"""Planner inputs of the sparse tier, computed from a replay directory without re-running localization.

Outputs in <out>/:
  rec.pkl        one row per (crop, decision, candidate, h, landmark family, K): outcome `ok`, the scalar
                 utilities (cnt, field_sum, harris_sum, n_co) and the FIF variants
  feat.pkl       one row per landmark that the planner sees: geometric features, field value (field),
                 Harris rank (h), the nine curvature invariants, the labels corr / inl, and a landmark id
  lwl/rec.pkl, lwl/grid_all.npy   LWL input (30 x 30 bins over all sparse points visible to the planner)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .. import paths
from ..geomfeat import FEATS9, feats9, normals, rank01, voxel_subsample
from ..localize import harris3d_scores
from ..replay.common import splat_vis
from ..sf3d import Crop
from .common import bin_grid, fif, load_landmarks

KEY = ['crop', 'dec', 'qry', 'mm', 'method', 'K']


def sparse_map(crop, mm, cache_dir):
    """Sparse points, field values on the candidate positions and Harris-3D, as in the replay."""
    Pn, T = crop.Pn, crop.T
    idx = voxel_subsample(Pn, T, mm / 1000.0)
    S = Pn[idx]
    f = crop.dir / f'field_s{mm:03d}.npz'
    if not f.exists():
        f = Path(cache_dir) / f'field_s{mm:03d}.npz'
    z = np.load(f)
    raw = z['raw'].astype(np.float64)
    harris8 = z['harris'] if 'harris' in z else harris3d_scores(S, 8, max_cand=len(S), seed=0).astype(np.float64)
    return idx, S, raw, harris8


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('replay', help='output directory of fvs.replay.indoor_sparse')
    ap.add_argument('--crops', required=True)
    ap.add_argument('--frames', default=str(paths.SCENEFUN3D))
    ap.add_argument('--cache', default='', help='field cache of the replay if the crops do not hold the field values')
    ap.add_argument('--lwl', action='store_true', help='also write the LWL input')
    ap.add_argument('--img_scale', type=float, default=0.5)
    ap.add_argument('--zdown', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    L = load_landmarks(a.replay, KEY)
    import glob
    Rw = pd.concat([pd.read_csv(f, dtype={'dec': str, 'qry': str}) for f in sorted(glob.glob(f'{a.replay}/rows_s*.csv'))])
    Rw = Rw.drop_duplicates(KEY)
    cols = ['cnt', 'field_sum', 'harris_sum', 'n_co', 'ok']
    Rw = {tuple(k): v for k, v in zip(Rw[KEY].itertuples(index=False, name=None), Rw[cols].itertuples(index=False))}
    print(f'records {len(L)}  rows {len(Rw)}', flush=True)
    by = defaultdict(list)
    for r in L:
        by[r['crop']].append(r)
    cache = Path(a.cache) if a.cache else Path(a.replay) / 'field_cache'
    recs, feats, lrec, g_all = [], [], [], []
    for ci, (cs, rr) in enumerate(sorted(by.items())):
        visit, cname = cs.split('/')
        crop = Crop(Path(a.crops) / visit / cname, a.frames, a.img_scale)
        Ks = crop.K
        H, W = crop.hw
        A = {}
        for mm in sorted({int(r['mm']) for r in rr}):
            idx, S, raw, harris8 = sparse_map(crop, mm, cache / visit / cname)
            nS = len(idx)
            Sw = crop.to_world(S)
            fd = feats9(S)
            c9 = np.stack([rank01(fd[k].astype(np.float64)) for k in FEATS9], 1)
            nrm = normals(Sw)[0]
            tree = cKDTree(Sw)
            dens = np.array([len(x) for x in tree.query_ball_point(Sw, 2 * mm / 1000.0)], np.int32)
            A[mm] = dict(Sw=Sw, Sw_t=torch.from_numpy(Sw.astype(np.float32)).to(dev), tree=tree, c9=c9, nrm=nrm,
                         dens=dens, raw_sorted={'field': np.sort(raw if mm >= 70 else raw[:nS]), 'harris': np.sort(raw[:nS])},
                         fieldS=rank01(raw[:nS]), hS=rankdata(harris8) / len(harris8), vS={}, ids={})
        for r in rr:
            mm = int(r['mm'])
            A_ = A[mm]
            meth = r['method']
            k = (cs, str(r['dec']), str(r['qry']), mm, meth, int(r['K']))
            row = Rw[k]
            v = r['vis']
            X = r['xw'][v].astype(np.float64)
            Tq = crop.pose(str(r['qry']))
            Tr = crop.pose(str(r['ref']))
            Eq = np.linalg.inv(Tq)
            Xc = X @ Eq[:3, :3].T + Eq[:3, 3]
            zc = np.maximum(Xc[:, 2], 1e-6)
            u = Ks[0, 0] * Xc[:, 0] / zc + Ks[0, 2]
            vv = Ks[1, 1] * Xc[:, 1] / zc + Ks[1, 2]
            rs = A_['raw_sorted'][meth]
            field_r = np.searchsorted(rs, r['field'][v].astype(np.float64)) / max(len(rs) - 1, 1)
            hr = r['h'][v].astype(np.float64)
            _, nn = A_['tree'].query(X) if len(X) else (None, np.zeros(0, int))
            c9 = A_['c9'][nn]
            rec = dict(crop=cs, dec=str(r['dec']), qry=str(r['qry']), ref=str(r['ref']), mm=mm, method=meth,
                       K=int(r['K']), ok=bool(r['ok']))
            for c in ('cnt', 'field_sum', 'harris_sum', 'n_co'):
                rec[c] = float(getattr(row, c))
            assert int(row.cnt) == len(X) and bool(row.ok) == rec['ok'], k
            rec.update(fif(X, Tq))
            recs.append(rec)
            if len(X):
                b1 = X - Tq[:3, 3]
                b2 = X - Tr[:3, 3]
                n1 = np.linalg.norm(b1, axis=1)
                n2 = np.linalg.norm(b2, axis=1)
                cosv = np.clip((b1 * b2).sum(1) / (n1 * n2 + 1e-12), -1, 1)
                nr = A_['nrm'][nn]
                uv = np.stack([u, vv], 1)
                dimg = np.array([len(x) - 1 for x in cKDTree(uv).query_ball_point(uv, 32.0)], np.int32)
                df = pd.DataFrame(dict(
                    z=zc, ang_ax=np.degrees(np.arctan2(np.hypot(Xc[:, 0], Xc[:, 1]), zc)),
                    un=u / W, vn=vv / H, rc=np.hypot(u / W - 0.5, vv / H - 0.5),
                    d_ref=n2, ang_view=np.degrees(np.arccos(cosv)), lscale=np.log(n1 / np.maximum(n2, 1e-6)),
                    inc_q=np.abs((b1 * nr).sum(1)) / (n1 + 1e-12), inc_r=np.abs((b2 * nr).sum(1)) / (n2 + 1e-12),
                    dens_img=dimg, dens3d=A_['dens'][nn], n_vis=np.full(len(X), len(X), np.int32),
                    field=field_r, h=hr))
                for j, g in enumerate(FEATS9):
                    df[g] = c9[:, j]
                df['corr'] = r['corr'][v]
                df['inl'] = r['inl'][v]
                # landmark id: index of the world position (5 decimals) within (crop, h, family)
                t = A_['ids'].setdefault(meth, {})
                df['lm'] = np.array([t.setdefault(p, len(t)) for p in map(tuple, np.round(X, 5))], np.int32)
                for c in ('crop', 'dec', 'qry', 'mm', 'method', 'K', 'ok'):
                    df[c] = rec[c]
                feats.append(df)
            if a.lwl and meth == 'field':
                lrec.append(rec)
                qk = str(r['qry'])
                if qk not in A_['vS']:
                    vS = splat_vis(A_['Sw_t'], A_['Sw_t'], Tq, Ks, (H, W), mm / 1000.0, a.zdown, mm / 1000.0, dev)
                    ii = np.where(vS)[0]
                    Y = A_['Sw'][ii]
                    Yc = Y @ Eq[:3, :3].T + Eq[:3, 3]
                    zy = np.maximum(Yc[:, 2], 1e-6)
                    uy = Ks[0, 0] * Yc[:, 0] / zy + Ks[0, 2]
                    vy = Ks[1, 1] * Yc[:, 1] / zy + Ks[1, 2]
                    A_['vS'][qk] = bin_grid(uy, vy, Yc, np.concatenate(
                        [np.stack([A_['fieldS'][ii], A_['hS'][ii]], 1), A_['c9'][ii]], 1), W, H)
                g_all.append(A_['vS'][qk])
        print(f'[{ci + 1}/{len(by)}] {cs} records {len(rr)}', flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_pickle(out / 'rec.pkl')
    F = pd.concat(feats, ignore_index=True)
    for c in ('crop', 'dec', 'qry', 'method'):
        F[c] = F[c].astype('category')
    for c in F.columns:
        if F[c].dtype == np.float64:
            F[c] = F[c].astype(np.float32)
    F.to_pickle(out / 'feat.pkl')
    if a.lwl:
        (out / 'lwl').mkdir(exist_ok=True)
        pd.DataFrame(lrec).to_pickle(out / 'lwl' / 'rec.pkl')
        np.save(out / 'lwl' / 'grid_all.npy', np.stack(g_all))
    print('->', out, len(recs), 'records;', len(F), 'landmark rows')


if __name__ == '__main__':
    main()
