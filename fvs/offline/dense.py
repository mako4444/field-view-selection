"""Planner inputs of the dense tier, computed from a replay directory without re-running localization.

The crop directories must hold harris3d.npy, the nine curvature invariants (gf_*.npy) and the
single-pass DetField values field_dense.npy (see fvs.prepare_dense).

Outputs in <out>/:
  rec.pkl       one row per (crop, decision, candidate, K): ok, cnt, field_sum, harris_sum, FIF variants
  feat.pkl      one row per landmark that the planner sees (geometric features, fv, h, invariants, corr, inl, lm)
  grid.npy      LWL input over the localizer landmarks (rows aligned with rec.pkl)
  grid_all.npy  LWL input over all crop points visible to the planner
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from .. import paths
from ..geomfeat import FEATS9, rank01
from ..replay.common import zbuf_vis
from ..sf3d import Crop
from .common import bin_grid, fif, load_landmarks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('replay', help='output directory of fvs.replay.indoor_dense')
    ap.add_argument('--crops', required=True)
    ap.add_argument('--frames', default=str(paths.SCENEFUN3D))
    ap.add_argument('--img_scale', type=float, default=0.5)
    ap.add_argument('--zdown', type=int, default=4)
    ap.add_argument('--ztol', type=float, default=0.05)
    ap.add_argument('--no_grids', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    L = [r for r in load_landmarks(a.replay, ['crop', 'dec', 'qry', 'method', 'K']) if r['method'] == 'harris']
    print(f'records {len(L)}', flush=True)
    by = defaultdict(list)
    for r in L:
        by[r['crop']].append(r)
    recs, feats, grids, grids_all = [], [], [], []
    for ci, (cs, rr) in enumerate(sorted(by.items())):
        visit, cname = cs.split('/')
        crop = Crop(Path(a.crops) / visit / cname, a.frames, a.img_scale, require_rgb=False)
        sd = crop.dir
        Ks, Pw = crop.K, crop.Pw
        H, W = crop.hw
        Nn = np.load(sd / 'pc_normal.npy').astype(np.float64).reshape(-1, 3)
        Nw = Nn @ crop.T[:3, :3]
        Nw /= np.maximum(np.linalg.norm(Nw, axis=1, keepdims=True), 1e-12)
        fv = np.load(sd / 'field_dense.npy').reshape(-1).astype(np.float64)
        assert len(fv) == len(Pw)
        hr = rank01(np.load(sd / 'harris3d.npy').reshape(-1))
        field_r = rank01(fv)
        C9 = np.stack([rank01(np.load(sd / f'{g}.npy')) for g in FEATS9], 1)
        allidx = np.unique(np.concatenate([r['idx'] for r in rr]))
        tree = cKDTree(Pw)
        cnt3 = np.zeros(len(Pw), np.int32)
        cnt3[allidx] = np.array([len(x) for x in tree.query_ball_point(Pw[allidx], 0.02)], np.int32)
        Pw_t = torch.from_numpy(Pw.astype(np.float32)).to(dev)
        gall_q = {}
        for r in rr:
            v = r['vis']
            idx = r['idx'][v].astype(np.int64)
            Tq = crop.pose(str(r['qry']))
            Tr = crop.pose(str(r['ref']))
            Eq = np.linalg.inv(Tq)
            X = Pw[idx]
            Xc = X @ Eq[:3, :3].T + Eq[:3, 3]
            z = np.maximum(Xc[:, 2], 1e-6)
            u = Ks[0, 0] * Xc[:, 0] / z + Ks[0, 2]
            vv = Ks[1, 1] * Xc[:, 1] / z + Ks[1, 2]
            rec = dict(crop=cs, dec=str(r['dec']), qry=str(r['qry']), ref=str(r['ref']), method=r['method'],
                       K=int(r['K']), ok=bool(r['ok']), cnt=len(idx), field_sum=float(fv[idx].sum()),
                       harris_sum=float(hr[idx].sum()))
            rec.update(fif(X, Tq))
            recs.append(rec)
            if not a.no_grids:
                grids.append(bin_grid(u, vv, Xc, np.concatenate([np.stack([field_r[idx], hr[idx]], 1), C9[idx]], 1),
                                      W, H))
                qk = str(r['qry'])
                if qk not in gall_q:
                    vq = zbuf_vis(Pw_t, Tq, Ks, (H, W), a.zdown, a.ztol, dev)
                    ii = np.where(vq)[0]
                    Yc = Pw[ii] @ Eq[:3, :3].T + Eq[:3, 3]
                    zy = np.maximum(Yc[:, 2], 1e-6)
                    uy = Ks[0, 0] * Yc[:, 0] / zy + Ks[0, 2]
                    vy = Ks[1, 1] * Yc[:, 1] / zy + Ks[1, 2]
                    gall_q[qk] = bin_grid(uy, vy, Yc, np.concatenate([np.stack([field_r[ii], hr[ii]], 1), C9[ii]], 1),
                                          W, H)
                grids_all.append(gall_q[qk])
            if v.any():
                zr = Xc[:, 2]
                ur = Ks[0, 0] * Xc[:, 0] / zr + Ks[0, 2]
                vr = Ks[1, 1] * Xc[:, 1] / zr + Ks[1, 2]
                b1 = X - Tq[:3, 3]
                b2 = X - Tr[:3, 3]
                n1 = np.linalg.norm(b1, axis=1)
                n2 = np.linalg.norm(b2, axis=1)
                cosv = np.clip((b1 * b2).sum(1) / (n1 * n2 + 1e-12), -1, 1)
                nrm = Nw[idx]
                uv = np.stack([ur, vr], 1)
                dimg = np.array([len(x) - 1 for x in cKDTree(uv).query_ball_point(uv, 32.0)], np.int32)
                df = pd.DataFrame(dict(
                    z=zr, ang_ax=np.degrees(np.arctan2(np.hypot(Xc[:, 0], Xc[:, 1]), zr)),
                    un=ur / W, vn=vr / H, rc=np.hypot(ur / W - 0.5, vr / H - 0.5),
                    d_ref=n2, ang_view=np.degrees(np.arccos(cosv)), lscale=np.log(n1 / np.maximum(n2, 1e-6)),
                    inc_q=np.abs((b1 * nrm).sum(1)) / (n1 + 1e-12), inc_r=np.abs((b2 * nrm).sum(1)) / (n2 + 1e-12),
                    dens_img=dimg, dens3d=cnt3[idx], n_vis=np.full(len(idx), len(idx), np.int32)))
                df['field'] = field_r[idx].astype(np.float32)
                df['h'] = hr[idx].astype(np.float32)
                for j, g in enumerate(FEATS9):
                    df[g] = C9[idx, j].astype(np.float32)
                df['corr'] = r['corr'][v]
                df['inl'] = r['inl'][v]
                df['lm'] = idx.astype(np.int32)
                df['crop'] = cs
                df['dec'] = str(r['dec'])
                df['qry'] = str(r['qry'])
                df['method'] = r['method']
                df['K'] = int(r['K'])
                df['ok'] = bool(r['ok'])
                feats.append(df)
        print(f'[{ci + 1}/{len(by)}] {cs} records {len(rr)}', flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_pickle(out / 'rec.pkl')
    D = pd.concat(feats, ignore_index=True)
    for c in ('crop', 'dec', 'qry', 'method'):
        D[c] = D[c].astype('category')
    for c in D.columns:
        if D[c].dtype == np.float64:
            D[c] = D[c].astype(np.float32)
    D.to_pickle(out / 'feat.pkl')
    if not a.no_grids:
        np.save(out / 'grid.npy', np.stack(grids))
        np.save(out / 'grid_all.npy', np.stack(grids_all))
    print('->', out, len(recs), 'records;', len(D), 'landmark rows')


if __name__ == '__main__':
    main()
