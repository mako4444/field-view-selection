"""Indoor experiment, sparse tier: replay of the view-selection decisions on SceneFun3D (or HM3D) crops.

The planner knows only a sparse map, i.e. the crop voxel-subsampled to h = 5 or 10 cm. Two landmark
families are evaluated: DetField landmarks (`field`; on the sparse points at 5 cm and on candidate
positions between them at 10 cm) and Harris-3D landmarks on the sparse points (`harris`, 8 neighbours).
For every candidate, the reference frame is the mapped frame that shares most sparse points with it;
the localizer then matches the landmarks visible in the reference frame against the candidate image.

Writes <out>/rows_s<i>.csv (one row per decision, candidate, h, landmark family and K) and
<out>/lm_s<i>.pkl (per-landmark records used by the offline features).
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata

from .. import paths
from ..detfield import load_model, query_field
from ..geomfeat import FEATS9, candidates, crop_scale, feats9, voxel_subsample
from ..localize import INDOOR, harris3d_scores, landmark_outcomes, localize_pair, radius_nms_3d, visible_mask
from ..sf3d import Crop, read_list
from ..superpoint import load_superpoint, superpoint_det_desc
from .common import add_protocol_args, make_decisions, mapped_pool, shard_items, splat_vis

ROTS = 4


def sparse_assets(crop, mm, model, dev, cache_dir):
    """Sparse map of one crop at voxel edge `mm` millimetres: point indices, Harris-3D (8 neighbours),
    candidate positions and the DetField values on them (`tta`: mean rank over 4 rotations, used to select
    landmarks; `raw`: single-pass value, used as utility). Stored values in `field_s<mm>.npz` of the crop
    directory (or of `cache_dir`) take precedence over recomputed ones."""
    Pn, T = crop.Pn, crop.T
    sc = crop_scale(T)
    idx = voxel_subsample(Pn, T, mm / 1000.0)
    S = Pn[idx]
    C, own = candidates(S, mm / 1000.0 * sc, 12, np.random.default_rng(0))
    f = crop.dir / f'field_s{mm:03d}.npz'
    if not f.exists():
        f = Path(cache_dir) / f'field_s{mm:03d}.npz'
    if f.exists():
        z = np.load(f)
        tta, raw = z['tta'], z['raw'].astype(np.float64)
        harris8 = z['harris'] if 'harris' in z else harris3d_scores(S, 8, max_cand=len(S), seed=0).astype(np.float64)
        if 'C' in z:
            C = z['C']
        return dict(idx=idx, C=C, own=own, tta=tta, raw=raw, harris8=harris8)
    harris8 = harris3d_scores(S, 8, max_cand=len(S), seed=0).astype(np.float64)
    fd = feats9(S)
    feat = np.stack([rankdata(fd[k].astype(np.float64)) / len(S) for k in FEATS9], 1)
    acc = np.zeros(len(C))
    raw = None
    for r in range(ROTS):
        if r == 0:
            Sr, Cr = S, C
        else:
            R = Rotation.random(random_state=1000 + r).as_matrix()
            mu = S.mean(0)
            Sr, Cr = (S - mu) @ R.T + mu, (C - mu) @ R.T + mu
        o = query_field(model, Sr, feat, Cr, dev)
        if r == 0:
            raw = o.copy()
        acc += rankdata(o) / len(C)
    tta = acc / ROTS
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(f, tta=tta, raw=raw.astype(np.float32), harris=harris8)
    return dict(idx=idx, C=C, own=own, tta=tta, raw=raw, harris8=harris8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--crops', required=True, help='directory of crop samples')
    ap.add_argument('--list', required=True, help='crop list (visit crop per line)')
    ap.add_argument('--frames', default=str(paths.SCENEFUN3D), help='root of the captures')
    ap.add_argument('--weights', default=str(paths.DETFIELD_SPARSE))
    ap.add_argument('--superpoint', default=str(paths.SUPERPOINT))
    ap.add_argument('--mms', default='50,100', help='voxel edges of the sparse map (mm)')
    ap.add_argument('--cache', default='', help='where to write field values that are not stored with the crops')
    ap.add_argument('--first', type=int, default=0, help='process only the first N crops of the shard')
    ap.add_argument('--out', required=True)
    add_protocol_args(ap)
    a = ap.parse_args()
    args = INDOOR
    dev = torch.device('cuda:0')
    Ks_list = [int(k) for k in a.Ks.split(',')]
    mms = [int(x) for x in a.mms.split(',')]
    root = Path(a.crops)
    items, si, sn = shard_items(read_list(a.list), a.shard)
    if a.first:
        items = items[:a.first]
    model, fn = load_model(a.weights, dev)
    assert fn == FEATS9
    sp = load_superpoint(a.superpoint, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) if a.cache else out / 'field_cache'
    rows, LMD = [], []
    for ci, (visit, cname) in enumerate(items):
        tc = time.time()
        crop = Crop(root / visit / cname, a.frames, args.img_scale)
        Ks, hw, Pw = crop.K, crop.hw, crop.Pw
        good, decisions, _ = make_decisions(crop, a, ci, dev)
        if not decisions:
            print(f'  skip {visit}/{cname}: no decision', flush=True)
            continue
        depth_c, det_c, desc_c = {}, {}, {}

        def depth(n):
            if n not in depth_c:
                depth_c[n] = crop.depth(n)
            return depth_c[n]

        def sp_frame(n):
            if n not in det_c:
                t = torch.from_numpy(crop.rgb(n)).float().to(dev) / 255.0
                a_, b_ = superpoint_det_desc(sp, t[None], chunk=1)
                det_c[n], desc_c[n] = a_[0], b_[0]
                depth(n)

        def real_vis(Xw, n):
            vm, uv, _ = visible_mask(Xw, crop.pose(n), Ks, depth(n), args.depth_tol)
            return vm & (uv[:, 0] >= 0) & (uv[:, 0] < hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < hw[0])

        for mm in mms:
            sa = sparse_assets(crop, mm, model, dev, cache / visit / cname)
            idx, Cn, own, tta, raw, harris8 = sa['idx'], sa['C'], sa['own'], sa['tta'], sa['raw'], sa['harris8']
            nS = len(idx)
            Sw = Pw[idx]
            Sw_t = torch.from_numpy(Sw.astype(np.float32)).to(dev)
            # landmark family -> (positions, selection score, field value, Harris value)
            if mm >= 70:
                fam = {'field': (Cn, tta, raw, harris8[own])}
            else:
                fam = {'field': (Cn[:nS], tta[:nS], raw[:nS], harris8)}
            fam['harris'] = (Cn[:nS], harris8, raw[:nS], harris8)
            U = {}
            for f_, (xn_, sel_, fr_, hv_) in fam.items():
                U[f_] = dict(xn=xn_, xw=crop.to_world(xn_), sel=sel_, field=fr_, h=rankdata(hv_) / len(hv_))
                U[f_]['xw_t'] = torch.from_numpy(U[f_]['xw'].astype(np.float32)).to(dev)
            svis_map, lm_cache = {}, {}

            def map_vis_S(n):
                if n not in svis_map:
                    svis_map[n] = real_vis(Sw, n)
                return svis_map[n]

            def landmarks(ref, f_, K):
                key = (ref, f_)
                if key not in lm_cache:
                    vm = real_vis(U[f_]['xw'], ref)
                    sub_ = np.where(vm)[0]
                    lm_cache[key] = sub_[radius_nms_3d(U[f_]['xn'][sub_], U[f_]['sel'][sub_],
                                                       args.nms_radius_3d, max(Ks_list))]
                return lm_cache[key][:K]

            tol = mm / 1000.0
            for di, (ia, pick, mp) in enumerate(decisions):
                mpn = [good[j] for j in mp]
                for jq in pick:
                    q = good[jq]
                    Tq = crop.pose(q)
                    vS = splat_vis(Sw_t, Sw_t, Tq, Ks, hw, tol, a.zdown, tol, dev)
                    pool, mapped = mapped_pool(crop, q, mpn, a)
                    cov_r = [int((vS & map_vis_S(m)).sum()) for m in pool]
                    ref = pool[int(np.argmax(cov_r))]
                    n_co = int((vS & map_vis_S(ref)).sum())
                    Tr = crop.pose(ref)
                    u = dict(crop=f'{visit}/{cname}', dec=f'{di}', mm=mm, ref_frame=ref, qry=q, mapped=int(mapped),
                             n_co=n_co)
                    sp_frame(ref)
                    sp_frame(q)
                    for f_ in fam:
                        va = splat_vis(Sw_t, U[f_]['xw_t'], Tq, Ks, hw, tol, a.zdown, tol, dev)
                        for K in Ks_list:
                            Lk = landmarks(ref, f_, K)
                            Lv = Lk[va[Lk]]
                            u2 = dict(u, method=f_, K=K, cnt=int(len(Lv)), field_sum=float(U[f_]['field'][Lv].sum()),
                                      harris_sum=float(U[f_]['h'][Lv].sum()))
                            r = localize_pair(U[f_]['xw'], Lk, ref, q, {ref: Tr, q: Tq}, Ks, depth_c, det_c, desc_c,
                                              hw, args, dev, return_debug=True)
                            XL = U[f_]['xw'][Lk]
                            corr, inl_ = landmark_outcomes(r.get('_dbg') if r is not None else None, XL, Tq, Ks, args)
                            LMD.append(dict(crop=f'{visit}/{cname}', dec=f'{di}', qry=q, ref=ref, method=f_, mm=mm, K=K,
                                            xw=XL.astype(np.float32), vis=va[Lk].copy(),
                                            field=U[f_]['field'][Lk].astype(np.float32), h=U[f_]['h'][Lk].astype(np.float32),
                                            corr=corr, inl=inl_, ok=bool(r['ok']) if r is not None else False))
                            if r is None:
                                r = dict(ok=False, e_R=np.nan, e_t=np.nan, n_inlier=0, n_match=0, n_kp3d=0)
                            u2.update({k: r[k] for k in ('ok', 'e_R', 'e_t', 'n_inlier', 'n_match', 'n_kp3d')})
                            rows.append(u2)
        okr = {mm: np.mean([r['ok'] for r in rows if r['crop'] == f'{visit}/{cname}' and r['mm'] == mm]) for mm in mms}
        print(f'[{ci + 1}/{len(items)}] {visit}/{cname} decisions {len(decisions)} '
              + ' '.join(f'{mm}mm ok {100 * v:.1f}%' for mm, v in okr.items()) + f'  {time.time() - tc:.0f}s', flush=True)
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / f'rows_s{si}.csv', index=False)
    pickle.dump(LMD, open(out / f'lm_s{si}.pkl', 'wb'))
    print('->', out / f'rows_s{si}.csv', len(rows), flush=True)


if __name__ == '__main__':
    main()
