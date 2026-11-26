"""Indoor experiment, dense tier: replay of the view-selection decisions on the full crop point clouds.

Landmarks are Harris-3D points of the dense crop (radius NMS 2 cm in normalised units). Planner-side
visibility uses a z-buffer of the crop points at the candidate pose. Decisions and candidates are those
of the sparse tier.

Writes <out>/rows_s<i>.csv and <out>/lm_s<i>.pkl (landmark indices into pc.npy).
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import paths
from ..geomfeat import rank_avg
from ..localize import INDOOR, landmark_outcomes, localize_pair, nms_gpu, visible_mask
from ..sf3d import Crop, read_list
from ..superpoint import load_superpoint, superpoint_det_desc
from .common import add_protocol_args, make_decisions, mapped_pool, shard_items, zbuf_vis


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--crops', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--frames', default=str(paths.SCENEFUN3D))
    ap.add_argument('--superpoint', default=str(paths.SUPERPOINT))
    ap.add_argument('--ztol', type=float, default=0.05)
    ap.add_argument('--first', type=int, default=0, help='process only the first N crops of the shard')
    ap.add_argument('--out', required=True)
    add_protocol_args(ap)
    a = ap.parse_args()
    args = INDOOR
    dev = torch.device('cuda:0')
    Ks_list = [int(k) for k in a.Ks.split(',')]
    root = Path(a.crops)
    items, si, sn = shard_items(read_list(a.list), a.shard)
    if a.first:
        items = items[:a.first]
    sp = load_superpoint(a.superpoint, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, LMD = [], []
    t0 = time.time()
    for ci, (visit, cname) in enumerate(items):
        crop = Crop(root / visit / cname, a.frames, args.img_scale)
        Ks, hw, Pw, Pn = crop.K, crop.hw, crop.Pw, crop.Pn
        hscore = rank_avg(np.load(crop.dir / 'harris3d.npy').reshape(-1))
        assert len(hscore) == len(Pn)
        Pw_t = torch.from_numpy(Pw.astype(np.float32)).to(dev)
        Pn_t = torch.from_numpy(Pn.astype(np.float32)).to(dev)
        good, decisions, _ = make_decisions(crop, a, ci, dev)
        if not decisions:
            print(f'  skip {visit}/{cname}: no decision', flush=True)
            continue
        vis_map, depth_c, det_c, desc_c = {}, {}, {}, {}

        def map_vis(n):
            if n not in vis_map:
                dd = crop.depth(n)
                depth_c[n] = dd
                vm, uv, _ = visible_mask(Pw, crop.pose(n), Ks, dd, args.depth_tol)
                vm &= (uv[:, 0] >= 0) & (uv[:, 0] < hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < hw[0])
                vis_map[n] = vm
            return vis_map[n]

        def sp_frame(n):
            if n not in det_c:
                t = torch.from_numpy(crop.rgb(n)).float().to(dev) / 255.0
                a_, b_ = superpoint_det_desc(sp, t[None], chunk=1)
                det_c[n], desc_c[n] = a_[0], b_[0]
                if n not in depth_c:
                    depth_c[n] = crop.depth(n)

        lm_cache = {}

        def landmarks(ref, K):
            if ref not in lm_cache:
                idx = np.where(map_vis(ref))[0]
                sel = nms_gpu(Pn_t[idx], torch.from_numpy(hscore[idx]).to(dev), args.nms_radius_3d,
                              max(Ks_list)).cpu().numpy()
                lm_cache[ref] = idx[sel]
            return lm_cache[ref][:K]

        for di, (ia, pick, mp) in enumerate(decisions):
            mpn = [good[j] for j in mp]
            for jq in pick:
                q = good[jq]
                Tq = crop.pose(q)
                vq = zbuf_vis(Pw_t, Tq, Ks, hw, a.zdown, a.ztol, dev)
                pool, mapped = mapped_pool(crop, q, mpn, a)
                cov_r = [int((vq & map_vis(m)).sum()) for m in pool]
                ref = pool[int(np.argmax(cov_r))]
                co = vq & map_vis(ref)
                Tr = crop.pose(ref)
                u = dict(crop=f'{visit}/{cname}', dec=f'{di}', ref_frame=ref, qry=q, mapped=int(mapped),
                         n_co=int(co.sum()))
                sp_frame(ref)
                sp_frame(q)
                for K in Ks_list:
                    Lk = landmarks(ref, K)
                    Lv = Lk[vq[Lk]]
                    u2 = dict(u, method='harris', K=K, cnt=int(len(Lv)))
                    r = localize_pair(Pw, Lk, ref, q, {ref: Tr, q: Tq}, Ks, depth_c, det_c, desc_c, hw, args, dev,
                                      return_debug=True)
                    corr, inl_ = landmark_outcomes(r.get('_dbg') if r is not None else None, Pw[Lk], Tq, Ks, args)
                    LMD.append(dict(crop=f'{visit}/{cname}', dec=f'{di}', qry=q, ref=ref, method='harris', K=K,
                                    idx=Lk.astype(np.int32), vis=vq[Lk].copy(), corr=corr, inl=inl_,
                                    ok=bool(r['ok']) if r is not None else False))
                    if r is None:
                        r = dict(ok=False, e_R=np.nan, e_t=np.nan, n_inlier=0, n_match=0, n_kp3d=0)
                    u2.update({k: r[k] for k in ('ok', 'e_R', 'e_t', 'n_inlier', 'n_match', 'n_kp3d')})
                    rows.append(u2)
        okr = np.mean([r['ok'] for r in rows if r['crop'] == f'{visit}/{cname}'])
        print(f'[{ci + 1}/{len(items)}] {visit}/{cname} decisions {len(decisions)} ok {100 * okr:.1f}%  '
              f'{time.time() - t0:.0f}s', flush=True)
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / f'rows_s{si}.csv', index=False)
    pickle.dump(LMD, open(out / f'lm_s{si}.pkl', 'wb'))
    print('->', out / f'rows_s{si}.csv', len(rows), flush=True)


if __name__ == '__main__':
    main()
