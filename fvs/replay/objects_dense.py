"""Object experiment, dense objects: replay of the view-selection decisions on dense ShapeNet point clouds.

Twenty reference directions, and around each one M = 8 candidates at exactly theta = 10 or 15 degrees.
Landmarks are the top-K points of DetField (`field`) or of Harris-3D (`harris`, 32 neighbours) after
radius NMS (0.02). The planner knows the CAD model and hence which points are visible in each view.

Writes <out>/rows.csv with one row per (object, reference, candidate, landmark family, K).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from .. import paths
from ..detfield import load_model, predict_field
from ..geomfeat import rank_avg
from ..localize import OBJECTS, harris3d_scores, localize_one_pair, point_vis, radius_nms_3d
from ..render import normalize_points_to_unit_cube, render_views, repair_mesh_normals_full, ring_dirs
from ..superpoint import load_superpoint, superpoint_det_desc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--objects', default=str(paths.OBJECTS / 'dense'), help='directory with <synset>/<model>/pc.npy')
    ap.add_argument('--list', default=str(paths.OBJECTS / 'dense_objects.txt'))
    ap.add_argument('--shapenet', default=str(paths.SHAPENET))
    ap.add_argument('--weights', default=str(paths.DETFIELD_OBJECTS))
    ap.add_argument('--superpoint', default=str(paths.SUPERPOINT))
    ap.add_argument('--n_models', type=int, default=66)
    ap.add_argument('--n_base', type=int, default=20)
    ap.add_argument('--M', type=int, default=8)
    ap.add_argument('--thetas', default='10,15')
    ap.add_argument('--Ks', default='32,64,128')
    ap.add_argument('--view_seed', type=int, default=101)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    args = OBJECTS
    dev = torch.device('cuda:0')
    rng = np.random.RandomState(args.seed)
    root = Path(a.objects)
    rows_l = [l.split() for l in Path(a.list).read_text().splitlines() if l.strip()]
    rng.shuffle(rows_l)
    Ks = [int(k) for k in a.Ks.split(',')]
    lms = ['field', 'harris']
    thetas = [float(x) for x in a.thetas.split(',')]
    from pytorch3d.io import load_objs_as_meshes
    sp = load_superpoint(a.superpoint, dev)
    model, feat_names = load_model(a.weights, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, done, t0 = [], 0, time.time()
    for syn, mid in rows_l:
        if done >= a.n_models:
            break
        d = root / syn / mid
        obj = Path(a.shapenet) / syn / mid / 'models' / 'model_normalized.obj'
        mesh = load_objs_as_meshes([str(obj)], device=dev, load_textures=False)
        vn, _ = normalize_points_to_unit_cube(mesh.verts_packed(), padding=0.1)
        mesh = mesh.update_padded(vn[None])
        Vv, Fr = mesh.verts_packed(), mesh.faces_packed()
        F, cn, fo, _ = repair_mesh_normals_full(Vv, Fr, crease_deg=args.crease_deg)
        pc = np.load(d / 'pc.npy').astype(np.float64)
        g = torch.Generator(device=dev)
        g.manual_seed(args.seed)
        ft = np.stack([rankdata(np.load(d / f'{fn_}.npy').astype(np.float64).reshape(-1)) / len(pc)
                       for fn_ in feat_names], 1)
        sal = predict_field(model, pc.astype(np.float32), dev, args.num_points_encoder, args.eval_chunk, g,
                            feat=ft).astype(np.float64)
        harris = harris3d_scores(pc, args.harris_knn, seed=args.seed)
        FS = {'field': rank_avg(sal), 'harris': rank_avg(harris)}
        dirs, items = ring_dirs(a.n_base, a.M, thetas, a.view_seed + done, dev)
        img, depth, valid, cams, w2v, K, C = render_views(Vv, F, cn, fo, dirs, args, dev)
        det, desc = superpoint_det_desc(sp, img, chunk=4)
        vis = {i: point_vis(pc, w2v[i], K, depth[i], valid[i], args.depth_tol, dev) for i in range(len(dirs))}
        kmax = max(Ks)
        kp_full = {'field': radius_nms_3d(pc, sal, args.nms_radius_3d, kmax),
                   'harris': radius_nms_3d(pc, harris.astype(np.float32), args.nms_radius_3d, kmax)}
        # field values (probabilities), Harris ranks mapped to the field values, and permuted field values
        RAW = {'field_sum': sal.astype(np.float32),
               'harris_matched': np.sort(sal)[np.argsort(np.argsort(FS['harris'], kind='stable'),
                                                         kind='stable')].astype(np.float32),
               'field_permuted': sal[np.random.RandomState(4321 + done).permutation(len(pc))].astype(np.float32)}
        util = {}
        for (ref, c, th, j) in items:
            co = vis[ref] & vis[c]
            u = dict(n_co=int(co.sum()))
            for lm in lms:
                for K_ in Ks:
                    Lk = kp_full[lm][:K_]
                    Lv = Lk[vis[ref][Lk] & vis[c][Lk]]
                    u[f'cnt|{lm}_K{K_}'] = int(len(Lv))
                    u[f'harris_sum|{lm}_K{K_}'] = float(FS['harris'][Lv].sum())
                    for fn, rv in RAW.items():
                        u[f'{fn}|{lm}_K{K_}'] = float(rv[Lv].sum())
            util[(ref, c)] = u
        jobs = [(K_, ref, c, th, j, m) for K_ in Ks for (ref, c, th, j) in items for m in lms]

        def _one(job):
            K_, ref, c, th, j, m = job
            r = localize_one_pair(kp_full[m][:K_], pc, ref, c, det, desc, depth, valid, w2v, K, args, dev)
            rec = dict(crop=f'{syn}/{mid}', method=m, K=K_, ref=ref, qry=c, theta=th, j=j)
            rec.update(r if r else dict(ok=False, e_R=np.nan, e_t=np.nan, n_kp3d=0, n_match=0, n_inlier=0))
            return rec

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(_one, jobs))
        for r in res:
            r.update(util[(r['ref'], r['qry'])])
        recs += res
        done += 1
        okm = {m: np.mean([r['ok'] for r in res if r['method'] == m]) for m in lms}
        print(f'[{done}/{a.n_models}] {syn}/{mid} views {len(dirs)} '
              + ' '.join(f'{m}={100 * v:.1f}%' for m, v in okm.items()) + f'  {time.time() - t0:.0f}s', flush=True)
        del img, det, desc, depth, valid
        torch.cuda.empty_cache()
        pd.DataFrame(recs).to_csv(out / 'rows.csv', index=False)
    print('->', out / 'rows.csv', len(recs))


if __name__ == '__main__':
    main()
