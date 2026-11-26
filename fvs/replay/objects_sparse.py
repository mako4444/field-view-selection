"""Object experiment, sparse objects: replay of the view-selection decisions on ShapeNet objects.

Each object is known as a sparse point set (voxel edge h = 0.02 or 0.04 of the normalised object).
Twelve reference directions are rendered, and around each one M = 8 candidate directions at exactly
theta = 10 or 15 degrees. Landmarks are the top-K points of DetField (`field`) or of Harris-3D
(`harris`) after radius NMS. The planner knows which landmarks the CAD model makes visible in each
view (model-based setting), never the candidate image.

Writes <out>/rows_s<i>.csv with one row per (object, h, reference, candidate, landmark family, K).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .. import paths
from ..detfield import load_model, predict_field
from ..geomfeat import sparse_object
from ..localize import OBJECTS, harris3d_scores, localize_one_pair, point_vis, radius_nms_3d
from ..render import render_views, repair_mesh_normals_full, ring_dirs
from ..superpoint import load_superpoint, superpoint_det_desc

HARRIS_KNN = {0.02: 16, 0.04: 8}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--objects', default=str(paths.OBJECTS / 'sparse'), help='directory with <synset>/<model>/dense.npy')
    ap.add_argument('--list', default=str(paths.OBJECTS / 'sparse_test.txt'))
    ap.add_argument('--shapenet', default=str(paths.SHAPENET))
    ap.add_argument('--weights', default=str(paths.DETFIELD_OBJECTS))
    ap.add_argument('--superpoint', default=str(paths.SUPERPOINT))
    ap.add_argument('--sps', default='0.02,0.04')
    ap.add_argument('--ks', default='16,32,64')
    ap.add_argument('--thetas', default='10,15')
    ap.add_argument('--n_base', type=int, default=12)
    ap.add_argument('--M', type=int, default=8)
    ap.add_argument('--nms_r', type=float, default=0.05)
    ap.add_argument('--shard', default='0,6', help='i,n: objects with index % n == i')
    ap.add_argument('--first', type=int, default=0, help='process only the first N objects of the shard')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    args = OBJECTS
    dev = torch.device('cuda')
    sps = [float(x) for x in a.sps.split(',')]
    ks = [int(x) for x in a.ks.split(',')]
    kmax = max(ks)
    thetas = [float(x) for x in a.thetas.split(',')]
    fams = ('field', 'harris')
    si, sn = (int(x) for x in a.shard.split(','))
    objs = [l.split() for l in open(a.list) if l.strip()]
    model, _ = load_model(a.weights, dev)
    from pytorch3d.io import load_objs_as_meshes
    import trimesh
    sp = load_superpoint(a.superpoint, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f'rows_s{si}.csv'
    recs_all, done = [], 0
    for oi, (syn, mid) in enumerate(objs):
        if oi % sn != si or (a.first and done >= a.first):
            continue
        t0 = time.time()
        obj = Path(a.shapenet) / syn / mid / 'models' / 'model_normalized.obj'
        mesh = load_objs_as_meshes([str(obj)], device=dev, load_textures=False)
        V0 = mesh.verts_packed()
        tv = np.asarray(trimesh.load(str(obj), force='mesh', process=False).vertices)
        lo, hi = tv.min(0), tv.max(0)
        ctr = torch.from_numpy((lo + hi) / 2).float().to(dev)
        vn = (V0 - ctr) / float((hi - lo).max() / 0.9)
        mesh = mesh.update_padded(vn[None])
        Vv, Fr = mesh.verts_packed(), mesh.faces_packed()
        F, cn, fop, _ = repair_mesh_normals_full(Vv, Fr, crease_deg=args.crease_deg)
        dirs, items = ring_dirs(a.n_base, a.M, thetas, args.seed + oi, dev)
        img, depth, valid, cams, w2v, K, C = render_views(Vv, F, cn, fop, dirs, args, dev)
        det, desc = superpoint_det_desc(sp, img, chunk=4)
        X = np.load(Path(a.objects) / syn / mid / 'dense.npy').astype(np.float64)
        jobs, util, LM_all = [], {}, {}
        for s in sps:
            S, feat = sparse_object(X, s, np.random.default_rng(0))
            tS = cKDTree(S)
            gg = torch.Generator(device=dev)
            gg.manual_seed(0)
            c1s = predict_field(model, S.astype(np.float32), dev, 8192, 100000, gg, feat=feat).astype(np.float64)
            hk = harris3d_scores(S, HARRIS_KNN[s], seed=0).astype(np.float64)
            hr = rankdata(np.nan_to_num(hk, neginf=-1e30)) / len(S)
            hq = np.sort(c1s)[np.argsort(np.argsort(hr, kind='stable'), kind='stable')]
            shuf = c1s[np.random.RandomState(4321 + oi).permutation(len(S))]
            visS = {i: point_vis(S, w2v[i], K, depth[i], valid[i], args.depth_tol, dev) for i in range(len(dirs))}
            LM = {}
            for fam in fams:
                sc = c1s if fam == 'field' else hk
                P = S[radius_nms_3d(S, sc.astype(np.float32), a.nms_r, kmax)]
                nn = tS.query(P)[1]
                vis = {i: point_vis(P, w2v[i], K, depth[i], valid[i], args.depth_tol, dev) for i in range(len(dirs))}
                LM[fam] = (P, nn, vis)
                LM_all[(s, fam)] = P
            for (ref, c, th, j) in items:
                n_co = int((visS[ref] & visS[c]).sum())
                for fam in fams:
                    P, nn, vis = LM[fam]
                    for k in ks:
                        m = vis[ref][:k] & vis[c][:k]
                        idx = nn[:k][m]
                        # utilities over the top-k landmarks visible in both the reference and the candidate:
                        # field_sum, count, Harris-matched (Harris ranks mapped to the field values), permuted
                        # field, Harris-sum (ranks) and co-visible sparse points
                        util[(s, fam, k, ref, c)] = dict(cnt=int(m.sum()), field_sum=float(c1s[idx].sum()),
                                                         harris_matched=float(hq[idx].sum()),
                                                         field_permuted=float(shuf[idx].sum()),
                                                         harris_sum=float(hr[idx].sum()), n_co=n_co)
                        jobs.append((s, fam, k, ref, c, th, j))

        def one(jb):
            s, fam, k, ref, c, th, j = jb
            P = LM_all[(s, fam)][:k]
            r = localize_one_pair(np.arange(len(P)), P, ref, c, det, desc, depth, valid, w2v, K, args, dev) \
                if len(P) >= 4 else None
            rec = dict(obj=oi, model=f'{syn}/{mid}', s=s, method=fam, K=k, ref=ref, qry=c, theta=th, j=j)
            rec.update(util[(s, fam, k, ref, c)])
            rec.update({kk: r[kk] for kk in ('ok', 'e_R', 'e_t', 'n_match', 'n_inlier')} if r else
                       dict(ok=False, e_R=np.nan, e_t=np.nan, n_match=0, n_inlier=0))
            return rec

        with ThreadPoolExecutor(a.workers) as ex:
            recs = list(ex.map(one, jobs))
        recs_all += recs
        done += 1
        pd.DataFrame(recs_all).to_csv(fname, index=False)
        print(f'[{done}] {oi} {syn}/{mid} views {len(dirs)} ' + ' '.join(
            f'{f}:{100 * np.mean([r["ok"] for r in recs if r["method"] == f]):.0f}' for f in fams)
              + f' {time.time() - t0:.0f}s', flush=True)
        del img, det, desc, depth, valid
        torch.cuda.empty_cache()
    print('->', fname, len(recs_all), flush=True)


if __name__ == '__main__':
    main()
