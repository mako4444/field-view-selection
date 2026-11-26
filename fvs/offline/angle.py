"""The `angle` planner (Hanlon et al., ICRA 2024): count the visible landmarks whose current viewing
geometry is close to one under which they were mapped.

Mapping observations: landmark visible (sensor depth) in a map frame, i.e. a frame more than `gap`
seconds away from every candidate of the decision (every `stride`-th frame of the video).
  ang_dm<t>  Davison and Murray (2002): a mapping observation with viewing-direction angle <= t degrees
             and distance ratio in [5/7, 7/5]; t in {15, 30, 45, 60}
  ang_rng    current distance and angle to the optical axis within the ranges of the mapping observations
Writes a pickle with one row per record.
"""
from __future__ import annotations

import argparse
import glob
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .. import paths
from ..localize import visible_mask
from ..sf3d import Crop, ts

THS = (15, 30, 45, 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('replay', help='replay output directory (lm_s*.pkl)')
    ap.add_argument('--crops', required=True)
    ap.add_argument('--frames', default=str(paths.SCENEFUN3D))
    ap.add_argument('--methods', default='field,harris')
    ap.add_argument('--gap', type=float, default=0.5)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--img_scale', type=float, default=0.5)
    ap.add_argument('--depth_tol', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    meths = set(a.methods.split(','))
    L = []
    for fp in sorted(glob.glob(f'{a.replay}/lm_s*.pkl')):
        L += [r for r in pickle.load(open(fp, 'rb')) if r['method'] in meths]
    by = defaultdict(list)
    for r in L:
        by[r['crop']].append(r)
    out = []
    for ci, (cs, rr) in enumerate(sorted(by.items())):
        visit, cname = cs.split('/')
        crop = Crop(Path(a.crops) / visit / cname, a.frames, a.img_scale)
        d, poses, Pw, Ks, hw = crop.video_dir, crop.poses, crop.Pw, crop.K, crop.hw
        names = crop.names[::a.stride]
        if 'idx' in rr[0]:
            U = np.unique(np.concatenate([r['idx'] for r in rr]).astype(np.int64))
            pos = {int(i): k for k, i in enumerate(U)}
            X = Pw[U]
            lm_of = lambda r: np.array([pos[int(i)] for i in r['idx'][r['vis']]], np.int64)
        else:
            allx = np.concatenate([r['xw'] for r in rr]).astype(np.float64)
            X, inv = np.unique(np.round(allx, 5), axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            offs = np.cumsum([0] + [len(r['xw']) for r in rr])
            ridx = {id(r): inv[offs[j]:offs[j + 1]] for j, r in enumerate(rr)}
            lm_of = lambda r: ridx[id(r)][r['vis']].astype(np.int64)
            U = X
        nF = len(names)
        obs = np.zeros((len(U), nF), bool)
        dirs = np.zeros((len(U), nF, 3), np.float32)
        dist = np.zeros((len(U), nF), np.float32)
        axang = np.zeros((len(U), nF), np.float32)
        tF = np.array([ts(n) for n in names])
        for j, n in enumerate(names):
            Tm = poses[ts(n)]
            dd = cv2.imread(str(d / 'hires_depth' / f'{n}.png'), -1).astype(np.float32) / 1000.0
            dd = cv2.resize(dd, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            vm, _, _ = visible_mask(X, Tm, Ks, dd, a.depth_tol)
            obs[:, j] = vm
            b = X - Tm[:3, 3]
            nb = np.linalg.norm(b, axis=1)
            dirs[:, j] = b / nb[:, None]
            dist[:, j] = nb
            ax = Tm[:3, 2] / np.linalg.norm(Tm[:3, 2])
            axang[:, j] = np.degrees(np.arccos(np.clip(dirs[:, j] @ ax, -1, 1)))
        cand = defaultdict(set)
        for r in rr:
            cand[str(r['dec'])].add(ts(str(r['qry'])))
        for r in rr:
            ii = lm_of(r)
            tc = np.array(sorted(cand[str(r['dec'])]))
            mF = np.all(np.abs(tF[:, None] - tc[None]) > a.gap, 1)
            rec = dict(crop=cs, dec=str(r['dec']), qry=str(r['qry']), K=int(r['K']), method=r['method'])
            if 'mm' in r:
                rec['mm'] = int(r['mm'])
            if len(ii) == 0 or not mF.any():
                rec.update({f'ang_dm{t}': 0 for t in THS})
                rec['ang_rng'] = 0
                out.append(rec)
                continue
            Tq = poses[ts(str(r['qry']))]
            bq = X[ii] - Tq[:3, 3]
            nq = np.linalg.norm(bq, axis=1)
            uq = bq / nq[:, None]
            axq = Tq[:3, 2] / np.linalg.norm(Tq[:3, 2])
            aq = np.degrees(np.arccos(np.clip(uq @ axq, -1, 1)))
            O = obs[ii][:, mF]
            Dm = dirs[ii][:, mF]
            Nm = dist[ii][:, mF]
            Am = axang[ii][:, mF]
            ang = np.degrees(np.arccos(np.clip(np.einsum('lk,lfk->lf', uq, Dm), -1, 1)))
            ratio = nq[:, None] / np.maximum(Nm, 1e-6)
            rok = (ratio >= 5 / 7) & (ratio <= 7 / 5) & O
            for t in THS:
                rec[f'ang_dm{t}'] = int((rok & (ang <= t)).any(1).sum())
            big = np.where(O, Nm, -np.inf).max(1)
            small = np.where(O, Nm, np.inf).min(1)
            abig = np.where(O, Am, -np.inf).max(1)
            asmall = np.where(O, Am, np.inf).min(1)
            rec['ang_rng'] = int(((nq >= small) & (nq <= big) & (aq >= asmall) & (aq <= abig)).sum())
            out.append(rec)
        print(f'[{ci + 1}/{len(by)}] {cs} landmarks {len(U)} map frames {nF}', flush=True)
    pd.DataFrame(out).to_pickle(a.out)
    print('->', a.out, len(out))


if __name__ == '__main__':
    main()
