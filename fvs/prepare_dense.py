"""Per-point inputs of the dense tier for every crop of a list: Harris-3D (32 neighbours) and the nine
curvature invariants. With --export_field, also the single-pass DetField value (field_dense.npy) for crops
that do not have it yet (GPU scatter operations make re-exported values differ in the last digits).

python -m fvs.prepare_dense --crops data/indoor/test --list data/splits/indoor_test.txt [--shard i n]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from . import paths
from .geomfeat import feats9_dense
from .localize import harris3d_scores
from .sf3d import read_list


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--crops', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--shard', type=int, nargs=2, default=(0, 1))
    ap.add_argument('--export_field', action='store_true')
    ap.add_argument('--weights', default=str(paths.DETFIELD_DENSE))
    a = ap.parse_args()
    model = None
    for k, (v, c) in enumerate(read_list(a.list)):
        if k % a.shard[1] != a.shard[0]:
            continue
        D = Path(a.crops) / v / c
        P = np.load(D / 'pc.npy').astype(np.float64)
        if not (D / 'harris3d.npy').exists():
            np.save(D / 'harris3d.npy', harris3d_scores(P, 32, max_cand=len(P), seed=0).astype(np.float32))
        if not (D / 'gf_sv64.npy').exists():
            for nm, arr in feats9_dense(P).items():
                np.save(D / f'{nm}.npy', arr)
        if a.export_field and not (D / 'field_dense.npy').exists():
            from .detfield import load_model, predict_field
            dev = torch.device('cuda')
            if model is None:
                model, fn = load_model(a.weights, dev)
            ft = np.stack([rankdata(np.load(D / f'{f}.npy').astype(np.float64)) / len(P) for f in fn], 1)
            g = torch.Generator(device=dev)
            g.manual_seed(0)
            o = predict_field(model, P, dev, 20000, 200000, g, feat=ft)
            np.save(D / 'field_dense.npy', np.asarray(o, dtype=np.float64).astype(np.float32))
        print(f'[{k}] {v}/{c} {len(P)} points', flush=True)


if __name__ == '__main__':
    main()
