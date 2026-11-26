"""Why the gain is small (development rooms).

1. Ideal static field p: arm G+P against G, and the share of that gain obtained by G+F.
2. Single landmarks: room-averaged AUC of the field and of p for a correct match of a visible landmark;
   for a correct match becoming a RANSAC inlier, AUC of the field and of the landmark depth (K = 96).
3. A planner that knows the true number of correct matches of each candidate (u_corr) against the oracle.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..planners.hgb import TIER, arms, ideal_static_field, load_feats, predict
from .indoor import norm
from .stats import cluster_ci, fmt_diff, per_decision


def room_auc(D, col, label='corr'):
    a = []
    for _, g in D.groupby(D.crop.str.split('/').str[0]):
        if g[label].nunique() == 2:
            a.append(roc_auc_score(g[label], g[col]))
    return float(np.mean(a))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tier', choices=['sparse', 'dense'], required=True)
    ap.add_argument('--feat', required=True, help='landmark table of the development rooms')
    ap.add_argument('--rec', required=True, help='rec.pkl of the development rooms')
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t = TIER[a.tier]
    method = t['methods'][0]
    f = t['field']
    Te = load_feats(a.feat)
    Te = Te[Te.method == method].reset_index(drop=True)
    for c in ('mm', 'K'):
        if c in Te:
            Te[c] = Te[c].astype(int)
    Te['pbar'] = ideal_static_field(Te, a.tier)
    Te = predict(a.tier, a.models, Te, with_pbar=True)
    res = dict(tier=a.tier, auc={c: room_auc(Te, c) for c in (f, 'pbar')})
    C = Te[(Te.K == 96) & Te['corr'].astype(bool)]
    res['auc_inlier_given_correct'] = {c: room_auc(C, c, 'inl') for c in (f, 'z')}
    A = list(arms(a.tier, with_pbar=True))
    U = Te.groupby(t['key'], observed=True).agg(**{f'u_{x}': (f'p_{x}', 'sum') for x in A},
                                               u_corr=('corr', 'sum')).reset_index()
    R = norm(pd.read_pickle(a.rec))
    R = R[R.method == method].copy()
    R = R.merge(U, on=t['key'], how='left')
    uc = [c for c in U.columns if c.startswith('u_')]
    R[uc] = R[uc].fillna(0.0)
    D = per_decision(R, uc, inner=['mm', 'K'] if a.tier == 'sparse' else ['K'])
    res['rates'] = {u: 100 * D[u].mean() for u in uc + ['random', 'oracle']}
    res['diffs'] = {}
    for x, y in (('u_G+P', 'u_G'), ('u_G+F', 'u_G')):
        res['diffs'][f'{x} - {y}'] = list(cluster_ci(D, x, y, 'room'))
    gap = D['u_G+P'].mean() - D['u_G'].mean()
    res['share_of_ideal_gain'] = 100 * (D['u_G+F'].mean() - D['u_G'].mean()) / gap
    print(f'== {a.tier} tier, {method} landmarks, {len(D)} decisions')
    print('   AUC correct match:', {k: round(v, 3) for k, v in res['auc'].items()})
    print('   AUC inlier | correct (K=96):', {k: round(v, 3) for k, v in res['auc_inlier_given_correct'].items()})
    for u in uc + ['random', 'oracle']:
        print(f'   {u:<10} {res["rates"][u]:5.1f}')
    for k, (m, lo, hi) in res['diffs'].items():
        print(f'   {k:<14} {fmt_diff(m, lo, hi)}')
    print(f'   share of the ideal-field gain obtained by G+F: {res["share_of_ideal_gain"]:.0f}%')
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
