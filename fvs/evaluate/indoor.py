"""Top-1 success of every planner on one split of the indoor experiment, and paired differences with
95% intervals from a bootstrap over rooms (or crops). Writes a JSON file used by the table scripts.

A decision is (crop, anchor); within a decision, top-1 success is averaged over K (and over h when both
map spacings are pooled).
"""
from __future__ import annotations

import argparse
import glob
import json

import pandas as pd

from .stats import cluster_ci, fmt_diff, per_decision

# FIF and angle variants chosen on the six validation rooms
VARIANTS = {('sparse', 'field'): ('fif_tr_qa', 'ang_dm15'), ('sparse', 'harris'): ('fif_me_ex', 'ang_rng'),
            ('dense', 'harris'): ('fif_ld_qa', 'ang_dm15')}
PAIRS = [('u_G+F', 'u_G'), ('u_G+F', 'u_G+H'), ('u_G+F', 'u_G+C9'), ('u_G+F+C9', 'u_G+C9'), ('u_G', 'cnt'),
         ('u_lwl', 'cnt'), ('cnt', 'fif'), ('field_sum', 'cnt'), ('u_G+H', 'u_G'), ('u_G+C9', 'u_G')]


def norm(D):
    for c in ('crop', 'dec', 'qry', 'method'):
        if c in D:
            D[c] = D[c].astype(str)
    for c in ('mm', 'K'):
        if c in D:
            D[c] = D[c].astype(int)
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tier', choices=['sparse', 'dense'], required=True)
    ap.add_argument('--rec', required=True, help='rec.pkl of fvs.offline')
    ap.add_argument('--plm', required=True, help='utilities of the learned planners (fvs.planners.hgb apply)')
    ap.add_argument('--angle', default='')
    ap.add_argument('--lwl', default='', help='LWL predictions (fvs.planners.lwl predict)')
    ap.add_argument('--covis', default='', help='dense tier: replay rows (glob) that hold n_co')
    ap.add_argument('--method', default='')
    ap.add_argument('--mm', default='all', help='sparse tier: 50, 100 or all')
    ap.add_argument('--level', default='room', choices=['room', 'crop'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sp = a.tier == 'sparse'
    method = a.method or ('field' if sp else 'harris')
    fif_v, ang_v = VARIANTS[(a.tier, method)]
    key = ['crop', 'dec', 'qry'] + (['mm'] if sp else []) + ['K']
    R = norm(pd.read_pickle(a.rec))
    R = R[R.method == method].copy()
    if sp and a.mm != 'all':
        R = R[R.mm == int(a.mm)]
    if sp:
        R = R[R.K.isin([64, 80, 96])]
    R['fif'] = R[fif_v]
    utils = ['cnt', 'n_co', 'field_sum', 'harris_sum', 'fif']
    if a.covis:
        C = norm(pd.concat([pd.read_csv(f, usecols=['crop', 'dec', 'qry', 'method', 'K', 'n_co'])
                            for f in sorted(glob.glob(a.covis))]))
        C = C[C.method == method].drop_duplicates(['crop', 'dec', 'qry', 'K'])
        R = R.merge(C[['crop', 'dec', 'qry', 'K', 'n_co']], on=['crop', 'dec', 'qry', 'K'], how='left')
    if 'n_co' not in R:
        utils.remove('n_co')
    P = norm(pd.read_pickle(a.plm))
    P = P[P.method == method]
    pc = [c for c in P.columns if c.startswith('u_')]
    R = R.merge(P[key + pc], on=key, how='left')
    R[pc] = R[pc].fillna(0.0)
    utils += pc
    if a.angle:
        A = norm(pd.read_pickle(a.angle))
        A = A[A.method == method]
        R = R.merge(A[key + [ang_v]].rename(columns={ang_v: 'angle'}), on=key, how='inner')
        utils.append('angle')
    if a.lwl:
        L = norm(pd.read_pickle(a.lwl))
        L = L[L.method == method]
        R = R.merge(L[key + ['u_lwl']], on=key, how='inner')
        utils.append('u_lwl')
    D = per_decision(R, utils, inner=['mm', 'K'] if sp else ['K'])
    rnd, orc = D['random'].mean(), D['oracle'].mean()
    res = dict(tier=a.tier, method=method, mm=a.mm, decisions=len(D),
               rooms=int(D.index.get_level_values(0).str.split('/').str[0].nunique()),
               records=len(R), fif_variant=fif_v, angle_variant=ang_v,
               rates={u: 100 * D[u].mean() for u in utils + ['random', 'oracle']}, diffs={})
    res['relative'] = {u: 100 * D[u].mean() / orc for u in utils + ['random']}
    print(f'== {a.tier} tier, {method} landmarks, h {a.mm}: {len(D)} decisions, {res["rooms"]} rooms, '
          f'{len(R)} records')
    print(f'   random {100 * rnd:.1f}  oracle {100 * orc:.1f}')
    for u in utils:
        print(f'   {u:<12} {100 * D[u].mean():5.1f}   relative to oracle {100 * D[u].mean() / orc:5.1f}')
    for x, y in PAIRS:
        if x in D and y in D:
            m, lo, hi = cluster_ci(D, x, y, a.level)
            res['diffs'][f'{x} - {y}'] = [m, lo, hi]
            print(f'   {x:>10} - {y:<8} {fmt_diff(m, lo, hi)}')
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
