"""Top-1 success of the planners in the object experiment and paired differences with 95% intervals
from a bootstrap over objects. A decision is (object, h, theta, j): the planner chooses among the
candidates at the same offset j around the different reference directions. Top-1 success is averaged
over K within a decision. Writes a JSON file used by the table scripts.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from .stats import cluster_ci, fmt_diff

UTILS = ['field_sum', 'cnt', 'harris_matched', 'field_permuted', 'harris_sum', 'n_co']
PAIRS = [('field_sum', 'cnt'), ('field_sum', 'harris_sum'), ('field_sum', 'field_permuted'),
         ('field_sum', 'harris_matched'), ('harris_matched', 'harris_sum')]


def decide(S, util_col, gk, Ks):
    out = {}
    for key, g in S[S.K.isin(Ks)].groupby(gk + ['K']):
        r = out.setdefault(key[:-1], {})
        ok = g.ok.values
        for u in UTILS:
            v = g[util_col(u, key[-1])].values
            r.setdefault(u, []).append(ok[v == v.max()].astype(float).mean())
        r.setdefault('random', []).append(g.ok.astype(float).mean())
        r.setdefault('oracle', []).append(float(g.ok.astype(bool).any()))
    D = pd.DataFrame({k: {u: np.mean(v) for u, v in r.items()} for k, r in out.items()}).T
    D.index = pd.MultiIndex.from_tuples(D.index, names=['crop'] + gk[1:])
    return D


def block(D):
    res = dict(decisions=len(D), rates={u: 100 * D[u].mean() for u in D.columns}, diffs={})
    for x, y in PAIRS:
        res['diffs'][f'{x} - {y}'] = list(cluster_ci(D, x, y, 'obj'))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('rows', help='replay output directory')
    ap.add_argument('--kind', choices=['sparse', 'dense'], required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = {}
    if a.kind == 'sparse':
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{a.rows}/rows_s*.csv'))])
        R = R.drop_duplicates(['model', 's', 'method', 'K', 'ref', 'qry'])
        R['ok'] = R['ok'].astype(float)
        Ks, sps = [16, 32, 64], sorted(R.s.unique())
        for fam in ('field', 'harris'):
            S = R[R.method == fam]
            for sset, lab in ((sps, 'pooled'),) + tuple(((x,), f'h{x:g}') for x in sps):
                D = decide(S[S.s.isin(sset)], lambda u, K: u, ['model', 's', 'theta', 'j'], Ks)
                res[f'{fam}/{lab}'] = block(D)
    else:
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{a.rows}/rows*.csv'))])
        Ks, ths = [32, 64, 128], [10.0, 15.0]
        for lm in ('field', 'harris'):
            S = R[(R.method == lm) & R.theta.isin(ths)]
            D = decide(S, lambda u, K, lm=lm: 'n_co' if u == 'n_co' else f'{u}|{lm}_K{K}',
                       ['crop', 'j', 'theta'], Ks)
            res[f'{lm}/pooled'] = block(D)
    for k, b in res.items():
        print(f'-- {k}: {b["decisions"]} decisions  ' + '  '.join(f'{u} {v:.1f}' for u, v in b['rates'].items()))
        for p, (m, lo, hi) in b['diffs'].items():
            print(f'   {p:<36} {fmt_diff(m, lo, hi)}')
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
