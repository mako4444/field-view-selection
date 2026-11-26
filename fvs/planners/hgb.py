"""Per-landmark learned planners: a gradient-boosted classifier predicts whether a landmark that the
planner sees will be matched correctly; the utility of a candidate is the sum of the predictions over
its visible landmarks. Arms differ only in their input features:
  G       geometric features (plus the map spacing h in the sparse tier)
  G+F     G + DetField value          G+H   G + Harris-3D rank
  G+C9    G + nine curvature invariants          G+F+C9
  G+P     G + ideal static field p (used by the diagnosis only)

  train:  python -m fvs.planners.hgb train --tier sparse --feats A/feat.pkl,B/feat.pkl --out DIR
  apply:  python -m fvs.planners.hgb apply --tier sparse --models DIR --feat C/feat.pkl --out u.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..geomfeat import FEATS9
from ..offline.common import GEO

HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
           random_state=0)
TRAIN_CAP = 400_000
TIER = {'sparse': dict(field='field', geo=GEO + ['mm'], key=['crop', 'dec', 'qry', 'mm', 'K', 'method'],
                       methods=('field', 'harris'), prop_key=['crop', 'mm']),
        'dense': dict(field='field', geo=GEO, key=['crop', 'dec', 'qry', 'K', 'method'],
                      methods=('harris',), prop_key=['crop'])}


def arms(tier, with_pbar=False):
    t = TIER[tier]
    G, f = t['geo'], t['field']
    A = {'G': G, 'G+F': G + [f], 'G+H': G + ['h'], 'G+C9': G + FEATS9, 'G+F+C9': G + [f] + FEATS9}
    if with_pbar:
        A['G+P'] = G + ['pbar']
    return A


def load_feats(path):
    D = pd.read_pickle(path)
    for c in ('crop', 'dec', 'qry', 'method'):
        D[c] = D[c].astype(str)
    return D


def ideal_static_field(D, tier, alpha=2.0):
    """Leave-one-decision-out fraction of correct matches of each landmark within its crop (and h),
    smoothed towards the crop mean with `alpha` pseudo-counts."""
    pk = TIER[tier]['prop_key']
    k = pk + ['lm']
    g = D.groupby(k, observed=True)['corr'].agg(['sum', 'count'])
    gd = D.groupby(k + ['dec'], observed=True)['corr'].agg(['sum', 'count'])
    prior = D.groupby(pk, observed=True)['corr'].mean().rename('prior')
    x = D[k + ['dec']].join(g, on=k).join(gd, on=k + ['dec'], rsuffix='_d').join(prior, on=pk)
    s = x['sum'] - x['sum_d']
    n = x['count'] - x['count_d']
    return ((s + alpha * x['prior']) / (n + alpha)).values.astype(np.float32)


def train(tier, feats, out, with_pbar=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Fs = [load_feats(f) for f in feats]
    methods = TIER[tier]['methods'][:1] if with_pbar else TIER[tier]['methods']
    for meth in methods:
        parts = []
        for F in Fs:
            X = F[F.method == meth].reset_index(drop=True)
            if with_pbar:
                X['pbar'] = ideal_static_field(X, tier)
            parts.append(X)
        tr = pd.concat(parts, ignore_index=True)
        tr = tr[tr.K == 96]
        if len(tr) > TRAIN_CAP:
            tr = tr.sample(TRAIN_CAP, random_state=0)
        for arm, cols in arms(tier, with_pbar).items():
            if arm != 'G+P' and with_pbar:
                continue
            m = HistGradientBoostingClassifier(**HGB).fit(tr[cols].values, tr['corr'].values)
            pickle.dump(dict(arm=arm, method=meth, cols=cols, model=m), open(out / f'hgb_{tier}_{meth}_{arm}.pkl', 'wb'))
            print(tier, meth, arm, 'trained', flush=True)


def predict(tier, models, Te, with_pbar=False):
    """Adds p_<arm> columns to the landmark table Te (one method)."""
    meth = Te.method.iloc[0]
    for arm in arms(tier, with_pbar):
        M = pickle.load(open(Path(models) / f'hgb_{tier}_{meth}_{arm}.pkl', 'rb'))
        Te[f'p_{arm}'] = M['model'].predict_proba(Te[M['cols']].values)[:, 1]
    return Te


def apply(tier, models, feat, methods=None):
    """Utility u_<arm> per record = sum of p_<arm> over its visible landmarks."""
    Te = load_feats(feat)
    outs = []
    for meth in methods or TIER[tier]['methods']:
        te = Te[Te.method == meth].copy()
        te = predict(tier, models, te)
        A = list(arms(tier))
        U = te.groupby(TIER[tier]['key'], observed=True)[[f'p_{x}' for x in A]].sum().reset_index()
        outs.append(U.rename(columns={f'p_{x}': f'u_{x}' for x in A}))
    return pd.concat(outs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['train', 'apply'])
    ap.add_argument('--tier', choices=['sparse', 'dense'], required=True)
    ap.add_argument('--feats', default='', help='training landmark tables (comma separated)')
    ap.add_argument('--feat', default='', help='landmark table to score')
    ap.add_argument('--models', default='')
    ap.add_argument('--with_pbar', action='store_true', help='train the G+P arm of the diagnosis')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a.tier, a.feats.split(','), a.out, a.with_pbar)
    else:
        U = apply(a.tier, a.models, a.feat)
        U.to_pickle(a.out)
        print('->', a.out, len(U))


if __name__ == '__main__':
    main()
