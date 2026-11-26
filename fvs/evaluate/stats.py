"""Decision-level top-1 success and clustered bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def top1(ok, util):
    """Success of the candidate with the highest utility; ties are averaged."""
    return ok[util == util.max()].astype(float).mean()


def per_decision(R, utils, inner=('K',)):
    """Top-1 success of every utility per decision (crop, dec), averaged over the `inner` groups
    (landmark budget K, and the map spacing h in the sparse tier). Adds `random` and `oracle`."""
    out = {}
    for key, g in R.groupby(['crop', 'dec'] + list(inner), sort=False):
        r = out.setdefault(key[:2], {u: [] for u in list(utils) + ['random', 'oracle']})
        ok = g.ok.values
        for u in utils:
            r[u].append(top1(ok, g[u].values))
        r['random'].append(ok.astype(float).mean())
        r['oracle'].append(float(ok.max()))
    D = pd.DataFrame({k: {u: np.mean(v) for u, v in r_.items()} for k, r_ in out.items()}).T
    D.index = pd.MultiIndex.from_tuples(D.index, names=['crop', 'dec'])
    return D


def cluster_ci(D, a, b, level='room', n=20000, seed=0):
    """Mean difference a - b (in points) with a 95% bootstrap interval over clusters
    (room = visit, or crop; for objects: object or category)."""
    d = (D[a] - D[b]).dropna()
    crops = d.index.get_level_values('crop')
    if level == 'room':
        keys = [c.split('/')[0] for c in crops]
    elif level == 'cat':
        keys = [c.split('/')[0] for c in crops]
    else:
        keys = list(crops)
    G = pd.DataFrame({'k': keys, 'd': d.values}).groupby('k').d
    s, w = G.sum().values, G.size().values
    idx = np.random.RandomState(seed).randint(0, len(s), (n, len(s)))
    bb = s[idx].sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(bb, [2.5, 97.5])
    return 100 * d.mean(), 100 * lo, 100 * hi


def fmt_diff(m, lo, hi):
    return f'{m:+5.1f} [{lo:+.1f}, {hi:+.1f}]' + ('*' if lo > 0 or hi < 0 else '')
