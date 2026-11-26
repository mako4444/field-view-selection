"""LWL (Di Giammarino et al., ECCV 2024) retrained on the outcomes of our localizer.

Input: 30 x 30 bins over the candidate image, each holding the mean of [u, v, x, y, z] of the landmarks
that fall into it (2d, 2dz or 2d3d channel sets), standardised per channel with empty bins set to 0
before standardisation. MLP 300-300-1 with dropout 0.5, weighted BCE, Adam, input noise of 1 px and
2 cm on occupied bins. The channel set, learning rate and epoch are chosen by decision-level top-1 on
the validation rooms; the chosen setting is then trained with seeds 0, 1, 2 and the test utility is
the mean predicted probability.

  train:   python -m fvs.planners.lwl train --train A,B --val V --grid grid_all.npy --out DIR
  predict: python -m fvs.planners.lwl predict --models DIR --data T --grid grid_all.npy --out pred.pkl
Each data directory holds rec.pkl and the grid file with aligned rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CH = {'u': 0, 'v': 1, 'x': 2, 'y': 3, 'z': 4}
INPUTS = {'2d': ['u', 'v'], '2dz': ['u', 'v', 'z'], '2d3d': ['u', 'v', 'x', 'y', 'z']}
NOISE = {'u': 1.0 / 960.0, 'v': 1.0 / 960.0 * 960 / 720, 'x': 0.02, 'y': 0.02, 'z': 0.02}


def load(d, grid):
    Rs, Gs = [], []
    for x in d.split(','):
        Rs.append(pd.read_pickle(Path(x) / 'rec.pkl'))
        Gs.append(np.load(Path(x) / grid, mmap_mode='r'))
    if len(Rs) == 1:
        return Rs[0], Gs[0]
    return pd.concat(Rs, ignore_index=True), np.concatenate([np.asarray(g) for g in Gs])


def top1(R, u):
    s = []
    for k, g in R.groupby(['crop', 'dec', 'K'] + (['mm'] if 'mm' in R else []), sort=False):
        v = g[u].values
        s.append((k[0], k[1], g.ok.values[v == v.max()].astype(float).mean()))
    S = pd.DataFrame(s, columns=['crop', 'dec', 't']).groupby(['crop', 'dec']).t.mean()
    return float(S.mean())


class MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.f = nn.Sequential(nn.Dropout(0.5), nn.Linear(n, 300), nn.ReLU(), nn.Dropout(0.5),
                               nn.Linear(300, 300), nn.ReLU(), nn.Dropout(0.5), nn.Linear(300, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def prep(G, cidx, mu, sd):
    X = np.asarray(G[..., cidx], np.float32)
    occ = np.asarray(G[..., 7], np.float32) > 0
    X = ((X - mu) / sd).astype(np.float32)
    return X.reshape(len(X), -1), occ


def train_eval(tr, va, cidx, lr, seed, epochs, dev, noise_sd, eval_every=10):
    (Rt, Gt), (Rv, Gv) = tr, va
    Xa = np.asarray(Gt[..., cidx], np.float32).reshape(-1, len(cidx))
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Xt, occt = prep(Gt, cidx, mu, sd)
    Xv, _ = prep(Gv, cidx, mu, sd)
    yt = Rt.ok.values.astype(np.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = MLP(Xt.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    pw = torch.tensor((1 - yt.mean()) / max(yt.mean(), 1e-6), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt_t = torch.from_numpy(Xt).to(dev)
    yt_t = torch.from_numpy(yt).to(dev)
    occ_t = torch.from_numpy(occt.reshape(len(occt), -1)).to(dev)
    nsd = torch.from_numpy((noise_sd / sd).astype(np.float32)).to(dev).repeat(occt.shape[1] * occt.shape[2])
    Xv_t = torch.from_numpy(Xv).to(dev)
    hist, best = [], (-1, None, None)
    nc = len(cidx)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(Xt_t), device=dev)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            x = Xt_t[b]
            mask = occ_t[b].repeat_interleave(nc, 1)
            x = x + torch.randn_like(x) * nsd * mask
            loss = lossf(m(x), yt_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % eval_every == 0:
            m.eval()
            with torch.no_grad():
                pv = torch.sigmoid(m(Xv_t)).cpu().numpy()
            t = top1(Rv.assign(u=pv), 'u')
            hist.append((ep, t))
            if t > best[0]:
                best = (t, ep, {k: v.detach().clone() for k, v in m.state_dict().items()})
    return hist, best, (mu, sd)


@torch.no_grad()
def predict_one(state, n, G, cidx, mu, sd, dev):
    m = MLP(n).to(dev)
    m.load_state_dict(state)
    m.eval()
    X, _ = prep(G, cidx, mu, sd)
    out = []
    for i in range(0, len(X), 4096):
        out.append(torch.sigmoid(m(torch.from_numpy(X[i:i + 4096]).to(dev))).cpu().numpy())
    return np.concatenate(out)


def cmd_train(a, dev):
    tr, va = load(a.train, a.grid), load(a.val, a.grid)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]
    sel = {}
    for name, names in INPUTS.items():
        cidx = [CH[n] for n in names]
        nsd = np.array([NOISE[n] for n in names], np.float32)
        best = None
        for lr in (1e-3, 1e-4):
            hist, b, _ = train_eval(tr, va, cidx, lr, seeds[0], a.epochs, dev, nsd)
            print(f'{name} lr {lr}: ' + ' '.join(f'{e}:{100 * t:.1f}' for e, t in hist), flush=True)
            if best is None or b[0] > best[0]:
                best = (b[0], b[1], lr)
        sel[name] = dict(chans=names, lr=best[2], epoch=best[1], val_top1=best[0])
    base = max(INPUTS, key=lambda n: sel[n]['val_top1'])
    s = sel[base]
    print(f'chosen: {base} lr {s["lr"]} epoch {s["epoch"]}', flush=True)
    cidx = [CH[n] for n in s['chans']]
    nsd = np.array([NOISE[n] for n in s['chans']], np.float32)
    for sd_ in seeds:
        _, b, (mu, sdv) = train_eval(tr, va, cidx, s['lr'], sd_, s['epoch'], dev, nsd, eval_every=s['epoch'])
        torch.save(dict(state=b[2], mu=mu, sd=sdv, cidx=cidx), out / f'lwl_seed{sd_}.pt')
    json.dump(dict(selection=sel, chosen=base, seeds=seeds), open(out / 'select.json', 'w'), indent=1)


def cmd_predict(a, dev):
    R, G = load(a.data, a.grid)
    sel = json.load(open(Path(a.models) / 'select.json'))
    P = []
    for sd_ in sel['seeds']:
        M = torch.load(Path(a.models) / f'lwl_seed{sd_}.pt', map_location='cpu', weights_only=False)
        P.append(predict_one(M['state'], len(M['cidx']) * 900, G, M['cidx'], M['mu'], M['sd'], dev))
    R = R.copy()
    R['u_lwl'] = np.mean(P, 0)
    R.to_pickle(a.out)
    print('->', a.out, len(R))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['train', 'predict'])
    ap.add_argument('--train', default='')
    ap.add_argument('--val', default='')
    ap.add_argument('--data', default='')
    ap.add_argument('--models', default='')
    ap.add_argument('--grid', default='grid_all.npy')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    if a.cmd == 'train':
        cmd_train(a, dev)
    else:
        cmd_predict(a, dev)


if __name__ == '__main__':
    main()
