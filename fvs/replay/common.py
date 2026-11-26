"""Offline-replay protocol of the indoor experiments: decisions, candidates, map frames and the
planner-side visibility. A decision is an anchor frame of a real video; its candidates are up to M
frames near the anchor with distinct viewing directions, and the map consists of the frames of the
same video recorded more than `gap` seconds away from every candidate."""
from __future__ import annotations

import argparse

import numpy as np
import torch

PROTOCOL = dict(n_dec=30, gap=0.5, ref_ang=20.0, ref_dt=0.6, R=1.0, min_sep=12.0, min_cand=4, anchor_sep=0.25,
                M=8, cov_min=0.15, map_stride=1, zdown=4, seed=0)


def add_protocol_args(ap: argparse.ArgumentParser):
    for k, v in PROTOCOL.items():
        ap.add_argument(f'--{k}', type=type(v), default=v)
    ap.add_argument('--Ks', default='64,80,96', help='landmark budgets')
    ap.add_argument('--shard', default='0/1', help='i/n: process crops i, i+n, ... of the list; the decision '
                                                   'seed is seed + position within the shard')


def zbuf_vis(Pw_t, T_c2w, Ks, hw, down, tol, dev):
    """Point visibility from a z-buffer of the points themselves (cells of `down` pixels)."""
    H, W = hw
    E = torch.from_numpy(np.linalg.inv(T_c2w)).float().to(dev)
    Q = Pw_t @ E[:3, :3].T + E[:3, 3]
    z = Q[:, 2]
    u = Ks[0, 0] * Q[:, 0] / z.clamp_min(1e-6) + Ks[0, 2]
    v = Ks[1, 1] * Q[:, 1] / z.clamp_min(1e-6) + Ks[1, 2]
    inb = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    gw, gh = int(np.ceil(W / down)), int(np.ceil(H / down))
    cell = (v.clamp(0, H - 1) // down).long() * gw + (u.clamp(0, W - 1) // down).long()
    zz = torch.where(inb, z, torch.full_like(z, 1e9))
    zmin = torch.full((gw * gh,), 1e9, device=dev)
    zmin = zmin.scatter_reduce(0, cell, zz, reduce='amin')
    return (inb & (z <= zmin[cell] + tol)).cpu().numpy()


def splat_vis(S_t, X_t, T_c2w, Ks, hw, spacing, down, tol, dev, rmax=3):
    """Visibility of X against a z-buffer in which every sparse point S covers a square of its voxel size."""
    H, W = hw
    E = torch.from_numpy(np.linalg.inv(T_c2w)).float().to(dev)
    gw, gh = int(np.ceil(W / down)), int(np.ceil(H / down))

    def proj(P):
        Q = P @ E[:3, :3].T + E[:3, 3]
        z = Q[:, 2]
        u = Ks[0, 0] * Q[:, 0] / z.clamp_min(1e-6) + Ks[0, 2]
        v = Ks[1, 1] * Q[:, 1] / z.clamp_min(1e-6) + Ks[1, 2]
        return u, v, z

    u, v, z = proj(S_t)
    ok = z > 0.1
    rc = torch.clamp(torch.round(Ks[0, 0] * spacing / z.clamp_min(0.1) / (2 * down)), 0, rmax).long()
    cu, cv = torch.floor(u / down).long(), torch.floor(v / down).long()
    zmin = torch.full((gw * gh,), 1e9, device=dev)
    for dy in range(-rmax, rmax + 1):
        for dx in range(-rmax, rmax + 1):
            m = ok & (rc >= max(abs(dx), abs(dy)))
            x_, y_ = cu + dx, cv + dy
            m &= (x_ >= 0) & (x_ < gw) & (y_ >= 0) & (y_ < gh)
            if m.any():
                zmin.scatter_reduce_(0, (y_[m] * gw + x_[m]), z[m], reduce='amin')
    u2, v2, z2 = proj(X_t)
    inb = (z2 > 0.1) & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    cell = (v2.clamp(0, H - 1) // down).long() * gw + (u2.clamp(0, W - 1) // down).long()
    return (inb & (z2 <= zmin[cell] + tol)).cpu().numpy()


def diverse_pick(cands, axes, min_sep, M):
    """Greedy farthest-direction selection with a minimum angle of `min_sep` degrees."""
    sel = [0]
    while len(sel) < M:
        A = np.degrees(np.arccos(np.clip(axes[cands] @ axes[[cands[s] for s in sel]].T, -1, 1)))
        dmin = A.min(1)
        dmin[sel] = -1
        j = int(np.argmax(dmin))
        if dmin[j] < min_sep:
            break
        sel.append(j)
    return [cands[s] for s in sel]


def make_decisions(crop, a, ci, dev):
    """Decisions of one crop: list of (anchor, candidate indices, map-frame indices) into `good`.
    The coverage filter and the anchors are computed on the dense crop geometry; this defines the
    experimental setting and is the same for every planner."""
    Pw_t = torch.from_numpy(crop.Pw.astype(np.float32)).to(dev)
    sub = np.random.RandomState(0).choice(len(crop.Pw), min(len(crop.Pw), 60000), replace=False)
    Ps_t = Pw_t[sub]
    cov, axis, ctr = {}, {}, {}
    for n in crop.names:
        Tn = crop.pose(n)
        cov[n] = float(zbuf_vis(Ps_t, Tn, crop.K, crop.hw, 8, 0.1, dev).mean())
        axis[n] = Tn[:3, 2] / np.linalg.norm(Tn[:3, 2])
        ctr[n] = Tn[:3, 3]
    from ..sf3d import ts
    good = [n for n in crop.names if cov[n] >= a.cov_min]
    if len(good) < 20:
        return good, [], cov
    A = np.stack([axis[n] for n in good])
    C = np.stack([ctr[n] for n in good])
    tg = np.array([ts(n) for n in good])
    Rg = np.stack([crop.pose(n)[:3, :3] for n in good])
    tr_ = np.einsum('aij,bij->ab', Rg, Rg)
    RA = np.degrees(np.arccos(np.clip((tr_ - 1) / 2, -1, 1)))
    DD = np.linalg.norm(C[:, None] - C[None], axis=2)
    TG = np.abs(tg[:, None] - tg[None])
    near_ok = (RA < a.ref_ang) & (DD < a.ref_dt)
    elig = (near_ok & (TG > a.gap)).any(1)
    rng = np.random.RandomState(a.seed + ci)
    order = rng.permutation(len(good))
    decisions, used_c = [], []
    for ia in order:
        if len(decisions) >= a.n_dec:
            break
        if not elig[ia] or any(np.linalg.norm(C[ia] - uc) < a.anchor_sep for uc in used_c):
            continue
        near = np.where((np.linalg.norm(C - C[ia], axis=1) < a.R) & elig)[0]
        near = [ia] + [j for j in near if j != ia]
        pick = diverse_pick(near, A, a.min_sep, a.M)
        cand_t = tg[pick]
        mpm = np.all(np.abs(tg[:, None] - cand_t[None]) > a.gap, 1)
        pick = [j for j in pick if (near_ok[j] & mpm).any()]
        if len(pick) < a.min_cand:
            continue
        mp = list(np.where(mpm)[0])
        mp = mp[::a.map_stride]
        if len(mp) < 5:
            continue
        decisions.append((ia, pick, mp))
        used_c.append(C[ia])
    return good, decisions, cov


def mapped_pool(crop, q, mpn, a):
    """Map frames within ref_ang / ref_dt of the candidate; all map frames if there is none."""
    Tq = crop.pose(q)
    inr = []
    for m in mpn:
        Tm = crop.pose(m)
        a_ = np.degrees(np.arccos(np.clip((np.trace(Tm[:3, :3].T @ Tq[:3, :3]) - 1) / 2, -1, 1)))
        if a_ < a.ref_ang and np.linalg.norm(Tm[:3, 3] - Tq[:3, 3]) < a.ref_dt:
            inr.append(m)
    mapped = len(inr) > 0
    return (inr if mapped else mpn), mapped


def shard_items(items, shard):
    si, sn = [int(x) for x in shard.split('/')]
    return items[si::sn], si, sn
