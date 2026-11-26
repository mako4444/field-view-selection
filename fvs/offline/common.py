"""Shared pieces of the offline planner inputs."""
from __future__ import annotations

import glob
import pickle

import numpy as np

NB = 30   # LWL bins per image side
GEO = ['z', 'ang_ax', 'un', 'vn', 'rc', 'd_ref', 'ang_view', 'lscale', 'inc_q', 'inc_r', 'dens_img', 'dens3d', 'n_vis']
FIF_VARIANTS = [f'fif_{s}_{v}' for s in ('ld', 'me', 'tr') for v in ('ex', 'qa')]


def load_landmarks(d, key_fields):
    """Per-landmark replay records of all shards, without duplicates."""
    L = []
    for fp in sorted(glob.glob(f'{d}/lm_s*.pkl')):
        L += pickle.load(open(fp, 'rb'))
    seen, out = set(), []
    for r in L:
        k = tuple(str(r[f]) if f in ('dec', 'qry') else r[f] for f in key_fields)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def skew(p):
    z = np.zeros(len(p))
    return np.stack([np.stack([z, -p[:, 2], p[:, 1]], 1),
                     np.stack([p[:, 2], z, -p[:, 0]], 1),
                     np.stack([-p[:, 1], p[:, 0], z], 1)], 1)


def fif(Xw, Twc, alpha=np.pi / 4):
    """Fisher information field (Zhang and Scaramuzza): log-det, smallest eigenvalue and trace of the
    bearing information of the visible landmarks, with exact or quadratic visibility weighting (sigma = 1)."""
    if len(Xw) < 3:
        return {f'fif_{s}_{v}': -1e9 for s in ('ld', 'me', 'tr') for v in ('ex', 'qa')}
    Rwc, twc = Twc[:3, :3], Twc[:3, 3]
    Rcw = Rwc.T
    pc = (Xw - twc) @ Rcw.T
    n = np.linalg.norm(pc, axis=1)
    f = pc / n[:, None]
    dfdpc = (np.eye(3)[None] - f[:, :, None] * f[:, None, :]) / n[:, None, None]
    A = np.concatenate([-np.broadcast_to(np.eye(3), (len(Xw), 3, 3)), skew(Xw)], 2)
    J = dfdpc @ Rcw[None] @ A
    JtJ = np.einsum('nai,naj->nij', J, J)
    c = f[:, 2]
    k1, k2, k3 = 0.7071, 0.5, -0.2071
    vis = {'ex': (c > np.cos(alpha)).astype(np.float64), 'qa': k1 * c ** 2 + k2 * c + k3}
    out = {}
    for k, v in vis.items():
        I = np.einsum('n,nij->ij', v, JtJ)
        sgn, ld = np.linalg.slogdet(I + 1e-9 * np.eye(6))
        out[f'fif_ld_{k}'] = float(ld) if sgn > 0 else -1e9
        out[f'fif_me_{k}'] = float(np.linalg.svd(I, compute_uv=False)[-1])
        out[f'fif_tr_{k}'] = float(np.trace(I))
    return out


def bin_grid(u, vv, Xc, extra, W, H):
    """LWL input of one candidate: per-bin mean of [u/W, v/H, x, y, z, extra[:2], count, extra[2:]]."""
    G = np.zeros((NB * NB, 17), np.float64)
    if len(u):
        bu = np.clip((u / W * NB).astype(int), 0, NB - 1)
        bv = np.clip((vv / H * NB).astype(int), 0, NB - 1)
        ch = np.concatenate([np.stack([u / W, vv / H, Xc[:, 0], Xc[:, 1], Xc[:, 2]], 1), extra[:, :2],
                             np.ones((len(u), 1)), extra[:, 2:]], 1)
        flat = bv * NB + bu
        for c in range(17):
            G[:, c] = np.bincount(flat, weights=ch[:, c], minlength=NB * NB)
        nz = G[:, 7] > 0
        G[nz] = np.where(np.arange(17) == 7, G[nz], G[nz] / G[nz][:, 7:8])
    return G.reshape(NB, NB, 17).astype(np.float16)
