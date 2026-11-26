"""Local geometric features of point clouds and the sparse-map construction."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

# The nine curvature invariants used as per-point inputs of DetField.
FEATS9 = ['gf_fro16', 'gf_fro32', 'gf_fro64', 'gf_g2_16', 'gf_g2_32', 'gf_g2_64', 'gf_sv16', 'gf_sv32', 'gf_sv64']


def curv(pts, nb, B=40000):
    """Quadric fit in the tangent frame of each neighbourhood: returns
    ||H||_F^2 = fxx^2 + 2 fxy^2 + fyy^2 and (fxx fyy - fxy^2)^2."""
    fro = np.empty(len(nb))
    g2 = np.empty(len(nb))
    for s in range(0, len(nb), B):
        P = pts[nb[s:s + B]]
        P = P - P.mean(1, keepdims=True)
        _, evec = np.linalg.eigh(np.einsum('mki,mkj->mij', P, P) / P.shape[1])
        Q = np.einsum('mki,mij->mkj', P, evec[:, :, ::-1])
        x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
        A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
        AtA = np.einsum('mki,mkj->mij', A, A) + np.eye(6)[None] * 1e-12
        c = np.linalg.solve(AtA, np.einsum('mki,mk->mi', A, z)[..., None])[..., 0]
        fxx, fxy, fyy = 2 * c[:, 3], c[:, 4], 2 * c[:, 5]
        fro[s:s + B] = fxx ** 2 + 2 * fxy ** 2 + fyy ** 2
        g2[s:s + B] = (fxx * fyy - fxy ** 2) ** 2
    f = lambda a: np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return f(fro), f(g2)


def harris_sv(pts, nb, k_h=0.04, B=40000):
    """Harris-3D response and surface variation (smallest eigenvalue / trace) of each neighbourhood."""
    h = np.empty(len(nb))
    sv = np.empty(len(nb))
    for s in range(0, len(nb), B):
        P = pts[nb[s:s + B]]
        P = P - P.mean(1, keepdims=True)
        kk = P.shape[1]
        w, evec = np.linalg.eigh(np.einsum('mki,mkj->mij', P, P) / kk)
        sv[s:s + B] = w[:, 0] / np.maximum(w.sum(1), 1e-18)
        Q = np.einsum('mki,mij->mkj', P, evec[:, :, ::-1])
        x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
        A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
        AtA = np.einsum('mki,mkj->mij', A, A) + np.eye(6)[None] * 1e-12
        c = np.linalg.solve(AtA, np.einsum('mki,mk->mi', A, z)[..., None])[..., 0]
        fx, fy, fxx, fxy, fyy = c[:, 1], c[:, 2], 2 * c[:, 3], c[:, 4], 2 * c[:, 5]
        EA = fx * fx + 2 * fxx * fxx + 2 * fxy * fxy
        EB = fy * fy + 2 * fyy * fyy + 2 * fxy * fxy
        EC = fx * fy + 2 * fxx * fxy + 2 * fxy * fyy
        h[s:s + B] = (EA * EB - EC * EC) - k_h * (EA + EB) ** 2
    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0), sv


def feats9_dense(P, workers=4):
    """The nine features on a dense point cloud (as stored next to each sample, float32)."""
    _, nb64 = cKDTree(P).query(P, 64, workers=workers)
    out = {}
    for kk in (16, 32, 64):
        out[f'gf_fro{kk}'], out[f'gf_g2_{kk}'] = curv(P, nb64[:, :kk])
        out[f'gf_sv{kk}'] = harris_sv(P, nb64[:, :kk])[1]
    return {k: v.astype(np.float32) for k, v in out.items()}


def feats9(P):
    """The nine features on a sparse point set (float64, not ranked)."""
    k = min(64, len(P))
    _, nb = cKDTree(P).query(P, k, workers=4)
    out = {}
    for kk in (16, 32, 64):
        out[f'gf_fro{kk}'], out[f'gf_g2_{kk}'] = curv(P, nb[:, :kk])
        out[f'gf_sv{kk}'] = harris_sv(P, nb[:, :kk])[1]
    return out


def rank_feats(fd, n):
    return np.stack([rankdata(fd[k].astype(np.float64)) / n for k in FEATS9], 1)


def crop_scale(T):
    """Scale s of a sample transform (pc = s * world + t)."""
    return float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))


def voxel_subsample(P_norm, T, spacing_m, seed=0):
    """Sparse map: one original point per world-space voxel of edge `spacing_m`. Returns sorted indices."""
    s = crop_scale(T)
    Pw = (P_norm - T[:3, 3]) / s
    perm = np.random.default_rng(seed).permutation(len(P_norm))
    key = np.floor(Pw[perm] / spacing_m).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return np.sort(perm[first])


def voxel_subsample_norm(P, vox, rng):
    perm = rng.permutation(len(P))
    _, first = np.unique(np.floor(P[perm] / vox).astype(np.int64), axis=0, return_index=True)
    return np.sort(perm[first])


def sparse_object(X, s, rng, max_enc=20000):
    """Sparse object point set with voxel edge `s`, and its ranked features."""
    S = X[voxel_subsample_norm(X, s, rng)][:max_enc]
    f = feats9(S)
    feat = np.stack([rankdata(f[k]) / len(S) for k in FEATS9], 1)
    return S, feat


def normals(P, k=16):
    _, nb = cKDTree(P).query(P, min(k, len(P)), workers=4)
    Q = P[nb] - P[nb].mean(1, keepdims=True)
    _, ev = np.linalg.eigh(np.einsum('mki,mkj->mij', Q, Q))
    return ev[:, :, 0], ev[:, :, 1], ev[:, :, 2]


def candidates(P, spacing, m, rng):
    """Query positions between sparse points: m samples per point in its tangent disc
    (radius 0.75 spacing, normal offset up to 0.25 spacing). The points themselves come first.
    Returns (positions, index of the owning sparse point)."""
    n, t1, t2 = normals(P)
    N = len(P)
    r = 0.75 * spacing * np.sqrt(rng.random((N, m)))
    th = 2 * np.pi * rng.random((N, m))
    h = 0.25 * spacing * (2 * rng.random((N, m)) - 1)
    off = (r * np.cos(th))[..., None] * t1[:, None] + (r * np.sin(th))[..., None] * t2[:, None] + h[..., None] * n[:, None]
    C = np.concatenate([P, (P[:, None] + off).reshape(-1, 3)])
    own = np.concatenate([np.arange(N), np.repeat(np.arange(N), m)])
    return C, own


def rank01(a):
    """Within-sample rank in [0, 1] (stable ties)."""
    a = np.asarray(a, np.float64).reshape(-1)
    r = np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(np.float64)
    return r / max(len(a) - 1, 1)


def rank_avg(a):
    """Average-tie rank / n, NaN treated as lowest."""
    a = np.nan_to_num(np.asarray(a, np.float64), nan=-np.inf)
    return (rankdata(a) / len(a)).astype(np.float32)
