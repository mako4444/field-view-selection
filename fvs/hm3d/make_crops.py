"""Crops of the HM3D room videos, built like the SceneFun3D crops: pose clusters along each video
(15 degrees / 0.5 m, 5 frames with stride 2), screening of extent [2, 6.5] m and planarity, the crop mesh
inside the frustums of the cluster (margin 0.3 m), and 480000 points per unit area sampled on the mesh
normalised to the unit cube. The screening uses points sampled uniformly on the glb surface. At most 30
crops per scene are drawn with seed 0.

python -m fvs.hm3d.make_crops 00017-oEPjPNSPmzL 00270-bDTsgcSK5Qr [--capture ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import paths
from ..render import normalize_points_to_unit_cube, repair_mesh_normals_full
from ..sf3d import aa2R, ts
from .glb import load_glb

ZMAX_CAP = 6.6
MAX_CROPS = 30
CROP_SEED = 0
SCREEN_DENSITY = 36000.0
CLUSTER_ANG, CLUSTER_DIST, SEED_GAP, MIN_PAIRS, FRAMES, STRIDE = 15.0, 0.5, 0.5, 1, 5, 2
ZMAX_PCT, MARGIN, EXT_LO, EXT_HI, MIN_LAM3, MIN_SCREEN = 99.0, 0.30, 2.0, 6.5, 0.03, 20000
PC_DENSITY, PC_MIN, PC_MAX, SAMPLE_SEED, CREASE = 480000.0, 2000, 700000, 42, 30.0


def read_capture(d):
    traj = {}
    for l in open(d / 'hires_poses.traj'):
        t = l.split()
        E = np.eye(4)
        E[:3, :3] = aa2R(np.array(list(map(float, t[1:4]))))
        E[:3, 3] = list(map(float, t[4:7]))
        traj[round(float(t[0]), 3)] = E
    names = sorted(f.stem for f in (d / 'hires_depth').glob('*.png'))
    return traj, [n for n in names if ts(n) in traj and (d / 'hires_wide' / f'{n}.jpg').exists()]


def intr6(d, name):
    return tuple(map(float, (d / 'hires_wide_intrinsics' / f'{name}.pincam').read_text().split()))


def pose_clusters(traj, names, ang_w, dist_w, seed_gap, min_pairs, pair_ang=(5.0, 10.0), pair_dist=(0.15, 1.0),
                  n_frames=0, stride=3):
    """Seed frames along the trajectory, each collecting its pose neighbours; returns [(seed, idx, pairs)]."""
    C = np.stack([-(traj[ts(n)][:3, :3].T @ traj[ts(n)][:3, 3]) for n in names])
    R = np.stack([traj[ts(n)][:3, :3] for n in names])
    seeds, last = [], None
    for i in range(len(names)):
        if last is None or np.linalg.norm(C[i] - C[last]) >= seed_gap:
            seeds.append(i)
            last = i
    out = []
    for i in seeds:
        tr = np.degrees(np.arccos(np.clip((np.trace(R[i].T @ R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        dd = np.linalg.norm(C - C[i], axis=1)
        idx = np.where((tr <= ang_w) & (dd <= dist_w))[0]
        if n_frames > 0 and len(idx) > n_frames:
            pos = int(np.searchsorted(idx, i))
            picked = {int(idx[min(pos, len(idx) - 1)])}
            k = 1
            while len(picked) < n_frames and k * stride < len(idx):
                for sgn in (1, -1):
                    q = pos + sgn * k * stride
                    if 0 <= q < len(idx):
                        picked.add(int(idx[q]))
                    if len(picked) >= n_frames:
                        break
                k += 1
            idx = np.array(sorted(picked))
        pairs = []
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                a, b = idx[x], idx[y]
                an = np.degrees(np.arccos(np.clip((np.trace(R[a].T @ R[b]) - 1) / 2, -1, 1)))
                ds = np.linalg.norm(C[a] - C[b])
                if pair_ang[0] <= an < pair_ang[1] and pair_dist[0] <= ds <= pair_dist[1]:
                    pairs.append((int(a), int(b)))
        if len(pairs) >= min_pairs:
            out.append((i, idx, pairs))
    return out


def adaptive_zmax(idx, names, d, pct, n_sample=6):
    zs = []
    for j in idx[::max(1, len(idx) // n_sample)]:
        dp = cv2.imread(str(d / 'hires_depth' / f'{names[j]}.png'), -1)
        if dp is None:
            continue
        v = dp[dp > 200].astype(np.float32) / 1000.0
        if len(v):
            zs.append(v)
    return float(np.percentile(np.concatenate(zs), pct)) if zs else 5.0


def cluster_frustum_mask(V, idx, names, traj, d, zmax, margin):
    keep = np.zeros(len(V), bool)
    for j in idx:
        E = traj[ts(names[j])]
        W, H, fx, fy, cx, cy = intr6(d, names[j])
        Q = V @ E[:3, :3].T + E[:3, 3]
        z = Q[:, 2]
        u = fx * Q[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * Q[:, 1] / np.maximum(z, 1e-6) + cy
        mx = margin * fx / np.maximum(z, 1e-6)
        my = margin * fy / np.maximum(z, 1e-6)
        keep |= ((z > max(0.3 - margin, 0.05)) & (z < zmax + margin) &
                 (u > -mx) & (u < W + mx) & (v > -my) & (v < H + my))
    return keep


def surface_sample_np(V, F, dens, seed=0):
    tri = V[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    n = int(a.sum() * dens)
    f = rng.choice(len(F), n, p=a / a.sum())
    u, v = rng.random((2, n))
    su = np.sqrt(u)
    return (1 - su)[:, None] * tri[f, 0] + (su * (1 - v))[:, None] * tri[f, 1] + (su * v)[:, None] * tri[f, 2]


def sample_surface(verts, faces, cnorm, n, gen):
    """Area-weighted surface points with crease normals interpolated barycentrically (inverse-CDF sampling)."""
    fv = verts[faces]
    area = 0.5 * torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=1).norm(dim=1)
    cdf = torch.cumsum(area / area.sum(), dim=0)
    cdf = cdf / cdf[-1].clamp_min(1e-20)
    r = torch.rand(n, device=verts.device, generator=gen)
    fid = torch.searchsorted(cdf, r).clamp_max(area.numel() - 1)
    u = torch.rand((n, 1), device=verts.device, generator=gen)
    v = torch.rand((n, 1), device=verts.device, generator=gen)
    su = u.sqrt()
    b = torch.cat([1.0 - su, su * (1.0 - v), su * v], dim=1)
    pts = (b[:, :, None] * fv[fid]).sum(1)
    nrm = (b[:, :, None] * cnorm[fid]).sum(1)
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return pts, nrm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scenes', nargs='+')
    ap.add_argument('--trajectories', default=str(paths.HM3D / 'trajectories'))
    ap.add_argument('--capture', default=str(paths.OUT / 'hm3d/capture'))
    ap.add_argument('--scenes_root', default=str(paths.HM3D_ACTLOC))
    ap.add_argument('--out', default=str(paths.OUT / 'hm3d/crops'))
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for scene in a.scenes:
        visit = scene.split('-')[0]
        cap = Path(a.capture) / visit
        videos = [k['video'] for k in json.load(open(Path(a.trajectories) / visit / 'rooms.json'))['kept']]
        V, F, _, _, _ = load_glb(next((Path(a.scenes_root) / scene).glob('*.glb')))
        V = V.astype(np.float64)
        P = surface_sample_np(V, F, SCREEN_DENSITY)
        ok, caps = [], {}
        for video in videos:
            d = cap / video
            traj, names = read_capture(d)
            caps[video] = (d, traj, names)
            cls = pose_clusters(traj, names, CLUSTER_ANG, CLUSTER_DIST, SEED_GAP, MIN_PAIRS, n_frames=FRAMES,
                                stride=STRIDE)
            for ci, (seed, idx, pr) in enumerate(cls):
                z99 = adaptive_zmax(idx, names, d, ZMAX_PCT)
                zmax = min(z99, ZMAX_CAP)
                km = cluster_frustum_mask(P, idx, names, traj, d, zmax, 0.0)
                if km.sum() < MIN_SCREEN:
                    continue
                Pc = P[km]
                ext = float(np.ptp(Pc, 0).max())
                Q = Pc - Pc.mean(0)
                ev = np.sort(np.linalg.eigvalsh(Q.T @ Q / len(Q)))[::-1]
                lam3 = float(ev[2] / ev.sum())
                if EXT_LO <= ext <= EXT_HI and lam3 >= MIN_LAM3:
                    ok.append((video, ci, seed, idx, zmax))
        pick = sorted(np.random.default_rng(CROP_SEED).choice(len(ok), min(MAX_CROPS, len(ok)), replace=False)) if ok else []
        for k in pick:
            video, ci, seed, idx, zmax = ok[k]
            d, traj, names = caps[video]
            vm = cluster_frustum_mask(V, idx, names, traj, d, zmax, MARGIN)
            fm = vm[F].all(1)
            cname = f'c{k:03d}'
            if fm.sum() < 1000:
                continue
            remap = -np.ones(len(V), np.int64)
            remap[vm] = np.arange(vm.sum())
            Vt = torch.from_numpy(V[vm]).float().to(dev)
            Ft = torch.from_numpy(remap[F[fm]]).long().to(dev)
            vn, T = normalize_points_to_unit_cube(Vt, padding=0.1)
            F2, cn, fo, st = repair_mesh_normals_full(vn, Ft, crease_deg=CREASE)
            fv = vn[F2]
            area = float(0.5 * torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=1).norm(dim=1).sum())
            n_pts = int(np.clip(PC_DENSITY * area, PC_MIN, PC_MAX))
            g = torch.Generator(device=dev)
            g.manual_seed(SAMPLE_SEED)
            pc, pn = sample_surface(vn, F2, cn, n_pts, g)
            dst = out / visit / cname
            dst.mkdir(parents=True, exist_ok=True)
            np.save(dst / 'pc.npy', pc.cpu().numpy().astype(np.float32))
            np.save(dst / 'pc_normal.npy', pn.cpu().numpy().astype(np.float16))
            np.savetxt(dst / 'transform.txt', T.cpu().numpy(), fmt='%1.8f')
            json.dump({'visit': visit, 'video': video}, open(dst / 'meta.json', 'w'))
            print(f'  {visit}/{cname} ({video} cluster {ci}): {n_pts} points', flush=True)


if __name__ == '__main__':
    main()
