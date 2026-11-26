"""2D-3D localization against a single reference image: landmark selection, SuperPoint
matching and EPnP-RANSAC. Used by every experiment."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as Fn

# Localizer settings of the indoor experiments (images at half resolution).
INDOOR = SimpleNamespace(
    img_scale=0.5, depth_tol=0.05, nms_radius_3d=0.02, kp2d_nms=4, kp2d_thr=0.002, kp2d_topk=4096,
    ratio=0.9, mutual=False, pnp_reproj=8.0, pnp_iters=2000, acc_rot=5.0, acc_trans=0.10)

# Localizer and renderer settings of the object experiments (normal-encoded renderings of the CAD model).
OBJECTS = SimpleNamespace(
    image_h=480, fov_deg=60.0, fill=0.88, nz_weight=0.7, crease_deg=30.0, view_chunk=8,
    nms_radius_3d=0.02, harris_knn=32, kp2d_nms=4, kp2d_thr=0.005, kp2d_topk=1024, depth_tol=0.03,
    ratio=0.9, mutual=False, pnp_reproj=4.0, pnp_iters=2000, acc_rot=10.0, acc_trans=0.05,
    num_points_encoder=8192, eval_chunk=100000, seed=0)


# ---------------------------------------------------------------- 2D keypoints and descriptors

def sample_desc(desc: torch.Tensor, uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bilinear sampling of a 1/8-resolution descriptor map at pixel positions, re-normalised."""
    g = uv.clone()
    g[:, 0] = g[:, 0] / (W - 1) * 2 - 1
    g[:, 1] = g[:, 1] / (H - 1) * 2 - 1
    out = Fn.grid_sample(desc[None], g[None, :, None, :], mode="bilinear", align_corners=True)[0, :, :, 0].T
    return out / out.norm(dim=1, keepdim=True).clamp_min(1e-6)


def nms_2d(det: torch.Tensor, radius: int, thr: float, topk: int):
    """Keypoints of a detection map: returns (uv (N,2), score (N,))."""
    mx = Fn.max_pool2d(det[None, None], radius * 2 + 1, 1, radius)[0, 0]
    m = (det >= mx) & (det > thr)
    idx = m.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.zeros((0, 2), device=det.device), torch.zeros((0,), device=det.device)
    sc = det[idx[:, 0], idx[:, 1]]
    if topk and len(sc) > topk:
        sel = torch.topk(sc, topk).indices
        idx, sc = idx[sel], sc[sel]
    uv = torch.stack([idx[:, 1].float(), idx[:, 0].float()], 1)
    return uv, sc


def match_2d_to_3d(d3: np.ndarray, d2: np.ndarray, ratio: float, mutual: bool):
    """Nearest 3D landmark for every query keypoint, with Lowe's ratio test. Returns (i3, i2)."""
    if len(d3) == 0 or len(d2) == 0:
        return np.zeros((0,), int), np.zeros((0,), int)
    sim = d2 @ d3.T
    n2 = min(2, sim.shape[1])
    if n2 == 2:
        part = np.argpartition(-sim, 1, axis=1)[:, :2]
        ps = np.take_along_axis(sim, part, 1)
        order = np.argsort(-ps, axis=1)
        top = np.take_along_axis(part, order, 1)
    else:
        top = np.argmax(sim, axis=1)[:, None]
    best = top[:, 0]
    if n2 == 2:
        s1 = sim[np.arange(len(sim)), top[:, 0]]
        s2 = sim[np.arange(len(sim)), top[:, 1]]
        d1 = np.sqrt(np.clip(2 - 2 * s1, 0, None))
        dd2 = np.sqrt(np.clip(2 - 2 * s2, 0, None))
        keep = d1 < ratio * np.maximum(dd2, 1e-12)
    else:
        keep = np.ones(len(sim), bool)
    if mutual:
        back = np.argmax(sim, axis=0)
        keep &= back[best] == np.arange(len(sim))
    sel = np.nonzero(keep)[0]
    return best[sel], sel


def pose_error(R, t, R_gt, t_gt):
    """Rotation error (degrees) and camera-centre distance."""
    cos = (np.trace(R @ R_gt.T) - 1) / 2
    e_R = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    c = -R.T @ t
    c_gt = -R_gt.T @ t_gt
    return e_R, float(np.linalg.norm(c - c_gt))


def pnp_ransac(X, uv, cm, args):
    import cv2
    return cv2.solvePnPRansac(X.astype(np.float64).reshape(-1, 1, 3), uv.astype(np.float64).reshape(-1, 1, 2),
                              cm, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=args.pnp_reproj,
                              confidence=0.999, iterationsCount=args.pnp_iters)


# ---------------------------------------------------------------- 3D landmarks

def radius_nms_3d(pts: np.ndarray, score: np.ndarray, radius: float, k: int) -> np.ndarray:
    """Greedy radius NMS in order of decreasing score; returns up to k indices."""
    order = np.argsort(-score)
    taken, out = np.zeros(len(pts), dtype=bool), []
    r2 = radius * radius
    for i in order:
        if taken[i]:
            continue
        out.append(i)
        if len(out) >= k:
            break
        d2 = ((pts - pts[i]) ** 2).sum(1)
        taken |= d2 < r2
    return np.array(out, dtype=np.int64)


def nms_gpu(P, s, radius, k):
    """Same greedy radius NMS as `radius_nms_3d`, on the GPU."""
    if len(P) == 0:
        return torch.zeros(0, dtype=torch.long, device=P.device)
    order = torch.argsort(-s)
    taken = torch.zeros(len(P), dtype=torch.bool, device=P.device)
    out, r2 = [], radius * radius
    Po = P[order]
    takeo = taken[order]
    while len(out) < k:
        free = (~takeo).nonzero()
        if len(free) == 0:
            break
        j = int(free[0])
        out.append(int(order[j]))
        d2 = ((Po - Po[j]) ** 2).sum(1)
        takeo |= d2 < r2
    return torch.tensor(out, dtype=torch.long, device=P.device)


def harris3d_scores(pts: np.ndarray, knn: int = 32, k_harris: float = 0.04, max_cand: int = 40000,
                    seed: int = 0) -> np.ndarray:
    """Harris-3D (Sipiran and Bustos, 2011) on a point cloud. Points outside a random subset of
    `max_cand` candidates get -inf."""
    import open3d as o3d
    n = len(pts)
    rng = np.random.RandomState(seed)
    cand = np.arange(n) if n <= max_cand else rng.choice(n, max_cand, replace=False)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    tree = o3d.geometry.KDTreeFlann(pc)
    nb = np.empty((len(cand), knn), dtype=np.int64)
    for i, ci in enumerate(cand):
        _, ids, _ = tree.search_knn_vector_3d(pts[ci], knn)
        ids = list(ids)
        while len(ids) < knn:
            ids.append(int(ci))
        nb[i] = ids[:knn]
    P = pts[nb].astype(np.float64)
    P = P - P.mean(axis=1, keepdims=True)
    cov = np.einsum('mki,mkj->mij', P, P) / knn
    _, evec = np.linalg.eigh(cov)
    R = evec[:, :, ::-1]
    Q = np.einsum('mki,mij->mkj', P, R)
    x, y, z = Q[:, :, 0], Q[:, :, 1], Q[:, :, 2]
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    AtA = np.einsum('mki,mkj->mij', A, A)
    Atz = np.einsum('mki,mk->mi', A, z)
    AtA += np.eye(6)[None] * 1e-12
    coef = np.linalg.solve(AtA, Atz[..., None])[..., 0]
    a1, a2, a3, a4, a5 = coef[:, 1], coef[:, 2], coef[:, 3], coef[:, 4], coef[:, 5]
    fx, fy, fxx, fxy, fyy = a1, a2, 2 * a3, a4, 2 * a5
    EA = fx * fx + 2 * fxx * fxx + 2 * fxy * fxy
    EB = fy * fy + 2 * fyy * fyy + 2 * fxy * fxy
    EC = fx * fy + 2 * fxx * fxy + 2 * fxy * fyy
    resp = (EA * EB - EC * EC) - k_harris * (EA + EB) ** 2
    out = np.full(n, -np.inf, dtype=np.float64)
    out[cand] = np.nan_to_num(resp, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return out


# ---------------------------------------------------------------- indoor: real images

def project(P: np.ndarray, T_c2w: np.ndarray, K: np.ndarray):
    w2c = np.linalg.inv(T_c2w)
    Xc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    ok = z > 1e-6
    u = K[0, 0] * Xc[:, 0] / np.where(ok, z, 1) + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / np.where(ok, z, 1) + K[1, 2]
    return np.stack([u, v], 1), z


def visible_mask(P, T_c2w, K, depth, tol):
    """Visibility of 3D points in a frame from its sensor depth."""
    uv, z = project(P, T_c2w, K)
    H, W = depth.shape
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    inb = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    dz = depth[v, u]
    return inb & (dz > 0) & (np.abs(z - dz) < tol), uv, z


def localize_pair(P_world, idx3d, ref, qry, poses, K, depths, dets, descs, hw, args, dev, return_debug=False):
    """Localize the query frame from landmarks `P_world[idx3d]` seen in the reference frame.
    Success = rotation error < acc_rot and camera-centre error < acc_trans (relative to the reference)."""
    Tr, Tq = poses[ref], poses[qry]
    Hc, Wc = hw
    P3 = P_world[idx3d]
    vis, _, _ = visible_mask(P3, Tr, K, depths[ref], args.depth_tol)
    uvc, _ = project(P3, Tr, K)
    vis &= (uvc[:, 0] >= 0) & (uvc[:, 0] < Wc) & (uvc[:, 1] >= 0) & (uvc[:, 1] < Hc)
    if vis.sum() < 4:
        return None
    P3 = P3[vis]
    uv_v = uvc[vis]
    d3 = sample_desc(descs[ref], torch.from_numpy(uv_v).float().to(dev), Hc, Wc).cpu().numpy()
    uv_q, _ = nms_2d(dets[qry], args.kp2d_nms, args.kp2d_thr, args.kp2d_topk)
    if len(uv_q) < 4:
        return None
    d2 = sample_desc(descs[qry], uv_q, Hc, Wc).cpu().numpy()
    i3, i2 = match_2d_to_3d(d3, d2, args.ratio, args.mutual)
    base = dict(n_kp3d=int(len(P3)), n_match=int(len(i3)), n_inlier=0, e_R=np.nan, e_t=np.nan, ok=False)
    if return_debug:
        base['_dbg'] = dict(P3=P3, kp2d_qry=uv_q.cpu().numpy(), i3=i3, i2=i2, inl=None)
    if len(i3) < 4:
        return base
    import cv2
    cm = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]], np.float64)
    good, rvec, tvec, inl = pnp_ransac(P3[i3], uv_q.cpu().numpy()[i2], cm, args)
    if not good or inl is None or len(inl) < 4:
        return base
    R, _ = cv2.Rodrigues(rvec)
    Rq_gt = Tq[:3, :3].T
    tq_gt = -Tq[:3, :3].T @ Tq[:3, 3]
    eR, et = pose_error(R, tvec.reshape(3), Rq_gt, tq_gt)
    base.update(n_inlier=int(len(inl)), e_R=eR, e_t=et, ok=bool(eR < args.acc_rot and et < args.acc_trans))
    if return_debug:
        base['_dbg'].update(inl=inl.reshape(-1))
    return base


def landmark_outcomes(dbg, XL, Tq, K, args):
    """Per-landmark labels of one localization: correct match (reprojects within the RANSAC
    threshold under the true pose) and RANSAC inlier."""
    corr = np.zeros(len(XL), bool)
    inl_ = np.zeros(len(XL), bool)
    if dbg is None:
        return corr, inl_
    key = {tuple(np.round(p_, 6)): i_ for i_, p_ in enumerate(XL)}
    pos = np.array([key.get(tuple(np.round(p_, 6)), -1) for p_ in dbg['P3']], int)
    i3 = np.asarray(dbg['i3'], int)
    i2 = np.asarray(dbg['i2'], int)
    if len(i3):
        uvg, zg = project(dbg['P3'][i3], Tq, K)
        good_ = (zg > 0) & (np.linalg.norm(uvg - dbg['kp2d_qry'][i2], axis=1) < args.pnp_reproj)
        pi = pos[i3]
        corr[pi[(pi >= 0) & good_]] = True
        if dbg['inl'] is not None:
            pin = pos[i3[np.asarray(dbg['inl'], int)]]
            inl_[pin[pin >= 0]] = True
    return corr, inl_


# ---------------------------------------------------------------- objects: rendered views

def w2v_to_opencv(w2v: torch.Tensor):
    """pytorch3d row-vector world-to-view matrix -> OpenCV (R, t)."""
    M = w2v.detach().cpu().numpy().T
    flip = np.diag([-1.0, -1.0, 1.0])
    R = flip @ M[:3, :3]
    t = flip @ M[:3, 3]
    return R.astype(np.float64), t.astype(np.float64)


def project_opencv(pts: np.ndarray, R: np.ndarray, t: np.ndarray, K: dict):
    Xc = pts @ R.T + t
    z = Xc[:, 2]
    ok = z > 1e-6
    u = K["cx"] + K["fx"] * Xc[:, 0] / np.where(ok, z, 1.0)
    v = K["cy"] + K["fy"] * Xc[:, 1] / np.where(ok, z, 1.0)
    return np.stack([u, v], 1), z


def point_vis(pc, w2v_i, K, depth_i, valid_i, tol, dev):
    """Visibility of points in a rendered view (bilinear depth lookup)."""
    H, W = int(K["H"]), int(K["W"])
    R, t = w2v_to_opencv(w2v_i)
    uv, z = project_opencv(pc, R, t, K)
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    inb = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    g = torch.from_numpy(uv).float().to(dev)
    g[:, 0] = g[:, 0] / (W - 1) * 2 - 1
    g[:, 1] = g[:, 1] / (H - 1) * 2 - 1
    dr = Fn.grid_sample(depth_i[None, None], g[None, :, None, :], mode="bilinear",
                        align_corners=True)[0, 0, :, 0].cpu().numpy()
    vl = valid_i.cpu().numpy()[v, u]
    return inb & vl & (dr > 0) & (np.abs(z - dr) < tol)


@torch.no_grad()
def localize_one_pair(idx3d, pc, ref, qry, det, desc, depth, valid, w2v, K, args, dev):
    """Localize rendered view `qry` from landmarks `pc[idx3d]` described in view `ref`."""
    H, W = int(K["H"]), int(K["W"])
    Rr, tr = w2v_to_opencv(w2v[ref])
    P3 = pc[idx3d]
    vis = point_vis(P3, w2v[ref], K, depth[ref], valid[ref], args.depth_tol, dev)
    if vis.sum() < 4:
        return None
    uv, _ = project_opencv(P3, Rr, tr, K)
    P3 = P3[vis]
    uv_v = uv[vis]
    uv_ref_kp, _ = nms_2d(det[ref], args.kp2d_nms, args.kp2d_thr, args.kp2d_topk)
    if len(uv_ref_kp) < 4:
        return None
    n_proj = int(len(P3))
    d3 = sample_desc(desc[ref], torch.from_numpy(uv_v).float().to(dev), H, W).cpu().numpy()
    uv_q, _ = nms_2d(det[qry], args.kp2d_nms, args.kp2d_thr, args.kp2d_topk)
    if len(uv_q) < 4:
        return None
    d2 = sample_desc(desc[qry], uv_q, H, W).cpu().numpy()
    base = dict(n_kp3d=n_proj, n_match=0, n_inlier=0, e_R=np.nan, e_t=np.nan, ok=False)
    i3, i2 = match_2d_to_3d(d3, d2, args.ratio, args.mutual)
    base["n_match"] = int(len(i3))
    if len(i3) < 4:
        return base
    import cv2
    cm = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]], np.float64)
    ok, rvec, tvec, inl = pnp_ransac(P3[i3], uv_q.cpu().numpy()[i2], cm, args)
    if not ok or inl is None or len(inl) < 4:
        return base
    R, _ = cv2.Rodrigues(rvec)
    Rg, tg = w2v_to_opencv(w2v[qry])
    eR, et = pose_error(R, tvec.reshape(3), Rg, tg)
    return dict(n_kp3d=n_proj, n_match=int(len(i3)), n_inlier=int(len(inl)), e_R=eR, e_t=et,
                ok=bool(eR < args.acc_rot and et < args.acc_trans))
