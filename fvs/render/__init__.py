"""Rendering of CAD models into normal-encoded images (the images that the object experiments
localize), and the candidate viewing directions."""
from __future__ import annotations

import numpy as np
import torch

from .framing import fit_camera_distance
from .mesh_repair import repair_mesh_normals_full
from .raster import make_raster_settings, rasterize, safe_settings_for
from .views import Kp3DViewConfig, _fibonacci_sphere_dirs, create_p3d_cameras, normalize_points_to_unit_cube

__all__ = ['render_views', 'ring_dirs', 'repair_mesh_normals_full', 'normalize_points_to_unit_cube']


@torch.no_grad()
def render_views(V, F, cnorm, face_open, dirs, args, dev):
    """Render the mesh from the given directions; the camera distance makes the object fill
    `args.fill` of the image. Image = 0.7 |n_z| + 0.3 normalised depth (camera-space normals).
    Returns (img, depth, valid, cams, w2v, K, C)."""
    from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
    from pytorch3d.structures import Meshes
    H = int(args.image_h)
    W = int(round(H * 4 / 3))
    dirs = dirs.to(dev).float()
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-20)
    n_views = int(dirs.shape[0])
    ds = [fit_camera_distance(V, dirs[i], args.fov_deg, W / H, fill=args.fill, d_init=2.0)[0] for i in range(n_views)]
    C = dirs * torch.tensor(ds, device=dev, dtype=torch.float32)[:, None]
    cfg = Kp3DViewConfig(num_views=n_views, cam_dist=2.0, image_w=W, image_h=H, fov_deg=args.fov_deg)
    cams, w2v, K = create_p3d_cameras(cfg=cfg, device=dev, override_C=C)
    fv = V[F]
    fgeo = torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=1)
    fgeo = fgeo / fgeo.norm(dim=1, keepdim=True).clamp_min(1e-20)
    ctr = fv.mean(1)
    Vh = torch.cat([V, torch.ones((len(V), 1), device=dev)], 1)
    radius = float((V.max(0).values - V.min(0).values).norm() * 0.5)
    rs = make_raster_settings((H, W))
    rs, _ = safe_settings_for(rs, Meshes(verts=[V], faces=[F]), cams[0], int(F.shape[0]))
    img = torch.zeros((n_views, H, W, 3), device=dev)
    depth = torch.zeros((n_views, H, W), device=dev)
    valid = torch.zeros((n_views, H, W), dtype=torch.bool, device=dev)
    for s in range(0, n_views, args.view_chunk):
        e = min(n_views, s + args.view_chunk)
        sub = cams[list(range(s, e))]
        fr = rasterize(Meshes(verts=[V], faces=[F]).extend(e - s), sub, rs)
        p2f, bary = fr.pix_to_face, fr.bary_coords
        vm = p2f[..., 0] >= 0
        attrs, dattrs = [], []
        for i in range(s, e):
            vc = C[i][None, :] - ctr
            vc = vc / vc.norm(dim=1, keepdim=True).clamp_min(1e-20)
            need = ((fgeo * vc).sum(1) < 0) & face_open
            attrs.append(cnorm * torch.where(need, -1.0, 1.0)[:, None, None])
            dattrs.append((Vh @ w2v[i])[:, 2:3][F])
        nw = interpolate_face_attributes(p2f, bary, torch.cat(attrs, 0).contiguous()).squeeze(3)
        nw = nw / nw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        nv = torch.einsum("vhwc,vcq->vhwq", nw, sub.R)
        dz = interpolate_face_attributes(p2f, bary, torch.cat(dattrs, 0).contiguous()).squeeze(3)[..., 0]
        dz = torch.where(vm, dz, torch.zeros_like(dz))
        d01 = ((dz - (C[s:e].norm(dim=1)[:, None, None] - radius)) / (2 * radius)).clamp(0, 1)
        nz = (1 - nv[..., 0] ** 2 - nv[..., 1] ** 2).clamp_min(0).sqrt()
        enc = torch.where(vm, args.nz_weight * nz + (1 - args.nz_weight) * d01, torch.zeros_like(d01))
        img[s:e] = enc[..., None].expand(-1, -1, -1, 3)
        depth[s:e], valid[s:e] = dz, vm
        del fr, p2f, bary, nw, nv
    return img, depth, valid, cams, w2v, K, C


def ring_dirs(n_base, M, theta_list, seed, dev):
    """Reference directions (Fibonacci sphere) and, around each, M candidate directions at exactly
    angle theta (random initial phase). Returns the directions and [(ref, cand, theta, j)]."""
    base = _fibonacci_sphere_dirs(n_base, device=dev)
    rng = np.random.RandomState(seed)
    e1 = torch.from_numpy(rng.randn(n_base, 3)).float().to(dev)
    e1 = e1 - (e1 * base).sum(1, keepdim=True) * base
    e1 = e1 / e1.norm(dim=1, keepdim=True)
    e2 = torch.cross(base, e1, dim=1)
    dirs, items = [base], []
    off = n_base
    for th in theta_list:
        t = np.radians(th)
        ph0 = rng.rand(n_base) * 2 * np.pi
        for j in range(M):
            ph = torch.from_numpy(ph0 + 2 * np.pi * j / M).float().to(dev)[:, None]
            e = torch.cos(ph) * e1 + torch.sin(ph) * e2
            dirs.append(float(np.cos(t)) * base + float(np.sin(t)) * e)
            items += [(i, off + i, float(th), j) for i in range(n_base)]
            off += n_base
    return torch.cat(dirs, 0), items
