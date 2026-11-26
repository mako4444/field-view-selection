"""Loading an HM3D glb scene and rendering RGB-D frames with pytorch3d (OpenCV camera convention,
pixel centres at integer coordinates, depth along the optical axis, 0 where nothing is hit)."""
from __future__ import annotations

import numpy as np
import torch
import trimesh


def load_glb(path):
    """All geometry of the glb in scene coordinates: V (n,3), F (m,3), UV (n,2), texture id per face, textures."""
    sc = trimesh.load(str(path), force='scene')
    V, F, UV, gid, tex = [], [], [], [], []
    off = 0
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        v = trimesh.transform_points(g.vertices, T)
        uv = getattr(g.visual, 'uv', None)
        img = None
        if uv is not None:
            mat = g.visual.material
            img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
        if img is None or uv is None:
            uv = np.zeros((len(v), 2))
            arr = np.full((1, 1, 3), 128, np.uint8)
        else:
            arr = np.asarray(img.convert('RGB'))
        V.append(v)
        F.append(np.asarray(g.faces) + off)
        UV.append(uv)
        gid.append(np.full(len(g.faces), len(tex)))
        tex.append(arr)
        off += len(v)
    return (np.concatenate(V).astype(np.float32), np.concatenate(F).astype(np.int64),
            np.concatenate(UV).astype(np.float32), np.concatenate(gid), tex)


class GlbScene:
    def __init__(self, path):
        self.V, self.F, self.UV, self.gid, tex = load_glb(path)
        self.dev = torch.device('cuda:0')
        self.tex = [torch.from_numpy(np.array(t, copy=True)).to(self.dev) for t in tex]
        self.V_t = torch.from_numpy(self.V).to(self.dev)
        self.F_t = torch.from_numpy(self.F).to(self.dev)
        self.UV_t = torch.from_numpy(self.UV).to(self.dev)
        self.gid_t = torch.from_numpy(self.gid).to(self.dev)
        self.max_faces_per_bin = len(self.F)

    @torch.no_grad()
    def render(self, K, c2w, W, H, znear=0.05):
        """Returns depth (H,W) float32 in metres and RGB (H,W,3) uint8."""
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        from pytorch3d.structures import Meshes
        dev = self.dev
        w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(dev)
        Xc = self.V_t @ w2c[:3, :3].T + w2c[:3, 3]
        Z = Xc[:, 2]
        Zs = Z.clamp_min(1e-4)
        u = float(K[0, 0]) * Xc[:, 0] / Zs + float(K[0, 2])
        v = float(K[1, 1]) * Xc[:, 1] / Zs + float(K[1, 2])
        fz, fu, fv = Z[self.F_t], u[self.F_t], v[self.F_t]
        keep = (fz > znear).any(1)
        front = (fz > znear).all(1)
        keep &= ~(front & ((fu < -1).all(1) | (fu > W).all(1) | (fv < -1).all(1) | (fv > H).all(1)))
        fid = torch.nonzero(keep).squeeze(1)
        s = 2.0 / H
        xn = W / H - (u + 0.5) * s
        yn = 1.0 - (v + 0.5) * s
        Vn = torch.stack([xn, yn, Z], 1)
        Vn = torch.where((Z > 1e-4)[:, None], Vn, torch.stack([Xc[:, 0] * 0 + 0, Xc[:, 1] * 0, Z], 1))
        m = Meshes(verts=[Vn], faces=[self.F_t[fid]])
        p2f, zbuf, bary, _ = rasterize_meshes(m, image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
                                              bin_size=None, max_faces_per_bin=min(self.max_faces_per_bin, len(fid)),
                                              perspective_correct=True, clip_barycentric_coords=False,
                                              cull_backfaces=False, z_clip_value=znear)
        p2f = p2f[0, ..., 0].reshape(-1)
        zb = zbuf[0, ..., 0].reshape(-1)
        b = bary[0, :, :, 0].reshape(-1, 3)
        hit = p2f >= 0
        depth = torch.where(hit, zb, torch.zeros_like(zb))
        rgb = torch.zeros((len(p2f), 3), device=dev)
        hi = torch.nonzero(hit).squeeze(1)
        f = fid[p2f[hi]]
        tri = self.F_t[f]
        bb = b[hi]
        uv = bb[:, 0:1] * self.UV_t[tri[:, 0]] + bb[:, 1:2] * self.UV_t[tri[:, 1]] + bb[:, 2:3] * self.UV_t[tri[:, 2]]
        g = self.gid_t[f]
        for k in torch.unique(g).tolist():
            mk = g == k
            T = self.tex[k]
            th_, tw_ = T.shape[:2]
            uu = torch.remainder(uv[mk, 0], 1.0) * (tw_ - 1)
            vv = (1 - torch.remainder(uv[mk, 1], 1.0)) * (th_ - 1)
            x0 = uu.floor().long().clamp(0, tw_ - 1)
            y0 = vv.floor().long().clamp(0, th_ - 1)
            x1 = (x0 + 1).clamp(max=tw_ - 1)
            y1 = (y0 + 1).clamp(max=th_ - 1)
            ax = (uu - x0)[:, None]
            ay = (vv - y0)[:, None]
            rgb[hi[mk]] = (T[y0, x0].float() * (1 - ax) * (1 - ay) + T[y0, x1].float() * ax * (1 - ay)
                           + T[y1, x0].float() * (1 - ax) * ay + T[y1, x1].float() * ax * ay)
        return (depth.reshape(H, W).cpu().numpy().astype(np.float32),
                rgb.reshape(H, W, 3).clamp(0, 255).round().byte().cpu().numpy())
