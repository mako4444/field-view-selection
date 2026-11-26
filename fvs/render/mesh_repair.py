"""Mesh normal repair: remove reversed duplicate faces, orient faces consistently within each
connected component, orient components outwards by visibility, and compute per-corner normals with
a 30 degree crease angle. Faces of open shells are flagged so that they can be flipped towards the camera."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def dedupe_reversed_faces(faces: np.ndarray) -> np.ndarray:
    seen: Dict[Tuple[int, int, int], int] = {}
    keep = []
    for i, f in enumerate(faces):
        k = tuple(sorted(int(x) for x in f))
        if k in seen:
            continue
        seen[k] = i
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def orient_faces(faces: np.ndarray, verts: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    F = faces.copy()
    n_faces = F.shape[0]

    edge_map: Dict[Tuple[int, int], list] = {}
    for fi in range(n_faces):
        a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        for (u, v) in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            forward = (u < v)
            edge_map.setdefault(key, []).append((fi, forward))

    visited = np.zeros(n_faces, dtype=bool)
    comp_id = np.full(n_faces, -1, dtype=np.int64)
    n_flipped = 0
    n_comp = 0

    for seed in range(n_faces):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        comp_id[seed] = n_comp
        while stack:
            fi = stack.pop()
            a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
            for (u, v) in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                fwd_me = (u < v)
                for (fj, _fwd_stored) in edge_map.get(key, ()):
                    if fj == fi or visited[fj]:
                        continue
                    a2, b2, c2 = int(F[fj, 0]), int(F[fj, 1]), int(F[fj, 2])
                    fwd_other = None
                    for (u2, v2) in ((a2, b2), (b2, c2), (c2, a2)):
                        k2 = (u2, v2) if u2 < v2 else (v2, u2)
                        if k2 == key:
                            fwd_other = (u2 < v2)
                            break
                    if fwd_other is None:
                        continue
                    if fwd_other == fwd_me:
                        F[fj, [1, 2]] = F[fj, [2, 1]]
                        n_flipped += 1
                    visited[fj] = True
                    comp_id[fj] = n_comp
                    stack.append(fj)
        n_comp += 1

    n_comp_flipped = 0
    for c in range(n_comp):
        sel = np.where(comp_id == c)[0]
        if sel.size == 0:
            continue
        fv = verts[F[sel]]
        vol = float(np.einsum("ij,ij->i",
                              fv[:, 0], np.cross(fv[:, 1], fv[:, 2])).sum()) / 6.0
        if abs(vol) > 1e-9:
            need_flip = vol < 0
        else:
            fn = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
            ctr = fv.reshape(-1, 3).mean(0)
            out = fv.mean(1) - ctr
            need_flip = float((fn * out).sum(1).sum()) < 0
        if need_flip:
            F[sel[:, None], np.array([1, 2])] = F[sel[:, None], np.array([2, 1])]
            n_comp_flipped += 1

    stats = {"n_components": float(n_comp),
             "n_face_flips": float(n_flipped),
             "n_component_flips": float(n_comp_flipped)}
    return F, comp_id, stats


def component_open_mask(faces: np.ndarray, comp_id: np.ndarray,
                        n_comp: int) -> np.ndarray:
    counts: Dict[int, Dict[Tuple[int, int], int]] = {}
    for fi in range(faces.shape[0]):
        c = int(comp_id[fi])
        d = counts.setdefault(c, {})
        a, b, cc = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        for (u, v) in ((a, b), (b, cc), (cc, a)):
            k = (u, v) if u < v else (v, u)
            d[k] = d.get(k, 0) + 1
    out = np.ones(n_comp, dtype=bool)
    for c, d in counts.items():
        out[c] = any(v != 2 for v in d.values())
    return out


@torch.no_grad()
def orient_components_by_visibility(
    verts: torch.Tensor,
    faces: torch.Tensor,
    comp_id: np.ndarray,
    n_comp: int,
    n_views: int = 64,
    image_size: int = 256,
    fov_deg: float = 60.0,
    cam_dist: float = 2.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    from .raster import (make_raster_settings, rasterize,
                                        safe_settings_for)
    from pytorch3d.structures import Meshes
    from .views import Kp3DViewConfig, create_p3d_cameras, _fibonacci_sphere_dirs

    dev = verts.device
    F = faces.shape[0]
    dirs = _fibonacci_sphere_dirs(n_views, device=dev)
    C = dirs * float(cam_dist)
    cfg = Kp3DViewConfig(num_views=n_views, cam_dist=cam_dist,
                         image_w=image_size, image_h=image_size, fov_deg=fov_deg)
    cams, _w2v, _K = create_p3d_cameras(cfg=cfg, device=dev, override_C=C)

    fv = verts[faces]
    fn = torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=1)
    fn = fn / fn.norm(dim=1, keepdim=True).clamp_min(1e-20)
    ctr = fv.mean(dim=1)                                    # (F,3)

    rs = make_raster_settings((image_size, image_size))

    _m1 = Meshes(verts=[verts], faces=[faces])
    _c1 = create_p3d_cameras(cfg=Kp3DViewConfig(num_views=1, cam_dist=cam_dist,
                                                image_w=image_size, image_h=image_size,
                                                fov_deg=fov_deg),
                             device=dev, override_C=dirs[0:1] * cam_dist)[0]
    rs, _bin_info = safe_settings_for(rs, _m1, _c1, int(faces.shape[0]))

    votes = torch.zeros(F, device=dev, dtype=torch.float64)
    chunk = 8
    for s in range(0, n_views, chunk):
        e = min(n_views, s + chunk)
        sub_cfg = Kp3DViewConfig(num_views=e - s, cam_dist=cam_dist, image_w=image_size,
                                 image_h=image_size, fov_deg=fov_deg)
        sub_cams, _w, _k = create_p3d_cameras(cfg=sub_cfg, device=dev, override_C=C[s:e])
        p2f = rasterize(Meshes(verts=[verts], faces=[faces]).extend(e - s),
                        sub_cams, rs).pix_to_face[..., 0]
        for j in range(e - s):
            pf = p2f[j]
            sel = pf[pf >= 0] % F
            if sel.numel() == 0:
                continue
            vcam = C[s + j][None, :] - ctr
            vcam = vcam / vcam.norm(dim=1, keepdim=True).clamp_min(1e-20)
            sgn = torch.sign((fn * vcam).sum(1))            # (F,)
            votes.index_add_(0, sel, sgn[sel].double())

    cid = torch.from_numpy(comp_id).to(dev)
    comp_vote = torch.zeros(n_comp, device=dev, dtype=torch.float64)
    comp_vote.index_add_(0, cid, votes)

    flip = comp_vote < 0
    F_out = faces.clone()
    fmask = flip[cid]
    if bool(fmask.any()):
        idx = torch.where(fmask)[0]
        F_out[idx[:, None], torch.tensor([1, 2], device=dev)] = \
            faces[idx[:, None], torch.tensor([2, 1], device=dev)]

    stats = {
        "vis_n_components_flipped": float(int(flip.sum())),
        "vis_n_faces_flipped": float(int(fmask.sum())),
        "vis_frac_components_ambiguous": float((comp_vote.abs() < 1e-9).float().mean()),
    }
    return F_out, stats


def face_corner_normals(verts: torch.Tensor, faces: torch.Tensor,
                        crease_deg: float = 30.0) -> torch.Tensor:
    V = verts.detach().cpu().numpy().astype(np.float64)
    F = faces.detach().cpu().numpy().astype(np.int64)
    nF = F.shape[0]
    nV = V.shape[0]

    fv = V[F]
    fn_raw = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    area = np.linalg.norm(fn_raw, axis=1)
    fn = fn_raw / np.clip(area, 1e-20, None)[:, None]
    cos_thr = float(np.cos(np.radians(float(crease_deg))))

    vi = F.ravel()
    fi = np.repeat(np.arange(nF, dtype=np.int64), 3)
    order = np.argsort(vi, kind="stable")
    vi_s, fi_s = vi[order], fi[order]
    counts = np.bincount(vi_s, minlength=nV)
    M = int(counts.max()) if nV else 0
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(vi_s.shape[0]) - np.repeat(start, counts)
    idx_pad = np.zeros((nV, M), dtype=np.int64)
    msk_pad = np.zeros((nV, M), dtype=bool)
    idx_pad[vi_s, pos] = fi_s
    msk_pad[vi_s, pos] = True

    out = np.empty((nF * 3, 3), dtype=np.float64)
    step = max(1, int(4_000_000 / max(M, 1)))
    for s in range(0, nF * 3, step):
        e = min(nF * 3, s + step)
        cv, cf = vi[s:e], fi[s:e]
        nbr = idx_pad[cv]                                    # (c,M)
        w = msk_pad[cv] & (np.einsum("cmd,cd->cm", fn[nbr], fn[cf]) >= cos_thr)
        out[s:e] = (fn[nbr] * (w * area[nbr])[..., None]).sum(axis=1)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    bad = nrm[:, 0] < 1e-12
    out[bad] = fn[fi[bad]]
    out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-20, None)
    return torch.from_numpy(out.reshape(nF, 3, 3)).to(device=verts.device, dtype=torch.float32)


def repair_mesh_normals_full(
    verts: torch.Tensor,
    faces: torch.Tensor,
    crease_deg: float = 30.0,
    use_visibility: bool = True,
    n_views: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    dev = verts.device
    V = verts.detach().cpu().numpy().astype(np.float64)
    F0 = faces.detach().cpu().numpy().astype(np.int64)

    keep = dedupe_reversed_faces(F0)
    F1 = F0[keep]
    F2, comp_id, st = orient_faces(F1, V)
    n_comp = int(st["n_components"])

    Ft = torch.from_numpy(F2).to(device=dev, dtype=faces.dtype)
    if use_visibility and n_comp > 0:
        Ft, vst = orient_components_by_visibility(verts, Ft, comp_id, n_comp, n_views=n_views)
        st.update(vst)

    open_comp = component_open_mask(Ft.detach().cpu().numpy().astype(np.int64), comp_id, n_comp)
    face_open = torch.from_numpy(open_comp[comp_id]).to(device=dev)

    cnorm = face_corner_normals(verts, Ft, crease_deg=crease_deg)

    st["n_faces_before"] = float(F0.shape[0])
    st["n_faces_after"] = float(Ft.shape[0])
    st["crease_deg"] = float(crease_deg)
    st["frac_open_components"] = float(open_comp.mean()) if n_comp else 0.0
    st["frac_faces_in_open_components"] = float(face_open.float().mean())
    return Ft, cnorm, face_open, st
