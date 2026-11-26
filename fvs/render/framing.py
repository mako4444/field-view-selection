"""Camera distance at which an object fills a given fraction of the image."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _project_ndc(verts: torch.Tensor, C: torch.Tensor, fov_deg: float,
                 aspect_wh: float) -> torch.Tensor:
    from pytorch3d.renderer.cameras import look_at_view_transform

    device = verts.device
    C = C.to(device=device, dtype=torch.float32).view(1, 3)
    at = torch.zeros_like(C)
    up0 = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    alt = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    vd = at - C
    vd = vd / vd.norm(dim=1, keepdim=True).clamp_min(1e-6)
    up = alt if float((vd * up0).sum().abs()) > 0.95 else up0

    R, T = look_at_view_transform(eye=C, at=at, up=up, device=device)
    Xv = verts @ R[0] + T[0]
    z = Xv[:, 2].clamp_min(1e-6)

    half_v = math.tan(0.5 * math.radians(float(fov_deg)))
    half_h = half_v * float(aspect_wh)
    ndc_x = (Xv[:, 0] / z) / half_h
    ndc_y = (Xv[:, 1] / z) / half_v
    return torch.stack([ndc_x, ndc_y], dim=-1)


def fit_camera_distance(
    verts: torch.Tensor,
    direction: torch.Tensor,
    fov_deg: float,
    aspect_wh: float,
    fill: float = 0.85,
    d_init: float = 2.0,
    iters: int = 6,
    d_min_margin: float = 1.05,
) -> Tuple[float, float]:
    device = verts.device
    dirn = direction.to(device=device, dtype=torch.float32).view(3)
    dirn = dirn / dirn.norm().clamp_min(1e-9)

    R_sphere = float(verts.norm(dim=1).max())
    d_floor = R_sphere * float(d_min_margin)

    def occupancy(dist: float) -> float:
        return float(_project_ndc(verts, dirn * dist, fov_deg, aspect_wh).abs().max())

    #
    #
    lo = d_floor
    if occupancy(lo) <= float(fill):
        return lo, occupancy(lo)
    hi = max(lo * 2.0, float(d_init))
    for _ in range(40):
        if occupancy(hi) <= float(fill):
            break
        hi *= 2.0
    for _ in range(int(max(iters, 40))):
        mid = 0.5 * (lo + hi)
        if occupancy(mid) > float(fill):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    d = hi
    return d, occupancy(d)
