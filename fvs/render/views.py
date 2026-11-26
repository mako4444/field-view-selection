"""Cameras on a sphere around an object (pytorch3d conventions)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import look_at_view_transform


@dataclass(frozen=True)
class Kp3DViewConfig:
    """Dataset-level view rendering config (single source of truth for vis re-render)."""
    num_views: int = 64
    cam_dist: float = 2.0
    image_w: int = 640
    image_h: int = 480
    fov_deg: float = 60.0  # VERTICAL FoV


def normalize_points_to_unit_cube(points: torch.Tensor, padding: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize points to roughly [-0.5, 0.5] with padding.
    Returns:
      points_norm: (N,3)
      transform T: (4,4) such that X_norm = (X - center) / scale
    """
    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    center = (mins + maxs) * 0.5
    scale = (maxs - mins).max()
    scale = scale / (1.0 - float(padding))

    pn = (points - center[None, :]) / scale

    T = torch.eye(4, device=points.device, dtype=torch.float32)
    T[:3, :3] = torch.eye(3, device=points.device, dtype=torch.float32) / scale
    T[:3, 3] = (-center / scale)
    return pn, T


def _fibonacci_sphere_dirs(n: int, device: torch.device) -> torch.Tensor:
    """Uniform-ish directions on sphere by Fibonacci lattice. Returns (n,3) unit."""
    if n <= 1:
        return torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)

    i = torch.arange(n, device=device, dtype=torch.float32)
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    golden_angle = 2.0 * math.pi * (1.0 - 1.0 / phi)

    y = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = golden_angle * i

    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=1)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True).clamp_min(1e-6)
    return dirs


def _sample_views_on_sphere_uniform_p3d(
    num_views: int,
    device: torch.device,
    dist: float,
    override_C: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (R,T,C) from PyTorch3D look_at_view_transform.

    PyTorch3D world->view convention:
      - column-vector: Xv = R * Xw + T
      - row-vector matrix (Transform3d): [x y z 1] @ M_w2v
    """
    if override_C is not None:
        C = override_C.to(device=device, dtype=torch.float32).view(num_views, 3)
    else:
        dirs = _fibonacci_sphere_dirs(num_views, device=device)
        C = dirs * float(dist)

    at = torch.zeros_like(C)
    up0 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)[None, :].repeat(num_views, 1)
    alt_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)[None, :].repeat(num_views, 1)

    view_dir = at - C
    view_dir = view_dir / torch.norm(view_dir, dim=1, keepdim=True).clamp_min(1e-6)
    dot = torch.abs((view_dir * up0).sum(dim=1, keepdim=True))
    up = torch.where(dot > 0.95, alt_up, up0)

    R, T = look_at_view_transform(eye=C, at=at, up=up, device=device)  # (V,3,3), (V,3)
    return R, T, C


def create_p3d_cameras(
    cfg: Kp3DViewConfig,
    device: torch.device,
    override_C: Optional[torch.Tensor] = None,
    projection: str = "persp",
) -> Tuple[Any, torch.Tensor, Dict[str, Any]]:
    proj = str(projection).lower()
    if proj not in ("persp", "ortho"):
        raise ValueError(f'unknown projection {projection!r}')

    V = int(cfg.num_views)
    H = int(cfg.image_h)
    W = int(cfg.image_w)

    R, T, _C = _sample_views_on_sphere_uniform_p3d(
        num_views=V, device=device, dist=float(cfg.cam_dist), override_C=override_C
    )

    cx = 0.5 * (W - 1)
    cy = 0.5 * (H - 1)

    if proj == "persp":
        # Intrinsics in pixels (VERTICAL FoV)
        fy = 0.5 * (H - 1) / math.tan(0.5 * math.radians(float(cfg.fov_deg)))
        fx = fy

        focal = torch.tensor([[fx, fy]], device=device, dtype=torch.float32).repeat(V, 1)
        pp = torch.tensor([[cx, cy]], device=device, dtype=torch.float32).repeat(V, 1)
        imsize = torch.tensor([[H, W]], device=device, dtype=torch.float32).repeat(V, 1)

        cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal,
            principal_point=pp,
            image_size=imsize,
            in_ndc=False,
            device=device,
        )
        K: Dict[str, Any] = {
            "projection": "persp",
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "H": int(H),
            "W": int(W),
            "num_views": int(V),
        }
    else:
        from pytorch3d.renderer import FoVOrthographicCameras

        half_y = float(cfg.cam_dist) * math.tan(0.5 * math.radians(float(cfg.fov_deg)))
        half_x = half_y * (float(W) / float(H))

        cameras = FoVOrthographicCameras(
            R=R,
            T=T,
            znear=0.01,
            zfar=float(cfg.cam_dist) * 4.0,
            max_y=half_y,
            min_y=-half_y,
            max_x=half_x,
            min_x=-half_x,
            device=device,
        )
        K = {
            "projection": "ortho",
            "fx": None,
            "fy": None,
            "cx": float(cx),
            "cy": float(cy),
            "half_x": float(half_x),
            "half_y": float(half_y),
            "H": int(H),
            "W": int(W),
            "num_views": int(V),
        }

    w2v_all = cameras.get_world_to_view_transform().get_matrix()  # (V,4,4) row-vector matrices
    return cameras, w2v_all, K
