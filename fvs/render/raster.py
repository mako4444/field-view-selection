"""Rasterization with binning, checked once per mesh against naive rasterization."""
from __future__ import annotations

from typing import Any


def make_raster_settings(image_size, *, blur_radius: float = 0.0,
                         faces_per_pixel: int = 1,
                         perspective_correct: bool = True,
                         **kw: Any):
    from pytorch3d.renderer import RasterizationSettings
    return RasterizationSettings(image_size=image_size, blur_radius=blur_radius,
                                 faces_per_pixel=faces_per_pixel,
                                 perspective_correct=perspective_correct, **kw)


def _with(settings, **over):
    from pytorch3d.renderer import RasterizationSettings
    f = dict(image_size=settings.image_size, blur_radius=settings.blur_radius,
             faces_per_pixel=settings.faces_per_pixel,
             perspective_correct=settings.perspective_correct,
             clip_barycentric_coords=settings.clip_barycentric_coords,
             cull_backfaces=settings.cull_backfaces,
             z_clip_value=settings.z_clip_value,
             max_faces_per_bin=settings.max_faces_per_bin,
             bin_size=settings.bin_size)
    f.update(over)
    return RasterizationSettings(**f)


def rasterize(meshes, cameras, settings):
    from pytorch3d.renderer import MeshRasterizer
    return MeshRasterizer(cameras=cameras, raster_settings=settings)(meshes)


def safe_settings_for(settings, mesh1, cameras1, n_faces: int):
    import torch

    cand = _with(settings, max_faces_per_bin=max(20000, n_faces // 2))
    naive = _with(settings, bin_size=0)
    ref = rasterize(mesh1, cameras1, naive).pix_to_face
    got = rasterize(mesh1, cameras1, cand).pix_to_face
    bad = int((ref != got).sum())
    if bad <= max(64, ref.numel() // 100000):
        return cand, {"max_faces_per_bin": cand.max_faces_per_bin, "mismatch": bad,
                      "fallback": False}
    safe = _with(settings, max_faces_per_bin=int(n_faces))
    got2 = rasterize(mesh1, cameras1, safe).pix_to_face
    return safe, {"max_faces_per_bin": int(n_faces), "mismatch": int((ref != got2).sum()),
                  "fallback": True, "mismatch_first": bad}
