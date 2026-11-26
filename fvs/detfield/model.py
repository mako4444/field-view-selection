"""DetField: a point encoder that writes features into voxel grids, and a decoder that
queries the detectability field at arbitrary 3D positions."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d import UNet3D


def normalize_3d_coordinate(p: torch.Tensor, padding: float = 0.1,
                            z_max: float = 0.5, z_min: float = -0.5) -> torch.Tensor:
    """Map coordinates of the padded unit cube to [0, 1)."""
    p_n = p / (1.0 + padding + 1e-8)
    xy = p_n[..., 0:2] + 0.5
    z = p_n[..., 2]
    z = torch.clamp(z, z_min, z_max)
    z = (z - z_min) / (z_max - z_min + 1e-8)
    out = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    return torch.clamp(out, 0.0, 1.0 - 1e-6)


def coordinate2index(x: torch.Tensor, reso: int) -> torch.Tensor:
    ijk = (x * reso).long()
    ijk = torch.clamp(ijk, 0, reso - 1)
    i, j, k = ijk[..., 0], ijk[..., 1], ijk[..., 2]
    return (i + reso * (j + reso * k)).long()


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """src (B,C,N), index (B,N) -> (B,C,dim_size)."""
    B, C, N = src.shape
    out = torch.zeros((B, C, dim_size), device=src.device, dtype=src.dtype)
    cnt = torch.zeros((B, 1, dim_size), device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(1).expand(-1, C, -1)
    out.scatter_add_(dim=2, index=idx, src=src)
    ones = torch.ones((B, 1, N), device=src.device, dtype=src.dtype)
    cnt.scatter_add_(dim=2, index=index.unsqueeze(1), src=ones)
    return out / (cnt + 1e-6)


class ResnetBlockFC(nn.Module):
    def __init__(self, size_in: int, size_out: int | None = None, size_h: int | None = None) -> None:
        super().__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        self.shortcut = None if size_in == size_out else nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        x_s = x if self.shortcut is None else self.shortcut(x)
        return x_s + dx


class GridEncoder(nn.Module):
    """PointNet with local pooling, scattered into one or more voxel grids refined by 3D U-Nets.
    Input (B,N,3+F): coordinates followed by F per-point features."""

    def __init__(self, c_dim=128, dim=3, hidden_dim=256, grid_resolution=32, padding=0.1, n_blocks=5,
                 extra_grids=()):
        super().__init__()
        self.reso_grid = grid_resolution
        self.padding = padding
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.unet3d = UNet3D(in_channels=c_dim, out_channels=c_dim, f_maps=max(16, c_dim // 4), num_levels=3,
                             layer_order="cr")
        self.extra_grids = tuple((int(r), int(c)) for r, c in extra_grids)
        if self.extra_grids:
            self.extra_fc_c = nn.ModuleList([nn.Linear(hidden_dim, c) for _, c in self.extra_grids])
            self.extra_unet3d = nn.ModuleList(
                [UNet3D(in_channels=c, out_channels=c, f_maps=max(8, c // 4), num_levels=3, layer_order="cr")
                 for _, c in self.extra_grids])

    def _pool_local_grid(self, index, net):
        B, N, H = net.shape
        fea = scatter_mean(net.permute(0, 2, 1), index, dim_size=self.reso_grid ** 3)
        gathered = fea.gather(dim=2, index=index.unsqueeze(1).expand(-1, H, -1))
        return gathered.permute(0, 2, 1)

    def _scatter_to_grid(self, p, c, reso, unet):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, reso)
        cc = int(c.shape[-1])
        fea_grid = scatter_mean(c.permute(0, 2, 1), index, dim_size=reso ** 3)
        fea_grid = fea_grid.reshape(p.size(0), cc, reso, reso, reso)
        return unet(fea_grid)

    def forward(self, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        p_xyz = p[..., :3]
        p_nor = normalize_3d_coordinate(p_xyz.clone(), padding=self.padding)
        idx_grid = coordinate2index(p_nor, self.reso_grid)
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self._pool_local_grid(idx_grid, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        out: Dict[str, torch.Tensor] = {"grid": self._scatter_to_grid(p_xyz, c, self.reso_grid, self.unet3d)}
        if self.extra_grids:
            grids = [out["grid"]]
            for k, (reso, _) in enumerate(self.extra_grids):
                grids.append(self._scatter_to_grid(p_xyz, self.extra_fc_c[k](net), reso, self.extra_unet3d[k]))
            out["grids"] = grids
        return out


class FieldDecoder(nn.Module):
    """Samples every grid at the query positions and decodes one logit per query.
    `feat_dim` > 0 appends per-query features to the in-voxel coordinate."""

    def __init__(self, c_dim=128, feat_dim=0, hidden_size=256, n_blocks=5, padding=0.1, extra_grids=()):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.extra_grids = tuple((int(r), int(c)) for r, c in extra_grids)
        total_c = c_dim + sum(c for _, c in self.extra_grids)
        self.fc_c = nn.ModuleList([nn.Linear(total_c, hidden_size) for _ in range(n_blocks)])
        self.feat_dim = int(feat_dim)
        self.fc_p = nn.Linear(3 + self.feat_dim, hidden_size)
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden_size, 1)
        self.padding = padding

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone().contiguous(), padding=self.padding)
        vgrid = 2.0 * p_nor[:, :, None, None].float() - 1.0
        feat = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode="bilinear")
        return feat.squeeze(-1).squeeze(-1)

    def local_coordinate(self, p, reso_grid):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        local = (p_nor * reso_grid) % 1.0
        return local * 2.0 - 1.0

    def forward(self, p: torch.Tensor, c_plane: dict, qfeat: torch.Tensor | None = None) -> torch.Tensor:
        grids = c_plane.get("grids") or [c_plane["grid"]]
        feats = [self.sample_grid_feature(p[:, :, :3], g).transpose(1, 2) for g in grids]
        c = torch.cat(feats, dim=2) if len(feats) > 1 else feats[0]
        p_in = self.local_coordinate(p[:, :, :3], max(g.shape[-1] for g in grids))
        if qfeat is not None:
            p_in = torch.cat([p_in.float(), qfeat.float()], dim=-1)
        net = self.fc_p(p_in.float())
        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        return self.fc_out(F.relu(net))


class DetField(nn.Module):
    def __init__(self, c_dim=128, hidden_dim=256, grid_reso=32, padding=0.1, n_blocks_enc=5, n_blocks_dec=5,
                 extra_grids=(), feat_dim=0, qfeat_dim=0):
        super().__init__()
        self.extra_grids = tuple((int(r), int(c)) for r, c in extra_grids)
        self.feat_dim = int(feat_dim)
        self.qfeat_dim = int(qfeat_dim)
        self.encoder = GridEncoder(c_dim=c_dim, dim=3 + self.feat_dim, hidden_dim=hidden_dim,
                                   grid_resolution=grid_reso, padding=padding, n_blocks=n_blocks_enc,
                                   extra_grids=self.extra_grids)
        self.decoder = FieldDecoder(c_dim=c_dim, feat_dim=self.qfeat_dim, hidden_size=hidden_dim,
                                    n_blocks=n_blocks_dec, padding=padding, extra_grids=self.extra_grids)
