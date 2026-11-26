"""Frozen DetField models: loading and inference."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from .model import DetField

__all__ = ["DetField", "load_model", "predict_field"]


def load_model(path, dev) -> Tuple[DetField, List[str]]:
    """Returns the model in eval mode and the names of its per-point input features
    (empty for models without feature inputs)."""
    ck = torch.load(path, map_location="cpu")
    model = DetField(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, list(ck.get("feat_names", []))


@torch.no_grad()
def predict_field(model: DetField, pc: np.ndarray, dev, num_points_encoder: int, chunk: int,
                  gen: torch.Generator, feat: np.ndarray | None = None) -> np.ndarray:
    """Field value (sigmoid) at every point of `pc`. The encoder sees a random subset of
    `num_points_encoder` points drawn with `gen`; `feat` (N,F) are per-point features in the order of `pc`."""
    pc_t = torch.from_numpy(pc).float().to(dev)
    ft_t = None if feat is None else torch.from_numpy(
        np.asarray(feat, dtype=np.float32).reshape(len(pc), -1)).to(dev)
    n = pc_t.shape[0]
    if n > num_points_encoder:
        idx = torch.randperm(n, device=dev, generator=gen)[:num_points_encoder]
        enc_pc = pc_t[idx]
        enc_ft = None if ft_t is None else ft_t[idx]
    else:
        enc_pc, enc_ft = pc_t, ft_t
    enc_in = enc_pc if enc_ft is None else torch.cat([enc_pc, enc_ft], dim=-1)
    grid_feat = model.encoder(enc_in[None, ...])
    out = torch.zeros((n,), device=dev)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        qf = None if (ft_t is None or not model.qfeat_dim) else ft_t[None, s:e, :]
        logit = model.decoder(pc_t[None, s:e, :], grid_feat, qfeat=qf)
        out[s:e] = torch.sigmoid(logit.squeeze(-1).squeeze(0))
    return out.cpu().numpy()


@torch.no_grad()
def query_field(model: DetField, S: np.ndarray, feat: np.ndarray, Q: np.ndarray, dev, chunk: int = 100000):
    """Field value at arbitrary query positions Q. The encoder sees all points S with their
    features; each query takes the features of its nearest point of S."""
    from scipy.spatial import cKDTree
    _, qn = cKDTree(S).query(Q, workers=4)
    St = torch.from_numpy(np.concatenate([S, feat], 1)).float().to(dev)[None]
    g = model.encoder(St)
    ft = torch.from_numpy(feat).float().to(dev)
    out = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        e = min(len(Q), s + chunk)
        q = torch.from_numpy(Q[s:e]).float().to(dev)[None]
        lg = model.decoder(q, g, qfeat=ft[torch.from_numpy(qn[s:e]).to(dev)][None])
        out[s:e] = torch.sigmoid(lg.reshape(-1)).cpu().numpy()
    return out
