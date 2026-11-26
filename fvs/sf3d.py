"""Reading SceneFun3D captures (and HM3D renderings stored in the same layout) and our crop samples.

A crop sample directory holds
  pc.npy         points of the crop in normalised coordinates (pc = s * world + t)
  pc_normal.npy  point normals in normalised coordinates
  transform.txt  4x4 matrix of the normalisation
  meta.json      {"visit": ..., "video": ...}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def aa2R(a):
    th = float(np.linalg.norm(a))
    if th < 1e-12:
        return np.eye(3)
    k = a / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def read_poses(d):
    """Camera-to-world poses keyed by timestamp (hires_poses.traj stores world-to-camera)."""
    poses = {}
    for l in open(Path(d) / 'hires_poses.traj'):
        t = l.split()
        E = np.eye(4)
        E[:3, :3] = aa2R(np.array(list(map(float, t[1:4]))))
        E[:3, 3] = list(map(float, t[4:7]))
        poses[round(float(t[0]), 3)] = np.linalg.inv(E)
    return poses


def intr(d, name):
    W, H, fx, fy, cx, cy = map(float, (Path(d) / 'hires_wide_intrinsics' / f'{name}.pincam').read_text().split())
    return int(W), int(H), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def ts(name):
    return round(float(name.split('_')[1]), 3)


def read_list(path):
    return [tuple(l.split()) for l in Path(path).read_text().strip().split('\n') if l.strip()]


class Crop:
    """A crop sample together with its capture: points in world coordinates, frames, poses, intrinsics."""

    def __init__(self, crop_dir, frames_root, img_scale=0.5, require_rgb=True):
        sd = Path(crop_dir)
        self.dir = sd
        meta = json.load(open(sd / 'meta.json'))
        self.video_dir = Path(frames_root) / meta['visit'] / meta['video']
        d = self.video_dir
        self.poses = read_poses(d)
        self.T = np.loadtxt(sd / 'transform.txt')
        self.Pn = np.load(sd / 'pc.npy').astype(np.float64)
        self.Pw = self.to_world(self.Pn)
        names = sorted(f.stem for f in (d / 'hires_depth').glob('*.png'))
        names = [n for n in names if ts(n) in self.poses and (not require_rgb or (d / 'hires_wide' / f'{n}.jpg').exists())]
        self.names = names
        W0, H0, K0 = intr(d, names[0])
        self.K = K0.copy()
        self.K[:2] *= img_scale
        self.hw = (int(round(H0 * img_scale)), int(round(W0 * img_scale)))

    def to_world(self, X):
        T = self.T
        return (X - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T

    def pose(self, name):
        return self.poses[ts(name)]

    def depth(self, name):
        import cv2
        dd = cv2.imread(str(self.video_dir / 'hires_depth' / f'{name}.png'), -1).astype(np.float32) / 1000.0
        return cv2.resize(dd, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_NEAREST)

    def rgb(self, name):
        import cv2
        im = cv2.resize(cv2.imread(str(self.video_dir / 'hires_wide' / f'{name}.jpg')), (self.hw[1], self.hw[0]),
                        interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
