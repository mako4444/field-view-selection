"""Render the room videos of an HM3D scene along the released trajectories, in the SceneFun3D capture
layout (hires_wide/*.jpg, hires_depth/*.png in mm, hires_wide_intrinsics/*.pincam, hires_poses.traj
storing world-to-camera axis-angle and translation). 1920 x 1440, fx = fy = 1600.

python -m fvs.hm3d.render_capture 00017-oEPjPNSPmzL [--trajectories data/hm3d/trajectories] [--capture outputs/hm3d/capture]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .. import paths
from .glb import GlbScene

W, H, FX = 1920, 1440, 1600.0
K = np.array([[FX, 0, (W - 1) / 2], [0, FX, (H - 1) / 2], [0, 0, 1.0]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scene')
    ap.add_argument('--trajectories', default=str(paths.HM3D / 'trajectories'))
    ap.add_argument('--capture', default=str(paths.OUT / 'hm3d/capture'))
    ap.add_argument('--scenes_root', default=str(paths.HM3D_ACTLOC))
    a = ap.parse_args()
    visit = a.scene.split('-')[0]
    tr = Path(a.trajectories) / visit
    rooms = json.load(open(tr / 'rooms.json'))['kept']
    G = GlbScene(next((Path(a.scenes_root) / a.scene).glob('*.glb')))
    for room in rooms:
        video = room['video']
        d = Path(a.capture) / visit / video
        z = np.load(tr / video / 'traj.npz')
        t, T = z['t'], z['c2w']
        for sub in ('hires_wide', 'hires_depth', 'hires_wide_intrinsics'):
            (d / sub).mkdir(parents=True, exist_ok=True)
        lines = []
        for ti, Ti in zip(t, T):
            w2c = np.linalg.inv(Ti)
            lines.append(f'{ti:.3f} ' + ' '.join(f'{x:.8f}' for x in [*Rotation.from_matrix(w2c[:3, :3]).as_rotvec(),
                                                                         *w2c[:3, 3]]))
        (d / 'hires_poses.traj').write_text('\n'.join(lines) + '\n')
        t0, done = time.time(), 0
        for ti, Ti in zip(t, T):
            name = f'{video}_{ti:011.3f}'
            jp, dp = d / 'hires_wide' / f'{name}.jpg', d / 'hires_depth' / f'{name}.png'
            if jp.exists() and dp.exists():
                continue
            dep, rgb = G.render(K, Ti, W, H)
            cv2.imwrite(str(dp), np.clip(np.round(dep * 1000), 0, 65535).astype(np.uint16))
            (d / 'hires_wide_intrinsics' / f'{name}.pincam').write_text(f'{W} {H} {FX} {FX} {K[0, 2]} {K[1, 2]}')
            cv2.imwrite(str(jp), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            done += 1
        print(f'{video}: {done} frames rendered of {len(t)}, {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
