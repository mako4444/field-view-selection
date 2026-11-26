"""Figure of one test decision of the indoor experiment: the sparse map coloured by the field percentile with
the cameras picked by G and G+F, and the two candidate images with their matches (inliers as green dots).
Field landmarks, h = 5 cm, K = 96. Needs the outputs of scripts/run_indoor.sh.

python scripts/fig_indoor.py --pick 466112/c002:11 --view 1.0,0.5,3.5 --fdepth 0.45 --out outputs/figures/teaser.pdf
"""
import argparse
import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from fvs import paths
from fvs.geomfeat import voxel_subsample
from fvs.localize import INDOOR, localize_pair, radius_nms_3d, visible_mask
from fvs.replay.common import PROTOCOL, make_decisions
from fvs.sf3d import Crop, read_list
from fvs.superpoint import load_superpoint, superpoint_det_desc

KEY = ['crop', 'dec', 'qry', 'mm', 'K']
C_GF, C_G = '#e08000', '#7f7f7f'


def rank01(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


class View:
    """Perspective virtual camera looking at `target` with `up` pointing up."""

    def __init__(self, eye, target, up):
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, u, f])
        self.e = eye

    def __call__(self, X):
        Q = (np.atleast_2d(X) - self.e) @ self.R.T
        return Q[:, 0] / Q[:, 2], Q[:, 1] / Q[:, 2], Q[:, 2]


def frustum(Tc, K0, W0, H0, depth):
    c = Tc[:3, 3]
    px = np.array([[0, 0], [W0, 0], [W0, H0], [0, H0]], float)
    rays = np.c_[(px[:, 0] - K0[0, 2]) / K0[0, 0], (px[:, 1] - K0[1, 2]) / K0[1, 1], np.ones(4)] * depth
    return c, rays @ Tc[:3, :3].T + c


def scene_view(poses, keys, ext):
    Ts = [poses[k] for k in keys]
    Cc = np.stack([t[:3, 3] for t in Ts])
    allR = np.stack([t[:3, 1] for t in poses.values()])
    up = -allR.mean(0)
    up /= np.linalg.norm(up)
    fw = np.mean([t[:3, 2] for t in Ts], 0)
    fh = fw - (fw @ up) * up
    fh = fh / np.linalg.norm(fh) if np.linalg.norm(fh) > 1e-3 else np.cross(up, [1, 0, 0])
    c0 = Cc.mean(0)
    return View(c0 - ext[1] * fh + ext[2] * up, c0 + ext[0] * fh, up)


def draw_map(ax, Sw, col, view, frusta, poses, K0, W0, H0, fdepth, fs, pt):
    import matplotlib.patheffects as pe
    from matplotlib.colors import Normalize
    x, y, z = view(Sw)
    m = z > 0.05
    o = np.argsort(col[m], kind='stable')
    ax.scatter(x[m][o], y[m][o], c=col[m][o], s=pt, cmap='viridis', norm=Normalize(0, 1), lw=0, rasterized=True)
    pts = []
    lw = 1.0
    for k, colr, lab in frusta:
        c, cor = frustum(poses[k], K0, W0, H0, fdepth)
        px, py, _ = view(np.vstack([c, cor]))
        st = dict(color=colr, lw=lw, solid_capstyle='round', zorder=4,
                  path_effects=[pe.Stroke(linewidth=2 * lw, foreground='w'), pe.Normal()])
        for i in range(4):
            ax.plot([px[0], px[1 + i]], [py[0], py[1 + i]], **st)
            ax.plot([px[1 + i], px[1 + (i + 1) % 4]], [py[1 + i], py[1 + (i + 1) % 4]], **st)
        fx, fy = px[1:].mean(), py[1:].mean()
        lx, ly = px[0] - 0.25 * (fx - px[0]), py[0] - 0.25 * (fy - py[0])
        ax.text(lx, ly, lab, color=colr, fontsize=fs, fontweight='bold', ha='center', va='center', zorder=6,
                path_effects=[pe.Stroke(linewidth=2.4, foreground='w'), pe.Normal()])
        pts += [np.c_[px, py], [[lx, ly]]]
    P = np.vstack(pts)
    xs = np.r_[P[:, 0], np.percentile(x[m], [2, 98])]
    ys = np.r_[P[:, 1], np.percentile(y[m], [2, 98])]
    dx, dy = np.ptp(xs), np.ptp(ys)
    ax.set_xlim(xs.min() - 0.04 * dx, xs.max() + 0.04 * dx)
    ax.set_ylim(ys.min() - 0.04 * dy, ys.max() + 0.04 * dy)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def draw_tile(ax, im, L, title, colr, fs):
    import matplotlib.patheffects as pe
    ax.imshow(im)
    uv, inl = L['uv'], L['inl']
    if len(uv):
        halo = [pe.Stroke(linewidth=1.5, foreground='w'), pe.Normal()]
        ax.scatter(uv[~inl, 0], uv[~inl, 1], marker='x', s=4, c=colr, lw=0.6, zorder=3, path_effects=halo)
        ax.scatter(uv[inl, 0], uv[inl, 1], marker='o', s=7, c='#00c000', edgecolors='w', linewidths=0.3, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(colr)
        s_.set_linewidth(1.4)
    ax.set_title(title, fontsize=fs, color=colr, fontweight='bold', pad=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pick', required=True, help='visit/crop:decision')
    ap.add_argument('--run', default=str(paths.OUT / 'indoor/test_sparse'), help='output folder of the test sparse tier')
    ap.add_argument('--crops', default=str(paths.INDOOR / 'test'))
    ap.add_argument('--list', default=str(paths.INDOOR / 'test.txt'))
    ap.add_argument('--frames', default=str(paths.SCENEFUN3D))
    ap.add_argument('--superpoint', default=str(paths.SUPERPOINT))
    ap.add_argument('--n_shards', type=int, default=4)
    ap.add_argument('--K', type=int, default=96)
    ap.add_argument('--mm', type=int, default=50)
    ap.add_argument('--view', default='1.0,1.6,1.3', help='virtual camera: ahead,back,high (m)')
    ap.add_argument('--fdepth', type=float, default=0.25)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    view_ext = [float(x) for x in a.view.split(',')]
    crop_id, dec = a.pick.split(':')
    rows = pd.concat([pd.read_csv(f, dtype={'dec': str, 'qry': str, 'ref_frame': str})
                      for f in sorted(glob.glob(f'{a.run}/replay/rows_s*.csv'))])
    rows = rows[(rows.method == 'field') & (rows.mm == a.mm) & (rows.K == a.K) & (rows.crop == crop_id) & (rows.dec == dec)]
    P = pd.read_pickle(f'{a.run}/planners.pkl')
    for c in ('crop', 'dec', 'qry', 'method'):
        P[c] = P[c].astype(str)
    P = P[(P.method == 'field')]
    g = rows.merge(P[KEY + ['u_G', 'u_G+F']], on=KEY, how='left').fillna({'u_G': 0.0, 'u_G+F': 0.0})
    rG, rF = g.loc[g['u_G'].idxmax()], g.loc[g['u_G+F'].idxmax()]
    visit, cname = crop_id.split('/')
    dev = torch.device('cuda')
    crop = Crop(Path(a.crops) / visit / cname, a.frames, INDOOR.img_scale)
    items = ['/'.join(x) for x in read_list(a.list)]
    ci = items.index(crop_id) // a.n_shards
    good, decisions, _ = make_decisions(crop, SimpleNamespace(**PROTOCOL), ci, dev)
    ia, pick, _ = decisions[int(dec)]
    assert [good[j] for j in pick] == list(g.qry), 'decision does not match the replay'
    idx = voxel_subsample(crop.Pn, crop.T, a.mm / 1000.0)
    z = np.load(crop.dir / f'field_s{a.mm:03d}.npz')
    nS = len(idx)
    Sw = crop.Pw[idx]
    xn, sel = crop.Pn[idx], z['tta'][:nS]
    xw = crop.to_world(xn)
    sp = load_superpoint(a.superpoint, dev)
    depth_c, det_c, desc_c = {}, {}, {}

    def prep(n):
        if n not in det_c:
            t = torch.from_numpy(crop.rgb(n)).float().to(dev) / 255.0
            d_, e_ = superpoint_det_desc(sp, t[None], chunk=1)
            det_c[n], desc_c[n] = d_[0], e_[0]
            depth_c[n] = crop.depth(n)

    def localize(q, ref):
        prep(ref)
        prep(q)
        vm, uv, _ = visible_mask(xw, crop.pose(ref), crop.K, depth_c[ref], INDOOR.depth_tol)
        vm &= (uv[:, 0] >= 0) & (uv[:, 0] < crop.hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < crop.hw[0])
        s_ = np.where(vm)[0]
        Lk = s_[radius_nms_3d(xn[s_], sel[s_], INDOOR.nms_radius_3d, 96)][:a.K]
        r = localize_pair(xw, Lk, ref, q, {ref: crop.pose(ref), q: crop.pose(q)}, crop.K, depth_c, det_c, desc_c,
                          crop.hw, INDOOR, dev, return_debug=True)
        dbg = r['_dbg']
        i2 = np.asarray(dbg['i2'], int)
        inl = np.zeros(len(i2), bool)
        if dbg['inl'] is not None:
            inl[np.asarray(dbg['inl'], int)] = True
        print(f'{q}: ok {r["ok"]}, {r["n_inlier"]} inliers / {r["n_match"]} matches, {r["e_R"]:.1f} deg, '
              f'{100 * r["e_t"]:.1f} cm')
        return dict(uv=dbg['kp2d_qry'][i2] if len(i2) else np.zeros((0, 2)), inl=inl, n_inlier=r['n_inlier'])

    LG, LF = localize(rG.qry, rG.ref_frame), localize(rF.qry, rF.ref_frame)
    assert LG['n_inlier'] == int(rG.n_inlier) and LF['n_inlier'] == int(rF.n_inlier)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'pdf.fonttype': 42,
                                'axes.linewidth': 0.5, 'savefig.dpi': 400})
    poses = {n: crop.pose(n) for n in set(g.qry) | {good[ia]}}
    W0, H0 = crop.hw[1] * 2, crop.hw[0] * 2
    K0 = crop.K.copy()
    K0[:2] /= INDOOR.img_scale
    W, gap, cbh = 7.16, 0.08, 0.30
    ims = [crop.rgb(q) for q in (rG.qry, rF.qry)]
    rr = ims[0].shape[1] / ims[0].shape[0]
    v = scene_view(poses, [rG.qry, rF.qry], view_ext)
    fig = plt.figure(figsize=(W, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_map(ax, Sw, rank01(z['raw'][:nS]), v, [(rG.qry, C_G, 'G'), (rF.qry, C_GF, 'G+F')], poses, K0, W0, H0,
             a.fdepth, 8, 1.2)
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    A = (xb - xa) / (yb - ya)
    th = (W - 2 * gap) / (A * (1 - cbh / 3.0) + 2 * rr)
    hm = th - cbh
    wm = A * hm
    tw = rr * th
    H = th + 0.18
    fig.set_size_inches(W, H)
    ax.set_position([0, cbh / H, wm / W, hm / H])
    bw = 0.5 * wm
    cax = fig.add_axes([(0.5 * wm - 0.5 * bw) / W, 0.09 / H, bw / W, 0.06 / H])
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), 'viridis'), cax=cax, orientation='horizontal')
    cb.set_ticks([])
    cb.outline.set_linewidth(0.3)
    cax.text(-0.02, 0.5, 'low', transform=cax.transAxes, ha='right', va='center', fontsize=6.5)
    cax.text(1.02, 0.5, 'high', transform=cax.transAxes, ha='left', va='center', fontsize=6.5)
    cax.text(0.5, 1.3, 'field percentile', transform=cax.transAxes, ha='center', va='bottom', fontsize=6.5)
    for k, (im, L, colr, nm) in enumerate(((ims[0], LG, C_G, 'G'), (ims[1], LF, C_GF, 'G+F (ours)'))):
        axi = fig.add_axes([(wm + gap + k * (tw + gap)) / W, 0, tw / W, th / H])
        draw_tile(axi, im, L, nm, colr, 8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches='tight', pad_inches=0.01)
    print('->', a.out)


if __name__ == '__main__':
    main()
