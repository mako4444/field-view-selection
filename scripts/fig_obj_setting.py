"""Figure of one decision of the object experiment: the sparse map inside the sphere of viewing directions
with the candidates picked by count (A) and field-sum (B), and the reference and candidate images of both.
Needs the replay output of the sparse objects (scripts/run_objects.sh).

python scripts/fig_obj_setting.py --pick 14,7,0.04,10 --K 16 --out outputs/figures/obj_setting.pdf
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from fvs import paths

A_COL, B_COL = '#7f7f7f', '#e08000'


def percentile(v):
    from scipy.stats import rankdata
    return (rankdata(v, method='average') - 1) / max(len(v) - 1, 1)


def render_normal_rgb(Vv, F, cn, fop, C, cams, views, args):
    """Colour normal maps for display (OpenGL convention), with the same normals and face flips as the
    grey encoding that the detector sees."""
    import torch
    from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
    from pytorch3d.structures import Meshes
    from fvs.render.raster import make_raster_settings, rasterize, safe_settings_for
    H = int(args.image_h)
    W = int(round(H * 4 / 3))
    fv = Vv[F]
    fgeo = torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=1)
    fgeo = fgeo / fgeo.norm(dim=1, keepdim=True).clamp_min(1e-20)
    ctr = fv.mean(1)
    rs = make_raster_settings((H, W))
    rs, _ = safe_settings_for(rs, Meshes(verts=[Vv], faces=[F]), cams[0], int(F.shape[0]))
    out = {}
    for i in views:
        fr = rasterize(Meshes(verts=[Vv], faces=[F]), cams[[i]], rs)
        vc = C[i][None, :] - ctr
        vc = vc / vc.norm(dim=1, keepdim=True).clamp_min(1e-20)
        need = ((fgeo * vc).sum(1) < 0) & fop
        att = cn * torch.where(need, -1.0, 1.0)[:, None, None]
        nw = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, att).squeeze(3)
        nw = nw / nw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        nv = torch.einsum("vhwc,vcq->vhwq", nw, cams[[i]].R)[0]
        gl = torch.stack([-nv[..., 0], nv[..., 1], -nv[..., 2]], -1)
        rgb = ((gl + 1) / 2).clamp(0, 1).cpu().numpy()
        vm = (fr.pix_to_face[0, ..., 0] >= 0).cpu().numpy()
        out[i] = np.where(vm[..., None], rgb, 1.0).astype(np.float32)
    return out


def recompute(a, oi, j, pairs):
    import cv2
    import torch
    import torch.nn.functional as Fn
    import trimesh
    from pytorch3d.io import load_objs_as_meshes
    from fvs.detfield import load_model, predict_field
    from fvs.geomfeat import sparse_object
    from fvs.localize import (OBJECTS, match_2d_to_3d, nms_2d, project_opencv, radius_nms_3d, sample_desc,
                              w2v_to_opencv)
    from fvs.render import render_views, repair_mesh_normals_full, ring_dirs
    from fvs.superpoint import load_superpoint, superpoint_det_desc
    args = OBJECTS
    objs = [l.split() for l in open(a.list) if l.strip()]
    syn, mid = objs[oi]
    dev = torch.device('cuda')
    model, _ = load_model(a.weights, dev)
    sp = load_superpoint(a.superpoint, dev)
    obj = Path(a.shapenet) / syn / mid / 'models' / 'model_normalized.obj'
    mesh = load_objs_as_meshes([str(obj)], device=dev, load_textures=False)
    V0 = mesh.verts_packed()
    tv = np.asarray(trimesh.load(str(obj), force='mesh', process=False).vertices)
    lo, hi = tv.min(0), tv.max(0)
    ctr = torch.from_numpy((lo + hi) / 2).float().to(dev)
    mesh = mesh.update_padded(((V0 - ctr) / float((hi - lo).max() / 0.9))[None])
    Vv, Fr = mesh.verts_packed(), mesh.faces_packed()
    F, cn, fop, _ = repair_mesh_normals_full(Vv, Fr, crease_deg=args.crease_deg)
    dirs, items = ring_dirs(12, 8, [10.0, 15.0], args.seed + oi, dev)
    img, depth, valid, cams, w2v, K, C = render_views(Vv, F, cn, fop, dirs, args, dev)
    det, desc = superpoint_det_desc(sp, img, chunk=4)
    X = np.load(Path(a.objects) / syn / mid / 'dense.npy').astype(np.float64)
    S, feat = sparse_object(X, a.s, np.random.default_rng(0))
    gg = torch.Generator(device=dev)
    gg.manual_seed(0)
    c1s = predict_field(model, S.astype(np.float32), dev, 8192, 100000, gg, feat=feat).astype(np.float64)
    P = S[radius_nms_3d(S, c1s.astype(np.float32), 0.05, 64)][:a.K]
    dec = [(ref, c) for (ref, c, th, jj) in items if np.isclose(th, a.theta) and jj == j]
    H, W = int(K['H']), int(K['W'])

    def visible(view):
        R_, t_ = w2v_to_opencv(w2v[view])
        uv, z = project_opencv(P, R_, t_, K)
        g = torch.from_numpy(uv).float().to(dev).clone()
        g[:, 0] = g[:, 0] / (W - 1) * 2 - 1
        g[:, 1] = g[:, 1] / (H - 1) * 2 - 1
        dr = Fn.grid_sample(depth[view][None, None], g[None, :, None, :], mode='bilinear',
                            align_corners=True)[0, 0, :, 0].cpu().numpy()
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
        inb = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv, inb & valid[view].cpu().numpy()[v, u] & (dr > 0) & (np.abs(z - dr) < args.depth_tol)

    cm = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]], np.float64)
    out = dict(S=S, c1s=c1s, dirs=dirs.cpu().numpy(), P=P, ref=np.array([r for r, _ in dec]),
               qry=np.array([c for _, c in dec]))
    views = sorted({v for pr in pairs for v in pr})
    rgb = render_normal_rgb(Vv, F, cn, fop, C, cams, views, args)
    for pi, (ref, qry) in enumerate(pairs):
        uvr, vr = visible(ref)
        idx = np.flatnonzero(vr)
        uv_v = uvr[idx]
        ur, _ = nms_2d(det[ref], args.kp2d_nms, args.kp2d_thr, args.kp2d_topk)
        d3 = sample_desc(desc[ref], torch.from_numpy(uv_v).float().to(dev), H, W).cpu().numpy()
        uq, _ = nms_2d(det[qry], args.kp2d_nms, args.kp2d_thr, args.kp2d_topk)
        d2 = sample_desc(desc[qry], uq, H, W).cpu().numpy()
        uq = uq.cpu().numpy()
        i3, i2 = match_2d_to_3d(d3, d2, args.ratio, args.mutual)
        inl = np.zeros(len(i3), bool)
        if len(i3) >= 4:
            ok, rv, tv_, il = cv2.solvePnPRansac(P[idx][i3].reshape(-1, 1, 3), uq[i2].astype(np.float64).reshape(-1, 1, 2),
                                                 cm, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=args.pnp_reproj,
                                                 confidence=0.999, iterationsCount=args.pnp_iters)
            if ok and il is not None and len(il) >= 4:
                inl[il.reshape(-1)] = True
        uvq_all, vq = visible(qry)
        out.update({f'p{pi}_refrgb': rgb[ref], f'p{pi}_qryrgb': rgb[qry], f'p{pi}_uvref': uv_v,
                    f'p{pi}_uvq': uq[i2], f'p{pi}_inl': inl, f'p{pi}_i3': np.asarray(i3, int),
                    f'p{pi}_detref': ur.cpu().numpy(), f'p{pi}_detqry': uq, f'p{pi}_uvqry': uvq_all[vq],
                    f'p{pi}_visref_idx': idx, f'p{pi}_visqry_idx': np.flatnonzero(vq)})
    return out


def draw_sides(ax, Z, ks, cols, labs, theta, Rs=0.8, el_deg=20.0, ps=1.2, fr_len=0.11):
    """Sphere of viewing directions with the sparse map (field percentile), the landmarks in red, the two
    picked candidates (solid frustums) with their references (dashed) and the theta circles of all
    twelve references."""
    from matplotlib import cm, colors
    from scipy.spatial import cKDTree
    dirs = Z['dirs']
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    cand = dirs[Z['qry']]
    refs = dirs[Z['ref']]
    m = cand[ks[0]] + cand[ks[1]]
    m /= np.linalg.norm(m)
    az = np.arctan2(m[0], m[2])
    key = np.concatenate([refs[list(ks)], cand[list(ks)]])
    best, best_sc, best_v = None, -9, None
    for e_ in np.radians(np.arange(el_deg, -61, -5)):
        v_ = np.array([np.cos(e_) * np.sin(az), np.sin(e_), np.cos(e_) * np.cos(az)])
        sc = float((key @ v_).min())
        if sc >= 0.25:
            best = v_
            break
        if sc > best_sc:
            best_v, best_sc = v_, sc
    v = best if best is not None else best_v
    up = np.array([0, 1.0, 0])
    ex = np.cross(up, v)
    ex /= np.linalg.norm(ex)
    ey = np.cross(v, ex)
    proj = lambda X: np.stack([X @ ex, X @ ey], -1)

    def curve(D, col, lw, zf=1, zb=0, ab=0.45):
        P2 = proj(Rs * D)
        front = D @ v >= 0
        for msk, ls, al, z in ((front, '-', 1.0, zf), (~front, (0, (1.5, 1.5)), ab, zb)):
            seg = np.where(msk, 0, np.nan)
            ax.plot(P2[:, 0] + seg, P2[:, 1] + seg, color=col, lw=lw, ls=ls, alpha=al, zorder=z)

    t = np.linspace(0, 2 * np.pi, 361)
    g = np.linspace(-1, 1, 301)
    GX, GY = np.meshgrid(g, g)
    rr = np.sqrt(GX ** 2 + GY ** 2)
    shade = np.where(rr <= 1, 1 - 0.06 * rr ** 3, np.nan)
    ax.imshow(shade, extent=(-Rs, Rs, -Rs, Rs), cmap="gray", vmin=0.55, vmax=1, origin="lower", zorder=0,
              interpolation='bilinear')
    ax.plot(Rs * np.cos(t), Rs * np.sin(t), color='0.55', lw=0.6, zorder=1)
    S = Z['S']
    pct = percentile(Z['c1s'])
    rgba = cm.viridis(colors.Normalize(0, 1)(pct ** 6))
    isk = np.zeros(len(S), bool)
    isk[cKDTree(S).query(Z['P'])[1]] = True
    rgba[isk] = colors.to_rgba('#e3001b')
    o = np.argsort(S @ v, kind='stable')
    pS = proj(S)
    ax.scatter(pS[o, 0], pS[o, 1], c=rgba[o], s=ps, lw=0, zorder=3, rasterized=True)
    th = np.radians(theta)

    def frustum(d, col, alpha, ls='-'):
        cpos = Rs * d
        fwd = -d
        u1 = np.cross(fwd, [0, 1.0, 0])
        u1 = u1 / (np.linalg.norm(u1) + 1e-9)
        u2 = np.cross(u1, fwd)
        h = fr_len * 0.45
        base = cpos + fwd * fr_len
        cor = [base + sx * h * u1 + sy * h * 0.75 * u2 for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        pc_, pcs = proj(cpos), [proj(c_) for c_ in cor]
        for q in pcs:
            ax.plot([pc_[0], q[0]], [pc_[1], q[1]], color=col, lw=0.6, ls=ls, alpha=alpha, zorder=6)
        loop = pcs + [pcs[0]]
        ax.plot([q[0] for q in loop], [q[1] for q in loop], color=col, lw=0.6, ls=ls, alpha=alpha, zorder=6)
        return pc_

    for k in range(len(cand)):
        if k in ks:
            continue
        r, c = refs[k], cand[k]
        u1 = np.cross(r, [0, 1.0, 0])
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(r, u1)
        circ = np.cos(th) * r[None] + np.sin(th) * (np.cos(t)[:, None] * u1 + np.sin(t)[:, None] * u2)
        P2 = proj(Rs * circ)
        front = circ @ v >= 0
        for msk, al in ((front, 0.9), (~front, 0.35)):
            seg = np.where(msk, 0, np.nan)
            ax.plot(P2[:, 0] + seg, P2[:, 1] + seg, color='0.35', lw=0.5, ls=(0, (1.6, 1.2)), alpha=al,
                    zorder=2 if al < 0.5 else 5)
        pcand = proj(Rs * c)
        ax.scatter([pcand[0]], [pcand[1]], s=3, color='0.35', alpha=0.9 if c @ v >= 0 else 0.35, lw=0, zorder=6)
    for k, col, lab in zip(ks, cols, labs):
        r, c = refs[k], cand[k]
        u1 = np.cross(r, [0, 1.0, 0])
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(r, u1)
        circ = np.cos(th) * r[None] + np.sin(th) * (np.cos(t)[:, None] * u1 + np.sin(t)[:, None] * u2)
        curve(circ, col, 0.8, zf=5, zb=4, ab=0.6)
        prf = frustum(r, col, 1.0 if r @ v >= 0 else 0.5, ls=(0, (1.6, 1.2)))
        pc = frustum(c, col, 1.0 if c @ v >= 0 else 0.5)
        ax.scatter([prf[0], pc[0]], [prf[1], pc[1]], s=3, color=col, lw=0, zorder=7)
        pr = proj(Rs * r)
        q = pr + (pr / (np.linalg.norm(pr) + 1e-9)) * 0.2
        ax.text(q[0], q[1], lab, color=col, fontsize=8.5, ha='center', va='center', fontweight='bold', zorder=7)
    lim = Rs * 1.12
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def crop_box(imgs, pad=12):
    m = np.zeros(imgs[0].shape[:2], bool)
    for im in imgs:
        m |= (im < 0.999).any(-1)
    ys, xs = np.nonzero(m)
    return max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad), max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad)


def draw_view(ax, im, uv3, det, col, hi_pt, box, hi_det=None):
    """Crosses: SuperPoint detections; red dots: visible landmarks; green dots: landmarks that became PnP
    inliers; bright green crosses: the detections matched to them."""
    x0, x1, y0, y1 = box
    ax.imshow(im, interpolation='lanczos')
    hit = np.zeros(len(det), bool)
    if hi_det is not None and len(hi_det) and len(det):
        hit[np.sqrt(((hi_det[:, None] - det[None]) ** 2).sum(-1)).argmin(1)] = True
    ax.scatter(uv3[~hi_pt, 0], uv3[~hi_pt, 1], s=3, marker='o', facecolor='#e3001b', edgecolor='none', lw=0, zorder=3)
    ax.scatter(uv3[hi_pt, 0], uv3[hi_pt, 1], s=9, marker='o', facecolor='#00c000', edgecolor='none', lw=0, zorder=3.5)
    ax.scatter(det[~hit, 0], det[~hit, 1], s=4, marker='x', color=col, lw=0.3, zorder=4)
    ax.scatter(det[hit, 0], det[hit, 1], s=7, marker='x', color='#00e000', lw=0.6, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_edgecolor(col)
        s_.set_linewidth(0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pick', default='14,7,0.04,10', help='object index in the list, j, h, theta')
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--rows', default=str(paths.OUT / 'objects_sparse'))
    ap.add_argument('--objects', default=str(paths.OBJECTS / 'sparse'))
    ap.add_argument('--list', default=str(paths.OBJECTS / 'sparse_test.txt'))
    ap.add_argument('--shapenet', default=str(paths.SHAPENET))
    ap.add_argument('--weights', default=str(paths.DETFIELD_OBJECTS))
    ap.add_argument('--superpoint', default=str(paths.SUPERPOINT))
    ap.add_argument('--out', default=str(paths.OUT / 'figures/obj_setting.pdf'))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    oi, j, s_, th_ = a.pick.split(',')
    oi, j, a.s, a.theta = int(oi), int(j), float(s_), float(th_)
    objs = [l.split() for l in open(a.list) if l.strip()]
    syn, mid = objs[oi]
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{a.rows}/rows_s*.csv'))], ignore_index=True)
    g = R[(R.model == f'{syn}/{mid}') & np.isclose(R.s, a.s) & np.isclose(R.theta, a.theta) & (R.method == 'field')
          & (R.K == a.K) & (R.j == j)].sort_values('ref')
    fi, ci = int(np.argmax(g.field_sum.values)), int(np.argmax(g.cnt.values))
    pairs = [(int(g.ref.values[ci]), int(g.qry.values[ci])), (int(g.ref.values[fi]), int(g.qry.values[fi]))]
    Z = recompute(a, oi, j, pairs)
    for pi, (ref, qry) in enumerate(pairs):
        row = g[g.ref == ref].iloc[0]
        print(f'{"AB"[pi]}: view {ref} -> {qry}: {int(Z[f"p{pi}_inl"].sum())} inliers (replay {int(row.n_inlier)})')
    k_c = int(np.flatnonzero(Z['qry'] == pairs[0][1])[0])
    k_f = int(np.flatnonzero(Z['qry'] == pairs[1][1])[0])
    fig = plt.figure(figsize=(7.16, 2.05))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1.75, 1, 1, 1, 1], wspace=0.05, left=0.0, right=0.998, top=0.9,
                  bottom=0.01)
    draw_sides(fig.add_subplot(gs[0, 0]), Z, [k_c, k_f], [A_COL, B_COL], ['A', 'B'], a.theta,
               ps=2.0 if a.s < 0.03 else 4.0)
    x0, x1, y0, y1 = crop_box([Z[f'p{q}_{k}'] for q in (0, 1) for k in ('refrgb', 'qryrgb')])
    H_, W_ = Z['p0_refrgb'].shape[:2]
    L = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cx = min(max(cx, L / 2), W_ - L / 2)
    cy = min(max(cy, L / 2), H_ - L / 2)
    box = (cx - L / 2, cx + L / 2, cy - L / 2, cy + L / 2)
    for pi, col, lab, c0 in ((0, A_COL, 'A', 1), (1, B_COL, 'B', 3)):
        inl = Z[f'p{pi}_inl'].astype(bool)
        gl = Z[f'p{pi}_visref_idx'][Z[f'p{pi}_i3'][inl]]
        ax1 = fig.add_subplot(gs[0, c0])
        ax2 = fig.add_subplot(gs[0, c0 + 1])
        draw_view(ax1, Z[f'p{pi}_refrgb'], Z[f'p{pi}_uvref'], Z[f'p{pi}_detref'], col,
                  np.isin(Z[f'p{pi}_visref_idx'], gl), box)
        draw_view(ax2, Z[f'p{pi}_qryrgb'], Z[f'p{pi}_uvqry'], Z[f'p{pi}_detqry'], col,
                  np.isin(Z[f'p{pi}_visqry_idx'], gl), box, hi_det=Z[f'p{pi}_uvq'][inl])
        ax1.set_title(f'{lab}: reference view', fontsize=7.5, pad=2, color=col, fontweight='bold')
        ax2.set_title(f'{lab}: candidate view', fontsize=7.5, pad=2, color=col, fontweight='bold')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print('->', a.out)


if __name__ == '__main__':
    main()
