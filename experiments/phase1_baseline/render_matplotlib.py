"""
Publication-quality point cloud renders using manual perspective projection.
Produces crisp, paper-ready images without requiring a display or GPU.

Strategy:
  - Downsample to ~300k pts (stratified per class so beam stays visible)
  - Sort by depth (painter's algorithm) for correct occlusion
  - Perspective projection with custom camera
  - 4 outputs: RGB, GT, Pred, triptych
  - Beam-zoom pair for the abstract diagram
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from PIL import Image, ImageDraw, ImageFont
import os

PLY_DIR = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/pointclouds/"
DATA    = "/home/ahmad/frozen_teacher_project/data/processed/s3dis/Area_5/office_39/"
RESULT  = "/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/result/"
OUT     = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/charts/"

CLASS_NAMES = [
    "ceiling","floor","wall","beam","column",
    "window","door","table","chair","sofa","bookcase","board","clutter",
]
# Perceptually-tuned class palette (float RGB)
CLASS_COLORS = np.array([
    [0.72, 0.72, 0.76],   # ceiling  – cool gray
    [0.62, 0.42, 0.20],   # floor    – warm brown
    [0.76, 0.83, 0.95],   # wall     – pale blue
    [1.00, 0.45, 0.00],   # beam     – vivid orange  ← diagnostic class
    [0.68, 0.22, 0.82],   # column   – purple
    [0.10, 0.88, 0.92],   # window   – cyan
    [0.95, 0.38, 0.62],   # door     – pink
    [0.24, 0.48, 0.85],   # table    – blue
    [0.92, 0.14, 0.14],   # chair    – red
    [0.18, 0.80, 0.38],   # sofa     – green
    [0.08, 0.22, 0.70],   # bookcase – dark blue
    [0.78, 0.88, 0.08],   # board    – yellow-green
    [0.45, 0.45, 0.45],   # clutter  – mid gray
], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Camera + projection
# ══════════════════════════════════════════════════════════════════════════════

def make_camera(pts: np.ndarray, az_deg=225.0, el_deg=32.0, dist_factor=0.62,
                fov_deg=75.0):
    """
    Interior corner view: camera placed just inside one ceiling corner,
    looking diagonally down-and-across the full room.
    Ceiling points are stripped before rendering so we see furniture/floor.
    Wide FOV (75°) captures the whole room in one shot.
    """
    mins   = pts.min(axis=0)
    maxs   = pts.max(axis=0)
    # lookat = interior center, slightly above floor level
    center = np.array([
        (mins[0] + maxs[0]) / 2,
        (mins[1] + maxs[1]) / 2,
        mins[2] + (maxs[2] - mins[2]) * 0.38,   # ~38% height = furniture zone
    ])
    diag   = float(np.linalg.norm(maxs - mins))
    az, el = np.radians(az_deg), np.radians(el_deg)
    # Place eye inside the room at the corner, just below ceiling
    eye    = np.array([
        mins[0] + (maxs[0]-mins[0]) * 0.06,     # 6% inside from corner
        mins[1] + (maxs[1]-mins[1]) * 0.06,
        maxs[2] - (maxs[2]-mins[2]) * 0.08,     # just below ceiling
    ])
    return eye, center, fov_deg


def project(pts: np.ndarray, eye, lookat, fov_deg, W, H):
    """Perspective projection → pixel (u,v) + depth."""
    fwd = lookat - eye;  fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]);  right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)

    d   = pts - eye                              # (N,3)
    z   = d @ fwd                                # depth
    x   = d @ right
    y   = d @ up

    f   = 1.0 / np.tan(np.radians(fov_deg / 2))
    u   = (f * x / z)  * (W / 2) + W / 2
    v   = -(f * y / z) * (H / 2) + H / 2
    return u, v, z


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════

def stratified_sample(pts, colors, labels=None, total=280_000, beam_boost=8):
    """
    Downsample to `total` points.  If labels provided, oversample beam (class 3)
    by `beam_boost`x to ensure it's visible despite being rare.
    """
    N = len(pts)
    if N <= total:
        return pts, colors, (labels if labels is not None else None)

    rng = np.random.default_rng(0)

    if labels is None:
        idx = rng.choice(N, total, replace=False)
        return pts[idx], colors[idx], None

    # class-stratified
    keep = []
    beam_mask = (labels == 3)
    n_beam    = beam_mask.sum()
    n_other   = N - n_beam
    budget_beam  = min(n_beam, int(total * beam_boost / (beam_boost + n_other / max(n_beam,1) * total / N)))
    budget_other = total - budget_beam

    if n_beam > 0:
        beam_idx = np.where(beam_mask)[0]
        keep.append(rng.choice(beam_idx, min(n_beam, budget_beam), replace=False))

    other_idx = np.where(~beam_mask)[0]
    keep.append(rng.choice(other_idx, min(n_other, budget_other), replace=False))

    idx = np.concatenate(keep)
    return pts[idx], colors[idx], labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Render single view → numpy RGB image
# ══════════════════════════════════════════════════════════════════════════════

def render_view(pts, rgb_colors, eye, lookat, fov_deg,
                W=1600, H=1200, pt_size=1.8, bg=(0.06, 0.07, 0.09)):
    u, v, z = project(pts, eye, lookat, fov_deg, W, H)

    # Clip to frustum
    mask = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[mask], v[mask], z[mask]
    c = rgb_colors[mask]

    # Depth sort (back to front)
    order = np.argsort(-z)
    u, v, c = u[order], v[order], c[order]

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, W);  ax.set_ylim(H, 0)
    ax.set_aspect("equal");  ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Depth-based point size (closer = slightly larger)
    z_norm  = (z[order] - z.min()) / (z.ptp() + 1e-8)
    sizes   = (pt_size * 100) * (0.5 + 0.5 * (1.0 - z_norm))   # matplotlib s= area units

    ax.scatter(u, v, c=c, s=sizes, linewidths=0, rasterized=True, alpha=0.92)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(H, W, 4)[:, :, :3]
    plt.close(fig)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Legend + title overlays (PIL)
# ══════════════════════════════════════════════════════════════════════════════

def _font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try: return ImageFont.truetype(path, size)
        except: pass
    return ImageFont.load_default()

def _font_reg(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        try: return ImageFont.truetype(path, size)
        except: pass
    return ImageFont.load_default()


def add_title(arr, text, font_size=42, bg=(15, 18, 23)):
    bar_h = font_size + 20
    bar   = np.full((bar_h, arr.shape[1], 3), bg, dtype=np.uint8)
    img   = Image.fromarray(np.vstack([bar, arr]))
    draw  = ImageDraw.Draw(img)
    font  = _font(font_size)
    bbox  = draw.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0]
    x     = max(12, (arr.shape[1] - tw) // 2)
    draw.text((x, 8), text, font=font, fill=(255, 255, 255))
    return np.array(img)


def add_legend(arr, used_classes, bg=(15, 18, 23)):
    swatch_w, swatch_h = 22, 22
    pad_x, pad_y = 16, 10
    col_w  = 148
    ncols  = 4
    nrows  = (len(used_classes) + ncols - 1) // ncols
    leg_h  = nrows * (swatch_h + pad_y) + pad_y

    bar  = np.full((leg_h, arr.shape[1], 3), bg, dtype=np.uint8)
    img  = Image.fromarray(np.vstack([arr, bar]))
    draw = ImageDraw.Draw(img)
    font = _font_reg(19)

    base_y = arr.shape[0] + pad_y
    for i, c in enumerate(used_classes):
        row, col = divmod(i, ncols)
        x = pad_x + col * col_w
        y = base_y + row * (swatch_h + pad_y)
        r, g, b = (CLASS_COLORS[c] * 255).astype(np.uint8).tolist()
        # Special border for beam
        border = (255, 120, 0) if c == 3 else (60, 60, 60)
        draw.rectangle([x, y, x+swatch_w, y+swatch_h], fill=(r, g, b), outline=border, width=2)
        label = CLASS_NAMES[c] + (" ← BEAM" if c == 3 else "")
        col_text = (255, 165, 40) if c == 3 else (220, 220, 220)
        draw.text((x + swatch_w + 6, y + 2), label, font=font, fill=col_text)

    return np.array(img)


def divider_v(h, w=4, color=(70, 70, 70)):
    return np.full((h, w, 3), color, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data …")
    coord  = np.load(DATA + "coord.npy").astype(np.float32)
    color  = np.load(DATA + "color.npy").astype(np.float32) / 255.0
    seg    = np.load(DATA + "segment.npy").reshape(-1).astype(int)
    pred   = np.load(RESULT + "Area_5-office_39_pred.npy").astype(int)

    gt_colors   = CLASS_COLORS[seg]
    pred_colors = CLASS_COLORS[pred]
    used        = sorted(c for c in np.unique(seg).tolist() if c != 0)  # skip ceiling
    print(f"  {len(coord):,} points  |  GT classes: {[CLASS_NAMES[c] for c in used]}")

    # Camera for full-room view
    eye, lookat, fov = make_camera(coord, az_deg=225, el_deg=28)
    print(f"  Camera eye={eye.round(1)}  lookat={lookat.round(1)}  fov={fov}°")

    W, H = 1600, 1200

    # ── Strip ceiling + front walls to expose room interior ───────────────────
    # Camera sits at the (min_x, min_y) corner — remove:
    #   1. Ceiling (class 0)
    #   2. Wall points (class 2) on the camera side of the room centre
    #      (dot product of wall→centre vs centre→camera < 0)

    room_centre = coord.mean(axis=0)
    eye_approx  = np.array([coord[:,0].min(), coord[:,1].min(), coord[:,2].max()])
    cam_dir_xy  = eye_approx[:2] - room_centre[:2]   # vector: centre → camera (XY only)

    # For each wall point: if it lies "toward" the camera relative to centre → cull
    wall_mask     = seg == 2
    wall_pts_xy   = coord[wall_mask, :2] - room_centre[:2]  # (N_wall, 2)
    toward_cam    = (wall_pts_xy * cam_dir_xy).sum(axis=1) > 0  # True = facing camera
    # Also cull the nearest 35% by distance in cam_dir so partial walls stay
    proj          = (wall_pts_xy * cam_dir_xy / (np.linalg.norm(cam_dir_xy)+1e-8)).sum(axis=1)
    proj_thresh   = np.percentile(proj[toward_cam], 60) if toward_cam.any() else 1e9
    cull_wall     = wall_mask.copy()
    cull_wall[wall_mask] = toward_cam & (proj > proj_thresh * 0.25)

    keep = ~((seg == 0) | cull_wall)   # drop ceiling + front walls

    # For RGB: same spatial mask (no GT labels) — drop top z + points near cam corner
    z_ceil_thresh = coord[:, 2].max() - (coord[:, 2].ptp() * 0.06)
    # approximate front-wall strip in XY for RGB
    px = coord[:, 0] - room_centre[0]
    py = coord[:, 1] - room_centre[1]
    toward_rgb = (px * cam_dir_xy[0] + py * cam_dir_xy[1]) > np.percentile(
        (px * cam_dir_xy[0] + py * cam_dir_xy[1]), 72
    )
    keep_rgb = ~(toward_rgb | (coord[:, 2] > z_ceil_thresh))

    coord_gt  = coord[keep];      color_gt  = color[keep]
    seg_gt    = seg[keep];        pred_gt   = pred[keep]
    coord_rgb = coord[keep_rgb];  color_rgb = color[keep_rgb]

    print(f"  After strip: GT {len(coord_gt):,} pts | RGB {len(coord_rgb):,} pts")

    # ── Downsample ────────────────────────────────────────────────────────────
    pts_s,   col_rgb_s, _     = stratified_sample(coord_rgb, color_rgb, None,       total=400_000)
    pts_s2,  col_gt_s,  seg_s = stratified_sample(coord_gt,  CLASS_COLORS[seg_gt],  seg_gt,  total=400_000, beam_boost=1)
    _,       col_pred_s, _    = stratified_sample(coord_gt,  CLASS_COLORS[pred_gt], seg_gt,  total=400_000, beam_boost=1)
    print(f"  Sampled {len(pts_s):,} pts (RGB) | {len(pts_s2):,} pts (GT/Pred)")

    # ── Render 3 views ────────────────────────────────────────────────────────
    renders = {}
    for label, pts_r, clr in [
        ("rgb",  pts_s,  col_rgb_s),
        ("gt",   pts_s2, col_gt_s),
        ("pred", pts_s2, col_pred_s),
    ]:
        print(f"  Rendering {label} …")
        renders[label] = render_view(pts_r, clr, eye, lookat, fov, W, H, pt_size=0.7)

    titles = {
        "rgb":  "True RGB Scan",
        "gt":   "Ground Truth Labels",
        "pred": "Sonata Prediction  (mIoU = 75.41%  |  Beam IoU = 0.19%)",
    }

    # Save individual panels (no legend on RGB)
    panels = {}
    for k in ("rgb", "gt", "pred"):
        arr = add_title(renders[k], titles[k], font_size=40)
        if k != "rgb":
            arr = add_legend(arr, used)
        panels[k] = arr
        out = os.path.join(OUT, f"office_39_open3d_{k}.png")
        Image.fromarray(arr).save(out, dpi=(300, 300))
        print(f"  saved → {out}")

    # ── Triptych ──────────────────────────────────────────────────────────────
    max_h = max(p.shape[0] for p in panels.values())
    bg3   = (15, 18, 23)

    def pad_panel(p):
        d = max_h - p.shape[0]
        if d <= 0: return p
        return np.vstack([p, np.full((d, p.shape[1], 3), bg3, dtype=np.uint8)])

    div = divider_v(max_h)
    trip = np.hstack([
        pad_panel(panels["rgb"]), div,
        pad_panel(panels["gt"]),  div,
        pad_panel(panels["pred"]),
    ])
    trip = add_title(trip,
        "office_39 — S3DIS Area 5  |  928,712 pts  |  Sonata (PointTransformerV3)",
        font_size=38, bg=bg3)
    out_trip = os.path.join(OUT, "office_39_triptych.png")
    Image.fromarray(trip).save(out_trip, dpi=(300, 300))
    print(f"  saved triptych → {out_trip}  ({trip.shape[1]}×{trip.shape[0]})")

    # ── Beam zoom pair ────────────────────────────────────────────────────────
    # Focus camera on top-of-room beam region
    z_thresh = coord[:, 2].max() - 1.0
    beam_gt_pts = coord[seg == 3] if (seg == 3).any() else coord[coord[:, 2] > z_thresh]
    top_pts     = coord[coord[:, 2] > z_thresh]
    zoom_center = (beam_gt_pts if len(beam_gt_pts) > 0 else top_pts).mean(axis=0)
    extents     = coord.max(axis=0) - coord.min(axis=0)
    diag_room   = float(np.linalg.norm(extents))

    az_z, el_z = np.radians(210), np.radians(18)
    eye_zoom = zoom_center + diag_room * 0.55 * np.array([
        np.cos(el_z)*np.cos(az_z), np.cos(el_z)*np.sin(az_z), np.sin(el_z)
    ])
    fov_zoom = 48.0

    # Only render the top portion points for zoom (faster, crisper)
    top_mask = coord[:, 2] > (coord[:, 2].max() - 2.5)
    pts_top  = coord[top_mask]
    seg_top  = seg[top_mask]
    pred_top = pred[top_mask]

    pts_z, cgt_z, sgt_z  = stratified_sample(pts_top, CLASS_COLORS[seg_top],  seg_top,  total=120_000, beam_boost=15)
    _,     cpd_z, _      = stratified_sample(pts_top, CLASS_COLORS[pred_top], seg_top,  total=120_000, beam_boost=15)

    print("  Rendering beam zoom GT …")
    arr_bgt  = render_view(pts_z, cgt_z, eye_zoom, zoom_center, fov_zoom, W, H, pt_size=2.5)
    print("  Rendering beam zoom Pred …")
    arr_bpd  = render_view(pts_z, cpd_z, eye_zoom, zoom_center, fov_zoom, W, H, pt_size=2.5)

    arr_bgt  = add_title(arr_bgt,  "Beam Region — Ground Truth  (orange = beam)", font_size=36)
    arr_bpd  = add_title(arr_bpd,  "Beam Region — Sonata  (beam predicted as ceiling/wall)", font_size=36)

    beam_classes = sorted(set(np.unique(seg_top).tolist()))
    arr_bgt  = add_legend(arr_bgt,  beam_classes)
    arr_bpd  = add_legend(arr_bpd,  beam_classes)

    max_h2 = max(arr_bgt.shape[0], arr_bpd.shape[0])
    def pad2(a):
        d = max_h2 - a.shape[0]
        return a if d <= 0 else np.vstack([a, np.full((d, a.shape[1], 3), bg3, dtype=np.uint8)])

    beam_pair = np.hstack([pad2(arr_bgt), divider_v(max_h2), pad2(arr_bpd)])
    beam_pair = add_title(beam_pair,
        "Critical Failure: Beam (orange) Predicted as Ceiling/Wall  |  IoU = 0.19%",
        font_size=34, bg=bg3)
    out_beam = os.path.join(OUT, "office_39_beam_zoom.png")
    Image.fromarray(beam_pair).save(out_beam, dpi=(300, 300))
    print(f"  saved beam zoom → {out_beam}  ({beam_pair.shape[1]}×{beam_pair.shape[0]})")

    print("\nAll renders complete.")


if __name__ == "__main__":
    main()
