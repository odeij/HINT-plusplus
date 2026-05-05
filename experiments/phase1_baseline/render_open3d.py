"""
Render office_39 point clouds with Open3D OffscreenRenderer.
Produces 3 side-by-side panels (True RGB / GT Classes / Prediction) plus
individual crops — publication quality for the HINT++ abstract diagram.
"""

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont
import os

PLY_DIR = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/pointclouds/"
OUT_DIR = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/charts/"

CLASS_NAMES = [
    "ceiling","floor","wall","beam","column",
    "window","door","table","chair","sofa",
    "bookcase","board","clutter",
]
CLASS_COLORS_F = np.array([
    [0.60,0.60,0.65],[0.55,0.38,0.20],[0.75,0.82,0.93],[1.00,0.50,0.05],
    [0.60,0.20,0.75],[0.10,0.85,0.90],[0.95,0.40,0.65],[0.25,0.45,0.80],
    [0.90,0.15,0.15],[0.20,0.75,0.35],[0.10,0.25,0.65],[0.75,0.85,0.10],
    [0.40,0.40,0.40],
], dtype=np.float32)

# ── Render settings ────────────────────────────────────────────────────────────
W, H       = 1600, 1200      # per-panel resolution
POINT_SIZE = 1.8             # world-space point radius (smaller = crisper)
BG         = (0.06, 0.06, 0.08)  # near-black background


def load_ply(name: str) -> o3d.geometry.PointCloud:
    path = os.path.join(PLY_DIR, name)
    pcd  = o3d.io.read_point_cloud(path)
    # normalise colors to [0,1] (PLY stores uint8, Open3D converts automatically)
    return pcd


def best_camera(pcd: o3d.geometry.PointCloud):
    """
    Compute a diagonal isometric viewpoint that shows the whole room.
    Returns (eye, lookat, up) in world coordinates.
    """
    pts   = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    extents = pts.max(axis=0) - pts.min(axis=0)
    diag    = float(np.linalg.norm(extents))

    # 35° elevation, 225° azimuth — classic architectural view
    az  = np.radians(225)
    el  = np.radians(35)
    eye = center + diag * 0.90 * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    up = np.array([0.0, 0.0, 1.0])
    return eye, center, up


def render_panel(pcd: o3d.geometry.PointCloud, eye, lookat, up) -> np.ndarray:
    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(list(BG) + [1.0])
    renderer.scene.scene.enable_sun_light(False)
    renderer.scene.scene.set_indirect_light_intensity(0)

    mat = rendering.MaterialRecord()
    mat.shader          = "defaultUnlit"
    mat.point_size      = POINT_SIZE

    renderer.scene.add_geometry("pcd", pcd, mat)

    fov   = 55.0
    near  = 0.01
    far   = 1000.0
    renderer.setup_camera(fov, lookat.tolist(), eye.tolist(), up.tolist(), near, far)

    img_o3d = renderer.render_to_image()
    renderer.scene.remove_geometry("pcd")
    del renderer
    return np.asarray(img_o3d)


def add_legend(img_arr: np.ndarray, used_classes: list[int]) -> np.ndarray:
    """Overlay a compact class-color legend on the bottom of the image."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = font_sm = ImageFont.load_default()

    swatch = 18
    pad    = 8
    col_w  = 130
    cols   = 4
    start_y = img.height - (len(used_classes) // cols + 2) * (swatch + pad) - 10

    for idx, c in enumerate(used_classes):
        row = idx // cols
        col = idx % cols
        x = pad + col * col_w
        y = start_y + row * (swatch + pad)
        r, g, b = (CLASS_COLORS_F[c] * 255).astype(np.uint8).tolist()
        draw.rectangle([x, y, x + swatch, y + swatch], fill=(r, g, b))
        draw.text((x + swatch + 5, y + 1), CLASS_NAMES[c], font=font_sm,
                  fill=(240, 240, 240))
    return np.array(img)


def title_bar(img_arr: np.ndarray, text: str, font_size: int = 38) -> np.ndarray:
    h_bar = font_size + 24
    bar   = np.full((h_bar, img_arr.shape[1], 3),
                    (int(BG[0]*255), int(BG[1]*255), int(BG[2]*255)), dtype=np.uint8)
    img   = Image.fromarray(np.vstack([bar, img_arr]))
    draw  = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img_arr.shape[1] - tw) // 2, 10), text, font=font, fill=(255, 255, 255))
    return np.array(img)


def main():
    print("Loading point clouds …")
    pcd_rgb  = load_ply("office_39_rgb.ply")
    pcd_gt   = load_ply("office_39_gt.ply")
    pcd_pred = load_ply("office_39_pred.ply")

    eye, lookat, up = best_camera(pcd_gt)
    print(f"Camera: eye={eye.round(2)}  lookat={lookat.round(2)}")

    # ── Classes present in GT ──────────────────────────────────────────────────
    seg  = np.load("/home/ahmad/frozen_teacher_project/data/processed/s3dis/Area_5/office_39/segment.npy").reshape(-1)
    used = sorted(np.unique(seg).tolist())
    print(f"GT classes present: {[CLASS_NAMES[c] for c in used]}")

    panels = []
    for pcd, label in [
        (pcd_rgb,  "True RGB Scan"),
        (pcd_gt,   "Ground Truth Labels"),
        (pcd_pred, "Sonata Prediction"),
    ]:
        print(f"  Rendering: {label} …")
        arr = render_panel(pcd, eye, lookat, up)
        arr = title_bar(arr, label, font_size=40)
        panels.append(arr)

    # ── Add legend to GT and Pred panels ──────────────────────────────────────
    panels[1] = add_legend(panels[1], used)
    panels[2] = add_legend(panels[2], used)

    # ── Save individual panels ─────────────────────────────────────────────────
    names = ["office_39_open3d_rgb.png", "office_39_open3d_gt.png", "office_39_open3d_pred.png"]
    for arr, name in zip(panels, names):
        out = os.path.join(OUT_DIR, name)
        Image.fromarray(arr).save(out, dpi=(300, 300))
        print(f"  saved → {out}  ({arr.shape[1]}×{arr.shape[0]})")

    # ── Triptych (3-panel side by side) ───────────────────────────────────────
    # Pad all panels to same height
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        pad = np.full((max_h - p.shape[0], p.shape[1], 3),
                      (int(BG[0]*255), int(BG[1]*255), int(BG[2]*255)), dtype=np.uint8)
        padded.append(np.vstack([p, pad]))

    divider = np.full((max_h, 4, 3),
                      (80, 80, 80), dtype=np.uint8)
    triptych = np.hstack([padded[0], divider, padded[1], divider, padded[2]])

    # Super-title
    triptych = title_bar(triptych,
        "office_39 — S3DIS Area 5  |  928,712 pts  |  Sonata (PointTransformerV3)",
        font_size=36)

    out_trip = os.path.join(OUT_DIR, "office_39_triptych.png")
    Image.fromarray(triptych).save(out_trip, dpi=(300, 300))
    print(f"\n  saved triptych → {out_trip}  ({triptych.shape[1]}×{triptych.shape[0]})")

    # ── Beam-focused crop ──────────────────────────────────────────────────────
    # Zoom camera into beam region (top of room)
    pts   = np.asarray(pcd_gt.points)
    z_max = pts[:, 2].max()
    z_min = pts[:, 2].max() - 0.8  # top 80 cm
    beam_pts = pts[pts[:, 2] > z_min]
    if len(beam_pts) > 0:
        beam_center = beam_pts.mean(axis=0)
        extents     = np.asarray(pcd_gt.points).max(axis=0) - np.asarray(pcd_gt.points).min(axis=0)
        diag        = float(np.linalg.norm(extents)) * 0.55
        az, el      = np.radians(220), np.radians(20)
        eye_beam    = beam_center + diag * np.array([
            np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)
        ])
        print("  Rendering: Beam zoom (GT) …")
        arr_beam = render_panel(pcd_gt, eye_beam, beam_center, up)
        arr_beam = title_bar(arr_beam, "Beam Region — Ground Truth  (orange = beam)", font_size=34)
        arr_beam = add_legend(arr_beam, used)

        print("  Rendering: Beam zoom (Pred) …")
        arr_pred_beam = render_panel(pcd_pred, eye_beam, beam_center, up)
        arr_pred_beam = title_bar(arr_pred_beam,
            "Beam Region — Sonata Prediction  (0.19% IoU)", font_size=34)
        arr_pred_beam = add_legend(arr_pred_beam, used)

        max_h2 = max(arr_beam.shape[0], arr_pred_beam.shape[0])
        def pad_h(a, h):
            d = h - a.shape[0]
            if d <= 0: return a
            p = np.full((d, a.shape[1], 3),
                        (int(BG[0]*255),int(BG[1]*255),int(BG[2]*255)), dtype=np.uint8)
            return np.vstack([a, p])
        beam_pair = np.hstack([pad_h(arr_beam, max_h2), divider, pad_h(arr_pred_beam, max_h2)])
        beam_pair = title_bar(beam_pair,
            "Beam Failure: Model Predicts Ceiling/Wall Instead of Beam",
            font_size=32)
        out_beam = os.path.join(OUT_DIR, "office_39_beam_zoom.png")
        Image.fromarray(beam_pair).save(out_beam, dpi=(300, 300))
        print(f"  saved beam zoom → {out_beam}  ({beam_pair.shape[1]}×{beam_pair.shape[0]})")

    print("\nAll renders complete.")


if __name__ == "__main__":
    main()
