"""
HINT++ Presentation — 5-Stage Story
Generates PLY + high-quality PNG renders for each stage:

  Stage 1: Input RGB scan
  Stage 2: Initial segmentation (Sonata/PTv3 predictions)
  Stage 3: Simulated human corrections (click markers on errors)
  Stage 4: Correction regions (semi-transparent purple blob)
  Stage 5: Refined segmentation (GT in corrected regions, pred elsewhere)

Room: Area_5 / conferenceRoom_2  —  1,922,357 pts  —  9.54m × 5.28m × 4.43m
Camera: corner (xmin, ymin), looking diagonally toward far walls. Ceiling +
two adjacent walls removed for clean interior view.
"""

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/ahmad/frozen_teacher_project/data/processed/s3dis/Area_5/conferenceRoom_2"
PRED_F   = ("/home/ahmad/frozen_teacher_project/repos/Pointcept"
            "/exp/sonata/semseg-sonata-s3dis/result/Area_5-conferenceRoom_2_pred.npy")

# ── S3DIS Semantic Classes ─────────────────────────────────────────────────────
NAMES = ["ceiling","floor","wall","beam","column","window","door",
         "table","chair","sofa","bookcase","board","clutter"]
CC = np.array([                        # class colours (uint8 RGB)
    [153,153,166],  # 0  ceiling   blue-grey
    [140, 97, 51],  # 1  floor     brown
    [191,209,237],  # 2  wall      pale blue
    [255,127, 13],  # 3  beam      orange
    [153, 51,191],  # 4  column    purple
    [ 26,217,230],  # 5  window    cyan
    [242,102,166],  # 6  door      pink
    [ 64,115,204],  # 7  table     blue
    [230, 38, 38],  # 8  chair     red
    [ 51,191, 89],  # 9  sofa      green
    [ 26, 64,166],  # 10 bookcase  dark blue
    [191,217, 26],  # 11 board     yellow-green
    [102,102,102],  # 12 clutter   grey
], dtype=np.uint8)

PURPLE = np.array([145, 80, 235], dtype=np.uint8)   # mask blob
ORANGE = np.array([255,165,  0], dtype=np.uint8)    # click marker
BG     = (1.0, 1.0, 1.0)                             # white background
W, H   = 2560, 1440
FOV    = 55.0
PSIZE  = 1.1   # point size for standard renders


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load():
    coord = np.load(DATA_DIR + "/coord.npy")
    color = np.load(DATA_DIR + "/color.npy")
    seg   = np.load(DATA_DIR + "/segment.npy").flatten().astype(np.int32)
    pred  = np.load(PRED_F).flatten().astype(np.int32)
    return coord, color, seg, pred


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CAMERA
# ═══════════════════════════════════════════════════════════════════════════════

def setup_camera(coord):
    """
    Place camera at the (xmin, ymin) corner of the room, elevated,
    looking diagonally toward the far (xmax, ymax) corner.
    This ensures the board (mounted on the far X wall) faces the camera.
    """
    xmin, ymin, zmin = coord.min(axis=0)
    xmax, ymax, zmax = coord.max(axis=0)

    # Camera just outside the near corner, 55 % of room height elevation
    eye    = np.array([xmin - 2.2, ymin - 2.2, zmin + (zmax - zmin) * 0.55])
    # Look toward the far side, slightly below mid-height
    lookat = np.array([(xmin+xmax)/2, (ymin+ymax)/2, zmin + (zmax-zmin)*0.32])
    up     = np.array([0.0, 0.0, 1.0])

    cam_corner = (xmin, ymin)   # the corner we're "cutting away"
    return eye, lookat, up, cam_corner


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ROOM MASK  (remove ceiling + two near walls)
# ═══════════════════════════════════════════════════════════════════════════════

def room_valid_mask(coord, seg, cam_corner, cam_eye=None,
                    wall_margin=0.50, cam_cutout_radius=5.2):
    """
    Remove:
    - All ceiling (class 0)
    - Wall slab at each near boundary (wall_margin from xmin/ymin)
    - Any remaining ceiling/wall within cam_cutout_radius of the camera eye
      (catches wall fragments that jut inward past the margin boundary)
    Objects (chairs, tables, boards …) are never removed.
    """
    xmin, ymin = coord[:,0].min(), coord[:,1].min()
    xmax, ymax = coord[:,0].max(), coord[:,1].max()
    cx, cy     = cam_corner

    near_x = xmin if cx < (xmin + xmax) / 2 else xmax
    near_y = ymin if cy < (ymin + ymax) / 2 else ymax

    rm_ceil   = (seg == 0)
    rm_wall_x = (seg == 2) & (np.abs(coord[:,0] - near_x) < wall_margin)
    rm_wall_y = (seg == 2) & (np.abs(coord[:,1] - near_y) < wall_margin)

    # Radius cutout: remove ceiling/wall within cam_cutout_radius of camera
    rm_near = np.zeros(len(coord), dtype=bool)
    if cam_eye is not None:
        xy_dist  = np.sqrt((coord[:,0] - cam_eye[0])**2 +
                           (coord[:,1] - cam_eye[1])**2)
        rm_near  = ((seg == 0) | (seg == 2)) & (xy_dist < cam_cutout_radius)

    return ~(rm_ceil | rm_wall_x | rm_wall_y | rm_near)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLICK OBJECTS  (the three misclassified regions)
# ═══════════════════════════════════════════════════════════════════════════════

def find_click_objects(coord, seg, pred):
    """
    Hard-coded to the three best visual errors in conferenceRoom_2:
      1. board  → clutter  (15 213 pts, tight 0.7 m cluster on far wall)
      2. chair  → floor    (1 151 pts, tight cluster in room centre)
      3. column → wall     (5 499 pts, far corner)
    """
    objects = []

    # ── Object 1: board predicted as clutter ──────────────────────────────────
    m1  = (seg == 11) & (pred == 12)
    pts = coord[m1]
    ctr = np.array([-8.48, -9.26, 1.47])       # known from analysis
    r   = 1.2
    in_cluster = np.linalg.norm(pts - ctr, axis=1) < r
    pts_c = pts[in_cluster]
    objects.append({
        "name": "board → clutter",
        "gt_class": 11, "pred_class": 12,
        "full_mask": m1,
        "center": pts_c.mean(axis=0),
        "radius": r,
        "n_pts": int(m1.sum()),
    })

    # ── Object 2: chair predicted as floor (densest 1.2 m cluster) ────────────
    m2   = (seg == 8) & (pred == 1)
    pts2 = coord[m2]
    ctr2 = np.array([-14.22, -8.74,  0.45])    # densest cell centroid + lift to seat
    r2   = 1.5
    # 2-D distance for clustering (chairs spread in Z, group by XY)
    in2  = np.linalg.norm(pts2[:,:2] - ctr2[:2], axis=1) < r2
    pts2_c = pts2[in2]
    if len(pts2_c) == 0:
        pts2_c = pts2
    objects.append({
        "name": "chair → floor",
        "gt_class": 8, "pred_class": 1,
        "full_mask": m2,
        "center": np.array([-14.22, -8.74, 0.55]),
        "radius": r2,
        "n_pts": int(in2.sum()),
    })

    # ── Object 3: column predicted as wall ────────────────────────────────────
    m3   = (seg == 4) & (pred == 2)
    pts3 = coord[m3]
    ctr3 = np.array([-8.75, -8.10, 1.20])
    # Take the 50th-percentile coherent core
    d3   = np.linalg.norm(pts3 - ctr3, axis=1)
    r3   = float(np.percentile(d3, 60)) + 0.3
    objects.append({
        "name": "column → wall",
        "gt_class": 4, "pred_class": 2,
        "full_mask": m3,
        "center": ctr3,
        "radius": r3,
        "n_pts": int(m3.sum()),
    })

    return objects


def build_mask_region(coord, objects):
    """All points within expanded_radius of each click cluster centre."""
    region = np.zeros(len(coord), dtype=bool)
    for obj in objects:
        r = obj["radius"] + 0.25   # slight expansion beyond click cluster
        d = np.linalg.norm(coord - obj["center"], axis=1)
        region |= (d < r)
    return region


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STAGE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_rgb(coord, color, valid):
    return coord[valid], color[valid]


def stage2_pred(coord, pred, valid):
    return coord[valid], CC[np.clip(pred[valid], 0, 12)]


def stage3_clicks(coord, pred, valid, objects):
    """Predictions + orange sphere + ring at each click centroid."""
    xyz  = coord[valid]
    rgb  = CC[np.clip(pred[valid], 0, 12)].copy()

    extra_pts, extra_rgb = [], []
    for obj in objects:
        c = obj["center"]
        r_sphere = max(obj["radius"] * 0.18, 0.15)

        # Solid sphere
        sph = _fibonacci_sphere(c, r_sphere, 1200)
        extra_pts.append(sph)
        extra_rgb.append(np.tile(ORANGE, (len(sph), 1)))

        # Horizontal targeting ring
        theta   = np.linspace(0, 2*np.pi, 600)
        r_ring  = obj["radius"] * 0.55
        ring    = np.column_stack([
            c[0] + r_ring*np.cos(theta),
            c[1] + r_ring*np.sin(theta),
            np.full(600, c[2]),
        ])
        extra_pts.append(ring)
        extra_rgb.append(np.tile(ORANGE, (600, 1)))

    all_xyz = np.vstack([xyz] + extra_pts)
    all_rgb = np.vstack([rgb] + extra_rgb)
    return all_xyz, all_rgb


def stage4_mask_ply(coord, pred, valid, objects, mask_region):
    """
    PLY representation of the mask: purple-tinted points inside the region
    plus a sparse shell of pure purple points at the boundary.
    """
    xyz = coord[valid]
    rgb = CC[np.clip(pred[valid], 0, 12)].copy()

    # Tint interior
    v_mask = mask_region[valid]
    rgb[v_mask] = (0.50 * rgb[v_mask].astype(float) +
                   0.50 * PURPLE.astype(float)).astype(np.uint8)

    # Shell
    rng = np.random.default_rng(42)
    extra_pts, extra_rgb = [], []
    for obj in objects:
        n_shell = 4000
        dirs    = rng.standard_normal((n_shell, 3))
        dirs   /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii   = obj["radius"] + rng.uniform(0.0, 0.25, n_shell)
        pts     = obj["center"] + dirs * radii[:,None]
        # Clip to room bounding box (loose)
        bbox_min = coord[obj["full_mask"]].min(axis=0) - 0.4
        bbox_max = coord[obj["full_mask"]].max(axis=0) + 0.4
        keep = np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)
        pts  = pts[keep]
        extra_pts.append(pts)
        extra_rgb.append(np.tile(PURPLE, (len(pts), 1)))

    all_xyz = np.vstack([xyz] + extra_pts)
    all_rgb = np.vstack([rgb] + extra_rgb)
    return all_xyz, all_rgb


def stage5_refined(coord, seg, pred, valid, mask_region):
    """GT in mask region, pred elsewhere — correction is spatially contained."""
    xyz   = coord[valid]
    seg_v = seg[valid]
    prd_v = pred[valid]
    msk_v = mask_region[valid]

    rgb = CC[np.clip(prd_v, 0, 12)].copy()
    rgb[msk_v] = CC[np.clip(seg_v[msk_v], 0, 12)]
    return xyz, rgb


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLY WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_ply(path, xyz, rgb):
    n  = len(xyz)
    hdr = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    data = np.zeros(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                               ("r","u1"), ("g","u1"), ("b","u1")])
    data["x"] = xyz[:,0]; data["y"] = xyz[:,1]; data["z"] = xyz[:,2]
    data["r"] = rgb[:,0]; data["g"] = rgb[:,1]; data["b"] = rgb[:,2]
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(data.tobytes())
    mb = os.path.getsize(path)/1e6
    print(f"    PLY  {os.path.basename(path):<40}  {n:>10,} pts  {mb:5.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def _make_pcd(xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    return pcd


def _render(pcd, eye, lookat, up, point_size=PSIZE):
    ren = rendering.OffscreenRenderer(W, H)
    ren.scene.set_background(list(BG) + [1.0])
    ren.scene.scene.enable_sun_light(False)
    ren.scene.scene.set_indirect_light_intensity(0)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    ren.scene.add_geometry("pcd", pcd, mat)
    ren.setup_camera(FOV, lookat.tolist(), eye.tolist(), up.tolist(), 0.01, 500.0)
    img = np.asarray(ren.render_to_image()).copy()
    ren.scene.remove_geometry("pcd")
    del ren
    return img


def render_with_mask_overlay(base_pcd, mask_pts, mask_rgb, eye, lookat, up):
    """
    1. Render base scene normally.
    2. Render mask points (purple, oversized) on pure black.
    3. Gaussian-blur the mask render for soft edges.
    4. Alpha-composite at 60 % opacity.
    """
    base_img = _render(base_pcd, eye, lookat, up)

    # Mask render — black background, big points for solid coverage
    mask_pcd = _make_pcd(mask_pts, mask_rgb)
    ren = rendering.OffscreenRenderer(W, H)
    ren.scene.set_background([0.0, 0.0, 0.0, 1.0])
    ren.scene.scene.enable_sun_light(False)
    ren.scene.scene.set_indirect_light_intensity(0)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 14.0
    ren.scene.add_geometry("mask", mask_pcd, mat)
    ren.setup_camera(FOV, lookat.tolist(), eye.tolist(), up.tolist(), 0.01, 500.0)
    raw_mask = np.asarray(ren.render_to_image()).copy()
    ren.scene.remove_geometry("mask")
    del ren

    # Soften edges with two Gaussian passes
    pil_m = Image.fromarray(raw_mask)
    pil_m = pil_m.filter(ImageFilter.GaussianBlur(radius=22))
    pil_m = pil_m.filter(ImageFilter.GaussianBlur(radius=10))
    soft  = np.array(pil_m).astype(float)

    # Alpha derived from brightness of the blurred mask, max 62 %
    alpha   = np.clip(soft.max(axis=2) / 255.0, 0, 1) * 0.62
    alpha3  = alpha[:,:,None]

    composite = (base_img.astype(float) * (1 - alpha3) +
                 soft                   *       alpha3)
    return np.clip(composite, 0, 255).astype(np.uint8)


def render_with_prominent_clicks(base_xyz, base_rgb, objects, eye, lookat, up):
    """
    Two-pass render: base cloud at normal point size, then orange sphere
    markers at point_size=14 on black background, max-blended on top.
    Markers are impossible to miss at any scale.
    """
    img_base = _render(_make_pcd(base_xyz, base_rgb), eye, lookat, up)

    marker_pts, marker_rgb = [], []
    for obj in objects:
        # Dense sphere at centroid
        sph = _fibonacci_sphere(obj["center"], max(obj["radius"] * 0.20, 0.18), 3000)
        marker_pts.append(sph)
        marker_rgb.append(np.tile(ORANGE, (len(sph), 1)))
        # Horizontal targeting ring
        theta  = np.linspace(0, 2*np.pi, 1200)
        r_ring = max(obj["radius"] * 0.60, 0.30)
        ring   = np.column_stack([
            obj["center"][0] + r_ring * np.cos(theta),
            obj["center"][1] + r_ring * np.sin(theta),
            np.full(1200, obj["center"][2]),
        ])
        marker_pts.append(ring)
        marker_rgb.append(np.tile(ORANGE, (1200, 1)))

    all_pts = np.vstack(marker_pts)
    all_rgb = np.vstack(marker_rgb)

    ren = rendering.OffscreenRenderer(W, H)
    ren.scene.set_background([0.0, 0.0, 0.0, 1.0])
    ren.scene.scene.enable_sun_light(False)
    ren.scene.scene.set_indirect_light_intensity(0)
    mat            = rendering.MaterialRecord()
    mat.shader     = "defaultUnlit"
    mat.point_size = 14.0
    ren.scene.add_geometry("mrk", _make_pcd(all_pts, all_rgb), mat)
    ren.setup_camera(FOV, lookat.tolist(), eye.tolist(), up.tolist(), 0.01, 500.0)
    img_mrk = np.asarray(ren.render_to_image()).copy()
    ren.scene.remove_geometry("mrk")
    del ren

    # Max-blend: wherever markers have colour, they win
    on_screen = img_mrk.max(axis=2) > 15
    result = img_base.copy()
    result[on_screen] = img_mrk[on_screen]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 8. POST-PROCESSING  (labels, legend, click markers)
# ═══════════════════════════════════════════════════════════════════════════════

def _fonts():
    try:
        bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
        reg  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        sm   = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        bold = reg = sm = ImageFont.load_default()
    return bold, reg, sm


def add_stage_label(arr, title, subtitle=None):
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    bold, reg, sm = _fonts()
    x, y = 32, H - 95
    for dx, dy, col in [(1,1,(200,200,200)), (0,0,(20,20,20))]:
        draw.text((x+dx, y+dy), title, font=bold, fill=col)
    if subtitle:
        for dx, dy, col in [(1,1,(200,200,200)), (0,0,(80,80,80))]:
            draw.text((x+dx, y+42+dy), subtitle, font=reg, fill=col)
    return np.array(img)


def add_legend(arr, class_ids):
    """Compact legend bottom-right, showing only present classes."""
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    _, _, sm = _fonts()
    sw, pad, col_w, cols = 16, 6, 140, 4
    n    = len(class_ids)
    rows = (n + cols - 1) // cols
    y0   = H - rows*(sw+pad) - 110
    x0   = W - cols*col_w - 12
    for idx, c in enumerate(class_ids):
        row, col = divmod(idx, cols)
        x = x0 + col*col_w
        y = y0 + row*(sw+pad)
        r,g,b = CC[c].tolist()
        draw.rectangle([x, y, x+sw, y+sw], fill=(r,g,b))
        for dx, dy, fc in [(1,1,(200,200,200)), (0,0,(30,30,30))]:
            draw.text((x+sw+4+dx, y+1+dy), NAMES[c], font=sm, fill=fc)
    return np.array(img)


def _project(pts, eye, lookat, up):
    """Project 3-D world points to 2-D image coordinates."""
    fwd   = lookat - eye;  fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up_c  = np.cross(right, fwd)
    f     = (H / 2) / np.tan(np.radians(FOV) / 2)
    out   = []
    for p in pts:
        v  = p - eye
        cx, cy, cz = np.dot(right,v), np.dot(up_c,v), np.dot(fwd,v)
        if cz < 0.1:
            out.append(None); continue
        px = int(f * cx / cz + W/2)
        py = int(-f * cy / cz + H/2)
        out.append((px, py))
    return out


def draw_click_markers(arr, objects, eye, lookat, up):
    """
    Orange circle + crosshair at the projected centroid of each click object,
    with a subtle glow ring.
    """
    img   = Image.fromarray(arr.copy())
    draw  = ImageDraw.Draw(img)
    _, reg, sm = _fonts()

    pixels = _project([o["center"] for o in objects], eye, lookat, up)

    for (px, py), obj in zip(pixels, objects):
        if px is None or not (20 < px < W-20 and 20 < py < H-20):
            continue
        R  = 58   # ring radius (px)
        Rg = 82   # glow radius

        # Outer glow ring
        draw.ellipse([px-Rg, py-Rg, px+Rg, py+Rg],
                     outline=(255, 220, 60), width=2)
        # Main bright ring
        draw.ellipse([px-R, py-R, px+R, py+R],
                     outline=(255, 155, 0), width=4)
        # Crosshair
        for x1, y1, x2, y2 in [
            (px-R-20, py,      px+R+20, py),
            (px,      py-R-20, px,      py+R+20),
        ]:
            draw.line([(x1, y1), (x2, y2)], fill=(255, 155, 0), width=3)
        # Centre dot
        draw.ellipse([px-7, py-7, px+7, py+7], fill=(255, 230, 120))

        # Label
        label = obj["name"].split("→")[1].strip()
        for dx, dy, fc in [(1, 1, (200,200,200)), (0, 0, (180, 80, 0))]:
            draw.text((px+R+12+dx, py-14+dy), f"← {label}", font=sm, fill=fc)

    return np.array(img)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fibonacci_sphere(center, radius, n):
    golden = (1 + 5**0.5) / 2
    i  = np.arange(n)
    th = np.arccos(1 - 2*(i+0.5)/n)
    ph = 2*np.pi*i/golden
    pts = center + radius * np.column_stack([
        np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    return pts


def save_png(arr, name):
    path = os.path.join(OUT_DIR, name)
    Image.fromarray(arr).save(path, dpi=(300,300))
    mb = os.path.getsize(path)/1e6
    print(f"    PNG  {name:<40}  {mb:5.1f} MB")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "─" * 62
    print(sep)
    print("  HINT++ Presentation Story Generator")
    print(sep)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\n[1/8] Loading data …")
    coord, color, seg, pred = load()
    print(f"      {len(coord):,} points")

    # ── Camera ───────────────────────────────────────────────────────────────
    print("[2/8] Camera …")
    eye, lookat, up, cam_corner = setup_camera(coord)
    print(f"      eye    {np.round(eye,2).tolist()}")
    print(f"      lookat {np.round(lookat,2).tolist()}")

    # ── Room mask ────────────────────────────────────────────────────────────
    print("[3/8] Removing ceiling + adjacent walls …")
    valid = room_valid_mask(coord, seg, cam_corner, cam_eye=eye)
    print(f"      {valid.sum():,} pts remain  ({100*valid.mean():.1f} %)")

    # ── Click objects ────────────────────────────────────────────────────────
    print("[4/8] Click objects …")
    objects = find_click_objects(coord, seg, pred)
    for o in objects:
        print(f"      {o['name']:<22}  {o['n_pts']:>6,} pts  "
              f"centre {np.round(o['center'],2).tolist()}")

    mask_region = build_mask_region(coord, objects)
    print(f"      mask region: {mask_region.sum():,} pts")

    # ── Build point clouds ───────────────────────────────────────────────────
    print("[5/8] Building 5 point clouds …")
    s1_xyz, s1_rgb = stage1_rgb(coord, color, valid)
    s2_xyz, s2_rgb = stage2_pred(coord, pred, valid)
    s3_xyz, s3_rgb = stage3_clicks(coord, pred, valid, objects)
    s4_xyz, s4_rgb = stage4_mask_ply(coord, pred, valid, objects, mask_region)
    s5_xyz, s5_rgb = stage5_refined(coord, seg, pred, valid, mask_region)

    # ── Write PLYs ───────────────────────────────────────────────────────────
    print("[6/8] Writing PLY files …")
    ply_specs = [
        ("stage1_input_rgb.ply",      s1_xyz, s1_rgb),
        ("stage2_initial_seg.ply",    s2_xyz, s2_rgb),
        ("stage3_clicks.ply",         s3_xyz, s3_rgb),
        ("stage4_region_mask.ply",    s4_xyz, s4_rgb),
        ("stage5_refined_seg.ply",    s5_xyz, s5_rgb),
    ]
    for name, xyz, rgb in ply_specs:
        write_ply(os.path.join(OUT_DIR, name), xyz, rgb)

    # ── Determine classes present (for legend) ───────────────────────────────
    used = sorted(set(np.unique(pred[valid]).tolist()) | set(np.unique(seg[valid]).tolist()))
    used = [c for c in used if c < 13]

    # ── Render PNGs ──────────────────────────────────────────────────────────
    print("[7/8] Rendering PNGs …")

    # Stage 1 — RGB
    print("  Stage 1 (RGB scan) …")
    img = _render(_make_pcd(s1_xyz, s1_rgb), eye, lookat, up)
    img = add_stage_label(img, "Input RGB Scan",
                          "conferenceRoom_2 · Area 5 · 1,922,357 pts")
    save_png(img, "stage1_input_rgb.png")

    # Stage 2 — Initial segmentation
    print("  Stage 2 (initial seg) …")
    img = _render(_make_pcd(s2_xyz, s2_rgb), eye, lookat, up)
    img = add_legend(img, used)
    img = add_stage_label(img, "Initial Segmentation (Frozen Teacher)",
                          "Sonata / PTv3 · mIoU 75.4 % · allAcc 92.7 % (S3DIS Area 5)")
    save_png(img, "stage2_initial_seg.png")

    # Stage 3 — Click markers (two-pass: base + large-point-size orange markers)
    print("  Stage 3 (click markers) …")
    img = render_with_prominent_clicks(s2_xyz, s2_rgb, objects, eye, lookat, up)
    img = draw_click_markers(img, objects, eye, lookat, up)
    img = add_legend(img, used)
    img = add_stage_label(img, "Human Corrections — Click Simulation",
                          "3 corrections: " + " · ".join(o["name"] for o in objects))
    save_png(img, "stage3_clicks.png")

    # Stage 4 — Region mask
    print("  Stage 4 (region mask) …")
    # Base = stage-2 prediction cloud
    base_pcd = _make_pcd(s2_xyz, s2_rgb)
    # Mask region cloud: all pts in expanded sphere, coloured purple
    mask_pts_list, mask_rgb_list = [], []
    for obj in objects:
        r   = obj["radius"] + 0.25
        in_r = np.linalg.norm(coord - obj["center"], axis=1) < r
        in_v = in_r & valid
        if in_v.sum() > 0:
            mask_pts_list.append(coord[in_v])
            mask_rgb_list.append(np.tile(PURPLE, (int(in_v.sum()), 1)))
    if mask_pts_list:
        m_pts = np.vstack(mask_pts_list)
        m_rgb = np.vstack(mask_rgb_list)
        img   = render_with_mask_overlay(base_pcd, m_pts, m_rgb, eye, lookat, up)
    else:
        img = _render(base_pcd, eye, lookat, up)
    # Overlay orange markers on the mask image too
    ren_mrk = rendering.OffscreenRenderer(W, H)
    ren_mrk.scene.set_background([0.0, 0.0, 0.0, 1.0])
    ren_mrk.scene.scene.enable_sun_light(False)
    ren_mrk.scene.scene.set_indirect_light_intensity(0)
    mat_mrk = rendering.MaterialRecord()
    mat_mrk.shader = "defaultUnlit"
    mat_mrk.point_size = 14.0
    mrk_pts, mrk_cols = [], []
    for obj in objects:
        sph = _fibonacci_sphere(obj["center"], max(obj["radius"]*0.20, 0.18), 3000)
        mrk_pts.append(sph)
        mrk_cols.append(np.tile(ORANGE, (len(sph), 1)))
    mrk_all_pts = np.vstack(mrk_pts)
    mrk_all_cols = np.vstack(mrk_cols)
    ren_mrk.scene.add_geometry("mrk", _make_pcd(mrk_all_pts, mrk_all_cols), mat_mrk)
    ren_mrk.setup_camera(FOV, lookat.tolist(), eye.tolist(), up.tolist(), 0.01, 500.0)
    img_mrk = np.asarray(ren_mrk.render_to_image()).copy()
    ren_mrk.scene.remove_geometry("mrk")
    del ren_mrk
    on_screen = img_mrk.max(axis=2) > 15
    img[on_screen] = img_mrk[on_screen]
    img = draw_click_markers(img, objects, eye, lookat, up)
    img = add_legend(img, used)
    img = add_stage_label(img, "Correction Regions — Spatial Masks",
                          "Semi-transparent blobs wrap each clicked object")
    save_png(img, "stage4_region_mask.png")

    # Stage 5 — Refined segmentation
    print("  Stage 5 (refined) …")
    img = _render(_make_pcd(s5_xyz, s5_rgb), eye, lookat, up)
    img = add_legend(img, used)
    img = add_stage_label(img, "Refined Segmentation",
                          "GT labels in corrected regions · predictions elsewhere")
    save_png(img, "stage5_refined_seg.png")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[8/8] Done.  All files → {OUT_DIR}")
    print()
    all_files = sorted(f for f in os.listdir(OUT_DIR)
                       if f.endswith((".ply",".png")) and f.startswith("stage"))
    for f in all_files:
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:<42}  {size/1e6:6.1f} MB")
    print(sep)


if __name__ == "__main__":
    main()
