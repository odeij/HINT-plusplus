import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter"
]

CLASS_COLORS = np.array([
    [0.65, 0.00, 0.65],  # 0  ceiling   - purple
    [0.56, 0.37, 0.00],  # 1  floor     - brown
    [0.50, 0.50, 0.50],  # 2  wall      - gray
    [0.00, 0.65, 0.65],  # 3  beam      - teal
    [0.00, 0.00, 0.65],  # 4  column    - dark blue
    [0.65, 0.65, 0.00],  # 5  window    - olive
    [0.65, 0.32, 0.00],  # 6  door      - orange
    [0.00, 0.65, 0.00],  # 7  table     - green
    [0.65, 0.00, 0.00],  # 8  chair     - red
    [0.20, 0.20, 0.80],  # 9  sofa      - blue
    [0.32, 0.00, 0.65],  # 10 bookcase  - violet
    [0.00, 0.75, 0.25],  # 11 board     - lime
    [0.65, 0.65, 0.65],  # 12 clutter   - light gray
], dtype=np.float64)


def show_legend(present_classes, title="Legend"):
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in present_classes
    ]
    fig, ax = plt.subplots(figsize=(3, len(present_classes) * 0.45 + 0.8))
    ax.legend(handles=patches, loc='center', fontsize=11, frameon=False)
    ax.axis('off')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def set_interior_camera(vis, coord):
    cx = (coord[:, 0].min() + coord[:, 0].max()) / 2
    cy = (coord[:, 1].min() + coord[:, 1].max()) / 2
    cz_floor = coord[:, 2].min()
    eye_height = cz_floor + 1.5
    ctr = vis.get_view_control()
    ctr.set_lookat([cx, cy, eye_height])
    ctr.set_front([0, -1, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.05)


def open_window(title, pcd, coord, left=True):
    vis = o3d.visualization.Visualizer()
    # Place GT on left half, pred on right half of screen
    x_pos = 0 if left else 910
    vis.create_window(window_name=title, width=900, height=900,
                      left=x_pos, top=50)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    set_interior_camera(vis, coord)
    return vis


def visualize_separate(room_path, pred_path):
    coord = np.load(os.path.join(room_path, "coord.npy")).astype(np.float64)
    gt    = np.load(os.path.join(room_path, "segment.npy")).astype(np.int64).flatten()
    pred  = np.load(pred_path).astype(np.int64).flatten()

    print(f"Points      : {len(coord)}")
    print(f"GT classes  : {np.unique(gt)}")
    print(f"Pred classes: {np.unique(pred)}")

    # ── size mismatch ────────────────────────────────────────────────────
    if len(pred) != len(coord):
        print(f"⚠️  Resampling pred {len(pred)} → {len(coord)}...")
        idx = np.round(np.linspace(0, len(pred) - 1, len(coord))).astype(np.int64)
        pred = pred[idx]

    # ── per-class IoU ────────────────────────────────────────────────────
    print("\n--- Per-class IoU ---")
    ious = []
    for c in range(13):
        tp = np.sum((gt == c) & (pred == c))
        fp = np.sum((gt != c) & (pred == c))
        fn = np.sum((gt == c) & (pred != c))
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float('nan')
        ious.append(iou)
        if not np.isnan(iou):
            print(f"  {CLASS_NAMES[c]:10s}: {iou:.3f}")
    valid = [i for i in ious if not np.isnan(i)]
    print(f"\n  mIoU: {np.mean(valid):.3f}")

    # ── build clouds ─────────────────────────────────────────────────────
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(coord)
    pcd_gt.colors = o3d.utility.Vector3dVector(CLASS_COLORS[gt])

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(coord)   # same coords, no offset
    pcd_pred.colors = o3d.utility.Vector3dVector(CLASS_COLORS[pred])

    # ── legends: two separate matplotlib windows ─────────────────────────
    present = sorted(set(gt.tolist()))
    fig_gt   = show_legend(present, title="Ground Truth Legend")
    fig_pred = show_legend(sorted(set(pred.tolist())), title="Prediction Legend")
    plt.show(block=False)
    plt.pause(0.5)

    # ── two separate Open3D windows ──────────────────────────────────────
    vis_gt   = open_window("Ground Truth",        pcd_gt,   coord, left=True)
    vis_pred = open_window("Model Predictions",   pcd_pred, coord, left=False)

    print("\n[Controls] Left-click: rotate | Scroll: zoom | Right-click: pan | Q: quit")
    print("Rotate both windows to the same angle for comparison.\n")

    # Run both windows in a shared loop
    while True:
        gt_alive   = vis_gt.poll_events()
        pred_alive = vis_pred.poll_events()
        vis_gt.update_renderer()
        vis_pred.update_renderer()
        if not gt_alive or not pred_alive:
            break

    vis_gt.destroy_window()
    vis_pred.destroy_window()
    plt.close('all')


# ── PATHS ────────────────────────────────────────────────────────────────
ROOM_PATH = os.path.expanduser(
    "~/Desktop/ramiVIPP/s3dis-compressed/s3dis-compressed/Area_5/office_19"
)
PRED_PATH = (
    "/home/ahmad/frozen_teacher_project/repos/Pointcept/"
    "exp/sonata/semseg-sonata-s3dis/result/Area_5-office_19_pred.npy"
)

visualize_separate(ROOM_PATH, PRED_PATH)