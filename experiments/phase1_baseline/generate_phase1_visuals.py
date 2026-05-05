"""
Phase 1 Baseline Visualization — HINT++
Sonata (PointTransformerV3) trained on S3DIS, evaluated on Area 5.

Outputs:
  charts/01_per_class_iou.png         — per-class IoU bar chart
  charts/02_training_curve.png        — validation mIoU over training
  charts/03_confusion_matrix.png      — normalized confusion matrix (Area 5)
  charts/04_per_room_miou.png         — per-room mIoU distribution
  pointclouds/<room>_gt.png           — ground truth segmentation (top-down)
  pointclouds/<room>_pred.png         — predicted segmentation (top-down)
  pointclouds/office_39_beam_focus.*  — beam class GT vs Pred comparison
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

# ── paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR  = "/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/result/"
RAW_DIR     = "/home/ahmad/Desktop/HINT++/datasets/Stanford3dDataset_v1.2/Area_5/"
TRAIN_LOG   = "/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/train.log"
OUT_CHARTS  = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/charts/"
OUT_PC      = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/pointclouds/"

os.makedirs(OUT_CHARTS, exist_ok=True)
os.makedirs(OUT_PC,     exist_ok=True)

# ── S3DIS 13-class palette ────────────────────────────────────────────────────
CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
# Vivid, perceptually distinct colors — one per class
CLASS_COLORS = np.array([
    [0.60, 0.60, 0.65],   # 0  ceiling   — cool gray
    [0.55, 0.38, 0.20],   # 1  floor     — brown
    [0.75, 0.82, 0.93],   # 2  wall      — sky blue
    [1.00, 0.50, 0.05],   # 3  beam      — vivid orange  ← focal class
    [0.60, 0.20, 0.75],   # 4  column    — purple
    [0.10, 0.85, 0.90],   # 5  window    — cyan
    [0.95, 0.40, 0.65],   # 6  door      — pink
    [0.25, 0.45, 0.80],   # 7  table     — cobalt blue
    [0.90, 0.15, 0.15],   # 8  chair     — red
    [0.20, 0.75, 0.35],   # 9  sofa      — green
    [0.10, 0.25, 0.65],   # 10 bookcase  — dark navy
    [0.75, 0.85, 0.10],   # 11 board     — yellow-green
    [0.40, 0.40, 0.40],   # 12 clutter   — dark gray
], dtype=np.float32)

IGNORE_LABEL = -1
NUM_CLASSES  = 13


def label_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Map integer labels [N] → float RGB [N,3]; unknown → black."""
    rgb = np.zeros((len(labels), 3), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask = labels == c
        rgb[mask] = CLASS_COLORS[c]
    return rgb


def load_room(room_name: str):
    """Return (xyz, rgb_raw, gt, pred) for a room, or None if files missing."""
    gt_path   = os.path.join(RESULT_DIR, f"Area_5-{room_name}_gt.npy")
    pred_path = os.path.join(RESULT_DIR, f"Area_5-{room_name}_pred.npy")
    raw_path  = os.path.join(RAW_DIR, room_name, f"{room_name}.txt")
    if not (os.path.exists(gt_path) and os.path.exists(pred_path) and os.path.exists(raw_path)):
        return None
    gt   = np.load(gt_path).astype(np.int32)
    pred = np.load(pred_path).astype(np.int32)
    data = np.loadtxt(raw_path, dtype=np.float32)      # [N, 6] X Y Z R G B
    xyz  = data[:, :3]
    rgb  = data[:, 3:] / 255.0
    return xyz, rgb, gt, pred


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Per-class IoU (from the test.log final results)
# ══════════════════════════════════════════════════════════════════════════════
def chart_per_class_iou():
    # Final test results from test.log (Area_5, full set)
    iou_vals = [
        0.9543, 0.9843, 0.8879, 0.0019, 0.5895,
        0.6745, 0.7978, 0.8594, 0.9110, 0.8038,
        0.8281, 0.8561, 0.6546,
    ]
    miou = np.mean(iou_vals)

    colors = [CLASS_COLORS[i].tolist() for i in range(NUM_CLASSES)]
    # Beam bar gets a red border to call it out
    edge_colors = ["#cc0000" if i == 3 else "none" for i in range(NUM_CLASSES)]
    edge_widths = [2.5       if i == 3 else 0.0    for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bars = ax.bar(CLASS_NAMES, iou_vals, color=colors,
                  edgecolor=edge_colors, linewidth=edge_widths, width=0.65)

    # Annotate each bar with its IoU value
    for bar, val in zip(bars, iou_vals):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(miou, color="#333333", linestyle="--", linewidth=1.5, label=f"mIoU = {miou:.4f}")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("IoU", fontsize=13)
    ax.set_title("Phase 1 Baseline — Per-Class IoU on S3DIS Area 5\n"
                 "Sonata (PointTransformerV3), Frozen Teacher", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # Beam annotation
    ax.annotate("Beam: 0.19%\n(critical failure)",
                xy=(3, 0.0019), xytext=(5, 0.25),
                arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.8),
                fontsize=10, color="#cc0000", fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUT_CHARTS, "01_per_class_iou.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart 1] saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Training / Validation mIoU curve
# ══════════════════════════════════════════════════════════════════════════════
def chart_training_curve():
    with open(TRAIN_LOG) as f:
        content = f.read()
    pattern = r"Val result: mIoU/mAcc/allAcc (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)"
    matches  = re.findall(pattern, content)
    miou_vals = np.array([float(m[0]) for m in matches])
    macc_vals = np.array([float(m[1]) for m in matches])
    steps     = np.arange(1, len(miou_vals) + 1)

    best_idx  = int(np.argmax(miou_vals))
    best_miou = miou_vals[best_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, miou_vals, color="#2563eb", linewidth=2.0, label="Val mIoU")
    ax.plot(steps, macc_vals, color="#16a34a", linewidth=1.5, linestyle="--",
            alpha=0.75, label="Val mAcc")
    ax.scatter(best_idx + 1, best_miou, color="#dc2626", s=90, zorder=5,
               label=f"Best mIoU = {best_miou:.4f}")
    ax.axhline(best_miou, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.5)

    ax.set_xlabel("Validation interval (every ~5 min of training)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Phase 1 Baseline — Validation mIoU During Training\n"
                 "Sonata on S3DIS (trained on Areas 1–4, 6; validated on Area 5)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.45, 0.82)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUT_CHARTS, "02_training_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart 2] saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Normalized Confusion Matrix (aggregate Area 5)
# ══════════════════════════════════════════════════════════════════════════════
def chart_confusion_matrix():
    all_gt   = []
    all_pred = []
    for fn in sorted(os.listdir(RESULT_DIR)):
        if fn.endswith("_gt.npy"):
            room = fn.replace("Area_5-", "").replace("_gt.npy", "")
            gt_path   = os.path.join(RESULT_DIR, fn)
            pred_path = os.path.join(RESULT_DIR, f"Area_5-{room}_pred.npy")
            if not os.path.exists(pred_path):
                continue
            gt   = np.load(gt_path).astype(np.int32)
            pred = np.load(pred_path).astype(np.int32)
            valid = (gt >= 0) & (gt < NUM_CLASSES)
            all_gt.extend(gt[valid])
            all_pred.extend(pred[valid])

    cm = confusion_matrix(all_gt, all_pred, labels=list(range(NUM_CLASSES)))
    # Normalize by true class (row-wise)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm.astype(np.float64) / row_sums

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized count (row = true class)")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title("Phase 1 Baseline — Normalized Confusion Matrix (Area 5, All Rooms)\n"
                 "Row = ground truth, column = prediction", fontsize=12, fontweight="bold")

    # Annotate cells where value ≥ 0.05
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if v >= 0.05:
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    # Red box around beam row
    rect = plt.Rectangle((-0.5, 2.5), NUM_CLASSES, 1, fill=False,
                         edgecolor="#cc0000", linewidth=2.0)
    ax.add_patch(rect)
    ax.text(NUM_CLASSES - 0.4, 3.0, "beam row", color="#cc0000",
            fontsize=8, va="center", ha="left")

    fig.tight_layout()
    out = os.path.join(OUT_CHARTS, "03_confusion_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart 3] saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Per-room mIoU distribution (box + scatter)
# ══════════════════════════════════════════════════════════════════════════════
def chart_per_room_miou():
    # Collected from test.log (all 68 rooms; some rooms have no GT → no mIoU)
    room_miou = {}
    with open("/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/test.log") as f:
        content = f.read()
    pattern = r"Area_5-(\S+?) \[(\d+)/68\]-\d+ Batch .+ mIoU ([\d.]+) \(([\d.]+)\)"
    for m in re.finditer(pattern, content):
        room, _, batch_miou, running_miou = m.groups()
        room_miou[room] = float(running_miou)

    # Group by room type
    type_map = {
        "office":        [],
        "hallway":       [],
        "conferenceRoom":[],
        "storage":       [],
        "WC":            [],
        "lobby":         [],
        "pantry":        [],
    }
    for room, v in room_miou.items():
        for key in type_map:
            if room.startswith(key):
                type_map[key].append(v)
                break

    labels = [k for k, vals in type_map.items() if vals]
    data   = [type_map[k] for k in labels]
    all_vals = [v for vals in data for v in vals]
    overall_mean = float(np.mean(all_vals))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5))

    group_colors = ["#3b82f6","#10b981","#f59e0b","#8b5cf6","#ef4444","#06b6d4","#ec4899"]
    for patch, color in zip(bp["boxes"], group_colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Scatter individual points
    for i, (vals, color) in enumerate(zip(data, group_colors[:len(labels)]), start=1):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color,
                   alpha=0.85, s=35, zorder=5)

    ax.axhline(overall_mean, color="#1f2937", linestyle="--", linewidth=1.5,
               label=f"Overall mIoU = {overall_mean:.4f}")
    ax.set_ylabel("mIoU (per room)", fontsize=12)
    ax.set_title("Phase 1 Baseline — Per-Room mIoU Distribution by Room Type\n"
                 "S3DIS Area 5 (68 rooms), Sonata Frozen Teacher", fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUT_CHARTS, "04_per_room_miou.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart 4] saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# POINT CLOUD — top-down 2D view (XY plane, coloured by class)
# ══════════════════════════════════════════════════════════════════════════════
def render_room_topdown(room_name: str, subsample: int = 6):
    """
    Render GT and Pred segmentation in a side-by-side top-down (X-Y) view.
    Saves as <room_name>_gt_vs_pred.png inside OUT_PC.
    subsample: keep every Nth point to keep file size manageable.
    """
    data = load_room(room_name)
    if data is None:
        print(f"  [pc] skipping {room_name} — missing files")
        return
    xyz, rgb_raw, gt, pred = data

    idx  = np.arange(0, len(xyz), subsample)
    xyz  = xyz[idx]
    gt   = gt[idx]
    pred = pred[idx]

    # Build legend patches (only classes present in this room)
    classes_present = sorted(set(gt.tolist()) | set(pred.tolist()))
    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in classes_present if 0 <= c < NUM_CLASSES]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, labels, title in zip(axes, [gt, pred], ["Ground Truth", "Prediction"]):
        colors = label_to_rgb(labels)
        # Top-down: X horizontal, Y vertical; point size scaled to density
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=0.6, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"{room_name} — {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.tick_params(labelsize=8)

    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 7),
               fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"Area 5 — {room_name} | Top-Down Segmentation View",
                 fontsize=14, fontweight="bold", y=1.01)

    out = os.path.join(OUT_PC, f"{room_name}_gt_vs_pred.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [pc] {room_name} saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# BEAM FOCUS — office_39 detailed comparison highlighting beam class
# ══════════════════════════════════════════════════════════════════════════════
def render_beam_focus(room_name: str = "office_39", subsample: int = 4):
    data = load_room(room_name)
    if data is None:
        print(f"  [beam] missing data for {room_name}")
        return
    xyz, rgb_raw, gt, pred = data

    idx  = np.arange(0, len(xyz), subsample)
    xyz_s  = xyz[idx]
    gt_s   = gt[idx]
    pred_s = pred[idx]

    beam_mask_gt   = gt_s == 3
    beam_mask_pred = pred_s == 3

    # ── Panel layout: 4 panels ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 9))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.30)

    # 1. Full GT top-down
    ax_gt = fig.add_subplot(gs[0, :2])
    ax_gt.scatter(xyz_s[:, 0], xyz_s[:, 1], c=label_to_rgb(gt_s),
                  s=0.5, linewidths=0)
    ax_gt.set_title("office_39 — Ground Truth (all classes)", fontsize=11, fontweight="bold")
    ax_gt.set_aspect("equal"); ax_gt.set_xlabel("X (m)"); ax_gt.set_ylabel("Y (m)")

    # 2. Full Pred top-down
    ax_pred = fig.add_subplot(gs[0, 2:])
    ax_pred.scatter(xyz_s[:, 0], xyz_s[:, 1], c=label_to_rgb(pred_s),
                    s=0.5, linewidths=0)
    ax_pred.set_title("office_39 — Prediction (all classes)", fontsize=11, fontweight="bold")
    ax_pred.set_aspect("equal"); ax_pred.set_xlabel("X (m)"); ax_pred.set_ylabel("Y (m)")

    # 3. Beam GT (orange) on gray backdrop
    ax_bgt = fig.add_subplot(gs[1, :2])
    gray = np.full((len(xyz_s), 3), 0.88, dtype=np.float32)
    ax_bgt.scatter(xyz_s[:, 0], xyz_s[:, 1], c=gray, s=0.5, linewidths=0)
    if beam_mask_gt.any():
        ax_bgt.scatter(xyz_s[beam_mask_gt, 0], xyz_s[beam_mask_gt, 1],
                       c=[CLASS_COLORS[3].tolist()], s=3.0, linewidths=0, zorder=3)
    n_beam_gt = beam_mask_gt.sum()
    ax_bgt.set_title(f"BEAM class — Ground Truth ({n_beam_gt:,} points)",
                     fontsize=11, fontweight="bold", color="#c2410c")
    ax_bgt.set_aspect("equal"); ax_bgt.set_xlabel("X (m)"); ax_bgt.set_ylabel("Y (m)")

    # 4. Model prediction coloured: missed beam (FN) in red, what the model called beam (TP/FP)
    ax_bpr = fig.add_subplot(gs[1, 2:])
    # Base: show predicted labels in normal palette
    ax_bpr.scatter(xyz_s[:, 0], xyz_s[:, 1], c=gray, s=0.5, linewidths=0)

    fn_mask = beam_mask_gt & (~beam_mask_pred)   # GT=beam, pred=other  → missed
    tp_mask = beam_mask_gt & beam_mask_pred       # GT=beam, pred=beam   → correct
    fp_mask = (~beam_mask_gt) & beam_mask_pred    # GT=other, pred=beam  → false alarm

    legend_handles = []
    if fn_mask.any():
        ax_bpr.scatter(xyz_s[fn_mask, 0], xyz_s[fn_mask, 1],
                       c=[[0.85, 0.1, 0.1]], s=5.0, linewidths=0, zorder=4)
        legend_handles.append(mpatches.Patch(color=[0.85, 0.1, 0.1],
                                             label=f"Missed beam / FN ({fn_mask.sum():,} pts)"))
    if tp_mask.any():
        ax_bpr.scatter(xyz_s[tp_mask, 0], xyz_s[tp_mask, 1],
                       c=[CLASS_COLORS[3].tolist()], s=5.0, linewidths=0, zorder=5)
        legend_handles.append(mpatches.Patch(color=CLASS_COLORS[3],
                                             label=f"Correct beam / TP ({tp_mask.sum():,} pts)"))
    if fp_mask.any():
        ax_bpr.scatter(xyz_s[fp_mask, 0], xyz_s[fp_mask, 1],
                       c=[[0.2, 0.6, 0.2]], s=5.0, linewidths=0, zorder=4)
        legend_handles.append(mpatches.Patch(color=[0.2, 0.6, 0.2],
                                             label=f"False beam / FP ({fp_mask.sum():,} pts)"))

    n_beam_pred = beam_mask_pred.sum()
    note = "Model predicted 0 beam points — complete miss" if n_beam_pred == 0 else f"{n_beam_pred:,} beam points predicted"
    ax_bpr.set_title(f"BEAM Errors — {note}", fontsize=11, fontweight="bold", color="#c2410c")
    ax_bpr.set_aspect("equal"); ax_bpr.set_xlabel("X (m)"); ax_bpr.set_ylabel("Y (m)")
    if legend_handles:
        ax_bpr.legend(handles=legend_handles, fontsize=9, loc="upper right")

    # Class legend for top two panels
    classes_in_room = sorted(set(gt_s.tolist()))
    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in classes_in_room if 0 <= c < NUM_CLASSES]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 8), fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle(
        "BEAM Class Failure Analysis — office_39 (Area 5)\n"
        "Sonata Phase 1 Baseline: Beam IoU = 0.19%  (mIoU = 75.41%)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    out = os.path.join(OUT_PC, "office_39_beam_focus.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [beam] saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(42)

    print("\n── Charts ─────────────────────────────────────────────────────")
    chart_per_class_iou()
    chart_training_curve()
    chart_confusion_matrix()
    chart_per_room_miou()

    print("\n── Point Cloud Renders (top-down) ─────────────────────────────")
    rooms_to_render = [
        "office_39",        # has beam — focal room
        "conferenceRoom_1", # large open space
        "office_1",         # typical office
        "hallway_1",        # corridor
        "storage_1",        # has beam, different room type
        "lobby_1",          # open lobby
    ]
    for room in rooms_to_render:
        render_room_topdown(room)

    print("\n── Beam Class Focus — office_39 ───────────────────────────────")
    render_beam_focus("office_39")

    print("\nDone. All outputs in experiments/phase1_baseline/")
