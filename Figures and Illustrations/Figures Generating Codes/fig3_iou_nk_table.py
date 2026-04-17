import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

CLASSES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
NUM_CLASSES = len(CLASSES)

RESULT_DIR = os.path.expanduser(
    "~/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/result"
)
S3DIS_DIR = os.path.expanduser(
    "~/Desktop/ramiVIPP/s3dis-compressed/s3dis-compressed"
)

def load_all():
    prob_files = sorted(glob.glob(os.path.join(RESULT_DIR, "Area_5-*_prob.npy")))
    if not prob_files:
        print("[ERROR] No *_prob.npy files found.")
        sys.exit(1)

    probs_list, sub_preds_list = [], []
    full_preds_list, full_gts_list = [], []
    skipped = 0

    for pf in prob_files:
        name        = os.path.basename(pf).replace("_prob.npy", "")
        room        = name.replace("Area_5-", "")
        pred_path   = os.path.join(RESULT_DIR, f"{name}_pred.npy")
        raw_gt_path = os.path.join(S3DIS_DIR, "Area_5", room, "segment.npy")

        if not os.path.exists(pred_path):
            print(f"  [WARN] Missing pred for {name}, skipping.")
            skipped += 1
            continue
        if not os.path.exists(raw_gt_path):
            print(f"  [WARN] Missing segment.npy for {room}, skipping.")
            skipped += 1
            continue

        probs   = np.load(pf).astype(np.float32)
        pred    = np.load(pred_path).astype(np.int64).flatten()
        full_gt = np.load(raw_gt_path).astype(np.int64).flatten()

        if pred.shape[0] != full_gt.shape[0]:
            print(f"  [WARN] pred/gt mismatch for {name}: "
                  f"pred={pred.shape[0]} gt={full_gt.shape[0]}, skipping.")
            skipped += 1
            continue

        probs_list.append(probs)
        sub_preds_list.append(probs.argmax(axis=1))
        full_preds_list.append(pred)
        full_gts_list.append(full_gt)

    print(f"Loaded {len(probs_list)} rooms ({skipped} skipped).")
    if not probs_list:
        print("[ERROR] No rooms loaded.")
        sys.exit(1)

    return (
        np.concatenate(probs_list,      axis=0),
        np.concatenate(sub_preds_list,  axis=0),
        np.concatenate(full_preds_list, axis=0),
        np.concatenate(full_gts_list,   axis=0),
    )

def compute_iou(full_preds, full_gts):
    valid = (full_gts >= 0) & (full_gts < NUM_CLASSES)
    p, g  = full_preds[valid], full_gts[valid]
    iou   = np.zeros(NUM_CLASSES)
    for k in range(NUM_CLASSES):
        tp = np.sum((p == k) & (g == k))
        fp = np.sum((p == k) & (g != k))
        fn = np.sum((p != k) & (g == k))
        denom = tp + fp + fn
        iou[k] = tp / denom if denom > 0 else 0.0
    return iou

def compute_nk(probs, sub_preds):
    max_probs = probs.max(axis=1)
    nk = np.zeros(NUM_CLASSES)
    for k in range(NUM_CLASSES):
        mask = sub_preds == k
        nk[k] = max_probs[mask].mean() if mask.sum() > 0 else 0.0
    return nk

def make_table_figure(iou, nk, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)

    order = np.argsort(nk)[::-1]
    iou_s = iou[order] * 100
    nk_s  = nk[order]
    cls_s = [CLASSES[i] for i in order]

    GREEN = "#d4edda"
    AMBER = "#fff3cd"
    RED   = "#f8d7da"

    def row_color(v):
        if v >= 70: return GREEN
        if v >= 30: return AMBER
        return RED

    fig_w  = 3.5
    row_h  = 0.22
    head_h = 0.30
    fig_h  = head_h + NUM_CLASSES * row_h + 0.25

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = [0.42, 0.29, 0.29]
    col_labels = ["Class", "IoU (%)", "n_k"]
    col_x      = [sum(col_widths[:i]) for i in range(3)]

    for lbl, x, w in zip(col_labels, col_x, col_widths):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1 - head_h / fig_h), w, head_h / fig_h,
            boxstyle="square,pad=0", lw=0, facecolor="#2c3e50",
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(x + w / 2, 1 - head_h / (2 * fig_h), lbl,
                ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", transform=ax.transAxes)

    for i, (cls, v_iou, v_nk) in enumerate(zip(cls_s, iou_s, nk_s)):
        y_frac = 1 - head_h / fig_h - (i + 1) * row_h / fig_h
        cy     = y_frac + (row_h / fig_h) / 2
        bg     = row_color(v_iou)

        for x, w in zip(col_x, col_widths):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y_frac), w, row_h / fig_h,
                boxstyle="square,pad=0", lw=0.4,
                edgecolor="#cccccc", facecolor=bg,
                transform=ax.transAxes, clip_on=False
            ))

        ax.text(col_x[0] + 0.01, cy, cls,
                ha="left", va="center", fontsize=7,
                fontfamily="monospace", transform=ax.transAxes)

        fw = "bold" if v_iou >= 70 else "normal"
        ax.text(col_x[1] + col_widths[1] / 2, cy, f"{v_iou:.1f}",
                ha="center", va="center", fontsize=7,
                fontweight=fw, transform=ax.transAxes)

        ax.text(col_x[2] + col_widths[2] / 2, cy, f"{v_nk:.3f}",
                ha="center", va="center", fontsize=7,
                transform=ax.transAxes)

    legend_items = [
        mpatches.Patch(facecolor=GREEN, edgecolor="#aaa", label="IoU >= 70%"),
        mpatches.Patch(facecolor=AMBER, edgecolor="#aaa", label="30% <= IoU < 70%"),
        mpatches.Patch(facecolor=RED,   edgecolor="#aaa", label="IoU < 30%"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=3,
              fontsize=6, frameon=False, handlelength=1.0, handleheight=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.text(0.5, -0.01,
             "Table 1. Per-class IoU and mean max-softmax confidence $n_k$ "
             "on S3DIS Area 5 (68 rooms), sorted by $n_k$ descending.",
             ha="center", va="top", fontsize=6, style="italic",
             transform=fig.transFigure)

    fig.savefig(os.path.join(out_dir, "fig3_iou_nk_table.pdf"), format="pdf")
    fig.savefig(os.path.join(out_dir, "fig3_iou_nk_table.png"), format="png", dpi=300)
    plt.close(fig)
    print("Saved -> figures/fig3_iou_nk_table.pdf")
    print("Saved -> figures/fig3_iou_nk_table.png")

    print(f"\n{'class':<12} {'IoU%':>7} {'n_k':>7}")
    print("-" * 28)
    for cls, v_iou, v_nk in zip(cls_s, iou_s, nk_s):
        print(f"{cls:<12} {v_iou:>7.1f} {v_nk:>7.3f}")
    print(f"\nmIoU = {iou.mean()*100:.2f}%  (eval log = 75.41%)")

if __name__ == "__main__":
    print("Loading data ...")
    probs, sub_preds, full_preds, full_gts = load_all()
    print(f"  subsampled points : {probs.shape[0]:,}")
    print(f"  full-res points   : {full_preds.shape[0]:,}")
    print("Computing IoU ...")
    iou = compute_iou(full_preds, full_gts)
    print("Computing n_k ...")
    nk  = compute_nk(probs, sub_preds)
    print("Rendering figure ...")
    make_table_figure(iou, nk)
    print("Done.")
