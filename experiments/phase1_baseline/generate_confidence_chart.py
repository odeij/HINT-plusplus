"""
Phase 1 — Model Confidence vs IoU Chart

Math
────
The model outputs accumulated softmax scores P ∈ ℝ^{N_vox × 13} at voxel
resolution (multiple sliding-window passes accumulate into P before argmax).

To get a proper probability distribution per voxel we normalize each row:

    p_i = P_i / Σ_k P_{ik}          (row-wise ℓ¹ normalization)

Shannon entropy of each voxel:

    H(p_i) = -Σ_k p_{ik} · log₂(p_{ik})      [bits]

Maximum possible entropy (uniform over 13 classes):

    H_max = log₂(13) ≈ 3.700

Normalized entropy ∈ [0, 1]:

    H_norm(p_i) = H(p_i) / H_max

Confidence ∈ [0, 1]:

    C(p_i) = 1 - H_norm(p_i)

Interpretation
    C = 1 → model puts all probability mass on one class  (certain)
    C = 0 → model is completely uniform over all classes  (maximally uncertain)

Aggregation
    We group voxels by their PREDICTED class (argmax of P_i) and report
    the mean confidence per class.  This answers:
      "When the model decides to label something as class X, how certain is it?"

    Note: prob.npy is stored at voxel resolution (before inverse interpolation
    back to the original point cloud), so direct GT-class grouping is not
    available without reconstructing the voxel→point mapping.
"""

import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULT_DIR = "/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/result/"
OUT        = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/charts/"

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
CLASS_COLORS = np.array([
    [0.60,0.60,0.65],[0.55,0.38,0.20],[0.75,0.82,0.93],[1.00,0.50,0.05],
    [0.60,0.20,0.75],[0.10,0.85,0.90],[0.95,0.40,0.65],[0.25,0.45,0.80],
    [0.90,0.15,0.15],[0.20,0.75,0.35],[0.10,0.25,0.65],[0.75,0.85,0.10],
    [0.40,0.40,0.40],
], dtype=np.float32)

NUM_CLASSES = 13
H_MAX       = np.log2(NUM_CLASSES)     # 3.700 bits

# ── Final test-set IoU from test.log ─────────────────────────────────────────
IOU = np.array([
    0.9543, 0.9843, 0.8879, 0.0019, 0.5895,
    0.6745, 0.7978, 0.8594, 0.9110, 0.8038,
    0.8281, 0.8561, 0.6546,
])


def compute_confidence_per_class():
    """
    Aggregate mean confidence and voxel count per PREDICTED class
    across all Area-5 rooms that have a _prob.npy file.
    """
    # accumulators: sum of confidence and count per class
    conf_sum   = np.zeros(NUM_CLASSES, dtype=np.float64)
    conf_count = np.zeros(NUM_CLASSES, dtype=np.int64)
    # also track distribution of predictions
    pred_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    prob_files = sorted(f for f in os.listdir(RESULT_DIR) if f.endswith("_prob.npy"))
    print(f"Processing {len(prob_files)} rooms...")

    for fname in prob_files:
        prob_raw = np.load(os.path.join(RESULT_DIR, fname)).astype(np.float64)

        # ── Normalize rows to proper probability simplex ──────────────────
        row_sums = prob_raw.sum(axis=1, keepdims=True).clip(min=1e-12)
        p = prob_raw / row_sums                                  # [N_vox, 13]

        # ── Shannon entropy per voxel ─────────────────────────────────────
        log_p = np.where(p > 0, np.log2(p.clip(min=1e-12)), 0.0)
        H     = -(p * log_p).sum(axis=1)                        # [N_vox]

        # ── Confidence ───────────────────────────────────────────────────
        C     = 1.0 - (H / H_MAX)                               # [N_vox]

        # ── Predicted class ───────────────────────────────────────────────
        pred_cls = prob_raw.argmax(axis=1)                       # [N_vox]

        # ── Accumulate ───────────────────────────────────────────────────
        for c in range(NUM_CLASSES):
            mask = pred_cls == c
            if mask.any():
                conf_sum[c]   += C[mask].sum()
                conf_count[c] += mask.sum()
        pred_count += np.bincount(pred_cls, minlength=NUM_CLASSES)

    mean_conf   = np.where(conf_count > 0, conf_sum / conf_count, 0.0)
    pred_frac   = pred_count / pred_count.sum()
    return mean_conf, pred_frac, conf_count


def make_chart(mean_conf, pred_frac, conf_count):
    fig, ax1 = plt.subplots(figsize=(15, 6.5))

    x     = np.arange(NUM_CLASSES)
    width = 0.42

    # ── Left axis: IoU bars ───────────────────────────────────────────────
    bars = ax1.bar(x - width/2, IOU, width=width,
                   color=[CLASS_COLORS[c].tolist() for c in range(NUM_CLASSES)],
                   alpha=0.90, label="IoU",
                   edgecolor=["#cc0000" if c==3 else "none" for c in range(NUM_CLASSES)],
                   linewidth=[2.0 if c==3 else 0 for c in range(NUM_CLASSES)])

    ax1.set_ylabel("IoU", fontsize=12, color="#1d4ed8")
    ax1.tick_params(axis="y", labelcolor="#1d4ed8")
    ax1.set_ylim(0, 1.15)

    # ── Right axis: Confidence bars ───────────────────────────────────────
    ax2 = ax1.twinx()
    conf_bars = ax2.bar(x + width/2, mean_conf, width=width,
                        color=[CLASS_COLORS[c].tolist() for c in range(NUM_CLASSES)],
                        alpha=0.45, hatch="///", edgecolor="white", linewidth=0.5,
                        label="Confidence  C = 1 − H/H_max")

    ax2.set_ylabel("Confidence  C = 1 − H(p)/log₂(13)", fontsize=12, color="#374151")
    ax2.tick_params(axis="y", labelcolor="#374151")
    ax2.set_ylim(0, 1.15)

    # ── Annotate values ───────────────────────────────────────────────────
    for i, (iou, conf, n) in enumerate(zip(IOU, mean_conf, conf_count)):
        # IoU label above left bar
        ax1.text(i - width/2, iou + 0.015, f"{iou:.3f}",
                 ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold", color="#1e3a8a")
        # Conf label above right bar
        if n > 0:
            ax2.text(i + width/2, conf + 0.015, f"{conf:.3f}",
                     ha="center", va="bottom", fontsize=7.5,
                     fontweight="bold", color="#374151")
        else:
            ax2.text(i + width/2, 0.02, "none\npred",
                     ha="center", va="bottom", fontsize=6.5, color="#888")

    # ── Beam annotation ───────────────────────────────────────────────────
    beam_idx = 3
    ax1.annotate(
        f"Beam IoU = {IOU[beam_idx]:.4f}\n"
        f"Conf = {mean_conf[beam_idx]:.3f}\n"
        f"({int(conf_count[beam_idx]):,} voxels predicted)",
        xy=(beam_idx, IOU[beam_idx]),
        xytext=(beam_idx + 1.6, 0.38),
        arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.8),
        fontsize=9.5, color="#cc0000", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc0000", alpha=0.9),
    )

    # ── Math annotation box ───────────────────────────────────────────────
    math_text = (
        "Confidence formula:\n"
        "  p_i = P_i / Σ P_ik\n"
        "  H(p) = −Σ p·log₂(p)\n"
        "  C = 1 − H(p) / log₂(13)"
    )
    ax1.text(0.01, 0.97, math_text, transform=ax1.transAxes,
             fontsize=8.5, va="top", ha="left",
             bbox=dict(boxstyle="round", fc="#f8fafc", ec="#94a3b8", alpha=0.92),
             fontfamily="monospace")

    # ── Axes formatting ───────────────────────────────────────────────────
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=10)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    ax1.set_title(
        "Phase 1 Baseline — Model Confidence vs IoU per Class (Area 5, All Rooms)\n"
        "Sonata (PointTransformerV3)  |  Confidence grouped by predicted class  "
        "|  H_max = log₂(13) ≈ 3.700 bits",
        fontsize=12, fontweight="bold",
    )

    # ── Legend ────────────────────────────────────────────────────────────
    h1 = mpatches.Patch(color="#4b70cc", alpha=0.90, label="IoU  (left axis, solid bar)")
    h2 = mpatches.Patch(color="#888888", alpha=0.50, hatch="///",
                         label="Confidence C = 1−H/H_max  (right axis, hatched bar)")
    ax1.legend(handles=[h1, h2], loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT, "05_confidence_vs_iou.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved → {out}")
    return out


def print_summary(mean_conf, pred_frac, conf_count):
    print("\n── Per-Class Summary ─────────────────────────────────────────────")
    print(f"{'Class':<12} {'IoU':>7} {'Confidence':>12} {'Voxels pred':>14}")
    print("-" * 50)
    for c in range(NUM_CLASSES):
        marker = "  ← BEAM" if c == 3 else ""
        print(f"{CLASS_NAMES[c]:<12} {IOU[c]:>7.4f} {mean_conf[c]:>12.4f} "
              f"{int(conf_count[c]):>14,}{marker}")
    print(f"\nH_max = log₂(13) = {H_MAX:.4f} bits")
    print(f"\nBeam: {int(conf_count[3]):,} voxels predicted as beam across all Area-5 rooms")


if __name__ == "__main__":
    mean_conf, pred_frac, conf_count = compute_confidence_per_class()
    print_summary(mean_conf, pred_frac, conf_count)
    make_chart(mean_conf, pred_frac, conf_count)
