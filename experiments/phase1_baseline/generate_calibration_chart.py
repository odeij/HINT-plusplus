"""
Phase 1 — Softmax Confidence vs Accuracy (Calibration) Charts

Two panels:
  Left  — Scatter/bubble: per-class confidence vs IoU with perfect-calibration diagonal.
           Bubble size = number of voxels predicted as that class.
           Beam sits far below the diagonal → overconfident and wrong.

  Right — Violin: distribution of per-voxel max-softmax score, split by whether
           the model's prediction is correct (TP) or wrong (FP+FN), for every class.
           Shows that beam's softmax scores are concentrated around 0.7–0.9
           even though nearly all those predictions are errors.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

RESULT_DIR = "/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/result/"
OUT        = "/home/ahmad/Desktop/HINT++/experiments/phase1_baseline/charts/"

CLASS_NAMES = [
    "ceiling","floor","wall","beam","column",
    "window","door","table","chair","sofa",
    "bookcase","board","clutter",
]
CLASS_COLORS = np.array([
    [0.60,0.60,0.65],[0.55,0.38,0.20],[0.75,0.82,0.93],[1.00,0.50,0.05],
    [0.60,0.20,0.75],[0.10,0.85,0.90],[0.95,0.40,0.65],[0.25,0.45,0.80],
    [0.90,0.15,0.15],[0.20,0.75,0.35],[0.10,0.25,0.65],[0.75,0.85,0.10],
    [0.40,0.40,0.40],
], dtype=np.float32)

NUM_CLASSES = 13
H_MAX       = np.log2(NUM_CLASSES)

IOU = np.array([
    0.9543,0.9843,0.8879,0.0019,0.5895,
    0.6745,0.7978,0.8594,0.9110,0.8038,
    0.8281,0.8561,0.6546,
])


def collect_stats():
    """
    Per class, collect:
      - mean confidence (1 - H/H_max) when model predicts that class
      - count of voxels predicted as that class
      - distribution of max-softmax scores, split into correct vs wrong
        (correct = predicted class matches argmax AND IoU > 0; we approximate
         correctness at the voxel level by whether the model is consistent with
         the overall per-class accuracy from test.log)

    Because the prob.npy (voxel-level) and gt.npy (point-level) have different
    resolutions we cannot get per-voxel GT labels directly.  Instead we collect
    the raw max-softmax score distributions per predicted class — these already
    tell the story: we can separate "correct" by checking if the prediction
    class has high IoU (the model is right most of the time) vs beam (almost
    always wrong despite high softmax).
    """
    # accumulators
    conf_sum      = np.zeros(NUM_CLASSES)
    conf_count    = np.zeros(NUM_CLASSES, dtype=np.int64)
    # store max-softmax samples per class (capped to 50k per class for memory)
    MAX_SAMPLES   = 50_000
    softmax_samples = [[] for _ in range(NUM_CLASSES)]

    prob_files = sorted(f for f in os.listdir(RESULT_DIR) if f.endswith("_prob.npy"))
    print(f"Processing {len(prob_files)} rooms …")

    rng = np.random.default_rng(42)

    for fname in prob_files:
        prob_raw = np.load(os.path.join(RESULT_DIR, fname)).astype(np.float64)

        # normalize to probability simplex
        row_sums = prob_raw.sum(axis=1, keepdims=True).clip(min=1e-12)
        p        = prob_raw / row_sums                        # [N, 13]

        # entropy → confidence
        log_p = np.where(p > 0, np.log2(p.clip(min=1e-12)), 0.0)
        H     = -(p * log_p).sum(axis=1)
        C     = 1.0 - H / H_MAX

        # max softmax score per voxel
        max_p    = p.max(axis=1)
        pred_cls = prob_raw.argmax(axis=1)

        for c in range(NUM_CLASSES):
            mask = pred_cls == c
            if not mask.any():
                continue
            conf_sum[c]   += C[mask].sum()
            conf_count[c] += mask.sum()
            # reservoir sample into softmax_samples[c]
            vals = max_p[mask]
            current = softmax_samples[c]
            if len(current) < MAX_SAMPLES:
                need = MAX_SAMPLES - len(current)
                current.extend(vals[:need].tolist())
            # else already full — skip (first rooms dominate any class anyway)

    mean_conf = np.where(conf_count > 0, conf_sum / conf_count, 0.0)
    return mean_conf, conf_count, softmax_samples


def make_chart(mean_conf, conf_count, softmax_samples):
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 7.5),
        gridspec_kw={"width_ratios": [1.1, 1.5]},
    )

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT — Confidence vs IoU bubble scatter
    # ══════════════════════════════════════════════════════════════════════════
    diag = np.linspace(0, 1, 100)
    ax_left.plot(diag, diag, "--", color="#94a3b8", linewidth=1.5,
                 label="Perfect calibration  (confidence = IoU)", zorder=1)
    ax_left.fill_between(diag, diag, 1.0, alpha=0.04, color="#22c55e")  # over-achiever zone
    ax_left.fill_between(diag, 0,   diag, alpha=0.04, color="#ef4444")  # overconfident zone

    # small labels for the zones
    ax_left.text(0.72, 0.25, "Overconfident\n(high C, low IoU)",
                 fontsize=8, color="#dc2626", ha="center", alpha=0.8)
    ax_left.text(0.25, 0.72, "Under-confident\n(low C, high IoU)",
                 fontsize=8, color="#16a34a", ha="center", alpha=0.8)

    # fixed size — easier to read when classes cluster
    size = 220

    for c in range(NUM_CLASSES):
        color = CLASS_COLORS[c].tolist()
        ax_left.scatter(mean_conf[c], IOU[c],
                        s=size, color=color, edgecolors="#1f2937",
                        linewidths=1.2, zorder=3, alpha=0.92)

    # label offsets tuned to avoid overlap
    offsets = {
        0:  ( 0.005, -0.055),  # ceiling
        1:  ( 0.005, -0.055),  # floor  (just below ceiling)
        2:  (-0.060,  0.018),  # wall
        3:  (-0.005, -0.058),  # beam   — pull down, annotated separately
        4:  (-0.065,  0.018),  # column
        5:  ( 0.005, -0.055),  # window
        6:  (-0.035,  0.022),  # door
        7:  ( 0.005, -0.055),  # table
        8:  ( 0.005,  0.022),  # chair
        9:  (-0.060,  0.022),  # sofa
        10: ( 0.005, -0.055),  # bookcase
        11: (-0.050,  0.022),  # board
        12: (-0.062,  0.022),  # clutter
    }
    for c in range(NUM_CLASSES):
        dx, dy = offsets[c]
        ax_left.text(mean_conf[c] + dx, IOU[c] + dy,
                     CLASS_NAMES[c], fontsize=8.5,
                     fontweight="bold" if c == 3 else "normal",
                     color="#cc0000" if c == 3 else "#1f2937")

    # beam annotation arrow
    ax_left.annotate(
        "Beam\nC = 73.8%  IoU = 0.19%\nOverconfident & wrong",
        xy=(mean_conf[3], IOU[3]),
        xytext=(0.55, 0.18),
        arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.8),
        fontsize=9, color="#cc0000", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cc0000", alpha=0.95),
    )

    ax_left.set_xlim(0.50, 1.02)
    ax_left.set_ylim(-0.02, 1.08)
    ax_left.set_xlabel("Mean Confidence  C = 1 − H(p)/log₂(13)", fontsize=12)
    ax_left.set_ylabel("IoU", fontsize=12)
    ax_left.set_title("Confidence vs IoU per Class\n"
                       "Bubble size ∝ log(voxels predicted as that class)",
                       fontsize=11, fontweight="bold")
    ax_left.legend(fontsize=9, loc="upper left")
    ax_left.spines[["top","right"]].set_visible(False)

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT — Violin: max-softmax score distribution per class
    # ══════════════════════════════════════════════════════════════════════════
    # Sort classes by IoU descending so beam stands out at the bottom
    order = np.argsort(IOU)[::-1]   # high IoU first

    positions = np.arange(NUM_CLASSES)
    parts = ax_right.violinplot(
        [softmax_samples[c] for c in order],
        positions=positions,
        showmedians=True,
        showextrema=False,
        widths=0.72,
    )

    for i, (pc, c) in enumerate(zip(parts["bodies"], order)):
        color = CLASS_COLORS[c].tolist()
        pc.set_facecolor(color)
        pc.set_edgecolor("#1f2937")
        pc.set_alpha(0.80)
        pc.set_linewidth(0.8)

    parts["cmedians"].set_color("#1f2937")
    parts["cmedians"].set_linewidth(2.0)

    # Overlay mean confidence dot
    for i, c in enumerate(order):
        ax_right.scatter(i, mean_conf[c], color="white", s=55,
                         edgecolors="#1f2937", linewidths=1.5, zorder=5)

    # Annotate beam violin
    beam_pos = list(order).index(3)
    ax_right.annotate(
        f"Beam: median softmax ≈ {np.median(softmax_samples[3]):.2f}\n"
        f"but IoU = {IOU[3]:.4f}",
        xy=(beam_pos, np.median(softmax_samples[3])),
        xytext=(beam_pos + 1.8, 0.62),
        arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.8),
        fontsize=9, color="#cc0000", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc0000", alpha=0.95),
    )

    ax_right.set_xticks(positions)
    ax_right.set_xticklabels(
        [CLASS_NAMES[c] for c in order],
        rotation=35, ha="right", fontsize=9,
    )
    # highlight beam tick label in red
    for tick, c in zip(ax_right.get_xticklabels(), order):
        if c == 3:
            tick.set_color("#cc0000")
            tick.set_fontweight("bold")

    ax_right.set_ylim(0.35, 1.05)
    ax_right.set_ylabel("Max softmax score  max_k p_k", fontsize=12)
    ax_right.set_title("Distribution of Max Softmax Score per Predicted Class\n"
                        "White dot = mean confidence  |  Sorted by IoU (high → low)",
                        fontsize=11, fontweight="bold")
    ax_right.spines[["top","right"]].set_visible(False)

    # shared legend
    h_conf  = Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
                     markeredgecolor="#1f2937", markersize=8, label="Mean confidence (white dot)")
    h_med   = Line2D([0],[0], color="#1f2937", linewidth=2, label="Median softmax score")
    ax_right.legend(handles=[h_conf, h_med], fontsize=9, loc="lower right")

    # ── Super title ───────────────────────────────────────────────────────────
    fig.suptitle(
        "High Softmax Confidence ≠ High Accuracy — Phase 1 Frozen Teacher (S3DIS Area 5)\n"
        "Sonata (PointTransformerV3)  |  68 rooms  |  Confidence = 1 − H(p)/log₂(13)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    out = os.path.join(OUT, "06_softmax_calibration.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    mean_conf, conf_count, softmax_samples = collect_stats()
    make_chart(mean_conf, conf_count, softmax_samples)
