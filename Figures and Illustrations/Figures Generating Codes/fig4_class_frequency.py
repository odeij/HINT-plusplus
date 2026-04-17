import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from matplotlib.patches import Patch

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
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
})

CLASSES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
NUM_CLASSES = len(CLASSES)

S3DIS_DATA_DIR = os.path.expanduser(
    "~/Desktop/ramiVIPP/s3dis-compressed/s3dis-compressed"
)

def load_gt_labels():
    pattern = os.path.join(S3DIS_DATA_DIR, "Area_5", "*", "segment.npy")
    seg_files = sorted(glob.glob(pattern))
    if not seg_files:
        print(f"[ERROR] No segment.npy files found under:\n  {pattern}")
        sys.exit(1)
    print(f"Found {len(seg_files)} rooms in Area_5.")
    labels = np.concatenate([
        np.load(f).astype(np.int64).flatten() for f in seg_files
    ])
    print(f"Total points: {labels.shape[0]:,}")
    return labels

def compute_frequency(labels):
    valid  = labels[(labels >= 0) & (labels < NUM_CLASSES)]
    counts = np.bincount(valid, minlength=NUM_CLASSES).astype(float)
    freq   = counts / counts.sum() * 100.0
    return freq, counts

def make_figure(freq, counts, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)

    order  = np.argsort(freq)[::-1]
    freq_s = freq[order]
    cls_s  = [CLASSES[i] for i in order]
    cnt_s  = counts[order].astype(int)

    DOMINANT = "#2176AE"
    NORMAL   = "#3BB5A0"
    RARE     = "#C0392B"

    bar_colors = []
    for i, f in enumerate(freq_s):
        if i < 3:
            bar_colors.append(DOMINANT)
        elif f < 2.0:
            bar_colors.append(RARE)
        else:
            bar_colors.append(NORMAL)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x    = np.arange(NUM_CLASSES)
    bars = ax.bar(x, freq_s, color=bar_colors, width=0.65, linewidth=0, zorder=3)

    for bar, f in zip(bars, freq_s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{f:.1f}%",
            ha="center", va="bottom", fontsize=5.5, color="#333333"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cls_s, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("% of total points", fontsize=7)
    ax.set_xlim(-0.6, NUM_CLASSES - 0.4)
    ax.set_ylim(0, freq_s.max() * 1.22)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_axisbelow(True)
    ax.set_title("Per-class Point Distribution: S3DIS Area 5", fontsize=8, pad=4)

    legend_elems = [
        Patch(facecolor=DOMINANT, label="Dominant (top 3)"),
        Patch(facecolor=NORMAL,   label="Mid-frequency"),
        Patch(facecolor=RARE,     label="Rare (< 2%)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right",
              fontsize=6, frameon=False, handlelength=1.0, handleheight=0.75)

    fig.text(
        0.5, -0.04,
        "Fig. 4. Per-class point frequency in S3DIS Area 5. "
        f"Top 3 classes account for ~{sum(freq_s[:3]):.0f}% of all points, "
        "motivating frequency-aware variance initialisation in HINT3D++ Phase 2.",
        ha="center", va="top", fontsize=5.5, style="italic",
        transform=fig.transFigure
    )

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_class_frequency.pdf"), format="pdf")
    fig.savefig(os.path.join(out_dir, "fig4_class_frequency.png"), format="png", dpi=300)
    plt.close(fig)
    print("Saved -> figures/fig4_class_frequency.pdf")
    print("Saved -> figures/fig4_class_frequency.png")

    print(f"\n{'class':<12} {'%':>7} {'points':>12}")
    print("-" * 34)
    for cls, f, c in zip(cls_s, freq_s, cnt_s):
        print(f"{cls:<12} {f:>7.2f}% {c:>12,}")

if __name__ == "__main__":
    labels = load_gt_labels()
    freq, counts = compute_frequency(labels)
    make_figure(freq, counts)
    print("Done.")
