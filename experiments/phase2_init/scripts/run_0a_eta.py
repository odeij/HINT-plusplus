"""
HINT++ Phase 2 — Sub-step 0A: per-class teacher confidence eta_k.

For each of the 68 Area-5 _prob.npy files (accumulated 13-class softmax at
voxel resolution), row-normalize to a probability simplex, compute Shannon
entropy with natural log, group voxels by argmax (= predicted class), and
accumulate per-class mean entropy. Then:

    eta_k = 1 - mean_H_k / ln(C),  C = 13

Inputs are streamed room by room — no full Area-5 tensor is held in memory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Default location of the frozen-teacher per-room outputs. Override on
# the command line with --prob-dir if your Pointcept results live
# somewhere else.
DEFAULT_PROB_DIR = Path(
    "/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/"
    "semseg-sonata-s3dis/result"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
NUM_CLASSES = 13
EPS = 1e-8

# Source: experiments/phase1_baseline/README.md (68-room
# evaluation). Replace with load from per_class_iou.json
# once save_gt_files.py has been verified and re-run to
# recover the 9 missing gt rooms.
IOU = np.array([
    0.9543, 0.9843, 0.8879, 0.0019, 0.5895,
    0.6745, 0.7978, 0.8594, 0.9110, 0.8038,
    0.8281, 0.8561, 0.6546,
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--prob-dir",
        type=Path,
        default=DEFAULT_PROB_DIR,
        help=("Directory containing 68 Area_5-*_prob.npy files "
              f"(default: {DEFAULT_PROB_DIR})"),
    )
    return p.parse_args()


def compute_eta(prob_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream the 68 _prob.npy files; return (eta_k, mean_H_k, n_rooms)."""
    sum_H = np.zeros(NUM_CLASSES, dtype=np.float64)
    count = np.zeros(NUM_CLASSES, dtype=np.int64)

    prob_files = sorted(f for f in os.listdir(prob_dir) if f.endswith("_prob.npy"))
    if len(prob_files) != 68:
        raise RuntimeError(
            f"Expected 68 _prob.npy files (Area 5 rooms), got {len(prob_files)}"
        )

    for fname in prob_files:
        prob = np.load(prob_dir / fname).astype(np.float32)
        # Row-normalize accumulated softmax -> probability simplex.
        row_sum = prob.sum(axis=1, keepdims=True).clip(min=1e-12)
        p = prob / row_sum                                          # (N, 13)
        # Shannon entropy with natural log; eps inside log avoids log(0).
        H = -(p * np.log(p + EPS)).sum(axis=1)                      # (N,)
        pred = p.argmax(axis=1)                                     # (N,)
        # Accumulate per predicted class.
        np.add.at(sum_H, pred, H)
        np.add.at(count, pred, 1)

    if (count == 0).any():
        missing = [CLASS_NAMES[i] for i in np.where(count == 0)[0]]
        raise RuntimeError(f"No voxels predicted for classes: {missing}")

    mean_H = sum_H / count
    eta = 1.0 - mean_H / np.log(NUM_CLASSES)
    return eta, mean_H, len(prob_files)


def save_outputs(eta: np.ndarray, mean_H: np.ndarray) -> None:
    eta_dict = {c: float(v) for c, v in zip(CLASS_NAMES, eta)}
    H_dict = {c: float(v) for c, v in zip(CLASS_NAMES, mean_H)}
    (OUT_DIR / "eta_k.json").write_text(json.dumps(eta_dict, indent=2))
    (OUT_DIR / "entropy_per_class.json").write_text(json.dumps(H_dict, indent=2))

    df = pd.DataFrame({
        "class": CLASS_NAMES,
        "iou_pct": (IOU * 100).round(2),
        "eta_k": eta.round(6),
    }).sort_values("eta_k", ascending=False)
    df.to_csv(OUT_DIR / "table1_eta.csv", index=False)

    # Figure 1: scatter, x=IoU, y=eta_k, with fitted line and Pearson r.
    r, _ = pearsonr(IOU, eta)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(IOU, eta, s=70, c="steelblue", edgecolor="black", zorder=3)
    for i, c in enumerate(CLASS_NAMES):
        ax.annotate(c, (IOU[i], eta[i]), textcoords="offset points",
                    xytext=(5, 4), fontsize=9)
    # Fitted line.
    slope, intercept = np.polyfit(IOU, eta, 1)
    xs = np.linspace(IOU.min() - 0.02, IOU.max() + 0.02, 100)
    ax.plot(xs, slope * xs + intercept, color="crimson", linewidth=1.2,
            linestyle="--", label=f"fit (r = {r:.3f})")
    ax.set_xlabel("Per-class IoU (Phase 1, Area 5)")
    ax.set_ylabel(r"$\eta_k$  (1 - mean_H / ln 13)")
    ax.set_title(r"Teacher confidence $\eta_k$ vs per-class IoU")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure1_iou_vs_eta.png", dpi=150)
    plt.close(fig)


def verify(eta: np.ndarray) -> None:
    print("\n=== Sub-step 0A verification ===")
    checks: list[tuple[str, bool, str]] = []

    c1 = bool(((eta >= 0) & (eta <= 1)).all())
    checks.append(("1. all eta_k in [0, 1]",
                   c1, f"min={eta.min():.4f}, max={eta.max():.4f}"))

    c2 = bool(np.isfinite(eta).all() and not np.isnan(eta).any())
    checks.append(("2. no nan or inf",
                   c2, "all finite" if c2 else "found NaN/Inf"))

    floor_eta = float(eta[CLASS_NAMES.index("floor")])
    c3 = floor_eta > 0.80
    checks.append(("3. eta_floor > 0.80",
                   c3, f"eta_floor = {floor_eta:.4f}"))

    # Beam: encodes the central HINT++ finding (confidence != correctness).
    # Beam IoU = 0.19% but the teacher is highly confident on its beam
    # predictions. We assert beam is the LEAST confident class and that it
    # sits meaningfully below the bulk distribution -- not that it is near
    # zero, which would imply self-aware uncertainty the teacher does not
    # have. See experiments/phase1_baseline/README.md.
    beam_idx = CLASS_NAMES.index("beam")
    beam_eta = float(eta[beam_idx])
    argmin_idx = int(np.argmin(eta))
    c4a = argmin_idx == beam_idx
    checks.append(("4a. argmin(eta_k) == beam",
                   c4a, f"argmin -> {CLASS_NAMES[argmin_idx]} ({eta[argmin_idx]:.4f})"))
    c4b = beam_eta < 0.80
    checks.append(("4b. eta_beam < 0.80 (below bulk distribution)",
                   c4b, f"eta_beam = {beam_eta:.4f}"))

    r, p = pearsonr(IOU, eta)
    c5 = r > 0
    checks.append(("5. Pearson(IoU, eta) > 0",
                   c5, f"r = {r:.4f}, p = {p:.2e}"))

    for name, ok, info in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Sub-step 0A FAILED verification")


def main() -> None:
    args = parse_args()
    print(f"Reading _prob.npy files from {args.prob_dir}")
    eta, mean_H, n = compute_eta(args.prob_dir)
    print(f"Processed {n} rooms")
    save_outputs(eta, mean_H)
    print("\nPer-class results (sorted by eta_k desc):")
    df = pd.read_csv(OUT_DIR / "table1_eta.csv")
    print(df.to_string(index=False))
    verify(eta)
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
