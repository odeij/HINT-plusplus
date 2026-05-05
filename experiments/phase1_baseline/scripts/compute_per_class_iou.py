"""Compute per-class IoU on S3DIS Area 5 from frozen-teacher gt/pred files.

Reads Area_5-*_gt.npy and Area_5-*_pred.npy from --prob-dir, accumulates
a global confusion matrix across all rooms, and writes per-class IoU
(TP / (TP + FP + FN)) plus mIoU to:

    experiments/phase1_baseline/results/per_class_iou.json

Output format:
    {
      "class_names": [...],
      "iou":         {"ceiling": 0.9543, ...},   # fractions in [0, 1]
      "miou":        75.41                         # percent
    }

Run before run_0a_eta.py; the latter loads this JSON instead of using
hardcoded numbers copied from a Pointcept eval log.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Default location of the frozen-teacher per-room outputs. Override with
# --prob-dir if your Pointcept results live elsewhere.
DEFAULT_PROB_DIR = Path(
    "/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/"
    "semseg-sonata-s3dis/result"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "results"

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
NUM_CLASSES = 13


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--prob-dir",
        type=Path,
        default=DEFAULT_PROB_DIR,
        help=("Directory containing Area_5-*_gt.npy and Area_5-*_pred.npy "
              f"(default: {DEFAULT_PROB_DIR})"),
    )
    return p.parse_args()


def compute_iou(prob_dir: Path) -> tuple[np.ndarray, float, int]:
    """Globally-accumulated per-class IoU on Area 5; returns (iou, miou_pct, n_rooms)."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    gt_files = sorted(f for f in os.listdir(prob_dir) if f.endswith("_gt.npy"))
    if not gt_files:
        raise RuntimeError(f"No _gt.npy files found in {prob_dir}")

    for gt_fname in gt_files:
        pred_fname = gt_fname.replace("_gt.npy", "_pred.npy")
        gt = np.load(prob_dir / gt_fname).reshape(-1).astype(np.int64)
        pred = np.load(prob_dir / pred_fname).reshape(-1).astype(np.int64)
        if gt.shape != pred.shape:
            raise RuntimeError(
                f"Shape mismatch in {gt_fname}: gt={gt.shape}, "
                f"pred={pred.shape}"
            )
        valid = (
            (gt >= 0) & (gt < NUM_CLASSES)
            & (pred >= 0) & (pred < NUM_CLASSES)
        )
        gt = gt[valid]
        pred = pred[valid]
        # cm[k, j] = number of points whose ground truth is k and prediction is j
        key = gt * NUM_CLASSES + pred
        cm += np.bincount(
            key, minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    tp = cm.diagonal().astype(np.int64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), 0.0)
    miou_pct = float(iou.mean()) * 100.0
    return iou, miou_pct, len(gt_files)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading gt/pred .npy from {args.prob_dir}")
    iou, miou_pct, n_rooms = compute_iou(args.prob_dir)
    print(f"Processed {n_rooms} Area 5 rooms")

    payload = {
        "class_names": list(CLASS_NAMES),
        "iou": {c: float(round(v, 6)) for c, v in zip(CLASS_NAMES, iou)},
        "miou": round(miou_pct, 2),
    }
    out_path = OUT_DIR / "per_class_iou.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print("\nPer-class IoU:")
    for c, v in zip(CLASS_NAMES, iou):
        print(f"  {c:10s}  {v * 100:6.2f}%")
    print(f"\nmIoU = {miou_pct:.2f}%")
    print(f"Outputs written to {out_path}")


if __name__ == "__main__":
    main()
