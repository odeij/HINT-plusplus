"""
save_gt_files.py
----------------
Extracts the subsampled ground-truth labels (same voxelization as used
during eval) for every Area_5 room and saves them as *_gt.npy files
alongside the existing *_pred.npy and *_prob.npy files.

No model inference. No GPU needed. Runs in ~2 minutes.

Usage:
    conda activate frozen_teacher
    cd ~/Desktop/ramiVIPP
    python save_gt_files.py
"""

import os, sys, glob
import numpy as np

RESULT_DIR = os.path.expanduser(
    "~/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/result"
)
POINTCEPT_DIR = os.path.expanduser(
    "~/frozen_teacher_project/repos/Pointcept"
)

# Add Pointcept to path so we can use its dataset loader
sys.path.insert(0, POINTCEPT_DIR)

# Load the experiment config so we use exactly the same data pipeline
import importlib.util

cfg_path = os.path.join(
    POINTCEPT_DIR,
    "exp/sonata/semseg-sonata-s3dis/config.py"
)

# Read config using Pointcept's config loader
from pointcept.utils.config import Config
cfg = Config.fromfile(cfg_path)

# Build the test dataset exactly as the tester does
from pointcept.datasets import build_dataset

print("Building test dataset (same pipeline as eval) ...")
test_dataset = build_dataset(cfg.data.test)
print(f"Dataset has {len(test_dataset)} rooms.")

# For each room, extract the subsampled segment labels and save as _gt.npy
saved, skipped, already = 0, 0, 0

for idx in range(len(test_dataset)):
    data_dict = test_dataset[idx]

    # Pointcept returns a list of fragments + metadata
    # The name and segment are at the top level before fragmentation
    # We need to access them the same way the tester does

    if isinstance(data_dict, list):
        # collated batch — take first item
        data_dict = data_dict[0]

    # Get room name
    name = data_dict.get("name", None)
    if name is None:
        print(f"  [{idx}] No name found, skipping.")
        skipped += 1
        continue

    # Only process Area_5 rooms
    if "Area_5" not in name:
        skipped += 1
        continue

    gt_save_path = os.path.join(RESULT_DIR, f"{name}_gt.npy")

    if os.path.isfile(gt_save_path):
        print(f"  {name}: already exists, skipping.")
        already += 1
        continue

    # Get segment — this is the subsampled GT at the same resolution as pred/prob
    if "origin_segment" in data_dict:
        segment = data_dict["origin_segment"]
    else:
        segment = data_dict.get("segment", None)

    if segment is None:
        print(f"  {name}: no segment found, skipping.")
        skipped += 1
        continue

    if hasattr(segment, "numpy"):
        segment = segment.numpy()
    segment = np.array(segment).astype(np.int16).flatten()

    # Verify shape matches pred file
    pred_path = os.path.join(RESULT_DIR, f"{name}_pred.npy")
    if os.path.isfile(pred_path):
        pred = np.load(pred_path)
        if pred.shape[0] != segment.shape[0]:
            print(f"  [WARN] {name}: pred {pred.shape[0]} vs gt {segment.shape[0]} — mismatch, skipping.")
            skipped += 1
            continue

    np.save(gt_save_path, segment)
    print(f"  {name}: saved {segment.shape[0]:,} points.")
    saved += 1

print(f"\nDone. Saved: {saved}  |  Already existed: {already}  |  Skipped: {skipped}")
