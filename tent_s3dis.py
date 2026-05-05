"""
TENT: Test-Time Entropy Minimization for PTv3/Sonata on S3DIS
=============================================================
Baseline experiment for HINT3D++ paper.

Purpose:
    Show that TENT (Wang et al., ICLR 2021) — the state-of-the-art
    fully test-time adaptation method — fails to improve rare classes
    (e.g., beam) on S3DIS Area 5, motivating HINT3D++'s human-governed
    exemplar-based approach.

Usage:
    conda activate frozen_teacher
    cd ~/Desktop/HINT++
    PYTHONPATH=~/Desktop/HINT++/Pointcept python tent_s3dis.py \
        --checkpoint checkpoints/model_best.pth \
        --data_root  s3dis-compressed/s3dis-compressed \
        --config     Pointcept/configs/sonata/semseg-sonata-v1m1-3c-s3dis-ft.py \
        --area 5 \
        --lr 0.00001 \
        --online
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ #
# Add Pointcept to path (idempotent — also works if PYTHONPATH set)   #
# ------------------------------------------------------------------ #
POINTCEPT_ROOT = Path(__file__).resolve().parent / "Pointcept"
if str(POINTCEPT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTCEPT_ROOT))


# ================================================================== #
#  S3DIS CLASS INFO                                                   #
# ================================================================== #

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter"
]
NUM_CLASSES = 13
IGNORE_INDEX = -1
BEAM_IDX = CLASS_NAMES.index("beam")  # 3


def _beam_confidence_stats(logits_fp32, segment_long):
    """Compute mean softmax probability for the beam class on (a) voxels
    whose ground-truth label is beam (recall-like confidence) and (b) all
    voxels (false-positive tendency).

    Args:
        logits_fp32:   (N, C) torch.float32 tensor on any device
        segment_long:  (N,)   torch.long    tensor on any device

    Returns:
        dict with keys:
            p_beam_on_beam   — float, NaN if no beam voxels in this room
            p_beam_on_all    — float
            n_beam_voxels    — int
            n_voxels         — int
    """
    probs    = torch.softmax(logits_fp32, dim=-1)
    p_beam   = probs[:, BEAM_IDX]                     # (N,)
    is_beam  = segment_long == BEAM_IDX               # (N,) bool
    n_beam   = int(is_beam.sum().item())
    n_total  = int(segment_long.shape[0])

    p_on_all = float(p_beam.mean().item())
    p_on_gt  = float(p_beam[is_beam].mean().item()) if n_beam > 0 else float("nan")

    return {
        "p_beam_on_beam": p_on_gt,
        "p_beam_on_all":  p_on_all,
        "n_beam_voxels":  n_beam,
        "n_voxels":       n_total,
    }


# ================================================================== #
#  TENT CORE                                                          #
# ================================================================== #

def configure_model_for_tent(model):
    """
    Prepare model for TENT adaptation.

    - Set full model to eval mode (disables dropout, fixes BN running stats)
    - Then set ONLY normalization layers to train mode
      (so LayerNorm computes fresh stats from test batch)
    - Freeze all parameters
    - Unfreeze only normalization affine params (weight=γ, bias=β)
    """
    model.eval()
    model.requires_grad_(False)

    norm_params = []
    norm_layer_count = 0

    for _, module in model.named_modules():
        is_norm = isinstance(module, (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
        ))

        if is_norm:
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True)
                norm_params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                norm_params.append(module.bias)
            norm_layer_count += 1

    total_params   = sum(p.numel() for p in model.parameters())
    adapted_params = sum(p.numel() for p in norm_params)
    pct            = 100.0 * adapted_params / max(total_params, 1)

    print(f"\n[TENT] Model configured:")
    print(f"  Norm layers found  : {norm_layer_count}")
    print(f"  Adapted params     : {adapted_params:,}  ({pct:.3f}% of total)")
    print(f"  Frozen params      : {total_params - adapted_params:,}")

    return model, norm_params


def entropy_loss(logits):
    """Mean Shannon entropy over all points in the batch."""
    probs = F.softmax(logits,     dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    H     = -(probs * log_p).sum(dim=-1)
    return H.mean()


# Cap on voxel count for the gradient-enabled forward. Matches the
# point_max used during S3DIS fine-tuning (SphereCrop point_max=204800),
# so the model is operating in a regime it has been trained on.
MAX_GRAD_VOXELS = 200_000


def _subsample_for_grad(input_dict, max_voxels):
    """Random-subsample a voxel-level dict to <= max_voxels points.

    Returns a dict with the keys the PTv3 backbone needs (coord, grid_coord,
    feat, offset). 'segment' is dropped so the segmentor stays in pure-test
    mode (no supervised CE loss path)."""
    n = input_dict["coord"].shape[0]
    if n <= max_voxels:
        return {
            "coord":      input_dict["coord"],
            "grid_coord": input_dict["grid_coord"],
            "feat":       input_dict["feat"],
            "offset":     input_dict["offset"],
        }
    device = input_dict["coord"].device
    idx, _ = torch.randperm(n, device=device)[:max_voxels].sort()
    return {
        "coord":      input_dict["coord"][idx],
        "grid_coord": input_dict["grid_coord"][idx],
        "feat":       input_dict["feat"][idx],
        "offset":     torch.tensor([idx.shape[0]], device=device, dtype=torch.long),
    }


def tent_adapt_batch(model, input_dict, optimizer, scaler):
    """
    One TENT adaptation step on a single room.

    Uses CUDA autocast (fp16) to match the model's AMP training setting and
    a voxel-count cap to keep activations in budget on a 24GB card. The
    gradient pass runs on a random subsample (matching the training crop
    size); predictions are then computed on the FULL room under no_grad
    using the just-updated weights — so IoU is over all original points.

    Returns (pred_voxel_full, H_before, H_after) with pred_voxel_full
    aligned to the full input_dict voxel grid.
    """
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    segment      = input_dict["segment"]  # (N,) long, voxel-level

    # ---- Step 1: Forward (before adaptation) — entropy + beam confidence ----
    with torch.no_grad(), autocast_ctx:
        out_before    = model(input_dict)
        logits_before = out_before["seg_logits"].float()
        H_before      = entropy_loss(logits_before).item()
        beam_before   = _beam_confidence_stats(logits_before, segment)
    del out_before, logits_before
    torch.cuda.empty_cache()

    # ---- Step 2: Subsampled forward + entropy loss (gradient enabled) ----
    grad_input = _subsample_for_grad(input_dict, MAX_GRAD_VOXELS)

    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        out    = model(grad_input)
        logits = out["seg_logits"]
        loss   = entropy_loss(logits.float())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    del out, logits, loss, grad_input
    torch.cuda.empty_cache()

    # ---- Step 3: Final forward with updated params on FULL room ----
    with torch.no_grad(), autocast_ctx:
        out_after    = model(input_dict)
        logits_after = out_after["seg_logits"].float()
        H_after      = entropy_loss(logits_after).item()
        beam_after   = _beam_confidence_stats(logits_after, segment)
        pred         = logits_after.argmax(dim=-1)

    pred_np = pred.cpu().numpy()
    del out_after, logits_after, pred
    torch.cuda.empty_cache()

    return pred_np, H_before, H_after, beam_before, beam_after


# ================================================================== #
#  IoU COMPUTATION                                                    #
# ================================================================== #

class IoUAccumulator:
    """Accumulates confusion matrix across rooms for mIoU computation."""

    def __init__(self, num_classes, ignore_index=-1):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.confusion    = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt):
        pred = np.asarray(pred).reshape(-1).astype(np.int64)
        gt   = np.asarray(gt).reshape(-1).astype(np.int64)
        valid = (
            (gt != self.ignore_index)
            & (gt >= 0) & (gt < self.num_classes)
            & (pred >= 0) & (pred < self.num_classes)
        )
        pred = pred[valid]
        gt   = gt[valid]
        key = gt * self.num_classes + pred
        self.confusion += np.bincount(
            key, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        tp = self.confusion.diagonal().astype(np.int64)
        fn = self.confusion.sum(axis=1) - tp
        fp = self.confusion.sum(axis=0) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), float('nan'))
        valid = [x for x in iou if not np.isnan(x)]
        miou = float(np.mean(valid)) if valid else 0.0
        return iou.tolist(), miou


# ================================================================== #
#  RESULTS TABLE                                                      #
# ================================================================== #

def print_results_table(iou_no_adapt, miou_no_adapt,
                        iou_tent,     miou_tent):
    print("\n" + "=" * 70)
    print("  TENT BASELINE RESULTS — S3DIS Area 5")
    print("=" * 70)
    print(f"  {'Class':<12}  {'No Adapt':>10}  {'TENT':>10}  {'Δ':>8}")
    print("-" * 70)

    for i, name in enumerate(CLASS_NAMES):
        iou_a = iou_no_adapt[i]
        iou_t = iou_tent[i]

        s_a = f"{iou_a*100:.1f}%" if not np.isnan(iou_a) else "  N/A"
        s_t = f"{iou_t*100:.1f}%" if not np.isnan(iou_t) else "  N/A"

        if not np.isnan(iou_a) and not np.isnan(iou_t):
            delta = (iou_t - iou_a) * 100
            s_d   = f"{delta:+.1f}%"
        else:
            s_d = "   N/A"

        marker = " <- BEAM (key class)" if name == "beam" else ""
        print(f"  {name:<12}  {s_a:>10}  {s_t:>10}  {s_d:>8}{marker}")

    print("-" * 70)
    print(f"  {'mIoU':<12}  {miou_no_adapt*100:>9.1f}%  {miou_tent*100:>9.1f}%  "
          f"{(miou_tent - miou_no_adapt)*100:>+7.1f}%")
    print("=" * 70)
    print()
    print("  PAPER NARRATIVE:")
    print("  " + "-" * 65)
    beam_idx = CLASS_NAMES.index('beam')
    if not np.isnan(iou_no_adapt[beam_idx]) and not np.isnan(iou_tent[beam_idx]):
        beam_delta = (iou_tent[beam_idx] - iou_no_adapt[beam_idx]) * 100
        if beam_delta <= 0:
            verb = "degrades" if beam_delta < 0 else "does not improve"
            print(f"  TENT {verb} beam IoU ({beam_delta:+.1f}%), confirming that entropy")
            print("  minimization is dominated by majority classes. This motivates")
            print("  HINT3D++'s human-governed exemplar approach, which explicitly")
            print("  targets rare classes through permission field signals.")
        else:
            print(f"  TENT improves beam by {beam_delta:+.1f}% — consider a different")
            print("  narrative angle or check if improvement is statistically significant.")
    print("=" * 70)


# ================================================================== #
#  BEAM CONFIDENCE TABLE                                              #
# ================================================================== #

def print_beam_confidence_table(beam_log, top_k=10):
    """Print per-room and aggregate beam-class confidence before/after TENT.

    Two columns matter:
      P(beam | gt=beam)   — mean softmax prob assigned to class beam on
                            voxels whose ground-truth label is beam.
                            Up = model is more confident the rare class
                            is itself. Down = TENT is suppressing it.

      P(beam | all)       — mean softmax prob assigned to class beam over
                            ALL voxels in the room. Up without P(beam|gt)
                            also going up means TENT is spreading beam
                            mass to non-beam points (false positives).
    """
    if not beam_log:
        return

    rooms_with_beam = [r for r in beam_log if r["n_beam_voxels"] > 0]

    print()
    print("=" * 95)
    print("  BEAM CONFIDENCE TRACKING — class index 3, before vs after TENT step")
    print("=" * 95)
    header = (f"  {'Room':<25}  {'#vox':>8} {'#beam':>7}  "
              f"{'P(beam|gt=beam)':>22}   {'P(beam|all)':>20}")
    print(header)
    print(f"  {'':<25}  {'':>8} {'':>7}  "
          f"{'before    after    Δ':>22}   {'before   after    Δ':>20}")
    print("-" * 95)

    # Show first top_k and last top_k rooms (so the user can read order effects)
    n = len(beam_log)
    if n <= 2 * top_k:
        rows_to_show = beam_log
    else:
        rows_to_show = beam_log[:top_k] + [None] + beam_log[-top_k:]

    for r in rows_to_show:
        if r is None:
            print(f"  {'...':<25}  {'...':>8} {'...':>7}  "
                  f"{'...':>22}   {'...':>20}")
            continue
        # P(beam | gt=beam) may be NaN for rooms with no beam
        if not np.isnan(r["p_beam_on_beam_b"]):
            d_gt = r["p_beam_on_beam_a"] - r["p_beam_on_beam_b"]
            s_gt = (f"{r['p_beam_on_beam_b']:.4f}  {r['p_beam_on_beam_a']:.4f} "
                    f"{d_gt:+.4f}")
        else:
            s_gt = f"{'  no beam in room':>22}"
        d_all = r["p_beam_on_all_a"] - r["p_beam_on_all_b"]
        s_all = (f"{r['p_beam_on_all_b']:.4f} {r['p_beam_on_all_a']:.4f} "
                 f"{d_all:+.4f}")
        print(f"  {r['room']:<25}  {r['n_voxels']:>8} {r['n_beam_voxels']:>7}  "
              f"{s_gt:>22}   {s_all:>20}")

    print("-" * 95)

    # ---- Aggregates ----
    # Macro: mean over rooms (each room weighted equally)
    macro_b_gt  = np.nanmean([r["p_beam_on_beam_b"] for r in beam_log])
    macro_a_gt  = np.nanmean([r["p_beam_on_beam_a"] for r in beam_log])
    macro_b_all = float(np.mean([r["p_beam_on_all_b"]  for r in beam_log]))
    macro_a_all = float(np.mean([r["p_beam_on_all_a"]  for r in beam_log]))

    # Micro: weighted by voxel count
    w_total       = np.array([r["n_voxels"]      for r in beam_log], dtype=np.float64)
    w_beam        = np.array([r["n_beam_voxels"] for r in beam_log], dtype=np.float64)
    micro_b_all   = float(np.sum([r["p_beam_on_all_b"]  * r["n_voxels"]      for r in beam_log])
                          / max(w_total.sum(), 1.0))
    micro_a_all   = float(np.sum([r["p_beam_on_all_a"]  * r["n_voxels"]      for r in beam_log])
                          / max(w_total.sum(), 1.0))
    if w_beam.sum() > 0:
        micro_b_gt = float(np.nansum(
            [(r["p_beam_on_beam_b"] if not np.isnan(r["p_beam_on_beam_b"]) else 0)
             * r["n_beam_voxels"] for r in beam_log]) / w_beam.sum())
        micro_a_gt = float(np.nansum(
            [(r["p_beam_on_beam_a"] if not np.isnan(r["p_beam_on_beam_a"]) else 0)
             * r["n_beam_voxels"] for r in beam_log]) / w_beam.sum())
    else:
        micro_b_gt = micro_a_gt = float("nan")

    print(f"  {'AGG (macro, n=' + str(len(beam_log)) + ' rooms)':<25}  "
          f"{'':>8} {'':>7}  "
          f"{macro_b_gt:.4f}  {macro_a_gt:.4f} {macro_a_gt-macro_b_gt:+.4f}   "
          f"{macro_b_all:.4f} {macro_a_all:.4f} {macro_a_all-macro_b_all:+.4f}")
    print(f"  {'AGG (micro, voxel-wt)':<25}  "
          f"{int(w_total.sum()):>8} {int(w_beam.sum()):>7}  "
          f"{micro_b_gt:.4f}  {micro_a_gt:.4f} {micro_a_gt-micro_b_gt:+.4f}   "
          f"{micro_b_all:.4f} {micro_a_all:.4f} {micro_a_all-micro_b_all:+.4f}")

    # ---- Per-room direction count: how many rooms moved up vs down? ----
    up_gt = sum(
        1 for r in rooms_with_beam
        if r["p_beam_on_beam_a"] > r["p_beam_on_beam_b"]
    )
    dn_gt = sum(
        1 for r in rooms_with_beam
        if r["p_beam_on_beam_a"] < r["p_beam_on_beam_b"]
    )
    up_all = sum(1 for r in beam_log if r["p_beam_on_all_a"] > r["p_beam_on_all_b"])
    dn_all = sum(1 for r in beam_log if r["p_beam_on_all_a"] < r["p_beam_on_all_b"])

    print("-" * 95)
    print(f"  Rooms with beam (n={len(rooms_with_beam)}): "
          f"P(beam|gt=beam) increased in {up_gt}, decreased in {dn_gt}")
    print(f"  All rooms       (n={len(beam_log)}): "
          f"P(beam|all)     increased in {up_all}, decreased in {dn_all}")

    # ---- Interpretation hint ----
    print()
    print("  INTERPRETATION:")
    print("  " + "-" * 65)
    delta_gt_micro  = micro_a_gt  - micro_b_gt  if not np.isnan(micro_b_gt) else 0
    delta_all_micro = micro_a_all - micro_b_all
    if delta_gt_micro < 0 and delta_all_micro >= 0:
        print("  TENT REDUCES confidence on actual beam voxels while leaving")
        print("  the all-points beam mass roughly unchanged. The model is")
        print("  becoming LESS sure of beam at the points where it should be sure.")
    elif delta_gt_micro < 0 and delta_all_micro < 0:
        print("  TENT suppresses the beam class everywhere — both on real")
        print("  beam voxels and overall. Entropy minimization is pushing")
        print("  predictions toward majority classes.")
    elif delta_gt_micro >= 0 and delta_all_micro > 0 and delta_all_micro > delta_gt_micro:
        print("  P(beam|all) rises faster than P(beam|gt=beam) — TENT is")
        print("  spreading beam probability to non-beam points (false positives).")
    elif delta_gt_micro >= 0:
        print(f"  P(beam|gt=beam) moved {delta_gt_micro:+.4f} — within typical")
        print("  noise; TENT is not meaningfully helping the rare class.")
    print("=" * 95)


# ================================================================== #
#  DATA LOADING — Pointcept val pipeline                              #
# ================================================================== #

def build_val_dataset(cfg, data_root, area):
    """Build an S3DISDataset that yields a single Area_X room per __getitem__,
    pre-processed with the val transforms from the config (GridSample,
    NormalizeColor, ToTensor, Collect with feat_keys=coord/color/normal)."""
    from pointcept.datasets import build_dataset

    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg["data_root"] = str(data_root)
    val_cfg["split"]     = f"Area_{area}"
    return build_dataset(val_cfg)


def to_device(data_dict, device):
    """Move tensors in dict to device. Leaves non-tensor values untouched."""
    out = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ================================================================== #
#  EVAL RUNNERS                                                       #
# ================================================================== #

def run_no_adapt(model, dataset, device):
    """Inference with NO adaptation."""
    print("\n[BASELINE] Running inference without adaptation...")
    acc = IoUAccumulator(NUM_CLASSES)

    model.eval()
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            inp    = to_device(sample, device)
            with autocast_ctx:
                out = model(inp)

            pred_voxel = out["seg_logits"].argmax(dim=-1).cpu().numpy()
            inverse    = sample["inverse"].cpu().numpy()
            origin_seg = sample["origin_segment"].cpu().numpy()
            pred_orig  = pred_voxel[inverse]

            acc.update(pred_orig, origin_seg)
            del out, inp
            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                print(f"  Room {i+1}/{len(dataset)}")

    return acc


def run_tent(model_original, dataset, device, lr=1e-5, online=True):
    """Run TENT adaptation.

    online=True   stateful: each room updates the model and passes
                  updated weights to the next room (realistic deployment)
    online=False  episodic: each room starts from the original checkpoint
                  (cleaner comparison, less dependent on room order)
    """
    print(f"\n[TENT] Running with lr={lr}, online={online}")
    acc         = IoUAccumulator(NUM_CLASSES)
    entropy_log = []
    beam_log    = []  # list of dicts: room_name + before/after beam stats

    def _record(i, room_name, b_before, b_after, H_before, H_after):
        beam_log.append({
            "idx":               i,
            "room":              room_name,
            "n_voxels":          b_before["n_voxels"],
            "n_beam_voxels":     b_before["n_beam_voxels"],
            "p_beam_on_beam_b":  b_before["p_beam_on_beam"],
            "p_beam_on_beam_a":  b_after["p_beam_on_beam"],
            "p_beam_on_all_b":   b_before["p_beam_on_all"],
            "p_beam_on_all_a":   b_after["p_beam_on_all"],
        })
        entropy_log.append((H_before, H_after))

    if online:
        model, norm_params = configure_model_for_tent(copy.deepcopy(model_original))
        optimizer = torch.optim.Adam(norm_params, lr=lr, betas=(0.9, 0.999))
        scaler    = torch.amp.GradScaler("cuda")

        for i in range(len(dataset)):
            sample = dataset[i]
            room_name = Path(dataset.data_list[i]).name
            inp    = to_device(sample, device)
            pred_voxel, H_before, H_after, b_before, b_after = tent_adapt_batch(
                model, inp, optimizer, scaler
            )

            inverse    = sample["inverse"].cpu().numpy()
            origin_seg = sample["origin_segment"].cpu().numpy()
            pred_orig  = pred_voxel[inverse]

            acc.update(pred_orig, origin_seg)
            _record(i, room_name, b_before, b_after, H_before, H_after)

            del inp
            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                print(f"  Room {i+1}/{len(dataset)}  "
                      f"H: {H_before:.4f} -> {H_after:.4f}  "
                      f"P(beam|gt=beam): {b_before['p_beam_on_beam']:.4f} -> "
                      f"{b_after['p_beam_on_beam']:.4f}")
    else:
        for i in range(len(dataset)):
            model, norm_params = configure_model_for_tent(copy.deepcopy(model_original))
            optimizer = torch.optim.Adam(norm_params, lr=lr, betas=(0.9, 0.999))
            scaler    = torch.amp.GradScaler("cuda")

            sample = dataset[i]
            room_name = Path(dataset.data_list[i]).name
            inp    = to_device(sample, device)
            pred_voxel, H_before, H_after, b_before, b_after = tent_adapt_batch(
                model, inp, optimizer, scaler
            )

            inverse    = sample["inverse"].cpu().numpy()
            origin_seg = sample["origin_segment"].cpu().numpy()
            pred_orig  = pred_voxel[inverse]

            acc.update(pred_orig, origin_seg)
            _record(i, room_name, b_before, b_after, H_before, H_after)

            del model, optimizer, norm_params, scaler, inp
            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                print(f"  Room {i+1}/{len(dataset)}  "
                      f"H: {H_before:.4f} -> {H_after:.4f}  "
                      f"P(beam|gt=beam): {b_before['p_beam_on_beam']:.4f} -> "
                      f"{b_after['p_beam_on_beam']:.4f}")

    avg_H_before = float(np.mean([h[0] for h in entropy_log]))
    avg_H_after  = float(np.mean([h[1] for h in entropy_log]))
    print(f"\n[TENT] Avg entropy: {avg_H_before:.4f} -> {avg_H_after:.4f} "
          f"(Δ = {avg_H_after - avg_H_before:+.4f})")

    print_beam_confidence_table(beam_log)

    return acc, beam_log


# ================================================================== #
#  MAIN                                                               #
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(description="TENT baseline for HINT3D++")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--data_root",  required=True,
                        help="Path to s3dis-compressed root directory "
                             "(parent of Area_1/.../Area_6)")
    parser.add_argument("--config",     required=True,
                        help="Path to Pointcept config .py file")
    parser.add_argument("--area",       type=int, default=5,
                        help="Test area (default: 5)")
    parser.add_argument("--lr",         type=float, default=1e-5,
                        help="TENT learning rate (default: 1e-5)")
    parser.add_argument("--online",     action="store_true",
                        help="Online mode: model accumulates across rooms")
    parser.add_argument("--max_rooms",  type=int, default=None,
                        help="Limit number of rooms (for quick testing)")
    parser.add_argument("--output",     type=str,
                        default="experiments/results/tent_baseline/results.npy",
                        help="Where to save the results .npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Build model from config ----
    from pointcept.utils.config import Config
    from pointcept.models import build_model

    print(f"\n[INFO] Loading config: {args.config}")
    cfg = Config.fromfile(args.config)

    print(f"[INFO] Loading model from {args.checkpoint}")
    model = build_model(cfg.model).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = (state.get("state_dict") if isinstance(state, dict) else None) \
                 or (state.get("model")   if isinstance(state, dict) else None) \
                 or state

    # Strip "module." prefix if present (DDP checkpoints)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    if not missing and not unexpected:
        print("[INFO] Checkpoint loaded with full key match.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total params: {total_params:,}")

    # ---- Build val dataset ----
    print(f"\n[INFO] Building val dataset for Area_{args.area}")
    dataset = build_val_dataset(cfg, args.data_root, args.area)

    if args.max_rooms is not None:
        # Truncate the data_list in place
        dataset.data_list = dataset.data_list[: args.max_rooms]

    print(f"[INFO] Loaded {len(dataset)} rooms")

    # ---- Run baseline (no adaptation) ----
    acc_base = run_no_adapt(model, dataset, device)
    iou_base, miou_base = acc_base.compute()

    # ---- Run TENT ----
    acc_tent, beam_log = run_tent(
        model_original = model,
        dataset        = dataset,
        device         = device,
        lr             = args.lr,
        online         = args.online,
    )
    iou_tent, miou_tent = acc_tent.compute()

    # ---- Print results ----
    print_results_table(iou_base, miou_base, iou_tent, miou_tent)

    # ---- Save ----
    results = {
        "no_adapt": {
            "miou":      miou_base,
            "per_class": {c: iou_base[i] for i, c in enumerate(CLASS_NAMES)},
        },
        "tent": {
            "miou":      miou_tent,
            "lr":        args.lr,
            "online":    args.online,
            "per_class": {c: iou_tent[i] for i, c in enumerate(CLASS_NAMES)},
        },
        "beam_log": beam_log,
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), results)
    print(f"\n[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()
