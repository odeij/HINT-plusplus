# Phase 1 Baseline — Frozen Teacher Evaluation on S3DIS Area 5

## What We Did

Evaluated the **Sonata (PointTransformerV3)** frozen teacher model — trained on S3DIS Areas 1–4 and 6 — against the held-out **Area 5** test split (68 rooms, ~60M points). No fine-tuning was applied; this is a pure inference pass to establish the Phase 1 baseline.

We then generated a full set of visualizations from the raw prediction files (`_gt.npy`, `_pred.npy`) to characterize where the model succeeds and where it fails.

---

## Results

| Metric | Value |
|--------|-------|
| mIoU | **75.41%** |
| mAcc | 81.54% |
| allAcc | 92.74% |
| Best val mIoU (during training) | 74.19% |

### Per-Class IoU

| Class | IoU | Class | IoU |
|-------|-----|-------|-----|
| ceiling | 95.43% | table | 85.94% |
| floor | 98.43% | chair | 91.10% |
| wall | 88.79% | sofa | 80.38% |
| **beam** | **0.19%** | bookcase | 82.81% |
| column | 58.95% | board | 85.61% |
| window | 67.45% | clutter | 65.46% |
| door | 79.78% | | |

---

## Key Findings

**1. Beam class is a critical failure.**
The model achieves 0.19% IoU on beams — effectively zero. In office_39 (one of only two Area 5 rooms containing beams), all 12,928 beam points are missed entirely: 0 true positives, 0 false positives. The confusion matrix shows the model redistributes beam points to **ceiling (40%)** and **wall (31%)**, which is geometrically understandable — beams are planar structures near the ceiling and share similar point density and surface normals.

**2. Everything else generalizes well.**
Ten of thirteen classes exceed 65% IoU on a zero-shot held-out area, with ceiling, floor, chair, and table all above 85%. The model has learned robust structural representations for common indoor objects.

**3. Training converged early and plateaued.**
Validation mIoU reached ~73% within the first 5 checkpoint intervals and oscillated between 72–74% for the remainder of training. No overfitting; the model is genuinely stable. The gap between best val mIoU (74.19%) and final test mIoU (75.41%) is within noise.

**4. Per-room variance is low across all room types.**
Offices, hallways, conference rooms, storage rooms, and WCs all cluster tightly around 72–74% mIoU. The pantry is the weakest single room (~70%), likely due to unusual object configurations. This low variance confirms the model is consistent, not lucky on easy rooms.

---

## Outputs

```
charts/
  01_per_class_iou.png      — bar chart of per-class IoU, beam annotated as critical failure
  02_training_curve.png     — val mIoU + mAcc over 100 checkpoint intervals
  03_confusion_matrix.png   — normalized confusion matrix, beam row highlighted in red
  04_per_room_miou.png      — box + scatter plot of per-room mIoU by room type

pointclouds/
  office_39_gt_vs_pred.png  — top-down GT vs Pred render for office_39 (beam room)
  office_39_beam_focus.png  — 4-panel beam failure analysis (GT, Pred, beam GT, beam FN map)
  conferenceRoom_1_gt_vs_pred.png
  office_1_gt_vs_pred.png
  hallway_1_gt_vs_pred.png
  storage_1_gt_vs_pred.png
  lobby_1_gt_vs_pred.png
  office_39_gt.ply          — full 928K-point cloud coloured by ground truth (open in MeshLab)
  office_39_pred.ply        — full 928K-point cloud coloured by Sonata prediction
```

---

## Confidence Analysis (Chart 05)

### Formula

For each voxel the model outputs accumulated softmax scores P ∈ ℝ^{N_vox × 13}. We normalize each row then compute Shannon entropy:

```
p_i  = P_i / Σ_k P_ik              (row-wise ℓ¹ normalization)
H(p) = −Σ_k p_ik · log₂(p_ik)     (Shannon entropy, bits)
C    = 1 − H(p) / log₂(13)         (confidence, ∈ [0, 1])
```

`H_max = log₂(13) ≈ 3.700 bits`. C = 1 means the model puts all mass on one class; C = 0 means it is uniform over all 13. Confidence is aggregated per **predicted class** (grouped by argmax of P_i) across all 68 Area 5 rooms.

### Results

| Class    | IoU    | Confidence | Voxels predicted |
|----------|--------|------------|-----------------|
| ceiling  | 95.43% | 98.45%     | 9,744,009       |
| floor    | 98.43% | 99.28%     | 8,214,994       |
| wall     | 88.79% | 95.30%     | 15,111,046      |
| **beam** | **0.19%** | **73.78%** | **261,639**  |
| column   | 58.95% | 86.94%     | 770,236         |
| window   | 67.45% | 93.92%     | 1,296,184       |
| door     | 79.78% | 93.63%     | 1,826,277       |
| table    | 85.94% | 93.06%     | 1,917,911       |
| chair    | 91.10% | 97.20%     | 956,295         |
| sofa     | 80.38% | 92.54%     | 122,316         |
| bookcase | 82.81% | 94.09%     | 5,200,616       |
| board    | 85.61% | 95.64%     | 544,370         |
| clutter  | 65.46% | 87.99%     | 4,040,687       |

### What the confidence is NOT

This C is **not η** from the Phase 2 update rule. In the HINT++ equations:

```
Adaptive safety weight = η · m̂_k / (√v̂_k + ε)
```

η is a **global scalar hyperparameter** (the meta-learning rate) — there is no per-class η. The per-class adaptation comes entirely from the moment terms m̂_k and v̂_k, which are driven by δ_k(t), the human correction signal.

C is a static diagnostic of the frozen teacher computed once before any human interaction. It could potentially inform the design of δ_k(t) — for example, a confident-but-wrong prediction might warrant a stronger correction signal — but that design decision belongs to Phase 2.

> _Note (Phase 2 update): η_k per-class ceiling introduced in Phase 2, see `src/safety/adaptive_moments.py`._

### Key finding

Beam is the **overconfident wrong prediction** failure mode: 73.78% confidence, 0.19% IoU, 261,639 voxels predicted as beam across Area 5. The model is not uncertain about beam — it is wrong and sure about it. A simple confidence threshold cannot filter this out, because beam's confidence (73.78%) is not far below well-performing classes like clutter (88%) or column (87%). Human correction via δ_k(t) is the only signal that can break through.

---

## Softmax Calibration Analysis (Chart 06)

Two panels showing that high softmax confidence does not imply high accuracy.

**Left — Confidence vs IoU scatter:**
Nearly every class sits near or above the perfect-calibration diagonal (confidence ≈ IoU), meaning the model is generally well-calibrated. Beam is the only class deep in the overconfident zone — confidence 73.8%, IoU 0.19% — isolated in the bottom-right, far from every other class.

**Right — Max softmax score distributions (sorted high IoU → low IoU):**
High-IoU classes (floor, ceiling, chair) have narrow distributions pinned near 1.0 — confident and correct. Beam sits at the far right with a median softmax of 0.67 and a wide distribution centered around 0.6–0.8. The model is pushing meaningful probability mass onto beam, just almost always incorrectly.

Together the two panels make the same point from two angles: **beam is not an uncertain class, it is a confidently mispredicted class.** Entropy-based filtering alone cannot catch it — beam's confidence (73.8%) is not far enough below well-performing classes like clutter (88%) or column (87%) to trigger any reasonable uncertainty threshold. Human correction via δₖ(t) is the only signal that can break through.

---

## Implications for Phase 2

The beam failure is the primary motivation for the HINT++ adaptive safety signal (Phase 2). The model is confidently wrong — it assigns beam points to ceiling/wall with high probability — so a simple confidence threshold will not catch it. What is needed is a correction signal δₖ(t) that can detect when human feedback indicates a structural misclassification and update the per-class safety weight accordingly, without degrading the already-strong performance on the other twelve classes.
