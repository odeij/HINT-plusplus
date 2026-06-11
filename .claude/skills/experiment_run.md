---
name: experiment-run
description: Configure, launch, and analyze HINT++ experiments. Use when setting up evaluations, ablation studies, hyperparameter sweeps, or the primary zero-shot deployment experiment. Triggers on mentions of experiment, ablation, evaluation, benchmark, ScanNet, SemanticKITTI, nuScenes.
---

# HINT++ Experiment Runner

## Completed milestones (keep this fresh)

| Experiment | Status | Headline result | Artefacts |
|---|---|---|---|
| Phase 1 — Frozen Teacher (S3DIS Area-5 test) | ✅ | mIoU **75.41%** | `checkpoints/model_best.pth`, `experiments/phase1_baseline/results/per_class_iou.json` |
| Phase 2 — Adaptive Moment init | ✅ | η_k, v_k buffers computed | `experiments/phase2_init/results/` |
| Cross-domain S3DIS → ScanNet zero-shot | ✅ | mIoU **42.03%** (gap −33.38 pp); conf>0.7 triages nothing | `experiments/cross_domain/` |

**Reusable cross-domain assets** (for future ablations / SemanticKITTI + nuScenes runs):
- Inference: `experiments/cross_domain/scripts/run_scannet_zero_shot.py` — loads Sonata S3DIS config, swaps the test dataset, runs Phase-1 TTA protocol, saves per-scene `_pred.npy` + `_prob.npy`.
- Analysis: `experiments/cross_domain/scripts/run_scannet_analysis.py` — per-class freq / conf / η_k / IoU with the ScanNet→S3DIS label map.
- Figures: `experiments/cross_domain/scripts/make_figures.py`.
- Run with `/home/ahmad/anaconda3/envs/frozen_teacher/bin/python` (has torch_scatter/spconv/flash-attn; Pointcept imported via `sys.path`, not pip-installed).
- ScanNet data: `/media/ahmad/One Touch/HINT++/data/scannet/{raw,processed}/` (external drive — main disk is full). Large prediction outputs are symlinked to the external drive.
- The same script pattern extends to SemanticKITTI and nuScenes for the Phase 7 zero-shot eval — swap the dataset type and the label map.

## Experiment Setup Protocol

1. Create Hydra config: `experiments/configs/{exp_name}.yaml`
2. Create run script: `experiments/scripts/run_{exp_name}.sh`
3. Results auto-save to: `experiments/results/{exp_id}/`
4. Log everything to Weights & Biases

## Primary Experiment: Zero-Shot Deployment

This is the most important experiment. Run it FIRST before any ablations.

**Setup:**
- Train HINT++ meta-learning loop on S3DIS with simulated human corrections
- Deploy to ScanNet, SemanticKITTI, nuScenes with ZERO target-domain tuning
- Compare against baselines

**Baselines (must include all three):**

| Method | Expected Violations | Tuning Required |
|--------|-------------------|-----------------|
| HINT-3D source thresholds | ~45-50% | None (but unsafe) |
| HINT-3D tuned per domain | ~18-20% | ~2 weeks per domain |
| HINT++ (ours) | Target <15% | None |

**Supporting evidence already collected (cross-domain validation):** On ScanNet zero-shot, the frozen teacher drops to 42.03% mIoU (−33.38 pp) and the HINT-3D global confidence threshold (conf > 0.7) fails to flag a single failure class — every wrong class has mean conf > 0.79. This pre-confirms the "HINT-3D source thresholds are unsafe" baseline qualitatively. The Phase 7 experiment must still quantify it as a *violation rate*, but the mechanism is already demonstrated.

**Success criteria (all must be YES):**
- [ ] Violations <15% on all three target domains without tuning
- [ ] HINT-3D source violations >40% (establishes the problem exists)
- [ ] Within 3% mIoU of exhaustive tuning
- [ ] Eliminates 2+ weeks of manual threshold tuning per domain

## Ablation Studies

Only run these AFTER the primary experiment succeeds.

1. **Moment components:** first-only vs second-only vs both
2. **Bias correction:** with vs without
3. **Asymmetric β:** symmetric(0.9,0.999) vs asymmetric(0.7,0.95) vs other combos
4. **Exemplar sampling:** recent vs balanced vs hard_neg vs domain_stratified
5. **Memory budget:** 100 vs 500 vs 1000 vs 5000 exemplars
6. **Initialization:** random vs frozen_teacher_confidence
7. **Monotonicity constraint:** with vs without

## Reporting Standards

Every experiment result must include:
- Violation rate (%) — primary metric
- mIoU (%) — accuracy metric
- Per-class IoU breakdown
- Wall-clock adaptation time (seconds)
- Memory footprint (MB)
- Mean ± std over ≥3 seeds (5 preferred)

## Config Template

```yaml
# experiments/configs/template.yaml
experiment:
  name: ${exp_name}
  seed: [42, 123, 456]
  
model:
  backbone: pointtransformer_v3
  pretrained: checkpoints/frozen_teacher_s3dis.pth
  
safety:
  beta1: 0.7
  beta2: 0.95
  eps: 1e-8
  eta: 1.0

memory:
  max_exemplars: 1000
  sampling: class_balanced
  
data:
  train: s3dis
  eval: [scannet, semantickitti, nuscenes]

logging:
  wandb_project: hint-plus-plus
  save_dir: experiments/results/
```

## Post-Experiment

1. Generate comparison table (markdown) from W&B data
2. Generate figures: violation rate bar chart, mIoU comparison, per-class heatmap
3. Save figures to `paper/figures/` with source scripts
4. Flag any result that violates success criteria as BLOCKING
