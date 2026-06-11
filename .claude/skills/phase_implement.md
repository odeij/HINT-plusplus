---
name: phase-implement
description: Implement a specific phase of the HINT++ seven-phase architecture. Use when creating new modules, writing phase code, or integrating phases together. Triggers on mentions of Phase 2-7, adaptive moments, permission field, monotone check, exemplar memory, or integration.
---

# HINT++ Phase Implementation

When implementing a HINT++ phase, follow this protocol strictly.

## Project status snapshot (keep this fresh)

- ✅ **Phase 1 — Frozen Teacher.** PTv3-Sonata trained on S3DIS Areas 1-4,6. Area-5 test mIoU **75.41%**. Checkpoint: `checkpoints/model_best.pth` (epoch 29, best_metric_value 0.7419). DO NOT modify.
- ✅ **Phase 2 — Adaptive Moment Safety Signals.** η_k buffer, v_k initialization, master CSV in `experiments/phase2_init/results/`. 19 tests pass.
- ✅ **Cross-Domain Validation (S3DIS → ScanNet zero-shot).** 312 scenes, **42.03% mIoU, gap −33.38 pp**. Conf > 0.7 threshold cannot triage failure. Four classes hit the beam pattern (ceiling, beam, column, board), plus a second tier of graded failures (sofa, bookcase). See `experiments/cross_domain/SCANNET_ZERO_SHOT_FINDINGS.md`.
- ⬅ **Phase 3 — Permission Field P(x)** is the next module to implement.
- ⬜ Phases 4–7 not started.

## Pre-Implementation Checklist

1. Confirm which phase number (2-7) is being implemented
2. Verify entry criteria from the previous phase are met (check tests pass)
3. Read existing code in `src/` to understand current interfaces
4. Identify which files will be created or modified
5. Re-read `experiments/cross_domain/SCANNET_ZERO_SHOT_FINDINGS.md` if the phase touches safety / permission / threshold logic — the cross-domain failure modes are the design target.

## Implementation Standards

All modules must:
- Inherit from `nn.Module`
- Include type hints on all function signatures
- Include docstrings with the Adam analogy correspondence where relevant
- Add numerical stability guards: `assert eps > 0`, `torch.isnan` checks
- Never use global state — all state lives in module attributes
- Follow the file structure: `src/{category}/{module_name}.py`

## Phase-Specific Guidance

### Phase 2 — Adaptive Moment Safety Signals
- Module: `src/safety/adaptive_moments.py`
- Class: `AdaptiveMomentSafety(nn.Module)`
- β₁ and β₂ are ASYMMETRIC by design (β₁ < β₂). Do not use Adam defaults.
- Bias correction is mandatory — early adaptation safety depends on it.
- Initialize moments from frozen teacher confidence, not zeros.
- Correspondence: δₖ(t)→g, m̂ₖ→consistency, v̂ₖ→noise, weight=η·m̂ₖ/(√v̂ₖ+ε)

### Phase 3 — Permission Field P(x)
- Module: `src/safety/permission_field.py`
- P(x) must output values in [0, 1] — use sigmoid-based gating
- Must be differentiable for gradient flow during meta-learning
- Spatial: operates on per-point features from the frozen teacher

**Cross-domain failure modes the permission field MUST absorb** (evidence: `experiments/cross_domain/SCANNET_ZERO_SHOT_FINDINGS.md`):
- **Be class-specific.** A single scalar gate fails: on ScanNet every wrong class still has mean conf > 0.79, so a global conf threshold (HINT-3D's conf > 0.7) filters nothing. P must be P_k(x) — per class k, per point x.
- **Allow a whole class to be gated off.** ceiling/beam/column/board have no analog on ScanNet; the model predicts them at conf 0.80–0.95 anyway. P_k(x) must be able to collapse to ≈ 0 across an entire class even when softmax is high.
- **Do not penalize reliable classes.** floor/wall/chair transfer well (IoU 0.94/0.75/0.78). P_k(x) must stay ≈ 1 for them — the field discriminates, it does not blanket-suppress.
- **Handle graded failure.** sofa/bookcase land at IoU 0.32–0.49 with conf ~0.9 — neither safe nor catastrophic. P_k(x) output must be continuous, not a binary gate.
- **Use η_k as the prior.** The Phase 2 per-class confidence ceiling η_k is the signal that separates "reliable" from "dangerous" before any correction arrives. Condition the permission field on η_k, not on raw softmax alone.

### Phase 4 — Monotone Safety Check
- Module: `src/safety/monotone_check.py`
- Property: P(x, t+1) ≥ P(x, t) when safety evidence increases
- Implement as a wrapper/constraint on the permission field
- Include formal property tests, not just value checks

### Phase 5 — Exemplar Memory
- Module: `src/memory/exemplar_memory.py`
- Store M sufficient statistics (mean, variance, count) per correction
- One exemplar per correction event — NOT full point cloud tensors
- Implement sampling strategies: recent_window, class_balanced, hard_negative, domain_stratified
- Enforce memory budget with eviction policy (priority-based)

### Phase 6 — Full Integration
- Module: `src/adaptation/hint_pp.py`
- Composes all phases into a single `HINTPlusPlus(nn.Module)`
- Meta-learning loop: outer loop over domains, inner loop over corrections
- Forward pass: frozen_teacher → adaptive_moments → permission_field → monotone_check → adapted_prediction

## Post-Implementation

1. Write tests in `tests/test_{module_name}.py` using synthetic point cloud data
2. Run: `pytest tests/test_{module_name}.py -v`
3. All tests must pass before committing
4. Tag the commit: `git tag phase-{N}-complete`
5. Update CLAUDE.md if any interfaces changed
6. **Update the "Project status snapshot" at the top of this file** — flip the phase marker to ✅ so the next session starts with accurate state. Forgetting this is how target goals get lost across sessions.
