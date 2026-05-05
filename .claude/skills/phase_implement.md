---
name: phase-implement
description: Implement a specific phase of the HINT++ seven-phase architecture. Use when creating new modules, writing phase code, or integrating phases together. Triggers on mentions of Phase 2-7, adaptive moments, permission field, monotone check, exemplar memory, or integration.
---

# HINT++ Phase Implementation

When implementing a HINT++ phase, follow this protocol strictly.

## Pre-Implementation Checklist

1. Confirm which phase number (2-7) is being implemented
2. Verify entry criteria from the previous phase are met (check tests pass)
3. Read existing code in `src/` to understand current interfaces
4. Identify which files will be created or modified

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
