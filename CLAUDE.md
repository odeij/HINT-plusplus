# HINT++ — Interactive Safe Test-Time Adaptation for 3D Segmentation

## Project
CVPR 2026 submission. First author: Odei. Lab: AUB Vision and Robotics Lab.
Supervisors: Dr. Daniel Asmar, Dr. Imad El-Hajj.
Predecessor: HINT-3D (accepted ICRA 2026).

Core contribution: "We enable zero-shot safe deployment of interactive TTA to unknown domains by meta-learning transferable safety patterns from human correction history."

## Tech Stack
- PyTorch, PointTransformer v3, Sonata (frozen teacher, trained on S3DIS)
- Evaluation: ScanNet, SemanticKITTI, nuScenes — all zero-shot, NO target tuning
- Experiment tracking: Weights & Biases
- Config management: Hydra
- Tests: pytest

## Architecture — Seven Phases
1. **Frozen Teacher** — PointTransformer v3 + Sonata on S3DIS ✅ COMPLETE
   Area 5 test mIoU **75.41%** (best val during training: 74.19% — do not confuse the two)
2. **Adaptive Moment Safety Signals** — Adam-style per-class safety
3. **Permission Field P(x)** — spatial gating for safe adaptation
4. **Monotone Safety Check** — safety never regresses
5. **Exemplar Memory** — M sufficient statistics, one per correction
6. **Full Integration** — meta-learning loop across domains
7. **CVPR Evaluation** — zero-shot deployment experiment

## Key Equations
- δₖ(t): correction signal for class k at time t
- m̂ₖ = β₁·mₖ + (1-β₁)·δₖ, bias-corrected: m̂ₖ/(1-β₁^t)
- v̂ₖ = β₂·vₖ + (1-β₂)·δₖ², bias-corrected: v̂ₖ/(1-β₂^t)
- ηₖ: per-class teacher-confidence ceiling, computed in `experiments/phase2_init/`, loaded as a buffer in `src/safety/adaptive_moments.py`
- Adaptive safety weight = η · ηₖ · m̂ₖ / (√v̂ₖ + ε)   (η is a global rate scalar; ηₖ is per-class)
- β₁ < β₂ (ASYMMETRIC by design — safety needs different smoothing)

## Coding Conventions
- All stateful components: `nn.Module`
- Type hints on all function signatures
- Docstrings on all public methods
- No global state. No hardcoded hyperparameters (use Hydra configs).
- Safety-critical code gets numerical stability assertions
- One module per phase, one test file per module

## Directory Structure
```
src/models/       — PointTransformer v3, Sonata
src/safety/       — Adaptive moments, permission field, monotone check
src/memory/       — Exemplar memory
src/adaptation/   — TTA loop, meta-learning, full HINT++ pipeline
src/utils/        — Data loading, metrics, visualization
experiments/      — configs/, scripts/, results/
paper/            — sections/, figures/, tables/
tests/            — Unit + integration tests per phase
```

## Critical Rules
- NEVER reference DynaCITY, Beirut, heritage, or funding in code, comments, or paper
- Run `pytest tests/` before every commit
- Safety test failures are BLOCKING — fix before any other work
- Experiment results go to `experiments/results/{experiment_id}/`
- All experiments: ≥3 seeds, report mean ± std

## Known Open Issues

Surfaced during research validation; tracked here so they are not forgotten as later phases land. See `experiments/phase2_init/RESEARCH_VALIDATION.md` for the full analysis.

- **Phase 2 sparse correction regime — CV > 1 for sofa, beam, column, window, board in 100-step synthetic experiments.** Meta-learning in Phase 6 requires either longer adaptation horizon or variance regularizer for sparse classes. Not a Phase 3 blocker.
- **Phase 4 running-max projection requires stable per-point identity across calls.** Voxelization strategy must preserve point identity or Phase 4 must define a spatial hashing scheme. Must be resolved in Phase 4 design session.

## Success Criteria (Phase 7)
- Violations <15% on unseen domains WITHOUT tuning
- HINT-3D source thresholds show >40% violations (establishes problem)
- Within 3% mIoU of exhaustive per-domain tuning
- Eliminates 2+ weeks of manual threshold tuning per domain

## Skills (.claude/skills/)

These skills exist and should be applied automatically based on context:

| Skill | File | When to use |
|---|---|---|
| `coding-discipline` | `coding_discipline.md` | Before writing or changing any code in this repo — think, simplify, stay surgical |
| `phase-implement` | `phase_implement.md` | Starting a new HINT++ phase module (Phase 2–7) |
| `safety-check` | `safety_check.md` | After any change to `src/safety/` or moment/permission/monotone code — runs at Opus high effort |
| `debug-phase` | `debugging_protocol.md` | NaN, divergence, failing test, unexpected result |
| `experiment-run` | `experiment_run.md` | Configuring or launching experiments, ablations, the primary zero-shot eval |
| `paper-section` | `paper_section.md` | Drafting `.tex` sections for the CVPR submission |

You can also invoke any skill explicitly: `/coding-discipline`, `/phase-implement`, `/safety-check`, `/debug-phase`, `/experiment-run`, `/paper-section`.
