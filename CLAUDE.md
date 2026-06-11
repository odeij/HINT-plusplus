# HINT++ — Safe and Persistent Human-Governed Test-Time Adaptation for 3D Semantic Segmentation

## Project
CVPR 2027 submission (deadline expected ~Nov 13, 2026 — verify on announcement; abstract ~1 week earlier).
First author: Odei. Lab: AUB Vision and Robotics Lab. Supervisors: Dr. Daniel Asmar, Dr. Imad El-Hajj.
Predecessor: HINT-3D (ICRA 2026). Design source of truth: `docs/HINTpp_Design_Memo_R1_2026-06-11.md`
(approved at G0; flaw diagnosis F1–F8 in §2). Deviations need a decision record in `docs/decisions/` BEFORE code.

**Contribution (exact sentence, never deviate):** "HINT++ is the first interactive TTA method in which
corrections maintain a longitudinal per-class trust state that spatially gates parameter updates, with
anytime-valid risk control — enabling safe deployment to unseen domains without per-domain tuning."

**Presentation model:** two loops, not seven phases. Inner loop (inherited from HINT-3D): click → region →
gated LoRA step (CE on region + KL on anchors). Outer loop (the contribution): correction outcome δ → trust
posterior → safety weight w → permission field + risk-controlled gate → governs the inner loop.
NOT a benchmark-dominance paper: every work item traces to safety-without-tuning, reliability-from-
imperfect-humans, or the risk certificate (memo §1) — anything else is scope creep.

## Tech Stack & Environment
- PyTorch, PointTransformer v3, Sonata. Hydra (wiring lands Phase 6; `configs/safety.yaml` — created with
  the Phase 2 R1 refactor — is the canonical hyperparameter source). Weights & Biases, pytest.
- Tests: `PYTHONPATH=. /home/ahmad/anaconda3/bin/python -m pytest tests/ -q` (base env has pytest + torch).
  GPU experiment scripts: `/home/ahmad/anaconda3/envs/frozen_teacher/bin/python` (spconv/flash-attn, NO pytest).
  ScanNet data + large predictions: external drive `/media/ahmad/One Touch/HINT++/data/` (symlinked).
- Teachers (frozen): S3DIS indoor ✅ (75.41% Area 5); outdoor PTv3 teacher pending DR-0002 (due 2026-06-19).
- Evaluation, two tracks, zero target tuning:
  - Indoor (primary): S3DIS→ScanNet, ScanNet→S3DIS (~8 shared classes; mapping table to be built
    and checked into `docs/`).
  - Outdoor (generality): Synth4D→SemanticKITTI, Synth4D→nuScenes (GIPSO/HGL protocol).

## Key Equations (R1 — supersedes all earlier versions; memo §3–§4)
- δₖ(t) ∈ {+1, −1}: signed OUTCOME of a correction event on class k, emitted after the gated update
  (+1 = local error on corrected region decreased; −1 = it did not, or re-corrected within T_rc).
- EMAs: mₖ = β₁mₖ + (1−β₁)δₖ; vₖ = β₂vₖ + (1−β₂)δₖ². β₁=0.7 < β₂=0.95, asserted in __init__.
- NO bias correction combined with informed priors. Prior pseudo-counts instead:
  λₖ = n₀/(n₀+Nₖ), Nₖ = cumulative per-class event count, n₀ = 5 (`configs/safety.yaml`: safety.n0)
  m̂ₖ = (1−λₖ)·m̃ₖ        (m̃, ṽ = zero-init bias-corrected internals; prior mean 0 ⇒ warmup damping)
  v̂ₖ = λₖ·vₖ(0) + (1−λₖ)·ṽₖ   (vₖ(0) = 0.5rₖ + 0.5uₖ from Sub-step 0B)
  wₖ = η·ηₖ·m̂ₖ/(√v̂ₖ+ε)   — SIGNED. Worked check: t=1, δ=+1, n₀=5, vₖ(0)=0.6 ⇒ w ≈ 0.204·η·ηₖ;
  cold start Nₖ=0 ⇒ λₖ=1 ⇒ wₖ=0 EXACTLY.
- Permission: P_raw,ₖ = σ(α·wₖ). Gate gₖ ∈ {CLOSED, OPEN} with hysteresis
  (open: P_raw>θ_hi=0.65 for c=2 consecutive events; close: P_raw<θ_lo=0.45 OR risk monitor trips).
  Gₖ = 1[OPEN]·max(0, 2P_raw−1); spatial G(x)=G_{ŷ(x)}; scales correction-loss gradient into LoRA.
- Risk monitor: anytime-valid confidence sequence on per-class harmful-update rate hₜ∈{0,1} (pooled
  fallback for rare classes); close gate when lower bound > α_risk (δ_conf=0.05). α_risk is the ONLY
  deployment-semantic knob.
- THE RUNNING MAX P_safe = max(P_safe, P_raw) IS RETIRED (flaw F2). Do not implement, test, or cite it —
  the term "monotone safety check" included.

## Formal claims
- Prop 1: zero corrections ⇒ output ≡ frozen teacher (zero-init LoRA). Verified numerically in tests.
- Prop 2: KL-stabilization budget bounds anchor drift (Pinsker: ‖p′−p‖₁ ≤ √(2KL)); per-round KL is logged.
- Thm 1: anytime-valid risk control — gates open only while harmful rate statistically consistent with ≤ α_risk.
Every paper claim must trace to a table, figure, or one of these statements.

## Status (2026-06-12)
- ✅ Phase 1 Frozen Teacher: `checkpoints/model_best.pth` (epoch 29), Area-5 test mIoU 75.41%. DO NOT modify.
- ✅ Phase 2 init artefacts: 0A ηₖ, 0B vₖ(0), 0C `phase2_init.pt` (frozen 2026-04-27, source-domain-only).
- 🔄 Phase 2 estimator `src/safety/adaptive_moments.py` (19 tests): pre-R1 on disk; the R1 refactor
  (λ-mixture, outcome δ, signed w) is the next commit, branch `feat/phase2-r1`.
- ✅ Cross-domain S3DIS→ScanNet zero-shot: 42.03% mIoU (gap −33.38 pp); ceiling/beam/column/board are
  overconfident-wrong (conf>0.78, IoU 0) — global conf>0.7 triages nothing. `experiments/cross_domain/`.
- ⬅ Phase 3 Permission Field next. ⬜ Phases 4–7, harness/, paper/ not started.

## Architecture — Phases (engineering schedule; paper presents two loops)
1. Frozen Teacher ✅  2. Adaptive Moment Safety Signals (R1 refactor)  3. Permission Field (signed w)
4. Risk-Controlled Permission Gate (hysteresis + monitor; renamed from "Monotone Safety Check")
5. Exemplar Memory (outcome events, recency-tempered replay)  6. Full Integration (zero-init rank-4 LoRA,
gated gradient, KL stabilization — unchanged)  7. Two-track CVPR evaluation (E1–E5, in that order).

## Critical Rules
- NEVER reference the project's grant program, sponsor organizations, host city, or application
  context in code, comments, commit messages, or paper text. (The four literal terms from the pre-R1
  Critical Rules are deliberately written nowhere in this repo; ask the supervisors if unsure.)
- `pytest tests/` before every commit. Safety test failures are BLOCKING (incl. gate-closure adversarial test).
- Every experiment runs through the eval harness (no bespoke loops). ≥3 seeds, mean ± std, matched click
  budgets; register every launch in `docs/experiments/registry.md`.
- No hand-edited numbers anywhere: tables and figures are generated by scripts from `experiments/results/`.
- Spec deviations require a decision record in `docs/decisions/` BEFORE code (see research-log skill).
- ScanNet/SemanticKITTI/nuScenes are zero-shot: never train, tune, or compute feedback statistics on them.
- Regions are training-free; learned maskers (AGILE3D/PinPoint3D/Point-SAM) never enter evaluation (F8).

## Success Criteria (Phase 7, R1 — metrics are model-agnostic, F6)
- Safety: lowest Regression Event Rate (% scenes with post-adapt mIoU < teacher − 1.0 pt) among all
  baselines at matched budget; worst-stream mIoU ≥ teacher − margin; Catastrophic Class Collapse
  (% streams with any shared class >10 IoU below teacher) ≈ 0; empirical harmful rate ≤ declared α_risk.
- Utility: shared-class mIoU within 3 pts of oracle-tuned HINT++ and ≥ HILTTA-selection at same budget;
  mIoU@k clicks, NoC@target, gain-per-click reported.
- Robustness: graceful degradation on noisy-oracle sweep p∈{0,10,20,30}% (E2); gate closes under
  adversarial burst (E3, BLOCKING).
- Cost: ≈0.5 s/round, <3% params updated, reported per method.

## Gates & Freezes (slips eat buffers, never gates)
G1 Jul 20 (gated vs ungated separation, real ScanNet mini-run) · G2 Aug 3 (primary table vs all families,
incl. HILTTA-selection — most dangerous baseline, runs FIRST) · outdoor descope checks Jul 24 / Aug 7 ·
new-experiment freeze Oct 17 · all-experiment freeze Oct 24 · submit ≥48 h early.

## Directory Map (✅ exists · ⬜ planned)
```
✅ src/safety/          adaptive_moments.py (Phase 2)      ⬜ src/{models,memory,adaptation,utils}/
✅ tests/               test_adaptive_moments.py           ⬜ harness/   runner, adapters, simulator
⬜ configs/             safety.yaml — lands with the Phase 2 R1 refactor   ⬜ paper/  sections|figures|tables
✅ experiments/         phase1_baseline/ phase2_init/ cross_domain/ results/   ⬜ experiments/configs/
✅ docs/                memo · decisions/ · experiments/registry.md · objections/ledger.md · reviews/ ·
                        LESSONS.md · changelog.md
✅ Pointcept/ flash-attention/   submodules — DO NOT EDIT  ✅ checkpoints/ s3dis-compressed/  data — DO NOT EDIT
```

## Skills (.claude/skills/) — applied automatically by context
| Skill | When |
|---|---|
| `coding-discipline` | Before writing or changing any code |
| `phase-implement` | Building/refactoring Phase 2–6 modules to the R1 spec |
| `safety-check` | After ANY change to src/safety/, the gate, estimator, or monitor — Opus, high effort |
| `debug-phase` | NaN, divergence, failing test, unexpected result |
| `eval-harness` | Building/altering the runner, adapters, simulator, metrics schema |
| `baseline-implement` | Wrapping any baseline behind the MethodAdapter interface |
| `experiment-run` | Configuring/launching E1–E5, ablations, reporting |
| `paper-section` | Drafting/revising any .tex section |
| `research-log` | Decision records, experiment registry, objection ledger, weekly gate reviews |
| `session-wrap` | End of EVERY working session — verify, summarize, distill lessons |

## Feedback loop
- `.claude/agents/test-runner.md` — runs the suite, reports failures only (use at session-wrap).
- `.claude/agents/safety-auditor.md` — adversarial audit; run on every safety-relevant diff before commit.
- PostToolUse hook (`.claude/settings.json`) runs the mapped pytest target on edits under
  `src/{safety,memory,adaptation}/`, `harness/`, `tests/` — failures come back as blocking feedback.
- `docs/LESSONS.md` — distilled lessons (cap 30); read before designing anything safety-critical.
