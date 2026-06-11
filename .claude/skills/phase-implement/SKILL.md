---
name: phase-implement
description: Build or refactor a HINT++ phase module (Phase 2–6) to the R1 spec. Use when creating phase code, integrating phases, or migrating pre-R1 modules. Triggers on adaptive moments, trust estimator, permission field, gate, risk monitor, exemplar memory, LoRA integration.
model: opusplan
---

# Phase Implementation — R1 spec

Source of truth: `docs/HINTpp_Design_Memo_R1_2026-06-11.md` (R1). Spec deviations need a decision
record in `docs/decisions/` BEFORE code. Tests first; safety failures are BLOCKING.

## Status snapshot (keep this fresh)
- ✅ **Phase 1 — Frozen Teacher.** PTv3-Sonata, S3DIS. Area-5 test mIoU **75.41%**.
  `checkpoints/model_best.pth` (epoch 29). DO NOT modify.
- ✅ **Phase 2 init artefacts.** 0A η_k (entropy ceiling), 0B v_k(0)=0.5r_k+0.5u_k, 0C `phase2_init.pt`
  (frozen 2026-04-27, source-domain-only). Scripts in `experiments/phase2_init/scripts/`.
- ✅ **Phase 2 estimator** — `src/safety/adaptive_moments.py`, R1 (λ-mixture on event-indexed Nₖ,
  outcome δ∈{−1,0,+1}, signed w). 29 tests incl. worked checks; safety-auditor PASS (2026-06-12).
- ✅ **Cross-domain S3DIS→ScanNet zero-shot.** 312 scenes, **42.03% mIoU (−33.38 pp)**. ceiling/beam/
  column/board: IoU 0 at conf > 0.78. Global conf>0.7 triages nothing. See
  `experiments/cross_domain/SCANNET_ZERO_SHOT_FINDINGS.md` — re-read before any safety/threshold work.
- ⬅ **Phase 3 — Permission Field** is next. ⬜ Phases 4–7 not started.

## Per-phase R1 specs and worked checks

### Phase 2 — Trust estimator (Adaptive Moment Safety Signals)
EMAs m_k=β₁m_k+(1−β₁)δ, v_k=β₂v_k+(1−β₂)δ², zero-init internals m̃,ṽ (bias-corrected);
λ_k=n₀/(n₀+N_k) with per-class event counts N_k, n₀=5 (`configs/safety.yaml`, key `safety.n0`);
m̂=(1−λ)m̃; v̂=λ·v_k(0)+(1−λ)ṽ; w=η·η_k·m̂/(√v̂+ε), SIGNED. β₁=0.7<β₂=0.95 asserted in `__init__`.
δ_k(t)∈{+1,−1} is the OUTCOME of a correction event, emitted after the gated update (+1 = local error
on the corrected region decreased; −1 = it did not, or re-correction within T_rc events).
**Worked check (must be a test):** t=1, δ=+1, n₀=5, v_k(0)=0.6 ⇒ m̃=1, λ=5/6, m̂=1/6, v̂=2/3,
w ≈ 0.204·η·η_k (assert ±10%). **Cold start:** N_k=0 ⇒ λ=1 ⇒ w=0 exactly.

### Phase 3 — Permission Field (signed w)
P_raw,k = σ(α·w_k); spatial field via predicted class, P_raw(x)=P_raw,ŷ(x). Design targets from the
ScanNet findings: per-class (a global threshold filters nothing); able to collapse a whole class to ≈0
even at softmax conf 0.95 (ceiling/beam/column/board have no target analog); ≈1 for reliable classes
(floor/wall/chair); continuous for graded failures (sofa/bookcase); conditioned on η_k, not raw softmax.

### Phase 4 — Risk-Controlled Permission Gate (NOT "monotone safety check" — renamed, F2)
Per-class state g_k∈{CLOSED,OPEN} with hysteresis: open after c=2 consecutive events P_raw>θ_hi=0.65;
close when P_raw<θ_lo=0.45 OR monitor trips. G_k=1[OPEN]·max(0,2P_raw−1); G(x)=G_ŷ(x) scales the
correction-loss gradient into LoRA only. Risk monitor: anytime-valid confidence sequence on per-class
harmful rate h_t∈{0,1} (pooled fallback for rare classes); trip when lower bound > α_risk (δ_conf=0.05).
α_risk is the ONLY deployment-semantic knob. Needs stable per-point identity across calls (voxel hash).

### Phase 5 — Exemplar Memory
Outcome EVENTS as sufficient statistics (class, δ, region stats, timestamp) — never raw point tensors.
Recency-tempered replay. Bounded size with eviction.

### Phase 6 — Full Integration
Zero-init LoRA rank 4 in last 2–3 PTv3 blocks; gated CE on corrected region + λ_stab·KL on
high-confidence anchors; two stop-gradients as in HINT-3D; Hydra wiring of `configs/`. Prop 1 numeric
identity test (zero corrections ⇒ output ≡ frozen teacher) is mandatory and BLOCKING.

### Phase 7 — Two-track evaluation
Everything through `harness/` (see `eval-harness` skill). E1–E5 per `experiment-run` skill.

## Forbidden patterns (grep your diff before committing)
- `max(P_safe, P_raw)` running max or any permission that can only loosen — RETIRED (F2).
- `1/(1−βᵗ)` bias correction on any nonzero-initialized statistic (F3) — priors enter via λ only.
- δ as occurrence/count instead of signed outcome (F4).
- Learned click-to-mask models (AGILE3D, PinPoint3D, Point-SAM) anywhere in evaluation paths (F8) —
  regions are training-free (radius default; geometric growing only via decision record).
- Tuning, training, or feedback statistics on ScanNet/SemanticKITTI/nuScenes (zero-shot targets).
- Metrics only HINT++ can score (F6) — headline metrics must be model-agnostic.

## Protocol
1. Confirm the phase and its memo section; read existing `src/` interfaces and the status snapshot.
2. Write the failing tests first — including the phase's worked check and forbidden-pattern guards.
3. Implement minimally (see `coding-discipline`); `nn.Module` for stateful components, type hints,
   docstrings, ε-guards and NaN assertions on safety-critical math.
4. Run the suite: `PYTHONPATH=. /home/ahmad/anaconda3/bin/python -m pytest tests/ -q`.
5. Any safety-relevant change → run the `safety-check` skill / `safety-auditor` agent before commit.
6. Update this snapshot section and `docs/changelog.md`; wrap with the `session-wrap` skill.
