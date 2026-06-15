# HINT++ Design Memo — Revision R1

**Date:** 2026-06-11 · **Status:** Approved at G0 · **Supersedes:** all pre-R1 design documents
**Provenance:** transcribed verbatim-in-substance from the approved R1 bootstrap work order
(2026-06-11). This file is the design source of truth; deviations require a decision record in
`docs/decisions/` BEFORE code.

---

## §1 Research goal

**Contribution sentence (exact, never deviate):** "HINT++ is the first interactive TTA method in which
corrections maintain a longitudinal per-class trust state that spatially gates parameter updates, with
anytime-valid risk control — enabling safe deployment to unseen domains without per-domain tuning."

The paper must demonstrate three things:

1. **Safety without tuning** — lowest regression rate at matched click budget on unseen domains.
2. **Reliability manufactured from imperfect humans** — graceful degradation with 10–30% corrupted
   corrections; the gate CLOSES under adversarial bursts.
3. **A certificate** — empirical harmful-update rate respects a declared risk budget α_risk, with
   persistence and no class collapse.

It is NOT a benchmark-dominance or max-mIoU paper. Any work not tracing to (1)–(3) is scope creep.

**Presentation model:** two loops, not seven phases. Inner loop (inherited from HINT-3D):
click → region → gated LoRA step (CE on region + KL on anchors). Outer loop (the contribution):
correction outcome δ → trust posterior → safety weight w → permission field + risk-controlled
gate → governs the inner loop.

## §2 Diagnosis — what was wrong with the pre-R1 pipeline (each flaw → its fix)

- **F1 Protocol unimplementable.** A 13-class S3DIS teacher cannot be evaluated on
  SemanticKITTI/nuScenes (disjoint label spaces, not domain shift). **Fix:** two tracks — Indoor
  primary: S3DIS→ScanNet and ScanNet→S3DIS on the shared-class intersection (~8 classes; build the
  mapping table); Outdoor generality: Synth4D→SemanticKITTI and Synth4D→nuScenes per the GIPSO/HGL
  protocol, with a new PTv3 teacher.
- **F2 The running-max gate `P_safe = max(P_safe, P_raw)` is a liveness property, not safety.**
  Permissions can only loosen; an adversarial burst opens a gate FOREVER. Its "Theorem" was true by
  construction. **Fix:** hysteresis state machine + anytime-valid risk monitor (§4). The running max
  is RETIRED everywhere — code, tests, docs, paper language (the term "monotone safety check"
  included).
- **F3 Estimator math incoherent.** Zero-init bias correction applied to a prior-initialized v
  inflates the prior ×19 at t=1 (β₂=0.95); "fixing" the correction creates instant near-full trust
  (m̂₁ = δ₁). **Fix:** prior pseudo-count mixture (§3). Never combine 1/(1−βᵗ) with nonzero init.
- **F4 δₖ was semantically undefined** (occurrence vs outcome). **Fix:** δₖ(t) ∈ {+1, −1} is the
  OUTCOME of a correction event, emitted after the gated update: +1 if local error on the corrected
  region decreased; −1 otherwise, or on re-correction of a previously fixed region within T_rc events.
- **F5 β rationale was backwards** ("slower than Adam" — 0.95 is faster). **Fix:** justify by
  effective-window ratio under sparse events (m ≈ 3.3-event window, v ≈ 20); keep β₁=0.7 < β₂=0.95
  asserted in `__init__`.
- **F6 Headline metric was circular** (permission-monotonicity violations: trivially 0 for us,
  undefined for baselines). **Fix:** model-agnostic metrics every method can score (§6).
- **F7 Positioning stale.** Latte++ (arXiv:2403.06461) now claims "Interactive TTA" for 3D with a
  promptable branch; HILTTA (arXiv:2405.18911) uses human labels for hyperparameter selection.
  **Fix:** the scoped contribution sentence (§1); baselines grouped by what the human signal is used
  for (§9); the HILTTA-selection baseline runs FIRST (Week 5) — it is the most dangerous comparison.
- **F8 Learned click-to-mask models (AGILE3D, PinPoint3D, Point-SAM) are trained on ScanNet →
  target-domain leakage.** **Fix:** regions are training-free (radius default; geometric growing only
  via decision record); learned maskers are deployment-demo/future-work only, never in evaluation
  numbers.

## §3 Trust estimator (canonical)

EMAs: mₖ = β₁mₖ + (1−β₁)δ; vₖ = β₂vₖ + (1−β₂)δ². β₁ = 0.7 < β₂ = 0.95, asserted in `__init__`.
m̃, ṽ = zero-init bias-corrected internals. Nₖ = cumulative event count for class k.

- λₖ = n₀/(n₀ + Nₖ), with n₀ = 5 (Hydra: `safety.n0`)
- m̂ₖ = (1−λₖ)·m̃ₖ  (prior mean 0 ⇒ warmup damping)
- v̂ₖ = λₖ·vₖ(0) + (1−λₖ)·ṽₖ, where vₖ(0) = 0.5rₖ + 0.5uₖ (existing Sub-step 0B)
- wₖ = η·ηₖ·m̂ₖ/(√v̂ₖ + ε) — **SIGNED**. ηₖ entropy ceiling (Sub-step 0A) unchanged; mₖ(0) = 0.

**Worked check that MUST hold in tests:** t=1, δ=+1, n₀=5, vₖ(0)=0.6 ⇒ w ≈ 0.20·η·ηₖ (±10%).
(m̃₁ = 1, λ₁ = 5/6, m̂₁ = 1/6 ≈ 0.167, v̂₁ = (5/6)·0.6 + (1/6)·1 = 0.667, w = 0.167/0.816 ≈ 0.204·η·ηₖ.)
**Cold-start identity that MUST hold in tests:** Nₖ = 0 ⇒ λₖ = 1 ⇒ m̂ₖ = 0 ⇒ wₖ = 0 EXACTLY.
Never combine 1/(1−βᵗ) bias correction with nonzero initialization.

## §4 Gate and risk monitor (canonical)

Permission: P_raw,ₖ = σ(α·wₖ). Per-class state gₖ ∈ {CLOSED, OPEN}: open after c=2 consecutive
events with P_raw > θ_hi = 0.65; close when P_raw < θ_lo = 0.45 OR the monitor trips.
Gₖ = 1[OPEN]·max(0, 2·P_raw − 1); spatial G(x) = G_{ŷ(x)} scales the correction gradient into LoRA
only.

Risk monitor: hₜ ∈ {0, 1} per correction event (prior gated updates degraded the region /
re-correction); anytime-valid confidence sequence on the per-class harmful rate (pooled fallback for
rare classes); trip when the lower bound > α_risk (δ_conf = 0.05). **α_risk is the ONLY
deployment-semantic knob.**

## §5 Evaluation protocol and datasets

Two tracks, zero target tuning. ScanNet/SemanticKITTI/nuScenes are zero-shot: never train, tune, or
compute feedback statistics on them.

- **Indoor (primary):** S3DIS→ScanNet and ScanNet→S3DIS on the shared-class intersection
  (~8 classes; mapping table to be built and checked into `docs/`).
- **Outdoor (generality):** Synth4D→SemanticKITTI and Synth4D→nuScenes per the GIPSO/HGL protocol,
  new PTv3 teacher. **Open question (DR-0002, due 2026-06-19):** outdoor SOURCE dataset —
  **Synth4D vs SynLiDAR**. Synth4D matches GIPSO/HGL most directly; SynLiDAR offers more scale and
  label diversity but a weaker protocol precedent. Decide before the outdoor teacher is launched;
  descope checks Jul 24 / Aug 7.

Regions are training-free (radius default; geometric growing only via decision record). Learned
maskers (AGILE3D, PinPoint3D, Point-SAM) never appear in evaluation numbers (F8).

## §6 Metrics (model-agnostic; every method can score them)

- **RER** — % scenes with post-adapt mIoU < teacher − 1.0 pt.
- **Worst-stream mIoU**; **CCC** — % streams with any shared class >10 IoU below teacher.
- **Risk-budget adherence** — empirical harmful rate vs declared α_risk.
- **mIoU@k clicks**, **NoC@target**, **gain-per-click**.
- **Cost:** s/round (~0.5 target), memory, % params updated.
- All ≥3 seeds, mean ± std, matched click budgets.

**Success criteria (Phase 7):** Safety — lowest RER among all baselines at matched budget;
worst-stream mIoU ≥ teacher − margin; CCC ≈ 0; empirical harmful rate ≤ declared α_risk
(instantiates Thm 1). Utility — shared-class mIoU within 3 pts of oracle-tuned HINT++ and
≥ HILTTA-selection at the same budget. Robustness — graceful degradation on the E2 noise sweep;
gate closes under the E3 adversarial burst (BLOCKING). Cost — ≈0.5 s/round, <3% params updated,
reported per method.

## §7 Formal claims

- **Prop 1:** zero corrections ⇒ output ≡ frozen teacher (zero-init LoRA; numeric identity test).
- **Prop 2:** KL-stabilization budget bounds anchor drift via Pinsker: ‖p′−p‖₁ ≤ √(2KL).
- **Thm 1:** anytime-valid risk control — gates open only while the harmful rate is statistically
  consistent with ≤ α_risk.

Every paper claim must trace to a table, figure, or one of these statements.

## §8 Adapter (unchanged from HINT-3D)

Zero-init LoRA rank 4 in the last 2–3 PTv3 blocks; gated CE on the corrected region +
λ_stab·KL on high-confidence anchors; two stop-gradients as in HINT-3D.

## §9 Baselines, grouped by what the human signal is used for

- **none:** frozen teacher, TENT, EATA/SAR, episodic reset, GIPSO+HGL (outdoor).
- **supervision:** HINT-3D source-thresholds, HINT-3D tuned, ungated-LoRA (= gate-off ablation).
- **selection:** HILTTA-style online threshold selection (same clicks; runs FIRST).
- **governance:** HINT++, oracle-tuned HINT++ (ceiling).

## §10 Experiments and gates

E1 primary two-track table → E2 noisy-oracle sweep p ∈ {0,10,20,30}% → E3 adversarial burst (gate
MUST close; blocking) → E4 persistence vs episodic reset → E5 ablations (only after E1 passes).

Gates (slips eat buffers, never gates): G1 Jul 20 — gated vs ungated separation on a real ScanNet
mini-run (fail → debug, pivot Jul 31) · G2 Aug 3 — full indoor table vs all families incl.
HILTTA-selection · outdoor descope checks Jul 24 / Aug 7 · new-experiment freeze Oct 17 ·
all-experiment freeze Oct 24 · submit ≥48 h early (CVPR 2027 deadline expected ~Nov 13, 2026).

## §11 Hard rules

- `pytest` before every commit; safety test failures are BLOCKING (incl. gate-closure adversarial test).
- Zero-shot targets never tuned or leaked; no feedback statistics computed on them.
- No hand-edited numbers: tables and figures are generated by scripts from `experiments/results/`.
- Decision record in `docs/decisions/` BEFORE any spec deviation.
- Every experiment runs through the eval harness — no bespoke loops.
- The forbidden-reference terms from the pre-R1 CLAUDE.md Critical Rules must have zero occurrences
  in code, comments, commit messages, and paper text.
