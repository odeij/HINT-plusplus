---
name: safety-check
description: Mandatory verification after ANY change to src/safety/, the trust estimator, permission field, gate, risk monitor, or exemplar memory. Adversarial mindset — try to break the property, don't confirm it. Safety failures are BLOCKING.
model: opus
effort: high
---

# Safety Check — R1 checklist

Run every applicable item whenever safety-relevant code changes. Spec: memo §3–§4
(`docs/HINTpp_Design_Memo_R1_2026-06-11.md`). Read `docs/LESSONS.md` first — past failure modes.

## 1. Estimator coherence (Phase 2)
- [ ] **Cold start:** N_k=0 ⇒ λ_k=1 ⇒ m̂_k=0 ⇒ w_k=0 EXACTLY (not approximately).
- [ ] **t=1 worked check:** δ=+1, n₀=5, v_k(0)=0.6 ⇒ w ≈ 0.204·η·η_k (±10%); derivation in test comment.
- [ ] **λ behavior:** λ_k = n₀/(n₀+N_k) uses PER-CLASS cumulative counts N_k, never the global step t;
      strictly monotone → 0 as events accumulate; prior never decays inside the EMA.
- [ ] **No bias-correction-with-prior:** `1/(1−βᵗ)` applies only to zero-init internals m̃, ṽ (F3).
- [ ] **Sign propagation:** a −1-only stream drives w_k < 0; w is SIGNED end-to-end (no abs, no clamp at 0
      before the gate's max(0, 2P_raw−1)).
- [ ] **β assertion:** β₁ < β₂ raised in `__init__`; β₁=0.7, β₂=0.95 in `configs/safety.yaml`.
- [ ] Numerical: ε>0 guards, no log/sqrt of negatives, finite at t=10⁴, float64 accumulators if needed.

## 2. Gate and monitor (Phases 3–4)
- [ ] **Gate closure under adversarial burst — BLOCKING test:** a burst of corrections engineered to
      open then poison a class MUST close its gate (monitor trip or θ_lo crossing). If this test is
      missing for gate code, write it before anything else.
- [ ] **Hysteresis correctness:** opens only after c=2 consecutive P_raw>θ_hi=0.65; closes on
      P_raw<θ_lo=0.45 OR monitor trip; consecutive counter resets on any sub-θ_hi event.
- [ ] **Conditional monotonicity (two-sided):** a test that positive-only evidence ⇒ P_raw,k
      non-decreasing, AND a test REQUIRING P_raw,k to decrease under negative evidence. A gate that
      cannot tighten is flaw F2 — the running max `max(P_safe, P_raw)` is RETIRED; flag ANY occurrence.
- [ ] **Monitor calibration, two-sided:** confidence sequence is anytime-valid (no fixed-n interval);
      trips when lower bound > α_risk; does NOT trip on h-streams genuinely below α_risk (false-trip
      rate checked on synthetic streams); pooled fallback engages for rare classes.
- [ ] G_k = 1[OPEN]·max(0, 2P_raw−1) ∈ [0,1]; spatial G(x)=G_ŷ(x) uses stable per-point identity.

## 3. Propositions
- [ ] **Prop 1 identity:** zero corrections ⇒ output ≡ frozen teacher, verified NUMERICALLY
      (zero-init LoRA; exact tensor equality on a fixed batch, not allclose-with-loose-tol).
- [ ] **Prop 2 KL logging:** the KL stabilization term is logged per round so the Pinsker bound
      ‖p′−p‖₁ ≤ √(2KL) is checkable from run artefacts.

## 4. Memory (Phase 5)
- [ ] Stores outcome-event sufficient statistics, never raw point tensors; bounded size + eviction.
- [ ] Recency-tempered replay weights verified on a synthetic stream (old events downweighted).

## 5. Leakage and rules
- [ ] No target-domain (ScanNet/SemanticKITTI/nuScenes) data, statistics, or tuning anywhere in the
      change; init artefacts remain source-only.
- [ ] No forbidden-reference terms (CLAUDE.md Critical Rules) in code, comments, or commit message.

## Execution
```bash
PYTHONPATH=. /home/ahmad/anaconda3/bin/python -m pytest tests/ -q --tb=short
```
Adversarial pass: for each property above, actively construct the input stream most likely to break
it (alternating δ, single-class floods, all-negative streams, t≫10³, empty classes) and check the
code path by hand or with a throwaway script. Record new attack patterns in the safety-auditor memory.

## Failure response — BLOCKING
STOP all other work. Identify the violated property; decide test bug vs code bug (fix the code, not
the test, unless the test contradicts memo §3–§4); re-run the full suite; all green before continuing.
A weakened test to make the suite pass is a stop-and-ask event, never a fix.
