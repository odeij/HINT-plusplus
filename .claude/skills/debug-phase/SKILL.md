---
name: debug-phase
description: Diagnose and fix bugs in HINT++ code. Use when tests fail, training diverges, NaN appears, or experiment results are unexpected. Triggers on mentions of bug, error, NaN, diverge, crash, failing test, unexpected results.
model: opus
effort: high
---

# HINT++ Debugging Protocol

## Triage: What Category Is This Bug?

### Category A — Numerical Instability (MOST COMMON)
Symptoms: NaN in outputs, loss exploding, gradients vanishing
Root causes:
- Division by √v̂ₖ + ε where ε is too small or missing
- Log of zero or negative values
- Moment accumulation overflow (float32 insufficient for long sequences)
- Sigmoid saturation killing gradients in permission field

Fixes:
1. Check all ε values are >0 and sufficient (try 1e-6 instead of 1e-8)
2. Add `torch.clamp(x, min=1e-7)` before log/sqrt operations
3. Use float64 for moment accumulators if t > 1000
4. If the permission sigmoid saturates, inspect the α scale in `P_raw = σ(α·w)` and the P_raw distribution (pile-up at 0/1)

### Category B — Logic Errors in Safety Properties (λ / gate-state checks)
Symptoms: trust moves the wrong way after events, w ≠ 0 before any correction, P_raw outside (0,1),
gate stuck OPEN or CLOSED, monitor never trips (or trips on clean streams)
Root causes:
- λ-mixture wrong: λₖ computed from the global step t instead of per-class event counts Nₖ;
  prior v_k(0) decaying inside the EMA instead of entering via λ; `1/(1−βᵗ)` applied to a
  nonzero-init buffer (flaw F3)
- β₁ and β₂ swapped
- Gate-state bugs: hysteresis thresholds swapped (θ_lo ≥ θ_hi), consecutive-open counter c not
  reset on a sub-θ_hi event, monitor trip not forcing CLOSED, h-events mislabeled
- Missing detach() causing unwanted gradient flow

Fixes:
1. Print the full chain per event: δ, Nₖ, λₖ, m̃ₖ, ṽₖ, m̂ₖ, v̂ₖ, wₖ, P_raw, gate state, monitor bound
2. λ checks: Nₖ=0 ⇒ λ=1 ⇒ w=0 EXACTLY; λ strictly monotone → 0 as events accumulate;
   t=1 worked check (δ=+1, n₀=5, v_k(0)=0.6 ⇒ w ≈ 0.204·η·ηₖ)
3. Two-sided gate check: +1-only stream ⇒ P_raw non-decreasing and gate opens after c=2 above θ_hi;
   −1-only stream ⇒ P_raw decreases and the gate CLOSES (a gate that cannot close is flaw F2)
4. Check β₁ < β₂ (asymmetric by design); trace gradients with `retain_grad()` on key tensors

### Category C — Domain Shift Failures
Symptoms: Works on S3DIS, fails on target domains. High violations on one domain but not others.
Root causes:
- Point density mismatch (indoor dense vs outdoor sparse)
- Class taxonomy mapping errors
- Feature distribution shift overwhelming the safety signals
- Exemplar memory over-fitted to source domain statistics

Fixes:
1. Visualize point clouds from source vs target — check scale, density
2. Verify class mapping tables are correct (manual inspection)
3. Check if adaptive moments are adapting or frozen (print m̂ₖ before/after)
4. Test with domain-stratified exemplar sampling

### Category D — Adaptation Instability
Symptoms: Loss oscillates, adaptation diverges over a stream, results vary wildly across seeds
Root causes:
- LoRA learning rate too high for the gated correction step
- KL stabilization weight λ_stab too low (anchor drift; check the Pinsker bound from logged KL)
- Replay batch too small for stable gradient estimates
- Conflicting gradients between correction CE and KL stabilization

Fixes:
1. Reduce the LoRA learning rate by 10x
2. Check logged per-round KL against the Prop 2 budget; raise λ_stab if anchors drift
3. Increase replay batch size
4. Plot loss curves per objective — find which one is unstable

## Debugging Workflow

1. **Reproduce:** Create minimal reproduction case with smallest possible data
2. **Isolate:** Test each phase module independently with synthetic data
3. **Bisect:** If integration broke something, find which phase interaction caused it
4. **Fix:** Change ONE thing at a time
5. **Verify:** Run full test suite after fix, not just the failing test
6. **Document:** Add a regression test for this specific failure mode

## Commands

```bash
# Run all tests with verbose output (base anaconda python has pytest; frozen_teacher env does NOT)
PYTHONPATH=. /home/ahmad/anaconda3/bin/python -m pytest tests/ -v --tb=long

# Run the estimator safety tests
PYTHONPATH=. /home/ahmad/anaconda3/bin/python -m pytest tests/test_adaptive_moments.py -v

# Check for NaN in a quick forward pass of the trust estimator
PYTHONPATH=. /home/ahmad/anaconda3/bin/python -c "
import torch
from src.safety.adaptive_moments import AdaptiveMomentSafety
m = AdaptiveMomentSafety(init_path='experiments/phase2_init/results/phase2_init.pt')
delta = torch.sign(torch.randn(13))
w = m(delta)
print('Any NaN:', torch.isnan(w).any().item())
"
```

## When to Escalate

If after 2 hours of debugging you cannot reproduce the issue with synthetic data, the problem is likely in data preprocessing or environment. Check:
- PyTorch version matches expected
- CUDA version compatible
- Dataset preprocessing pipeline hasn't changed
- Random seeds are actually fixed
