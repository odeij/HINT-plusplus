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
4. Add gradient checkpointing if sigmoid is saturating

### Category B — Logic Errors in Safety Properties
Symptoms: Monotonicity violated, P(x) outside [0,1], violations increasing
Root causes:
- Bias correction formula wrong (t indexing starts at 0 vs 1)
- β₁ and β₂ swapped
- Permission field update direction inverted
- Missing detach() causing unwanted gradient flow

Fixes:
1. Print intermediate values: m̂ₖ, v̂ₖ, safety weight, P(x) at each step
2. Verify monotonicity with synthetic test: feed strictly increasing evidence, check P never decreases
3. Check β₁ < β₂ (asymmetric by design)
4. Trace gradient flow with `retain_grad()` on key tensors

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

### Category D — Training Instability
Symptoms: Loss oscillates, meta-learning doesn't converge, results vary wildly across seeds
Root causes:
- Meta-learning rate too high (outer loop)
- Inner loop steps insufficient
- Exemplar batch too small for stable gradient estimates
- Conflicting gradients between safety and accuracy objectives

Fixes:
1. Reduce outer learning rate by 10x
2. Increase inner loop steps (try 5 → 10 → 20)
3. Increase exemplar batch size
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
# Run all tests with verbose output
pytest tests/ -v --tb=long

# Run specific safety tests
pytest tests/test_safety.py -v

# Check for NaN in a quick forward pass
python -c "
import torch
from src.adaptation.hint_pp import HINTPlusPlus
model = HINTPlusPlus(num_classes=13)
x = torch.randn(1024, 3)  # synthetic point cloud
out = model(x)
print('Any NaN:', torch.isnan(out).any().item())
"
```

## When to Escalate

If after 2 hours of debugging you cannot reproduce the issue with synthetic data, the problem is likely in data preprocessing or environment. Check:
- PyTorch version matches expected
- CUDA version compatible
- Dataset preprocessing pipeline hasn't changed
- Random seeds are actually fixed
