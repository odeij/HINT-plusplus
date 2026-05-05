---
name: safety-check
description: Verify safety properties of HINT++ code. Use when modifying permission fields, monotone checks, adaptive moment computations, or any code in src/safety/. Also use proactively after code changes to safety-critical modules.
model: opus
effort: high
---

# HINT++ Safety Verification

This skill runs at HIGH effort on OPUS. Safety-critical code demands maximum reasoning depth.

## Mandatory Checks

Run ALL of these whenever safety-related code changes:

### 1. Numerical Stability
- [ ] All denominators have ε > 0 guard
- [ ] No log(0) or log(negative) possible
- [ ] sqrt() only applied to non-negative values
- [ ] No overflow risk in moment accumulation (check dtype, use float64 if needed)
- [ ] Add `assert not torch.isnan(output).any()` after critical computations

### 2. Permission Field Bounds
- [ ] P(x) ∈ [0, 1] for all inputs — verify via sigmoid or clamp
- [ ] P(x) is differentiable (needed for meta-learning gradient flow)
- [ ] Gradient through P(x) does not vanish (check sigmoid saturation regions)

### 3. Monotonicity Guarantee
- [ ] P(x, t+1) ≥ P(x, t) when new safety evidence is positive
- [ ] Monotonicity holds even with noisy corrections (adversarial test)
- [ ] No code path can decrease P(x) when safety evidence accumulates

### 4. Moment Computation Correctness
- [ ] Bias correction applied: m̂ = m / (1 - β₁^t), v̂ = v / (1 - β₂^t)
- [ ] t counter increments correctly (starts at 1, not 0)
- [ ] β₁ ≠ β₂ (asymmetric by design — flag if someone sets them equal)
- [ ] Initialization from frozen teacher confidence, not zeros

### 5. Memory Safety
- [ ] Exemplar memory has bounded size (max_exemplars enforced)
- [ ] Eviction policy works correctly when memory is full
- [ ] No memory leak from accumulated tensors (check .detach() usage)
- [ ] M statistics are sufficient (can reconstruct needed info without raw data)

## Test Command

```bash
pytest tests/test_safety.py tests/test_adaptive_moments.py tests/test_permission_field.py tests/test_monotone_check.py -v --tb=long
```

## If ANY Test Fails

**STOP. Do not proceed with other work.** Safety failures are BLOCKING.

1. Identify the exact property that was violated
2. Determine if it's a test bug or a code bug
3. Fix the code (not the test) unless the test is genuinely wrong
4. Re-run all safety tests
5. All must pass before continuing

## Edge Cases to Test

- All corrections are zero (no human input yet)
- All corrections are identical (degenerate case)
- Extreme class imbalance (1 correction for rare class, 1000 for common)
- Adversarial corrections (deliberately wrong)
- Very long adaptation sequences (t > 10000, check for numerical drift)
