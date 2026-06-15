# Attack patterns

- Per-class N_k vs global t leakage: after the bias-correction refactor, drive a class to
  N_k=100 and another to N_k=1 in the SAME module, then check m_hat denominators differ
  ((1-β^100) vs (1-β^1)); also spam no-event (all-zero) forwards so t≫N_k and confirm the
  first real event still reproduces the worked check. Catches t bleeding into λ or β^t exponents.
- float32 β^N underflow: drive 10^4 single-class events; β1^N → 0.0 exactly in float32, so the
  bias denom (1-β^N) clamps to 1.0 — verify w stays finite and clamp_min(1e-12) is never
  load-bearing for any valid N (smallest N=1 ⇒ 1-β = 0.05..0.3, far from the floor).
- m_k_0 validation honesty: forge an init .pt with a nonzero m_k_0 and confirm the loader RAISES
  (retired pre-R1 prior must not silently enter a zero-init EMA); also confirm a .pt with m_k_0
  absent is accepted. A loader that ignores stale m_k_0 reintroduces F3.
- Test-change ledger audit: when an estimator test file is rewritten, diff removed assertions and
  confirm the deleted "cold-start identity" (m_hat(1)=δ) was the F3 bug itself, not a real
  property being weakened. Compare every kept assertion to memo §3, not to the new code.
- Sign symmetry under v=δ²: run mirrored all-+1 and all-−1 streams; w must be exactly antisymmetric
  (sum ≈ 0) because v uses δ². Asymmetry reveals an abs() or clamp sneaking into the signed path.
- Running-max residue grep: search src/safety + tests for p_safe|running.?max|high.?water|
  ratchet|monotone; a "monotone" hit on λ-decreasing-toward-0 is benign, a P_safe high-water mark
  is an automatic CRITICAL (flaw F2).
