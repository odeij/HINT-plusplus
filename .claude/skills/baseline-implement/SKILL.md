---
name: baseline-implement
description: Wrap a baseline method (TENT, EATA, SAR, GIPSO, HGL, HILTTA-selection, HINT-3D variants, episodic reset) behind the MethodAdapter interface. Use whenever adding or porting any comparison method. Triggers on baseline, TENT, EATA, SAR, GIPSO, HGL, HILTTA, adapter.
model: sonnet
---

# Baseline Implementation — R1 rules

Baselines exist to make the E1 table honest. The bar: a reviewer who loves the baseline should agree
our port is fair.

## Adapter-or-nothing
Every baseline is a `MethodAdapter` (`reset` / `predict` / `adapt_round` — see `eval-harness`).
No baseline gets a bespoke loop, special data access, or its own metric computation. If a method
cannot fit the adapter (e.g., needs target pretraining), it is reported as out-of-protocol in the
paper, not shoehorned in.

## Reproduce-one-published-number gate (per baseline, BEFORE it enters E1)
Port the method, then reproduce ONE published number from its paper (or the GIPSO/HGL protocol
tables) within tolerance (±1 mIoU pt or the paper's own seed std). Record in
`docs/experiments/registry.md` as `EXP-*-repro`. A baseline that misses its repro number is a bug in
our port until proven otherwise — it does NOT enter the primary table.

## Matched-budget invariants (checked by the harness, asserted in tests)
- Same click sequences: identical (stream, seed, budget) ⇒ identical simulator clicks for every method.
- Same frozen teacher checkpoint and preprocessing for every indoor method; same for outdoor.
- Same compute envelope reported: s/round, memory, % params updated — measured, not quoted.
- Zero target tuning for EVERY method: baseline hyperparameters come from their papers' source-side
  defaults. Oracle-tuned HINT++ is the only deliberate exception (labeled ceiling).

## Per-baseline porting notes
- **TENT on PTv3:** PTv3 uses LayerNorm, not BatchNorm — adapt LN affine params (γ, β), not BN
  running stats; entropy minimization over the same voxel set the teacher predicts on. Existing
  starting point: `tent_s3dis.py` at repo root (pre-adapter; wrap it, don't extend it).
- **EATA/SAR:** keep their sample-filtering and anti-forgetting pieces intact (that's their
  contribution); Fisher/EWC anchors computed on SOURCE data only.
- **Episodic reset:** call `reset()` per scene — trivially safe, the persistence foil for E4.
- **GIPSO+HGL (outdoor only):** follow their published protocol verbatim; pending DR-0002 source
  dataset; geometric propagation stays training-free.
- **HILTTA-selection:** human labels spend on online hyperparameter selection — SAME click budget as
  HINT++, clicks consumed from the same simulator stream. The most dangerous comparison; runs FIRST.
- **HINT-3D source-thresholds / tuned:** the predecessor with its thresholds as published (source)
  vs oracle-tuned per domain (upper foil); shares the inner loop with HINT++ by construction.
- **Ungated-LoRA:** HINT++ with the gate forced OPEN and w bypassed — doubles as the gate-off
  ablation; MUST share every other line of code with HINT++ (one flag, not a fork).

## Group labels (paper + configs use these exact group names)
`signal=none` frozen, TENT, EATA/SAR, episodic reset, GIPSO+HGL · `signal=supervision` HINT-3D
variants, ungated-LoRA · `signal=selection` HILTTA-selection · `signal=governance` HINT++,
oracle-tuned HINT++.

## Checklist before a baseline PR merges
- [ ] Adapter only; no harness edits smuggled in.
- [ ] Repro number recorded in the registry with config hash.
- [ ] Source of every hyperparameter cited (paper section/table) in the config file.
- [ ] Leakage scan: no target statistics, no target tuning, no learned masker (F8).
- [ ] Runs under both noise and burst simulator modes without crashing (E2/E3 readiness).
