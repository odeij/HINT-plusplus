# LESSONS.md — distilled, durable lessons

Hard cap: 30 entries; prune the least-load-bearing first. New entries come from session-wrap and
must pass all four acceptance criteria: (1) generalizes beyond the incident that produced it,
(2) actionable as a concrete check or rule, (3) not already derivable from CLAUDE.md, the memo, or
a skill, (4) stated with the trigger context in which to apply it.

- **L1 — A gate that can only open is a liveness property, not a safety property.** (from F2) Any
  "safety" state that is monotone toward permissiveness (running max, ratchet, high-water mark) is
  an eventual-permission guarantee; one adversarial burst opens it forever. Trigger: designing any
  gate/threshold state. Check: identify the closing mechanism and the adversarial input that should
  trip it; write that test first.
- **L2 — Never combine zero-init bias correction with nonzero initialization.** (from F3) The
  1/(1−βᵗ) factor assumes a zero-initialized EMA; applied to a prior-initialized statistic it
  inflates the prior (×19 at t=1 for β=0.95), and "fixing" it by dropping the prior creates instant
  full trust. Trigger: any EMA with informed initialization. Check: use a pseudo-count prior
  mixture (λ = n₀/(n₀+N)) instead; hand-verify t=1.
- **L3 — Justify smoothing constants by effective window under the REAL event rate.** (from F5)
  "β=0.95 is slower/safer than Adam" was backwards; corrections are sparse events, not dense
  gradient steps, so reason in events: β=0.7 ≈ 3.3-event window, β=0.95 ≈ 20. Trigger: choosing or
  defending any EMA/decay constant. Check: state the window in events and why that horizon is right.
- **L4 — Check the training set of every pretrained component against the evaluation domains.**
  (from F8) Click-to-mask models (AGILE3D, PinPoint3D, Point-SAM) are trained on ScanNet — using
  them in a ScanNet evaluation is target-domain leakage even if "only" for region proposal.
  Trigger: importing any learned module into the pipeline. Check: list its training corpora; if any
  overlaps a zero-shot target, it is demo-only.
- **L5 — Self-scored metrics are circular.** (from F6) A headline metric defined by the method's
  own internal state (e.g., permission-monotonicity violations) is trivially perfect for the method
  and undefined for baselines. Trigger: defining any headline or success metric. Check: can every
  baseline produce a number for it without adopting our machinery? If not, replace it.
- **L6 — Every estimator gets a worked-number cold-start test.** Hand-compute the estimator's
  output at t=0 (must be the designed neutral value) and t=1 (a worked example with all constants
  substituted, asserted ±10%) before trusting any downstream behavior; this is what caught F3.
  Trigger: implementing or refactoring any estimator. Check: the worked numbers appear in the test
  file with the derivation in a comment.
- **L7 — A doc that references a file vouches for its existence.** (from the R1 bootstrap
  verification) Marking a planned artifact as existing (or citing it in present tense) sends future
  sessions hunting for files that aren't there — the bootstrap's only CRITICAL verification finding
  was exactly this. Trigger: writing any doc/skill/spec that names a path. Check: `test -e` every
  referenced path; what doesn't exist gets an explicit planned marker and the event that creates it.
- **L8 — Per-entity counts in a spec mean per-entity indexing everywhere.** (from the Phase 2 R1
  refactor) A spec written in scalar notation (mₖ = β₁mₖ + (1−β₁)δ) hid a real design choice: with
  per-class Nₖ and δ∈{±1}, the EMAs must be event-indexed — updates masked to the event's class,
  every β-exponent and λ computed from Nₖ, never from global time, and no decay of untouched
  entities. Trigger: implementing any per-class/per-entity estimator from scalar-notation equations.
  Check: ask "what advances this index?" for every exponent and mixture weight before coding.
