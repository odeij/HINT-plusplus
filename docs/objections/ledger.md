# Objection Ledger

Standing reviewer objections and their current rebuttals. Each entry: the objection at full
strength, the rebuttal, and the evidence that must exist by submission. Update rebuttals when
evidence lands; an objection without evidence by the all-experiment freeze (Oct 24) is a risk item
for the Monday gate review.

## OBJ-1 · Five knobs ("no per-domain tuning" is a sham)

**Objection.** Coined as "five knobs" against the pre-R1 design; under R1 the count is nine:
β₁, β₂, n₀, α, θ_hi, θ_lo, c, λ_stab, T_rc. Claiming "no per-domain tuning" while shipping nine
hyperparameters is dishonest.

**Rebuttal.** Every knob except α_risk is fixed ONCE, on source-side validation, and reused
unchanged across all four target streams in both tracks — that is precisely the claim "without
per-domain tuning," and E1 instantiates it with zero target tuning. α_risk is the single
deployment-semantic knob: it declares a risk budget (a semantics choice, like a significance
level), it does not tune performance. Oracle-tuned HINT++ is reported as a ceiling so the cost of
NOT tuning is quantified rather than hidden. **Evidence needed:** one global config in the repo;
E1 table footnoted "identical settings across all streams."

## OBJ-2 · Why not episodic reset?

**Objection.** Resetting the adapter after every scene is simpler and trivially prevents long-run
degradation; the persistent trust state is complexity without payoff.

**Rebuttal.** Reset discards exactly the thing corrections pay for: longitudinal evidence about
which classes' corrections help in this domain. A reset method must re-learn trust from zero every
scene, so its per-click utility cannot compound, and it has no memory with which to manufacture
reliability from imperfect humans (E2) or to keep a gate closed against a repeat adversary (E3).
Episodic reset is in the baseline suite (signal = none), and E4 measures persistence vs reset
head-to-head at matched budgets. **Evidence needed:** E4 table.

## OBJ-3 · Within 3 points of oracle-tuned — marginal gain

**Objection.** If oracle-tuned HINT++ is only ~3 mIoU points better, the contribution is marginal.

**Rebuttal.** The comparison is inverted: oracle tuning is unavailable at deployment (it requires
labeled target data per domain), so "within 3 points of the oracle at matched budget, with the
lowest regression-event rate and zero target tuning" means the safety machinery costs almost
nothing in utility while removing the per-domain tuning that makes current interactive TTA
undeployable. The paper's claim is safety without tuning (memo §1), not benchmark dominance.
**Evidence needed:** E1 utility-vs-safety columns; success criteria in memo §6.

## OBJ-4 · This is just ITTA / HILTTA

**Objection.** Latte++ already claims "Interactive TTA" for 3D; HILTTA already uses human labels
in TTA. The novelty is gone.

**Rebuttal.** Group methods by what the human signal is USED FOR. Latte++ routes prompts into a
promptable branch to refine predictions (supervision at inference); HILTTA spends labels on
hyperparameter selection. Neither maintains a longitudinal per-class trust state, neither spatially
gates parameter updates by that state, and neither carries an anytime-valid risk certificate — the
three clauses of the contribution sentence. We run HILTTA-style selection as a first-class baseline
at the same click budget — scheduled FIRST because it is the most dangerous comparison — and group
related work by signal use so the distinction is structural, not rhetorical. **Evidence needed:**
HILTTA-selection row in the E1 table; related-work clusters in the paper.

## OBJ-5 · Humans are unreliable; trust amplifies bad corrections

**Objection.** Real operators mislabel. A method that converts corrections into parameter-update
permissions will amplify operator error.

**Rebuttal.** δ is an OUTCOME, not an operator claim: it is emitted after the gated update by
measuring whether local error on the corrected region decreased, and re-correction within T_rc
counts as failure — so bad corrections register as −1 and drive trust (and the gate) down, they do
not accumulate as trust. E2 quantifies graceful degradation at 10–30% corruption; E3 is a BLOCKING
test that the gate closes under an adversarial burst; the risk monitor is anytime-valid, so the
α_risk guarantee survives optional stopping. **Evidence needed:** E2 sweep curve; E3 gate-closure
trace; harmful-rate ≤ α_risk adherence column.

## OBJ-6 · The theorem only holds under its assumptions

**Objection.** Thm 1 is a statistics-flavored decoration: its guarantee evaporates if its
assumptions are violated in deployment.

**Rebuttal.** Thm 1 claims exactly what an anytime-valid confidence sequence provides and no more:
gates remain open only while the observed harmful-update rate is statistically consistent with
≤ α_risk at δ_conf = 0.05. hₜ is an observed event outcome (region degraded / re-corrected), not a
modeled quantity; no distributional assumption over domains is introduced, and validity under
optional stopping is the defining property of the construction. The paper instantiates the theorem
empirically (risk-budget adherence in E1–E3) rather than leaving it on paper, and the claims
section maps Thm 1 to those tables (memo §7: every claim traces to a table, figure, or formal
statement). **Evidence needed:** adherence numbers per stream; assumptions stated next to Thm 1.
