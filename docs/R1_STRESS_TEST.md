# HINT++ R1 — Adversarial Stress Test (full record)

![verdict](https://img.shields.io/badge/verdict-GO--WITH--FIXES-yellow) ![root](https://img.shields.io/badge/root%20cause-δ%2Fh_t%20undefined-red) ![novelty](https://img.shields.io/badge/scoped%20novelty-survives-brightgreen) ![scale](https://img.shields.io/badge/82%20agents%20·%203.9M%20tokens-blue) ![date](https://img.shields.io/badge/run-2026--06--16-lightgrey)

> **What this is.** The complete record of an adversarial multi-agent stress test of the HINT++ R1
> design, run before committing five weeks to the Phase-3 build. 8 hostile agents each tried to
> *kill* a part of R1 (grounded in the real spec + code), every serious finding was put through a
> 3-vote refutation panel, and the survivors were synthesized into a verdict plus a hostile
> reviewer-2 rejection. Spec under test: [`HINTpp_Design_Memo_R1_2026-06-11.md`](HINTpp_Design_Memo_R1_2026-06-11.md).
> Raw output: `tasks/w9o7oi65z.output` (571 KB) in the session dir.

---

## TL;DR

**R1 is a sound research *direction* with a defensible, scoped novelty — but the R1 *spec as written*
is not yet a sound road to submission, because its central measured quantity (δ / h_t) is undefined
and its G1 schedule is fictional.** Verdict: **GO-WITH-FIXES**, conditional on **4 decision records
before Phase 3 code.**

Three things to hold in mind, because they are easy to misread:

1. **The per-finding severities are inflated.** The refutation panel graded **69 of 72 votes
   OVERSTATED**, only **2 REAL_HOLE**, **1 FALSE** — and **0 ALREADY_HANDLED**. So the attack labels
   ("CRITICAL ×10") overstate each issue *individually* — but every issue is a **genuine gap not
   covered by the existing objection ledger** (nobody voted "already handled"), and the synthesis
   found that several individually-overstated findings **compound** into a few real problems.
2. **The root is one hole with several faces:** δ (the trust signal) and h_t (the certificate's
   signal) have **no defined measurement operator** at deployment, where targets are zero-shot with
   no labels. This single gap propagates into the trust state, the gate, the certificate, and every
   safety metric. *This is exactly the hole flagged in the prior single-read review — now confirmed
   empirically (grep'd absent in docs/spec/code).*
3. **The scoped "first" survives** a fresh 2023–2026 literature sweep — but the sweep found **5
   citations the prior scan missed** (chiefly **InterPCSeg**, arXiv:2403.06401, the nearest
   3D-on-S3DIS/ScanNet inner-loop relative) that must be added before submission.

None of the findings is fatal-and-unfixable; each has a concrete, mostly-cheap fix. But fixes 1–4
below must be written **before** Phase 3 code, or the project builds a safety stack on an undefined
signal and an inverted prior.

---

## Methodology

| Phase | What ran | Agents |
|---|---|---|
| **Attack** | 8 hostile agents, each grounded in the real files (memo, CLAUDE.md, `adaptive_moments.py`, tests, ledger, scan), each trying to *kill* one dimension | 8 |
| **Verify** | every CRITICAL/HIGH finding → 3 perspective-diverse refuters (check spec/code · check ledger/scan · steelman the defense) trying to *refute* it | 24 × 3 = 72 |
| **Synthesize** | one synthesis agent (rank + verdict + go/no-go) + one hostile CVPR reviewer-2 (rejection narrative) | 2 |

**Attack dimensions (4 findings each → 28 total):** ① the trust signal δ + estimator; ② the spatial
gate; ③ the certificate (Thm 1) + monitor; ④ "no per-domain tuning"; ⑤ experimental design;
⑥ internal coherence; ⑦ feasibility & schedule. ⑧ A parallel literature counterexample hunt
(web, 10+ searches) tested the "first" claim.

**Scoring rule:** a finding was "confirmed" if ≥2 of 3 refuters graded it REAL_HOLE **or** OVERSTATED
(i.e. real, even if severity-inflated). All 24 CRITICAL/HIGH findings cleared that bar; the 4 MEDIUM
findings were not sent to the panel.

### The verification reality-check (read this before the severities)

```
72 refutation votes across 24 serious findings:
  OVERSTATED      ███████████████████████████████████████████████████████████████████  69
  REAL_HOLE       ██                                                                     2
  FALSE           █                                                                      1
  ALREADY_HANDLED                                                                        0
```

- **OVERSTATED dominates** → the attack framing exaggerates each finding's severity. Do **not** read
  "10 CRITICAL" as "10 paper-killers."
- **0 ALREADY_HANDLED** → none of these 28 is covered by the 6 existing objections in the ledger.
  They are *new* gaps.
- The only two **REAL_HOLE** votes landed on **(#1) the δ/h_t measurement operator** and **(#8/#9)
  the monitor multiplicity/endogeneity** — i.e. the panel's strongest "this is genuinely unresolved"
  signal points at exactly the root cause.

---

## Verdict (synthesis)

> R1 is a sound *research direction* with a genuinely defensible, scoped novelty claim — but the R1
> *spec as written* is not yet a sound *road to submission*, because its central measured quantity is
> undefined and its schedule is fictional.
>
> **Solid:** the novelty cell survives a fresh scan; the Phase-2 estimator is real and tested (29
> tests, worked checks, cold-start w=0); Prop 1 is clean and checkable; the two-loop framing and the
> motivating cross-domain failure (42.03% mIoU; ceiling/beam/column/board confidently-wrong at
> conf>0.78, IoU 0) are real and compelling.
>
> **Fragile and load-bearing:** the δ/h_t measurement gap is the root; the source prior is
> *anti-correlated* with target danger on the worst class; the headline metrics are *blind* to the
> dangerous classes (no target GT); and the G1 schedule is a 6-workstream zero-slack chain on a
> near-empty codebase.
>
> **GO-WITH-FIXES. RECONSIDER is not warranted** — the novelty cell is unoccupied, the core mechanism
> is coherent once δ is operationalized, and every hole has a concrete fix. But if the
> before-Phase-3 conditions are not in decision records, the project is building a safety stack on an
> undefined signal and an inverted prior, and should pause.

### Strongest single objection (verbatim)

> The trust signal δ (and the identical certificate signal h_t) has **no defined measurement
> operator**, and this single undefined number is the common root of the four most damaging findings.
> Verified by grep that the memo defines δ/h_t only semantically ("local error on the corrected
> region decreased" / "region degraded") and that **no operator** — no held-out probe, reserved
> fraction, or simulator/oracle definition — exists anywhere in docs/, spec, ROADMAP, or code
> (`adaptive_moments.py` merely range-validates δ∈{−1,0,+1}). At deployment the targets are zero-shot
> with no labels, so "error decreased" cannot be measured against ground truth. Two exhaustive
> readings, both fatal as written: **(a)** against the human click label on the region the LoRA step
> just trained CE on ⇒ circular, δ≈+1 by construction for every class regardless of harm; **(b)** an
> unlabeled proxy (entropy/confidence) ⇒ anti-correlated with truth on exactly the confidently-wrong
> dangerous classes, so δ=+1 fires precisely on harmful updates. The fix is cheap (define δ on a
> held-out neighborhood; validate informativeness on labeled SOURCE data; state the certificate is
> validated under the simulator's oracle, not live), but until it is written in a DR, this is the
> objection a sharp reviewer reaches first.

---

## The 4 compounding root problems

| # | Problem | Verified evidence | Consequence |
|---|---|---|---|
| **R1** | **δ/h_t measurement operator undefined** | grep: no operator in docs/spec/code; `adaptive_moments.py` only range-validates δ | trust state, Thm 1, and every safety metric are downstream of a number that doesn't exist |
| **R2** | **Source prior inverted on the worst class** | `phase2_init.pt`: ceiling v_k(0)=**0.0422** (lowest/least-conservative of 13), η_k=0.9845; ceiling is the most dangerous ScanNet class (conf 0.95, IoU 0) | cold-start safety for source-only classes can't come from the prior — must come from the gate, and the paper must prove it |
| **R3** | **Metrics blind to the dangerous classes** | the 4 dangerous classes have no ScanNet GT (has_gt=False) → contribute IoU 0 by construction, excluded from CCC, no GT to score δ | the paper would demonstrate safety on already-safe classes while the advertised failure is invisible to every headline metric |
| **R4** | **G1 schedule is fiction** | only `adaptive_moments.py` (199 lines) exists; `harness/`, `src/{models,adaptation,memory}` absent; registry empty | a 6-workstream zero-slack serial chain + a from-scratch HINT-3D reimpl in ~5 weeks, one student |

---

## Action plan (the go/no-go conditions)

**BEFORE Phase 3 code (this week) — mandatory:**
1. **DR: define the δ/h_t measurement operator** (held-out neighborhood or reserved-fraction), and
   state plainly that the certificate is validated under the **simulated oracle** in E1–E5, **not**
   claimed live at deployment. This re-words what Thm 1 may claim. *(R1)*
2. **DR + cross-domain validation** of η_k / v_k(0) against per-class target danger on the existing
   S3DIS→ScanNet run; document that the **gate, not the prior**, carries cold-start safety for
   source-only classes. *(R2)*
3. **DR: pin every undefined knob** (T_rc, λ_stab, α_risk default, region radius, α, θ_hi, θ_lo, c)
   to source-side values in `configs/safety.yaml`, with a config-sync test. *(knobs)*
4. **DR-0003: de-risk G1** — de-scope to a harness-free gated-vs-ungated script; add a ~Jul 6
   inner-loop smoke milestone; add a self-contained behavioral done-when on SOURCE data; pin every
   inner-loop hyperparameter. *(R4)*

**BEFORE G1 (Jul 20):** gate state-machine env-only unit test for E3 closure (decouple the BLOCKING
property from the GPU harness); DR confronting `G(x)=G_{ŷ(x)}` indexing (route suppression through
the **target** class's gate).

**BEFORE G2 (Aug 3):** add the no-GT spurious-prediction-suppression metric and/or make ScanNet→S3DIS
the primary safety direction; pre-register a **paired** (RER ≤ X AND gain-over-teacher ≥ Y) acceptance
on a Pareto plot; commit to a named betting/e-process confidence sequence with family-wise control +
the masked-rare-class BLOCKING ablation; close DR-0002 and sequence outdoor **after** G1.

---

## Ranked confirmed holes (19)

Severity is the *attacker's* label; recall the panel graded most OVERSTATED. "Blocks" = the phase/gate
the fix must precede.

| # | Sev | Hole | Address via | Blocks |
|---|---|---|---|---|
| 1 | CRIT | δ/h_t measurement operator undefined at deployment | DR + spec + experiment | Phase 3, Thm 1 wording |
| 2 | CRIT | Metrics blind to dangerous classes (no target GT) | DR + protocol redesign | Phase 7 metrics, G2 |
| 3 | CRIT | Source prior v_k(0) anti-correlated with target danger | DR + experiment (cheap) | Phase 3, "no-tuning" framing |
| 4 | CRIT | G1 = 6-workstream zero-slack chain on empty codebase | DR-0003 (re-sequence) | G1 → G2 → submission |
| 5 | CRIT | HINT-3D inner loop reverse-engineered, no reference/baseline | DR + spike | inner-loop deliverable |
| 6 | CRIT | RER gameable by inaction (frozen teacher wins it) | spec (success criteria) | Phase 7, pre-register before G2 |
| 7 | CRIT | Gate circularity: G(x)=G_{ŷ(x)} keys on the wrong (hallucinated) class | DR + experiment | Phase 3/4, before G1 |
| 8 | HIGH | Per-class CS, no family-wise control; pooling masks one bad rare class | DR + experiment | Phase 4 monitor, Thm 1 |
| 9 | CRIT | h_t endogenous (closed-loop) and = δ=−1; anytime-validity unaddressed | DR + experiment | Phase 4, Thm 1 assumptions |
| 10 | HIGH | "Spatially" unearned — G is a 13-entry scalar via argmax | phase-design + ablation | "spatially" clause |
| 11 | HIGH | Informative δ (delayed −1) suppressed by β/n0; harm already committed | DR (T_rc) + test | Phase 4 dynamics, E3 |
| 12 | HIGH | T_rc, λ_stab, α_risk default, region radius have no value | DR + config + test | Phase 3/4, G1/G2 |
| 13 | HIGH | E3 trivially passable by an over-conservative gate (no liveness side) | spec (two-sided E3) | Phase 4 / E3 protocol |
| 14 | HIGH | α_risk: no value; one value may not serve indoor(−33pp) + outdoor(+6pp) | DR + experiment | Phase 7 E1, outdoor descope |
| 15 | HIGH | 3 seeds: no power for rare-event RER/CCC; matched-budget undefined cross-family | experiment design | Phase 7 stats, HILTTA |
| 16 | HIGH | h_t not i.i.d.; anytime-validity needs an unnamed martingale construction | DR + experiment | Phase 4 (folds into #9) |
| 17 | HIGH | Outdoor: 40-day teacher unstarted, descope checks placed too late | DR (close DR-0002) | outdoor track |
| 18 | HIGH | Hysteresis/α magic numbers couple to gate liveness | DR + config + experiment | Phase 3/4 (folds #12) |
| 19 | HIGH | Harness + noise/burst simulator absent; single point of failure incl. E3 | build-order + tests | Phase 7, E3 |

### The seven CRITICALs, expanded

- **#1 δ/h_t undefined** — see the strongest-objection box above. *Fix:* δ on a held-out neighborhood
  (never the trained-on points) or a reserved fraction of the clicked region; certificate validated
  under the E1–E5 simulated oracle; a source-validation experiment proving per-class δ=+1 rate
  correlates with *true* region improvement and is not ≈1 everywhere.
- **#2 metrics blind** — ceiling/beam/column/board have IoU 0 on ScanNet *by construction* (no
  analog), so they cannot move mIoU/RER and are excluded from CCC. *Fix:* add a **spurious-prediction
  suppression rate** (fraction of confidently-predicted no-analog-class points the gate keeps CLOSED
  — measurable without GT), and/or make **ScanNet→S3DIS** the primary safety direction (S3DIS has GT
  for all 13 classes).
- **#3 prior inverted** — ceiling v_k(0)=0.042 is the *least* conservative prior yet ceiling is the
  most dangerous target class; beam's protection (0.60) is an accident of source rarity. *Fix:* plot
  η_k/v_k(0) vs per-class target IoU on the existing run; document that the gate carries cold-start
  safety for source-only classes; E1 reports per-class gate-open rates for the 4 classes.
- **#4 schedule** — *Fix:* DR-0003 de-scopes G1 to a harness-free script, adds a Jul 6 inner-loop
  smoke milestone and an explicit Jul 31 pivot criterion, and builds the gate state machine as an
  env-only pytest test so the BLOCKING E3 property is checkable without the GPU harness.
- **#5 inner loop** — full spec is 2 sentences (memo §8); block indices, LR/steps, λ_stab, stop-grad
  placement, anchor threshold, radius→token mapping all undefined; no HINT-3D code/numbers to
  validate against; Prop 1 identity is trivially true and tests nothing about learning. *Fix:* DR
  pinning every hyperparameter + a self-contained behavioral done-when on ≥10 S3DIS source scenes
  (gated LoRA raises corrected-region IoU ≥ X pp vs frozen, KL drift below the Pinsker budget) + a
  5-day spike before the harness.
- **#6 RER gameable** — Prop 1 ⇒ frozen teacher / all-gates-closed give RER=0 exactly; the certificate
  is auto-satisfied by a mostly-closed gate; E3 passes for an over-conservative gate. *Fix:* a
  **paired pre-registered** (RER ≤ X AND gain-over-teacher ≥ Y) acceptance on a Pareto plot; report
  gate-open rate + #updates; disqualify a method with RER=0 and gate-open≈0.
- **#7 gate circularity** — `G(x)=G_{ŷ(x)}` looks up trust under the model's *predicted* class, which
  cross-domain is the wrong one; and because rare dangerous classes never reach c=2 events their gate
  stays CLOSED, so corrections on hallucinated-ceiling points get gradient 0 — the gate *freezes the
  hallucination it was built to fix*. *Fix:* route the suppression gradient through the **target**
  (human-asserted) class's gate; index by ŷ(x) only off-region; E5 ablation showing the gate drives
  dangerous-class predicted-frequency down.

### The 4 MEDIUM findings (not panel-verified)

- **λ-mixture never forgets** — N_k only increases, so λ_k→0 *permanently*; the conservative prior
  switches off forever per class, in tension with "longitudinal" and the E4 persistence experiment.
  *Fix:* DR on a λ-floor or windowed N_k; an E4-adjacent test where a class flips easy→hard mid-stream
  and the gate must re-close.
- **Prop 2 orthogonal** — Pinsker bounds drift on the *already-trusted* anchor set, not correctness on
  at-risk points. *Fix:* scope Prop 2 as a **stability** lemma, not part of the harm certificate; add
  the measured ‖p′−p‖₁ vs √(2KL) figure (logging ≠ verifying).
- **Oracle ceiling strawman** — if oracle-tuned HINT++ tunes only α_risk, the ceiling is artificially
  low and "within 3 pts" is hollow. *Fix:* let the oracle tune **all** source-fixed knobs (+ recompute
  η_k/v_k(0) on target) per stream; report the honest gap.
- **Latte++ vs ITTA naming** — memo §2-F7 and OBJ-4 still say "Latte++"; the scan establishes the
  interactive method is **ITTA**. *Fix:* reconcile memo + ledger before the related-work section.

---

## Complete finding register (28, by attack dimension)

Each dimension produced 4 findings. Tally = refutation votes (O=OVERSTATED, R=REAL_HOLE, F=FALSE).

| Dim | Findings (severity · vote tally) |
|---|---|
| **① Trust signal δ** | δ operator undefined (CRIT · 2O/1R) · informative δ is the suppressed delayed −1 (HIGH · 2O/1F) · v_k(0)=0.5r+0.5u blends non-comparable quantities (HIGH · 3O) · λ never forgets (MED) |
| **② Spatial gate** | G keyed on wrong class / freezes hallucination (CRIT · 3O) · "spatially" is a per-class scalar (HIGH · 3O) · θ/c/α magic numbers (HIGH · 3O) · rare classes never reach c=2 → gates never open (HIGH · 3O) |
| **③ Certificate / Thm 1** | h_t no GT on targets (CRIT · 3O) · per-class CS no family-wise control + pooling masks bad class (CRIT · 1R/2O) · h_t not i.i.d., no martingale named (HIGH · 3O) · h_t ≡ δ=−1, endogenous closed loop (CRIT · 3O) |
| **④ No per-domain tuning** | source prior anti-correlated with target danger (CRIT · 3O) · T_rc/λ_stab uncommitted (HIGH · 3O) · α_risk no value, may differ per track (HIGH · 3O) · oracle ceiling strawman (MED) |
| **⑤ Experimental design** | metrics blind to dangerous classes (CRIT · 3O) · RER gameable by inaction (CRIT · 3O) · E3 trivially passable, no liveness side (HIGH · 3O) · 3 seeds no power + matched-budget undefined (HIGH · 3O) |
| **⑥ Internal coherence** | h_t = δ=−1 redundancy (CRIT · 3O) · 4 knobs unvalued (HIGH · 3O) · Prop 2 / Thm 1 untested vs the "every claim traces" rule (HIGH · 3O) · Latte++/ITTA contradiction (MED) |
| **⑦ Feasibility / schedule** | G1 zero-slack chain on empty codebase (CRIT · 3O) · inner loop unvalidatable reimpl (CRIT · 3O) · outdoor teacher unstarted, descope too late (HIGH · 3O) · harness/simulator absent = single point of failure (HIGH · 3O) |

---

## Novelty hunt — the scoped "first" survives

**Verdict:** After a fresh 2023–2026 fan-out (10+ searches + full-text checks), **no single paper
occupies the governance cell** — none simultaneously (a) uses correction *outcomes* as the signal,
(b) keeps a longitudinal per-class trust state that gates updates, and (c) carries anytime-valid risk
control on harmful updates. The three clauses exist only separately. **The contribution sentence must
stay exactly scoped** — drop the qualifiers and it is false, because ITTA already owns "interactive
TTA for 3D."

**Citations the prior scan missed (add before submission; not novelty kills):**

| Paper | Why it matters | Why it doesn't pre-empt |
|---|---|---|
| **InterPCSeg / "Refining Segmentation On-the-Fly"** (arXiv:2403.06401) | **Nearest inner-loop relative** — interactive 3D TTT on the *same* S3DIS/ScanNet, click→TTT→stabilization-energy, ~identical inner loop | signal=supervision; no trust state, no spatial trust-gating, no risk certificate — the entire outer loop is absent |
| **Monitoring Risks in TTA** (arXiv:2507.08721) | **Closest on the certificate axis** — anytime-valid CS on running test risk | only *alarms* (doesn't gate updates); aggregate risk not per-class harmful rate; global not per-class; no human |
| **You Point, I Learn** (arXiv:2503.06717) | online adaptation from clicks under shift (2D medical) | signal=supervision; no trust posterior/gate/certificate |
| **Kontogianni et al.** (ECCV'20, arXiv:1911.12709) | the "learn from corrections at test time" root | 2D, supervision, no governance |
| **DC-TTA** (ICCV'25, arXiv:2506.23104 — *verify*) | recent click-driven TTA of SAM | 2D, supervision, divide-and-conquer merge not a trust gate |

> ⚠️ **Verify before citing.** The agent itself caught that an automated summarizer mischaracterized
> 2507.08721 as "governs parameter updates" — the full text says it only alarms. Treat all
> newly-surfaced IDs (esp. any 2026-dated) as *to-verify against the HTML* before they enter the paper.

**Residual novelty risk:** horizon (a new entrant could land in the cell before Nov 2026 — re-run this
exact scan at the Oct 17 freeze and pre-submission); positioning (add the 5 citations; reconcile the
ITTA naming in memo §2-F7 / OBJ-4).

---

## Reviewer-2 — the rejection you must survive

*Generated by a hostile-CVPR-reviewer agent shown only the confirmed weaknesses. This is the
worst-case framing — deliberately harsher than the calibrated synthesis above, which weighed the
"OVERSTATED" grades. Where reviewer-2 says "fatal," the panel says "real, fix before Phase 3."*

> **Recommendation: Reject (confidence 4/5).**
>
> I credit the scoped novelty — the governance cell is genuinely unoccupied. But novelty in an
> unoccupied cell is not a result, and the central technical object the method rests on — the outcome
> signal δ and its twin h_t — is **undefined at deployment**.
>
> - **W1 (fatal).** δ's measurement operator does not exist on a zero-shot target. Either δ is against
>   the click label the LoRA just trained on (circular, δ≈+1 for every class) or an unlabeled proxy
>   (anti-correlated with truth on exactly the confidently-wrong classes). Everything downstream is
>   built on a signal whose definition is absent.
> - **W2 (fatal).** The certificate controls a GT-free, optimistically-censored proxy h_t, not the
>   safety property. The 4 dangerous classes have no GT analog, so no re-correction proxy can fire
>   harmful on them — Thm 1 is provably true about the wrong quantity and structurally silent about
>   the exact failure the paper exists to absorb.
> - **W3.** The gate is keyed on ŷ(x) — the broken prediction — and because rare dangerous classes
>   never reach c=2 events, their gate stays closed and zeroes the very corrections meant to demote
>   the hallucination. The gate enforces the hallucination it was designed to suppress.
> - **W4.** "Spatially" is a 13-entry per-class scalar painted on by argmax — a load-bearing word the
>   math doesn't earn.
> - **W5.** Source priors are inverted for the worst class (ceiling v_k(0)=0.042, the least
>   conservative, yet the most dangerous). "Fixed on source" is per-domain tuning relocated to the
>   source taxonomy.
> - **W6.** θ_hi/θ_lo/c/α are unjustified magic numbers; T_rc, λ_stab, region radius, α_risk have no
>   value — yet "every knob is fixed once on source" is the knob-count rebuttal. You can't fix what
>   isn't set.
> - **W7.** ~8–13 per-class confidence sequences with no family-wise control; CS width is widest on
>   the rarest (dangerous) classes; pooling dilutes a bad rare class below α_risk; h_t is
>   autocorrelated with no named martingale construction; h_t and δ=−1 are one statistic in a closed
>   loop.
> - **W8.** The evaluation is blind to the motivating failure (dangerous classes unscoreable), and RER
>   is won by the frozen teacher (Prop 1 ⇒ RER=0). The three demonstrations are jointly satisfiable by
>   a near-inert method; no minimum gain-over-teacher is set; 3 seeds give no power.
> - **W9.** Two of three formal claims have no experiment; Prop 1 is trivially true and tests nothing
>   about learning.
> - **W10.** The G1 artifacts don't exist yet and the inner loop is an unvalidatable reimplementation.
>
> **What would change my score to accept:** (1) define δ/h_t deployably and prove it's not circular
> (evaluate on a held-out labeled shift, show δ→−1 on confidently-wrong classes); (2) fix the
> gate-keying circularity (index by the corrected label; prove the gate can demote a hallucinated
> class); (3) make the certificate honest (named martingale valid under the autocorrelated closed-loop
> stream, family-wise control, non-masking pooling, null-calibration showing false-trip ≤ δ_conf);
> (4) evaluate on classes that can actually fail + a minimum gain-over-teacher + a two-sided E3; (5)
> commit every knob to a source-side-fitted value and show a *single* α_risk holds across both tracks.
> **Points (1) and (2) are non-negotiable** — without a non-circular, danger-aligned δ and a gate that
> can act on hallucinated classes, the contribution sentence describes a mechanism that, by the
> paper's own data, does the opposite of what it claims exactly where it matters most.

---

## What this changes

The stress test does **not** overturn R1 — it sharpens it. The contribution is novel and the science
is salvageable; the fixes are mostly cheap and several reuse the *existing* cross-domain run. But the
sequencing in [`ROADMAP.md`](ROADMAP.md) now has a prerequisite stage: **four decision records before
Phase 3 code** (δ/h_t operator, prior-vs-gate safety, knob values, schedule de-risk). The reviewer-2
non-negotiables (define δ; fix gate keying) map onto R1 and #7 and must be resolved in the spec before
the permission field is built.

**Recommended immediate move:** open the four DRs (especially the δ/h_t operator), patch the
ITTA naming + add the 5 citations to the scan, then start Phase 3 on the corrected spec.

<sub>Run 2026-06-16 · 82 agents · 3.9M subagent tokens · 902 tool uses · 16 min wall. 28 findings, 24
panel-verified (69 OVERSTATED / 2 REAL_HOLE / 1 FALSE / 0 ALREADY_HANDLED), 9 novelty candidates. Raw
JSON: `tasks/w9o7oi65z.output`. This document records an internal audit; no external claim is asserted
as fact without the per-finding verification noted above, and newly-surfaced citations are flagged
to-verify.</sub>
