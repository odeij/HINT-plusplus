---
name: paper-section
description: Draft or revise any .tex section of the HINT++ CVPR 2027 paper. Use for introduction, related work, method, experiments, conclusion, rebuttals, or abstract. Triggers on paper, draft, section, LaTeX, writing, rebuttal.
model: opus
---

# Paper Sections — R1 framing

Spec: memo §1 (`docs/HINTpp_Design_Memo_R1_2026-06-11.md`). Check `docs/objections/ledger.md` before
drafting — every section should pre-empt its standing objections.

## Contribution sentence (exact, never deviate)
"HINT++ is the first interactive TTA method in which corrections maintain a longitudinal per-class
trust state that spatially gates parameter updates, with anytime-valid risk control — enabling safe
deployment to unseen domains without per-domain tuning."

## Method section order — two loops, not seven phases
1. Problem setup: streams, clicks, frozen teacher, zero target tuning.
2. **Inner loop** (inherited from HINT-3D, credited as such): click → training-free region →
   gated LoRA step (CE on region + λ_stab·KL on high-confidence anchors; zero-init rank-4 LoRA).
3. **Outer loop** (the contribution): correction outcome δ → trust posterior (λ-mixture estimator)
   → signed safety weight w → permission field P_raw = σ(αw) → risk-controlled gate (hysteresis +
   anytime-valid monitor) → governs the inner loop.
4. Formal statements: Prop 1 (zero-correction identity), Prop 2 (Pinsker drift bound),
   Thm 1 (anytime-valid risk control) — assumptions stated next to each.
Every equation numbered and referenced; Algorithm 1 shows both loops.

## Related work — five clusters (position against each, in this order)
1. **Reliable / safe TTA** (degradation-aware, reset-based, conservative updates).
2. **Human-in-the-loop TTA** — ITTA, Latte++ (prompts refine predictions), HILTTA (labels select
   hyperparameters), PinPoint3D (interactive 3D masks). Distinguish by what the human signal is USED
   FOR; none maintain longitudinal trust that gates parameter updates under a risk certificate.
3. **3D TTA / domain adaptation** (GIPSO, HGL, point-cloud TTA).
4. **Human-gated autonomy / shared control** (operator-governed learning systems).
5. **Risk monitoring + conformal / anytime-valid inference** (confidence sequences, e-values).

## Framing — ALWAYS / NEVER
ALWAYS: "safe deployment without per-domain tuning" · "graceful degradation under corrupted
corrections" · "empirical harmful-update rate respects the declared budget α_risk" · "lowest
regression-event rate at matched click budget" · "within N points of the oracle-tuned ceiling".
NEVER: "state-of-the-art" · "significantly outperforms" · "X% improvement over tuned baseline" ·
benchmark-dominance framing of any kind · the retired terms "monotone safety check" / running max ·
the project's grant program, sponsors, host city, or application context (Critical Rules) ·
meta-learning language (pre-R1 contribution — retired).

## Claims-traceability rule
Every claim traces to a table, figure, or one of Prop 1 / Prop 2 / Thm 1. While drafting, annotate
each claim with `% trace: <table/fig/stmt>`; a claim with no trace is cut or hedged. Numbers are
script-generated from `experiments/results/` — never typed by hand (including in the abstract).

## Section notes
- **Intro:** open with the deployment problem (per-domain tuning makes interactive TTA undeployable);
  three demonstrations (safety without tuning / reliability from imperfect humans / certificate);
  contribution sentence verbatim; contributions list maps 1:1 to experiments.
- **Experiments:** E1 primary two-track table FIRST; then E2 noise sweep, E3 burst (gate-closure
  trace figure), E4 persistence, E5 ablations. Include failure analysis (ceiling/beam/column/board
  overconfident-wrong transfer) and the per-stream worst case, not just means.
- **Conclusion:** restate contribution, honest limitations (single-click protocol, simulator-based
  corrections, indoor-primary), future work (learned maskers as deployment demo only).

## LaTeX macros (R1)
```latex
\newcommand{\dk}{\delta_k(t)}        \newcommand{\wk}{w_k}
\newcommand{\Praw}{P_{\mathrm{raw},k}} \newcommand{\Gk}{G_k}
\newcommand{\arisk}{\alpha_{\mathrm{risk}}} \newcommand{\method}{HINT\texttt{++}}
```
