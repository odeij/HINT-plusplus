---
name: research-log
description: Maintain the research log — decision records, experiment registry, objection ledger, weekly gate reviews, changelog. Use when deviating from the R1 memo, registering an experiment, logging a reviewer objection, or running the Monday gate review.
model: sonnet
---

# Research Log — formats and rituals

The log lives in `docs/`. Rule one (R1, Critical Rules): any deviation from the memo gets a decision
record BEFORE code. Rule two: if it isn't in the registry, the experiment didn't happen.

## Decision records — `docs/decisions/DR-XXXX-<slug>.md`
Sequential IDs, never reused. Statuses: `PENDING` (with **Due** date) → `accepted` / `rejected`;
later reversals get a new DR that marks the old one `superseded by DR-YYYY`.
```markdown
# DR-XXXX: <Title>
**Status:** accepted | rejected | PENDING | superseded by DR-YYYY
**Date:** YYYY-MM-DD · **Due:** YYYY-MM-DD (if PENDING) · **Deciders:** <names>
## Context      — why a decision is needed; link the memo section it touches
## Options      — numbered, each with its main trade-off
## Decision     — one paragraph; empty until decided
## Consequences — what changes in code/docs/experiments; the commits that implement it
```
Open DRs: check `grep -l PENDING docs/decisions/` at every session start; overdue DRs are the first
agenda item of the gate review. Currently pending: DR-0002 (outdoor source dataset, due 2026-06-19).

## Experiment registry — `docs/experiments/registry.md`
Append one row per launch (including failed runs and baseline repro runs):
`| EXP-YYYYMMDD-NN | date | experiment | branch @ SHA | config hash | seeds | status | headline | artefacts |`
Statuses: `running` / `done` / `failed` / `descoped`. The row links to
`experiments/results/{exp_id}/`; numbers in the row come from the MetricsJSON, typed once, verbatim.

## Objection ledger — `docs/objections/ledger.md`
One entry per standing reviewer objection: the objection AT FULL STRENGTH, the current rebuttal, and
**Evidence needed** (the table/figure that must exist by submission). Update the rebuttal when
evidence lands. New objections (from supervisors, reading groups, related-work updates) get appended
with the next OBJ-N id. An objection whose evidence is missing at the Oct 24 freeze is a red risk.

## Weekly gate review — Mondays, `docs/reviews/YYYY-MM-DD-gate-review.md`
```markdown
# Gate review YYYY-MM-DD (week N to <next gate>)
**Next gate:** <G1 Jul 20 | G2 Aug 3 | descope Jul 24 / Aug 7 | freeze Oct 17 / Oct 24>
**On track?** GREEN / AMBER / RED — one sentence why.
## Since last review   — shipped work, registry rows added, DRs closed
## Gate burndown       — what must be true at the gate; what's missing; days of buffer left
## Risks               — top 3, each with the action that retires it
## Decisions needed    — overdue/expiring DRs, descope calls
## Next week           — the ONE thing that must land, then the rest
```
Slips eat buffers, never gates: a slipped gate is a descope decision (DR), not a date move.

## Changelog — `docs/changelog.md`
One line per week, prepended: `YYYY-MM-DD — <what actually changed>`. Written at session-wrap of the
last session that week.

## Hygiene
- Convert relative dates ("next Friday") to absolute dates in every log entry.
- Link artefacts by path, never paste numbers from memory.
- The log is append-mostly: edit only to update statuses/rebuttals, never to rewrite history.
