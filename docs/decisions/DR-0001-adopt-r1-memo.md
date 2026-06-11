# DR-0001: Adopt the R1 design memo as project source of truth

**Status:** accepted · **Date:** 2026-06-11 · **Deciders:** Odei (with supervisor sign-off at G0)

## Context

The pre-R1 pipeline had eight diagnosed flaws (unimplementable outdoor protocol, a liveness-only
running-max gate, incoherent estimator bias correction, undefined δ semantics, backwards β
rationale, a circular headline metric, stale positioning vs Latte++/HILTTA, and target-domain
leakage through learned maskers). The R1 memo resolves each one:
[HINTpp_Design_Memo_R1_2026-06-11.md](../HINTpp_Design_Memo_R1_2026-06-11.md), §2 (diagnosis) and
§3–§11 (canonical spec).

## Options

1. Adopt R1 wholesale (memo as single source of truth; all docs, skills, code refactored to match).
2. Patch the old spec incrementally, keeping the seven-phase presentation and running-max gate.

## Decision

The repo adopts the R1 spec wholesale. The running max P_safe = max(P_safe, P_raw) and the term
"monotone safety check" are retired everywhere. All future deviations from the memo require a
decision record in this directory BEFORE code.

## Consequences

- The R1 bootstrap commit: CLAUDE.md and all skills rewritten to R1, research-log scaffold
  (decisions, experiments registry, objections ledger, reviews, changelog, LESSONS), feedback loop
  (test-runner and safety-auditor subagents, PostToolUse test hook).
- The Phase 2 estimator refactor (pseudo-count prior mixture, outcome δ, signed w) follows
  immediately on `feat/phase2-r1`.
- Phase 4 is renamed "Risk-Controlled Permission Gate"; its old open-issue about running-max point
  identity is moot, but stable per-point identity remains a requirement for the spatial permission
  field G(x).
