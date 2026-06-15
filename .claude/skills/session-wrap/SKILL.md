---
name: session-wrap
description: End-of-session ritual — verify tests via subagent, write a ≤10-line summary, distill 0–3 lesson candidates against the acceptance criteria, update LESSONS.md and the changelog. Use at the end of every working session, before the final commit is pushed.
model: sonnet
---

# Session Wrap — end-of-session ritual

Run this before closing any session that changed code or docs. Order matters.

## 1. Verify (via subagent, not inline)
Dispatch the `test-runner` agent (`.claude/agents/test-runner.md`). It returns failures only.
- ALL GREEN → proceed. Any failure → the session is not over; fix or explicitly hand off the
  failure in the summary as the next session's first task (never wrap with silent reds).
- If safety-relevant files changed this session and the `safety-auditor` agent has not run on the
  final diff, run it now (its findings block the wrap the same way test failures do).

## 2. Summary (≤10 lines, in the final session message and `docs/changelog.md` if week's end)
- What changed (files/modules, one line each, with commit SHAs).
- What was verified (test counts, audits run).
- What is explicitly NOT done / handed off, with the exact next step.
- Any DR opened/closed; any registry rows added.

## 3. Lessons (0–3 candidates, most sessions produce 0–1)
Distill candidate lessons from anything that surprised, broke, or got reworked this session.
Score each against ALL FOUR acceptance criteria (from `docs/LESSONS.md` header):
1. **Generalizes** beyond the incident that produced it.
2. **Actionable** as a concrete check or rule (someone can DO something with it).
3. **Not derivable** from CLAUDE.md, the memo, or an existing skill/lesson (no duplicates —
   if it sharpens an existing lesson, edit that lesson instead).
4. **Trigger-stated** — names the situation in which to apply it.
Accept only candidates passing all four; append to `docs/LESSONS.md` as
`- **L<next> — <one-line rule>.** (origin) <2–4 sentence body> Trigger: <when>. Check: <what to do>.`
Cap is 30: if full, prune the least load-bearing lesson in the same commit (say which and why).

## 4. Bookkeeping
- Update the status snapshots that changed: `phase-implement` SKILL (phase status),
  `experiment-run` SKILL (milestone table), CLAUDE.md Status block.
- `docs/changelog.md`: prepend the week line if this is the week's last session.
- Registry: any experiment launched this session has its row (status updated if it finished).
- Open DRs: flag any due within 7 days in the summary.

## 5. Hand-off prompt (when the next session's task is known)
End by OUTPUTTING (not executing) a work-order prompt for the next session: the task, the memo
sections that govern it, the REAL file paths and interfaces it touches, its worked checks, and its
stop conditions. Pre-fill with discovered reality, never with assumed structure.

## Anti-patterns
- Wrapping with failing tests "to be fixed next time" without naming them in the summary.
- Lesson inflation: process notes ("we ran tests") are not lessons; neither are spec restatements.
- Editing LESSONS.md outside this ritual (drive-by lessons skip the acceptance criteria).
