---
name: coding-discipline
description: Behavioral guardrails for writing code in the HINT++ repo. Bias toward caution, simplicity, and surgical changes over speed and cleverness. Use whenever modifying source under Pointcept/, src/safety/, src/memory/, src/adaptation/, or experiments/. For trivial edits (typo, rename, import sort) skim only the relevant section.
---

# HINT++ Coding Discipline

These rules reduce the most common LLM coding mistakes — overconfident assumptions, speculative abstractions, scope creep, and "make it work" loops without verification. They are biased toward caution; for trivial tasks use judgment, but never skip the project rules from `CLAUDE.md` (no DynaCITY, ≥3 seeds, pytest before commit, safety failures are blocking).

---

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing anything in this repo:

- **State assumptions explicitly.** If you are about to use Adam defaults `(β₁=0.9, β₂=0.999)` say so out loud — HINT++ is asymmetric `β₁ < β₂` by design (e.g. `0.7, 0.95`). If you are unsure whether a tensor is on CPU or GPU, whether voxelized or raw, whether `xyz` or `xyz+rgb` — ask, don't guess.
- **If multiple interpretations exist, present them.** "Implement the safety check" can mean a runtime assertion, a property test in `tests/`, or a wrapper module. List the options, recommend one, wait.
- **If a simpler approach exists, say so.** Push back when warranted. A 30-line `nn.Module` that subclasses `torch.optim.Adam` is not better than 10 lines of explicit moment update if the explicit version makes the bias-correction term legible.
- **If something is unclear, stop.** Name what's confusing. Ask. The cost of one clarifying question is far less than re-deriving the math after you've written 200 lines that quietly transposed `m̂ₖ` and `v̂ₖ`.

Repo-specific things you should always confirm before writing:

- **Which structure are we in?** `CLAUDE.md` describes a planned `src/safety/`, `src/memory/`, `src/adaptation/` layout. The actual repo today is `Pointcept/` + `flash-attention/` + `experiments/phase1_baseline/`. If the user says "add Phase 2," confirm whether that means scaffolding a new `src/` tree or extending Pointcept.
- **Which phase boundary?** Phase 1 (frozen teacher) is complete. Phases 2–7 are spec only. Don't import from a module that doesn't exist yet — scaffold it explicitly.
- **Which dataset?** Training is S3DIS. Evaluation targets (ScanNet, SemanticKITTI, nuScenes) are zero-shot — never tune on them.

---

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- **No features beyond what was asked.** If the task is "add bias correction," do not also add a learning-rate schedule, a logger, or a `__repr__` method.
- **No abstractions for single-use code.** A `BaseSafetyMixin` with one subclass is worse than two functions. Three similar lines beat a premature class hierarchy.
- **No "flexibility" or "configurability" that wasn't requested.** Hydra configs already exist for hyperparameters — don't add a `mode` argument with seven enum values "in case we need it later."
- **No error handling for impossible scenarios.** Trust internal calls and PyTorch guarantees. Validate only at boundaries: dataset loading, user/CLI input, external checkpoints. A `try/except` around `tensor.sum()` is noise.

Concrete tells: if a Phase 2 module hits 200 lines, ask whether it should be 50. A senior engineer reading `AdaptiveMomentSafety` should see the four-line correspondence to Adam (`δ → m̂ → v̂ → weight`) without scrolling. If they can't, rewrite.

What the project DOES require, that may look like over-engineering but isn't:

- Numerical stability assertions on safety-critical code (`assert eps > 0`, `torch.isnan` checks). Keep these.
- Type hints on all signatures, docstrings on public methods. Keep these.
- Tests with synthetic point clouds for every new module. Keep these.

---

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- **Don't "improve" adjacent code, comments, or formatting.** If you are fixing one bias-correction term in `adaptive_moments.py`, do not also refactor the constructor, rename a private attribute, or reformat a docstring.
- **Don't refactor things that aren't broken.** Pointcept's config registry, builder pattern, and DDP launcher work. Leave them.
- **Match existing style.** Pointcept uses snake_case modules, registry-based factories, and Hydra-style configs. FlashAttention v4 uses CuTeDSL and JIT caches. Don't impose a different style on either.
- **Mention orphans, don't delete them.** If you spot pre-existing dead code while making your change, say so and move on. Removing it is a separate, explicit task.

When your changes create orphans:

- Remove imports / variables / functions that **your** changes made unused.
- Don't remove pre-existing dead code unless asked. The test: every changed line should trace directly to the user's request.

Repo-specific care:

- **Never modify `flash-attention/` upstream files** unless explicitly asked. It is a submodule of an external project; changes there create merge pain.
- **Don't touch `Pointcept/` core engines** (`engines/train.py`, `engines/test.py`) without saying so first. They are shared infrastructure.
- **Don't edit anything under `checkpoints/`, `datasets/`, or `s3dis-compressed/`** — those are data, not code.
- **Generated artifacts** (`experiments/results/{exp_id}/`, paper figures, `.npy` predictions) are written by scripts, not edited by hand.

---

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform vague tasks into verifiable goals before writing code:

| Vague | Verifiable |
|---|---|
| "Add validation" | "Write tests for invalid inputs (NaN δₖ, β₁ ≥ β₂, eps ≤ 0), then make them pass" |
| "Fix the bug" | "Write a test that reproduces the NaN in the bias-correction term, then make it pass" |
| "Refactor X" | "Ensure `pytest tests/test_X.py` passes before and after, with the same outputs on a fixed seed" |
| "Implement Phase 3" | "`PermissionField(nn.Module)` exists, P(x) ∈ [0,1] verified by property test, gradient flows, monotonicity test passes" |

For multi-step tasks, state the plan before executing:

```
1. [step]   → verify: [check]
2. [step]   → verify: [check]
3. [step]   → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") force the user to clarify after every diff.

Project-specific verification gates that are non-negotiable:

- `pytest tests/` passes before any commit. Safety tests are BLOCKING.
- All experiments report mean ± std over **≥3 seeds**. A single-seed result is not a result.
- Numerical sanity: no NaN in forward pass on synthetic data, no division by zero, ε guards present.
- For Phase 7 / primary experiment: violations <15% on **all three** target domains, within 3% mIoU of per-domain tuning, AND HINT-3D source thresholds show >40% violations on the same domains.

---

## 5. HINT++-Specific Pitfalls

Things that have specifically tripped up implementations of this method or similar ones. Check yourself against these:

- **β₁ vs β₂ swap.** Using Adam defaults (`0.9, 0.999`) silently destroys the asymmetric-smoothing argument. Always pass them explicitly, always assert `β₁ < β₂` in `__init__`.
- **Bias correction off-by-one.** `t` starts at 1, not 0. `m̂ = m / (1 - β₁^t)` blows up if `t=0`. Add a unit test for `t=1` and `t=10000`.
- **Permission field saturation.** A naive sigmoid can produce gradients on the order of `1e-30`. If meta-learning isn't moving, check `P(x)` distribution — if it's piled at 0 or 1, the gating has saturated.
- **Storing tensors in exemplar memory.** Phase 5 stores **sufficient statistics** (mean, variance, count) per correction event. If you find yourself appending point-cloud tensors to a list, stop.
- **Target-domain leakage.** ScanNet / SemanticKITTI / nuScenes are zero-shot. Never train on them, never compute target stats and feed them back, never tune hyperparameters on their dev sets.
- **Forbidden references.** Never put DynaCITY, Beirut, heritage, or funding mentions in any code, comment, docstring, or paper text. This includes commit messages.
- **Confidence ≠ correctness.** The Phase 1 baseline shows beam = 0.19% IoU but 73.78% confidence. Any "uncertainty-based" filter you write should be tested against this case before you trust it.

---

## 6. When to Use This Skill vs Others

- **Use this skill** for general edits, bug fixes, refactors, scaffolding — any time you are about to write or change code in this repo and want a sanity gate.
- **Use `phase-implement`** when starting a new HINT++ phase module (Phase 2–7).
- **Use `safety-check`** after any change to `src/safety/` or to moment / permission / monotone code.
- **Use `debug-phase`** when something is broken — NaN, divergence, failing test, unexpected result.
- **Use `experiment-run`** when configuring or launching experiments.
- **Use `paper-section`** when drafting `.tex` for the CVPR submission.

These guidelines are working if: fewer unnecessary lines in diffs, fewer rewrites due to overcomplication, and clarifying questions arrive **before** implementation rather than as apologies after.
