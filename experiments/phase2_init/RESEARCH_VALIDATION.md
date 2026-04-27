# Research Validation — Pre-Phase 3

This document records four research-level verification checks performed against the existing HINT++ codebase before Phase 3 (Permission Field) begins. Performed: 2026-04-27.

## Check 1: Contribution Claim Traceability

**Claim under audit:**
> "We treat per-class safety sensitivity as a quantity estimated from human correction history using adaptive moment statistics, yielding safety weights that are responsive to consistent feedback and conservative under sparse or noisy corrections."

**Component-by-component trace:**

| Claim component | Code location | Mechanism |
|---|---|---|
| "per-class safety sensitivity as a quantity estimated" | `src/safety/adaptive_moments.py` `m_k`, `v_k` buffers (registered shape `(num_classes,) = (13,)`) | One scalar per class for each moment; nothing global, nothing collapsed. |
| "from human correction history" | `forward(delta: Tensor)` consumes a per-class correction signal δ_k(t) at each call | Module is stateful — `m_k`, `v_k`, `t` accumulate across calls. The "history" is the sequence of forward calls. |
| "using adaptive moment statistics" | Lines computing `m_k.mul_(β₁).add_(δ, α=1-β₁)` and `v_k.mul_(β₂).addcmul_(δ, δ, value=1-β₂)`, then bias correction `/(1-β^t)` | Direct Adam-style EMA + bias correction. β₁<β₂ asymmetric by design. |
| "responsive to consistent feedback" | m̂_k captures consistent direction. When δ has consistent sign, m̂_k grows; when it flips, m̂_k contracts. | Empirically verified by Check 2 — floor in dense regime climbs to mean w_k ≈ 0.82 over t ≥ 50. |
| "conservative under sparse corrections" | δ_k = 0 → m_k stays at 0 (β₁·0 + (1-β₁)·0 = 0) → m̂_k = 0 → w_k = 0 by construction | Empirically verified — uncorrected classes returned EXACTLY 0.0 across all 100 timesteps in B and C. Not "below 0.15" — **identically zero**. |
| "conservative under noisy corrections" | Noisy δ inflates v̂_k (sum of δ²); √v̂_k in the denominator suppresses w_k | Empirically: sofa (freq 0.005, noise σ=0.1 → SNR ≈ 1:20) gives mean w_k = -0.067 with std 0.26 over t ≥ 50 — the noise dominates and w_k oscillates around zero. The estimator does not commit to a direction it cannot resolve. |
| `η_k` per-class ceiling | Loaded from `phase2_init.pt`, registered as immutable buffer | Per-class teacher-confidence ceiling derived from Phase 1 entropy; multiplies w_k element-wise. |

**Critical question — what makes w_k a SAFETY weight rather than just an adaptive weight?**

Honest answer: **nothing in the current code does**. The math implemented in `AdaptiveMomentSafety` is exactly Adam — a per-parameter (here per-class) adaptive step-size. Three things give it "safety" framing today, none of which are mathematical guarantees:

1. **Intent.** The buffers are named `safety` and the docstring describes the eventual use. Naming is not a guarantee.
2. **The v_k(0) prior** (Phase 2 Sub-step 0B) was deliberately constructed to bias rare-and-uncertain classes toward higher initial variance — i.e., toward smaller initial w_k. This is a design choice that *encodes* safety priors into the initialization, but the math operating on it is generic.
3. **The η_k per-class ceiling** caps the per-class scale of w_k by the teacher's confidence on that class. This is also a prior, not a runtime guarantee.

The actual safety semantics — w_k controlling whether and how strongly the model is permitted to deviate from the frozen teacher — does not exist yet in the codebase. **It will be built in Phase 3.**

### Phase 3 Entry Criterion (the contract Phase 3 must satisfy)

Phase 3 lifts the per-class scalar w_k(t) ∈ ℝ^13 to a per-point spatial permission field P(x, t) ∈ [0, 1] over all points x in the scene. The mathematical contract is: for each point x, let k(x) = argmax_c π_teacher(x)_c be the teacher's predicted class at x. Then `P(x, t) = σ(α · w_{k(x)}(t) + β)` where σ is the logistic function and α > 0, β ∈ ℝ are scalar hyperparameters fixed at construction. The required preserved properties are (i) **boundedness**: P(x, t) ∈ [0, 1] for all x, t (forced by σ); (ii) **monotonicity in w_k**: ∂P(x, t) / ∂w_{k(x)}(t) > 0 (forced by σ being strictly increasing and α > 0) — higher safety weight on the predicted class at x translates to more permission to deviate from the teacher at x; (iii) **differentiability**: P(x, t) is differentiable in w_k(t) so meta-learning (Phase 6) can backpropagate through the gating; (iv) **per-class shared scale**: every point sharing class k receives the same scalar argument α·w_k + β to σ, so adaptation behavior on a class is jointly governed (no per-point free parameters that could be tuned to individual points). Any monotone transformation other than logistic (e.g., clipped ReLU, smooth-max) is permissible *only if* all four properties remain provable. Phase 3 must include a property test that exercises (i)–(iii) on a synthetic w_k sweep across the realistic range observed in Check 2 below.

---

## Check 2: Sparse Correction Behavior

Three synthetic experiments were run by driving the existing `AdaptiveMomentSafety` module (no new code; the existing forward loop). For each experiment, δ_k(t) = `freq_k + N(0, σ=0.1)` for active classes, δ_k(t) = 0 for inactive classes; n_steps = 100; seed = 42 per experiment. `freq_k` is the per-class S3DIS training-split frequency loaded from `experiments/phase2_init/results/freq_k.json`.

### Experiment A — Dense regime (all 13 classes corrected)

w_k(t) at key timesteps:

| class | t=1 | t=10 | t=50 | t=100 | active? |
|---|---:|---:|---:|---:|:---:|
| floor | 0.2340 | 0.6688 | 0.7477 | 1.1776 | Y |
| beam | 0.0103 | 0.0135 | 0.3778 | 0.3620 | Y |
| sofa | 0.0084 | -0.0066 | -0.0871 | -0.0670 | Y |

Uncorrected check: **N/A (all classes active in A)**

Convergence/stability over t ∈ [50, 100], all 13 active classes:

| class | mean (t≥50) | std (t≥50) | CV |
|---|---:|---:|---:|
| ceiling | 0.8686 | 0.1594 | 0.184 |
| floor | 0.8215 | 0.1758 | 0.214 |
| wall | 0.8657 | 0.0755 | 0.087 |
| beam | 0.1600 | 0.2034 | 1.271 |
| column | 0.1194 | 0.2108 | 1.766 |
| window | 0.1920 | 0.3239 | 1.687 |
| door | 0.5021 | 0.1945 | 0.387 |
| table | 0.2404 | 0.2381 | 0.990 |
| chair | 0.4954 | 0.2859 | 0.577 |
| sofa | -0.0671 | 0.2622 | 3.910 |
| bookcase | 0.4628 | 0.2798 | 0.605 |
| board | 0.1451 | 0.2532 | 1.745 |
| clutter | 0.5025 | 0.1599 | 0.318 |

### Experiment B — Sparse regime (3 classes corrected: floor, beam, clutter)

w_k(t) at key timesteps:

| class | t=1 | t=10 | t=50 | t=100 | active? |
|---|---:|---:|---:|---:|:---:|
| floor | 0.2595 | 0.6075 | 0.8936 | 0.9435 | Y |
| beam | 0.0081 | 0.0082 | -0.0292 | 0.1125 | Y |
| sofa | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |
| ceiling | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |

Uncorrected classes max |w_k(t)| over **all** t ∈ [1, 100]: **0.000000**
All uncorrected |w_k| < 0.15 at all t: **True (vacuously — value is exactly 0)**

Convergence/stability over t ∈ [50, 100], 3 active classes:

| class | mean (t≥50) | std (t≥50) | CV |
|---|---:|---:|---:|
| floor | 0.8286 | 0.2148 | 0.259 |
| beam | 0.1060 | 0.1411 | 1.331 |
| clutter | 0.4919 | 0.1707 | 0.347 |

### Experiment C — Extreme sparse regime (2 classes corrected: beam, clutter)

w_k(t) at key timesteps:

| class | t=1 | t=10 | t=50 | t=100 | active? |
|---|---:|---:|---:|---:|:---:|
| floor | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |
| beam | 0.0126 | 0.0296 | 0.3048 | 0.1742 | Y |
| sofa | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |
| ceiling | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N |

Uncorrected classes max |w_k(t)| over **all** t ∈ [1, 100]: **0.000000**
All uncorrected |w_k| < 0.15 at all t: **True (vacuously — value is exactly 0)**

Convergence/stability over t ∈ [50, 100], 2 active classes:

| class | mean (t≥50) | std (t≥50) | CV |
|---|---:|---:|---:|
| beam | 0.0737 | 0.1599 | 2.170 |
| clutter | 0.5570 | 0.1840 | 0.330 |

### Reporting note (no interpretation)

Two facts about the design surface in the numbers above without commentary:
- For inactive classes, w_k(t) is **identically zero at every timestep**, not "approximately zero" or "below 0.15". The cold-start property m_k(0) = 0 combined with δ_k = 0 input keeps m_k = 0 forever, so m̂_k = 0 / (1-β₁^t) = 0 exactly.
- CV varies by ~40× across active classes in A (wall: 0.087 to sofa: 3.910). This is the headline number; you assess whether the spread matches the research plan.

---

## Check 3: Monotone Invariant Specification

**Q1. What is the exact monotone invariant HINT++ guarantees?**

Formal property: for every spatial location x and every timestep t ≥ 1,

```
P(x, t) ≥ P(x, t-1)
```

This is a **one-way ratchet on permission**. Once the system has granted permission level p at point x, no subsequent correction can revoke it. The "safety never regresses" claim is interpreted under the convention that P encodes *permission to deviate from the frozen teacher* — higher P means more freedom to adapt, lower P means held closer to teacher behavior. Under this invariant, the bound `KL(π_t(x) || π_teacher(x)) ≤ g(P(x, t))` for any monotone g translates to a non-decreasing KL budget: **once we have certified the deployed model is allowed to deviate by amount d at x, that certification cannot be retracted**. This is the dual of the "KL budget can only tighten" reading — they are the same property under different sign conventions; the form above is the one to implement because it composes cleanly with stochastic moment estimators (the running max of a fluctuating signal is well-defined; "tightening" is not a single-valued operation).

This invariant is **stronger** than the conditional form in `phase-implement` skill (`P(x, t+1) ≥ P(x, t) when safety evidence increases`). The unconditional running-max form removes the dependence on a definition of "safety evidence" — it holds even when the underlying w_k(t) decreases. This is the specification Phase 4 must enforce.

**Q2. Does Phase 3 use w_k(t) directly to set P(x), or some monotone transformation of w_k(t)?**

**It must NOT use w_k(t) directly.** w_k(t) is provably non-monotone: m̂_k can decrease whenever the correction signal δ flips sign, and √v̂_k can grow under noisy input. Check 2 demonstrates this empirically — beam in Experiment B oscillates from 0.0081 → 0.0082 → -0.0292 → 0.1125, hitting both signs.

Phase 3 should compute a per-step **raw permission proposal** P_raw(x, t) ∈ [0, 1] from w_k(t) using the logistic transformation specified in Check 1's Phase 3 Entry Criterion. P_raw is *not* monotone in t — it inherits the non-monotonicity of w_k. This is correct and expected; the monotonicity property is enforced by Phase 4, not Phase 3.

**Q3. What interface must Phase 3 expose so that Phase 4 can enforce the monotone invariant without modifying Phase 3 internals?**

Phase 3 exposes a `PermissionField(nn.Module)` whose `forward` returns the raw, possibly-fluctuating proposal:

```python
class PermissionField(nn.Module):
    def forward(
        self,
        w_k: Tensor,                    # shape (num_classes,) — current safety weights
        teacher_logits: Tensor,         # shape (N_points, num_classes)
    ) -> Tensor:                        # shape (N_points,) in [0, 1]
        """Return raw P_raw(x, t). Non-monotone in t by design.
        Monotone projection is Phase 4's responsibility."""
```

Phase 4 wraps this with a stateful running-max projection:

```python
class MonotoneSafety(nn.Module):
    def __init__(self, field: PermissionField, num_points_max: int):
        self.field = field
        self.register_buffer("P_running_max", torch.zeros(num_points_max))

    def forward(self, w_k, teacher_logits, point_index):
        P_raw = self.field(w_k, teacher_logits)
        idx = point_index   # stable identity per scene point
        P_safe = torch.maximum(self.P_running_max[idx], P_raw)
        self.P_running_max[idx] = P_safe.detach()
        return P_safe
```

The contract is therefore: Phase 3 computes "what's the right P given current evidence"; Phase 4 enforces "but never below any P previously certified at this point". The two phases are decoupled — Phase 3 is stateless w.r.t. monotonicity; Phase 4 is stateless w.r.t. how P is computed. **Phase 3 has no awareness of Phase 4 and exposes no API specifically for it.** Phase 4 simply consumes Phase 3's output as a black box.

One open design issue this surfaces: Phase 4 needs a stable per-point identity (`point_index` above) to maintain its running max across calls. If point clouds are voxelized differently across calls, point identity is not preserved. This is a Phase 4 implementation concern but should be flagged now: Phase 3 must, alongside `P_raw`, return or accept a stable point-identity tensor (e.g., voxel hash) for Phase 4 to key its state on.

---

## Check 4: Zero-Shot Claim Verification

**Files loaded at deployment time** (the only file `AdaptiveMomentSafety.__init__` reads):

```
experiments/phase2_init/results/phase2_init.pt
```

This file contains exactly four entries: `class_names: List[str]`, `eta_k: Tensor[13]`, `v_k_0: Tensor[13]`, `m_k_0: Tensor[13]`.

**Provenance of each entry:**

| Entry | Source script | Source data | Dataset / split |
|---|---|---|---|
| `eta_k` | `experiments/phase2_init/scripts/run_0a_eta.py` | `/home/ahmad/frozen_teacher_project/repos/Pointcept/exp/sonata/semseg-sonata-s3dis/result/Area_5-*_prob.npy` (68 files) | **S3DIS Area 5 — held-out test split of SOURCE domain** |
| `v_k_0` | `run_0b_freq.py` (consumes `eta_k.json`) | `s3dis-compressed/s3dis-compressed/Area_{1,2,3,4,6}/*/segment.npy` (204 rooms) for `freq_k`, plus `eta_k` from 0A | **S3DIS Areas 1, 2, 3, 4, 6 — training split of SOURCE domain**, plus derivative of Area 5 source-domain confidence |
| `m_k_0` | `run_0c_master.py` | `np.zeros(13)` — no data dependency | None (cold-start) |
| `class_names` | `run_0c_master.py` | hard-coded canonical S3DIS 13-class list | None |

**Q1. What files are loaded at deployment time?**

Exactly one: `phase2_init.pt`. Its contents are summarized above.

**Q2. Does anything in the pipeline use target-domain data (ScanNet, SemanticKITTI, nuScenes) to compute or update η_k or freq_k?**

**No.** Verified by direct trace:
- `run_0a_eta.py` reads only `Area_5-*_prob.npy` files generated by the frozen teacher on **S3DIS Area 5**. Path is `/home/ahmad/frozen_teacher_project/.../semseg-sonata-s3dis/result/` — the model name in the path (`s3dis`) and the file naming (`Area_5-*`) confirm source.
- `run_0b_freq.py` reads only `s3dis-compressed/s3dis-compressed/Area_{1..4,6}/*/segment.npy`. All five training areas are S3DIS source.
- `run_0c_master.py` consumes only `eta_k.json` and `v_k_0.json`, both pure-source.
- `AdaptiveMomentSafety.forward` reads no files; mutates `m_k`, `v_k` solely from the δ argument supplied by the caller.
- The δ argument's origin is human correction at deployment time — that is on-target by design (it has to be, because the system has to receive corrections in order to adapt). But δ is a **signal of human intent**, not a target-domain statistic; it does not encode label distributions, voxel counts, or any property of the target dataset that would constitute leakage.

**Q3. Is `phase2_init.pt` computed once from source domain and never recomputed during deployment?**

Yes. The three `run_0*.py` scripts are run **before deployment** (offline, on the source dataset). `phase2_init.pt` is then a frozen artifact. `AdaptiveMomentSafety` opens it read-only via `torch.load` once in `__init__` and never writes it back. `eta_k` is registered as an immutable buffer (the running estimators `m_k`, `v_k`, `t` are mutable, but they evolve from δ — not from any recomputed prior).

There is also no code path that triggers re-running 0A/0B/0C as part of the deployment loop. The scripts have to be invoked manually via `python experiments/phase2_init/scripts/run_0*.py`.

**Verdict on the zero-shot claim:** **PRESERVED.** The Phase 2 initialization pipeline is fully source-domain (S3DIS Areas 1–6) and frozen at deployment. No target-domain (ScanNet, SemanticKITTI, nuScenes) data is touched at any point in the documented data flow.

One mild caveat to note for the paper, not a violation: η_k is computed from S3DIS **Area 5**, which is also the held-out test split used to report the 75.41% Phase 1 mIoU. That number is the frozen-teacher baseline (no adaptation), so η_k does not bias it. But if any future ablation reports HINT++ deployed on Area 5, the η_k computed from the same partition would constitute a mild form of source-side test-time leakage. This is irrelevant to the zero-shot deployment claim (Area 5 is not a target domain) but should be mentioned in the experimental section if Area 5 is ever used as a deployment target for sanity checks.

---

## Verdict

The implementation through Phase 2 Sub-step 0 is aligned with the research plan on **two dimensions** (claim traceability of moment-based safety estimation, and zero-shot data-flow purity) and **partially aligned** on **two dimensions** (the distinction between "adaptive weight" and "safety weight", and the monotone invariant).

What is solid: every component of the contribution claim except the "safety" semantics traces directly to code; the v_k(0) prior and η_k ceiling encode meaningful priors verified empirically; the inactive-class identity (w_k ≡ 0) is not a numerical accident but a structural property of the cold-start design; and the source-only data flow is clean.

What needs to land in Phase 3 and Phase 4 to close the gaps: (i) **Phase 3 must implement the w_k → P(x) lift specified in Check 1's Entry Criterion** — this is what turns w_k from "an adaptive scalar" into "a safety control"; (ii) **Phase 4 must enforce the unconditional running-max monotonicity specified in Check 3** — this is what makes "safety never regresses" a guarantee rather than an aspiration. Until (i) and (ii) are in place, the paper claim "yielding safety weights" is technically defensible only as "yielding adaptive weights designed to be plugged into a safety mechanism that does not yet exist". With (i) and (ii) in place, the claim becomes mathematically defensible.

A secondary observation surfaced by Check 2: in Experiment A, several active classes (sofa, beam, column, window, board) show coefficient of variation > 1 over t ∈ [50, 100] — meaning the running estimate's standard deviation exceeds its mean. This is a research finding worth investigating before Phase 6 meta-learning: the moment estimator may not converge cleanly for classes where the correction-signal SNR is poor (rare classes with σ on the order of the signal magnitude). It is not a blocker for Phase 3 — Phase 3 only needs w_k as a current value, not as a converged estimate — but it suggests the meta-learner in Phase 6 will need either a longer adaptation horizon or a regularizer that explicitly penalizes high-variance w_k regimes. **Flagging now so it is not surprising later.**
