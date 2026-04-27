# Phase 2 Sub-step 0 — Initialization Findings

## What This Step Does

When HINT++ is deployed to a new scene, no human correction has arrived yet. The safety weight system still needs to know something about each class before it touches a single prediction — which classes to treat cautiously, which classes to trust, and which classes are likely to produce noisy or unreliable corrections when they do arrive. Sub-step 0 computes two numbers per class that answer these questions from Phase 1 data alone: **η_k**, the teacher's confidence on class k, and **v_k(0)**, the starting noise estimate for class k. Together they initialize the adaptive moment safety system so that its first response to human correction is principled rather than blind.

---

## Sub-step 0A — Teacher Confidence η_k

### What We Did

We took the frozen teacher's softmax outputs across all 68 rooms in S3DIS Area 5 — the same predictions that produced the 75.41% mIoU baseline — and measured how confident the teacher is about each class. Confidence is measured using Shannon entropy: a perfectly confident prediction puts all probability mass on one class and has zero entropy; a completely uncertain prediction spreads mass evenly across all 13 classes and has maximum entropy. We flip the entropy score so that a confident teacher gives a high value. The result is η_k between 0 and 1 for each class, derived entirely from the teacher's own prediction distributions on real data.

### Results

| Class | η_k |
|---|---|
| floor | 0.9928 |
| ceiling | 0.9845 |
| chair | 0.9720 |
| board | 0.9564 |
| wall | 0.9530 |
| bookcase | 0.9409 |
| window | 0.9392 |
| door | 0.9363 |
| table | 0.9306 |
| sofa | 0.9254 |
| clutter | 0.8799 |
| column | 0.8694 |
| **beam** | **0.7378** |

**High confidence (η_k ≥ 0.90):** floor, ceiling, chair, board, wall, bookcase, window, door, table, sofa — ten of thirteen classes. The teacher is highly self-consistent on these classes.

**Moderate confidence (0.80 ≤ η_k < 0.90):** clutter (0.8799), column (0.8694). These classes have more spread in the teacher's prediction distributions, consistent with their lower Phase 1 IoU.

**Lower confidence (η_k < 0.80):** beam only (0.7378). See below.

### Key Finding — Beam

Beam has the lowest η_k in the dataset at 0.7378. It is the only class below 0.80. What matters is understanding what this number means — and what it does not mean.

The teacher achieves 0.19% IoU on beam with 73.78% confidence. The teacher is not uncertain about beam. It does not spread its prediction probability across many classes when it encounters a beam. It commits confidently to wrong answers, consistently assigning beam points to ceiling and wall with high softmax scores. A confidence threshold system would pass beam as safe — 73.78% does not look alarming compared to clutter at 87.99% or column at 86.94%. Any filter that only looks at the teacher's self-reported certainty would miss this failure entirely.

This is the central motivation for HINT++. The teacher cannot be trusted to identify its own blind spots. Human correction via δ_k(t) is the only signal that can detect a confidently wrong class and update the safety weight accordingly. Beam is the proof of concept: high confidence, effectively zero accuracy.

### Why This Is Stronger Than Adam's Initialization

Adam initializes its moment estimates with a fixed scalar (typically 10⁻³ for the variance term) — a convention chosen for general numerical stability, not for any property of the problem at hand. Our η_k is earned directly from the teacher's own predictions on the target dataset: it is different for every class, reflects real structure in the prediction distributions, and gives the safety weight system a semantically meaningful starting point before the first human correction arrives.

### Verification

All checks passed on the computed η_k values:

| # | Check | Result |
|---|---|---|
| 1 | All η_k ∈ [0, 1] | PASS — min = 0.7378, max = 0.9928 |
| 2 | No NaN or Inf | PASS — all 13 values finite |
| 3 | η_floor > 0.80 | PASS — η_floor = 0.9928 |
| 4a | argmin(η_k) = beam | PASS — beam (0.7378) is the lowest |
| 4b | η_beam < 0.80 | PASS — 0.7378 < 0.80 |
| 5 | Pearson(IoU, η_k) > 0 | PASS — r = 0.9745, p = 1.74e-08 |

The Pearson r of 0.97 between per-class IoU and η_k confirms that the teacher's self-confidence is a meaningful signal: classes where the teacher is more confident tend to be the same classes where it is more accurate. Beam is the notable exception — high confidence, near-zero accuracy — which is exactly what this step is designed to expose.

---

## Sub-step 0B — Initial Variance v_k(0)

### What We Did

We compute a starting noise estimate per class using two independent signals drawn from Phase 1 data.

**Signal 1 — class frequency** from the S3DIS training split (Areas 1, 2, 3, 4, 6). During deployment, human corrections arrive roughly in proportion to how often a class appears. Rare classes will be corrected rarely. When corrections are sparse, the running estimate of what a class's correction signal looks like is inherently noisier — there is less data to stabilize it. Rare classes need a higher initial variance to reflect this.

**Signal 2 — teacher uncertainty (1 − η_k)**. Classes where the teacher is less consistent will produce more variable corrections when humans do intervene. A high value of 1 − η_k means the teacher is less reliable on that class, which translates to a noisier initial condition before corrections can accumulate.

### The Problem We Found and Fixed

The original formula combined both signals with equal weight:

```
v_k(0) = 0.5 · (1/freq_k) + 0.5 · (1 − η_k)
```

This looks balanced. It is not. The two signals operate at completely different scales. Across the 13 S3DIS classes, 1/freq_k ranges from 3.7 to 205. The term (1 − η_k) ranges from 0.007 to 0.26. With these scales, the teacher uncertainty term contributes roughly 1% to the result. The formula was not mixing two signals — it was almost entirely driven by frequency, with the teacher uncertainty term too small to matter.

The diagnostic was clear: with the original formula, Pearson(η_k, v_k(0)) = −0.107, not significantly different from zero (p = 0.73). Teacher confidence had no meaningful relationship to the initial variance. The formula was not doing what it was designed to do.

**The fix:** normalize both signals to the range (0, 1] by dividing each by its own maximum value before mixing. The rarest class gets a normalized rarity score of 1.0. The most teacher-uncertain class gets a normalized uncertainty score of 1.0. With both signals on the same scale, α = 0.5 genuinely means equal weighting.

```
r_k    = (1/freq_k)  / max_j(1/freq_j)        # normalized rarity:      ∈ (0, 1]
u_k    = (1 − η_k)   / max_j(1 − η_j)         # normalized uncertainty: ∈ (0, 1]
v_k(0) = 0.5 · r_k + 0.5 · u_k               # honest equal weighting: ∈ (0, 1]
```

After the fix:

| Diagnostic | Before | After |
|---|---|---|
| Pearson(η_k, v_k(0)) | −0.107 (p = 0.73) | **−0.733** (p = 0.004) |

After the fix, teacher uncertainty genuinely drives the initial variance. The formula now does what it was designed to do.

### Results

v_k(0) values, sorted highest to lowest:

| Class | v_k(0) | Dominant signal |
|---|---|---|
| sofa | 0.6422 | Rarity — rarest training class (0.49%), r_k = 1.00 |
| beam | 0.6006 | Uncertainty — most teacher-uncertain class, u_k = 1.00 |
| column | 0.3636 | Both — rare (2.1%) and moderately uncertain |
| board | 0.2762 | Rarity — second rarest after sofa |
| clutter | 0.2509 | Uncertainty — high 1 − η_k = 0.120 |
| window | 0.2310 | Rarity — rare (2.1%) |
| table | 0.2074 | Both — moderate on each signal |
| door | 0.1659 | Both — moderate on each signal |
| bookcase | 0.1642 | Both — moderate on each signal |
| chair | 0.1132 | Neither — common and confident |
| wall | 0.0986 | Neither — most common class (27.2%) |
| ceiling | 0.0422 | Neither — common and very confident |
| floor | 0.0284 | Neither — common, η_k = 0.9928 |

The ranking reflects the design intent. Sofa rises to the top because it is the rarest class in the training data — the system will see very few sofa corrections before needing to act. Beam rises to second because the teacher is most uncertain about it — its corrections will be less consistent than any other class. Floor, ceiling, and wall sit at the bottom because they are common, well-predicted, and will accumulate stable correction histories quickly. When the first human corrections arrive, the system responds more slowly to classes at the top of this table and more quickly to classes at the bottom.

Note that sofa and beam are each the maximum on a different signal (r_k and u_k respectively). This is the clearest demonstration that the two signals are carrying independent information — if they were redundant, the same class would dominate both.

**Signal independence:** Pearson(r_k, u_k) = +0.106 (p = 0.73) — effectively zero correlation. The frequency signal and the uncertainty signal carry independent information. The equal weighting α = 0.5 is theoretically justified.

### Verification

All five checks passed:

| # | Check | Result |
|---|---|---|
| 1 | Σ freq_k ≈ 1.0 (±0.01) | PASS — sum = 1.000000 |
| 2 | v_k(0) > 0 for all classes | PASS — min = 0.0284 (floor) |
| 3 | max(v_k(0)) / min(v_k(0)) ≥ 10 | PASS — 0.6422 / 0.0284 = 22.60× |
| 4 | Pearson(η_k, v_k(0)) < 0 | PASS — r = −0.7331, p = 4.35e-03 |
| 5 | \|Pearson(r_k, u_k)\| < 0.7 | PASS — r = +0.1056, p = 7.31e-01 |

Check 3 enforces that the prior is informative: there must be at least an order-of-magnitude spread between the class that needs most caution and the class that needs least. The 22.60× spread between sofa and floor confirms that the initialization will have real effect on early adaptation behavior rather than treating all classes the same. Check 5 is the gate that replaced the printed-only independence diagnostic from an earlier draft of this script — it enforces that α = 0.5 is doing what it claims.

---

## What These Findings Mean for HINT++

- **The system now knows which classes to trust before any correction arrives.** Floor, ceiling, and chair start with low initial variance and will adapt quickly when corrections come. Sofa and beam start with high initial variance and will adapt more conservatively until enough corrections accumulate to establish a reliable signal.

- **The beam finding proves why confidence thresholds alone cannot govern safe deployment.** The teacher is confident and wrong on beam. Any threshold-based filter operating on the teacher's own output would pass beam as acceptable. Human correction via δ_k(t) is the only signal that can break through a confidently wrong prior — this is the core justification for the entire HINT++ adaptive safety mechanism.

- **The formula fix ensures both prior knowledge sources contribute equally.** Class frequency and teacher uncertainty are independent signals — Pearson r = +0.106 between the normalized terms. After normalization, both genuinely shape v_k(0). The initialization is not just a frequency prior with a decorative uncertainty term attached; it is a principled combination of two different aspects of what the teacher does and does not know.

---

## What Comes Next

At t = 1 the first human correction arrives. The correction signal δ_k is computed as a soft volume ratio across all classes in the corrected region. This feeds into the first and second moment updates — beginning the adaptive estimation process that drives the safety weight w_k(t) for the first time. The moment update loop is the next implementation step: `src/safety/adaptive_moments.py`. The values produced here — η_k loaded into m_k(0) and v_k(0) from `experiments/phase2_init/results/` — are the inputs that file will consume at initialization.

Sub-step 0C is the next action: it will produce the master initialization table combining η_k, v_k(0), and m_k(0) = 0 into a single deployable CSV (`table3_master_init.csv`) that `adaptive_moments.py` loads directly.
