---
name: paper-section
description: Draft or revise a section of the HINT++ CVPR 2026 paper in LaTeX. Use when writing introduction, related work, method, experiments, or conclusion sections. Triggers on mentions of paper, draft, section, LaTeX, writing.
model: opus
effort: medium
---

# HINT++ Paper Section Writer

## Core Contribution (NEVER deviate)

"We enable zero-shot safe deployment of interactive TTA to unknown domains by meta-learning transferable safety patterns from human correction history."

## LaTeX Conventions

- One file per section: `paper/sections/{section_name}.tex`
- Use `\input{sections/{name}}` in main.tex
- CVPR two-column format
- Define macros for repeated notation:
  ```latex
  \newcommand{\dk}{\delta_k(t)}       % correction signal
  \newcommand{\mhat}{\hat{m}_k}       % first moment
  \newcommand{\vhat}{\hat{v}_k}       % second moment
  \newcommand{\Px}{P(\mathbf{x})}     % permission field
  \newcommand{\method}{HINT\texttt{++}}
  ```

## Section Guidelines

### Introduction
- Open with deployment problem (2-3 weeks tuning is impractical)
- Establish fixed thresholds as domain-specific hyperparameters
- State contribution with emphasis hierarchy: zero-shot → unknown domains → transferable patterns → correction history
- NEVER mention DynaCITY, Beirut, heritage, or funding

### Related Work
- Subsections: TTA, Safe RL, Meta-Learning, 3D Domain Adaptation, Interactive Segmentation
- Position HINT++ in the gap: no existing method handles safety + transfer + zero-shot + 3D
- Cite from the ~45 paper reference library

### Method
- Start with problem formulation
- Present the Adam analogy with full mathematical correspondence
- Seven components in logical order (not phase order)
- Include the algorithm box (Algorithm 1)
- Every equation numbered and referenced

### Experiments
- Primary result FIRST: zero-shot deployment table
- Headline: 3.7× violation reduction vs source baseline
- Frame as deployment capability, NOT benchmark dominance
- Ablations in subsections
- Include failure analysis

### Conclusion
- Restate contribution (one sentence)
- Summarize key results
- Honest limitations
- Future work

## Framing Rules

ALWAYS frame as:
- "achieves safe deployment without target-domain tuning"
- "3.7× violation reduction compared to source-domain baseline"
- "comparable accuracy to exhaustive per-domain tuning"

NEVER frame as:
- "X% improvement over tuned baseline"
- "state-of-the-art results"
- "significantly outperforms"
- Any reference to DynaCITY or funding context

## Figure/Table Generation

When generating figures or tables from experiment data:
1. Source data from `experiments/results/`
2. Generate with matplotlib/pgfplots
3. Save figure source script to `paper/figures/gen_{name}.py`
4. Save compiled figure to `paper/figures/{name}.pdf`
5. Captions must be self-contained (reader understands without body text)
