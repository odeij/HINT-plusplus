# HINT++ Research System: Complete Architecture for CVPR 2026

## 0. Reality Check: Mythos and What You Can Actually Use

**Mythos is not available to you.** Claude Mythos (codenamed Capybara) was released April 8, 2026 as a restricted preview for cybersecurity work under Project Glasswing. Access is limited to ~40 organizations (Amazon, Apple, Microsoft, CrowdStrike, etc.) for defensive security. It is not available via API, Claude Code, or any subscription tier. Do not plan around it.

**What you CAN use today:**

| Model | Strengths | Cost (per M tokens) | Use for |
|-------|-----------|---------------------|---------|
| Opus 4.6 | Deepest reasoning, 1M context, Agent Teams | $15/$75 | Architecture design, paper reasoning, complex debugging |
| Sonnet 4.6 | 98% of Opus coding at 5x cheaper, fast | $3/$15 | Daily coding, writing drafts, routine tasks |
| Haiku 4.5 | Fastest, cheapest | $0.25/$1.25 | Linting, simple lookups, repetitive transforms |

---

## 1. Decision Analysis: Is Claude Max Worth It?

### The Verdict: Yes, Max 5x ($100/mo). Not Max 20x.

**Why Max 5x specifically:**

- **Claude Code access is included.** This is the killer feature. Claude Code is your terminal-based coding agent — it reads your entire HINT++ codebase, plans changes across files, runs tests. Without Max, you'd need Pro ($20/mo) + separate API billing for heavy sessions.
- **5x usage = ~225+ messages per 5-hour window.** For a research workflow where you alternate between deep thinking sessions and implementation, this is sufficient. You'll rarely exhaust it if you route models correctly.
- **Opus 4.6 becomes default in Claude Code on Max.** Pro defaults to Sonnet. Max gives you Opus when you need it without manual API configuration.
- **1M token context window.** On Max, Opus automatically gets 1M context in Claude Code with no extra configuration. Your entire HINT++ codebase fits in one session.

**Why NOT Max 20x ($200/mo):**

You're a single researcher, not a team running parallel agents all day. 5x gives you ~5 deep sessions per day. If you exhaust that, the problem is workflow efficiency, not quota.

**Why NOT API-only ($0 subscription + pay-per-token):**

A heavy API user doing HINT++-scale work would spend $50–150/month on tokens alone, without the convenience of Claude Code's integrated codebase understanding, file editing, and test execution. Max bundles this. The breakeven is roughly 3 hours of daily Claude Code use.

**Why NOT local models:**

Local open-weight models (DeepSeek V3.2, Qwen 3) score 72–74% on SWE-bench vs. Opus's 80.8%. For safety-critical research code (your permission fields, monotone checks), the gap matters. Use local models for bulk preprocessing or dataset scripts, not core algorithm work.

### When Higher Throughput Actually Helps Research

Throughput translates to better research outcomes in exactly three scenarios:

1. **Rapid experiment iteration loops.** When you're tuning β₁, β₂ asymmetry and need to modify code → run experiment → analyze results → modify code in tight cycles. Hitting rate limits here kills momentum.
2. **Long-context paper writing sessions.** Revising the full HINT++ paper (method + experiments + related work) in a single session requires sustained context. Running out of quota mid-revision means losing all accumulated context.
3. **Multi-file codebase refactoring.** When integrating Phase 5 (Exemplar Memory) into Phase 6 (Full Integration), Claude Code needs to reason across 10+ files simultaneously. This is token-heavy.

Throughput does NOT help for: literature search (web search is the bottleneck), simple coding tasks, or brainstorming (you're the bottleneck, not the model).

---

## 2. System Architecture

### 2.1 Local Environment (Your PC)

```
~/hint-plus-plus/
├── CLAUDE.md                    # Project-wide Claude Code instructions
├── .claude/
│   ├── settings.json            # Model defaults, permissions
│   └── skills/                  # Claude Code skills (SKILL.md files)
│       ├── phase-implement/     # Phase-specific implementation skill
│       ├── safety-check/        # Safety constraint verification skill
│       ├── experiment-run/      # Experiment execution skill
│       └── paper-section/       # Paper writing skill
├── src/
│   ├── models/                  # PointTransformer v3, Sonata
│   ├── safety/                  # Adaptive moments, permission field, monotone check
│   ├── memory/                  # Exemplar memory (M statistics)
│   ├── adaptation/              # TTA loop, meta-learning
│   └── utils/                   # Data loading, metrics, visualization
├── experiments/
│   ├── configs/                 # Hydra/YAML experiment configs
│   ├── scripts/                 # Run scripts per experiment
│   └── results/                 # Auto-organized by experiment ID
├── paper/
│   ├── sections/                # One .tex file per section
│   ├── figures/                 # Generated figures + source scripts
│   └── tables/                  # Auto-generated from experiment results
├── docs/                        # Obsidian vault symlink target
└── tests/                       # Unit + integration tests per phase
```

### 2.2 CLAUDE.md (Project Root — Claude Code Reads This Every Session)

```markdown
# HINT++ — Interactive Safe Test-Time Adaptation for 3D Segmentation

## Project
CVPR 2026 submission. Core contribution: zero-shot safe deployment of
interactive TTA to unknown domains by meta-learning transferable safety
patterns from human correction history.

## Tech Stack
- PyTorch, PointTransformer v3, Sonata
- Training: S3DIS | Eval: ScanNet, SemanticKITTI, nuScenes
- Experiment tracking: Weights & Biases
- Config: Hydra

## Conventions
- Type hints on all functions. Docstrings on all public methods.
- nn.Module for anything stateful. No global state.
- Tests required before merging any phase.
- Safety-critical code (permission field, monotone check) gets
  numerical stability assertions: check for NaN, check ε > 0.

## Architecture Phases
1. Frozen Teacher (COMPLETE)
2. Adaptive Moment Safety Signals — Adam-style per-class safety
3. Permission Field P(x)
4. Monotone Safety Check
5. Exemplar Memory (M stats, not tensors)
6. Full Integration
7. CVPR Evaluation

## Key Equations
- δₖ(t): correction signal for class k at time t
- m̂ₖ = β₁·mₖ + (1-β₁)·δₖ  (bias-corrected)
- v̂ₖ = β₂·vₖ + (1-β₂)·δₖ²  (bias-corrected)
- Safety weight = η·m̂ₖ/(√v̂ₖ + ε)

## Rules
- NEVER reference DynaCITY in code comments, docstrings, or paper text
- Run `pytest tests/` before committing
- Experiment results go to experiments/results/{experiment_id}/
```

### 2.3 Claude Code Skills (for Claude Code terminal agent)

These are different from the .yml skill files we created earlier. The .yml files are for Claude.ai Projects (the web interface). Claude Code uses SKILL.md files in `.claude/skills/`. You need BOTH:

**Claude.ai Projects** (web interface): Use the 6 .yml skill files we created (researcher, paper writer, reviewer simulator, code architect, experiment designer, literature monitor). These are for deep thinking sessions — paper writing, review simulation, literature analysis.

**Claude Code** (terminal): Use SKILL.md files for hands-on coding work. Here are the ones you need:

#### `.claude/skills/phase-implement/SKILL.md`
```markdown
---
name: phase-implement
description: Implement a specific phase of the HINT++ seven-phase architecture.
  Handles module creation, testing, and integration.
model: opusplan
---

When implementing a HINT++ phase:

1. Check entry criteria for this phase are met
2. Create the module under src/ following nn.Module pattern
3. Add numerical stability checks for all division operations (ε guards)
4. Write unit tests with synthetic point cloud data
5. Verify backward compatibility with completed phases
6. Log to W&B: phase name, test pass/fail, key metrics

Phase-specific guidance:
- Phase 2: β₁, β₂ are ASYMMETRIC by design. Do not default to Adam's 0.9/0.999.
- Phase 3: Permission field must be differentiable. Use sigmoid-based gating.
- Phase 4: Monotonicity = non-decreasing in safety evidence. Assert this in tests.
- Phase 5: Store M sufficient statistics (mean, variance, count), NOT full tensors.
```

#### `.claude/skills/experiment-run/SKILL.md`
```markdown
---
name: experiment-run
description: Configure, launch, and analyze HINT++ experiments.
  Use when running evaluations, ablations, or the primary zero-shot deployment test.
model: sonnet
---

Experiment workflow:
1. Create config in experiments/configs/{exp_name}.yaml
2. Create run script in experiments/scripts/run_{exp_name}.sh
3. Results auto-save to experiments/results/{exp_id}/
4. Generate summary table comparing against baselines

Primary experiment baselines:
- HINT-3D source thresholds: ~45-50% violations (unsafe, no tuning)
- HINT-3D tuned per domain: ~18-20% violations (~2 weeks/domain)
- HINT++ zero-shot: target <15% violations (no tuning)

Always report: violation rate, mIoU, per-class IoU, wall-clock time.
Always run ≥3 seeds. Report mean ± std.
```

#### `.claude/skills/safety-check/SKILL.md`
```markdown
---
name: safety-check
description: Verify safety properties of HINT++ code. Use when modifying
  permission fields, monotone checks, or adaptive moment computations.
model: opus
effort: high
---

Safety verification checklist:
1. No division by zero: all denominators have ε > 0
2. Permission field P(x) ∈ [0, 1] — clamp or sigmoid
3. Monotonicity: P(x, t+1) ≥ P(x, t) when safety evidence increases
4. Bias correction applied before using moments
5. No silent NaN propagation — add torch.isnan assertions
6. Exemplar memory bounded — enforce max size, LRU or priority eviction

Run: pytest tests/test_safety.py -v
If ANY safety test fails, do NOT proceed. Fix first.
```

### 2.4 Model Routing Strategy in Claude Code

Set in `.claude/settings.json`:
```json
{
  "model": "opusplan",
  "effort": "medium"
}
```

This gives you Opus for planning, Sonnet for execution — automatically. Override per-task:

| Task | Model | Effort | Why |
|------|-------|--------|-----|
| Architecture decisions | opus | high | Need deepest reasoning for safety proofs |
| Daily coding | sonnet | medium | 98% quality, 5x cheaper |
| Bulk file transforms | haiku | low | Renaming, formatting, simple refactors |
| Safety-critical code | opus | high | Permission field, monotone check — correctness matters |
| Paper section drafts | opus | medium | Long-context coherence for multi-page sections |
| Experiment scripts | sonnet | medium | Straightforward config generation |
| Debugging | opus → sonnet | high → medium | Start Opus for diagnosis, switch to Sonnet for fix |

### 2.5 Experiment Tracking & Versioning

| Tool | Purpose | Why This One |
|------|---------|-------------|
| **Weights & Biases** | Experiment tracking, hyperparameter sweeps | Best integration with PyTorch, free for academics |
| **Git + GitHub** | Code versioning | Standard. Branch per phase. |
| **Hydra** | Experiment configuration | Composable YAML configs, CLI overrides, auto-logs |
| **DVC** (optional) | Dataset/model versioning | Only if you need to share large checkpoints across machines |

Skip MLflow (redundant with W&B), skip Docker for now (you're a single researcher, not deploying), skip schedulers (Hydra multirun handles sweeps).

---

## 3. Workflow Design: The Research Pipeline

### Stage 1: Literature → Positioning (Weeks 1–2 of any phase)

**Where:** Claude.ai web interface with the `hint_plus_plus_researcher.yml` and `hint_pp_lit_monitor.yml` skill files loaded into a Claude Project.

**How:**
1. Open a Claude Project called "HINT++ Research"
2. Upload the researcher and lit monitor skill files as project knowledge
3. Ask targeted questions: "Search for papers published since January 2026 on test-time adaptation with safety constraints for 3D point clouds"
4. The researcher agent decomposes into sub-questions, searches, synthesizes
5. Feed findings into your Obsidian vault

**Model:** Opus 4.6 in claude.ai (select in model picker). Deep reasoning for literature synthesis.

### Stage 2: Idea Refinement → Validation (Weeks 2–3)

**Where:** Claude.ai with `hint_pp_reviewer_sim.yml` loaded.

**How:**
1. Write your proposed approach in 1–2 paragraphs
2. Run the reviewer simulator: "Simulate 3 hostile CVPR reviewers for this approach"
3. Iterate until you can defend every claim
4. Lock the approach before coding

**Key principle:** Validate the idea BEFORE writing code. The reviewer simulator exists to kill bad ideas early, not to polish bad ideas late.

### Stage 3: Implementation (Weeks 3–8 per phase)

**Where:** Claude Code in terminal.

**How:**
1. Start session: `claude --model opusplan`
2. Phase entry: `/phase-implement` → specify which phase
3. Code in tight loops: design module → write tests → implement → run tests → fix
4. Safety-critical code triggers `/safety-check` automatically (Claude detects it from the skill description)
5. Commit working phases to Git with tags: `git tag phase-2-complete`

**Model routing:**
- Start each session in `opusplan` mode
- For complex debugging: `/model opus` + `/effort high`
- For boilerplate: `/model sonnet`

### Stage 4: Experiments (Weeks 8–12)

**Where:** Claude Code for script generation, local GPU for execution, W&B for tracking.

**How:**
1. `/experiment-run` skill generates configs and run scripts
2. Execute on GPU: `python -m experiments.scripts.run_primary`
3. Results auto-log to W&B
4. Use Claude.ai with `hint_pp_experiment_designer.yml` to design ablations based on initial results
5. Generate paper tables/figures directly from W&B data

**Critical:** Run the primary experiment (zero-shot deployment) FIRST. If violations are >15% with zero tuning, debug before running ablations. Don't waste GPU time on ablations of a broken system.

### Stage 5: Paper Writing (Weeks 10–16)

**Where:** Claude.ai with `hint_pp_paper_writer.yml` loaded.

**How:**
1. Write each section as a separate .tex file
2. Use the paper writer skill for drafting — it enforces the locked-in framing
3. After drafting, run the reviewer simulator on each section
4. Use Claude Code for LaTeX compilation and figure generation

**Parallel track:** Start the introduction and related work during Stage 3. Method section during Stage 4. Results and conclusion during Stage 5.

---

## 4. Optimization Strategy: Maximizing LLM Effectiveness

### 4.1 Prompt Structuring for HINT++

**The CLAUDE.md file is your single highest-leverage prompt.** Claude Code reads it at session start. Every instruction there applies to every interaction without re-prompting. Put your notation conventions, phase structure, and safety rules there — NOT in individual messages.

**Per-message prompts should be surgical:**

Bad:
> "Can you help me implement the adaptive moment safety signals? I need a module that computes per-class safety sensitivity using Adam-style moment estimation with correction signals and bias correction."

Good:
> "Implement Phase 2: AdaptiveMomentSafety(nn.Module). Inputs: per-class correction signals δₖ(t). Outputs: adaptive safety weights. Follow the Adam correspondence in CLAUDE.md. Include bias correction. Test with synthetic data where classes 0-3 have consistent corrections and classes 4-6 have noisy corrections."

The second prompt is better because it names the exact module, specifies inputs/outputs, references the existing documentation (CLAUDE.md), and gives a concrete test scenario.

### 4.2 Token Efficiency Tactics

1. **Use `opusplan` mode.** Opus reasons about architecture (expensive but valuable), Sonnet writes the code (cheap, fast). Saves 60–80% vs. Opus for everything.

2. **Effort levels matter.** `/effort low` for simple file reads. `/effort high` only for safety-critical reasoning. Default `medium` for everything else.

3. **Don't re-explain context.** Your CLAUDE.md and skills carry context automatically. If you find yourself re-explaining what HINT++ is, your CLAUDE.md is incomplete.

4. **Batch related changes.** Instead of 5 separate requests to modify 5 files, say "Update all files in src/safety/ to use the new ε=1e-8 constant." Claude Code handles multi-file edits in one turn.

5. **Use Haiku for mechanical work.** Renaming variables across the codebase, reformatting imports, generating boilerplate configs — switch to Haiku.

### 4.3 Iterative Refinement Loops

The most productive pattern for research code:

```
[You] Write hypothesis as a 1-line comment in the code
      ↓
[Claude Code, Opus] Design the test that would validate/invalidate it
      ↓
[Claude Code, Sonnet] Implement the test
      ↓
[GPU] Run the test
      ↓
[You] Interpret results, update hypothesis
      ↓
[Claude Code, Opus] Reason about what went wrong / what to try next
      ↓
Repeat
```

The key insight: YOU own the hypothesis. Claude owns the implementation. Never let Claude generate hypotheses about your research direction — it doesn't understand the reviewer landscape, the competitive positioning, or your advisor's preferences. Use Claude to test YOUR ideas faster, not to generate ideas.

---

## 5. Coding + Research Integration

### 5.1 Codebase Structure for Rapid Iteration

**One module per phase. One test file per module. One config per experiment.**

```python
# src/safety/adaptive_moments.py — Phase 2

class AdaptiveMomentSafety(nn.Module):
    """Per-class safety sensitivity via Adam-style moment estimation.

    Correspondence:
        δₖ(t) → correction signal (like gradient in Adam)
        m̂ₖ   → correction consistency (like first moment)
        v̂ₖ   → correction noise (like second moment)
        weight = η · m̂ₖ / (√v̂ₖ + ε)
    """

    def __init__(self, num_classes: int, beta1: float = 0.7, beta2: float = 0.95,
                 eps: float = 1e-8, eta: float = 1.0):
        # β₁ < β₂ by design: safety needs slower noise estimation
        super().__init__()
        ...
```

**Principles:**
- Each phase is a self-contained `nn.Module` with its own forward pass
- Phase N+1 imports Phase N's output, never its internals
- Configs are Hydra YAML — never hardcode hyperparameters
- Every module has a `tests/test_{module}.py` with synthetic data tests

### 5.2 Using Claude Without Degrading Understanding

This is the real risk. Three rules:

1. **Never accept code you can't explain line-by-line.** If Claude generates a permission field computation and you can't derive why the sigmoid gating preserves monotonicity, you don't understand your own method. Ask Claude to EXPLAIN, then rewrite it yourself.

2. **Write the math first, code second.** Derive the adaptive moment equations on paper (or in LaTeX). THEN ask Claude to implement them. If you start with code, you'll end up fitting your theory to Claude's implementation choices rather than the reverse.

3. **Own the ablation interpretations.** Claude can generate tables. You must explain WHY asymmetric β works better. The understanding lives in the interpretation, not the numbers.

---

## 6. Execution Plan: Weekly System to CVPR Submission

### Timeline: April 2026 → September–November 2026

Assuming CVPR 2026 deadline is ~November 2026 (verify when announced).

#### Phase 2: Adaptive Moment Safety Signals (Weeks 1–4, April–May)

| Week | Focus | Deliverable | Claude Tool |
|------|-------|-------------|-------------|
| 1 | Literature deep dive on adaptive thresholds | Obsidian notes, positioning doc | Claude.ai + Researcher skill |
| 2 | Module design, test specification | `AdaptiveMomentSafety` API + test cases | Claude Code, Opus |
| 3 | Implementation + unit tests | Passing tests with synthetic data | Claude Code, Sonnet |
| 4 | Integration with frozen teacher | End-to-end forward pass works | Claude Code, opusplan |

**Exit criteria:** Synthetic corrections → adaptive weights that differ per class. Classes with consistent corrections get high weights. Classes with noisy corrections get low weights.

#### Phase 3: Permission Field (Weeks 5–7, May–June)

| Week | Focus | Deliverable | Claude Tool |
|------|-------|-------------|-------------|
| 5 | Permission field design | P(x) specification + differentiability proof | Claude.ai, Opus |
| 6 | Implementation + monotonicity tests | Module + passing safety tests | Claude Code, Opus |
| 7 | Integration with Phase 2 | Adaptive moments → permission field pipeline | Claude Code, opusplan |

#### Phase 4: Monotone Safety Check (Week 8, June)

| Week | Focus | Deliverable | Claude Tool |
|------|-------|-------------|-------------|
| 8 | Safety guarantee implementation | Monotone check module + formal property tests | Claude Code, Opus (effort: high) |

#### Phase 5: Exemplar Memory (Weeks 9–11, June–July)

| Week | Focus | Deliverable | Claude Tool |
|------|-------|-------------|-------------|
| 9 | Memory design (M stats, sampling strategies) | Memory module API | Claude.ai, Experiment Designer |
| 10 | Implementation with memory budget experiments | Module + memory profiling | Claude Code, Sonnet |
| 11 | Sampling strategy comparison | Ablation results (recent vs. balanced vs. hard neg) | Claude Code + GPU |

#### Phase 6: Full Integration (Weeks 12–14, July–August)

| Week | Focus | Deliverable | Claude Tool |
|------|-------|-------------|-------------|
| 12 | End-to-end pipeline assembly | Full HINT++ forward pass | Claude Code, opusplan |
| 13 | Meta-learning loop implementation | Train on S3DIS with simulated corrections | Claude Code, Opus |
| 14 | Debugging + optimization | Stable training, no NaN, reasonable loss curves | Claude Code, Opus (effort: high) |

#### Phase 7: Experiments + Paper (Weeks 15–24, August–November)

| Week | Focus | Deliverable | Claude Tool |
|------|-------|-------------|-------------|
| 15–16 | PRIMARY experiment: zero-shot deployment | Results on ScanNet, SemanticKITTI, nuScenes | GPU + W&B |
| 17 | Checkpoint 5 evaluation | YES/NO on all 4 success criteria | You (no Claude needed) |
| 18–19 | Ablation studies | 7 ablation tables | Claude Code + Experiment Designer |
| 20–21 | Paper draft: method + experiments | Complete .tex sections | Claude.ai + Paper Writer |
| 22 | Internal review simulation | 4 simulated reviews + rebuttals | Claude.ai + Reviewer Simulator |
| 23 | Paper revision based on simulated reviews | Final draft | Claude.ai + Paper Writer |
| 24 | Submission prep | Camera-ready, supplementary, code release | Claude Code + LaTeX |

### Weekly Ritual (Every Monday)

1. **15 min:** Review W&B dashboard. Any experiments running? Any results to interpret?
2. **15 min:** Check the week's milestone against the table above. On track?
3. **30 min:** Plan the week's Claude sessions. What needs Opus? What's Sonnet-level?
4. **5 min:** Run literature monitor check (if relevant new papers dropped)

### Monthly Checkpoint

- **End of each month:** Meet with Dr. Asmar / Dr. El-Hajj. Present current phase status.
- **Bring:** W&B plots, passing test counts, and the reviewer simulator's latest objections.
- **Ask:** "Does this phase's output support the paper's zero-shot deployment claim?"

---

## 7. Cost Projection

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Claude Max 5x | $100 | Covers Claude Code + web interface |
| W&B | $0 | Free for academics |
| GPU (if needed) | $0–50 | Use university cluster; cloud only for overflow |
| **Total** | **$100–150/mo** | |

Over 6 months (May–October): **$600–900 total.**

This is extremely cost-effective for a CVPR submission. A single conference trip costs more.

---

## 8. Summary: The Stack

```
┌─────────────────────────────────────────────────────┐
│                   CLAUDE.AI (WEB)                    │
│  Projects with skill files loaded:                   │
│  • Researcher         • Paper Writer                 │
│  • Reviewer Simulator • Literature Monitor           │
│  • Experiment Designer                               │
│  Model: Opus 4.6 for deep thinking sessions          │
├─────────────────────────────────────────────────────┤
│                 CLAUDE CODE (TERMINAL)                │
│  CLAUDE.md + .claude/skills/:                        │
│  • phase-implement    • experiment-run               │
│  • safety-check                                      │
│  Model: opusplan (Opus plans, Sonnet executes)       │
├─────────────────────────────────────────────────────┤
│                   LOCAL ENVIRONMENT                   │
│  • Git repo (src/, experiments/, paper/, tests/)     │
│  • Obsidian vault (knowledge graph, wikilinks)       │
│  • Hydra configs + W&B tracking                      │
│  • GPU for training + evaluation                     │
├─────────────────────────────────────────────────────┤
│                   SUBSCRIPTION                       │
│  Claude Max 5x ($100/mo)                             │
│  Opus 4.6 + Sonnet 4.6 + Haiku 4.5                  │
│  1M token context in Claude Code                     │
└─────────────────────────────────────────────────────┘
```
