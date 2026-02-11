# NLP Research Workflow

## Project Overview

This is a **Multimodal NLP (Vision-Language) research** project with a structured 5-phase research lifecycle:

| Phase | Activity | Directory |
|-------|----------|-----------|
| 1 | Hypothesis | `docs/01-hypothesis/` |
| 2 | Experiment Design | `docs/02-experiment-design/` |
| 3 | Experiment Run | `docs/03-experiment-log/` |
| 4 | Result Analysis | `docs/04-results/` |
| 5 | Paper Writing / Iterate | `docs/05-paper/` |

## Directory Structure

```
docs/                              # Research documents
  .research-status.json            # Current research state tracker
  01-hypothesis/features/          # Research questions & hypotheses
  02-experiment-design/features/   # Experiment design docs
  03-experiment-log/               # Execution logs
  04-results/features/             # Analysis reports
  05-paper/sections/               # Paper section drafts
experiments/
  configs/                         # YAML experiment configurations
  scripts/                         # Training/evaluation scripts
results/
  tables/                          # Result tables (CSV, Markdown)
  figures/                         # Plots, visualizations
paper/                             # LaTeX paper source
data/                              # Datasets (gitignored)
```

## Commands

The `/research` skill provides the unified command interface:

```
/research hypothesis [topic]  → Hypothesis definition + literature review
/research design [topic]      → Experiment design + config YAML generation
/research run [topic]         → Experiment execution guide + environment snapshot
/research analyze [topic]     → Result analysis + statistical verification
/research paper [topic]       → Paper section writing
/research iterate [topic]     → Next experiment improvement plan
/research status              → Current research status overview
/research next                → Next step guidance
```

## Conventions

### Experiment ID Format
`EXP-YYYYMMDD-NNN` (e.g., `EXP-20260211-001`)

### File Naming
- Hypothesis: `docs/01-hypothesis/features/TOPIC-hypothesis.md`
- Design: `docs/02-experiment-design/features/EXP-ID-design.md`
- Log: `docs/03-experiment-log/EXP-ID-log.md`
- Results: `docs/04-results/features/EXP-ID-analysis.md`
- Paper: `docs/05-paper/sections/SECTION-NAME.md`
- Config: `experiments/configs/EXP-ID-config.yaml`

### Metric Reporting
- Always report mean and standard deviation (mean +/- std) over multiple runs
- Use consistent decimal places (4 for most metrics, 2 for percentages)
- Include confidence intervals for key results
- Report statistical significance (p-value < 0.05)

### Document Language
- Technical documents: English
- Comments/notes: Korean or English (author preference)

## Agents

| Agent | Model | Role |
|-------|-------|------|
| literature-reviewer | sonnet | Paper search, summarization, hypothesis support |
| experiment-designer | sonnet | Experiment design, config YAML generation, GPU check |
| result-analyzer | sonnet | Result analysis, statistical tests, visualization |
| paper-writer | opus | Paper section writing (Markdown + LaTeX hints) |

## State Tracking

Research state is tracked in `docs/.research-status.json`. The status file records:
- Current research phase
- Phase completion status
- List of experiments and their states
- Iteration count

Always update the status file when transitioning between phases.
