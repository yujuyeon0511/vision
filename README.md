# NLP Research Workflow

Multimodal NLP (Vision-Language) research workflow powered by Claude Code.

## Research Lifecycle

| Phase | Activity | Description |
|-------|----------|-------------|
| 1 | Hypothesis | Define research questions, literature review |
| 2 | Experiment Design | Model architecture, dataset, metrics, baselines |
| 3 | Experiment Run | Execute experiments, log environments |
| 4 | Result Analysis | Statistical verification, error analysis |
| 5 | Paper Writing / Iterate | Write paper or plan next experiment |

## Directory Structure

```
docs/                        # Research documents (by phase)
  01-hypothesis/features/    # Research questions & hypotheses
  02-experiment-design/features/  # Experiment configurations
  03-experiment-log/         # Execution logs
  04-results/features/       # Analysis reports
  05-paper/sections/         # Paper drafts
experiments/
  configs/                   # YAML experiment configs
  scripts/                   # Training/eval scripts
results/
  tables/                    # Result tables (CSV/MD)
  figures/                   # Plots and visualizations
paper/                       # LaTeX paper source
data/                        # Datasets (gitignored)
```

## Commands

```
/research hypothesis [topic]  # Define hypothesis + literature review
/research design [topic]      # Design experiment + generate config
/research run [topic]         # Execution guide + environment snapshot
/research analyze [topic]     # Result analysis + statistical tests
/research paper [topic]       # Write paper sections
/research iterate [topic]     # Plan next experiment improvements
/research status              # Current research status
/research next                # Next step guidance
```

## Experiment ID Format

`EXP-YYYYMMDD-NNN` (e.g., `EXP-20260211-001`)
