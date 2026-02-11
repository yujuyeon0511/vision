# NLP Research Conventions

## Experiment ID Format

```
EXP-YYYYMMDD-NNN
```

- `YYYYMMDD`: Date the experiment was created
- `NNN`: Sequential number within the day (001, 002, ...)
- Example: `EXP-20260211-001`

## File Naming Conventions

### Documents
| Phase | Pattern | Example |
|-------|---------|---------|
| Hypothesis | `{topic}-hypothesis.md` | `vl-alignment-hypothesis.md` |
| Design | `{EXP-ID}-design.md` | `EXP-20260211-001-design.md` |
| Log | `{EXP-ID}-log.md` | `EXP-20260211-001-log.md` |
| Analysis | `{EXP-ID}-analysis.md` | `EXP-20260211-001-analysis.md` |
| Paper | `{section-name}.md` | `introduction.md` |

### Configs
```
experiments/configs/{EXP-ID}-config.yaml
```

### Results
```
results/tables/{EXP-ID}-{description}.csv
results/figures/{EXP-ID}-{description}.png
```

## Metric Reporting Rules

### Format
- Report as **mean +/- std** over multiple runs
- Decimal places:
  - Accuracy, F1, BLEU, etc.: **4 decimal places** (e.g., 0.8534 +/- 0.0021)
  - Percentages: **2 decimal places** (e.g., 85.34% +/- 0.21%)
  - Loss values: **4 decimal places**
  - Perplexity: **2 decimal places**

### Multiple Runs
- Default seeds: `[42, 123, 456]`
- Always run at minimum **3 seeds** for reported results
- Report individual run results in appendix if needed

### Statistical Significance
- Use **paired t-test** or **bootstrap test** for comparison
- Report p-values
- Significance threshold: **p < 0.05**
- Mark significant results with asterisk (*) in tables

### Tables
- Use **bold** for best results in each column
- Use _underline_ for second-best
- Include parameter count and FLOPs where relevant
- Always include baseline reproductions (not just reported numbers)

## Document Conventions

### Language
- Technical content: **English**
- Internal notes/comments: Korean or English (author preference)
- Paper drafts: **English**

### Status Labels
Documents use these status labels:
- `Draft` - Initial version, work in progress
- `Under Review` - Awaiting feedback
- `Approved` - Finalized and approved
- `Archived` - No longer active

### Cross-references
- Reference other documents using relative paths: `See [design](../../02-experiment-design/features/EXP-ID-design.md)`
- Reference experiments by their EXP-ID

## Version Control

### Commit Message Format
```
[phase] brief description

phase: hypothesis | design | experiment | analysis | paper
```

Examples:
```
[hypothesis] define RQ for VL alignment study
[design] add ablation configs for EXP-20260211-001
[experiment] log results for EXP-20260211-001
[analysis] add significance tests for main results
[paper] draft introduction section
```

### Branch Strategy
- `main`: Stable research state
- `exp/{EXP-ID}`: Experiment branches
- `paper/{venue}`: Paper preparation branches
