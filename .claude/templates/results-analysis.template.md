# Results Analysis: {{EXP_ID}}

> Research Phase: **4 - Result Analysis**
> Experiment Ref: {{EXP_LOG_REF}}
> Analyzed: {{DATE}}
> Status: In Progress | Complete

---

## 1. Main Results

### Performance Comparison
| Method | Metric1 | Metric2 | Metric3 | Params | FLOPs |
|--------|---------|---------|---------|--------|-------|
| Baseline 1 | | | | | |
| Baseline 2 | | | | | |
| **Ours** | | | | | |

> All results: mean +/- std over 3 runs (seeds: 42, 123, 456)

### Best Configuration
<!-- Which hyperparameters/settings produced the best results? -->

## 2. Statistical Verification

### Significance Tests
| Comparison | Test | Statistic | p-value | Significant? |
|-----------|------|-----------|---------|--------------|
| Ours vs Baseline 1 | paired t-test | | | |
| Ours vs Baseline 2 | paired t-test | | | |

### Confidence Intervals
| Method | Metric | 95% CI |
|--------|--------|--------|
| Ours | | [, ] |

### Effect Size
| Comparison | Cohen's d | Interpretation |
|-----------|-----------|----------------|
| | | |

## 3. Ablation Results

| Variant | Metric1 | Metric2 | Delta from Full |
|---------|---------|---------|----------------|
| Full model | | | - |
| w/o component A | | | |
| w/o component B | | | |

### Key Takeaways from Ablation
1.
2.

## 4. Error Analysis

### Error Categories
| Category | Count | % | Example |
|----------|-------|---|---------|
| | | | |
| | | | |
| | | | |

### Failure Cases
<!-- Describe representative failure cases -->

#### Case 1
- Input:
- Expected:
- Predicted:
- Likely cause:

### Performance by Data Subset
| Subset | N | Metric | Notes |
|--------|---|--------|-------|
| | | | |

## 5. Hypothesis Verification

### H1: {{HYPOTHESIS_TEXT}}
- **Result**: Supported / Partially Supported / Not Supported
- **Evidence**:
- **Caveats**:

### Additional Findings
<!-- Unexpected results or insights -->

## 6. Visualization

### Figures
| Figure | Path | Description |
|--------|------|-------------|
| Fig 1 | `results/figures/` | |
| Fig 2 | `results/figures/` | |

### Tables
| Table | Path | Description |
|-------|------|-------------|
| Tab 1 | `results/tables/` | |

## 7. Comparison with Related Work

| Method | Our Repro | Reported | Gap | Notes |
|--------|----------|----------|-----|-------|
| | | | | |

## 8. Conclusions

### Summary
<!-- 2-3 sentence summary of key findings -->

### Strengths
1.
2.

### Limitations
1.
2.

### Recommendations
<!-- What should be done next? -->
- [ ] Proceed to paper writing
- [ ] Run additional experiments
- [ ] Iterate on method design

---

## Notes
<!-- Detailed observations, reviewer feedback, discussion points -->
