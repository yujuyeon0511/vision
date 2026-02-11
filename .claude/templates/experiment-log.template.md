# Experiment Log: {{EXP_ID}}

> Research Phase: **3 - Experiment Run**
> Design Ref: {{DESIGN_REF}}
> Started: {{START_DATE}}
> Ended: {{END_DATE}}
> Status: Running | Completed | Failed | Aborted

---

## 1. Environment Snapshot

### Hardware
```
hostname:
GPU:
GPU count:
VRAM:
CPU:
RAM:
Disk:
```

### Software
```
OS:
Python:
PyTorch:
CUDA:
cuDNN:
transformers:
```

### Git State
```
branch:
commit:
dirty: yes/no
```

## 2. Config Snapshot

> Full config: `experiments/configs/{{EXP_ID}}-config.yaml`

```yaml
# Key parameters at runtime (copy from actual config)
```

## 3. Execution Timeline

| Time | Event | Notes |
|------|-------|-------|
| {{START_DATE}} | Training started | |
| | | |
| | | |
| {{END_DATE}} | Training completed | |

## 4. Training Progress

### Loss Curve
<!-- Reference to saved plot or inline description -->
- Initial loss:
- Final loss:
- Best validation loss: (epoch )

### Key Checkpoints
| Epoch | Train Loss | Val Loss | Val Metric | Saved |
|-------|-----------|----------|------------|-------|
| | | | | |
| | | | | |

### GPU Utilization
```
Average GPU util: %
Average VRAM usage:  GB / GB
Peak VRAM usage:  GB
```

## 5. Issues and Resolutions

| # | Issue | Resolution | Impact |
|---|-------|-----------|--------|
| 1 | | | |

## 6. Outputs

### Checkpoints
- Best model: `checkpoints/{{EXP_ID}}/best.pt`
- Last model: `checkpoints/{{EXP_ID}}/last.pt`

### Logs
- Training log: `outputs/{{EXP_ID}}/train.log`
- WandB run: <!-- URL if applicable -->

### Generated Files
| File | Path | Description |
|------|------|-------------|
| | | |

## 7. Quick Summary

### Did the experiment succeed?
<!-- Yes/No with brief explanation -->

### Key observations
1.
2.
3.

### Recommended next steps
<!-- What should be done next based on this run? -->

---

## Notes
<!-- Runtime observations, anomalies, things to remember -->
