# Experiment Design: {{EXP_ID}}

> Research Phase: **2 - Experiment Design**
> Topic: {{TOPIC}}
> Hypothesis Ref: {{HYPOTHESIS_REF}}
> Created: {{DATE}}
> Status: Draft | Approved | Running | Completed

---

## 1. Objective

### Goal
<!-- What does this experiment aim to test or demonstrate? -->

### Hypothesis Being Tested
<!-- Which specific hypothesis does this experiment address? -->

## 2. Model Architecture

### Base Model
<!-- Pretrained model or architecture starting point -->
- Model:
- Parameters:
- Source:

### Modifications
<!-- What changes are made to the base model? -->

### Architecture Diagram
<!-- ASCII diagram or reference to figure -->
```
[Input] → [Encoder] → [Fusion] → [Decoder] → [Output]
```

## 3. Dataset

### Training Data
| Dataset | Size | Split | Description |
|---------|------|-------|-------------|
| | | Train | |
| | | Val | |
| | | Test | |

### Preprocessing
<!-- Data preprocessing steps -->
1.
2.
3.

### Data Augmentation
<!-- Augmentation strategies if applicable -->

## 4. Hyperparameters

### Training
```yaml
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 0.01
batch_size: 32
max_epochs: 50
warmup_steps: 1000
scheduler: cosine
gradient_clip: 1.0
seed: [42, 123, 456]
```

### Model-specific
```yaml
# Add model-specific hyperparameters
```

### Search Space (if tuning)
| Hyperparameter | Range | Scale |
|---------------|-------|-------|
| learning_rate | [1e-5, 1e-3] | log |
| | | |

## 5. Baselines

| # | Method | Paper/Source | Expected Performance | Notes |
|---|--------|-------------|---------------------|-------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## 6. Ablation Study

| # | Variant | What's Changed | Purpose |
|---|---------|---------------|---------|
| 1 | Full model | Nothing (reference) | Upper bound |
| 2 | | | |
| 3 | | | |

## 7. Evaluation Metrics

### Primary Metrics
| Metric | Formula/Tool | Target |
|--------|-------------|--------|
| | | |

### Secondary Metrics
| Metric | Formula/Tool | Purpose |
|--------|-------------|---------|
| | | |

### Statistical Tests
- Significance test: paired t-test / bootstrap
- Confidence level: 95%
- Number of runs: 3 (seeds: 42, 123, 456)

## 8. Computational Requirements

### Hardware
- GPU:
- VRAM per GPU:
- Number of GPUs:
- Estimated training time:

### Software
- Framework: PyTorch
- Key libraries:
- CUDA version:
- Python version:

## 9. Config File

> Auto-generated config saved to: `experiments/configs/{{EXP_ID}}-config.yaml`

---

## Notes
<!-- Design decisions, trade-offs, alternatives considered -->
