# Visual Drift Compensation (VDC) - Method Diagram

**Date:** 2025-02-13

---

## Overview Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISUAL DRIFT COMPENSATION                     │
│                        Training Framework                        │
└─────────────────────────────────────────────────────────────────┘

Input: Image + Text (Language L2)
Goal: Preserve visual grounding while adapting to L2

┌────────────────────────┐
│   Frozen Vision        │
│   Encoder (CLIP)       │
│   ViT-L/14             │
└────────┬───────────────┘
         │
         │ Visual tokens
         │ [N × D_v]
         ▼
┌────────────────────────┐
│   Visual Projector     │
│   (3-layer MLP)        │◄────── Component 4: Modality-Aware LoRA
│   + LoRA adaptation    │         rank[layer] ∝ MIR[layer]
└────────┬───────────────┘
         │
         │ Projected visual tokens
         │ [N × D_llm]
         │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
    ┌────────────────┐       ┌────────────────────┐
    │ Component 1:   │       │ Component 3:       │
    │ Drift Tracking │       │ Visual Alignment   │
    │                │       │                    │
    │ Measure MMD    │       │ L_align =          │
    │ to anchors     │       │ MSE(v_mllm, v_vfm) │
    └────┬───────────┘       └────────┬───────────┘
         │                            │
         │ L_drift                    │ L_align
         │                            │
         └────────────┬───────────────┘
                      │
         ┌────────────▼────────────┐
         │   Loss Aggregation      │
         │                         │
         │ L = L_task + λ1·L_drift │
         │          + λ2·L_contrast│
         │          + λ3·L_align   │
         └─────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │   Backpropagation       │
         │   Update:               │
         │   - Projector (LoRA)    │
         │   - LLM (LoRA)          │
         └─────────────────────────┘
```

---

## Component 1: Drift Tracking & Compensation

```
┌───────────────────────────────────────────────────────────────┐
│              DRIFT TRACKING & COMPENSATION                    │
└───────────────────────────────────────────────────────────────┘

Step 1: Select Anchor Concepts
┌─────────────────────────────────────────────────┐
│ Anchor Set (50 concepts)                        │
│ - Universal objects: person, car, tree, etc.    │
│ - Spatial relations: above, left, inside, etc.  │
│ - Attributes: red, large, round, etc.           │
│                                                 │
│ Bilingual pairs: (image, text_en, text_l2)     │
└─────────────────────────────────────────────────┘

Step 2: Encode Anchors at Training Step t
┌──────────────────────┐         ┌──────────────────────┐
│  Anchor Images       │         │  Anchor Texts (L2)   │
│  ┌───┐ ┌───┐ ┌───┐  │         │  "사람" "차" "나무"    │
│  │img│ │img│ │img│  │         │  "person" "car"      │
│  └───┘ └───┘ └───┘  │         │  "tree"              │
└──────┬───────────────┘         └──────┬───────────────┘
       │                                │
       │ Vision Encoder                 │ LLM Embedder
       ▼                                ▼
┌──────────────────────┐         ┌──────────────────────┐
│  v_anchor_t          │         │  l_anchor_t          │
│  [50 × D_llm]        │         │  [50 × D_llm]        │
└──────┬───────────────┘         └──────┬───────────────┘
       │                                │
       └────────────┬───────────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │  Compute L_drift    │
          │                     │
          │  MMD(v_anchor_t,    │
          │      l_anchor_t)    │
          └─────────────────────┘

Step 3: Backprop through Projector
┌─────────────────────────────────────────────────┐
│  Gradient flow:                                 │
│  ∂L_drift/∂θ_projector                          │
│                                                 │
│  Effect: Projector learns to keep visual tokens │
│          close to LLM's current representation  │
│          of same concepts                       │
└─────────────────────────────────────────────────┘
```

---

## Component 2: Contrastive Visual Grounding

```
┌───────────────────────────────────────────────────────────────┐
│          CONTRASTIVE VISUAL GROUNDING REGULARIZATION          │
└───────────────────────────────────────────────────────────────┘

For each training example:

Path 1: With Visual Information
┌──────────┐     ┌──────────┐
│  Image   │────▶│  Vision  │
│  ┌────┐  │     │  Encoder │
│  │    │  │     └────┬─────┘
│  └────┘  │          │
└──────────┘          │ visual_tokens
                      │
           ┌──────────▼──────────┐
           │   Projector         │
           └──────────┬──────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │                      │
Text ─────▶│    LLM Decoder       │────▶ logits_visual
           │    (with visual      │
           │     tokens)          │
           └──────────────────────┘


Path 2: Text-Only (No Visual Information)
┌──────────────────────┐
│                      │
Text ─────▶│    LLM Decoder       │────▶ logits_text
           │    (text-only)       │
           │                      │
           └──────────────────────┘


Contrastive Loss
┌─────────────────────────────────────────────────┐
│  L_contrast = -log[P(target | visual + text)    │
│                   ────────────────────────  ]    │
│                    P(target | text only)         │
│              + KL(P_visual || P_text_frozen)     │
│                                                  │
│  Effect: Penalize model if outputs are similar  │
│          with and without image                  │
└─────────────────────────────────────────────────┘
```

---

## Component 3: Dynamic Visual Alignment

```
┌───────────────────────────────────────────────────────────────┐
│            DYNAMIC VISUAL REPRESENTATION ALIGNMENT            │
└───────────────────────────────────────────────────────────────┘

Multi-Layer Alignment Strategy

┌──────────────────────────────────────────────────────────────┐
│                   MLLM vs Frozen VFM                         │
│                                                              │
│  Layer 1 (Early)   [────── Large Gap ──────]  ⚠️ Low weight │
│  Layer 4 (Early)   [────── Large Gap ──────]  ⚠️ Low weight │
│                                                              │
│  Layer 8 (Middle)  [─── Moderate Gap ───]    ✓ High weight  │
│  Layer 12 (Middle) [─── Moderate Gap ───]    ✓ High weight  │
│  Layer 16 (Middle) [─── Moderate Gap ───]    ✓ High weight  │
│                                                              │
│  Layer 20 (Deep)   [── Small Gap ──]          ⚠️ Low weight │
│  Layer 24 (Deep)   [── Small Gap ──]          ⚠️ Low weight │
└──────────────────────────────────────────────────────────────┘
        ▲                                           ▲
        │                                           │
   Modality Gap                                Alignment
   (from ICCV 2025)                            Priority


Adaptive Weighting
┌─────────────────────────────────────────────────┐
│  λ3_adaptive = λ3_base × (1 + α × drift_mag)    │
│                                                 │
│  Low drift  → Low alignment (allow adaptation)  │
│  High drift → High alignment (preserve visual)  │
└─────────────────────────────────────────────────┘

Loss Computation
┌─────────────────────────────────────────────────┐
│  for layer in [8, 12, 16]:  # Middle layers     │
│      v_mllm = extract_features(mllm, layer)     │
│      v_vfm  = extract_features(frozen_vfm, layer)│
│      L_align += MSE(v_mllm, v_vfm) × weight[layer]│
└─────────────────────────────────────────────────┘
```

---

## Component 4: Modality-Aware LoRA

```
┌───────────────────────────────────────────────────────────────┐
│              LAYER-WISE MODALITY-AWARE LoRA                   │
└───────────────────────────────────────────────────────────────┘

Step 1: Compute Modality Integration Rate (MIR)
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  MIR[layer] = cross_modal_attention_score[layer]            │
│               ──────────────────────────────────            │
│                  total_attention[layer]                      │
│                                                              │
│  Layer 1-4:   MIR ≈ 0.1  (low integration)                  │
│  Layer 8-16:  MIR ≈ 0.6  (high integration)                 │
│  Layer 20-24: MIR ≈ 0.3  (moderate integration)             │
└──────────────────────────────────────────────────────────────┘

Step 2: Allocate LoRA Rank
┌─────────────────────────────────────────────────┐
│  rank[layer] = base_rank × MIR[layer]           │
│                                                 │
│  Example (base_rank = 16):                      │
│  - Layer 1:  rank = 16 × 0.1 = 2                │
│  - Layer 8:  rank = 16 × 0.6 = 10               │
│  - Layer 16: rank = 16 × 0.6 = 10               │
│  - Layer 24: rank = 16 × 0.3 = 5                │
│                                                 │
│  Visual Projector: rank = 16 × 1.5 = 24         │
└─────────────────────────────────────────────────┘

Step 3: LoRA Application
┌────────────────────────────────────┐
│  Standard Transformer Layer:       │
│  output = W × input                │
│                                    │
│  With LoRA:                        │
│  output = W × input                │
│          + (B_r × A_r) × input     │
│              ↑                     │
│         rank[layer]                │
│                                    │
│  Only train A_r, B_r matrices      │
└────────────────────────────────────┘
```

---

## Training Flow Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                     TRAINING ITERATION                        │
└───────────────────────────────────────────────────────────────┘

1. Batch Input
   ┌─────────────────────────────────────┐
   │ Regular batch:                      │
   │ - Images (B × 3 × H × W)            │
   │ - Texts in L2 (B × seq_len)         │
   │ - Targets (B × seq_len)             │
   │                                     │
   │ + Anchor batch (every N steps):     │
   │ - Anchor images (50)                │
   │ - Anchor texts L2 (50)              │
   └─────────────────────────────────────┘
                    │
                    ▼
2. Forward Pass
   ┌─────────────────────────────────────┐
   │ a) Standard forward with images     │
   │    → logits_visual                  │
   │                                     │
   │ b) Text-only forward (for L_contrast)│
   │    → logits_text                    │
   │                                     │
   │ c) Extract intermediate features    │
   │    → v_mllm[layers]                 │
   │    → v_vfm[layers] (frozen)         │
   │                                     │
   │ d) Process anchors (if anchor step) │
   │    → v_anchor, l_anchor             │
   └─────────────────────────────────────┘
                    │
                    ▼
3. Loss Computation
   ┌─────────────────────────────────────┐
   │ L_task = CrossEntropy(logits, target)│
   │                                     │
   │ L_contrast = -log(P_visual/P_text)  │
   │            + KL(P_v || P_t_frozen)  │
   │                                     │
   │ L_align = Σ MSE(v_mllm[l], v_vfm[l])│
   │           l ∈ middle_layers         │
   │                                     │
   │ L_drift = MMD(v_anchor, l_anchor)   │
   │           (only on anchor steps)    │
   │                                     │
   │ L_total = L_task                    │
   │         + λ1 × L_drift              │
   │         + λ2 × L_contrast           │
   │         + λ3 × L_align              │
   └─────────────────────────────────────┘
                    │
                    ▼
4. Backward Pass
   ┌─────────────────────────────────────┐
   │ Update parameters:                  │
   │                                     │
   │ ✓ Visual Projector LoRA (rank=24)  │
   │ ✓ LLM LoRA (layer-wise ranks)      │
   │ ✗ Vision Encoder (frozen)           │
   │ ✗ LLM base weights (frozen)         │
   └─────────────────────────────────────┘
                    │
                    ▼
5. Logging & Monitoring
   ┌─────────────────────────────────────┐
   │ Track over time:                    │
   │ - L_drift (should decrease)         │
   │ - MMD(v_anchor, l_anchor)           │
   │ - Visual attention allocation       │
   │ - Task performance                  │
   └─────────────────────────────────────┘
```

---

## Comparison: Standard LoRA vs VDC

```
┌───────────────────────────────────────────────────────────────┐
│                    STANDARD LoRA TRAINING                     │
└───────────────────────────────────────────────────────────────┘

Image ──▶ [Vision Enc] ──▶ [Projector] ──┐
                          (frozen)       │
                                         ▼
Text (L2) ─────────────────────────────▶ [LLM + LoRA]
                                              │
                                              ▼
                                          Prediction

Loss: L_task only

Problem:
- Visual tokens stay fixed while LLM adapts to L2
- LLM embedding space shifts
- Visual tokens become out-of-distribution
- Model learns to ignore visual information ❌


┌───────────────────────────────────────────────────────────────┐
│                    VDC TRAINING                               │
└───────────────────────────────────────────────────────────────┘

Image ──▶ [Vision Enc] ──▶ [Projector + LoRA] ──┐
                          (adapts to LLM)       │
                                                │
                          ┌─────────────────────┘
                          │
                          ├─▶ L_drift (track LLM shift)
                          │
                          ├─▶ L_align (preserve details)
                          │
                          ▼
Text (L2) ─────────────▶ [LLM + LoRA]
                              │
                              ├─▶ L_contrast (force visual use)
                              │
                              ▼
                          Prediction

Loss: L_task + L_drift + L_contrast + L_align

Solution:
- Visual tokens adapt to track LLM shift ✓
- Drift is measured and compensated ✓
- Contrastive loss forces visual grounding ✓
- Alignment preserves visual details ✓
```

---

## Expected Training Dynamics

```
┌───────────────────────────────────────────────────────────────┐
│              DRIFT OVER TRAINING (Expected Plot)              │
└───────────────────────────────────────────────────────────────┘

Drift Magnitude
    │
    │  ╔════════════════╗ Standard LoRA (uncontrolled drift)
 15 │  ║               ╱
    │  ║              ╱
    │  ║             ╱
 10 │  ║            ╱
    │  ║           ╱
    │  ║          ╱
  5 │  ║  ┌──────────── VDC (controlled drift)
    │  ║  │     ╱‾‾‾‾‾
    │  ║  │    ╱
  0 │  ╚══╧═══╧───────────────────▶ Training Steps
         0   5k   10k   15k   20k


Visual Attention Allocation
    │
100%│  ┌────────────────────────── VDC (stable)
    │  │
    │  │ ╲
 75%│  │  ╲___________________
    │  │           Standard LoRA
    │  │                ╲
 50%│  │                 ╲
    │  │                  ╲______
    │  │
    │  │
  0 │  └──────────────────────────▶ Training Steps
         0   5k   10k   15k   20k


Visual Grounding Accuracy
    │
 80%│  ┌─────────┬────────── VDC
    │  │         │
 70%│  │         ╲
    │  │          ╲____ Standard LoRA
 60%│  │
    │  │
 50%│  │
    │  │
    │  │
  0 │  └──────────────────────────▶ Training Steps
         0   5k   10k   15k   20k
```

---

## Implementation Checklist

### Core Components
- [ ] Drift measurement module
  - [ ] Anchor concept dataset (50 bilingual pairs)
  - [ ] MMD computation
  - [ ] Drift tracking logger

- [ ] Contrastive visual grounding
  - [ ] Dual forward pass (with/without image)
  - [ ] Contrastive loss computation
  - [ ] Text-only model freezing

- [ ] Visual alignment
  - [ ] Multi-layer feature extraction
  - [ ] Layer-wise MSE computation
  - [ ] Adaptive weighting based on drift

- [ ] Modality-aware LoRA
  - [ ] MIR computation per layer
  - [ ] Dynamic rank allocation
  - [ ] LoRA module integration

### Training Infrastructure
- [ ] Multi-loss aggregation
- [ ] Anchor batch sampling
- [ ] Gradient flow validation
- [ ] Checkpointing strategy

### Evaluation
- [ ] Drift measurement tools
- [ ] Visual grounding benchmarks
- [ ] Attention visualization
- [ ] Multilingual evaluation suite
