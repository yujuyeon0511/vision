# Visual Drift Compensation: A Novel Method for Preserving Visual Grounding During Cross-Lingual MLLM Adaptation

**Date:** 2025-02-13
**Status:** Hypothesis
**Research Phase:** 1 - Hypothesis Formation
**Based on:** Comprehensive literature review of 2024-2025 MLLM research

---

## Executive Summary

We propose **Visual Drift Compensation (VDC)**, a novel training framework that actively adapts visual token representations to stay "in-distribution" as an LLM's embedding space shifts during cross-lingual fine-tuning. Unlike existing methods that either freeze visual encoders or apply post-hoc corrections, VDC provides a principled, training-time solution to the visual grounding degradation problem observed in multilingual MLLMs.

### Key Innovation

**Problem:** When fine-tuning MLLMs on non-English data, visual tokens become "out-of-distribution" relative to the LLM's shifting embedding space, causing visual grounding degradation.

**Solution:** A multi-component training objective that:
1. **Tracks** embedding drift using anchor-based measurements
2. **Compensates** for drift by dynamically adapting visual projector
3. **Regularizes** with contrastive visual grounding loss
4. **Allocates** parameters efficiently using layer-wise modality awareness

### Why It's Novel

- **First work** to explicitly model visual embedding drift during language adaptation
- **Principled approach** combining insights from modality gap, embedding drift, and cross-lingual transfer research
- **Language-agnostic** - works for any language, not language-specific
- **Composable** - integrates with standard fine-tuning pipelines (LoRA, full fine-tuning)
- **Training-time solution** - prevents misalignment rather than correcting it post-hoc

---

## 1. Problem Statement

### 1.1 Empirical Observation

**Finding from literature:** MLLMs exhibit consistent performance degradation from English → target language → foreign languages, even for visual concepts seen during pre-training ([Traveling Across Languages, arXiv 2505.15075](https://arxiv.org/html/2505.15075v1)).

**Critical insight:** Models fail to leverage visual memory multilingually, indicating a fundamental disconnect between multimodal training and cross-lingual use.

### 1.2 Mechanistic Hypothesis

When fine-tuning an MLLM on language L2:

1. **LLM embedding space shifts** to accommodate L2 linguistic patterns
2. **Visual tokens remain fixed** (frozen encoder) or change independently (unfrozen)
3. **Embedding space mismatch** → visual tokens become out-of-distribution
4. **Result:** LLM increasingly ignores visual information, relies on language priors

### 1.3 Current Solutions and Their Limitations

| Approach | Method | Limitation |
|----------|--------|------------|
| **Freeze visual encoder** | Standard practice | Visual tokens can't adapt to LLM shift |
| **Unfreeze visual encoder** | Joint training | Expensive, destabilizes training, no principled guidance |
| **VIRAL** | Align to fixed VFM | Assumes LLM embedding space is stable (breaks during language adaptation) |
| **Contrastive decoding** | Inference-time fix | Treats symptom, not cause |
| **Multilingual pre-training** | Train from scratch | Expensive, not practical for adaptation |

### 1.4 Gap in Literature

**No existing work addresses:**
- Visual token distribution drift during language adaptation
- Dynamic co-adaptation of visual projector with LLM
- Training objective that maintains visual grounding across language shifts

---

## 2. Proposed Method: Visual Drift Compensation (VDC)

### 2.1 Overview

VDC consists of four complementary components:

```
L_total = L_task + λ1·L_drift + λ2·L_contrast + λ3·L_align

where:
- L_task: Standard language modeling loss
- L_drift: Drift compensation loss
- L_contrast: Contrastive visual grounding loss
- L_align: Visual representation alignment loss
```

### 2.2 Component 1: Anchor-Based Drift Tracking

**Motivation:** Quantify how much visual token distributions shift as LLM adapts.

**Inspiration:**
- A3D2 (anchor dragging drift for domain adaptation)
- Cross-modal redundancy research (bimodal vs. unimodal atoms)
- Harmful embedding drift (BOOSTER, ICLR 2025)

**Method:**

1. **Select anchor visual-linguistic concepts:**
   - Use bilingual image-text pairs that should remain consistent
   - Examples: universal objects (person, car, tree), spatial relations (above, left)

2. **Track embedding drift:**
   ```python
   # At training step t
   v_anchor_t = encode_visual(anchor_images)  # Visual tokens for anchors
   l_anchor_t = embed_text(anchor_texts_L2)   # LLM embeddings for anchor concepts

   # Measure drift from initial state (t=0)
   drift = MMD(v_anchor_t, v_anchor_0) + Wasserstein(l_anchor_t, l_anchor_0)
   ```

3. **Drift compensation loss:**
   ```python
   L_drift = MMD(project(v_anchor_t), l_anchor_t)
   ```

   This loss encourages the visual projector to adapt such that projected visual tokens remain close to LLM's current representation of the same concepts.

**Key difference from VIRAL:** VIRAL aligns to a fixed VFM. We align to the **dynamically changing** LLM embedding space.

### 2.3 Component 2: Contrastive Visual Grounding Regularization

**Motivation:** Explicitly force the model to use visual information, preventing visual neglect.

**Inspiration:**
- CLOC (region-text contrastive loss, ICML 2025)
- DPA (Data-augmented Phrase-level Alignment)
- Instruction Contrastive Decoding (ACL 2024)

**Method:**

During training, for each multimodal example (image, text, target):

1. **Forward pass with image:**
   ```python
   logits_visual = model(visual_tokens + text_tokens)
   ```

2. **Forward pass without image (text-only):**
   ```python
   logits_text = model(text_tokens_only)
   ```

3. **Contrastive loss:**
   ```python
   L_contrast = -log(P(target | visual + text) / P(target | text))
                + KL(P_visual || P_text_frozen)
   ```

**Effect:** Penalizes the model if it produces similar outputs with and without the image, forcing visual dependency.

**Key difference from existing work:**
- CLOC: Pre-training only
- ICD: Inference-time decoding
- VDC: Training-time regularization during language adaptation

### 2.4 Component 3: Dynamic Visual Representation Alignment

**Motivation:** Preserve fine-grained visual details while adapting to LLM shift.

**Inspiration:**
- VIRAL (Visual Representation Alignment)
- Modality gap research (layer-wise alignment)

**Method:**

1. **Layer-wise visual alignment:**
   ```python
   for layer_idx in critical_layers:  # Middle layers from modality gap research
       v_mllm = extract_visual_features(mllm, layer_idx)
       v_vfm = extract_features(frozen_vfm, layer_idx)

       L_align += MSE(v_mllm, v_vfm)
   ```

2. **Adaptive weighting based on drift:**
   ```python
   # Increase alignment weight when drift is high
   λ3_adaptive = λ3_base * (1 + α * drift_magnitude)
   ```

**Key difference from VIRAL:** We adaptively increase alignment strength when drift is detected, balancing adaptation with preservation.

### 2.5 Component 4: Layer-Wise Modality-Aware LoRA

**Motivation:** Allocate adaptation capacity efficiently based on modality integration.

**Inspiration:**
- Dynamic Mixture of Curriculum LoRA Experts
- Modality gap layer analysis (ICCV 2025)
- SMoLoRA (ICCV 2025)

**Method:**

1. **Compute Modality Integration Rate (MIR) per layer:**
   ```python
   # From ICCV 2025 paper: deeper layers have higher integration
   MIR[layer] = cross_modal_attention_score(layer) / total_attention(layer)
   ```

2. **Allocate LoRA rank proportionally:**
   ```python
   rank[layer] = base_rank * MIR[layer]
   ```

3. **Separate LoRA for visual projector:**
   ```python
   # Visual projector gets higher rank allocation
   rank[projector] = base_rank * β  # β > 1
   ```

**Effect:** Layers with high visual-linguistic integration (middle-to-deep) get more adaptation capacity.

**Key difference:** First LoRA allocation strategy based on modality integration analysis.

---

## 3. Theoretical Justification

### 3.1 Information-Theoretic View

Let:
- `V` = visual information
- `T_L1` = text in language L1 (English)
- `T_L2` = text in language L2 (target)
- `Y` = task output

**Ideal multilingual MLLM:**
```
I(V; Y | T_L1) ≈ I(V; Y | T_L2)
```
Visual information is equally used regardless of language.

**Current MLLMs (empirical):**
```
I(V; Y | T_L1) >> I(V; Y | T_L2)
```
Visual information is underutilized in L2.

**VDC objective:**
Maximize `I(V; Y | T_L2)` while fine-tuning on L2 data by:
1. **L_drift:** Maintains `I(V; T_L2)` - visual-linguistic mutual information
2. **L_contrast:** Increases conditional entropy `H(Y | T_L2) - H(Y | V, T_L2)`
3. **L_align:** Preserves `I(V; V_frozen)` - visual detail preservation

### 3.2 Embedding Geometry View

From modality gap research:
- Vision and language embeddings are geometrically separated
- Bimodal atoms carry cross-modal alignment signal
- Language adaptation shifts language embedding manifold

**VDC strategy:**
1. Track shift of language manifold (drift measurement)
2. Translate visual manifold accordingly (drift compensation)
3. Maintain bimodal atoms (contrastive regularization)
4. Preserve within-visual structure (alignment)

---

## 4. Experimental Design

### 4.1 Research Questions

**RQ1:** Does visual token distribution drift measurably during language adaptation?
- **Metric:** MMD, Wasserstein distance of visual tokens before/after adaptation
- **Expected:** Significant drift (p < 0.001)

**RQ2:** Does VDC preserve visual grounding better than baselines?
- **Baselines:** Standard LoRA, VIRAL, vision encoder unfreezing
- **Metrics:** Visual grounding accuracy (VQA, counting, spatial)
- **Expected:** VDC > VIRAL > unfreezing > standard LoRA

**RQ3:** Is VDC language-agnostic?
- **Test languages:** Korean, Japanese, Arabic, Thai (diverse scripts/linguistic families)
- **Expected:** Consistent improvement across all languages

**RQ4:** What is the contribution of each component?
- **Ablation:** Remove L_drift, L_contrast, L_align, layer-wise LoRA
- **Expected:** All components contribute, L_drift is most critical

### 4.2 Datasets

**Language Adaptation:**
- Korean: AIHub multimodal data
- Japanese: JaVQA, Japanese COCO captions
- Arabic: XM3600 (Arabic subset)
- Thai: XM3600 (Thai subset)

**Visual Grounding Evaluation:**
- Cross-lingual VQA: xGQA
- Object counting: TallyQA (translated)
- Spatial reasoning: What's Up (multilingual)
- Hallucination: POPE (multilingual version)

### 4.3 Baselines

1. **Standard LoRA:** Adapt LLM with LoRA, freeze visual encoder
2. **VIRAL:** Add visual representation alignment loss
3. **Unfreezing:** Jointly train visual encoder + LLM
4. **Multilingual Pre-training:** Train from scratch (upper bound)

### 4.4 Metrics

**Visual Grounding:**
- VQA accuracy
- Counting MAE
- Spatial relation accuracy
- Hallucination rate (POPE)

**Language Performance:**
- Perplexity on target language
- Language understanding benchmarks

**Efficiency:**
- Training FLOPs
- Memory usage
- Inference speed

---

## 5. Expected Results

### 5.1 Quantitative Predictions

| Method | VQA (Korean) | Counting MAE | Hallucination ↓ | Training Cost |
|--------|--------------|--------------|-----------------|---------------|
| Standard LoRA | 65.2 | 2.8 | 35.6% | 1x |
| VIRAL | 68.4 | 2.3 | 32.1% | 1.2x |
| Unfreezing | 71.2 | 2.0 | 28.9% | 5x |
| **VDC (ours)** | **73.8** | **1.7** | **24.3%** | **1.5x** |
| Multilingual Pre-train | 76.1 | 1.4 | 21.2% | 50x |

### 5.2 Qualitative Predictions

**Drift Measurement:**
- Significant visual token drift in standard LoRA
- VDC shows minimal drift throughout training

**Attention Patterns:**
- Standard LoRA: Decreasing attention to visual tokens over training
- VDC: Stable visual attention allocation

**Cross-Lingual Consistency:**
- Standard LoRA: Performance English >> Korean >> Japanese
- VDC: More uniform performance across languages

---

## 6. Novel Contributions

### 6.1 Scientific Contributions

1. **First mechanistic explanation** of visual grounding degradation during language adaptation
2. **Novel drift measurement framework** for multimodal embedding spaces
3. **Theoretical connection** between modality gap and cross-lingual transfer

### 6.2 Technical Contributions

1. **Training-time solution** vs. inference-time band-aids
2. **Composable framework** that works with any MLLM architecture
3. **Efficient implementation** using LoRA and selective parameter updates

### 6.3 Practical Contributions

1. **Language-agnostic method** applicable to any language
2. **Reduced need for multilingual pre-training**
3. **Better resource utilization** for multilingual MLLM deployment

---

## 7. Comparison to Related Work

### 7.1 Visual Representation Alignment (VIRAL)

| Aspect | VIRAL | VDC |
|--------|-------|-----|
| **Alignment target** | Fixed VFM | Dynamic LLM embedding |
| **Problem setting** | General fine-tuning | Language adaptation |
| **Drift awareness** | No | Yes (core component) |
| **Contrastive loss** | No | Yes |
| **LoRA strategy** | Uniform | Layer-wise modality-aware |
| **Cross-lingual eval** | No | Yes |

**Why VDC is better for language adaptation:** VIRAL assumes stable LLM embedding space, which breaks during language shift.

### 7.2 Anchor Dragging Drift (A3D2)

| Aspect | A3D2 | VDC |
|--------|------|-----|
| **Problem** | Domain adaptation | Language adaptation |
| **Modality shift** | Visual domain → visual domain | Language L1 → language L2 |
| **Application** | Cross-domain transfer | Cross-lingual transfer |
| **Training objective** | Adversarial alignment | Contrastive + alignment |

**Why VDC is novel:** A3D2 handles visual domain shift; VDC handles linguistic shift while preserving visual grounding.

### 7.3 Contrastive Decoding (ICD)

| Aspect | ICD | VDC |
|--------|-----|-----|
| **When** | Inference time | Training time |
| **Mechanism** | Contrast outputs ±visual | Contrast training gradients |
| **Cost** | 2x inference | Minimal training overhead |
| **Prevents problem** | No | Yes |

**Why VDC is better:** Training-time prevention vs. inference-time correction.

### 7.4 Vision Encoder Unfreezing

| Aspect | Unfreezing | VDC |
|--------|------------|-----|
| **Visual adaptation** | Yes | Yes |
| **Guidance** | None (joint training) | Principled (drift-aware) |
| **Efficiency** | 5x cost | 1.5x cost |
| **Stability** | Requires low LR | Stable |

**Why VDC is better:** Efficient, stable, principled adaptation.

---

## 8. Potential Challenges and Mitigation

### 8.1 Challenge: Anchor Selection

**Problem:** Which visual-linguistic concepts should be anchors?

**Mitigation:**
- Use universal concepts with cross-lingual translations
- Automatically select based on semantic stability
- Ablation study on anchor set size/diversity

### 8.2 Challenge: Hyperparameter Sensitivity

**Problem:** Balancing λ1, λ2, λ3 weights.

**Mitigation:**
- Grid search on validation set
- Adaptive weighting based on drift magnitude
- Provide default values from pilot experiments

### 8.3 Challenge: Computational Overhead

**Problem:** Additional loss computations.

**Mitigation:**
- L_drift: Computed on anchor subset (1-5% of batch)
- L_contrast: Text-only forward pass is fast (no visual encoder)
- L_align: Only on critical layers (3-5 layers)

### 8.4 Challenge: Generalization to Other Languages

**Problem:** Will it work for low-resource languages?

**Mitigation:**
- Design is language-agnostic
- No language-specific components
- Test on typologically diverse languages (Korean, Arabic, Thai)

---

## 9. Broader Impact

### 9.1 Multilingual AI Accessibility

- Reduces cost of adapting MLLMs to non-English languages
- Enables better visual understanding for global users
- Democratizes multimodal AI beyond English-centric models

### 9.2 Foundation Model Adaptation

- General framework applicable beyond language adaptation
- Could extend to domain adaptation, task adaptation
- Provides principled approach to fine-tuning foundation models

### 9.3 Cross-Modal Learning Theory

- Advances understanding of embedding space dynamics
- Connects modality gap with cross-lingual transfer
- Provides measurement framework for multimodal drift

---

## 10. Timeline and Milestones

### Phase 1: Proof of Concept (4 weeks)

- [ ] Implement drift measurement on existing Korean MLLM
- [ ] Quantify visual token drift during language adaptation
- [ ] Baseline visual grounding evaluation

### Phase 2: Method Development (6 weeks)

- [ ] Implement VDC components incrementally
- [ ] Ablation studies on each component
- [ ] Hyperparameter tuning

### Phase 3: Comprehensive Evaluation (6 weeks)

- [ ] Multi-language experiments (Korean, Japanese, Arabic, Thai)
- [ ] Comparison with all baselines
- [ ] Statistical analysis and visualization

### Phase 4: Paper Writing (4 weeks)

- [ ] Results compilation
- [ ] Theoretical analysis refinement
- [ ] Paper submission to EMNLP/NeurIPS

---

## 11. Success Criteria

### Minimum Viable Contribution

1. **Empirical proof:** Visual token drift exists and is measurable
2. **Method validation:** VDC improves visual grounding on ≥2 languages
3. **Efficiency:** Training cost < 2x baseline

### Strong Contribution

1. All minimum criteria +
2. Consistent improvement across ≥4 typologically diverse languages
3. Ablation showing all components contribute
4. Theoretical analysis explaining why VDC works

### Exceptional Contribution

1. All strong criteria +
2. State-of-the-art visual grounding on multilingual benchmarks
3. General framework applicable to other adaptation scenarios
4. Open-source implementation with reproducible results

---

## 12. References

See companion document: `visual-grounding-preservation-literature-review.md` for comprehensive references.

### Key Foundational Papers

- **VIRAL:** Visual Representation Alignment for MLLMs (arXiv 2509.07979)
- **A3D2:** Adversarial Alignment with Anchor Dragging Drift (ACL 2025)
- **Modality Gap:** Cross-Modal Redundancy and Geometry (arXiv 2602.06218)
- **Cross-Lingual Degradation:** Traveling Across Languages (arXiv 2505.15075)
- **CLOC:** Contrastive Localized Language-Image Pre-Training (ICML 2025)
- **Visual Attention Sink:** See What You Are Told (ICLR 2025)
- **BOOSTER:** Harmful Embedding Drift (ICLR 2025)

---

## Conclusion

Visual Drift Compensation represents a novel, principled approach to preserving visual grounding during cross-lingual MLLM adaptation. By explicitly modeling and compensating for embedding space dynamics, VDC addresses a critical gap in current multilingual multimodal AI research. The proposed method combines insights from multiple research areas (modality gap, embedding drift, contrastive learning) into a unified framework that is both theoretically motivated and practically implementable.

**Next step:** Begin Phase 1 proof-of-concept experiments to validate the core hypothesis that visual token drift is measurable and correlated with visual grounding degradation.
