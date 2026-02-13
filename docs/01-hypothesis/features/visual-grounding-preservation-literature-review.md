# Literature Review: Visual Grounding Preservation in Non-English MLLM Adaptation
**Date:** 2025-02-13
**Status:** Literature Review
**Goal:** Identify gaps in existing methods for preserving visual grounding during non-English language adaptation of multimodal LLMs

---

## Executive Summary

This literature review surveys recent research (2024-2025) across five critical areas to identify gaps for a novel method that preserves visual grounding during non-English language adaptation of MLLMs. The review reveals that **while individual components exist (attention steering, visual representation alignment, contrastive training), no existing work systematically addresses the embedding drift problem that occurs when adapting MLLMs to new languages while maintaining visual grounding.**

### Key Gap Identified

**None of the surveyed methods address the core problem: visual tokens becoming "out-of-distribution" as the LLM's embedding space shifts during language adaptation.** This represents a significant opportunity for novel research.

---

## 1. Attention Key-Space Manipulation in Transformers

### 1.1 Representation Steering and Activation Engineering (2024-2025)

**Paper:** Instruction Attention Boosting (INSTABOOST)
**Venue:** ICLR 2025
**Key Technique:** Applies multiplicative boost to attention scores of instruction tokens
**Limitation:** Static steering vectors that don't adapt to semantic context
**Difference from our goal:** Steers generation behavior, not visual token adaptation during training

**Paper:** Conceptor-Based Affine Steering
**Year:** 2025
**Key Technique:** Conceptors as soft projection matrices for provably optimal affine steering
**Limitation:** Focuses on controlling LLM behavior at inference time, not training-time adaptation
**Difference from our goal:** No consideration of visual modality or cross-lingual drift

**Paper:** Control Reinforcement Learning (CRL) via SAE Features
**Year:** 2025
**Key Technique:** Uses sparse autoencoders to extract interpretable features for steering
**Limitation:** Designed for text-only models, no visual component
**Difference from our goal:** Inference-time intervention, not training-time visual-linguistic co-adaptation

### 1.2 Visual Attention Analysis

**Paper:** "See What You Are Told: Visual Attention Sink in Large Multimodal Models"
**Venue:** ICLR 2025
**Authors:** Seil Kang et al., Yonsei University
**Key Finding:** MLLMs allocate high attention to specific visual tokens even when irrelevant
**Solution:** Visual Attention Redistribution (VAR) - training-free method
**Limitation:** Inference-time fix, doesn't prevent the underlying misalignment during training
**Difference from our goal:** Addresses symptom (attention sink) not cause (embedding drift during fine-tuning)

### Gap Analysis
- **No work on key-space adaptation during cross-lingual fine-tuning**
- All methods are either inference-time interventions or text-only
- None address the fundamental problem of keeping visual keys aligned as LLM embedding space shifts

---

## 2. Visual Token Adaptation During LLM Fine-Tuning

### 2.1 Visual Representation Alignment

**Paper:** VIRAL (Visual Representation Alignment for MLLMs)
**Year:** 2025 (arXiv: 2509.07979)
**Authors:** KAIST CVLAB
**Key Technique:** Regularization that aligns intermediate visual features with pretrained VFM representations
**Training Objective:** L_VIRAL = MSE(f_MLLM^visual, f_VFM^frozen)
**Results:** +9.4% average improvement, especially on vision-centric tasks
**Limitation:**
- Aligns to a **fixed** frozen VFM
- Doesn't account for LLM embedding space drift during language adaptation
- English-centric evaluation only
**Difference from our goal:**
- VIRAL maintains alignment to original vision encoder
- Our goal: adapt visual representations to **track** LLM's shifting embedding space during language fine-tuning

**Critical insight:** VIRAL prevents forgetting of visual details, but assumes LLM embedding space is stable. This assumption breaks during cross-lingual adaptation.

### 2.2 Vision Encoder Unfreezing

**Paper:** Various 2024-2025 studies on vision encoder unfreezing
**Key Finding:** Unfreezing vision encoder improves performance but causes language capability degradation
**Training Challenge:** Requires very low learning rate for stable optimization
**Observation:** Multi-modality performance ↑, language capability ↓
**Limitation:**
- High computational cost
- No principled method to balance visual and language objectives
- Doesn't specifically address cross-lingual transfer
**Difference from our goal:** Brute-force approach without understanding embedding space dynamics

### 2.3 Dynamic Projector Adaptation

**Paper:** Language-guidance Visual Projector (LVP)
**Year:** 2025
**Key Technique:** Uses text features to dynamically select important visual tokens
**Limitation:** Token selection, not representation adaptation
**Difference from our goal:** Addresses efficiency, not distribution shift

**Paper:** Visual Concept Modeling (VCM)
**Key Technique:** 85% FLOPs reduction through dynamic token selection
**Limitation:** Inference-time optimization, not training-time adaptation

### Gap Analysis
- **VIRAL is closest but doesn't handle embedding drift**
- No work on co-adapting visual projector as LLM adapts to new language
- All methods either freeze visual encoder or unfreeze without principled guidance

---

## 3. Contrastive/Grounding-Aware Training Objectives for MLLMs

### 3.1 Visual Hallucination Reduction

**Paper:** Data-augmented Phrase-level Alignment (DPA)
**Year:** 2025
**Venue:** Various EMNLP/ACL proceedings
**Training Objective:**
```
L_DPA = log(P_hallucinated / P_correct) + KL(model || frozen_reference)
```
**Limitation:** Requires paired correct/hallucinated examples, no cross-lingual component
**Difference from our goal:** Post-hoc correction vs. proactive grounding preservation

**Paper:** Hallucination-targeted DPO (HDPO)
**Year:** 2024-2025
**Key Technique:** Preference optimization targeting hallucination
**Limitation:** Requires preference pairs, language-specific annotations
**Difference from our goal:** Doesn't address visual token distribution shift

**Paper:** Semantic Curriculum Preference Optimization (SCPO)
**Key Technique:** Extension of DPO to MLLMs
**Limitation:** Keeps visual input fixed, only varies text responses
**Difference from our goal:** Doesn't adapt visual representations

### 3.2 Contrastive Training Methods

**Paper:** Contrastive Localized Language-Image Pre-Training (CLOC)
**Venue:** ICML 2025
**Authors:** Apple ML Research
**Key Technique:** Region-text contrastive loss complementing CLIP
**Training Objective:**
```
L_CLOC = L_CLIP + L_region-text
```
**Results:** High-quality regional embeddings, drop-in CLIP replacement
**Limitation:** Pre-training only, not for fine-tuning/adaptation
**Difference from our goal:** Doesn't address language shift during fine-tuning

**Paper:** Dual Contrastive Decoding
**Year:** 2024
**Key Technique:** Contrasts model outputs with/without visual prompts
**Limitation:** Inference-time intervention
**Difference from our goal:** Doesn't prevent misalignment during training

### 3.3 Contrastive Decoding for Visual Grounding

**Paper:** Instruction Contrastive Decoding (ICD)
**Venue:** ACL 2024
**Key Technique:** Contrastive decoding to mitigate hallucinations
**Limitation:** Inference-time method

**Paper:** TWIST & SCOUT
**Venue:** ICCV 2025
**Key Technique:** Twin-expert stepwise tuning for visual grounding
**Limitation:** Requires separate expert modules, increases model complexity
**Difference from our goal:** Architectural change vs. training objective change

### Gap Analysis
- **All contrastive methods are either pre-training or inference-time**
- No contrastive loss that explicitly maintains visual grounding during language fine-tuning
- No work on using contrastive objectives to prevent visual token drift

---

## 4. Cross-Lingual Transfer in Multimodal Models

### 4.1 Cross-Lingual Performance Degradation

**Paper:** "Traveling Across Languages: Benchmarking Cross-Lingual Consistency in MLLMs"
**Year:** 2025 (arXiv: 2505.15075)
**Key Finding:** Performance consistently declines from English → local language → foreign language
**Critical Observation:** Models that "saw" landmarks during training fail to leverage visual memory multilingually
**Implication:** **Fundamental disconnect between multimodal training and multilingual use**
**Gap:** No explanation or solution for why visual grounding degrades with language

**Paper:** MultimodalX
**Year:** 2024-2025
**Key Approach:** Multilingual multimodal pre-training
**Limitation:** Requires massive multilingual data from scratch
**Difference from our goal:** Pre-training vs. efficient adaptation

### 4.2 Zero-Shot Cross-Lingual Transfer

**Paper:** "Multilingual Multimodal Pre-training for Zero-Shot Cross-Lingual Transfer"
**Key Finding:** Visual concepts are universal but require proper alignment with text in each language
**Observation:** Conventional models handle this poorly → performance gap
**Limitation:** Proposes expensive pre-training, not efficient fine-tuning

**Paper:** "Multilingual Diversity Improves Vision-Language Representations"
**Year:** 2024 (arXiv: 2405.16915)
**Key Finding:** Multilingual data improves vision-language representations
**Limitation:** Doesn't explain WHY monolingual fine-tuning degrades visual grounding

### 4.3 Regional Language Models

**Paper:** Qwen2.5
**Languages:** 29+ including Chinese, Japanese, Korean, Thai, Arabic, Vietnamese
**Modality:** Multimodal (text + images)
**Limitation:** Black-box approach, no analysis of visual grounding across languages

**Paper:** SeaLLMs
**Languages:** Southeast Asian languages (Thai, Vietnamese, Indonesian, etc.)
**Focus:** Language modeling
**Limitation:** Primarily text-focused, minimal multimodal evaluation

### Gap Analysis
- **Strong empirical evidence that visual grounding degrades cross-lingually**
- **No mechanistic understanding of WHY this happens**
- **No proposed solution beyond expensive multilingual pre-training**
- Critical opportunity: explain and fix the visual grounding degradation during language adaptation

---

## 5. Representation Drift / Embedding Drift

### 5.1 Drift Detection and Quantification

**Paper:** "Handling LLM Model Drift in Production"
**Year:** 2024-2025
**Key Techniques:**
- Population Stability Index (PSI)
- KL Divergence for distribution shift
- Embedding-based drift detection
**Limitation:** Monitoring methods, not prevention methods
**Application:** Production ML systems, not research

**Paper:** "Embedding Drift in Fine-Tuning"
**Key Finding:** Statistical properties of embeddings change during fine-tuning
**Mitigation:** Drift detection triggers retraining
**Limitation:** Reactive approach, doesn't prevent drift

### 5.2 Harmful Embedding Drift

**Paper:** BOOSTER - "Tackling Harmful Fine-Tuning for Large Language Models"
**Venue:** ICLR 2025
**Key Concept:** "Harmful embedding drift" - drift of embeddings over alignment data
**Measurement:** Distance between embeddings before/after fine-tuning
**Mitigation:** Vaccine, CTRL - minimax solutions at alignment stage
**Limitation:**
- Focuses on safety alignment, not visual-linguistic alignment
- Text-only models
**Difference from our goal:** We need to measure and compensate for visual-linguistic drift, not safety drift

### 5.3 Anchor-Based Drift Compensation

**Paper:** "LLM Alignment with Anchor Words Tuning"
**Venue:** EMNLP 2025 Findings
**Key Technique:** Select anchor words, optimize hidden states to maximize value distinction
**Limitation:** Text-only, single modality
**Difference from our goal:** Need visual-linguistic anchors

**Paper:** "Adversarial Alignment with Anchor Dragging Drift (A3D2)"
**Venue:** ACL 2025
**Key Technique:** Multimodal domain adaptation with partially shifted modalities
**Important:** **Addresses modality drift in domain adaptation!**
**Limitation:** Domain adaptation (different visual domains), not language adaptation
**Closest work to our goal but different problem setting**

**Paper:** Cross-Modal Feature Alignment and MMD
**Venue:** WACV 2025
**Key Technique:** Maximum Mean Discrepancy (MMD) for cross-modal alignment robustness
**Limitation:** Prompt tuning context, not language adaptation

### 5.4 Modality Gap

**Paper:** "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning"
**Venue:** ICLR 2025
**Key Finding:** Vision and language modalities are embedded "at arm's length" (geometric separation)
**Proposed Solution:** Parameter sharing between encoders + intra-modality separation objective
**Limitation:** Pre-training approach, doesn't address drift during fine-tuning

**Paper:** "Diffusion Bridge: Leveraging Diffusion Model to Reduce the Modality Gap"
**Venue:** CVPR 2025
**Key Technique:** Diffusion models to learn intrinsic distribution of text embeddings
**Limitation:** Computational overhead, doesn't address language adaptation scenario

**Paper:** "Deciphering Cross-Modal Alignment via Modality Integration Rate"
**Venue:** ICCV 2025
**Key Finding:**
- Modality gap is large at shallow layers
- Gap narrows through middle layers
- Alignment achieved at deeper layers
**Implication:** Layer-wise analysis is crucial
**Difference from our goal:** Static analysis, not dynamic adaptation

**Paper:** "Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings"
**Year:** 2025 (arXiv: 2602.06218)
**Key Finding:**
- Sparse bimodal atoms carry cross-modal alignment signal
- Unimodal atoms explain modality gap and can be removed
**Critical Insight:** **Decomposition of embeddings into unimodal and bimodal components**
**Potential application:** Use this decomposition to track drift in bimodal components

### Gap Analysis
- **Strong evidence that embedding drift exists and can be measured**
- **A3D2 addresses modality drift in domain adaptation** (closest related work)
- **No work on embedding drift during language adaptation**
- **No anchor-based methods for visual-linguistic co-adaptation**

---

## 6. Catastrophic Forgetting in MLLMs

### 6.1 Continual Learning for MLLMs

**Paper:** "Investigating the Catastrophic Forgetting in Multimodal Large Language Models"
**Year:** 2024
**Key Finding:** Fine-tuned MLLMs fail to retain pre-trained performance
**Benchmark:** EMT (Evaluating MulTimodality)
**Observation:** Early-stage fine-tuning on images improves alignment of text and visual features

**Paper:** "Modality-Inconsistent Continual Learning of MLLMs"
**Key Problem:** Sequential tasks with different modality requirements
**Limitation:** Focuses on task sequences, not language adaptation

**Paper:** SMoLoRA - "Exploring and Defying Dual Catastrophic Forgetting"
**Venue:** ICCV 2025
**Key Technique:** Sparse Mixture of LoRA experts
**Observation:** Uniform LoRA allocation causes redundancy and under-adaptation
**Proposed:** Layer-wise curriculum for LoRA allocation
**Limitation:** Continual vision-language learning, not language adaptation

### 6.2 Mitigation Approaches

**Paper:** "Model Tailor"
**Venue:** CVPR 2024
**Key Technique:** Tailored parameter updates to mitigate forgetting
**Limitation:** General continual learning, not cross-lingual specific

**Paper:** MoInCL - Pseudo-target Generation
**Key Technique:** Synthesize input-target pairs for effective forgetting mitigation
**Limitation:** Requires generating synthetic data

**Paper:** MAFED - Modality-Aware Feature Distillation
**Key Technique:** Mix old samples with new data + feature distillation
**Limitation:** Requires storing old data

### Gap Analysis
- **Catastrophic forgetting is well-studied in continual learning**
- **Visual feature forgetting during language adaptation is not specifically addressed**
- **No work connecting embedding drift with catastrophic forgetting of visual grounding**

---

## 7. Parameter-Efficient Fine-Tuning for MLLMs

### 7.1 LoRA Adaptations

**Paper:** VIRAL (Visual Representation Alignment)
**LoRA Usage:** 3-layer MLP projector with LoRA for efficient adaptation
**Finding:** LoRA achieves comparable performance to full fine-tuning

**Paper:** "Dynamic Mixture of Curriculum LoRA Experts"
**Year:** 2025
**Key Finding:** Uniform LoRA allocation is inefficient
**Proposed:** Layer-wise curriculum - prioritize layers that contribute most
**Limitation:** Doesn't address modality-specific adaptation

**Paper:** LoRA-PRO
**Venue:** ICLR 2025
**Key Question:** Are low-rank adapters properly optimized?
**Finding:** Learning rate strategies matter for LoRA matrices A and B

**Paper:** VLoRA - Visual LoRA
**Key Technique:** Generates low-rank perceptual weights for visual information
**Limitation:** Limited documentation, unclear how it differs from standard LoRA

**Paper:** MMLoRA - Multitask Memory LoRA
**Application:** Multimodal speech emotion recognition
**Limitation:** Different domain (audio+text, not vision+text)

### Gap Analysis
- **LoRA is widely used for efficient MLLM adaptation**
- **No LoRA variant specifically designed for visual token adaptation during language fine-tuning**
- **Dynamic/layer-wise LoRA allocation is emerging but not modality-aware**

---

## Summary of Key Gaps

### Critical Gap: No Systematic Solution for Visual Embedding Drift During Language Adaptation

| Problem Component | Existing Work | Gap |
|------------------|---------------|-----|
| **Embedding drift measurement** | BOOSTER (safety drift), A3D2 (domain drift) | No visual-linguistic drift during language adaptation |
| **Visual representation preservation** | VIRAL (fixed VFM alignment) | Doesn't track LLM embedding shift |
| **Contrastive grounding** | CLOC (pre-training), ICD (inference) | No training-time contrastive loss for adaptation |
| **Cross-lingual degradation** | Empirical observations only | No mechanistic explanation or solution |
| **Attention adaptation** | VAR (inference-time), INSTABOOST (text-only) | No training-time visual key adaptation |
| **Modality gap** | Static analysis, pre-training solutions | No dynamic adaptation during fine-tuning |

### The Novel Method We Can Propose

**Core Innovation:** A training objective that actively adapts visual token representations to stay "in-distribution" as the LLM's embedding space shifts during language fine-tuning.

**Key Components:**

1. **Anchor-Based Drift Tracking**
   - Inspired by: A3D2, anchor words tuning
   - Novel: Track drift of visual-linguistic bimodal atoms (from modality gap research)
   - Measure: Distance between visual token distributions before/after language update

2. **Dynamic Visual Projector Adaptation**
   - Inspired by: VIRAL's alignment objective
   - Novel: Instead of aligning to fixed VFM, align to keep visual tokens in-distribution relative to LLM's current embedding space
   - Loss: L_drift = MMD(visual_tokens_current, visual_tokens_adapted)

3. **Contrastive Visual Grounding Regularization**
   - Inspired by: CLOC, DPA
   - Novel: Image-conditioned vs. text-only contrastive loss during language fine-tuning
   - Loss: L_contrast = -log[P(output|image,text) / P(output|text)]

4. **Layer-Wise Modality-Aware LoRA**
   - Inspired by: Dynamic LoRA curriculum, modality gap layer analysis
   - Novel: Allocate LoRA ranks based on layer-wise modality integration rate
   - Deeper layers (high integration) get more visual adaptation budget

**Why It's Novel:**

1. **First work to explicitly model and compensate for visual embedding drift during language adaptation**
2. **Principled motivation from modality gap and embedding drift literature**
3. **Language-agnostic** - works for any language, not just Korean
4. **Composable** - can be added to standard fine-tuning pipelines
5. **Addresses root cause** - not inference-time band-aid

**Comparison to Closest Related Work:**

| Method | VIRAL | A3D2 | Our Method |
|--------|-------|------|------------|
| Problem | Visual detail loss | Domain shift | Language shift |
| Solution | Align to fixed VFM | Anchor dragging | Dynamic co-adaptation |
| When | During any fine-tuning | Domain adaptation | Language adaptation |
| Visual adaptation | No | Yes | Yes |
| LLM tracking | No | No | Yes |

---

## Recommended Next Steps

1. **Quantify the problem first:**
   - Fine-tune InternVL on Korean data
   - Measure visual token distribution drift using MMD/Wasserstein distance
   - Show visual grounding degradation on multilingual benchmarks

2. **Baseline comparisons:**
   - VIRAL (fixed VFM alignment)
   - Standard LoRA fine-tuning
   - Vision encoder unfreezing

3. **Proposed method components:**
   - Start with drift measurement + simple compensation
   - Add contrastive regularization
   - Finally add dynamic LoRA allocation

4. **Evaluation:**
   - Multilingual benchmarks (not just Korean)
   - Visual grounding tasks (VQA, counting, spatial reasoning)
   - Language performance (to show no degradation)

---

## Sources

### Attention Key-Space Manipulation
- [Instruction Attention Boosting - ICLR 2025](https://openreview.net/pdf?id=xDyJVMnab8)
- [Conceptor-Based Affine Steering - OpenReview](https://openreview.net/forum?id=0Yu0eNdHyV)
- [Control Reinforcement Learning via SAE Features](https://arxiv.org/html/2602.10437)
- [Visual Attention Sink - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Paper-Conference.pdf)
- [Visual Attention Sink - OpenReview](https://openreview.net/forum?id=7uDI7w5RQA)

### Visual Token Adaptation
- [VIRAL - Visual Representation Alignment](https://arxiv.org/pdf/2509.07979)
- [VIRAL - GitHub](https://github.com/cvlab-kaist/VIRAL)
- [PruneVid - Visual Token Pruning](https://aclanthology.org/2025.findings-acl.1024.pdf)
- [Token Pruning in MLLMs](https://aclanthology.org/2025.findings-acl.802.pdf)
- [Fast-Slow Efficient Training](https://arxiv.org/html/2602.03815)
- [Language-Guided Visual Projector](https://openreview.net/forum?id=PxBzxO02Ef)

### Contrastive Training & Visual Grounding
- [Towards Visual Grounding Survey - TPAMI 2025](https://github.com/linhuixiao/Awesome-Visual-Grounding)
- [Contrastive Region Guidance - ECCV 2024](https://dl.acm.org/doi/10.1007/978-3-031-72986-7_12)
- [Dual Contrastive Decoding](https://dl.acm.org/doi/10.1145/3743093.3770933)
- [TWIST & SCOUT - ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Bhowmik_TWIST__SCOUT_Grounding_Multimodal_LLM-Experts_by_Forget-Free_Tuning_ICCV_2025_paper.pdf)
- [VGent - Visual Grounding](https://arxiv.org/abs/2512.11099)

### Visual Hallucination Reduction
- [Awesome MLLM Hallucination](https://github.com/showlab/Awesome-MLLM-Hallucination)
- [MLLM Hallucination Survey](https://arxiv.org/pdf/2404.18930)
- [Antidote - CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Antidote_A_Unified_Framework_for_Mitigating_LVLM_Hallucinations_in_Counterfactual_CVPR_2025_paper.pdf)
- [DPA - Data-augmented Phrase-level Alignment](https://aclanthology.org/2025.findings-acl.850.pdf)
- [SCPO - Semantic Curriculum Preference Optimization](https://arxiv.org/html/2509.24491)

### Cross-Lingual Transfer
- [Traveling Across Languages](https://arxiv.org/html/2505.15075v1)
- [MultimodalX - Multilingual Pretraining](https://bhakta-works.medium.com/multimodalx-advancing-vision-language-models-with-multilingual-multimodal-pretraining-ea260ef1a4a8)
- [Multilingual Diversity Improves VLMs](https://arxiv.org/abs/2405.16915)
- [Multilingual LLM Survey - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11783891/)
- [Speaking in Code: Southeast Asia LLMs](https://carnegieendowment.org/research/2025/01/speaking-in-code-contextualizing-large-language-models-in-southeast-asia?lang=en)
- [SeaLLMs - Southeast Asian Languages](https://huggingface.co/SeaLLMs)

### Embedding Drift & Representation Drift
- [Handling LLM Model Drift](https://www.rohan-paul.com/p/ml-interview-q-series-handling-llm)
- [BOOSTER - Harmful Fine-Tuning - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/a7ac8a21e5a27e7ab31a5f42a0117bdb-Paper-Conference.pdf)
- [LLM Alignment with Anchor Words - EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.317.pdf)
- [A3D2 - Anchor Dragging Drift - ACL 2025](https://aclanthology.org/2025.acl-long.967/)
- [Cross-Modal Feature Alignment - WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Sun_Cross-Modal_Feature_Alignment_and_MMD_Improve_Robustness_of_Prompt_Tuning_WACV_2025_paper.pdf)

### Modality Gap
- [Mitigate the Gap - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc1de06a58ba1db43538a37e076e466d-Paper-Conference.pdf)
- [Mind the Modality Gap - Remote Sensing](https://www.sciencedirect.com/science/article/pii/S092427162500245X)
- [Cross-Modal Redundancy and Geometry](https://arxiv.org/html/2602.06218)
- [Deciphering Cross-Modal Alignment - ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_Deciphering_Cross-Modal_Alignment_in_Large_Vision-Language_Models_via_Modality_Integration_ICCV_2025_paper.pdf)
- [Accept the Modality Gap - CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ramasinghe_Accept_the_Modality_Gap_An_Exploration_in_the_Hyperbolic_Space_CVPR_2024_paper.pdf)
- [Diffusion Bridge - CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Diffusion_Bridge_Leveraging_Diffusion_Model_to_Reduce_the_Modality_Gap_CVPR_2025_paper.pdf)

### Catastrophic Forgetting
- [LLM Continual Learning Survey - CSUR 2025](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
- [Investigating Catastrophic Forgetting in MLLMs](https://www.semanticscholar.org/paper/Investigating-the-Catastrophic-Forgetting-in-Large-Zhai-Tong/a281094d05e96b7cca044fdd87ff7c3c65649e20)
- [Modality-Inconsistent Continual Learning](https://arxiv.org/html/2412.13050)
- [SMoLoRA - ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.pdf)

### LoRA & Parameter-Efficient Fine-Tuning
- [Dynamic Mixture of Curriculum LoRA Experts](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2025_Dynamic%20Mixture%20of%20Curriculum%20LoRA%20Experts%20for%20Continual%20Multimodal%20Instruction%20Tuning.pdf)
- [LoRA-PRO - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/ea184f920a0f0f8d8030aa1bd7ac9fd4-Paper-Conference.pdf)
- [LoRA Original Paper](https://arxiv.org/abs/2106.09685)

### Contrastive Learning & Pre-training
- [CLOC - Contrastive Localized Pre-Training - ICML 2025](https://arxiv.org/html/2410.02746v1)
- [CLOC - Apple ML Research](https://machinelearning.apple.com/research/contrastive-localized)
- [Instruction Contrastive Tuning - OpenReview](https://openreview.net/forum?id=OZdr2mV5EI)
- [Aligning VLMs with Contrastive Learning - Amazon](https://assets.amazon.science/36/5c/19734bdf4fdb8da3cc809590c05d/aligning-vision-language-models-with-contrastive-learning.pdf)

### Vision Encoders & Architecture
- [Words Over Pixels - IJCAI 2025](https://www.ijcai.org/proceedings/2025/1164.pdf)
- [Florence-VL - CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Florence-VL_Enhancing_Vision-Language_Models_with_Generative_Vision_Encoder_and_Depth-Breadth_CVPR_2025_paper.pdf)
- [Vision Language Models Survey - 26K Papers](https://arxiv.org/pdf/2510.09586)
- [ViCA: Vision-Only Cross-Attention](https://arxiv.org/html/2602.07574v1)

### General MLLM Resources
- [Awesome Multimodal LLMs](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- [Efficient MLLMs Survey - Springer](https://link.springer.com/article/10.1007/s44267-025-00099-6)
- [Will MLLMs Achieve Deep Understanding - Frontiers](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2025.1683133/full)
