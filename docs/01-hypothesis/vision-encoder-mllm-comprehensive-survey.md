# Comprehensive Survey: Vision Encoders & MLLM Architectures for Korean Bilingual MLLM

> **Research Phase:** 1 - Hypothesis (Literature Review)
> **Created:** 2026-02-12
> **Author:** Juyeon
> **Status:** Draft
> **Purpose:** Inform vision encoder improvement strategy for Korean+English MLLM based on InternVL framework

---

## Table of Contents

1. [Master MLLM Architecture Comparison](#1-master-mllm-architecture-comparison)
2. [Vision Encoder Deep Analysis](#2-vision-encoder-deep-analysis)
3. [Training Data Composition](#3-training-data-composition-by-model)
4. [Feature Extraction Improvement Methods](#4-feature-extraction-improvement-methods)
5. [Projector Architecture Comparison](#5-projector-architecture-comparison)
6. [Position Encoding Methods](#6-position-encoding-methods)
7. [Token Compression Methods](#7-token-compression-methods)
8. [Dynamic Resolution Strategies](#8-dynamic-resolution-strategies)
9. [Trends & Future Directions](#9-trends--future-directions-2024-2025)
10. [Research Gaps & Proposed Direction](#10-research-gaps--proposed-direction)

---

## 1. Master MLLM Architecture Comparison

### 1A. Architecture Overview

| Model | Vision Encoder | Enc Params | Resolution | Projector | LLM Decoder | LLM Params | Total Params |
|-------|---------------|-----------|------------|-----------|-------------|-----------|-------------|
| **InternVL 2.5-78B** | InternViT-6B | 6B | 448 tiles (1-12) | 2-layer MLP | Qwen2.5-72B | 72B | ~78B |
| **InternVL 3.0-78B** | InternViT-6B + V2PE | 6B | 448 tiles (dynamic) | 2-layer MLP | Qwen2.5-72B | 72B | ~78B |
| **Qwen2-VL-72B** | ViT (2D-RoPE) | 675M | Native dynamic | MLP (2x2 merge) | Qwen2-72B | 72B | ~73B |
| **Qwen2.5-VL-72B** | ViT (WinAttn+2D-RoPE) | 675M | Native dynamic | MLP (2x2 merge) | Qwen2.5-72B | 72B | ~73B |
| **LLaVA-OneVision-72B** | SigLIP-SO400M | 400M | AnyRes (up to 10x) | 2-layer MLP | Qwen2-72B | 72B | ~72B |
| **LLaVA-NeXT-72B** | SigLIP-SO400M | 400M | AnyRes (672 max) | 2-layer MLP | Qwen-1.5-72B | 72B | ~72B |
| **Cambrian-1-34B** | CLIP+SigLIP+ConvNeXt+DINOv2 | ~5B combined | Variable | SVA | Hermes-Yi-34B | 34B | ~39B |
| **VILA-1.5-40B** | SigLIP-SO400M | 400M | Dynamic | 2-layer MLP | Yi-34B | 34B | ~34B |
| **Phi-4-Multimodal** | SigLIP-400M (LLM2CLIP) | 440M+370M LoRA | 448x448 | MLP | Phi-4-Mini | 5.6B | 5.6B |
| **Idefics3-8B** | SigLIP-SO400M | 400M | 384 (pixel shuffle 4x) | MLP | Llama-3-8B | 8B | ~8B |
| **MiniCPM-V 2.6** | SigLIP-SO400M | 400M | Up to 1344x1344 | Perceiver (64 tokens) | Qwen2-7B | 7B | ~8B |
| **DeepSeek-VL2** | SigLIP-SO400M-384 | 400M | 384 tiles (dynamic) | MLP | DeepSeekMoE | Various | 16-28B (1-4.5B active) |
| **Molmo-72B** | OpenAI ViT-L/14 | 304M | 336 (multi-crop) | MLP | Qwen2-72B | 72B | ~72B |
| **Pixtral-12B** | Pixtral-ViT (RoPE-2D) | 400M | Native variable | Linear | Mistral-Nemo-12B | 12B | ~12B |
| **CogVLM2-19B** | EVA2-CLIP-E | ~5B | Up to 1344x1344 | Visual Expert | Llama-3-8B | 8B | ~19B |
| **mPLUG-Owl3-7B** | SigLIP-SO400M | 400M | 384x384 | Hyper Attention Block | Qwen2-7B | 7B | ~8B |
| **Monkey-9.8B** | ViT-BigG (CLIP) | 1.9B | Up to 1344x896 | Resampler | Qwen-7B | 7.7B | ~9.8B |
| **VARCO-VISION-2.0-14B** | SigLIP2 | ~400M | 384 (AnyRes) | MLP (LLaVA-OV style) | Qwen3-14B | 14B | ~14B |
| **Ovis-1.6-9B** | SigLIP-SO400M | 400M | Dynamic high-res | Visual Embedding Table | Gemma2-9B | 9B | ~10B |
| **MM1.5-30B** | ViT-H (CLIP, DFN-5B) | ~632M | High-res multi-crop | C-Abstractor (144 tokens) | Custom LLM-30B | 30B | ~30B |

### 1B. Training Pipeline & Resources

| Model | Stages | LoRA/Full FT | Framework | Total Data | Compute |
|-------|--------|-------------|-----------|-----------|---------|
| **InternVL 2.5** | 3 (MLP warmup → ViT incr. → Full FT) | Full FT (S2) | DeepSpeed | ~120B tokens | - |
| **InternVL 3.0** | 3 (Native MM PT → SFT → MPO) | Full FT | DeepSpeed | ~200B tokens | - |
| **Qwen2-VL** | 3+ (ViT PT → VL PT → SFT) | Full FT | Custom | ~1.4T tokens | - |
| **Qwen2.5-VL** | 3+ (ViT scratch → VL PT → SFT) | Full FT | Custom | ~4.1T tokens | - |
| **LLaVA-OneVision** | 4 (Align→Knowledge→SI→OV) | S1: Proj only; rest Full | DeepSpeed | ~5.8M samples | ~$16K A100 |
| **Cambrian-1** | 2 (SVA train → Instruction) | S1: SVA only; S2: SVA+LLM | TPU (JAX) | 9.5M samples | TPU-V4-512 |
| **VILA-1.5** | 3 (Align→PT→SFT) | Full FT (S2/S3) | DeepSpeed | 53M pairs | - |
| **Phi-4-Multimodal** | 4 (Proj→Joint ViT→LoRA LLM→Joint) | S3: LoRA; others Full | Custom | 1.1T img-text+5T text | 512 A100, 28d |
| **MiniCPM-V 2.6** | 3 (Align→PT→SFT) | Full FT | Custom | ~1T tokens | - |
| **DeepSeek-VL2** | 3 (Align→VL PT→SFT) | Full FT | Custom | ~800B tokens | 128-336 A100, 7-14d |
| **Molmo** | 2 (Caption PT→SFT) | Full FT | Custom | ~600K images | - |
| **CogVLM2** | 4 (PT1→PT2→SFT1→SFT2) | Visual Expert trainable | Custom | 1.5B+40M pairs | 8-node A100 |
| **Monkey** | 2 (Align→Instruction) | **LoRA only** (LLM frozen) | Custom | 1.44M samples | **8x RTX 3090** |
| **VARCO-VISION-2.0** | 4 (Align→PT→SFT→DPO) | Full FT (S4: full model) | Custom | 36.9B tokens | - |
| **MM1.5** | 3 (PT→Continual PT→SFT) | Full (all unfrozen) | Custom | 2B pairs+600M interleaved+2T text | - |

### 1C. Benchmark Scores (Largest Variant)

| Model | Params | MMBench-EN | MMMU | ChartQA | DocVQA | OCRBench | MathVista |
|-------|--------|-----------|------|---------|--------|----------|-----------|
| **InternVL 3.0-78B** | 78B | **89.0** | **72.2** | **89.7** | 95.4 | **906** | **79.6** |
| **InternVL 2.5-78B** | 78B | 88.5 | 70.0 | 88.3 | 95.1 | 854 | 72.3 |
| **Qwen2.5-VL-72B** | 73B | 88.6 | 70.2 | 89.5 | **96.4** | 885 | 74.8 |
| **Qwen2-VL-72B** | 73B | 86.5 | 64.5 | 88.3 | 96.5 | 855 | 70.5 |
| **LLaVA-OneVision-72B** | 72B | 86.6 | 57.4 | 84.9 | 93.5 | -- | 66.5 |
| **Pixtral-Large** | 124B | -- | ~65 | 87.2 | 93.2 | -- | 69.4 |
| **Cambrian-1-34B** | 34B | 81.4 | 49.7 | 73.3 | -- | -- | 53.2 |
| **DeepSeek-VL2** | 4.5B active | 83.1 | ~50 | -- | 93.3 | 834 | -- |
| **VARCO-VISION-2.0-14B** | 14B | ~82 | ~48 | -- | -- | -- | -- |
| **MiniCPM-V 2.6** | 8B | ~78 | ~49 | -- | -- | -- | -- |
| **Phi-3.5-Vision** | 4.2B | 81.9 | 43.0 | -- | -- | -- | 43.9 |

---

## 2. Vision Encoder Deep Analysis

### 2A. Encoder Master Comparison

| Encoder | Params | Layers | Heads | Hidden Dim | Patch | Default Res | Position Encoding | Pre-training Obj. | Training Data |
|---------|--------|--------|-------|-----------|-------|------------|-------------------|-------------------|---------------|
| CLIP ViT-L/14 | 307M | 24 | 16 | 1024 | 14 | 224/336 | Learnable absolute | Contrastive (softmax) | WIT 400M / LAION-2B |
| CLIP ViT-H/14 | 986M | 32 | 16 | 1280 | 14 | 224 | Learnable absolute | Contrastive (softmax) | LAION-2B |
| CLIP ViT-bigG/14 | ~1.8B | 48 | 16 | 1664 | 14 | 224 | Learnable absolute | Contrastive (softmax) | LAION-2B |
| **SigLIP SO400M/14** | ~400M | 27 | 16 | 1152 | 14 | 384 | Learnable absolute | Contrastive (sigmoid) | WebLI 10B |
| SigLIP2 SO400M/14 | ~400M | 27 | 16 | 1152 | 14 | 384 | Learnable absolute | Sigmoid+caption+self-distill | WebLI |
| **InternViT-6B** | 5.5B | 45 | 25 | 3200 | 14 | 448 | Learnable abs (V2PE in v3) | Contrastive+generative | Custom mixed |
| **Qwen2-VL ViT** | ~675M | 32 | 16 | 1280 | 14 | Dynamic | **2D-RoPE** | Contrastive (DFN init) | DFN+custom |
| **Qwen2.5-VL ViT** | ~675M | 32 | 16 | 1280 | 14 | Dynamic | **2D-RoPE** | From scratch | Custom |
| EVA-02-L | 304M | 24 | 16 | 1024 | 14 | 224/448 | **2D-RoPE** | MIM (CLIP teacher) | IN-22K+LAION |
| EVA-02-E | ~4.4B | 64 | 16 | 1792 | 14 | 224 | **2D-RoPE** | MIM+CLIP | LAION-2B |
| DFN ViT-H/14 | ~986M | 32 | 16 | 1280 | 14 | 224/378 | Learnable absolute | Contrastive (softmax) | DFN-5B (43B filtered) |
| NaViT | Varies | Varies | Varies | Varies | 16 | Native (any) | Factorized fractional | Contrastive/classification | JFT-4B |
| ConvNeXt-XXL | ~846M | [3,4,30,3] | N/A | [384-3072] | 4 (stem) | Variable | Implicit (CNN) | Supervised/CLIP | LAION/IN-22K |

### 2B. Individual Encoder Analysis

#### CLIP ViT (OpenAI / OpenCLIP)

**Contrastive Learning:** InfoNCE loss with softmax normalization. For N image-text pairs, computes NxN similarity matrix with symmetric cross-entropy. Requires large batch sizes (32K+).

**Position Encoding:** Learnable absolute position embeddings, fixed to training resolution, interpolated for different resolutions. **Weakness:** Does not generalize well to unseen resolutions.

**Strengths:** Strong zero-shot transfer, well-aligned vision-language space, extensive ecosystem.
**Weaknesses:** Fixed resolution, weak at fine-grained spatial/localization, limited OCR capability, CLS token may not capture spatial details.

**Used by:** LLaVA, LLaVA-1.5, Molmo, Cambrian-1, Phi-3-Vision, BLIP-2, MiniGPT-4

#### SigLIP / SigLIP 2 (Google)

**Key Difference from CLIP:**

| Aspect | CLIP (Softmax) | SigLIP (Sigmoid) |
|--------|---------------|-----------------|
| Loss type | Multi-class (softmax CE) | Binary (sigmoid per pair) |
| Normalization | Global (full batch) | Per-pair (independent) |
| Optimal batch | Very large (32K-98K) | Smaller works (32K) |
| Memory efficiency | Lower (full NxN matrix) | Higher (independent pairs) |

**SoViT-400m/14 (Shape-Optimized):** 27 layers, 16 heads, hidden=1152, MLP=4304, patch=14. Dimensions determined by **scaling laws** ("Getting ViT in Shape"). Default 384x384 → 729 tokens.

**Training Data:** WebLI ~10B images, ~12B alt-texts, **109 languages** (multilingual).

**SigLIP 2 (2025) additions:** Captioning pretraining, self-distillation (after 80% training), masked prediction, online data curation, referring expression prediction, grounded captioning. Significantly improved localization and dense prediction.

**Why MLLMs switched to SigLIP:** Better at practical batch sizes, higher resolution (384 vs 224/336), multilingual data, better empirical MLLM performance, 90.3% top-1 fine-tuned ImageNet.

**Used by:** LLaVA-OneVision, DeepSeek-VL2, MiniCPM-V, VILA, mPLUG-Owl3, Ovis, Idefics2/3, Phi-4, Eagle/Eagle2, PaliGemma

#### InternViT-6B (OpenGVLab)

**Architecture:** 5.5B params (last 3 of 48 blocks removed in V1.5+), 45 layers, 25 heads, hidden=3200, MLP=12800, patch=14, 448x448/tile. RMSNorm + QK-Norm. Output after 4th-to-last block used for MLLM.

**Dynamic Resolution (Tiling):** Images divided into 448x448 tiles (1-12 training, up to 40 inference for 4K). Each tile independently encoded. Global thumbnail for context.

**Pixel Unshuffle:** 1024 tokens/tile → **256 tokens/tile** (4x reduction). Groups 2x2 adjacent tokens, 3200 → 12800 channel dim before MLP projection.

**Progressive Training:** Pre-train ViT with contrastive → train with small LLM → scale to larger LLM. ViT trained once, reusable across LLMs.

**Version History:**
- V1.0: 48 layers, initial
- V1.5: 45 layers, dynamic tiling
- V2.0/2.5: Improved data, OCR
- V3.0: **V2PE** (variable position increments for visual tokens)

**Key Insight:** Largest open-source vision encoder (5.5B) **reduces data dependency by 10x** — InternVL 2.5 uses only 120B tokens vs Qwen2-VL's 1.4T.

**Strengths:** Largest open encoder, excellent OCR/document/chart, effective pixel unshuffle, progressive training.
**Weaknesses:** Very large (latency+memory), tiling boundary artifacts, absolute position encoding (pre-V3.0).

#### Qwen2-VL / Qwen2.5-VL ViT (Alibaba)

**Architecture:** ~675M params, 32 layers, 16 heads, hidden=1280, patch=14. Images resized to multiples of 28.

**Native Dynamic Resolution (No Tiling!):**
1. Images resized so H,W are multiples of 28
2. Patchified with stride 14 → variable-length sequence
3. Variable-length sequence directly processed by ViT
4. No padding, no resizing, no tiling

**2D-RoPE Detailed Mechanism:**
- Embedding dimensions split into two halves
- First half: **height position** (row index) rotation
- Second half: **width position** (column index) rotation
- Attention score naturally encodes **relative 2D spatial distance**
- **Key property:** Generalizes to any resolution without interpolation

**M-RoPE (Multimodal Rotary Position Embedding):**

| Component | Text | Image | Video |
|-----------|------|-------|-------|
| Temporal | Same ID | Constant | Increments/frame |
| Height | Same ID | Varies by row | Varies by row |
| Width | Same ID | Varies by col | Varies by col |

**Qwen2.5-VL Improvements:**
- ViT trained from scratch (not DFN init)
- **Window attention** (only 4 full-attention layers, rest window max 8x8) → linear compute scaling
- SwiGLU FFN + RMSNorm
- Actual image scale coordinates (not normalized)

**Strengths:** True native dynamic resolution, 2D-RoPE resolution generalization, unified multimodal position encoding, window attention efficiency.
**Weaknesses:** Smaller encoder (675M vs 5.5B), variable sequence complicates batching, proprietary data.

#### EVA-CLIP / EVA-02 (BAAI)

**Innovations:** SwiGLU FFN, Sub-LN, **2D-RoPE**, Mean Pooling (no CLS token).

**Training:** Stage 1: MIM (reconstruct CLIP vision features of masked patches) on IN-22K. Stage 2: CLIP contrastive on LAION-2B. EVA-02-E: 4.4B vision, 82.0% zero-shot ImageNet.

**Used by:** CogVLM/CogVLM2 (with visual expert modules per transformer layer).

#### DFN ViT (Apple/LAION)

**Core Concept:** Data quality over architecture. Train small "filter" model on high-quality data (CC-12M), use it to score/filter billions of pairs, train CLIP on filtered data. Standard ViT architecture.

**Result:** DFN ViT-H/14 achieves **84.4%** IN-1K zero-shot with same architecture but better-filtered 5B dataset (from 43B raw).

**Relevance:** Qwen2-VL initializes from DFN ViT. **Data quality > architecture innovation.**

#### NaViT (Google DeepMind)

**Patch n' Pack:** Processes images at native resolution/aspect ratio by packing patches from multiple images into single sequence. Masked self-attention prevents cross-image attention. Factorized positional embeddings (separate height/width, fractional).

**Efficiency:** 5x more images per training step, ViT's performance at 4x less compute. Padding < 2%.

**Influence:** Inspired Qwen2-VL's native dynamic resolution approach.

### 2C. Encoder Adoption in Major MLLMs

| Encoder Family | Models Using It | Key Property |
|---------------|----------------|-------------|
| **SigLIP-SO400M** | LLaVA-OV, VILA, MiniCPM-V, DeepSeek-VL2, mPLUG-Owl3, Ovis, VARCO, Phi-4, Idefics2/3 | **Dominant choice**; sigmoid loss |
| **InternViT-6B** | InternVL 2.0/2.5/3.0/3.5 | Largest; 10x data reduction |
| **CLIP ViT-L/14** | Molmo, Cambrian-1, LLaVA-1.5, Phi-3 | Most proven, fully open |
| **EVA2-CLIP-E** | CogVLM/CogVLM2 | Large-scale EVA pre-trained |
| **Custom from-scratch** | Pixtral (RoPE-2D), Qwen2-VL/2.5-VL (2D-RoPE) | Native variable resolution |
| **Multi-encoder** | Cambrian-1 (4 enc), Eagle/Eagle2 (SigLIP+ConvNeXt) | Complementary semantic+spatial |

---

## 3. Training Data Composition by Model

### 3A. InternVL 2.5 / 3.0

**InternVL 2.5 SFT Data (16.3M samples, ~120B tokens):**

| Modality | Token Proportion |
|----------|-----------------|
| Single-image | 45.92% |
| Video | 39.79% |
| Multi-image | 9.37% |
| Pure-text | 4.92% |

**Quality Filtering:** LLM scoring (0-10, threshold 7), repetition detection (threshold 3), heuristic rules, JPEG compression augmentation.

**Key Insight:** Only ~120B tokens (1/10 of competitors) while achieving SOTA — emphasizes data quality.

**InternVL 3.0 Native Multimodal Pre-training:**

| Data Type | Tokens | Ratio |
|-----------|--------|-------|
| Multimodal (image-text, video-text, interleaved) | ~150B | 75% |
| Pure language | ~50B | 25% |
| **Total** | **~200B** | - |

### 3B. Qwen2-VL / Qwen2.5-VL

| Model | Pre-training Tokens | Key Data |
|-------|-------------------|----------|
| Qwen2-VL | ~1.4T | Image-text, OCR, interleaved, VQA, pure text |
| Qwen2.5-VL | **~4.1T** (3.4x increase) | + Grounding, document omni-parsing (HTML), video |

### 3C. LLaVA-OneVision (Curriculum Learning)

| Stage | Name | Samples | Trainable |
|-------|------|---------|-----------|
| S1 | Language-Image Alignment | 558K (LCS-558K) | MLP projector |
| S1.5 | High-Quality Knowledge | 4M (captions+OCR) | ViT+MLP+LLM |
| S2 | Single-Image Instruction | 3.2M (Doc/Chart/OCR/Math/Language) | All |
| S3 | OneVision (Multi-modal) | 1.6M (50% SI + 35% MI + 22% Video) | All |

### 3D. VARCO-VISION 2.0 (Korean+English)

| Stage | Text Tokens | Image Tokens | Total |
|-------|------------|--------------|-------|
| S1: Alignment | 4M | 560M | 564M |
| S2: Basic SFT | 760M | 4.7B | 5.46B |
| S3: Advanced SFT | 5.7B | 25B | 30.7B |
| S4: DPO | 61M | 93M | 154M |
| **Total** | **6.5B** | **30.4B** | **36.9B** |

### 3E. Other Models Summary

| Model | Pre-training Volume | SFT Volume | Key Data Innovation |
|-------|-------------------|------------|---------------------|
| DeepSeek-VL2 | ~800B tokens (70% VL / 30% text) | Diverse QA | MoE with text preservation |
| VILA-1.5 | 51M images (COYO-25M + MMC4-25M + ShareGPT4V-1M) | 1.8-5.9M | Interleaved+caption blending |
| Cambrian-1 | Pre-trained backbones | 7M curated (from 10M filtered) | 7M beats 10M |
| Phi-4-Multimodal | 1.1T img-text + 5T text | Included | Heavy synthetic data |
| CogVLM2 | 1.5B pairs + 40M grounding | 300K+50K pref | Iterative refinement |
| Molmo/PixMo | 712K captions (speech→text) | ~5.5M QA | **No VLM synthetic data** |
| MM1.5 | 2B pairs + 600M interleaved + 2T text | Curated | Optimal: 45% caption + 45% interleaved + 10% text |

### 3F. Korean Multimodal Training Data Availability

| Resource | Type | Size |
|----------|------|------|
| VARCO-VISION training data | Training | 36.9B tokens (KO+EN) |
| HAN Dataset | Training | Korean Heritage VL descriptions |
| K-DTCBench | Benchmark | 240 questions (doc, table, chart) |
| KOFFVQA | Benchmark | 275 questions (10 subcategories) |
| K-MMBench / K-SEED / K-MMStar | Benchmark | Korean MCQA |

**Challenge:** Very limited Korean-specific training data. Most resources are benchmarks, not training data.

---

## 4. Feature Extraction Improvement Methods

### 4A. LLaVA-SP (ICCV 2025) — Visual Spatial Tokens

**Paper:** Lou et al., [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Lou_LLaVA-SP_Enhancing_Visual_Representation_with_Visual_Spatial_Tokens_for_MLLMs_ICCV_2025_paper.pdf) | [GitHub](https://github.com/CnFaker/LLaVA-SP)

**Core Idea:** Adds **6 spatial visual tokens** via convolutional kernels from ViT patch features.

**Architecture (3 components):**
1. **Spatial Feature Extractor (SFE):** Trainable convolution matrices extract 6 tokens. Two modes:
   - **Cropping:** Progressive central-to-outward crops (local→global, fine-grained)
   - **Pooling:** Adaptive average pooling at multiple scales (abstract→specific)
2. **Detail Feature Integrator (DFI):** Cross-attention fusing spatial tokens with fine-grained feature maps
3. **Parallel MLPs:** Standard projectors for original ViT tokens and spatial tokens

**Results:** Surpasses fully-trained LLaVA-1.5 on **10/11 benchmarks** with LoRA fine-tuning. Nearly identical inference latency (only 6 extra tokens). Convolution outperforms transformer blocks for spatial extraction.

**Key Insight:** Spatial structure in ViT features is severely underutilized by MLP projectors. A small number of spatially-aware tokens provides significant gains with minimal overhead.

### 4B. Multi-Scale / Pyramid Feature Methods

| Method | Key Innovation | Venue | Result |
|--------|---------------|-------|--------|
| **LLaVA-UHD v2** | Hiwin Transformer + Inverse Semantic Pyramid | arXiv 2024 | +3.7% avg, **+9.3% DocVQA** |
| **PIIP-LLaVA** | Inverted parameter pyramid (high-res→small model) | NeurIPS 2024 Spotlight | 1-2% gain at 40-60% compute |
| **Mini-Gemini** | Dual encoders: low-res embed + high-res refine | arXiv 2024 | No extra tokens |
| **XComposer2-4KHD** | Dynamic patches 336→4K HD | NeurIPS 2024 | Matches GPT-4V on 10/16 benchmarks |
| **Cambrian-1** | SVA with 4 encoders, 576 tokens (1/5 standard) | NeurIPS 2024 | Exceeds LLaVA-NeXT |
| **Eagle / Eagle 2** | SigLIP+ConvNeXt channel concatenation | ICLR 2025 Spotlight | Matches frontier VLMs |
| **BRAVE** | MEQ-Former combining 8 encoders | ECCV 2024 Oral | SOTA captioning/VQA |

### 4C. Vision Encoder Improvement Techniques

**Fine-tuning vs Freezing:**

| Strategy | Pros | Cons | When |
|----------|------|------|------|
| Fully Frozen | Preserves features, efficient | Limits alignment | Limited compute |
| Fully Unfrozen | Maximum adaptation | Expensive, forgetting | Sufficient data |
| LoRA on encoder | Parameter-efficient | Limited expressivity | Moderate budget |
| Selective unfreezing | Balanced | Requires layer selection | Most scenarios |

**AIMv2 (CVPR 2025 Highlight):** Autoregressive vision pre-training. Causal multimodal decoder: regresses image patches → decodes text. AIMv2-3B: **89.5% ImageNet-1k** (frozen trunk). **Outperforms CLIP, SigLIP, DINOv2.**

**"Are Bigger Encoders Always Better?" (arXiv 2408.00620):** **No.** Architecture > size. FastViTHD: 8x smaller, 20x faster than ViT-L/14. Encoder diversity > encoder size.

### 4D. Encoder-Free Approaches

| Method | Architecture | vs. Encoder-Based | Venue |
|--------|-------------|-------------------|-------|
| **Fuyu-8B** | Direct patch embedding to LLM | Average | 2023 |
| **EVE/EVEv2** | PEL+PAL with CLIP alignment | Rivals similar capacity | NeurIPS 2024 / ICCV 2025 |
| **VoRA** | Vision as mergeable LoRA in LLM | Near-zero inference overhead | 2025 |

---

## 5. Projector Architecture Comparison

| Projector | Representative Model | Token Change | Spatial Awareness | Params | Strength |
|-----------|---------------------|-------------|-------------------|--------|----------|
| **MLP (2-layer)** | LLaVA, InternVL | 1-to-1 | None | <2M | Simplest, surprisingly effective |
| **Q-Former** | BLIP-2 | N→K (K<<N) | Limited | ~188M | Strong compression |
| **Perceiver Resampler** | Flamingo, MiniCPM-V | N→K | Limited | ~100M+ | Extreme compression (64 tokens) |
| **C-Abstractor** | Honeybee, MM1/1.5 | N→K | Good (conv) | Moderate | Locality preservation |
| **SVA** | Cambrian-1 | Dynamic | Spatially-aware | Moderate | Multi-encoder fusion |
| **Hiwin Transformer** | LLaVA-UHD v2 | Hierarchical | Multi-scale | Moderate | Inverse semantic pyramid |
| **Visual Expert** | CogVLM/CogVLM2 | 1-to-1 | Deep (per-layer) | +5B | Deepest integration |
| **Visual Embedding Table** | Ovis | Probabilistic | Structural alignment | Moderate | Novel tokenization |
| **Hyper Attention Block** | mPLUG-Owl3 | Dual attention | Cross-modal | Moderate | 87.8% inference reduction |
| **TokenPacker** | TokenPacker | 75-89% reduction | Region-aware | Moderate | Drop-in MLP replacement |
| **LLaVA-SP SFE+DFI** | LLaVA-SP | +6 tokens only | Conv spatial | Small | Minimal overhead, high gain |

**MM1 Key Finding:** Image encoder + resolution + token count matter most; **connector design is comparatively negligible.**

---

## 6. Position Encoding Methods

| Method | Type | Resolution Generalization | 2D Aware | Used By |
|--------|------|--------------------------|----------|---------|
| Learnable Absolute | Parametric | Poor (needs interpolation) | No (1D) | CLIP, SigLIP, InternViT v1-v2.5 |
| **2D-RoPE** | Relative, rotary | **Excellent** (inherent) | Yes | **Qwen2-VL, EVA-02** |
| **M-RoPE** | Relative, rotary, multimodal | **Excellent** | Yes (+temporal) | **Qwen2-VL, Qwen2.5-VL** |
| Factorized Fractional | Parametric, factorized | Good | Yes | NaViT |
| **V2PE** | Flexible increments | Good | Configurable | **InternVL 3.0** |
| Implicit (CNN) | None (inherent locality) | Natural | Yes | ConvNeXt |

**Recommendation:** 2D-RoPE or M-RoPE — best trade-off of resolution generalization, 2D spatial awareness, and LLM compatibility (most LLMs already use RoPE).

---

## 7. Token Compression Methods

| Method | Compression | Training-Free? | Applied Where | Venue |
|--------|------------|----------------|---------------|-------|
| **PyramidDrop** | 55% FLOPs, 40% training time | Yes (inference) | Inside LLM layers | CVPR 2025 |
| **FastV** | ~50% tokens | Yes | After layer K in LLM | ECCV 2024 Oral |
| **LLaVA-PruMerge** | **14.4x** | No | Before LLM (projector) | ICCV 2025 |
| **MQT** | 2-256 tokens (elastic) | Yes (inference) | Query transformer | NeurIPS 2024 |
| **TokenPacker** | 75-89% | No | Visual projector | IJCV 2025 |
| **Pixel Unshuffle** | 4x (1024→256/tile) | N/A | After ViT | InternVL |
| **MLP 2x2 Merge** | 4x | N/A | After ViT | Qwen2-VL/2.5-VL |

---

## 8. Dynamic Resolution Strategies

| Strategy | Resolution | Tokens | Position Enc. | Pros | Cons | Used By |
|----------|-----------|--------|--------------|------|------|---------|
| **Fixed** | 224/336/384 | Fixed | Standard | Simple | Loses detail | CLIP, SigLIP default |
| **Dynamic Tiling** | 448 tiles (1-40) | 256/tile | Tile position | Handles high-res | Boundary artifacts | InternVL, DeepSeek-VL2 |
| **Native Dynamic** | Any resolution | Variable | 2D-RoPE | No artifacts, true preservation | Variable seq, complex batching | **Qwen2-VL/2.5-VL** |
| **AnyRes** | Grid of sub-images | Fixed/tile | Standard | Global+local | More tokens | LLaVA-OV/NeXT, VARCO |
| **Patch n' Pack** | Native | Packed | Factorized | <2% padding, efficient | Complex implementation | NaViT |
| **Scale-then-Compress** | 896 → compressed | Compressed | Standard | Balanced | Two-step | NVILA |
| **Perceiver Resampler** | Up to 1.8M pixels | 64-96 fixed | N/A | Extreme efficiency | Information loss | MiniCPM-V |

**2025 Trend:** Field moving toward **native resolution**. Qwen2.5-VL demonstrated native + window attention matches or beats tiling. LLaVA-UHD v3 and UniViTAR further validate this.

---

## 9. Trends & Future Directions (2024-2025)

### Major Trends

**1. Native Resolution is Winning.** Qwen2.5-VL proved native resolution + 2D-RoPE + window attention matches/beats tiling. Linear compute scaling via window attention.

**2. SigLIP Replacing CLIP.** Sigmoid loss (better small batches), higher resolution (384), multilingual data (109 languages). SigLIP2 adds captioning, self-distillation, grounding.

**3. Token Compression is Essential.** PyramidDrop, FastV, PruMerge, MQT, TokenPacker all show most visual tokens are redundant in deeper LLM layers. Progressive/elastic compression most promising.

**4. Multi-Encoder Ensembles are Powerful but Expensive.** Eagle, Cambrian-1, BRAVE: no single encoder captures everything. Simple channel concatenation is effective.

**5. Autoregressive Vision Pre-training Emerging.** AIMv2 (CVPR 2025 Highlight) outperforms CLIP/SigLIP/DINOv2. Potential paradigm shift.

**6. Encoder Size < Architecture Design.** FastVLM 85x speedup with 3.4x smaller encoder. Focus shifting from scaling to design optimization.

**7. Data Quality > Data Quantity.** InternVL 2.5 (120B tokens) competitive with Qwen2-VL (1.4T). Molmo SOTA with 600K images. Cambrian 7M beats 10M.

### Emerging Research Directions

1. Progressive Visual Compression within encoder (LLaVA-UHD v3)
2. Unified autoregressive vision-language pre-training (AIMv2)
3. Elastic inference with adaptive token budgets (MQT)
4. Frequency-domain processing for dense inputs (DocPedia)
5. Modality-specific LoRA adaptation (VoRA, MM-LoRA)
6. 4K+ resolution with efficient architectures
7. Hybrid CNN+Transformer encoders (FastViTHD)

---

## 10. Research Gaps & Proposed Direction

### Identified Gaps

1. **No native-resolution InternViT:** InternViT uses tiling (boundary artifacts) while Qwen2.5-VL proves native+2D-RoPE is superior
2. **InternViT's absolute position encoding is suboptimal:** 2D-RoPE (Qwen2.5-VL) and V2PE (InternVL3) show relative position is better, but full 2D-RoPE integration into InternViT unexplored
3. **Spatial information underutilized by MLP projector:** LLaVA-SP shows 6 spatial tokens give huge gains; InternVL's simple MLP leaves this on the table
4. **No multi-scale feature extraction in InternVL:** PIIP-LLaVA, LLaVA-UHD v2 show multi-scale features critical for document/chart tasks
5. **Window attention not applied to InternViT:** Qwen2.5-VL's window attention achieves linear compute scaling — would make InternViT-6B much more practical
6. **Token compression beyond pixel unshuffle:** InternVL's pixel unshuffle is static 4x; progressive/adaptive methods (PyramidDrop, MQT) could do better
7. **Korean-specific vision encoder optimization absent:** No work on vision encoder optimized for Korean text density (Hangul character complexity)
8. **SigLIP2's multi-objective training not tested with large encoders:** SigLIP2 adds captioning+self-distill+grounding to 400M encoder; untested at InternViT 6B scale

### Proposed Research Direction

**Improve InternViT vision encoder through architectural modernization** while maintaining InternVL's proven ViT-MLP-LLM framework:

1. **2D-RoPE Integration:** Replace learnable absolute PE with 2D-RoPE in InternViT for native resolution generalization
2. **Window Attention:** Add window attention layers (Qwen2.5-VL style) for linear compute scaling at high resolution
3. **Spatial Feature Enhancement:** Integrate LLaVA-SP style spatial tokens or multi-scale feature extraction
4. **Progressive Token Compression:** Replace static pixel unshuffle with adaptive compression (PyramidDrop/MQT hybrid)
5. **Korean Text-Aware Optimization:** Fine-tune with Korean OCR focus, leveraging Hangul character density awareness

---

## References

### InternVL Series
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [InternVL 2.5 (arXiv:2412.05271)](https://arxiv.org/abs/2412.05271)
- [InternVL 3.0 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [V2PE (arXiv:2412.09616)](https://arxiv.org/abs/2412.09616)

### Qwen-VL Series
- [Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)

### Vision Encoders
- [SigLIP (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343)
- [SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [EVA-02 (arXiv:2303.11331)](https://arxiv.org/abs/2303.11331)
- [DFN (arXiv:2309.17425)](https://arxiv.org/abs/2309.17425)
- [NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [AIMv2 (arXiv:2411.14402)](https://arxiv.org/abs/2411.14402)

### Feature Extraction & Projectors
- [LLaVA-SP (ICCV 2025)](https://github.com/CnFaker/LLaVA-SP)
- [LLaVA-UHD v2 (arXiv:2412.13871)](https://arxiv.org/abs/2412.13871)
- [PIIP-LLaVA (arXiv:2501.07783)](https://arxiv.org/abs/2501.07783)
- [Honeybee/C-Abstractor (CVPR 2024)](https://arxiv.org/abs/2312.06742)
- [PyramidDrop (arXiv:2410.17247)](https://arxiv.org/abs/2410.17247)
- [FastV (ECCV 2024)](https://github.com/pkunlp-icler/FastV)
- [LLaVA-PruMerge (arXiv:2403.15388)](https://arxiv.org/abs/2403.15388)
- [MQT (arXiv:2405.19315)](https://arxiv.org/abs/2405.19315)
- [TokenPacker (arXiv:2407.02392)](https://arxiv.org/abs/2407.02392)

### Multi-Encoder & Hybrid
- [Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Eagle/Eagle 2 (arXiv:2408.15998)](https://arxiv.org/abs/2408.15998)
- [BRAVE (arXiv:2404.07204)](https://arxiv.org/abs/2404.07204)
- [Mini-Gemini (arXiv:2403.18814)](https://arxiv.org/abs/2403.18814)

### Major MLLMs
- [LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [DeepSeek-VL2 (arXiv:2412.10302)](https://arxiv.org/abs/2412.10302)
- [Molmo/PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Pixtral (arXiv:2410.07073)](https://arxiv.org/abs/2410.07073)
- [CogVLM2 (arXiv:2408.16500)](https://arxiv.org/abs/2408.16500)
- [MiniCPM-V (arXiv:2408.01800)](https://arxiv.org/abs/2408.01800)
- [VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533) | [NVILA (arXiv:2412.04468)](https://arxiv.org/abs/2412.04468)
- [Phi-4-Multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- [mPLUG-Owl3 (arXiv:2408.04840)](https://arxiv.org/abs/2408.04840)
- [MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611) | [MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)

### Bilingual / Korean
- [VARCO-VISION-2.0 (arXiv:2509.10105)](https://arxiv.org/abs/2509.10105)
- [VARCO-VISION (arXiv:2411.19103)](https://arxiv.org/abs/2411.19103)
- [K-DTCBench](https://huggingface.co/datasets/NCSOFT/K-DTCBench)
- [KOFFVQA](https://github.com/maum-ai/KOFFVQA)

### Encoder-Free & Novel Approaches
- [EVE (arXiv:2406.11832)](https://arxiv.org/abs/2406.11832)
- [VoRA](https://github.com/Hon-Wong/VoRA)
- [Vary (arXiv:2312.06109)](https://arxiv.org/abs/2312.06109)
- [DocPedia (arXiv:2311.11810)](https://arxiv.org/abs/2311.11810)
- [FastVLM (Apple)](https://github.com/apple/ml-fastvlm)

---

**End of Comprehensive Survey**
