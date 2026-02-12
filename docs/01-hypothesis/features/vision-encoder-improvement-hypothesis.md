# Hypothesis: Vision Encoder Improvement for Korean Bilingual MLLM

> Research Phase: **1 - Hypothesis**
> Created: 2026-02-12
> Author: Juyeon
> Status: Draft
> Base Model: InternVL Framework
> Full Literature Review: See [vision-encoder-mllm-comprehensive-survey.md](../vision-encoder-mllm-comprehensive-survey.md)

---

## 1. Research Question (RQ)

### Primary RQ
Can architectural modernization of InternViT's vision encoder — incorporating 2D-RoPE, window attention, and spatial feature enhancement — produce superior visual feature extraction for a Korean+English bilingual MLLM, surpassing both vanilla InternVL and data-centric bilingual approaches (VARCO-VISION-2.0)?

### Sub-questions
1. **RQ1 (Position Encoding):** Does replacing InternViT's learnable absolute position embeddings with 2D-RoPE enable native dynamic resolution processing and improve performance on text-dense Korean documents/charts compared to the current tiling approach?
2. **RQ2 (Compute Efficiency):** Does adding window attention (Qwen2.5-VL style) to InternViT-6B achieve linear compute scaling while maintaining or improving feature quality, making the 6B encoder practical for high-resolution inputs?
3. **RQ3 (Spatial Features):** Does augmenting the vision encoder output with spatial feature tokens (LLaVA-SP style) or multi-scale pyramid features (PIIP/LLaVA-UHD v2 style) improve structured visual content understanding (charts, tables, documents) for bilingual inputs?
4. **RQ4 (Token Compression):** Does adaptive token compression (PyramidDrop or MQT-style) outperform static pixel unshuffle (4x) in preserving information-dense regions (Korean text, chart details) while reducing overall token count?
5. **RQ5 (Synergy):** Do these encoder improvements exhibit synergistic effects when combined, and do they complement the existing projector and decoder improvements proposed in the original TBI-MLLM hypothesis?

## 2. Hypothesis

### Main Hypothesis
**H1:** Architectural modernization of InternViT through (1) 2D-RoPE for native dynamic resolution, (2) window attention for efficient high-resolution processing, and (3) spatial feature enhancement (spatial tokens or multi-scale features) will produce a vision encoder that significantly outperforms vanilla InternViT on bilingual Korean+English visual understanding tasks, particularly for text-dense content (documents, charts, tables, OCR).

### Sub-Hypotheses

**H1a (2D-RoPE):** Replacing learnable absolute position embeddings with 2D-RoPE in InternViT will:
- Enable native resolution processing without tiling artifacts
- Improve generalization to unseen resolutions
- Specifically benefit Korean document understanding where Hangul's higher character density requires finer spatial resolution

**H1b (Window Attention):** Adding window attention layers (keeping only a few global attention layers) will:
- Reduce compute from quadratic to near-linear with input resolution
- Enable processing of 2K-4K resolution images within the same memory budget
- Maintain feature quality by preserving local attention patterns that matter most for text/chart recognition

**H1c (Spatial Enhancement):** Augmenting encoder output with spatial features will:
- Recover spatial structure lost in standard ViT feature extraction
- Improve structured content (chart axes, table cells, document layout) recognition by 3-5%
- Provide particularly large gains for Korean text where character boundary precision is critical

**H1d (Adaptive Compression):** Replacing static pixel unshuffle with adaptive token compression will:
- Preserve more tokens for information-dense regions (text, chart elements)
- Reduce tokens in background/simple regions
- Improve efficiency without accuracy loss

### Alternative Hypotheses

**H2:** 2D-RoPE alone is sufficient — window attention and spatial features provide negligible additional gains once position encoding is fixed (position encoding is the bottleneck).

**H3:** Window attention + 2D-RoPE is the complete solution — spatial feature enhancement is unnecessary because the improved attention pattern already captures spatial structure (Qwen2.5-VL proves this).

**H4:** Multi-scale features (PIIP style) are more effective than spatial tokens (LLaVA-SP style) for structured content, as they provide richer multi-resolution information rather than just 6 additional tokens.

**H5:** The vision encoder improvements are complementary to but independent of decoder improvements — combining vision encoder modernization with the TBI-MLLM decoder merging strategy yields additive rather than synergistic gains.

### Null Hypothesis
**H0:** Architectural modernization of InternViT does not provide statistically significant improvements over vanilla InternViT when trained with the same data and training recipe for bilingual visual understanding tasks.

## 3. Literature Review Summary

### 3A. Key Findings from Comprehensive Survey (50+ papers reviewed)

**Vision Encoder Landscape (2024-2025):**

| Finding | Evidence | Implication |
|---------|----------|-------------|
| SigLIP replacing CLIP as default | 10+ major MLLMs switched | Sigmoid loss + multilingual data superior |
| 2D-RoPE enables native resolution | Qwen2-VL, EVA-02, Pixtral | Superior to absolute PE + tiling |
| Window attention achieves linear scaling | Qwen2.5-VL | Makes large ViTs practical |
| InternViT-6B reduces data needs 10x | InternVL 2.5 (120B tokens vs 1.4T) | Large encoder compensates for less data |
| Spatial tokens give big gains cheaply | LLaVA-SP: 10/11 benchmarks, +6 tokens | MLP projector underutilizes spatial info |
| Multi-scale features help documents | LLaVA-UHD v2: +9.3% DocVQA | Critical for text-dense content |
| Token compression essential | PyramidDrop 55% FLOPs, MQT elastic | Most visual tokens redundant in deep layers |
| Data quality > data quantity | InternVL 2.5, Molmo, Cambrian | Filtering > scaling |
| Encoder size < architecture design | FastVLM 85x speedup, "Bigger Not Always Better" | Optimize architecture, not just scale |
| Autoregressive pre-training emerging | AIMv2 beats CLIP/SigLIP/DINOv2 | Potential paradigm shift |

**Competitive Model Analysis:**

| Model | Vision Encoder Strategy | Key Strength | Weakness for Korean |
|-------|----------------------|-------------|---------------------|
| InternVL 3.0 | InternViT-6B + V2PE + tiling | Largest encoder, SOTA benchmarks | Tiling artifacts, no 2D-RoPE |
| Qwen2.5-VL | 675M ViT + 2D-RoPE + window attn | Native resolution, efficient | Smaller encoder (675M vs 6B) |
| VARCO-VISION 2.0 | SigLIP2 + AnyRes | Korean bilingual, 4-stage training | Data-centric, standard encoder |
| LLaVA-OneVision | SigLIP-SO400M + AnyRes | Curriculum learning | No encoder innovation |

**Key Insight:** No model combines InternViT's large-scale encoder (6B, 10x data efficiency) with Qwen2.5-VL's architectural innovations (2D-RoPE, window attention). This is the gap we target.

### 3B. Key Papers for This Hypothesis

| # | Paper | Year | Venue | Key Contribution | Direct Relevance |
|---|-------|------|-------|-----------------|------------------|
| 1 | Qwen2.5-VL | 2025 | arXiv | 2D-RoPE + window attn in ViT, native resolution | Architecture template for InternViT modernization |
| 2 | LLaVA-SP | 2025 | ICCV | 6 spatial tokens via conv kernels, +10/11 benchmarks | Spatial feature extraction method |
| 3 | InternVL 3.0 | 2025 | arXiv | V2PE, native MM pre-training, InternViT-6B | Base model to improve |
| 4 | PIIP-LLaVA | 2025 | NeurIPS/TPAMI | Inverted parameter pyramid, 40-60% compute | Multi-scale approach for InternViT |
| 5 | LLaVA-UHD v2 | 2024 | arXiv | Hiwin Transformer, inverse semantic pyramid, +9.3% DocVQA | Multi-scale projector design |
| 6 | PyramidDrop | 2025 | CVPR | Progressive token dropping, 55% FLOPs reduction | Adaptive token compression |
| 7 | MQT | 2024 | NeurIPS | Elastic 2-256 token budget from single model | Adaptive inference |
| 8 | EVA-02 | 2024 | arXiv | 2D-RoPE + SwiGLU + MIM in ViT | Proves 2D-RoPE works in large ViTs |
| 9 | NaViT | 2023 | NeurIPS | Patch n' Pack, native resolution, 4x less compute | Native resolution inspiration |
| 10 | AIMv2 | 2025 | CVPR Highlight | Autoregressive vision pre-training, beats CLIP/SigLIP | Future encoder pre-training direction |
| 11 | Eagle 2 | 2025 | ICLR Spotlight | SigLIP+ConvNeXt channel concat, simple but effective | Multi-encoder fusion baseline |
| 12 | FastVLM | 2025 | CVPR | Hybrid conv+transformer, 85x faster TTFT | Efficient architecture design |
| 13 | TokenPacker | 2025 | IJCV | 75-89% visual token compression, drop-in MLP replacement | Token compression alternative |
| 14 | VARCO-VISION 2.0 | 2025 | arXiv | Korean bilingual VLM, 4-stage, SigLIP2+Qwen3 | Korean MLLM competitor |

### 3C. Research Gap

**No existing work applies Qwen2.5-VL's architectural innovations (2D-RoPE, window attention) to InternViT-6B.** This matters because:

1. InternViT-6B's **size advantage** (6B vs 675M) provides 10x data efficiency — critical when Korean VL data is scarce
2. Qwen2.5-VL's **architectural innovations** are proven to improve resolution handling and compute efficiency
3. **Combining both** could yield the best of both worlds: large-scale feature quality + efficient native resolution

Additionally:
- LLaVA-SP's spatial token enhancement has not been tested with InternViT or for Korean text
- Multi-scale features (PIIP) have been tested with InternViT but not combined with 2D-RoPE
- Adaptive token compression has not been applied to InternViT's pixel unshuffle stage
- No vision encoder has been specifically optimized for Korean character density

### 3D. Positioning

This work takes a **vision-encoder-centric approach** to bilingual MLLM improvement:

| Approach | Representative | Focus | Our Advantage |
|----------|---------------|-------|---------------|
| Data-centric | VARCO-VISION 2.0 | Training data quality/quantity | We improve the encoder itself |
| Projector-centric | Honeybee, LLaVA-UHD v2 | Better vision-language bridge | We improve what comes before the projector |
| Decoder-centric | TBI-MLLM (our prior work) | LLM adaptation for Korean | Complementary — encoder improvements feed better features to decoder |
| **Encoder-centric (ours)** | This work | Vision encoder architecture | Fundamental feature quality improvement |

## 4. Approach Overview

### Proposed Method: InternViT-Next (Working Name)

Modernize InternViT-6B with proven architectural innovations while preserving its scale advantage:

```
Input Image (any resolution)
    │
    ▼
┌──────────────────────────────────────────────┐
│  InternViT-Next (Modernized InternViT-6B)    │
│                                               │
│  Modification 1: 2D-RoPE                     │
│  - Replace learnable absolute PE              │
│  - Native dynamic resolution (no tiling)      │
│  - 2D spatial relationship encoding           │
│                                               │
│  Modification 2: Window Attention             │
│  - 41 window attention layers (8x8 window)    │
│  - 4 global attention layers (every ~11th)    │
│  - Linear compute scaling with resolution     │
│                                               │
│  Modification 3: Spatial Feature Extraction   │
│  Option A: LLaVA-SP style (+6 spatial tokens) │
│  Option B: PIIP multi-scale pyramid features  │
│  Option C: Both combined                      │
│                                               │
│  Modification 4: Adaptive Token Compression   │
│  - Replace static pixel unshuffle             │
│  - Content-aware: more tokens for text/detail  │
│  - Fewer tokens for background/simple regions  │
└──────────┬───────────────────────────────────┘
           │ Enhanced visual features
           ▼
┌──────────────────────────────────────────────┐
│  Projector (MLP or Enhanced)                  │
│  - Baseline: InternVL's 2-layer MLP           │
│  - Optional: Spatial-aware MLP (LLaVA-SP DFI) │
└──────────┬───────────────────────────────────┘
           │ Visual tokens
           ▼
┌──────────────────────────────────────────────┐
│  LLM Decoder                                  │
│  - InternLM3 or Qwen2.5                      │
│  - Korean adaptation (from TBI-MLLM work)    │
└──────────┬───────────────────────────────────┘
           │
           ▼
        Output
```

### Ablation Plan

| Variant | 2D-RoPE | Window Attn | Spatial Features | Adaptive Compress | Purpose |
|---------|---------|-------------|-----------------|-------------------|---------|
| **A0** (Baseline) | ✗ | ✗ | ✗ | ✗ | InternVL 2.5 reproduction |
| **A1** | ✓ | ✗ | ✗ | ✗ | 2D-RoPE only |
| **A2** | ✓ | ✓ | ✗ | ✗ | + Window attention |
| **A3a** | ✓ | ✓ | SP (6 tokens) | ✗ | + LLaVA-SP spatial tokens |
| **A3b** | ✓ | ✓ | PIIP (multi-scale) | ✗ | + PIIP pyramid features |
| **A3c** | ✓ | ✓ | SP + PIIP | ✗ | + Both combined |
| **A4** | ✓ | ✓ | Best of A3 | ✓ | + Adaptive compression |
| **A5** | ✓ | ✓ | Best of A3 | ✓ | Full InternViT-Next |

### Key Innovation

1. **First 2D-RoPE integration into InternViT-6B** — combining the largest open-source vision encoder's scale with native dynamic resolution
2. **Window attention for 6B-scale ViT** — enabling practical high-resolution processing with InternViT-6B, reducing compute from O(n²) to O(n)
3. **Spatial feature enhancement for bilingual OCR** — first application of spatial token injection (LLaVA-SP) to address Korean Hangul's higher character density
4. **Adaptive token compression for multilingual content** — content-aware compression that preserves Korean text regions while reducing tokens in simple areas

### Expected Contribution

1. **InternViT-Next:** Modernized vision encoder combining InternViT-6B's scale with Qwen2.5-VL's architectural innovations
2. **Bilingual feature extraction:** Vision encoder optimized for mixed Korean+English visual content
3. **Comprehensive ablation:** Systematic study of each architectural component's contribution
4. **Practical efficiency:** Window attention + adaptive compression make InternViT-6B practical for high-resolution bilingual documents
5. **Complementary to TBI-MLLM:** Vision encoder improvements that integrate with projector and decoder innovations

## 5. Feasibility Assessment

### Data Availability
- [x] Required datasets identified
  - Pre-existing InternVL training pipeline + datasets
  - Korean OCR/document data from VARCO-VISION benchmarks
  - Standard benchmarks: MMBench, MMMU, ChartQA, DocVQA, OCRBench, MathVista, K-DTCBench
- [x] Data access confirmed (public datasets + InternVL open-source data pipeline)
- [x] Key challenge: Limited Korean VL training data (mitigated by InternViT-6B's 10x data efficiency)

### Computational Resources
- [ ] GPU requirements estimated
  - 2D-RoPE replacement: Minimal additional compute (replace PE layer, fine-tune)
  - Window attention modification: Requires re-training attention layers
  - Full InternViT-Next training: ~4-8x A100-80GB with DeepSpeed ZeRO-3
  - Ablation study (8 variants): ~50-100 GPU-days on A100-80GB
- [ ] Training time estimated
  - Per variant: 3-7 days on 4x A100-80GB
  - Full ablation: ~4-8 weeks with sequential execution, ~2 weeks with parallel
- [ ] Storage requirements estimated
  - InternViT-6B checkpoints: ~12GB each × 8 variants = ~96GB
  - Training data: Shared with existing InternVL pipeline
  - Results/logs: ~5GB

### Timeline
| Phase | Duration | Milestone |
|-------|----------|-----------|
| Literature review completion | 1 week | This document + survey |
| 2D-RoPE integration & testing | 2 weeks | A1 variant working |
| Window attention integration | 2 weeks | A2 variant working |
| Spatial feature experiments | 2 weeks | A3a/A3b/A3c variants |
| Adaptive compression | 1 week | A4 variant |
| Full ablation + Korean evaluation | 3 weeks | All variants benchmarked |
| Analysis + paper writing | 3 weeks | Paper draft |
| **Total** | **~14 weeks** | |

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 2D-RoPE destabilizes InternViT-6B training | Medium | High | Start from EVA-02's proven 2D-RoPE implementation; progressive training |
| Window attention degrades global features | Low | Medium | Keep 4 global attention layers (Qwen2.5-VL proven); evaluate on global tasks |
| Spatial features don't help for Korean | Low | Low | LLaVA-SP already proven for general tasks; Korean benefit is incremental |
| InternViT-6B too expensive with modifications | Medium | Medium | Window attention reduces compute; can fall back to InternViT-300M for quick experiments |
| Korean training data insufficient | Medium | High | Leverage InternViT-6B's 10x data efficiency; synthetic data augmentation |
| Modifications don't compose well | Low | High | Ablation study will catch this early; can select best subset |

## 6. Success Criteria

### Quantitative

**Primary Metrics (vs InternVL 2.5 baseline):**
- **DocVQA:** >= +3% absolute (motivated by LLaVA-UHD v2's +9.3%)
- **ChartQA:** >= +2% absolute
- **OCRBench:** >= +30 points
- **K-DTCBench (Korean):** >= VARCO-VISION-2.0 performance
- **MMBench-EN:** >= InternVL 2.5 baseline (no degradation)
- **MMMU:** >= InternVL 2.5 baseline (no degradation)
- **MathVista:** >= InternVL 2.5 baseline (no degradation)

**Efficiency Metrics:**
- **Inference FLOPs:** <= 70% of vanilla InternViT-6B at same resolution (via window attention)
- **Max practical resolution:** >= 2x improvement (from ~5K to ~10K pixels)
- **Tokens per image:** <= 80% of current (via adaptive compression)

**Ablation Criteria:**
- Each modification contributes >= 0.5% average gain (statistically significant, p < 0.05)
- Combined modifications show >= 1% synergistic gain beyond sum of individual gains

### Qualitative
- Novel architectural contribution publishable at top venue (CVPR/NeurIPS/ICLR/ACL)
- Clear ablation story demonstrating each modification's role
- Practical efficiency gains that make InternViT-6B more deployable
- Generalizable approach applicable to any large-scale vision encoder

---

## Notes

<!-- Additional thoughts, discussions, open questions -->
- InternViT-6B의 2D-RoPE 전환 시 기존 pretrained weights를 최대한 활용하는 방법 연구 필요
  - EVA-02는 처음부터 2D-RoPE로 학습했지만, 우리는 기존 weights에서 전환해야 함
  - Qwen2.5-VL은 ViT를 처음부터 학습했으므로 직접 비교 불가
  - 가능한 전략: (1) position embedding layer만 재학습, (2) progressive unfreezing, (3) adapter-based transition
- Window attention 구현 시 FlashAttention2와의 호환성 확인 필요
- LLaVA-SP의 convolution kernel이 InternViT-6B의 3200 hidden dim에서도 효율적인지 검증 필요
- PIIP의 inverted parameter allocation이 이미 6B인 InternViT에서 어떻게 적용될지 설계 필요
- 한국어 OCR 데이터셋이 부족하므로, 합성 데이터 생성 파이프라인 (SSR 또는 LLM augmentation) 고려
- 기존 TBI-MLLM의 projector/decoder 개선과 어떻게 통합할지 integration plan 필요
- 모델 크기: InternViT-6B (full) vs InternViT-300M (quick experiments) → 300M으로 먼저 검증 후 6B에 적용
- Qwen2.5-VL의 window attention은 ViT 내부에서 적용, InternViT에도 동일하게 적용 가능한지 확인
- AIMv2의 autoregressive pre-training을 InternViT에 적용하는 것은 장기적 연구 방향으로 고려

**Full literature review and detailed model comparison:** See [vision-encoder-mllm-comprehensive-survey.md](../vision-encoder-mllm-comprehensive-survey.md)
