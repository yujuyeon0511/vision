# Hypothesis: Bilingual MLLM Architectural Innovation

> Research Phase: **1 - Hypothesis**
> Created: 2026-02-11
> Author: Juyeon
> Status: Draft

---

## 1. Research Question (RQ)

### Primary RQ
Can architectural innovations in the vision encoder, projector, and LLM decoder of an English-centric open-source MLLM (InternVLM) yield superior bilingual (Korean+English) performance across diverse visual understanding tasks — beyond what data-driven approaches alone can achieve?

### Sub-questions
1. **RQ1 (Vision Encoder):** Does multi-scale vision encoding with 2D-RoPE improve Korean text-rich image understanding (charts, tables, documents) compared to InternVL's default tiling approach, given the higher character density of Hangul?
2. **RQ2 (Projector):** Does a locality-preserving projector (e.g., C-Abstractor with pyramid feature injection) outperform InternVL's MLP projector for structured visual content (charts, tables) in both Korean and English?
3. **RQ3 (Decoder):** Among decoder adaptation strategies — (a) standard fine-tuning, (b) model merging (DARE+TIES), (c) fine-tuning + merging, and (d) fine-tuning + merging + MoLE routing — which yields the best trade-off between Korean acquisition and English retention?
4. **RQ4 (Synergy):** Do the three architectural improvements exhibit synergistic effects when combined, yielding gains greater than the sum of individual improvements?

## 2. Hypothesis

### Main Hypothesis
**H1:** A tripartite architectural innovation — (1) multi-scale vision encoding, (2) locality-preserving projector, and (3) optimized decoder adaptation (fine-tuning, merging, or their combination with sparse experts) — applied to InternVLM will produce a bilingual Korean+English MLLM that outperforms both the base InternVLM and existing bilingual models (VARCO-VISION-2.0) across chart, table, document, OCR, math, reasoning, and general VQA tasks.

### Alternative Hypotheses
**H2:** Only one or two of the three innovations are necessary; the third does not contribute significant gains beyond the first two (partial synergy).

**H3:** The projector innovation (C-Abstractor + pyramid features) alone is sufficient for most improvements, as it directly addresses the vision-language information bottleneck that disproportionately affects Korean text.

**H4:** Standard fine-tuning of the decoder on bilingual data (without merging or MoLE) is sufficient when paired with vision encoder and projector improvements, suggesting the decoder is not the bottleneck.

**H5:** Fine-tuning followed by merging (FT→Merge pipeline) outperforms both pure fine-tuning and pure merging by combining the strengths of gradient-based learning and weight-space composition.

### Null Hypothesis
**H0:** Architectural innovations in vision encoder, projector, and decoder do not provide statistically significant improvements over standard fine-tuning of InternVLM with Korean+English data for bilingual visual understanding tasks.

## 3. Literature Review

### Key Papers

| # | Paper | Year | Venue | Key Finding | Relevance |
|---|-------|------|-------|-------------|-----------|
| 1 | InternVL2.5 | 2024 | arXiv | Progressive training + data quality; ViT-MLP-LLM with dynamic resolution (1-12 tiles of 448x448) | Base model; MLP projector leaves room for improvement |
| 2 | InternVL3 (V2PE) | 2025 | arXiv | Variable Visual Position Encoding with different increments for visual vs. textual tokens; up to 1M token support | Directly applicable for bilingual dense text handling |
| 3 | VARCO-VISION-2.0 | 2025 | arXiv | Korean-English bilingual VLM; 14B variant; layout-aware OCR; 8th on OpenCompass | Primary competitor; data-driven approach to bilingual MLLM |
| 4 | Qwen2-VL (2D-RoPE) | 2024 | arXiv | Removes absolute position embeddings; 2D-RoPE for native dynamic resolution; M-RoPE decomposition | Superior resolution handling applicable to our vision encoder |
| 5 | LLaVA-UHD v2 | 2024 | arXiv | Hierarchical Window Transformer (Hiwin); inverse semantic pyramid; cross-scale windows | Multi-scale features critical for documents/charts |
| 6 | Honeybee (C-Abstractor) | 2024 | CVPR Highlight | Conv-based projector preserves spatial locality; outperforms Perceiver Resampler (53.5 vs 43.9 @M=144) | Strong alternative to InternVL's MLP projector |
| 7 | DARE+TIES | 2024 | NeurIPS | Drop and rescale fine-tuned weights + resolve parameter sign conflicts; state-of-the-art model merging | Applicable to merging English decoder with Korean-adapted LLM |
| 8 | MoME | 2024 | NeurIPS | Mixture of Vision Experts + Mixture of Language Experts for task interference in MLLMs | Language-specific expert routing for bilingual decoder |
| 9 | Branch-and-Merge (BaM) | 2024 | EMNLP Findings | Iteratively merges models fine-tuned on data subsets; reduces catastrophic forgetting | Language transfer via model merging |
| 10 | Model Merging for Low-Resource Lang. | 2024 | EMNLP Findings | Merges models with distinct capabilities without additional training | Korean adaptation without expensive continual pre-training |
| 11 | Table-LLaVA | 2024 | ACL | First large-scale multimodal instruction-tuning dataset for table understanding | Training methodology for table capabilities |
| 12 | CharXiv | 2024 | NeurIPS | Benchmark exposing failures on reasoning-heavy chart questions | Reveals current MLLM chart reasoning limitations |
| 13 | Self-Synthesized Rehearsal (SSR) | 2024 | ACL | LLM generates synthetic instances for continual learning without storing old data | Applicable to Korean addition without forgetting |
| 14 | PIIP-LLaVA | 2025 | arXiv | Efficient multi-scale processing with inverted parameter allocation | Efficient pyramid approach |
| 15 | PyramidDrop | 2024 | arXiv | Progressive token dropping across model stages; efficiency with minimal performance loss | Training/inference acceleration |

### Research Gap

**No existing work combines all three architectural dimensions** (vision encoder, projector, decoder) for bilingual MLLMs. Current bilingual approaches (VARCO-VISION) rely primarily on data-driven methods. Key gaps:

1. **C-Abstractor untested for multilingual OCR** — spatial locality preservation unexplored for mixed Korean+English text
2. **Model merging (DARE+TIES) unapplied to MLLM decoders** — proven for LLMs but not for multimodal decoders
3. **Multi-scale features not combined with advanced projectors** — LLaVA-UHD v2 uses standard attention; Honeybee uses single-scale
4. **Language interference at decoder level unaddressed** — MoME handles task interference but not language interference
5. **V2PE untested for bilingual dense text** — different character densities of Hangul vs. Latin not considered

### Positioning

This work extends InternVLM with a **tripartite architectural innovation** that addresses bilingual MLLM challenges at each component level. Unlike VARCO-VISION's data-centric approach, we propose a **method-centric approach** that improves the architecture itself, making the model inherently more capable of handling bilingual visual content. Each innovation is grounded in proven techniques (2D-RoPE, C-Abstractor, DARE+TIES, MoLE) but their combination and application to bilingual MLLMs is novel.

## 4. Approach Overview

### Proposed Method

**Architecture: Tripartite Innovation for Bilingual MLLM (TBI-MLLM)**

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  Multi-Scale Vision Encoder         │
│  InternViT + 2D-RoPE               │
│  → Pyramid features (L1, L2, L3)   │
│  → PyramidDrop for efficiency       │
└──────────┬──────────────────────────┘
           │ Multi-scale features
           ▼
┌─────────────────────────────────────┐
│  Pyramid C-Abstractor (Projector)   │
│  Conv blocks + SE attention         │
│  + Cross-scale feature fusion       │
│  → Locality-preserving tokens       │
└──────────┬──────────────────────────┘
           │ Visual tokens
           ▼
┌─────────────────────────────────────┐
│  Decoder Adaptation (4 variants)    │
│  3a: Fine-tuning only (LoRA/full)   │
│  3b: DARE+TIES merging only         │
│  3c: Fine-tune → Merge              │
│  3d: FT → Merge → MoLE routing     │
└──────────┬──────────────────────────┘
           │
           ▼
        Output
```

**Three stages of innovation:**

1. **Stage 1 — Multi-Scale Vision Encoding:**
   - Replace InternViT's absolute position embeddings with 2D-RoPE
   - Extract pyramid features at multiple scales (1/4, 1/2, 1x resolution)
   - Apply PyramidDrop to reduce redundant tokens while preserving information

2. **Stage 2 — Pyramid C-Abstractor:**
   - Replace InternVL's MLP projector with extended C-Abstractor
   - Input: multi-scale pyramid features from Stage 1
   - Conv bottleneck blocks + Squeeze-and-Excitation for cross-scale attention
   - Adaptive pooling to produce fixed-size locality-preserving visual tokens

3. **Stage 3 — Decoder Adaptation (4 variants to compare):**
   - **3a. Fine-tuning only:** LoRA/full fine-tune of InternVL decoder on Korean+English bilingual data
   - **3b. Merging only:** DARE+TIES merge of original English decoder with separately Korean-adapted decoder (no joint fine-tuning)
   - **3c. Fine-tune → Merge:** Fine-tune on Korean data first, then DARE+TIES merge with original to recover English performance
   - **3d. Fine-tune → Merge → MoLE:** Same as 3c, plus add lightweight MoLE layers (2-3 experts: KO-specialist, EN-specialist, shared) with language-aware routing

### Key Innovation

1. **Pyramid C-Abstractor** — first projector combining multi-scale vision features with locality-preserving convolutions for bilingual structured content understanding
2. **Systematic decoder adaptation study** — comprehensive comparison of fine-tuning, merging, FT→Merge pipeline, and FT→Merge→MoLE for bilingual language transfer in MLLMs
3. **Synergistic three-component architecture** — first unified framework addressing bilingual MLLM challenges at every architectural level

### Expected Contribution

1. A novel **method-centric approach** for bilingual MLLM development (vs. data-centric)
2. **Pyramid C-Abstractor** as a general-purpose multi-scale projector for MLLMs
3. **Decoder adaptation recipe** — systematic comparison revealing optimal strategy (fine-tuning vs. merging vs. hybrid) for adding new languages to MLLMs
4. Comprehensive ablation study demonstrating the contribution of each component
5. State-of-the-art bilingual (Korean+English) results across 7 visual understanding domains

## 5. Feasibility Assessment

### Data Availability
- [x] Required datasets identified
  - Korean VQA: K-DTCBench, VARCO-VISION benchmarks
  - English VQA: VQAv2, GQA, TextVQA, OCR-Bench
  - Charts: ChartQA, CharXiv, mChartQA
  - Tables: Table-LLaVA data, WTQ, TabFact
  - Documents: DocVQA, InfoVQA
  - Math: MathVista, MathVerse
  - General: MMBench, MMStar, SEED-Bench
- [x] Data access confirmed (public datasets)
- [x] Data size/format understood

### Computational Resources
- [ ] GPU requirements estimated
  - Vision encoder modification: ~4x A100 80GB (LoRA fine-tune of InternViT)
  - Projector training: ~2x A100 80GB (projector parameters only)
  - Decoder merging: ~4x A100 80GB (Korean fine-tune + DARE+TIES merge)
  - MoLE training: ~4x A100 80GB (expert FFN layers)
  - Full pipeline: ~8x A100 80GB
- [ ] Training time estimated
  - Each stage: 2-5 days on 4x A100
  - Total: ~2-4 weeks including ablations
- [ ] Storage requirements estimated
  - Model checkpoints: ~200GB (multiple variants)
  - Datasets: ~100GB
  - Results/logs: ~10GB

### Timeline
| Phase | Duration | Milestone |
|-------|----------|-----------|
| Literature review | 1 week | Completed (this document) |
| Stage 1: Vision encoder | 2 weeks | Multi-scale InternViT with 2D-RoPE |
| Stage 2: Projector | 2 weeks | Pyramid C-Abstractor trained |
| Stage 3: Decoder merging | 2 weeks | Language-aware merged decoder |
| Integration + ablation | 2 weeks | Full TBI-MLLM + ablation tables |
| Analysis | 1 week | Statistical analysis + visualizations |
| Writing | 2 weeks | Paper draft |
| **Total** | **~12 weeks** | |

### Risks and Mitigations
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Multi-scale vision encoder destabilizes training | Medium | High | Start with frozen InternViT, only train 2D-RoPE adapter |
| C-Abstractor doesn't scale to pyramid features | Low | Medium | Fall back to per-scale C-Abstractor + concatenation |
| DARE+TIES merging degrades both languages | Medium | High | Tune merging coefficients; use BaM iterative approach |
| MoLE router doesn't learn meaningful routing | Medium | Medium | Pre-assign experts with language labels for warm start |
| Insufficient Korean training data for domain tasks | Medium | High | Use SSR (self-synthesized rehearsal) + translation augmentation |
| GPU budget insufficient for full experiments | Low | High | Prioritize most impactful component (projector first) |

## 6. Success Criteria

### Quantitative
- **Korean VQA (K-DTCBench):** >= VARCO-VISION-2.0 performance
- **English VQA (VQAv2, TextVQA):** >= 95% of InternVL2.5 baseline (minimal forgetting)
- **ChartQA:** >= +3% absolute over InternVL2.5 baseline
- **DocVQA:** >= +3% absolute over InternVL2.5 baseline
- **TableVQA (WTQ):** >= +2% absolute over InternVL2.5 baseline
- **OCR-Bench:** top-3 among open-source models of similar parameter count
- **MathVista:** >= InternVL2.5 baseline (no degradation)
- **MMBench (EN+KO):** >= +2% average over InternVL2.5 baseline
- **Ablation:** Each component contributes >= 1% average gain (statistically significant, p < 0.05)

### Qualitative
- Novel architectural contributions publishable at top venue (ACL/EMNLP/NeurIPS/CVPR)
- Clear ablation story showing each component's role
- Generalizable recipe for adding new languages to English-centric MLLMs
- Efficient enough for practical deployment (inference latency comparable to InternVL2.5)

---

## Notes
<!-- Additional thoughts, discussions, open questions -->
- InternVLM 모델 사이즈 결정 필요: 2B vs 8B vs 26B → 8B가 연구 효율성과 성능의 균형점
- Korean tokenizer 확장 여부 검토: InternLM2의 기본 vocab에 한국어 토큰이 부족할 수 있음
- VARCO-VISION-2.0이 data-centric approach의 upper bound를 보여줌 → 우리는 method-centric으로 차별화
- 3개 component를 모두 합쳤을 때의 training recipe 설계가 가장 어려운 부분이 될 것
- 각 component를 incremental하게 추가하는 ablation study가 논문의 핵심 contribution

**Full literature review:** See [literature-review-bilingual-mllm-architecture.md](../literature-review-bilingual-mllm-architecture.md)
