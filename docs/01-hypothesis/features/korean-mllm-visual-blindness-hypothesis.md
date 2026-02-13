# Hypothesis: Visual Blindness in Korean-Finetuned Multimodal LLMs

> Research Phase: **1 - Hypothesis**
> Created: 2026-02-13
> Author: Juyeon
> Status: Draft

---

## 1. Research Question (RQ)

### Primary RQ
**Why do open-source vision encoder + LLM decoder multimodal models fail to properly utilize visual information when fine-tuned on Korean datasets, and how can this "visual blindness" be resolved?**

### Sub-questions
1. **RQ1**: Does Korean fine-tuning cause catastrophic forgetting of pre-trained visual understanding capabilities in the LLM decoder?
2. **RQ2**: Does the attention mechanism in the LLM decoder systematically de-prioritize visual tokens in favor of Korean text tokens after language-specific fine-tuning?
3. **RQ3**: Does the projector/connector between vision encoder and LLM degrade or become misaligned during Korean fine-tuning, leading to visual representation collapse?
4. **RQ4**: To what extent does the scarcity and quality of Korean multimodal training data contribute to visual grounding failure?
5. **RQ5**: What is the optimal training strategy (staged training, selective freezing, regularization) to preserve visual understanding while adapting to Korean?

## 2. Hypothesis

### Main Hypothesis
**H1: Korean fine-tuning of MLLMs causes "visual attention collapse" — a systematic shift in the LLM decoder's attention distribution away from visual tokens toward Korean text tokens, because the adapted attention key-space becomes text-centric for Korean while visual token keys remain out-of-distribution.**

This hypothesis is grounded in the finding that LLM backbones pre-trained on text develop attention key-spaces that primarily reflect textual statistics. When the LLM is further adapted to Korean, this text-centric bias intensifies for Korean tokens specifically, causing visual tokens to be treated as noise rather than information.

### Alternative Hypotheses

**H2 (Catastrophic Forgetting)**: Korean fine-tuning erases the LLM's learned ability to process visual features, even though visual tokens are still being injected. The model has forgotten *how to read* visual tokens rather than *choosing to ignore* them.

**H3 (Projector Misalignment)**: The MLP projector, originally trained to map vision encoder features into the LLM's embedding space, becomes misaligned as the LLM's embedding space shifts during Korean fine-tuning. The projector still maps correctly to the *original* embedding space, but the LLM no longer interprets that space the same way.

**H4 (Data Distribution Mismatch)**: Korean multimodal datasets are too small, lack diversity, or have poor visual grounding quality (e.g., generic captions that don't require looking at images), causing the model to learn text-only shortcuts.

**H5 (Compound Effect)**: The visual blindness is not caused by any single factor but results from the multiplicative interaction of H1-H4, where each factor alone causes minor degradation but their combination leads to complete visual grounding failure.

### Null Hypothesis
**H0**: Korean fine-tuning does not inherently cause visual blindness. The observed performance degradation is solely due to insufficient data quantity/quality or suboptimal hyperparameters, and can be resolved by simply scaling up Korean multimodal training data without any architectural or training strategy changes.

## 3. Literature Review

### Key Papers

| # | Paper | Year | Venue | Key Finding | Relevance |
|---|-------|------|-------|-------------|-----------|
| 1 | Unveiling Intrinsic Text Bias in MLLMs through Attention Key-Space Analysis | 2024 | ArXiv | LLM attention key-space is text-centric; visual keys are out-of-distribution, causing systematic text preference | **Core mechanism** — explains why Korean tokens dominate visual tokens |
| 2 | See What You Are Told: Visual Attention Sink in Large Multimodal Models | 2025 | ICLR | MLLMs allocate high attention to specific visual tokens regardless of relevance ("attention sink"); deep layers ignore visual semantics | Explains *how* visual tokens become ignored in decoder |
| 3 | VIRAL: Visual Representation Alignment for MLLMs | 2024 | ArXiv | Visual representations diverge from vision encoder outputs during text-supervised training ("visual representation misalignment") | Mechanism for projector degradation during fine-tuning |
| 4 | Hidden in Plain Sight: VLMs Overlook Their Visual Representations | 2024 | ArXiv | VLMs fail to use accessible visual information; inherit strong language priors from LLM | Confirms visual information is present but unused |
| 5 | Words Over Pixels? Rethinking Vision in MLLMs | 2025 | IJCAI | LLM exhibits strong language prior that overrides visual evidence; vision features "washed out" in deep layers | Language dominance over vision in generation |
| 6 | VARCO-VISION: Korean Vision-Language Models | 2024 | ArXiv | 4-stage training (alignment → basic SFT → advanced SFT → DPO); full model unfreezing in final stage; Korean OCR/grounding success | **Best practice** for Korean MLLM training |
| 7 | No Language Data Left Behind: CJK Language Datasets | 2024 | ArXiv | Korean has **0.0%** representation in vision-language datasets; massive domain shift for Korean multimodal tasks | Quantifies Korean data scarcity |
| 8 | TWIST & SCOUT: Forget-Free Tuning for MLLMs | 2025 | ICCV | Fine-tuning erases pre-trained visual understanding; TWIST framework preserves visual grounding | Solution: forget-free tuning |
| 9 | SMoLoRA: Dual Catastrophic Forgetting in Visual Instruction Tuning | 2025 | ICCV | Both visual and language capabilities can be forgotten simultaneously during continual tuning | Dual forgetting risk |
| 10 | Mitigating Catastrophic Forgetting in Target Language Adaptation | 2024 | ArXiv | Performance collapse approaching 0 BLEU in language adaptation; more severe with large datasets | Severity quantification |
| 11 | Attention Debiasing for Token Pruning in VLMs | 2024 | ArXiv | Attention systematically biased toward later tokens; inflated scores to padding tokens; recency bias from LLM | Structural attention issues |
| 12 | Token Pruning in MLLMs | 2025 | ACL Findings | Visual information migrates to text tokens within first few layers; vision tokens unnecessary in deep layers | Visual info transfer dynamics |
| 13 | Bridging Writing Manner Gap in Visual Instruction Tuning | 2025 | ArXiv | Writing style mismatch between instructions and base LLM causes capability degradation | Korean writing style shift compounds problem |
| 14 | DC-CLIP: Multilingual CLIP Compression | 2025 | Pattern Recognition | CLIP performance degrades for non-English; "curse of multilinguality" dilutes representations | Vision encoder's English bias |
| 15 | Traveling Across Languages: Cross-Lingual Consistency in MLLMs | 2025 | ArXiv | Large gap between English performance and zero-shot cross-lingual transfer in VLMs | Cross-lingual transfer failure |
| 16 | Learning without Forgetting for VLMs (PROOF) | 2025 | IEEE TPAMI | Task-specific projections on frozen encoders; new projections expanded without destroying old ones | Solution: expandable projections |
| 17 | Debiasing MLLMs via Penalization of Language Priors | 2024 | ArXiv | Visual Debias Decoding contrasts correct vs. meaningless image to force visual attention | Solution: training-free debiasing |
| 18 | Multi-modal Preference Alignment Remedies VIT Degradation | 2024 | ArXiv | VQA datasets lack diversity of original text instruction datasets; Korean VQA even less diverse | Data quality concern |
| 19 | Honeybee: Locality-enhanced Projector for MLLM | 2024 | CVPR | Standard MLP projectors may not preserve locality and fine-grained visual information | Projector limitation |
| 20 | Continual Pre-training Mitigates Forgetting in Language and Vision | 2024 | Neural Networks | Self-supervised continual pre-training sufficient to mitigate forgetting without CL strategies | Solution: continual pre-training |

### Research Gap

Despite extensive research on (1) visual attention mechanisms in MLLMs, (2) catastrophic forgetting in LLMs, and (3) multilingual NLP, **there is almost no work specifically investigating the intersection of all three** — how non-English fine-tuning causes visual blindness in MLLMs. Key gaps include:

1. **No systematic analysis** of visual token attention patterns before vs. after Korean/CJK fine-tuning
2. **No quantitative measurement** of projector alignment degradation as a function of language-specific training
3. **No comparison of mitigation strategies** (VIRAL, attention redistribution, staged training, LoRA) specifically for the Korean MLLM setting
4. **No established Korean multimodal benchmark** that separately measures visual grounding vs. language generation quality

### Positioning

This work bridges three research streams:
- **MLLM visual attention analysis** (Papers 1, 2, 4, 5, 11, 12): We extend these analyses to the multilingual fine-tuning setting
- **Catastrophic forgetting in MLLMs** (Papers 3, 8, 9, 10): We investigate forgetting specifically induced by language adaptation
- **Korean/CJK multimodal AI** (Papers 6, 7, 14, 15): We provide root cause analysis and systematic solutions

## 4. Approach Overview

### Proposed Method
A systematic diagnostic and mitigation framework for visual blindness in Korean-finetuned MLLMs:

**Phase 1: Diagnosis** — Quantify the visual blindness phenomenon
- Measure attention distribution over visual vs. text tokens at each layer, before and after Korean fine-tuning
- Analyze projector output similarity to original vision encoder representations (cosine similarity, CKA)
- Probe intermediate representations for visual information retention (linear probing)
- Compare performance on image-dependent vs. image-independent Korean VQA questions

**Phase 2: Root Cause Isolation** — Ablation study to isolate contributing factors
- Controlled experiments: freeze/unfreeze vision encoder, projector, LLM subsets
- Vary Korean data quality and quantity
- Measure each factor's individual and combined contribution to visual blindness

**Phase 3: Mitigation** — Develop and evaluate solutions
- **Visual Alignment Regularization**: VIRAL-style loss to maintain vision encoder alignment during Korean fine-tuning
- **Attention Redistribution**: Redistribute attention away from text-biased sinks toward visual tokens
- **Staged Korean Adaptation**: Progressive unfreezing with alignment preservation (inspired by VARCO-VISION)
- **Projector Re-alignment**: Post-fine-tuning projector re-calibration with small high-quality visual data

### Key Innovation
1. **First systematic study** of visual attention collapse specifically induced by non-English language adaptation in MLLMs
2. **Diagnostic framework** that can quantify the degree and source of visual blindness
3. **Unified mitigation strategy** combining attention, alignment, and training strategies tailored for Korean MLLM adaptation

### Expected Contribution
1. **Empirical understanding**: Detailed analysis of why and how Korean fine-tuning causes visual blindness
2. **Diagnostic toolkit**: Reusable tools for measuring visual attention health in multilingual MLLMs
3. **Training recipe**: Practical guidelines for training Korean MLLMs that maintain strong visual grounding
4. **Benchmark**: Korean visual grounding evaluation protocol that distinguishes visual understanding from language generation

## 5. Feasibility Assessment

### Data Availability
- [x] Required datasets identified
  - Korean VQA: AI Hub Korean VQA, K-VQA
  - Korean image captioning: AI Hub, COCO-Ko (translated)
  - Korean OCR: AI Hub document/scene OCR datasets
  - English baselines: LLaVA-Instruct, ShareGPT4V, COCO
- [x] Data access confirmed (AI Hub, public datasets)
- [x] Data size/format understood

### Computational Resources
- [x] GPU requirements estimated: 2x A100-80GB (available)
- [x] Training time estimated:
  - Phase 1 (Diagnosis): ~5-7 GPU-days (attention analysis, probing)
  - Phase 2 (Ablation): ~20-30 GPU-days (controlled experiments)
  - Phase 3 (Mitigation): ~15-25 GPU-days (solution training and evaluation)
- [x] Storage requirements estimated: ~500GB (models + data + checkpoints)

### Timeline
| Phase | Duration | Milestone |
|-------|----------|-----------|
| Literature review | 1 week | Completed (this document) |
| Phase 1: Diagnosis | 2 weeks | Attention analysis report, visual blindness quantification |
| Phase 2: Root Cause Isolation | 2-3 weeks | Ablation results, factor importance ranking |
| Phase 3: Mitigation | 3-4 weeks | Solution comparison, best recipe identified |
| Analysis & Writing | 2 weeks | Paper draft with full results |

### Risks and Mitigations
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Visual blindness varies by model architecture | High | Medium | Test on 3+ architectures (InternVL, LLaVA, Qwen-VL) |
| Korean VQA data too low quality for diagnosis | Medium | High | Curate small high-quality Korean visual grounding set |
| Attention analysis doesn't reveal clear patterns | Low | High | Use additional probing methods (CKA, linear probes, representational similarity) |
| Mitigation works on small scale but not large | Medium | Medium | Validate on both 2B and 7B+ parameter models |
| Training instability when unfreezing vision encoder | High | Medium | Use LoRA for vision encoder, careful learning rate scheduling |

## 6. Success Criteria

### Quantitative
- **Diagnosis**: Demonstrate ≥20% relative decrease in visual token attention weight after Korean fine-tuning vs. baseline
- **Root Cause**: Identify factors explaining ≥80% of the visual blindness variance through ablation
- **Mitigation**: Recover ≥90% of original visual grounding performance while maintaining Korean language performance
  - Korean VQA accuracy: ≥ baseline (English-trained) - 2%
  - Korean image captioning CIDEr: ≥ baseline - 5%
  - Visual grounding accuracy: ≥ 85% of English-only model performance

### Qualitative
- Clear, interpretable visualization of attention patterns showing visual blindness
- Practical training recipe that other researchers can follow for Korean MLLM development
- Insights transferable to other non-English languages (Japanese, Chinese, etc.)

---

## Notes

### 핵심 관찰 (Core Observations)
- 한국어 파인튜닝 후 모델이 이미지를 "보지 못하는" 현상은 단일 원인이 아닌 복합적 요인의 결과일 가능성이 높음
- 가장 유력한 원인: LLM의 attention key-space가 텍스트 중심으로 형성되어 있어, 한국어 적응 후 시각 토큰이 더욱 out-of-distribution이 됨
- VARCO-VISION (NCSOFT)이 4단계 학습으로 성공한 사례가 있으므로, 학습 전략이 핵심
- 한국어 vision-language 데이터가 전체의 0.0% (CJK 데이터셋 조사 기준) → 데이터 부족이 근본적 제약

### 기존 연구와의 차별점
- 기존 가설 (bilingual-mllm-arch, vision-encoder-improvement)은 아키텍처 개선에 초점
- 이 가설은 **"왜 한국어로 학습하면 이미지를 못 보는가"** 라는 근본 원인 분석에 초점
- 진단 → 원인 분석 → 해결의 체계적 접근
