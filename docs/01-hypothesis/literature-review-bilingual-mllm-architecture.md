# Literature Review: Architectural Innovations for Bilingual (Korean+English) Multimodal LLMs

**Research Focus:** Architectural approaches (vision encoder, projector, decoder merging) for improving bilingual MLLM performance across diverse visual understanding tasks (chart, table, document, OCR, math, reasoning, general VQA), with InternVL as the base model.

**Date:** 2026-02-11
**Agent:** literature-reviewer (Sonnet 4.5)

---

## 1. InternVL Series: Architecture and Capabilities

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **InternVL2: Scaling Vision Foundation Model** | Chen et al. 2024 | CVPR 2024 (Oral) | ViT-MLP-LLM architecture with 6B InternViT vision encoder; dynamic resolution (1-12 tiles of 448×448); pixel unshuffle for token reduction (256 tokens per tile) | Baseline architecture for our work; demonstrates scalability of vision encoder |
| **InternVL2.5: Expanding Performance Boundaries** | OpenGVLab, Dec 2024 | arXiv | Progressive training strategy; enhanced data quality; parameter range 1B-78B | Shows importance of training data curation beyond pure architecture |
| **InternVL3: Advanced Training and Test-Time Recipes** | OpenGVLab, 2025 | arXiv | Variable Visual Position Encoding (V2PE); smaller position increments for visual tokens | Directly applicable for extending context length in our bilingual setup |
| **V2PE: Variable Visual Position Encoding** | Zhang et al. 2024 | arXiv 2412.09616 | Novel positional encoding with variable increments for visual vs. textual tokens; supports up to 1M token multimodal sequences when trained on 256K | Critical for handling high-resolution documents and long chart sequences in Korean+English |

**Key Insights:**
- InternVL uses a simple MLP projector, leaving room for improvement with advanced projector architectures
- Dynamic resolution support is essential but current approach (tiling) may not be optimal for structured content
- V2PE shows that treating visual and textual tokens differently in position encoding yields significant gains

---

## 2. Multilingual and Bilingual MLLMs

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **VARCO-VISION: Korean Vision-Language Models** | NC AI, Nov 2024 | arXiv 2411.19103 | First open-source Korean-English bilingual VLM; supports grounding, referring, OCR; released 5 Korean benchmarks | Direct competitor/reference for Korean MLLM; shows feasibility of bilingual approach |
| **VARCO-VISION-2.0** | NC AI, Sep 2025 | arXiv 2509.10105 | 14B and 1.7B variants; multi-image understanding; layout-aware OCR with spatial location prediction; 8th on OpenCompass VLM leaderboard | State-of-the-art Korean-English bilingual MLLM; our target to match/exceed |
| **Thunder-LLM: Korean Adaptation with Minimal Resources** | 2025 | arXiv 2506.21595 | Efficient Korean adaptation approach for LLMs | Shows resource-efficient language adaptation is possible |
| **Bilingual Adaptation of Monolingual Foundation Models** | 2024 | arXiv 2407.12869 | Two-stage approach: vocabulary expansion + embedding training, then full model continual pre-training on bilingual corpus | Directly applicable strategy for adding Korean to InternVL |

**Key Insights:**
- Vocabulary expansion is a necessary first step for Korean adaptation
- Multi-stage training (embeddings first, then full model) reduces catastrophic forgetting
- Korean benchmarks exist but may need augmentation for our diverse task set

---

## 3. Vision Encoder Improvements for MLLMs

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **Qwen2-VL: Perception at Any Resolution** | Qwen Team, Sep 2024 | arXiv 2409.12191 | Removes absolute position embeddings; introduces 2D-RoPE for spatial info; M-RoPE decomposes position into temporal/height/width; native dynamic resolution | Superior dynamic resolution approach compared to InternVL's tiling; 2D-RoPE could improve chart/table understanding |
| **Qwen2.5-VL Technical Report** | Qwen Team, Mar 2025 | arXiv 2502.13923 | Enhanced native resolution processing; 2×2 token compression via MLP | Shows continued evolution of resolution handling |
| **LLaVA-UHD v2: High-Resolution Semantic Pyramid** | Zhang et al. 2024 | arXiv 2412.13871 | Hierarchical Window Transformer (Hiwin); constructs inverse semantic pyramid by injecting low-level details into high-level features; cross-scale windows | Multi-scale feature fusion critical for documents/charts; could replace simple tiling |
| **Parameter-Inverted Image Pyramid Networks (PIIP)** | 2025 | arXiv 2501.07783 | PIIP-LLaVA uses efficient multi-scale processing with inverted parameter allocation | Efficient alternative to traditional pyramid approaches |
| **PyramidDrop: Visual Redundancy Reduction** | 2024 | arXiv 2410.17247 | Progressively drops image tokens across model stages; pyramid-like token reduction; efficiency gains with minimal performance loss | Could accelerate training and inference for our bilingual model |
| **FastVLM: Efficient Vision Encoding** | Apple, CVPR 2025 | Apple ML Research | Hybrid architecture visual encoder for high-resolution images; fast and efficient visual query processing | Production-ready efficiency improvements |
| **Res-Bench: Resolution Robustness** | 2024 | arXiv 2510.16926 | Evaluates resolution robustness across native and patch-based methods; identifies critical factors: processing mechanism, question type, training data distribution | Important for ensuring Korean and English text robustness at various resolutions |

**Key Insights:**
- 2D-RoPE and M-RoPE are superior to absolute position embeddings for vision
- Multi-scale/pyramid features significantly improve structured content understanding
- Token reduction strategies can maintain performance while improving efficiency

---

## 4. Projector Architecture Innovations

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **Honeybee: Locality-Enhanced Projector** | Cha et al. 2024 | CVPR 2024 (Highlight) | C-Abstractor uses convolution blocks (ResNet bottleneck + SE) + adaptive pooling; D-Abstractor uses deformable attention; preserves local spatial context vs. Perceiver Resampler | Strong alternative to InternVL's MLP; C-Abstractor outperforms Resampler (53.5 vs 43.9 @ M=144); critical for chart/table spatial structure |
| **Perceiver Resampler (in Flamingo)** | DeepMind 2022, still used 2024 | Multiple | Cross-attention with learned queries; fixed output tokens; can lose spatial information | Baseline for comparison; known weakness in spatial tasks |
| **MiniCPM-V Compression Layer** | 2024 | arXiv 2408.01800 | One-layer Perceiver Resampler for token compression | Shows Perceiver still competitive in simple compression tasks |
| **CROME: Cross-Modal Adapter** | 2024 | arXiv 2408.06610 | Gated cross-modal adapter; unifies vision and language representations before LLM; keeps LLM and vision encoder frozen | Parameter-efficient alternative; could enable efficient Korean adaptation |
| **Vision as LoRA (VoRA)** | Hon-Wong 2024 | GitHub | Internalizes visual capabilities via vision-specific LoRA layers in LLM; encoder-free approach | Radical rethinking of vision-language connection; potential efficiency gains |

**Key Insights:**
- Convolutional projectors (C-Abstractor) preserve spatial locality better than attention-based ones
- Perceiver Resampler's information loss is a known issue for spatial tasks
- LoRA-based approaches offer parameter efficiency crucial for bilingual adaptation

---

## 5. LLM Decoder Feature/Logit Merging

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **TIES-Merging** | 2024 | Multiple | Addresses task interference by allowing models with largest weight updates to take precedence; handles redundancy and parameter sign disagreement | Could merge English-centric InternVL with Korean-adapted LLM decoder |
| **DARE: Drop And REscale** | 2024 | Multiple | Randomly resets fine-tuned weights to base values; rescales weights to maintain output expectations; combines with TIES | Complementary to TIES; better preserves base model capabilities |
| **Differentiable DARE-TIES** | NeurIPS 2024 | OpenReview | Optimizes merging parameters via gradient descent instead of black-box optimization; more efficient for high-dimensional parameters | State-of-the-art merging; directly applicable to our Korean adaptation |
| **Evolutionary Model Merging** | 2024 | Nature Machine Intelligence | Automatic merging via evolutionary algorithms; creates hybrid models without extensive training | Alternative to manual merging hyperparameter tuning |
| **Mitigating Catastrophic Forgetting via Model Merging** | 2024 | ACL Findings EMNLP | Branch-and-Merge (BaM): iteratively merges models fine-tuned on data subsets; yields lower magnitude but higher quality weight changes | Specifically designed for language transfer; highly relevant |
| **Model Merging for Low-Resource Languages** | 2024 | ACL Findings EMNLP | Merges models with distinct capabilities without additional training; alternative to continual pre-training | Shows merging can bypass expensive continual training |
| **MoME: Mixture of Multimodal Experts** | 2024 | NeurIPS 2024 | Mixture of Vision Experts (MoVE) + Mixture of Language Experts (MoLE); addresses task interference in generalist MLLMs | Could use multiple Korean/English experts in decoder; elegant solution to bilingual task interference |

**Key Insights:**
- Model merging is a viable alternative to continual pre-training for language adaptation
- DARE+TIES combination is current best practice
- MoME-style mixture of experts could handle bilingual inference more elegantly than single decoder

---

## 6. Document/Chart/Table Understanding Specialists

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **Table-LLaVA** | 2024 | ACL 2024 | First large-scale multimodal instruction-tuning and pre-training dataset for table understanding; generalist tabular MLLM | Provides training methodology for table-specific capabilities |
| **Donut: OCR-free Document Understanding** | Kim et al. 2022, cited in 2024 | ECCV 2022 | Vision encoder + text decoder; learns to read via next-word prediction conditioned on image and previous text; no OCR pipeline | Efficient baseline for document understanding; 1.3s per document |
| **Pix2Struct: Screenshot Parsing** | Google 2024 | Multiple | Image-encoder-text-decoder based on ViT; variable resolution rendering; specialized for screenshots and charts | Higher accuracy than Donut but slower; good ablation baseline |
| **mPLUG-DocOwl 1.5: Unified Structure Learning** | 2024 | arXiv 2403.12895 | OCR-free with unified structure learning for various document types | Shows unified approach across document types is feasible |
| **LLaVA-Chart** | Zeng et al. 2024 | IEEE VIS 2024 | Visualization-referenced instruction tuning for chart QA; addresses ChartQA benchmark limitations | Directly applicable to our chart understanding requirements |
| **CharXiv: Charting Gaps in Chart Understanding** | 2024 | NeurIPS 2024 | New benchmark exposing failures of open-source models on reasoning-heavy chart questions | Reveals that current MLLMs struggle with complex chart reasoning |
| **ChartSketcher: Reasoning with Multimodal Feedback** | 2025 | arXiv 2505.19076 | Multimodal reflection mechanism for chart understanding; outperforms GPT-4o on chart-specific datasets | Agentic approach may be needed for complex chart reasoning |
| **mChartQA: Universal Benchmark** | 2024 | arXiv 2404.01548 | Universal benchmark for multimodal chart QA across diverse chart types | Evaluation benchmark for our chart capabilities |

**Key Insights:**
- OCR-free approaches (Donut, Pix2Struct) are mature and fast
- Chart reasoning remains challenging even for frontier models
- Specialized instruction tuning data is critical for chart/table performance
- Agentic/reflection approaches may be needed for complex reasoning

---

## 7. Multilingual Capability Without Catastrophic Forgetting

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **Self-Synthesized Rehearsal (SSR)** | 2024 | ACL 2024 | LLM generates synthetic instances via in-context learning; refines with latest model; selects diverse high-quality instances for rehearsal | Avoids need to store previous training data; applicable to Korean addition |
| **Continual Learning of LLMs: Comprehensive Survey** | Wang et al. 2025 | ACM Computing Surveys | Reviews methods for continual learning in LLMs (1B-7B); finds forgetting intensifies with scale in this range | Critical survey for understanding forgetting patterns |
| **Spurious Forgetting in Continual Learning** | 2024 | OpenReview | Identifies and addresses spurious forgetting phenomena in language model continual learning | Helps distinguish real vs. spurious forgetting |
| **Replay Adapter for Multilingual Continual Learning** | 2024 | Multiple | Decouples replay optimization from individual language adapters; maintains cross-lingual generalization with minimal parameters | Elegant architecture for bilingual continual learning |
| **Continual Learning for Low-Resource Languages** | 2025 | arXiv 2601.05874 | Strategies for adding low-resource languages to LLMs via continual learning | Directly applicable if treating Korean as lower-resource than English |

**Key Insights:**
- Catastrophic forgetting is more severe in 1B-7B parameter range (InternVL range)
- Adapter-based approaches minimize forgetting while maintaining efficiency
- Synthetic data generation (SSR) can replace expensive data retention

---

## 8. Mixture of Experts for Decoder Efficiency

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **DeepSeek-V3** | 2024-2025 | Multiple | 671B parameters, 37B activated per token; MoE-based frontier model | Shows MoE is production-ready at scale |
| **Grok-1** | xAI 2024 | Multiple | 314B parameters, 25% active per token; MoE-based LLM | Commercial deployment of MoE |
| **Mixture of Experts Explained** | HuggingFace 2024 | Blog | Router/gate network selects sparse expert subset per token; replaces FFN layers in transformer | Good tutorial for implementation |
| **MoE LLMs in Practice** | NVIDIA 2024-2025 | Technical Blog | MoE enables 10× faster token generation at 1/10 cost; runs optimally on Blackwell NVL72 | Production efficiency considerations |
| **Med-MoE: Domain-Specific Experts** | Jiang et al. 2024 | ACL Findings EMNLP | Lightweight medical VLM using domain-specific expert mixture | Shows MoE works in specialized domains; could apply domain experts per task type |

**Key Insights:**
- MoE is production-ready and used in frontier models (GPT-4, Grok, DeepSeek)
- Sparse activation maintains performance while reducing compute
- Could use separate experts for Korean vs. English, or for different visual task types

---

## 9. LLaVA-OneVision and Recent Strong Baselines

| Paper | Authors/Year | Venue | Key Method | Relevance |
|-------|--------------|-------|------------|-----------|
| **LLaVA-OneVision** | LMMS Lab, Aug 2024 | Blog | Single model for image/multi-image/video; SO400M vision encoder + Qwen2; 4-stage training (pretrain, mid, final-image, onevision); 0.5/7/72B variants | Strong baseline with proven multi-stage training; shows importance of data curriculum |
| **LLaVA-OneVision-Qwen2** | 2024 | HuggingFace | 32K context window; strong transfer across modalities | Context length important for documents/charts in Korean |

**Key Insights:**
- Multi-stage training with increasingly complex data is effective
- Single model can handle diverse tasks (single-image, multi-image, video)
- Qwen2 LLM backbone is strong and multilingual-friendly

---

## Research Gaps and Opportunities

### 1. **No Existing Work Combining All Three:**
- **Vision encoder improvement** + **Advanced projector** + **Decoder merging** for bilingual MLLMs
- Most work focuses on one dimension only

### 2. **Limited Bilingual MLLM Research:**
- Only VARCO-VISION addresses Korean-English bilingual MLLMs
- No systematic study of architectural choices for bilingual performance
- Gap: Which architectural component matters most for bilingual capability?

### 3. **Projector Architecture Underexplored for Structured Content:**
- Honeybee's C-Abstractor shows promise but hasn't been tested with:
  - Multilingual text-rich images (Korean+English OCR)
  - Document layout understanding
  - Multi-scale chart features
- Gap: Does C-Abstractor's spatial locality preservation help bilingual OCR?

### 4. **Model Merging for Multilingual MLLMs Unexplored:**
- DARE+TIES proven for LLMs, but no application to MLLM decoders
- MoME uses experts but not for language-specific routing
- Gap: Can we merge English InternVL decoder with Korean-adapted LLM and route by language?

### 5. **Multi-Scale Features Not Combined with Advanced Projectors:**
- LLaVA-UHD v2 uses hierarchical features but with standard attention
- Honeybee uses advanced projector but with single-scale features
- Gap: Pyramid features + C-Abstractor = best of both worlds?

### 6. **V2PE Not Tested for Bilingual Dense Text:**
- V2PE designed for long contexts, but not evaluated on:
  - Mixed Korean+English documents
  - Dense text with different character densities (Hangul vs. Latin)
- Gap: Does V2PE improve bilingual OCR by allowing finer position granularity for dense Korean text?

### 7. **Task Interference in Bilingual Settings:**
- MoME addresses task interference (vision tasks) but not language interference
- Gap: Do Korean and English interfere at the decoder level? Can language-specific experts help?

---

## Key Insight: Proposed Novel Contribution

### **Tripartite Architectural Innovation for Bilingual MLLMs**

**Hypothesis:** Combining three architectural innovations will yield superior bilingual (Korean+English) performance across diverse visual understanding tasks:

1. **Multi-Scale Vision Encoding with Dynamic Resolution**
   - Use Qwen2-VL's 2D-RoPE for native dynamic resolution
   - Add LLaVA-UHD v2's hierarchical pyramid features for multi-scale semantics
   - Hypothesis: Korean text (denser character information) benefits more from multi-scale features than English

2. **Locality-Preserving Projector with Pyramid Feature Injection**
   - Replace InternVL's MLP with Honeybee's C-Abstractor
   - Extend C-Abstractor to accept pyramid features from vision encoder
   - Hypothesis: Preserving spatial locality is critical for bilingual OCR where Korean and English text may be interspersed

3. **Language-Aware Decoder Merging with Sparse Experts**
   - Merge English-centric InternVL decoder with Korean-adapted LLM using DARE+TIES
   - Add lightweight language-routing MoLE (2-3 experts: Korean-specialist, English-specialist, shared)
   - Hypothesis: Sparse expert routing reduces interference between languages while maintaining shared visual reasoning

**Why This Could Work:**
- Each component addresses a specific bilingual challenge:
  - Multi-scale vision: handles different character densities (Korean Hangul vs. Latin)
  - C-Abstractor: preserves spatial layout for mixed-language documents
  - Language experts: reduces catastrophic forgetting and task interference
- No prior work combines all three
- Each component has independent evidence of effectiveness
- Synergistic effects likely (e.g., better vision features → better projector → better expert routing)

**Expected Improvements:**
- **Korean OCR/Document Understanding:** +5-10% over VARCO-VISION-2.0 (multi-scale + locality preservation)
- **English Performance Retention:** >95% of InternVL2 baseline (DARE+TIES merging + shared expert)
- **Chart/Table Understanding:** +3-5% over InternVL2 (pyramid features + C-Abstractor spatial preservation)
- **Training Efficiency:** 30-50% fewer tokens needed (expert routing reduces interference)

**Key Risk:** Complexity of combining three innovations may make debugging difficult. Mitigation: Ablation study with incremental addition (baseline → vision → projector → decoder).

---

## Recommended Next Steps

1. **Immediate:** Design experiment to ablate each component independently
2. **Priority 1:** Implement C-Abstractor + pyramid features (likely biggest gain for charts/tables)
3. **Priority 2:** Implement DARE+TIES merging for Korean decoder adaptation (reduces catastrophic forgetting)
4. **Priority 3:** Add language-aware MoLE (polish, may be overkill if P1+P2 already work well)
5. **Evaluation:** Use VARCO-VISION benchmarks + ChartQA + custom bilingual document benchmark

---

## Sources

### InternVL Series
- [InternVL GitHub Repository](https://github.com/OpenGVLab/InternVL)
- [InternVL2.5 Blog Post](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)
- [InternVL3 Blog Post](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)
- [InternVL2.5 Technical Report](https://arxiv.org/abs/2412.05271)
- [V2PE: Variable Visual Position Encoding](https://arxiv.org/abs/2412.09616)
- [V2PE Project Page](https://zzdhybthu.github.io/V2PE.github.io/)

### Multilingual MLLMs
- [Thunder-LLM: Korean LLM Adaptation](https://arxiv.org/html/2506.21595v1)
- [Best Open Source LLM for Korean](https://www.siliconflow.com/articles/en/best-open-source-llm-for-korean)
- [Korean Language Model Merging Research](https://www.sciencedirect.com/science/article/pii/S0952197625016884)
- [VARCO-VISION Korean Vision-Language Models](https://arxiv.org/abs/2411.19103)
- [VARCO-VISION-2.0 Technical Report](https://arxiv.org/abs/2509.10105)
- [Bilingual Adaptation of Foundation Models](https://arxiv.org/abs/2407.12869)

### Vision Encoder Improvements
- [Qwen2-VL: Any Resolution Perception](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [LLaVA-UHD v2: High-Resolution Semantic Pyramid](https://arxiv.org/html/2412.13871)
- [Parameter-Inverted Image Pyramid Networks](https://arxiv.org/html/2501.07783)
- [PyramidDrop: Visual Redundancy Reduction](https://arxiv.org/html/2410.17247v1)
- [FastVLM: Efficient Vision Encoding](https://machinelearning.apple.com/research/fast-vision-language-models)
- [Res-Bench: Resolution Robustness Benchmark](https://arxiv.org/html/2510.16926v1)

### Projector Architectures
- [Honeybee: Locality-Enhanced Projector (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Cha_Honeybee_Locality-enhanced_Projector_for_Multimodal_LLM_CVPR_2024_paper.pdf)
- [Honeybee GitHub Repository](https://github.com/khanrc/honeybee)
- [MiniCPM-V Architecture](https://arxiv.org/html/2408.01800v1)
- [CROME: Cross-Modal Adapter](https://arxiv.org/html/2408.06610v1)
- [VoRA: Vision as LoRA](https://github.com/Hon-Wong/VoRA)
- [Design Choices for Vision Language Models](https://huggingface.co/blog/gigant/vlm-design)

### Model Merging
- [Differentiable DARE-TIES (NeurIPS 2024)](https://openreview.net/forum?id=4jqff9QeUD)
- [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models)
- [Model Merging Methods Survey](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications)
- [NVIDIA: Model Merging for LLMs](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)
- [Evolutionary Model Merging (Nature Machine Intelligence)](https://www.nature.com/articles/s42256-024-00975-8)
- [Mitigating Catastrophic Forgetting via Model Merging](https://aclanthology.org/2024.findings-emnlp.1000/)
- [Model Merging for Low-Resource Languages](https://aclanthology.org/2024.findings-emnlp.508/)

### Document/Chart/Table Understanding
- [Table-LLaVA (ACL 2024)](https://github.com/SpursGoZmy/Table-LLaVA)
- [Donut: OCR-free Document Understanding](https://arxiv.org/abs/2111.15664)
- [mPLUG-DocOwl 1.5](https://arxiv.org/html/2403.12895)
- [LLaVA-Chart (IEEE VIS 2024)](https://github.com/zengxingchen/ChartQA-MLLM)
- [CharXiv: Chart Understanding Benchmark (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/cdf6f8e9fd9aeaf79b6024caec24f15b-Paper-Datasets_and_Benchmarks_Track.pdf)
- [ChartSketcher: Reasoning with Multimodal Feedback](https://arxiv.org/html/2505.19076)
- [mChartQA: Universal Benchmark](https://arxiv.org/pdf/2404.01548)

### Catastrophic Forgetting
- [Self-Synthesized Rehearsal (ACL 2024)](https://aclanthology.org/2024.acl-long.77/)
- [Continual Learning of LLMs: Survey (ACM Computing Surveys 2025)](https://dl.acm.org/doi/10.1145/3735633)
- [Continual Learning Survey GitHub](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
- [Spurious Forgetting in Continual Learning](https://openreview.net/forum?id=ScI7IlKGdI)
- [Continual Learning for Low-Resource Languages](https://arxiv.org/html/2601.05874)

### Mixture of Experts
- [Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
- [MoE LLMs Survey](https://cameronrwolfe.substack.com/p/moe-llms)
- [NVIDIA: MoE in LLM Architectures](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)
- [MoME: Mixture of Multimodal Experts (NeurIPS 2024)](https://arxiv.org/abs/2407.12709)
- [MoME GitHub Repository](https://github.com/JiuTian-VL/MoME)
- [Med-MoE: Domain-Specific Experts](https://aclanthology.org/2024.findings-emnlp.221/)

### LLaVA-OneVision and Baselines
- [LLaVA-OneVision Blog](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
- [LLaVA-OneVision-Qwen2-7B Model](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)
- [LLaVA-OneVision Documentation](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md)

### Efficient Tuning
- [LoRA Adapters for Vision-Language Models](https://medium.com/@hexiangnan/a-practical-guide-to-training-lora-adapters-for-vision-language-models-using-pytorch-0f64c74af7fa)
- [CROME: Cross-Modal Adapters](https://arxiv.org/html/2408.06610v1)
- [Vision as LoRA (VoRA)](https://github.com/Hon-Wong/VoRA)

---

**End of Literature Review**
