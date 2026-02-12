# Focused PhD Research Directions: Vision Encoder for Multilingual MLLMs

> Research Phase: **1 - Hypothesis Refinement**
> Created: 2026-02-12
> Purpose: Narrow from 4-modification engineering plan to 1 deep, novel research contribution
> Target Venues: CVPR / NeurIPS / ICLR / ACL
> Constraints: 2x A100 80GB, ~3-4 months

---

## Executive Summary

After extensive literature review of 2024-2025 publications, I analyze six potential research directions and recommend **Direction 2 (Content-Adaptive Visual Token Compression)** as the strongest PhD-level research topic, with **Direction 1 (2D-RoPE for Text-Dense Multilingual Encoding)** as a strong secondary option. The analysis below provides detailed justification.

**Key insight from the literature review:** The field has moved extremely fast in 2025. Several directions that seemed novel 6 months ago now have significant prior work. The winning strategy is not "combine known techniques" (engineering paper) but "deeply investigate one mechanism with novel theoretical insight and strong empirical analysis" (research paper).

---

## Direction 1: Semantic-Aware Position Encoding for Text-Dense Multilingual Vision

### Title
**"Beyond Spatial Coordinates: Content-Adaptive 2D Position Encoding for Text-Dense Vision Understanding in Multilingual MLLMs"**

### Core Research Question
Can a position encoding scheme that adapts to local content characteristics (text density, script type, visual complexity) — rather than encoding only geometric coordinates — significantly improve vision encoder performance on text-dense multilingual documents, and what is the theoretical basis for this improvement?

### Why This Is PhD-Level
- **Novelty:** Current 2D-RoPE (Qwen2-VL, Qwen2.5-VL) and XD-RoPE (HunyuanOCR) encode pure spatial coordinates. SaPE2 (May 2025) showed content-adaptive PE helps, but only tested on CIFAR10/100 — never at scale, never for text-dense images, never for multilingual content. This direction would be the first to design a position encoding that is *semantically aware of text regions and script types*.
- **Theoretical depth:** Requires formal analysis of why standard 2D-RoPE underperforms for text-dense images (attention distribution analysis, frequency domain analysis of position embeddings for different script densities).
- **Not just engineering:** Goes beyond "replace abs PE with RoPE" to ask *what information should position encodings carry* for document understanding.

### Key Technical Approach
1. **Analysis phase:** Attention map visualization comparing 2D-RoPE vs absolute PE on text-dense Korean/English mixed documents. Quantify the "position encoding bottleneck" — where does spatial PE fail for text vs natural images?
2. **Method:** Design a content-adaptive position encoding (CaPE) that modulates position frequencies based on local patch complexity:
   - Text-dense patches: higher frequency (finer spatial discrimination)
   - Natural scene patches: standard frequency
   - Use a lightweight content classifier (frozen, from ViT early layers) to predict per-patch complexity scores
   - Modulate 2D-RoPE frequencies based on these scores
3. **Script-aware extension:** Different scripts have different spatial information needs (Korean Hangul: dense 2D block characters; Latin: 1D horizontal flow; Chinese: dense but square). Design script-aware frequency modulation.
4. **Integration:** Apply to InternViT-300M (quick experiments) and InternViT-6B (final results).

### What Makes It Top-Tier
- Novel formulation: position encoding as a content-adaptive mechanism, not just coordinate encoding
- Strong analysis component showing *why* standard PE fails for text-dense multilingual content
- Applicable beyond the specific model — any ViT-based MLLM can use this
- Addresses a real problem: text-dense document understanding is a critical MLLM weakness

### Feasibility (2x A100 80GB, 3-4 months)
- **Feasible with InternViT-300M:** Position encoding modification is lightweight. Fine-tuning InternViT-300M with LoRA on 2x A100 is straightforward.
- **InternViT-6B:** Tight but doable with DeepSpeed ZeRO-3. May need to limit to LoRA fine-tuning.
- **Risk:** Content-adaptive PE adds per-patch computation. Must keep overhead minimal (<5% FLOPs increase).
- **Timeline:** 2 weeks analysis + 4 weeks method + 4 weeks experiments + 2 weeks paper = 12 weeks

### Related Work & Gap
| Paper | Year | What It Does | What's Missing |
|-------|------|-------------|---------------|
| Qwen2-VL / Qwen2.5-VL | 2024-2025 | 2D-RoPE for native resolution | Pure geometric PE, no content awareness |
| HunyuanOCR (XD-RoPE) | 2025 | 4-subspace RoPE (text, H, W, time) | Fixed subspace decomposition, not adaptive |
| SaPE2 | 2025 | Semantic-aware PE based on local content | Only tested on CIFAR10/100, not scaled, not for text |
| EVA-02 | 2024 | 2D-RoPE + MIM pretraining | Standard RoPE, no content adaptation |
| Pixtral | 2024 | RoPE-2D for multi-image sequences | No content awareness |

**Gap:** No work adapts position encoding frequencies based on local content complexity for text-dense multilingual documents.

### Expected Contribution
1. CaPE: A content-adaptive position encoding that outperforms 2D-RoPE on text-dense benchmarks
2. Theoretical analysis: Why standard PE underperforms for text-dense images (frequency analysis)
3. Multi-script analysis: How different writing systems create different PE requirements
4. Drop-in module applicable to any ViT-based MLLM

### Assessment
- **Novelty:** HIGH (SaPE2 opens the door but only scratches the surface)
- **Depth:** HIGH (requires theoretical + empirical analysis)
- **Feasibility:** MEDIUM-HIGH (position encoding is lightweight but analysis is complex)
- **Impact:** MEDIUM-HIGH (addresses real problem but somewhat niche — document/OCR focused)
- **Competition risk:** MEDIUM (SaPE2 shows others are thinking about this; need to move fast)

---

## Direction 2: Content-Adaptive Visual Token Compression for Text-Dense MLLMs

### Title
**"Where to Look Matters: Text-Density-Guided Visual Token Compression for Efficient Document Understanding in MLLMs"**

### Core Research Question
Can a visual token compression mechanism that explicitly models text density and content importance — allocating more tokens to text-rich regions and fewer to background — outperform both static compression (pixel unshuffle) and existing adaptive methods (PyramidDrop, ATP-LLaVA) on document understanding tasks, while providing theoretical guarantees on information preservation?

### Why This Is PhD-Level
- **Novelty:** Existing token compression methods (PyramidDrop CVPR 2025, ATP-LLaVA CVPR 2025, IPCV Dec 2025, PVC CVPR 2025) use attention scores, CLS attention, or layer-wise similarity for importance scoring. **None** explicitly models text density as the compression signal. For document/OCR tasks — the fastest-growing MLLM application — text density is the *correct* importance signal.
- **Theoretical depth:** Formalize visual token compression as an information-theoretic problem: given a budget of K tokens, how to allocate them to maximize mutual information I(visual_tokens; text_answer)? Derive bounds on compression ratio vs. information loss for text-dense vs. natural images.
- **Strong empirical gap:** VTC-Bench (Oct 2025) showed that "simple image downsampling consistently outperforms many advanced compression methods" — this embarrassing finding means current methods lack the right inductive bias. Text-density guidance provides that bias.

### Key Technical Approach
1. **Text density estimation module (TDE):**
   - Lightweight CNN head attached to early ViT layers (layers 3-6)
   - Produces per-patch text density score: P(text | patch)
   - Pre-trained on synthetic text-density maps (generated from OCR annotations)
   - Frozen during main training — zero additional training cost for MLLM
2. **Density-guided token budget allocation:**
   - Given N input patches and target budget K tokens
   - Allocate K_i tokens to region i proportional to text density score
   - Within each region, use attention-based selection (similar to ATP-LLaVA) for fine-grained selection
   - Result: text-rich regions keep 90%+ tokens, background compressed to 10%
3. **Information recycling from compressed tokens:**
   - Compressed (dropped) tokens are not simply discarded
   - Aggregate compressed tokens within each local window via weighted mean (weights = attention scores)
   - Inject aggregated representation back as a "summary token" per window
   - Inspired by IPCV's Neighbor-Guided Reconstruction but with text-density-informed aggregation
4. **Integration into ViT encoder (not just LLM stage):**
   - Key insight: Most token compression works at the LLM stage (after projector). This misses the ViT's own quadratic cost.
   - Our compression operates *inside* the ViT at intermediate layers (like IPCV, LLaVA-UHD v3), reducing both ViT and LLM computation
   - Progressive compression: layer 12 (50% tokens for background), layer 24 (50% of remaining background) = aggressive background compression while preserving all text tokens

### What Makes It Top-Tier
- **Novel inductive bias:** Text density as the compression signal — obvious in hindsight, but nobody has done it
- **Information-theoretic formulation:** Not just "drop tokens" but "optimally allocate a token budget under information constraints"
- **Strong baselines exist for comparison:** PyramidDrop (CVPR 2025), ATP-LLaVA (CVPR 2025), IPCV (Dec 2025), PVC (CVPR 2025), MQT (NeurIPS 2024), TokenPacker (IJCV 2025)
- **Addresses the VTC-Bench embarrassment:** Explains *why* current methods fail (wrong importance signal) and provides the fix
- **Practical impact:** Document understanding is the #1 commercial MLLM application; more efficient document processing has massive real-world value

### Feasibility (2x A100 80GB, 3-4 months)
- **Highly feasible:**
  - Text density estimation: lightweight CNN head, minimal compute
  - Token compression module: implemented as masking/selection operations — nearly zero parameter overhead
  - Can prototype on LLaVA-1.5/LLaVA-NeXT first (fits easily on 2x A100), then validate on InternVL
  - Training: Only fine-tune compression module + LoRA on ViT. No full model training needed.
- **Risk:** Text density estimation quality may vary for handwritten / artistic text
- **Timeline:** 1 week analysis + 3 weeks method implementation + 4 weeks experiments + 2 weeks ablation + 2 weeks paper = 12 weeks

### Related Work & Gap
| Paper | Year | Venue | Compression Signal | Stage | Text-Aware? |
|-------|------|-------|-------------------|-------|-------------|
| PyramidDrop | 2025 | CVPR | Layer-wise similarity | LLM | No |
| ATP-LLaVA | 2025 | CVPR | Importance scores per layer | LLM | No |
| IPCV | 2025 | arXiv | Attention + neighbor reconstruction | ViT | No |
| PVC | 2025 | CVPR | Progressive merging | ViT+LLM | No |
| MQT | 2024 | NeurIPS | Matryoshka query selection | Projector | No |
| TokenPacker | 2025 | IJCV | Coarse-to-fine interpolation | Projector | No |
| AdaptInfer | 2025 | arXiv | Text-guided attention pruning | LLM | Text-query guided, not text-density |
| TokenCarve | 2025 | arXiv | Information-preserving selection | LLM | Implicitly focuses on text regions |
| DeepSeek-OCR | 2025 | arXiv | 16x encoder compression | Encoder | OCR-specific, not general compression |
| **Ours** | 2026 | - | **Text density map** | **ViT (inside)** | **Yes (explicit)** |

**Gap:** No method explicitly uses text density as the compression signal. No method operates inside the ViT with content-aware budget allocation. No method provides information-theoretic analysis of compression-vs-accuracy tradeoffs for text-dense images.

### Expected Contribution
1. Text-Density-Guided Compression (TDC): first token compression method with explicit text-awareness
2. Information-theoretic analysis: formal bounds on compression ratio vs. information loss
3. Intra-ViT compression: efficient compression inside the encoder, not just post-projector
4. SOTA efficiency-accuracy tradeoff on DocVQA, OCRBench, ChartQA, InfoVQA
5. Analysis of VTC-Bench failure modes and why text-density bias fixes them

### Assessment
- **Novelty:** VERY HIGH (explicit text-density signal is genuinely new; information-theoretic formulation adds depth)
- **Depth:** HIGH (theory + method + comprehensive experiments)
- **Feasibility:** VERY HIGH (lightweight module, can prototype quickly, strong baselines for comparison)
- **Impact:** VERY HIGH (document understanding is the #1 MLLM application; efficiency is always valued)
- **Competition risk:** MEDIUM (many groups working on token compression, but none with this angle)

---

## Direction 3: Vision Encoder Pre-training — From Contrastive to Autoregressive-Generative

### Title
**"Rethinking Vision Encoder Pre-training for MLLMs: Unifying Autoregressive and Spatial Objectives for Document-Aware Visual Features"**

### Core Research Question
Can a vision encoder pre-trained with a hybrid objective — combining autoregressive token prediction (AIMv2-style), spatial layout prediction, and text-reading objectives — produce features that are inherently better for document understanding in MLLMs than CLIP/SigLIP-pretrained encoders, and does this eliminate the need for expensive post-hoc adaptation?

### Why This Is PhD-Level
- **Paradigm contribution:** AIMv2 (CVPR 2025 Highlight) showed autoregressive pretraining beats CLIP/SigLIP. OpenVision2 showed you can drop the text encoder entirely. But nobody has designed a pretraining objective *specifically optimized for MLLM downstream tasks*, especially document understanding.
- **Addresses fundamental mismatch:** CLIP was designed for image-text matching. MLLMs use vision encoders for *generation* (feeding features to an autoregressive LLM). The pretraining-finetuning mismatch is real and documented.
- **Deep theoretical question:** What properties should vision features have for optimal LLM consumption?

### Key Technical Approach
1. **Hybrid pretraining objective:**
   - Autoregressive patch prediction (from AIMv2): predicts next patch given previous patches
   - Text reading objective: predict text content visible in the image patches (requires text-annotated images)
   - Layout-aware spatial objective: predict spatial relationships between text regions
2. **Architecture:** Standard ViT with 2D-RoPE + the above objectives
3. **Data:** Use existing captioned datasets (DataComp-1B) + document datasets with OCR annotations (IIT-CDIP, DocVQA train)
4. **Evaluation:** Compare with SigLIP, CLIP, AIMv2, OpenVision2 as vision backbone for same MLLM (e.g., LLaVA-NeXT)

### What Makes It Top-Tier
- Potential paradigm shift paper: "how should we pretrain vision encoders for MLLMs?"
- Builds on CVPR 2025 Highlight (AIMv2) — timely and relevant
- Could redefine the default vision encoder for the field

### Feasibility (2x A100 80GB, 3-4 months)
- **VERY RISKY:** Vision encoder pretraining from scratch requires massive compute
  - AIMv2 used 128 GPUs for the largest model
  - Even at ViT-B/16 scale, pretraining requires ~32-64 GPU-days
  - With 2x A100, this could take 4-8 weeks just for pretraining, leaving no time for MLLM integration
- **Potential mitigation:** Start from AIMv2 checkpoint and *continue* pretraining with the additional objectives (not from scratch). But this reduces novelty.
- **Timeline:** 4 weeks pretraining + 4 weeks MLLM integration + 3 weeks experiments + 2 weeks paper = 13 weeks (extremely tight)

### Related Work & Gap
| Paper | Year | Venue | Pretraining Objective | For MLLMs? | Document-Aware? |
|-------|------|-------|----------------------|-----------|----------------|
| CLIP/SigLIP/SigLIP2 | 2021-2025 | Various | Contrastive | No (adapted) | No |
| AIMv2 | 2025 | CVPR Highlight | Autoregressive | Yes (designed for) | No |
| OpenVision2 | 2025 | arXiv | Generative captioning only | Yes | No |
| DINOv2 | 2024 | TMLR | Self-supervised distillation | No | No |
| **Ours** | 2026 | - | **AR + Text-reading + Layout** | **Yes** | **Yes** |

### Expected Contribution
1. Document-aware pretraining objective for vision encoders
2. Analysis of pretraining-downstream mismatch in MLLM vision encoders
3. Publicly released pretrained encoder checkpoints

### Assessment
- **Novelty:** VERY HIGH (new pretraining paradigm)
- **Depth:** VERY HIGH (fundamental question about vision features for LLMs)
- **Feasibility:** LOW (pretraining is extremely compute-intensive; 2x A100 is insufficient for convincing results)
- **Impact:** VERY HIGH if successful (could be highly cited)
- **Competition risk:** HIGH (Apple, Google, UCSC-VLAA all actively working on this)

---

## Direction 4: Spatial-Aware Vision-Language Alignment via Structured Visual Tokens

### Title
**"Structured Visual Tokens: Encoding Spatial Layout Hierarchy in Vision-Language Alignment for Document Understanding"**

### Core Research Question
Can injecting explicit spatial structure (reading order, column boundaries, table grids, text-line groupings) into visual tokens — as structured tokens between the vision encoder and LLM — improve document and chart understanding beyond what flat visual token sequences provide, and does this bridge the spatial reasoning gap in current MLLMs?

### Why This Is PhD-Level
- **Addresses a fundamental problem:** Current MLLMs flatten 2D visual features into 1D token sequences, losing spatial structure. The LLM must *reconstruct* spatial relationships from flat tokens — an unnecessary burden.
- **Goes beyond LLaVA-SP:** LLaVA-SP adds 6 spatial tokens with simple convolution. This is a good start but lacks depth. We propose a full spatial hierarchy encoding with rich structural information.
- **Connects to spatial reasoning literature:** Spatial-MLLM (May 2025), SpatialVLM (CVPR 2024), ViCA2 all show spatial reasoning is a major MLLM weakness.

### Key Technical Approach
1. **Layout structure extraction:** Use a lightweight layout detector (e.g., LayoutLMv3 head, or simple convolution-based) to identify text regions, tables, charts, figures from ViT features
2. **Hierarchical spatial tokens:**
   - Level 1: Page-level layout token (encodes overall structure: single-column, multi-column, etc.)
   - Level 2: Region-level tokens (encode each detected region's type and bounding box)
   - Level 3: Relationship tokens (encode reading order, adjacency, containment)
3. **Integration via cross-attention:** Structured tokens attend to visual tokens via cross-attention, then are prepended to the visual token sequence
4. **Total overhead:** ~20-50 additional tokens (vs LLaVA-SP's 6), but encoding much richer information

### What Makes It Top-Tier
- Novel representation: structured spatial tokens as a bridge between vision encoder and LLM
- Principled approach to spatial reasoning in MLLMs
- Directly addresses the #1 weakness of MLLMs on structured documents

### Feasibility (2x A100 80GB, 3-4 months)
- **Feasible:** The structured token module is lightweight. Training only involves the spatial module + LoRA on the MLLM. Can work with LLaVA-1.5/LLaVA-NeXT as the base model.
- **Risk:** Layout detection quality may limit overall performance. Need robust layout features from ViT.
- **Timeline:** 2 weeks layout module + 3 weeks structured tokens + 4 weeks experiments + 2 weeks paper = 11 weeks

### Related Work & Gap
| Paper | Year | Key Approach | Limitation |
|-------|------|-------------|-----------|
| LLaVA-SP | 2025 (ICCV) | 6 spatial tokens via convolution | Too few tokens, no explicit structure |
| Spatial-MLLM | 2025 | Dual encoder (2D + 3D spatial) | Focused on 3D spatial reasoning, not documents |
| DocLayLLM | 2025 (CVPR) | Layout-aware document extension | Document-specific, not general MLLM |
| SpatialVLM | 2024 (CVPR) | Spatial reasoning training data | Data-driven, no architectural change |
| **Ours** | 2026 | **Hierarchical spatial structure tokens** | - |

### Expected Contribution
1. Structured Visual Tokens (SVT) as a general-purpose spatial encoding for MLLM
2. Analysis of spatial information loss in flat token sequences
3. Gains on document, chart, table benchmarks with minimal token overhead

### Assessment
- **Novelty:** MEDIUM-HIGH (builds on LLaVA-SP; spatial tokens are a natural extension)
- **Depth:** MEDIUM (more architectural than theoretical)
- **Feasibility:** HIGH (lightweight module, straightforward training)
- **Impact:** MEDIUM-HIGH (document/chart focused)
- **Competition risk:** MEDIUM (LLaVA-SP team and DocLayLLM team may extend their work)

---

## Direction 5: Multi-Script Vision Encoding — Optimizing ViT for Diverse Writing Systems

### Title
**"Script-Adaptive Vision Transformers: Multi-Scale Feature Selection for Cross-Script Document Understanding in MLLMs"**

### Core Research Question
Do different writing systems (Korean Hangul, Chinese Hanzi, Latin alphabet, Arabic, Devanagari) require different visual processing strategies (patch size, attention window size, feature scale) in vision transformers, and can a script-adaptive ViT that dynamically selects processing parameters per-region outperform one-size-fits-all encoders on multilingual document understanding?

### Why This Is PhD-Level
- **Unexplored territory:** No existing work studies how ViT architectural parameters should vary across writing systems. This is a genuinely novel research question.
- **Linguistic grounding:** Korean Hangul has compositional block structure (onset-nucleus-coda within square blocks). Chinese has dense stroke patterns. Latin is horizontally sparse. These have measurably different optimal patch sizes and attention windows.
- **Connects CV to linguistics:** Brings writing system typology into vision architecture design.

### Key Technical Approach
1. **Analysis phase:** Systematic study of ViT feature quality across writing systems at different scales:
   - Vary patch size (14, 16, 28) and measure OCR accuracy per script
   - Vary attention window size and measure text recognition accuracy per script
   - Visualize attention maps for different scripts
2. **Script-adaptive processing:**
   - Lightweight script detector (from early ViT layers) identifies script type per region
   - Dynamic patch size or multi-scale feature selection based on detected script
   - Korean/Chinese regions: smaller effective patches (finer detail for dense characters)
   - Latin/Arabic regions: standard patches
3. **Implementation:** Multi-scale ViT branches (similar to PIIP) but with script-based routing instead of resolution-based

### What Makes It Top-Tier
- Genuinely novel research question with no prior work
- Interdisciplinary (CV + computational linguistics)
- Practical implications for multilingual MLLM deployment

### Feasibility (2x A100 80GB, 3-4 months)
- **Feasible for analysis:** The analysis study (patch size vs. script accuracy) is straightforward and highly publishable even without a full method.
- **Challenging for full method:** Multi-scale routing adds complexity. PIIP-style architecture on InternViT-6B would be tight on 2x A100.
- **Fallback:** Even if the adaptive method doesn't fully work, the analysis alone (showing that different scripts need different processing) is a valuable contribution.
- **Data challenge:** Need multilingual document datasets with diverse scripts. DocML (14 languages) from HunyuanOCR could help.
- **Timeline:** 3 weeks analysis + 3 weeks method + 4 weeks experiments + 2 weeks paper = 12 weeks

### Related Work & Gap
| Paper | Year | Focus | Multi-Script? |
|-------|------|-------|--------------|
| dots.ocr | 2025 | Unified multilingual OCR (109 languages) | One-size-fits-all encoder |
| HunyuanOCR | 2025 | End-to-end OCR VLM (14+ languages) | XD-RoPE but same processing for all scripts |
| Pangea | 2025 | Multilingual MLLM (39 languages) | Data-driven multilingual, no script-aware encoding |
| Hangul ViT | 2025 | Korean character recognition | Korean-only, not multi-script |
| **Ours** | 2026 | **Script-adaptive ViT processing** | **First to study and optimize per-script** |

### Expected Contribution
1. First systematic analysis of ViT feature quality across writing systems
2. Script-adaptive vision processing for multilingual MLLMs
3. Design guidelines: recommended ViT configurations per writing system
4. Multi-script document understanding benchmark results

### Assessment
- **Novelty:** VERY HIGH (genuinely unexplored)
- **Depth:** HIGH (analysis component is very strong)
- **Feasibility:** MEDIUM (analysis is easy; full method is harder)
- **Impact:** MEDIUM (somewhat niche — multilingual document understanding)
- **Competition risk:** LOW (nobody is working on this specific question)

---

## Direction 6: Progressive Intra-Encoder Compression for Ultra-High-Resolution Vision

### Title
**"Encode More, Transmit Less: Progressive Token Compression Inside Vision Transformers for Native Ultra-High-Resolution MLLM Processing"**

### Core Research Question
Can progressive token compression applied *inside* the ViT encoder (not after it) — reducing tokens at intermediate layers based on local information density — enable native processing of 4K-8K resolution images within practical memory budgets (80GB) while preserving fine-grained details for document/chart understanding?

### Why This Is PhD-Level
- **Architectural innovation:** Most token compression happens *after* the ViT or *inside the LLM*. Compressing inside the ViT is much harder (bidirectional attention vs. causal) but much more impactful (reduces the ViT's own quadratic cost).
- **LLaVA-UHD v3 opened the door but left questions:** Their PVC method uses windowed compression at fixed layer positions. We ask: what is the *optimal* compression schedule? Should it be learned? Content-adaptive?
- **Enables new capability:** Native 4K-8K processing has been impractical. If we can compress inside the ViT, we unlock new use cases (full-page document processing, large charts).

### Key Technical Approach
1. **Baseline reproduction:** Implement LLaVA-UHD v3's PVC (windowed compression at fixed layers)
2. **Learned compression schedule:**
   - Instead of fixed compression at layers L={12, 24}, learn which layers to compress and by how much
   - Use a lightweight policy network (similar to NAS) to determine compression ratio per layer
   - Train with REINFORCE or Gumbel-Softmax to optimize compression schedule
3. **Content-adaptive compression within the encoder:**
   - At each compression layer, compute per-token importance (attention entropy, gradient magnitude)
   - Merge/drop low-importance tokens, keep high-importance ones
   - Progressive: 100% -> 80% -> 60% -> 40% token survival rate
4. **Memory optimization:** Use gradient checkpointing + windowed attention to enable 4K+ processing on 80GB GPUs

### What Makes It Top-Tier
- Extends LLaVA-UHD v3 (ICLR 2026 level work) with learned compression scheduling
- Enables new capability (native 4K-8K processing)
- Strong systems contribution (practical memory optimization)

### Feasibility (2x A100 80GB, 3-4 months)
- **Feasible:** Working inside the ViT is well-understood. LLaVA-UHD v3 code is open-source as starting point.
- **Risk:** Learned compression schedule adds training complexity. May need to simplify to rule-based schedule.
- **Risk:** 4K-8K resolution experiments are memory-intensive even with compression. Need careful engineering.
- **Timeline:** 2 weeks baseline + 3 weeks method + 4 weeks experiments + 2 weeks paper = 11 weeks

### Related Work & Gap
| Paper | Year | Where Compression Happens | Learned Schedule? | Content-Adaptive? |
|-------|------|--------------------------|-------------------|-------------------|
| LLaVA-UHD v3 | 2025 | Inside ViT (fixed layers) | No | No (windowed) |
| PVC | 2025 (CVPR) | ViT + LLM | No | Progressive but fixed |
| IPCV | 2025 | Inside ViT | No | Attention-based |
| PyramidDrop | 2025 (CVPR) | LLM layers | No | Similarity-based |
| FastVLM | 2025 (CVPR) | Encoder design | No | Architecture-level |
| **Ours** | 2026 | **Inside ViT (learned layers)** | **Yes** | **Yes (content-adaptive)** |

### Expected Contribution
1. Learned progressive compression schedule for ViT encoders
2. Native 4K-8K resolution processing within 80GB memory budget
3. Analysis of optimal compression-vs-layer relationship in ViTs
4. Practical efficiency gains: 2-4x speedup over LLaVA-UHD v3 at same quality

### Assessment
- **Novelty:** MEDIUM-HIGH (extends LLaVA-UHD v3 with learned scheduling)
- **Depth:** MEDIUM-HIGH (learned schedule is interesting but not deeply theoretical)
- **Feasibility:** MEDIUM (memory engineering for 4K+ is challenging)
- **Impact:** HIGH (efficiency + new capability)
- **Competition risk:** HIGH (LLaVA-UHD team, OpenGVLab, and others actively working here)

---

## Comparative Analysis

| Criterion (weight) | Dir 1: CaPE | Dir 2: TDC | Dir 3: Pretraining | Dir 4: SVT | Dir 5: Multi-Script | Dir 6: Progressive |
|---------------------|-------------|-------------|-------------------|------------|--------------------|--------------------|
| **Novelty** (25%) | HIGH | VERY HIGH | VERY HIGH | MEDIUM-HIGH | VERY HIGH | MEDIUM-HIGH |
| **Depth** (20%) | HIGH | HIGH | VERY HIGH | MEDIUM | HIGH | MEDIUM-HIGH |
| **Feasibility** (25%) | MEDIUM-HIGH | VERY HIGH | LOW | HIGH | MEDIUM | MEDIUM |
| **Impact** (15%) | MEDIUM-HIGH | VERY HIGH | VERY HIGH | MEDIUM-HIGH | MEDIUM | HIGH |
| **Competition Risk** (15%) | MEDIUM | MEDIUM | HIGH | MEDIUM | LOW | HIGH |
| **Weighted Score** | 3.6 | **4.3** | 3.2 | 3.1 | 3.4 | 3.0 |

Scoring: VERY HIGH=5, HIGH=4, MEDIUM-HIGH=3.5, MEDIUM=3, LOW=2 (for risk, inverted: LOW risk=5, MEDIUM=3, HIGH=1)

---

## Final Recommendation

### Primary Recommendation: Direction 2 — Content-Adaptive Visual Token Compression (TDC)

**Why this is the best choice:**

1. **Highest feasibility:** Lightweight module, can prototype on LLaVA in 1-2 weeks, scale to InternVL later. Fits comfortably on 2x A100. No pretraining needed.

2. **Genuine novelty with clear gap:** Despite 10+ token compression papers in 2025, none uses text density as the compression signal. The VTC-Bench finding that "simple downsampling beats advanced compression" is an embarrassment for the field — your paper would explain *why* (wrong importance signal) and fix it.

3. **Strong theoretical component:** Information-theoretic formulation (optimal token budget allocation under information constraints) elevates this from "another compression method" to a principled contribution.

4. **Maximum impact:** Document understanding is the fastest-growing MLLM application. Efficient document processing is commercially critical. This paper addresses both accuracy and efficiency.

5. **Clear experimental plan:** 7+ strong baselines (PyramidDrop, ATP-LLaVA, IPCV, PVC, MQT, TokenPacker, AdaptInfer), standard benchmarks (DocVQA, OCRBench, ChartQA, InfoVQA, TextVQA), and the VTC-Bench framework for fair comparison.

6. **Extensible to Korean/multilingual:** The text-density mechanism naturally helps for Korean (higher character density = more tokens preserved). This connects to your broader bilingual MLLM goal without requiring it as the sole focus.

### Secondary Recommendation: Direction 5 — Multi-Script Vision Encoding

**Why this is a strong backup:**

1. **Lowest competition risk:** Nobody is studying this. You would own this research direction.

2. **The analysis alone is publishable:** Even if the full adaptive method doesn't work, a systematic study showing "different scripts need different ViT configurations" is a valuable empirical finding for EMNLP/NAACL/ACL.

3. **Connects to your core expertise:** As someone building a Korean bilingual MLLM, you have domain knowledge that most CV researchers lack. The linguistic insight about Hangul's compositional block structure vs. Latin's sequential flow is genuinely novel in the CV context.

4. **Risk:** Impact may be perceived as limited to multilingual settings. Mitigate by framing broadly: "content-adaptive vision processing" where scripts are one instance of content types requiring different processing.

### What NOT to Pursue

- **Direction 3 (Pretraining):** Too compute-intensive for 2x A100. Apple, Google, and UCSC-VLAA have massive GPU clusters; you cannot compete on pretraining experiments. Save this for when you have more resources.

- **Direction 6 (Progressive Compression):** Too close to LLaVA-UHD v3 and IPCV. The competition risk is very high, and the novelty (learned schedule) may not be perceived as sufficient by reviewers.

---

## Suggested Research Plan for Direction 2 (TDC)

### Phase 1: Analysis & Motivation (Weeks 1-2)
- Reproduce VTC-Bench finding on your setup
- Analyze attention maps of existing compression methods on text-dense images
- Show quantitatively that text regions are disproportionately compressed by existing methods
- Write the motivation section of the paper

### Phase 2: Method Design & Implementation (Weeks 3-5)
- Implement text density estimation module (lightweight CNN on early ViT features)
- Implement density-guided token budget allocation
- Implement information recycling for compressed tokens
- Integrate into LLaVA-1.5/LLaVA-NeXT (smaller model for fast iteration)

### Phase 3: Core Experiments (Weeks 6-9)
- Compare against 7+ baselines on DocVQA, OCRBench, ChartQA, InfoVQA, TextVQA
- Ablation study: TDE quality, compression ratio, layer placement, recycling mechanism
- Efficiency measurements: FLOPs, latency, memory usage
- Scale to InternVL (with InternViT-300M or InternViT-6B with LoRA)

### Phase 4: Analysis & Theory (Weeks 10-11)
- Information-theoretic analysis of compression bounds
- Visualization of text-density maps and compression decisions
- Per-script analysis (Korean vs English text density and compression behavior)
- VTC-Bench evaluation to show improvement over prior methods

### Phase 5: Paper Writing (Weeks 11-14)
- Draft paper following CVPR/NeurIPS format
- Target: CVPR 2027 (deadline likely ~Nov 2026) or NeurIPS 2026 (deadline likely ~May 2026)
- If targeting NeurIPS 2026: compress to 10-12 weeks total

---

## Sources

- [Qwen2-VL: 2D-RoPE Architecture](https://arxiv.org/html/2409.12191v1)
- [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923)
- [SaPE2: 2D Semantic-Aware Position Encoding](https://arxiv.org/abs/2505.09466)
- [HunyuanOCR Technical Report (XD-RoPE)](https://arxiv.org/html/2511.19575v1)
- [SigLIP 2: Multilingual Vision-Language Encoders](https://arxiv.org/abs/2502.14786)
- [AIMv2: Multimodal Autoregressive Pre-training](https://arxiv.org/abs/2411.14402)
- [OpenVision 2: Generative Pretrained Visual Encoders](https://arxiv.org/abs/2509.01644)
- [PyramidDrop: Visual Redundancy Reduction (CVPR 2025)](https://github.com/Cooperx521/PyramidDrop)
- [ATP-LLaVA: Adaptive Token Pruning (CVPR 2025)](https://cvpr.thecvf.com/virtual/2025/poster/33610)
- [IPCV: Information-Preserving Compression](https://arxiv.org/abs/2512.18747)
- [PVC: Progressive Visual Token Compression (CVPR 2025)](https://github.com/OpenGVLab/PVC)
- [MQT-LLaVA: Matryoshka Query Transformer](https://gordonhu608.github.io/mqtllava/)
- [TokenPacker: Efficient Visual Projector (IJCV 2025)](https://github.com/CircleRadon/TokenPacker)
- [AdaptInfer: Adaptive Token Pruning with Text Guidance](https://arxiv.org/abs/2508.06084)
- [TokenCarve: Information-Preserving Compression](https://arxiv.org/html/2503.10501v1)
- [DeepSeek-OCR: Context Optical Compression](https://arxiv.org/html/2510.18234v1)
- [VTC-Bench: Evaluation Framework for Visual Token Compression](https://arxiv.org/abs/2510.07143)
- [LLaVA-UHD v3: Progressive Visual Compression](https://github.com/thunlp/LLaVA-UHD)
- [LLaVA-UHD v2: Hierarchical Window Transformer](https://arxiv.org/abs/2412.13871)
- [LLaVA-SP: Visual Spatial Tokens (ICCV 2025)](https://arxiv.org/abs/2507.00505)
- [Spatial-MLLM: Visual-based Spatial Intelligence](https://arxiv.org/abs/2505.23747)
- [PIIP-LLaVA: Parameter-Inverted Image Pyramid](https://arxiv.org/abs/2501.07783)
- [InternVL 2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)
- [InternVL 3.0](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)
- [FastVLM: Efficient Vision Encoding](https://machinelearning.apple.com/research/fast-vision-language-models)
- [dots.ocr: Multilingual Document Parsing](https://www.marktechpost.com/2025/08/16/meet-dots-ocr-a-new-1-7b-vision-language-model-that-achieves-sota-performance-on-multilingual-document-parsing/)
- [Pangea: Multilingual Multimodal LLM](https://neulab.github.io/Pangea/)
- [DocLayLLM: Document Understanding (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_DocLayLLM_An_Efficient_Multi-modal_Extension_of_Large_Language_Models_for_CVPR_2025_paper.pdf)
- [Survey of Multimodal Token Compression](https://arxiv.org/pdf/2507.20198)
- [TokenFD: Token-level Text Image Foundation Model (ICCV 2025)](https://arxiv.org/html/2503.02304)
- [Text or Pixels? Token Efficiency of Visual Text (Oct 2025)](https://arxiv.org/html/2510.18279v1)
