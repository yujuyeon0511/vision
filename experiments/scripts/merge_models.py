"""
TBI-MLLM Model Merging Script (DARE+TIES)
===========================================
Merges English-base and Korean-finetuned models using DARE+TIES.

Usage:
    python experiments/scripts/merge_models.py \
        --method dare_ties \
        --model_en results/EXP-20260211-001/E2/seed_42/phase2/last \
        --model_ko results/EXP-20260211-001/F4/seed_42/phase3a/last \
        --lambda_en 0.6 --lambda_ko 0.4 \
        --dare_drop_rate 0.9 \
        --output_dir results/EXP-20260211-001/F4/seed_42/phase3b_merged

    # Grid search for optimal lambda
    python experiments/scripts/merge_models.py \
        --method dare_ties \
        --model_en path/to/en --model_ko path/to/ko \
        --grid_search \
        --output_dir results/merge_grid_search
"""

import argparse
import copy
import logging
import os
from collections import OrderedDict

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tbi-mllm-merge")


# ---------------------------------------------------------------------------
# DARE (Drop And REscale)
# ---------------------------------------------------------------------------

def dare_drop(task_vector: dict, drop_rate: float, seed: int = 42) -> dict:
    """Apply DARE: randomly drop a fraction of weight deltas and rescale.

    Args:
        task_vector: dict of {param_name: delta_tensor} (fine-tuned - base)
        drop_rate: fraction of deltas to zero out (e.g., 0.9 = drop 90%)
        seed: random seed for reproducibility

    Returns:
        Pruned and rescaled task vector.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    result = {}
    total_params = 0
    dropped_params = 0

    for name, delta in task_vector.items():
        if delta.dim() == 0:
            result[name] = delta.clone()
            continue

        # Create binary mask: 1 = keep, 0 = drop
        mask = torch.bernoulli(
            torch.full_like(delta, 1.0 - drop_rate, dtype=torch.float32),
            generator=rng,
        ).to(delta.dtype)

        # Rescale kept values: divide by (1 - drop_rate) to maintain expected magnitude
        rescale_factor = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 0.0
        result[name] = delta * mask * rescale_factor

        total_params += delta.numel()
        dropped_params += (mask == 0).sum().item()

    logger.info(
        f"DARE: dropped {dropped_params:,}/{total_params:,} params "
        f"({dropped_params / max(total_params, 1) * 100:.1f}%), rescale={1.0 / (1.0 - drop_rate):.2f}x"
    )
    return result


# ---------------------------------------------------------------------------
# TIES (TrIm, Elect Sign, mErge)
# ---------------------------------------------------------------------------

def ties_merge(task_vectors: list, weights: list, top_k_percent: float = 0.2) -> dict:
    """Apply TIES merging to multiple task vectors.

    Steps:
    1. TRIM: Keep only top-K% parameters by magnitude per task vector
    2. ELECT SIGN: Resolve sign conflicts by majority vote (mass-based)
    3. MERGE: Average the agreeing parameters with given weights

    Args:
        task_vectors: list of task vector dicts
        weights: list of merging weights (e.g., [0.6, 0.4])
        top_k_percent: fraction of parameters to keep (per vector)

    Returns:
        Merged task vector.
    """
    assert len(task_vectors) == len(weights), "Must have same number of vectors and weights"

    all_param_names = set()
    for tv in task_vectors:
        all_param_names.update(tv.keys())

    merged = {}
    for name in all_param_names:
        deltas = []
        w = []
        for tv, weight in zip(task_vectors, weights):
            if name in tv:
                deltas.append(tv[name])
                w.append(weight)

        if not deltas:
            continue

        if deltas[0].dim() == 0:
            # Scalar: weighted average
            merged[name] = sum(d * wi for d, wi in zip(deltas, w))
            continue

        # Step 1: TRIM — keep top-K% by magnitude
        trimmed = []
        for delta in deltas:
            threshold = torch.quantile(delta.abs().float(), 1.0 - top_k_percent)
            mask = delta.abs() >= threshold
            trimmed.append(delta * mask)

        # Step 2: ELECT SIGN — resolve sign conflicts by mass
        # For each parameter position, compute the "mass" for positive and negative
        pos_mass = torch.zeros_like(trimmed[0], dtype=torch.float32)
        neg_mass = torch.zeros_like(trimmed[0], dtype=torch.float32)

        for delta, weight in zip(trimmed, w):
            pos_mask = delta > 0
            neg_mask = delta < 0
            pos_mass += delta.abs().float() * pos_mask.float() * weight
            neg_mass += delta.abs().float() * neg_mask.float() * weight

        elected_sign = torch.where(pos_mass >= neg_mass, torch.ones_like(pos_mass), -torch.ones_like(pos_mass))

        # Step 3: MERGE — average parameters that agree with elected sign
        numerator = torch.zeros_like(trimmed[0], dtype=torch.float32)
        denominator = torch.zeros_like(trimmed[0], dtype=torch.float32)

        for delta, weight in zip(trimmed, w):
            # Only include parameters that agree with the elected sign
            sign_agree = (delta.sign() == elected_sign) | (delta == 0)
            numerator += delta.float() * sign_agree.float() * weight
            denominator += sign_agree.float() * weight

        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)
        merged[name] = (numerator / denominator).to(trimmed[0].dtype)

    return merged


# ---------------------------------------------------------------------------
# Full DARE+TIES pipeline
# ---------------------------------------------------------------------------

def dare_ties_merge(
    base_state_dict: dict,
    finetuned_state_dicts: list,
    lambdas: list,
    dare_drop_rate: float = 0.9,
    ties_top_k: float = 0.2,
    seed: int = 42,
) -> dict:
    """Full DARE+TIES merging pipeline.

    Args:
        base_state_dict: pre-trained base model weights
        finetuned_state_dicts: list of fine-tuned model weights
        lambdas: merging weights for each fine-tuned model
        dare_drop_rate: DARE drop rate
        ties_top_k: TIES trim percentage
        seed: random seed

    Returns:
        Merged state dict.
    """
    logger.info(f"DARE+TIES merging: {len(finetuned_state_dicts)} models, "
                f"lambdas={lambdas}, dare_drop={dare_drop_rate}, ties_top_k={ties_top_k}")

    # Step 1: Compute task vectors (delta = finetuned - base)
    task_vectors = []
    for i, ft_sd in enumerate(finetuned_state_dicts):
        tv = {}
        for name in base_state_dict:
            if name in ft_sd:
                tv[name] = ft_sd[name].float() - base_state_dict[name].float()
        task_vectors.append(tv)
        logger.info(f"  Task vector {i}: {len(tv)} parameters")

    # Step 2: Apply DARE to each task vector
    dare_vectors = []
    for i, tv in enumerate(task_vectors):
        dare_tv = dare_drop(tv, dare_drop_rate, seed=seed + i)
        dare_vectors.append(dare_tv)

    # Step 3: Apply TIES to merge the DARE-pruned task vectors
    merged_tv = ties_merge(dare_vectors, lambdas, top_k_percent=ties_top_k)

    # Step 4: Add merged task vector back to base
    merged_sd = OrderedDict()
    for name, param in base_state_dict.items():
        if name in merged_tv:
            merged_sd[name] = (param.float() + merged_tv[name].float()).to(param.dtype)
        else:
            merged_sd[name] = param.clone()

    logger.info(f"Merged model: {len(merged_sd)} parameters")
    return merged_sd


# ---------------------------------------------------------------------------
# Simple merging methods
# ---------------------------------------------------------------------------

def linear_merge(state_dicts: list, weights: list) -> dict:
    """Simple weighted average of state dicts."""
    merged = OrderedDict()
    for name in state_dicts[0]:
        merged[name] = sum(
            sd[name].float() * w for sd, w in zip(state_dicts, weights)
            if name in sd
        ).to(state_dicts[0][name].dtype)
    return merged


def slerp_merge(sd1: dict, sd2: dict, t: float = 0.5) -> dict:
    """Spherical linear interpolation between two state dicts."""
    merged = OrderedDict()
    for name in sd1:
        if name not in sd2:
            merged[name] = sd1[name].clone()
            continue
        v1 = sd1[name].float().flatten()
        v2 = sd2[name].float().flatten()

        # Compute cosine angle
        dot = torch.dot(v1, v2)
        norm1, norm2 = v1.norm(), v2.norm()
        if norm1 < 1e-8 or norm2 < 1e-8:
            merged[name] = ((1 - t) * sd1[name].float() + t * sd2[name].float()).to(sd1[name].dtype)
            continue

        cos_angle = torch.clamp(dot / (norm1 * norm2), -1.0, 1.0)
        angle = torch.acos(cos_angle)

        if angle.abs() < 1e-6:
            merged[name] = ((1 - t) * sd1[name].float() + t * sd2[name].float()).to(sd1[name].dtype)
        else:
            sin_angle = torch.sin(angle)
            w1 = torch.sin((1 - t) * angle) / sin_angle
            w2 = torch.sin(t * angle) / sin_angle
            result = (w1 * v1 + w2 * v2).reshape(sd1[name].shape)
            merged[name] = result.to(sd1[name].dtype)

    return merged


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search_lambdas(
    base_sd: dict,
    ft_sds: list,
    dare_drop_rate: float,
    ties_top_k: float,
    output_dir: str,
    tokenizer,
    model_cls,
    seed: int = 42,
):
    """Grid search over lambda values to find optimal merging weights."""
    lambda_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    for lam_en in lambda_values:
        lam_ko = 1.0 - lam_en
        logger.info(f"\n--- Grid search: lambda_en={lam_en}, lambda_ko={lam_ko} ---")

        merged_sd = dare_ties_merge(
            base_sd, ft_sds, [lam_en, lam_ko],
            dare_drop_rate, ties_top_k, seed,
        )

        # Save merged model
        variant_dir = os.path.join(output_dir, f"lambda_en{lam_en}_ko{lam_ko}")
        os.makedirs(variant_dir, exist_ok=True)

        # Create a model and load the merged weights
        model = model_cls.from_pretrained(
            args.model_en,  # use EN model config
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.load_state_dict(merged_sd, strict=False)
        model.save_pretrained(variant_dir)
        tokenizer.save_pretrained(variant_dir)

        results.append({
            "lambda_en": lam_en,
            "lambda_ko": lam_ko,
            "model_path": variant_dir,
        })
        logger.info(f"  Saved to: {variant_dir}")

    # Save grid search results
    import json
    with open(os.path.join(output_dir, "grid_search_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nGrid search complete. {len(results)} variants saved to {output_dir}")
    logger.info("Run evaluate.py on each variant to find the optimal lambda.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TBI-MLLM Model Merging (DARE+TIES)")
    parser.add_argument("--method", type=str, default="dare_ties",
                        choices=["dare_ties", "ties", "dare", "linear", "slerp"],
                        help="Merging method")
    parser.add_argument("--model_en", type=str, required=True, help="Path to English (base) model")
    parser.add_argument("--model_ko", type=str, required=True, help="Path to Korean fine-tuned model")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Path to pre-trained base model (for task vector computation). Defaults to model_en.")
    parser.add_argument("--lambda_en", type=float, default=0.6, help="Weight for English model")
    parser.add_argument("--lambda_ko", type=float, default=0.4, help="Weight for Korean model")
    parser.add_argument("--dare_drop_rate", type=float, default=0.9, help="DARE drop rate")
    parser.add_argument("--ties_top_k", type=float, default=0.2, help="TIES top-K percent to keep")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search over lambda values")
    return parser.parse_args()


def main():
    global args
    args = parse_args()

    logger.info(f"Method: {args.method}")
    logger.info(f"English model: {args.model_en}")
    logger.info(f"Korean model: {args.model_ko}")
    logger.info(f"Lambda: EN={args.lambda_en}, KO={args.lambda_ko}")

    # Load models
    logger.info("Loading English model...")
    model_en = AutoModel.from_pretrained(
        args.model_en, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    sd_en = model_en.state_dict()

    logger.info("Loading Korean model...")
    model_ko = AutoModel.from_pretrained(
        args.model_ko, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    sd_ko = model_ko.state_dict()

    tokenizer = AutoTokenizer.from_pretrained(args.model_en, trust_remote_code=True)

    # Base model for task vector computation
    if args.model_base:
        logger.info("Loading base model for task vectors...")
        model_base = AutoModel.from_pretrained(
            args.model_base, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        sd_base = model_base.state_dict()
        del model_base
    else:
        sd_base = sd_en  # treat EN as the base

    # Grid search mode
    if args.grid_search:
        grid_search_lambdas(
            sd_base, [sd_en, sd_ko],
            args.dare_drop_rate, args.ties_top_k,
            args.output_dir, tokenizer, type(model_en), args.seed,
        )
        return

    # Merge
    if args.method == "dare_ties":
        merged_sd = dare_ties_merge(
            sd_base, [sd_en, sd_ko],
            [args.lambda_en, args.lambda_ko],
            args.dare_drop_rate, args.ties_top_k, args.seed,
        )
    elif args.method == "ties":
        # TIES without DARE
        tv_en = {n: sd_en[n].float() - sd_base[n].float() for n in sd_base if n in sd_en}
        tv_ko = {n: sd_ko[n].float() - sd_base[n].float() for n in sd_base if n in sd_ko}
        merged_tv = ties_merge([tv_en, tv_ko], [args.lambda_en, args.lambda_ko], args.ties_top_k)
        merged_sd = OrderedDict()
        for name, param in sd_base.items():
            if name in merged_tv:
                merged_sd[name] = (param.float() + merged_tv[name]).to(param.dtype)
            else:
                merged_sd[name] = param.clone()
    elif args.method == "dare":
        # DARE without TIES (simple weighted average after dropping)
        tv_en = {n: sd_en[n].float() - sd_base[n].float() for n in sd_base if n in sd_en}
        tv_ko = {n: sd_ko[n].float() - sd_base[n].float() for n in sd_base if n in sd_ko}
        dare_en = dare_drop(tv_en, args.dare_drop_rate, args.seed)
        dare_ko = dare_drop(tv_ko, args.dare_drop_rate, args.seed + 1)
        merged_sd = OrderedDict()
        for name, param in sd_base.items():
            delta = torch.zeros_like(param.float())
            if name in dare_en:
                delta += dare_en[name] * args.lambda_en
            if name in dare_ko:
                delta += dare_ko[name] * args.lambda_ko
            merged_sd[name] = (param.float() + delta).to(param.dtype)
    elif args.method == "linear":
        merged_sd = linear_merge([sd_en, sd_ko], [args.lambda_en, args.lambda_ko])
    elif args.method == "slerp":
        merged_sd = slerp_merge(sd_en, sd_ko, t=args.lambda_ko)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Free memory
    del model_en, model_ko, sd_en, sd_ko
    torch.cuda.empty_cache()

    # Save merged model
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving merged model to: {args.output_dir}")

    # Reload model structure and load merged weights
    model_merged = AutoModel.from_pretrained(
        args.model_en,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_merged.load_state_dict(merged_sd, strict=False)
    model_merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save merge metadata
    import json
    metadata = {
        "method": args.method,
        "model_en": args.model_en,
        "model_ko": args.model_ko,
        "model_base": args.model_base,
        "lambda_en": args.lambda_en,
        "lambda_ko": args.lambda_ko,
        "dare_drop_rate": args.dare_drop_rate,
        "ties_top_k": args.ties_top_k,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "merge_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Merge complete!")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Weights: EN={args.lambda_en}, KO={args.lambda_ko}")


if __name__ == "__main__":
    main()
