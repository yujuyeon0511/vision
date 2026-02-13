"""
TBI-MLLM MoLE (Mixture of Language Experts) Training Script
=============================================================
Initializes MoLE experts from merged/KO/EN checkpoints and trains the router.

Architecture:
  - Replace FFN in top decoder layers with 3-expert MoE FFN
  - Expert 0: Korean specialist (init from KO fine-tuned model)
  - Expert 1: English specialist (init from original EN model)
  - Expert 2: Shared expert (init from DARE+TIES merged model)
  - Router: learned 2-layer MLP with language hint input

Usage:
    deepspeed --hostfile experiments/configs/multi-node/hostfile \
        --master_addr 192.168.0.28 --master_port 29500 \
        experiments/scripts/train_mole.py \
        --config experiments/configs/EXP-20260211-001-config.yaml \
        --variant F4 --phase 3c_mole --seed 42 \
        --init_merged results/.../phase3b_merged \
        --init_ko results/.../phase3a/last \
        --init_en results/.../phase2/last \
        --output_dir results/.../phase3c \
        --deepspeed experiments/configs/multi-node/ds_config_zero2_multinode.json
"""

import argparse
import copy
import json
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tbi-mllm-mole")


# ---------------------------------------------------------------------------
# MoLE modules
# ---------------------------------------------------------------------------

class LanguageRouter(nn.Module):
    """Router that dispatches tokens to language-specific experts.

    Input features: hidden_state (from decoder) + optional language_hint
    Output: expert weights [num_experts]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 3,
        router_hidden_dim: int = 256,
        temperature: float = 1.0,
        use_language_hint: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_language_hint = use_language_hint

        input_dim = hidden_dim + (1 if use_language_hint else 0)

        self.router = nn.Sequential(
            nn.Linear(input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_experts),
        )

    def forward(self, hidden_states: torch.Tensor, language_ids: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            language_ids: [batch, seq_len] with 0=EN, 1=KO (optional)

        Returns:
            routing_weights: [batch, seq_len, num_experts] (softmax probabilities)
            routing_logits: [batch, seq_len, num_experts] (raw logits for aux losses)
        """
        if self.use_language_hint and language_ids is not None:
            lang_feature = language_ids.unsqueeze(-1).float()
            router_input = torch.cat([hidden_states, lang_feature], dim=-1)
        else:
            router_input = hidden_states

        logits = self.router(router_input)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights, logits


class MoLEFFN(nn.Module):
    """Mixture of Language Experts FFN layer.

    Replaces a single FFN with num_experts parallel FFNs + router.
    """

    def __init__(
        self,
        original_ffn: nn.Module,
        num_experts: int = 3,
        hidden_dim: int = 4096,
        router_hidden_dim: int = 256,
        temperature: float = 1.0,
        use_language_hint: bool = True,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Create expert copies (will be initialized from different checkpoints later)
        self.experts = nn.ModuleList([
            copy.deepcopy(original_ffn) for _ in range(num_experts)
        ])

        # Router
        self.router = LanguageRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            router_hidden_dim=router_hidden_dim,
            temperature=temperature,
            use_language_hint=use_language_hint,
        )

    def forward(self, hidden_states: torch.Tensor, language_ids: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            language_ids: [batch, seq_len]

        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss: auxiliary load balancing loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get routing weights
        routing_weights, routing_logits = self.router(hidden_states, language_ids)

        # Compute expert outputs (all experts, weighted sum)
        expert_outputs = torch.stack(
            [expert(hidden_states) for expert in self.experts],
            dim=2,
        )  # [batch, seq_len, num_experts, hidden_dim]

        # Weighted combination
        output = torch.einsum("bse,bsed->bsd", routing_weights, expert_outputs)

        # Compute auxiliary load-balancing loss
        aux_loss = self._compute_load_balance_loss(routing_weights, routing_logits)

        return output, aux_loss

    def _compute_load_balance_loss(self, weights: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute Switch Transformer-style load balancing loss.

        Encourages equal distribution of tokens across experts.
        """
        # f_i: fraction of tokens routed to expert i
        # P_i: average routing probability for expert i
        num_experts = self.num_experts
        f = weights.mean(dim=[0, 1])  # [num_experts]
        P = F.softmax(logits, dim=-1).mean(dim=[0, 1])  # [num_experts]

        # Loss = N * sum(f_i * P_i)
        # Minimum when f_i = P_i = 1/N (uniform distribution)
        loss = num_experts * (f * P).sum()
        return loss


class MoLERouter_Z_Loss(nn.Module):
    """Router z-loss to prevent logit explosion."""

    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """z-loss = mean(log(sum(exp(logits)))^2)"""
        log_z = torch.logsumexp(logits, dim=-1)
        return self.weight * (log_z ** 2).mean()


# ---------------------------------------------------------------------------
# Model surgery: inject MoLE into decoder
# ---------------------------------------------------------------------------

def inject_mole(model, mole_config: dict) -> nn.Module:
    """Replace FFN layers in specified decoder layers with MoLE FFN.

    Args:
        model: InternVL model
        mole_config: MoLE configuration dict

    Returns:
        Modified model with MoLE layers.
    """
    layers_to_replace = mole_config.get("layers_to_replace", [16, 20, 24, 28])
    num_experts = mole_config.get("num_experts", 3)
    router_cfg = mole_config.get("router", {})

    # Get the decoder (LLM) module
    llm = getattr(model, "language_model", None) or getattr(model, "llm", None)
    if llm is None:
        raise RuntimeError("Could not find LLM module in model")

    # Navigate to the transformer layers
    # InternLM2: model.layers[i].feed_forward
    layers = None
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        layers = llm.model.layers
    elif hasattr(llm, "layers"):
        layers = llm.layers
    elif hasattr(llm, "transformer") and hasattr(llm.transformer, "layers"):
        layers = llm.transformer.layers

    if layers is None:
        raise RuntimeError("Could not find decoder layers. Adjust inject_mole() for your model architecture.")

    hidden_dim = None
    replaced = 0

    for layer_idx in layers_to_replace:
        if layer_idx >= len(layers):
            logger.warning(f"Layer {layer_idx} does not exist (model has {len(layers)} layers), skipping")
            continue

        layer = layers[layer_idx]

        # Find the FFN module
        ffn = getattr(layer, "feed_forward", None) or getattr(layer, "mlp", None)
        if ffn is None:
            logger.warning(f"Could not find FFN in layer {layer_idx}, skipping")
            continue

        # Infer hidden dim from the FFN
        if hidden_dim is None:
            for p in ffn.parameters():
                hidden_dim = p.shape[-1] if p.dim() >= 2 else p.shape[0]
                break

        # Create MoLE FFN
        mole_ffn = MoLEFFN(
            original_ffn=ffn,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            router_hidden_dim=router_cfg.get("hidden_dim", 256),
            temperature=router_cfg.get("temperature", 1.0),
            use_language_hint=router_cfg.get("use_language_hint", True),
            capacity_factor=mole_config.get("expert_capacity_factor", 1.25),
        )

        # Replace FFN with MoLE FFN
        if hasattr(layer, "feed_forward"):
            layer.feed_forward = mole_ffn
        elif hasattr(layer, "mlp"):
            layer.mlp = mole_ffn

        replaced += 1
        logger.info(f"  Replaced layer {layer_idx} FFN with MoLE ({num_experts} experts)")

    logger.info(f"MoLE injection complete: {replaced}/{len(layers_to_replace)} layers replaced")
    return model


def initialize_experts(model, init_ko_path: str, init_en_path: str, init_merged_path: str, mole_config: dict):
    """Initialize MoLE experts from different checkpoints.

    Expert 0: Korean specialist (from KO fine-tuned model)
    Expert 1: English specialist (from original EN model)
    Expert 2: Shared (from merged model — already loaded as the base)
    """
    layers_to_replace = mole_config.get("layers_to_replace", [16, 20, 24, 28])

    # Load source models
    logger.info(f"Loading KO model for expert init: {init_ko_path}")
    sd_ko = AutoModel.from_pretrained(
        init_ko_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).state_dict()

    logger.info(f"Loading EN model for expert init: {init_en_path}")
    sd_en = AutoModel.from_pretrained(
        init_en_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).state_dict()

    # Get the decoder layers
    llm = getattr(model, "language_model", None) or getattr(model, "llm", None)
    layers = None
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        layers = llm.model.layers
    elif hasattr(llm, "layers"):
        layers = llm.layers
    elif hasattr(llm, "transformer") and hasattr(llm.transformer, "layers"):
        layers = llm.transformer.layers

    for layer_idx in layers_to_replace:
        if layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]
        mole_ffn = getattr(layer, "feed_forward", None) or getattr(layer, "mlp", None)

        if not isinstance(mole_ffn, MoLEFFN):
            continue

        # Initialize experts from source models
        # Expert 0: Korean specialist
        _copy_ffn_weights(mole_ffn.experts[0], sd_ko, layer_idx)
        # Expert 1: English specialist
        _copy_ffn_weights(mole_ffn.experts[1], sd_en, layer_idx)
        # Expert 2: Shared (already initialized from merged model via deepcopy)
        # No action needed — it already has the merged weights

        logger.info(f"  Layer {layer_idx}: experts initialized (KO, EN, shared)")

    del sd_ko, sd_en
    torch.cuda.empty_cache()
    logger.info("Expert initialization complete")


def _copy_ffn_weights(expert_ffn: nn.Module, source_sd: dict, layer_idx: int):
    """Copy FFN weights from a source state dict to an expert module."""
    expert_sd = expert_ffn.state_dict()

    # Build mapping: find FFN-related keys for this layer in the source
    # Common patterns: model.layers.{idx}.feed_forward.* or model.layers.{idx}.mlp.*
    layer_prefix_patterns = [
        f"language_model.model.layers.{layer_idx}.feed_forward.",
        f"language_model.model.layers.{layer_idx}.mlp.",
        f"llm.model.layers.{layer_idx}.feed_forward.",
        f"llm.model.layers.{layer_idx}.mlp.",
        f"model.layers.{layer_idx}.feed_forward.",
        f"model.layers.{layer_idx}.mlp.",
    ]

    for source_key, source_val in source_sd.items():
        for prefix in layer_prefix_patterns:
            if source_key.startswith(prefix):
                # Extract the local key name
                local_key = source_key[len(prefix):]
                if local_key in expert_sd:
                    expert_sd[local_key] = source_val.clone()
                break

    expert_ffn.load_state_dict(expert_sd, strict=False)


# ---------------------------------------------------------------------------
# Custom Trainer for MoLE
# ---------------------------------------------------------------------------

class MoLETrainer(Trainer):
    """Custom Trainer that handles MoLE auxiliary losses."""

    def __init__(self, *args, load_balance_weight=0.01, z_loss_weight=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_balance_weight = load_balance_weight
        self.z_loss_weight = z_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with auxiliary MoLE losses."""
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Collect auxiliary losses from MoLE layers
        aux_loss = torch.tensor(0.0, device=loss.device)
        n_mole_layers = 0

        llm = getattr(model, "language_model", None) or getattr(model, "llm", None)
        if llm is not None:
            layers = None
            if hasattr(llm, "model") and hasattr(llm.model, "layers"):
                layers = llm.model.layers
            elif hasattr(llm, "layers"):
                layers = llm.layers

            if layers is not None:
                for layer in layers:
                    ffn = getattr(layer, "feed_forward", None) or getattr(layer, "mlp", None)
                    if isinstance(ffn, MoLEFFN) and hasattr(ffn, "_last_aux_loss"):
                        aux_loss = aux_loss + ffn._last_aux_loss
                        n_mole_layers += 1

        if n_mole_layers > 0:
            aux_loss = aux_loss / n_mole_layers * self.load_balance_weight

        total_loss = loss + aux_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ---------------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------------

def setup_training(config: dict, args):
    """Set up MoLE training."""
    mole_config = config["model"]["decoder"]["mole"]

    # Load merged model as base
    logger.info(f"Loading merged model: {args.init_merged}")
    model = AutoModel.from_pretrained(
        args.init_merged,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.init_merged, trust_remote_code=True)

    # Inject MoLE layers
    logger.info("Injecting MoLE layers...")
    model = inject_mole(model, mole_config)

    # Initialize experts from source models
    logger.info("Initializing experts...")
    initialize_experts(model, args.init_ko, args.init_en, args.init_merged, mole_config)

    # Freeze everything except MoLE components (router + experts)
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze MoLE components
    mole_params = 0
    for name, param in model.named_parameters():
        if "router" in name or "experts" in name:
            param.requires_grad = True
            mole_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable MoLE params: {mole_params:,} / {total_params:,} ({mole_params / total_params * 100:.2f}%)")

    return model, tokenizer


def build_mole_training_args(config: dict, args) -> TrainingArguments:
    """Build training arguments for MoLE phase."""
    phase3_cfg = config["training"]["phases"]["phase3"]

    # MoLE uses the mole_balanced data mixing
    # Steps: D4/F4 config specifies 25000 steps for MoLE training
    variant_cfg = config.get("variant", {})
    stages = None
    if isinstance(phase3_cfg, dict) and "stages" in phase3_cfg:
        stages = phase3_cfg["stages"]
    elif isinstance(variant_cfg.get("training", {}).get("phases", {}).get("phase3", {}), dict):
        stages = variant_cfg["training"]["phases"]["phase3"].get("stages", [])

    mole_steps = 25000
    if stages:
        for s in stages:
            if s.get("name") == "mole_training":
                mole_steps = s.get("steps", 25000)
                break

    mole_config = config["model"]["decoder"]["mole"]

    return TrainingArguments(
        output_dir=args.output_dir,
        max_steps=mole_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        weight_decay=0.05,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=100,
        eval_steps=2500,
        save_steps=5000,
        save_total_limit=3,
        dataloader_num_workers=8,
        seed=args.seed,
        report_to=["wandb", "tensorboard"],
        run_name=f"MoLE_{config['experiment']['id']}_{args.variant}_seed{args.seed}",
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TBI-MLLM MoLE Training")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    parser.add_argument("--variant", type=str, required=True, help="Experiment variant (D4 or F4)")
    parser.add_argument("--phase", type=str, default="3c_mole", help="Phase identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--init_merged", type=str, required=True, help="Path to DARE+TIES merged model")
    parser.add_argument("--init_ko", type=str, required=True, help="Path to Korean fine-tuned model")
    parser.add_argument("--init_en", type=str, required=True, help="Path to English/base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config
    with open(args.config) as f:
        full_config = yaml.safe_load(f)

    # Get variant config
    from train import get_variant_config
    config = get_variant_config(full_config, args.variant)

    logger.info(f"{'=' * 60}")
    logger.info(f"MoLE Training: {args.variant} (seed={args.seed})")
    logger.info(f"  Merged model: {args.init_merged}")
    logger.info(f"  KO model: {args.init_ko}")
    logger.info(f"  EN model: {args.init_en}")
    logger.info(f"{'=' * 60}")

    # Setup model with MoLE
    model, tokenizer = setup_training(config, args)

    # Build dataset (balanced KO+EN for router training)
    from train import build_dataset, InternVLDataCollator

    # Override data config to use mole_balanced mixing
    phase3_data = config["training"]["phases"]["phase3"].get("data", {})
    mole_data = phase3_data.get("mole_balanced", phase3_data.get("standard", {}))
    config["training"]["phases"]["phase3"]["data"] = {
        "sources": mole_data.get("sources", []),
        "mixing_weights": mole_data.get("mixing_weights", []),
        "max_length": phase3_data.get("max_length", 2048),
    }

    processor = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            args.init_merged, trust_remote_code=True,
        )
    except Exception:
        pass

    dataset = build_dataset(config, 3, tokenizer, processor)
    collator = InternVLDataCollator(tokenizer, processor, max_length=2048)

    # Training arguments
    training_args = build_mole_training_args(config, args)

    # Create MoLE trainer
    mole_config = config["model"]["decoder"]["mole"]
    trainer = MoLETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        load_balance_weight=mole_config.get("load_balancing_loss_weight", 0.01),
        z_loss_weight=mole_config.get("router_z_loss_weight", 0.001),
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(os.path.join(args.output_dir, "last"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "last"))

    # Save MoLE metadata
    metadata = {
        "variant": args.variant,
        "seed": args.seed,
        "init_merged": args.init_merged,
        "init_ko": args.init_ko,
        "init_en": args.init_en,
        "mole_config": mole_config,
    }
    with open(os.path.join(args.output_dir, "mole_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"MoLE training complete! Model saved to {args.output_dir}/last")


if __name__ == "__main__":
    main()
