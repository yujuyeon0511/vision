"""
TBI-MLLM Training Script
=========================
Main training script for all experiment variants (A0-F4).
Supports 3-phase progressive training with DeepSpeed multi-node.

Usage:
    # Single variant, single seed
    deepspeed --hostfile experiments/configs/multi-node/hostfile \
        --master_addr 192.168.0.28 --master_port 29500 \
        experiments/scripts/train.py \
        --config experiments/configs/EXP-20260211-001-config.yaml \
        --variant A1 --phase 1 --seed 42 \
        --output_dir results/EXP-20260211-001/A1/seed_42/phase1 \
        --deepspeed experiments/configs/multi-node/ds_config_zero2_multinode.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tbi-mllm-train")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_variant_config(full_config: dict, variant: str) -> dict:
    """Merge global config with variant-specific overrides."""
    base = {
        "experiment": full_config["experiment"],
        "model": full_config["model"].copy(),
        "training": full_config["training"].copy(),
        "data": full_config["data"],
        "hardware": full_config["hardware"],
    }
    variant_overrides = full_config["experiments"].get(variant)
    if variant_overrides is None:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(full_config['experiments'].keys())}")

    # Apply model overrides
    if "model" in variant_overrides:
        for component, settings in variant_overrides["model"].items():
            if component in base["model"]:
                if isinstance(settings, dict):
                    for k, v in settings.items():
                        if isinstance(v, dict) and k in base["model"][component]:
                            base["model"][component][k].update(v)
                        else:
                            base["model"][component][k] = v

    # Apply training overrides
    if "training" in variant_overrides:
        for phase_name, phase_cfg in variant_overrides["training"].get("phases", {}).items():
            if phase_name in base["training"]["phases"]:
                base["training"]["phases"][phase_name].update(phase_cfg)

    base["variant"] = variant_overrides
    base["variant_name"] = variant
    return base


def get_phase_config(config: dict, phase: int) -> dict:
    """Get training config for a specific phase (1, 2, or 3)."""
    phase_key = f"phase{phase}"
    phase_cfg = config["training"]["phases"].get(phase_key)
    if phase_cfg is None:
        raise ValueError(f"Phase {phase} not found in config")
    if not phase_cfg.get("enabled", True):
        raise ValueError(f"Phase {phase} is disabled for variant {config.get('variant_name', '?')}")
    return phase_cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VLMDataset(Dataset):
    """Vision-Language dataset loader.

    Loads pre-processed data from disk. Each sample is a JSON line with:
    - image_path: path to image file
    - conversations: list of {from, value} dicts
    """

    def __init__(self, data_dir: str, tokenizer, processor, max_length: int = 2048):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        # Load from jsonl files in directory
        for jsonl_file in sorted(self.data_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

        # Fallback: load from json file
        if not self.samples:
            json_file = self.data_dir / "data.json"
            if json_file.exists():
                with open(json_file) as f:
                    self.samples = json.load(f)

        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Delegate actual processing to the model's processor
        # InternVL uses its own chat template and image processing
        return sample


def build_dataset(config: dict, phase: int, tokenizer, processor) -> Dataset:
    """Build training dataset for a given phase."""
    phase_cfg = get_phase_config(config, phase)
    data_cfg = phase_cfg.get("data", {})
    max_length = data_cfg.get("max_length", 2048)

    sources = data_cfg.get("sources", [])
    mixing_weights = data_cfg.get("mixing_weights", [1.0 / len(sources)] * len(sources))

    datasets = []
    data_paths = config["data"]["paths"]

    for source in sources:
        source_path = data_paths.get(source)
        if source_path is None:
            logger.warning(f"Data source '{source}' not found in paths config, skipping")
            continue
        if not Path(source_path).exists():
            logger.warning(f"Data path does not exist: {source_path}, skipping")
            continue
        ds = VLMDataset(source_path, tokenizer, processor, max_length)
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        raise RuntimeError("No valid datasets found. Check data paths and download datasets first.")

    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_base_model(config: dict):
    """Load InternVL2.5 base model with trust_remote_code."""
    model_name = config["experiment"]["base_model"]
    cache_dir = config["experiment"].get("cache_dir")

    logger.info(f"Loading base model: {model_name}")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    return model, tokenizer


def configure_trainable_params(model, config: dict, phase: int):
    """Freeze/unfreeze parameters based on phase and variant config."""
    phase_cfg = get_phase_config(config, phase)
    trainable = phase_cfg.get("trainable", {})

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze based on config
    unfrozen_count = 0
    total_count = 0

    if trainable.get("vision_encoder", False):
        vision_module = getattr(model, "vision_model", None) or getattr(model, "visual", None)
        if vision_module is not None:
            # Check if we should use LoRA for vision encoder
            ve_lora = config["model"]["vision_encoder"].get("lora", {})
            if ve_lora.get("enabled", False):
                layers_to_train = ve_lora.get("layers_to_train", [])
                for name, param in vision_module.named_parameters():
                    layer_idx = _extract_layer_idx(name)
                    if layer_idx is not None and layer_idx in layers_to_train:
                        param.requires_grad = True
                        unfrozen_count += 1
                    total_count += 1
            else:
                for param in vision_module.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
                    total_count += 1

    if trainable.get("projector", False):
        proj_module = (
            getattr(model, "mlp1", None)
            or getattr(model, "projector", None)
            or getattr(model, "mm_projector", None)
        )
        if proj_module is not None:
            for param in proj_module.parameters():
                param.requires_grad = True
                unfrozen_count += 1
                total_count += 1

    if trainable.get("llm_decoder", False):
        llm_module = (
            getattr(model, "language_model", None)
            or getattr(model, "llm", None)
        )
        if llm_module is not None:
            # Check for LoRA on decoder
            dec_lora = config["model"]["decoder"].get("lora", {})
            if dec_lora.get("enabled", False):
                # Apply LoRA (handled separately in apply_lora)
                pass
            else:
                for param in llm_module.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
                    total_count += 1

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Phase {phase} trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.2f}%)"
    )

    return model


def apply_decoder_lora(model, config: dict):
    """Apply LoRA to the decoder (LLM) if enabled."""
    dec_lora = config["model"]["decoder"].get("lora", {})
    if not dec_lora.get("enabled", False):
        return model

    llm_module = getattr(model, "language_model", None) or getattr(model, "llm", None)
    if llm_module is None:
        logger.warning("Could not find LLM module for LoRA, skipping")
        return model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=dec_lora.get("rank", 64),
        lora_alpha=dec_lora.get("alpha", 16),
        lora_dropout=dec_lora.get("dropout", 0.05),
        target_modules=dec_lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        modules_to_save=dec_lora.get("modules_to_save", []),
    )

    llm_module = get_peft_model(llm_module, lora_config)
    llm_module.print_trainable_parameters()

    # Replace back into the parent model
    if hasattr(model, "language_model"):
        model.language_model = llm_module
    elif hasattr(model, "llm"):
        model.llm = llm_module

    return model


def _extract_layer_idx(param_name: str):
    """Extract layer index from parameter name like 'encoder.layers.18.attn.qkv.weight'."""
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Data collator for InternVL
# ---------------------------------------------------------------------------

class InternVLDataCollator:
    """Collate function for InternVL-style multi-modal data.

    Handles variable-length conversations and dynamic image tiling.
    """

    def __init__(self, tokenizer, processor, max_length=2048):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        # InternVL's preprocessing handles image loading and tokenization
        # This is a simplified collator — actual implementation depends on
        # InternVL's data pipeline which uses dynamic_preprocess + build_transform
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []

        for sample in batch:
            # Process through InternVL's chat template
            # The actual processing depends on the model's specific format
            processed = self._process_sample(sample)
            if processed is not None:
                input_ids_list.append(processed["input_ids"])
                attention_mask_list.append(processed["attention_mask"])
                labels_list.append(processed["labels"])
                if "pixel_values" in processed:
                    pixel_values_list.append(processed["pixel_values"])

        if not input_ids_list:
            return None

        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
        if pixel_values_list:
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        return result

    def _process_sample(self, sample):
        """Process a single VLM sample into model inputs."""
        from PIL import Image

        try:
            # Load image
            image_path = sample.get("image", sample.get("image_path", ""))
            image = None
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")

            # Build conversation text
            conversations = sample.get("conversations", [])
            text_parts = []
            for turn in conversations:
                role = turn.get("from", turn.get("role", ""))
                content = turn.get("value", turn.get("content", ""))
                if role in ("human", "user"):
                    if image is not None and "<image>" not in content:
                        content = "<image>\n" + content
                    text_parts.append(f"<|user|>\n{content}")
                elif role in ("gpt", "assistant"):
                    text_parts.append(f"<|assistant|>\n{content}")

            full_text = "\n".join(text_parts)

            # Tokenize
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            # Labels: mask user tokens (set to -100)
            labels = input_ids.clone()
            # Simple heuristic: mask everything before the last assistant response
            assistant_token = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)
            if len(assistant_token) > 0:
                for i in range(len(labels) - len(assistant_token)):
                    if labels[i:i + len(assistant_token)].tolist() == assistant_token:
                        labels[:i + len(assistant_token)] = -100
                        break

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            # Process image through model's vision processor
            if image is not None and self.processor is not None:
                pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
                result["pixel_values"] = pixel_values

            return result

        except Exception as e:
            logger.warning(f"Failed to process sample: {e}")
            return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_args(config: dict, phase: int, args) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config."""
    phase_cfg = get_phase_config(config, phase)
    batch_cfg = phase_cfg.get("batch_size", {})
    opt_cfg = phase_cfg.get("optimizer", {})
    sched_cfg = phase_cfg.get("scheduler", {})
    precision_cfg = phase_cfg.get("precision", {})

    # Determine learning rate (may be dict for multi-component)
    lr = opt_cfg.get("lr", 1e-5)
    if isinstance(lr, dict):
        # Use the highest LR as the base, apply per-group scaling in optimizer
        lr = max(lr.values())

    output_dir = args.output_dir or os.path.join(
        config["experiment"]["output_dir"],
        config["variant_name"],
        f"seed_{args.seed}",
        f"phase{phase}",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # we use max_steps instead
        max_steps=phase_cfg.get("max_steps", 10000),
        per_device_train_batch_size=batch_cfg.get("per_device", 2),
        gradient_accumulation_steps=batch_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=lr,
        weight_decay=opt_cfg.get("weight_decay", 0.05),
        adam_beta1=opt_cfg.get("betas", [0.9, 0.95])[0],
        adam_beta2=opt_cfg.get("betas", [0.9, 0.95])[1],
        adam_epsilon=opt_cfg.get("eps", 1e-8),
        max_grad_norm=config["training"]["regularization"].get("gradient_clip_norm", 1.0),
        lr_scheduler_type=sched_cfg.get("type", "cosine"),
        warmup_steps=sched_cfg.get("warmup_steps", 500),
        bf16=precision_cfg.get("mixed_precision", "bf16") == "bf16",
        fp16=precision_cfg.get("mixed_precision", "bf16") == "fp16",
        gradient_checkpointing=precision_cfg.get("gradient_checkpointing", False),
        logging_steps=phase_cfg.get("logging_steps", 100),
        eval_steps=phase_cfg.get("eval_steps", 2500),
        save_steps=phase_cfg.get("save_steps", 5000),
        save_total_limit=config.get("logging", {}).get("checkpoints", {}).get("save_total_limit", 5),
        dataloader_num_workers=config["data"]["dataloader"].get("num_workers", 8),
        dataloader_pin_memory=config["data"]["dataloader"].get("pin_memory", True),
        seed=args.seed,
        data_seed=args.seed,
        report_to=_get_report_to(config),
        run_name=f"{config['experiment']['id']}_{config['variant_name']}_phase{phase}_seed{args.seed}",
        deepspeed=args.deepspeed if args.deepspeed else None,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    return training_args


def _get_report_to(config: dict) -> list:
    """Determine which logging backends to use."""
    report_to = []
    logging_cfg = config.get("logging", {})
    if logging_cfg.get("wandb", {}).get("enabled", False):
        report_to.append("wandb")
    if logging_cfg.get("tensorboard", {}).get("enabled", False):
        report_to.append("tensorboard")
    return report_to if report_to else ["none"]


def train_phase(config: dict, phase: int, args):
    """Run a single training phase."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting Phase {phase} for variant {config['variant_name']} (seed={args.seed})")
    logger.info(f"{'=' * 60}")

    # Load model
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        model = AutoModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.resume_from,
            trust_remote_code=True,
        )
    else:
        model, tokenizer = load_base_model(config)

    # Apply LoRA if enabled
    model = apply_decoder_lora(model, config)

    # Configure trainable parameters for this phase
    model = configure_trainable_params(model, config, phase)

    # Get processor for image handling
    processor = getattr(model, "processor", None)
    if processor is None:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                config["experiment"]["base_model"],
                trust_remote_code=True,
                cache_dir=config["experiment"].get("cache_dir"),
            )
        except Exception:
            logger.warning("Could not load image processor, images will not be processed")

    # Build dataset
    dataset = build_dataset(config, phase, tokenizer, processor)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Build training arguments
    training_args = build_training_args(config, phase, args)

    # Data collator
    collator = InternVLDataCollator(tokenizer, processor, max_length=2048)

    # Configure wandb
    if "wandb" in training_args.report_to:
        import wandb
        wandb_cfg = config.get("logging", {}).get("wandb", {})
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "tbi-mllm")
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Train
    if args.resume_from and os.path.isdir(args.resume_from):
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "last"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, "last"))
    logger.info(f"Phase {phase} complete. Model saved to {training_args.output_dir}/last")

    return os.path.join(training_args.output_dir, "last")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TBI-MLLM Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--variant", type=str, required=True, help="Experiment variant (A0, A1, B1, ..., F4)")
    parser.add_argument("--phase", type=int, default=None, help="Training phase (1, 2, or 3). If not set, runs all enabled phases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config JSON path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    full_config = load_config(args.config)
    config = get_variant_config(full_config, args.variant)

    logger.info(f"Experiment: {config['experiment']['id']}")
    logger.info(f"Variant: {args.variant} ({config['variant'].get('name', '')})")
    logger.info(f"Seed: {args.seed}")

    # Determine which phases to run
    if args.phase is not None:
        phases = [args.phase]
    else:
        phases = []
        for p in [1, 2, 3]:
            phase_key = f"phase{p}"
            phase_cfg = config["training"]["phases"].get(phase_key, {})
            if phase_cfg.get("enabled", False):
                phases.append(p)

    if not phases:
        logger.info(f"No training phases enabled for variant {args.variant}. Nothing to do.")
        if args.variant == "A0":
            logger.info("A0 is zero-shot evaluation only. Use evaluate.py instead.")
        return

    logger.info(f"Phases to run: {phases}")

    # Run phases sequentially
    checkpoint_path = args.resume_from
    for phase in phases:
        if checkpoint_path and phase > 1:
            args.resume_from = checkpoint_path
        checkpoint_path = train_phase(config, phase, args)

    logger.info("All training phases complete!")
    logger.info(f"Final model: {checkpoint_path}")


if __name__ == "__main__":
    main()
