"""
TBI-MLLM Evaluation Script
============================
Evaluates trained models on all benchmarks with statistical reporting.

Usage:
    python experiments/scripts/evaluate.py \
        --model results/EXP-20260211-001/A1/seed_42/phase3/last \
        --benchmarks mmbench_en mmbench_ko chartqa docvqa textvqa k_dtcbench \
        --output results/tables/EXP-20260211-001-A1-seed42-results.csv \
        --num_gpus 2

    # Full evaluation with all benchmarks
    python experiments/scripts/evaluate.py \
        --model results/EXP-20260211-001/F4/seed_42/phase3c/last \
        --config experiments/configs/EXP-20260211-001-config.yaml \
        --all_benchmarks \
        --output results/tables/EXP-20260211-001-F4-seed42-results.csv

    # Aggregate over seeds
    python experiments/scripts/evaluate.py \
        --aggregate \
        --result_files results/tables/EXP-20260211-001-F4-seed*-results.csv \
        --output results/tables/EXP-20260211-001-F4-aggregated.csv
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tbi-mllm-eval")


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    # General VQA
    "mmbench_en": {"metric": "accuracy", "language": "en", "priority": "high"},
    "mmbench_ko": {"metric": "accuracy", "language": "ko", "priority": "high"},
    "mmstar": {"metric": "accuracy", "language": "en", "priority": "high"},
    "seed_bench": {"metric": "accuracy", "language": "en", "priority": "medium"},
    # OCR & Text
    "ocr_bench": {"metric": "f1", "language": "en", "priority": "high"},
    "textvqa": {"metric": "vqa_score", "language": "en", "priority": "high"},
    # Chart
    "chartqa": {"metric": "relaxed_accuracy", "language": "en", "priority": "high"},
    "charxiv": {"metric": "accuracy", "language": "en", "priority": "high"},
    "mchartqa": {"metric": "accuracy", "language": "multi", "priority": "medium"},
    # Table
    "wtq": {"metric": "accuracy", "language": "en", "priority": "high"},
    "tabfact": {"metric": "accuracy", "language": "en", "priority": "medium"},
    # Document
    "docvqa": {"metric": "anls", "language": "en", "priority": "high"},
    "infovqa": {"metric": "anls", "language": "en", "priority": "medium"},
    # Math
    "mathvista": {"metric": "accuracy", "language": "en", "priority": "high"},
    "mathverse": {"metric": "accuracy", "language": "en", "priority": "medium"},
    # Korean
    "k_dtcbench": {"metric": "accuracy", "language": "ko", "priority": "critical"},
    "komm_bench": {"metric": "accuracy", "language": "ko", "priority": "high"},
}

# Aggregated metric groups
AGGREGATED_METRICS = {
    "korean_average": ["k_dtcbench", "mmbench_ko", "komm_bench"],
    "english_average": ["mmbench_en", "textvqa", "ocr_bench", "docvqa", "chartqa", "mathvista"],
    "structured_content_average": ["chartqa", "docvqa", "wtq"],
    "overall_average": None,  # all benchmarks
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, num_gpus: int = 1):
    """Load model for evaluation."""
    logger.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if num_gpus > 1:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()

    model.eval()
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def load_benchmark_data(benchmark_name: str, data_dir: str) -> list:
    """Load benchmark data from directory."""
    bench_dir = Path(data_dir)
    if not bench_dir.exists():
        logger.warning(f"Benchmark directory not found: {bench_dir}")
        return []

    samples = []

    # Try loading from various formats
    for pattern in ["*.jsonl", "test.jsonl", "val.jsonl", "*.json", "test.json", "val.json"]:
        for fpath in sorted(bench_dir.glob(pattern)):
            if fpath.suffix == ".jsonl":
                with open(fpath) as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            else:
                with open(fpath) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    elif isinstance(data, dict) and "data" in data:
                        samples.extend(data["data"])
            if samples:
                break
        if samples:
            break

    logger.info(f"  {benchmark_name}: {len(samples)} samples loaded")
    return samples


def generate_response(model, tokenizer, image_path: str, question: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the model given an image and question."""
    from PIL import Image

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "num_beams": 1,
    }

    try:
        # InternVL2.5 uses model.chat() API
        if hasattr(model, "chat"):
            image = None
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")

            if image is not None:
                response = model.chat(
                    tokenizer,
                    pixel_values=_preprocess_image(model, image),
                    question=question,
                    generation_config=generation_config,
                )
            else:
                response = model.chat(
                    tokenizer,
                    pixel_values=None,
                    question=question,
                    generation_config=generation_config,
                )
            return response

        # Fallback: standard generate
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input portion
        response = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        return response.strip()

    except Exception as e:
        logger.warning(f"Generation failed: {e}")
        return ""


def _preprocess_image(model, image):
    """Preprocess image for InternVL models."""
    if hasattr(model, "build_transform"):
        transform = model.build_transform(is_train=False)
        pixel_values = transform(image).unsqueeze(0).to(model.device).to(torch.bfloat16)
        return pixel_values
    return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_accuracy(predictions: list, references: list) -> float:
    """Exact match accuracy."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if _normalize_answer(pred) == _normalize_answer(ref)
    )
    return correct / len(predictions)


def compute_relaxed_accuracy(predictions: list, references: list, tolerance: float = 0.05) -> float:
    """Relaxed accuracy for ChartQA (numeric tolerance)."""
    if not predictions:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_norm = _normalize_answer(pred)
        ref_norm = _normalize_answer(ref)
        if pred_norm == ref_norm:
            correct += 1
        else:
            try:
                pred_num = float(pred_norm.replace(",", "").replace("%", ""))
                ref_num = float(ref_norm.replace(",", "").replace("%", ""))
                if ref_num != 0 and abs(pred_num - ref_num) / abs(ref_num) <= tolerance:
                    correct += 1
                elif ref_num == 0 and abs(pred_num) <= tolerance:
                    correct += 1
            except (ValueError, ZeroDivisionError):
                pass
    return correct / len(predictions)


def compute_vqa_score(predictions: list, references_list: list) -> float:
    """VQA v2.0 score: min(count/3, 1) where count = matching annotators."""
    if not predictions:
        return 0.0
    total_score = 0.0
    for pred, refs in zip(predictions, references_list):
        pred_norm = _normalize_answer(pred)
        if isinstance(refs, str):
            refs = [refs]
        count = sum(1 for r in refs if _normalize_answer(r) == pred_norm)
        total_score += min(count / 3.0, 1.0)
    return total_score / len(predictions)


def compute_anls(predictions: list, references: list, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity."""
    if not predictions:
        return 0.0
    total = 0.0
    for pred, ref in zip(predictions, references):
        if isinstance(ref, list):
            score = max(_normalized_levenshtein(pred, r) for r in ref)
        else:
            score = _normalized_levenshtein(pred, ref)
        total += score if score >= threshold else 0.0
    return total / len(predictions)


def compute_f1(predictions: list, references: list) -> float:
    """Token-level F1 score."""
    if not predictions:
        return 0.0
    total_f1 = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = set(_normalize_answer(pred).split())
        ref_tokens = set(_normalize_answer(ref).split())
        if not ref_tokens:
            total_f1 += 1.0 if not pred_tokens else 0.0
            continue
        common = pred_tokens & ref_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        total_f1 += 2 * precision * recall / (precision + recall)
    return total_f1 / len(predictions)


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity."""
    s1 = _normalize_answer(s1)
    s2 = _normalize_answer(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    distance = _levenshtein_distance(s1, s2)
    return 1.0 - distance / max_len


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


METRIC_FUNCTIONS = {
    "accuracy": compute_accuracy,
    "relaxed_accuracy": compute_relaxed_accuracy,
    "vqa_score": compute_vqa_score,
    "anls": compute_anls,
    "f1": compute_f1,
}


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_benchmark(
    model,
    tokenizer,
    benchmark_name: str,
    data_dir: str,
    metric_name: str,
    batch_size: int = 4,
) -> dict:
    """Evaluate model on a single benchmark."""
    logger.info(f"Evaluating: {benchmark_name} (metric: {metric_name})")

    samples = load_benchmark_data(benchmark_name, data_dir)
    if not samples:
        return {"benchmark": benchmark_name, "metric": metric_name, "score": None, "n_samples": 0}

    predictions = []
    references = []

    start_time = time.time()

    for i, sample in enumerate(samples):
        # Extract question and answer
        question = sample.get("question", sample.get("prompt", sample.get("text", "")))
        answer = sample.get("answer", sample.get("label", sample.get("ground_truth", "")))
        image_path = sample.get("image", sample.get("image_path", ""))

        # Make the image path absolute if relative
        if image_path and not os.path.isabs(image_path):
            image_path = os.path.join(data_dir, image_path)

        pred = generate_response(model, tokenizer, image_path, question)
        predictions.append(pred)
        references.append(answer)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            logger.info(f"  {benchmark_name}: {i + 1}/{len(samples)} ({speed:.1f} samples/s)")

    elapsed = time.time() - start_time

    # Compute metric
    metric_fn = METRIC_FUNCTIONS.get(metric_name, compute_accuracy)
    score = metric_fn(predictions, references)

    result = {
        "benchmark": benchmark_name,
        "metric": metric_name,
        "score": round(score, 4),
        "n_samples": len(samples),
        "time_seconds": round(elapsed, 1),
    }

    logger.info(f"  {benchmark_name}: {score:.4f} ({len(samples)} samples, {elapsed:.1f}s)")
    return result


def evaluate_all(model, tokenizer, config: dict, benchmarks: list, batch_size: int = 4) -> list:
    """Run evaluation on multiple benchmarks."""
    results = []
    eval_cfg = config.get("evaluation", {}).get("benchmarks", {})

    for bench_name in benchmarks:
        bench_cfg = eval_cfg.get(bench_name, BENCHMARKS.get(bench_name, {}))
        data_dir = bench_cfg.get("path", "")

        if not data_dir:
            # Infer from default structure
            data_dir = os.path.join("data", "benchmarks", bench_name)

        metric_name = bench_cfg.get("metric", "accuracy")
        result = evaluate_benchmark(model, tokenizer, bench_name, data_dir, metric_name, batch_size)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def compute_aggregated_metrics(results: list) -> dict:
    """Compute aggregated metrics from individual benchmark results."""
    score_map = {r["benchmark"]: r["score"] for r in results if r["score"] is not None}

    aggregated = {}
    for group_name, bench_list in AGGREGATED_METRICS.items():
        if bench_list is None:
            # overall: all benchmarks
            scores = [s for s in score_map.values()]
        else:
            scores = [score_map[b] for b in bench_list if b in score_map]

        if scores:
            aggregated[group_name] = round(np.mean(scores), 4)
        else:
            aggregated[group_name] = None

    return aggregated


def save_results_csv(results: list, aggregated: dict, output_path: str):
    """Save evaluation results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "metric", "score", "n_samples", "time_seconds"])

        for r in results:
            writer.writerow([r["benchmark"], r["metric"], r["score"], r["n_samples"], r.get("time_seconds", "")])

        # Add aggregated metrics
        writer.writerow([])
        writer.writerow(["--- Aggregated ---", "", "", "", ""])
        for group_name, score in aggregated.items():
            writer.writerow([group_name, "average", score, "", ""])

    logger.info(f"Results saved to: {output_path}")


def aggregate_seeds(result_files: list, output_path: str):
    """Aggregate results across multiple seeds. Report mean +/- std."""
    all_results = {}  # benchmark -> list of scores

    for fpath in result_files:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["benchmark"].startswith("---") or not row["score"]:
                    continue
                bench = row["benchmark"]
                score = float(row["score"])
                all_results.setdefault(bench, []).append(score)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "mean", "std", "ci_95_lower", "ci_95_upper", "n_seeds"])

        for bench, scores in sorted(all_results.items()):
            scores = np.array(scores)
            mean = np.mean(scores)
            std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
            n = len(scores)
            # 95% CI using t-distribution
            from scipy import stats
            if n > 1:
                t_val = stats.t.ppf(0.975, df=n - 1)
                margin = t_val * std / np.sqrt(n)
            else:
                margin = 0.0
            writer.writerow([
                bench,
                f"{mean:.4f}",
                f"{std:.4f}",
                f"{mean - margin:.4f}",
                f"{mean + margin:.4f}",
                n,
            ])

    logger.info(f"Aggregated results saved to: {output_path}")


def print_results_table(results: list, aggregated: dict):
    """Print results as a formatted table."""
    print("\n" + "=" * 70)
    print(f"{'Benchmark':<25} {'Metric':<20} {'Score':>10} {'Samples':>10}")
    print("-" * 70)

    for r in results:
        score_str = f"{r['score']:.4f}" if r["score"] is not None else "N/A"
        print(f"{r['benchmark']:<25} {r['metric']:<20} {score_str:>10} {r['n_samples']:>10}")

    print("-" * 70)
    for group_name, score in aggregated.items():
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"{group_name:<25} {'average':<20} {score_str:>10}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TBI-MLLM Evaluation Script")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--config", type=str, default=None, help="Experiment config YAML for benchmark paths")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Benchmarks to evaluate")
    parser.add_argument("--all_benchmarks", action="store_true", help="Evaluate on all benchmarks")
    parser.add_argument("--high_priority", action="store_true", help="Evaluate only high/critical priority benchmarks")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for inference")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    # Aggregation mode
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results across seed files")
    parser.add_argument("--result_files", nargs="+", default=None, help="CSV files to aggregate (supports glob)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Aggregation mode
    if args.aggregate:
        files = []
        for pattern in (args.result_files or []):
            files.extend(glob.glob(pattern))
        if not files:
            logger.error("No result files found for aggregation")
            sys.exit(1)
        logger.info(f"Aggregating {len(files)} result files")
        aggregate_seeds(files, args.output)
        return

    # Evaluation mode
    if not args.model:
        logger.error("--model is required for evaluation")
        sys.exit(1)

    # Load config for benchmark paths
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Determine benchmarks
    if args.benchmarks:
        benchmarks = args.benchmarks
    elif args.all_benchmarks:
        benchmarks = list(BENCHMARKS.keys())
    elif args.high_priority:
        benchmarks = [
            b for b, info in BENCHMARKS.items()
            if info["priority"] in ("critical", "high")
        ]
    else:
        # Default: high priority only
        benchmarks = [
            b for b, info in BENCHMARKS.items()
            if info["priority"] in ("critical", "high")
        ]

    logger.info(f"Benchmarks to evaluate: {benchmarks}")

    # Load model
    model, tokenizer = load_model(args.model, args.num_gpus)

    # Run evaluation
    results = evaluate_all(model, tokenizer, config, benchmarks, args.batch_size)

    # Compute aggregated metrics
    aggregated = compute_aggregated_metrics(results)

    # Print and save
    print_results_table(results, aggregated)
    save_results_csv(results, aggregated, args.output)

    # Also save as JSON for programmatic access
    json_output = args.output.replace(".csv", ".json")
    with open(json_output, "w") as f:
        json.dump({
            "results": results,
            "aggregated": aggregated,
            "model": args.model,
        }, f, indent=2)
    logger.info(f"JSON results saved to: {json_output}")


if __name__ == "__main__":
    main()
