"""Orchestrate the full DPO + LoRA training loop.

Usage: uv run python scripts/09_run_loop.py --config configs/loop.yaml
       uv run python scripts/09_run_loop.py --config configs/loop.yaml --run_name my_experiment
"""
import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def run_step(cmd: str, description: str):
    """Run a subprocess and check for errors."""
    logger.info(f"\n{'=' * 60}")
    logger.info(description)
    logger.info(f"{'=' * 60}")
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        logger.error(f"FAILED: {description}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/loop.yaml")
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument(
        "--run_name", default=None,
        help="Run name (default: auto-generated timestamp like run_20260222_141700)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    K = cfg["loop"]["iterations"]
    prefix = cfg["loop"]["ollama_model_prefix"]
    quant = cfg["gguf"]["quantization"]

    # Generate run name if not provided
    run_name = args.run_name
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_flag = f"--run_name {run_name}"

    logger.info(f"Starting training loop: {K} iterations from iter {args.start_iter}")
    logger.info(f"Run name: {run_name}")

    for k in range(args.start_iter, K):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# ITERATION {k}")
        logger.info(f"{'#' * 60}")

        # Determine Ollama model for this iteration
        if k == 0:
            ollama_model = cfg["base_model"]["ollama"]
        else:
            ollama_model = f"{prefix}:iter{k:03d}"

        # Step 1: Generate samples
        run_step(
            f"uv run python scripts/01_generate.py --iter {k} --config {args.config} "
            f"--ollama_model {ollama_model} {run_flag}",
            f"Iteration {k}: Generating code samples with {ollama_model}",
        )

        # Step 2: Score samples
        run_step(
            f"uv run python scripts/02_score.py --iter {k} --config {args.config} {run_flag}",
            f"Iteration {k}: Scoring samples in environment",
        )

        # Step 3: Build DPO pairs
        run_step(
            f"uv run python scripts/03_build_dpo.py --iter {k} --config {args.config} {run_flag}",
            f"Iteration {k}: Building DPO preference pairs",
        )

        # Step 4: Train DPO LoRA
        run_step(
            f"uv run python scripts/04_train_dpo_lora.py --iter {k} --config {args.config} {run_flag}",
            f"Iteration {k}: Training DPO LoRA adapter",
        )

        # Step 5: Merge LoRA into base
        run_step(
            f"uv run python scripts/05_merge_lora.py --iter {k} --config {args.config} {run_flag}",
            f"Iteration {k}: Merging LoRA into base model",
        )

        # Step 6: Convert to GGUF
        next_iter = k + 1
        run_step(
            f"bash scripts/06_convert_to_gguf.sh {next_iter} {quant} {run_name}",
            f"Iteration {k}: Converting merged model to GGUF",
        )

        # Step 7: Create Ollama model
        run_step(
            f"bash scripts/07_ollama_create.sh {next_iter} {quant} {run_name}",
            f"Iteration {k}: Creating Ollama model {prefix}:iter{next_iter:03d}",
        )

        # Step 8: Evaluate
        run_step(
            f"uv run python scripts/08_eval.py --iter {next_iter} --config {args.config} {run_flag}",
            f"Iteration {k}: Evaluating {prefix}:iter{next_iter:03d}",
        )

        # Step 9: Cleanup — delete temporary merged HF weights and old GGUF
        hf_base = Path(cfg["paths"]["hf_base_dir"]) / run_name
        merged_hf = hf_base / f"iter_{next_iter:03d}"
        if merged_hf.exists():
            logger.info(f"Cleaning up merged HF weights: {merged_hf}")
            shutil.rmtree(merged_hf)

        if k > 0:
            gguf_base = Path(cfg["paths"]["gguf_dir"]) / run_name
            old_gguf = gguf_base / f"iter_{k:03d}"
            if old_gguf.exists():
                logger.info(f"Cleaning up old GGUF: {old_gguf}")
                shutil.rmtree(old_gguf)

        logger.info(f"\nIteration {k} complete. Next model: {prefix}:iter{next_iter:03d}")

        # Print progress so far
        from src.progress import print_progress_table
        print_progress_table(run_name=run_name)

    logger.info(f"\n{'#' * 60}")
    logger.info(f"# TRAINING LOOP COMPLETE ({K} iterations)")
    logger.info(f"{'#' * 60}")

    from src.progress import print_progress_table
    print_progress_table(run_name=run_name)


if __name__ == "__main__":
    main()
