"""Orchestrate the full DPO + LoRA training loop.

Usage: uv run python scripts/09_run_loop.py --config configs/loop.yaml
       uv run python scripts/09_run_loop.py --config configs/loop.yaml --run_name my_experiment
"""
import argparse
import logging
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

    # Generate run name if not provided
    run_name = args.run_name
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_flag = f"--run_name {run_name}"

    logger.info(f"Starting training loop: {K} iterations from iter {args.start_iter}")
    logger.info(f"Run name: {run_name}")

    # Clear stale progress CSVs when starting fresh (not resuming)
    if args.start_iter == 0:
        from src.progress import _get_csv_paths
        for csv_path in _get_csv_paths(run_name):
            p = Path(csv_path)
            if p.exists():
                p.unlink()
                logger.info(f"Cleared stale progress file: {p}")

    for k in range(args.start_iter, K):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# ITERATION {k}")
        logger.info(f"{'#' * 60}")

        # Step 1: Generate samples (loads base + merges adapters 0..k-1)
        run_step(
            f"uv run python scripts/01_generate.py --iter {k} --config {args.config} {run_flag}",
            f"Iteration {k}: Generating code samples",
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

        # Step 4: Train DPO LoRA (loads base + merges adapters 0..k-1, trains adapter k)
        run_step(
            f"uv run python scripts/04_train_dpo_lora.py --iter {k} --config {args.config} {run_flag}",
            f"Iteration {k}: Training DPO LoRA adapter",
        )

        # Step 5: Evaluate (loads base + merges adapters 0..k)
        next_iter = k + 1
        run_step(
            f"uv run python scripts/08_eval.py --iter {next_iter} --config {args.config} {run_flag}",
            f"Iteration {k}: Evaluating model with {next_iter} adapters merged",
        )

        logger.info(f"\nIteration {k} complete.")

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
