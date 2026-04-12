"""Orchestrate the full training loop (DPO or GRPO).

Usage: uv run python scripts/09_run_loop.py --config configs/loop.yaml
       uv run python scripts/09_run_loop.py --config configs/loop.yaml --run_name my_experiment
"""
import argparse
import csv
import logging
import os
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


def _wandb_enabled(report_to) -> bool:
    """TRL's report_to accepts a string or list; treat any of them as enabling wandb."""
    if isinstance(report_to, str):
        return report_to == "wandb" or report_to == "all"
    if isinstance(report_to, list):
        return "wandb" in report_to or "all" in report_to
    return False


def _read_iter_row(csv_path: str, iter_num: int) -> dict | None:
    """Read the row for `iter_num` from a progress CSV, or None if absent."""
    p = Path(csv_path)
    if not p.exists():
        return None
    with open(p) as f:
        for row in csv.DictReader(f):
            if int(row.get("iter", -1)) == iter_num:
                return row
    return None


def _log_iteration_metrics(wandb_run, run_name: str, k: int):
    """Push the per-iteration train/eval CSV rows to the orchestrator W&B run."""
    if wandb_run is None:
        return
    from src.progress import _get_csv_paths
    train_csv, eval_csv = _get_csv_paths(run_name)
    payload: dict = {}
    train_row = _read_iter_row(train_csv, k)
    if train_row:
        payload.update({
            "train/n_samples": int(train_row["n_samples"]),
            "train/n_success": int(train_row["n_success"]),
            "train/success_rate": float(train_row["success_rate"].rstrip("%")) / 100,
            "train/mean_reward": float(train_row["mean_reward"]),
            "train/max_reward": float(train_row["max_reward"]),
        })
    # eval CSV is appended for iter k+1 (the post-training eval)
    eval_row = _read_iter_row(eval_csv, k + 1)
    if eval_row:
        payload.update({
            "eval/n_samples": int(eval_row["n_samples"]),
            "eval/n_success": int(eval_row["n_success"]),
            "eval/success_rate": float(eval_row["success_rate"].rstrip("%")) / 100,
            "eval/mean_reward": float(eval_row["mean_reward"]),
            "eval/mean_best_reward": float(eval_row["mean_best_reward"]),
        })
    if payload:
        wandb_run.log(payload, step=k)


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

    from src.config import validate_config
    validate_config(cfg)

    K = cfg["loop"]["iterations"]
    gcs_bucket = cfg.get("paths", {}).get("gcs_bucket")

    # Generate run name if not provided
    run_name = args.run_name
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_flag = f"--run_name {run_name}"

    logger.info(f"Starting training loop: {K} iterations from iter {args.start_iter}")
    logger.info(f"Run name: {run_name}")
    if gcs_bucket:
        logger.info(f"GCS sync enabled: gs://{gcs_bucket}/runs/{run_name}")

    if gcs_bucket:
        from src.gcs_sync import subtrees_for_config
        gcs_subtrees = subtrees_for_config(cfg)

    # If resuming, pull any prior artifacts from GCS first so iter k can read
    # the adapters/rollouts from iters 0..k-1.
    if args.start_iter > 0 and gcs_bucket:
        from src.gcs_sync import pull_run
        logger.info(f"Pulling prior run artifacts from GCS for resume at iter {args.start_iter}")
        pull_run(gcs_bucket, run_name, gcs_subtrees)

    # Clear stale progress CSVs when starting fresh (not resuming)
    if args.start_iter == 0:
        from src.progress import _get_csv_paths
        for csv_path in _get_csv_paths(run_name):
            p = Path(csv_path)
            if p.exists():
                p.unlink()
                logger.info(f"Cleared stale progress file: {p}")

    trainer_type = cfg["loop"].get("trainer", "dpo")
    logger.info(f"Trainer: {trainer_type}")

    # Set up W&B for the orchestrator + propagate env vars to training
    # subprocesses so TRL's built-in W&B integration groups every iteration's
    # training run under the same `run_name` group.
    wandb_run = None
    report_to = cfg.get("training", {}).get("report_to", "none")
    if _wandb_enabled(report_to):
        os.environ.setdefault(
            "WANDB_PROJECT",
            cfg.get("training", {}).get("wandb_project", "lora4vr"),
        )
        os.environ["WANDB_RUN_GROUP"] = run_name
        # Pin a stable run id + name for the trainer subprocess so every
        # iteration's GRPOTrainer.train() resumes the same W&B run instead
        # of creating a fresh one. Without this you get one disconnected
        # curve per outer iteration.
        os.environ["WANDB_RUN_ID"] = f"{run_name}-trainer"
        os.environ["WANDB_NAME"] = f"{run_name}-trainer"
        os.environ["WANDB_RESUME"] = "allow"
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                group=run_name,
                name=f"{run_name}-orchestrator",
                job_type="loop",
                config={
                    "trainer": trainer_type,
                    "iterations": K,
                    "start_iter": args.start_iter,
                    **{f"loop/{k_}": v for k_, v in cfg.get("loop", {}).items()},
                    **{f"training/{k_}": v for k_, v in cfg.get("training", {}).items()},
                    **{f"lora/{k_}": v for k_, v in cfg.get("lora", {}).items()},
                },
                resume="allow",
            )
            logger.info(f"W&B orchestrator run: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"Failed to init W&B: {e} — continuing without orchestrator metrics")
            wandb_run = None

    for k in range(args.start_iter, K):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# ITERATION {k}")
        logger.info(f"{'#' * 60}")

        if trainer_type == "grpo":
            # GRPO handles generation + scoring + optimization internally
            run_step(
                f"uv run python scripts/04_train_grpo_lora.py --iter {k} --config {args.config} {run_flag}",
                f"Iteration {k}: GRPO training (generate + score + optimize)",
            )
        else:
            # DPO pipeline: generate -> score -> build pairs -> train
            run_step(
                f"uv run python scripts/01_generate.py --iter {k} --config {args.config} {run_flag}",
                f"Iteration {k}: Generating code samples",
            )

            run_step(
                f"uv run python scripts/02_score.py --iter {k} --config {args.config} {run_flag}",
                f"Iteration {k}: Scoring samples in environment",
            )

            run_step(
                f"uv run python scripts/03_build_dpo.py --iter {k} --config {args.config} {run_flag}",
                f"Iteration {k}: Building DPO preference pairs",
            )

            run_step(
                f"uv run python scripts/04_train_dpo_lora.py --iter {k} --config {args.config} {run_flag}",
                f"Iteration {k}: Training DPO LoRA adapter",
            )

        # Evaluate (loads base + merges adapters 0..k)
        next_iter = k + 1
        run_step(
            f"uv run python scripts/08_eval.py --iter {next_iter} --config {args.config} {run_flag}",
            f"Iteration {k}: Evaluating model with {next_iter} adapters merged",
        )

        logger.info(f"\nIteration {k} complete.")

        # Print progress so far
        from src.progress import print_progress_table
        print_progress_table(run_name=run_name)

        # Log per-iteration train/eval metrics to W&B alongside TRL's
        # per-step metrics from the training subprocess.
        _log_iteration_metrics(wandb_run, run_name, k)

        # Push this iteration's artifacts to GCS. rsync is diff-based so
        # re-uploading every iteration is cheap.
        if gcs_bucket:
            from src.gcs_sync import push_run
            push_run(gcs_bucket, run_name, gcs_subtrees)

    logger.info(f"\n{'#' * 60}")
    logger.info(f"# TRAINING LOOP COMPLETE ({K} iterations)")
    logger.info(f"{'#' * 60}")

    print_progress_table(run_name=run_name)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
