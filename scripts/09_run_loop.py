"""Orchestrate the full training loop (DPO or GRPO).

Usage: uv run python scripts/09_run_loop.py --config configs/loop_grpo_branched_slosh.yaml
       uv run python scripts/09_run_loop.py --config configs/loop_grpo_branched_slosh.yaml --run_name my_experiment
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


def _log_iteration_metrics(wandb_run, run_name: str, k: int, step: int):
    """Push the per-iteration train/eval CSV rows to the orchestrator W&B run.

    `step` is the wandb step axis position to log at; it must be monotonically
    increasing across calls because wandb silently drops out-of-order steps.
    """
    if wandb_run is None:
        return
    from src.progress import _get_csv_paths
    train_csv, eval_csv = _get_csv_paths(run_name)
    payload: dict = {"train/iter": k}
    train_row = _read_iter_row(train_csv, k)
    if train_row:
        payload.update({
            "train/n_samples": int(train_row["n_samples"]),
            "train/n_success": int(train_row["n_success"]),
            "train/success_rate": float(train_row["success_rate"].rstrip("%")) / 100,
            "train/iter_mean_reward": float(train_row["mean_reward"]),
            "train/iter_max_reward": float(train_row["max_reward"]),
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
        # `n_settled` / `settle_rate` are new in PR #69; old rows from
        # prior runs won't have them. Skip silently rather than failing.
        if eval_row.get("n_settled") not in (None, ""):
            payload["eval/n_settled"] = int(eval_row["n_settled"])
        if eval_row.get("settle_rate") not in (None, ""):
            payload["eval/settle_rate"] = (
                float(eval_row["settle_rate"].rstrip("%")) / 100
            )
    if payload:
        wandb_run.log(payload, step=step)


def _forward_trainer_log_history(wandb_run, run_name: str, k: int, step_offset: int) -> int:
    """Read the trainer subprocess's dumped log_history and forward each entry
    to the orchestrator's W&B run, offsetting global_step so the curve is
    monotonic across outer iterations. Returns the new offset to use next time.
    """
    import json
    log_path = Path("data/progress") / run_name / f"trainer_log_iter_{k:03d}.json"
    if not log_path.exists():
        return step_offset
    try:
        with open(log_path) as f:
            entries = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read trainer log {log_path}: {e}")
        return step_offset

    max_local_step = 0
    for entry in entries:
        local_step = int(entry.get("step", 0) or 0)
        max_local_step = max(max_local_step, local_step)
        # Strip the local step + epoch keys; everything else is a metric.
        payload = {
            f"train/{key}": val
            for key, val in entry.items()
            if key not in ("step", "epoch") and isinstance(val, (int, float))
        }
        if not payload:
            continue
        payload["train/iter"] = k
        wandb_run.log(payload, step=step_offset + local_step)
    return step_offset + max_local_step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                id=f"{run_name}-orchestrator",
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

    # Running offset for trainer log_history forwarding: each iteration's
    # log_history starts at global_step=0, so we add this offset to get a
    # monotonic step axis across all outer iterations in the orchestrator run.
    trainer_step_offset = 0

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
        elif trainer_type == "grpo_branched":
            # Branched-rollout multi-turn GRPO (N=2, VinePPO-style). The
            # subprocess generates the G_1 × G_2 tree, computes per-turn
            # advantages, and dumps trainer_log_iter_*.json the same way
            # 04_train_grpo_lora.py does, so the rest of this loop is unchanged.
            run_step(
                f"uv run python scripts/05_train_grpo_branched.py --iter {k} --config {args.config} {run_flag}",
                f"Iteration {k}: Branched GRPO training (G1×G2 tree + per-turn advantages)",
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

        # Forward the trainer subprocess's per-step log_history to the
        # orchestrator's W&B run with a monotonic step offset, so the
        # loss / reward / KL curves are continuous across outer iterations.
        if wandb_run is not None and trainer_type in ("grpo", "grpo_branched"):
            trainer_step_offset = _forward_trainer_log_history(
                wandb_run, run_name, k, trainer_step_offset
            )

        # Log per-iteration train/eval CSV metrics AFTER forwarding the
        # trainer log, so they land at a step strictly past every per-step
        # metric we just logged. wandb silently drops out-of-order steps,
        # which is why iter 1+ CSV metrics were missing before.
        _log_iteration_metrics(wandb_run, run_name, k, step=trainer_step_offset)
        trainer_step_offset += 1

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
