"""Score generated code samples by executing in environment.

Usage: python scripts/02_score.py --iter 0 --config <config.yaml>
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runner import score_sample
from src.dataset import load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    iter_num = args.iter
    score_cfg = cfg["scoring"]

    rollouts_base = Path(cfg["paths"]["rollouts_dir"])
    if args.run_name:
        rollouts_base = rollouts_base / args.run_name
    rollout_dir = rollouts_base / f"iter_{iter_num:03d}"
    samples = load_jsonl(str(rollout_dir / "samples.jsonl"))

    # Load tasks to get env_code and scenarios
    tasks = load_jsonl(cfg["paths"]["tasks_train"])
    task_map = {t["task_id"]: t for t in tasks}

    all_scores = []
    for sample in samples:
        task_id = sample["task_id"]
        sample_id = sample["sample_id"]
        code = sample.get("code", "")

        if not sample.get("generation_success", False) or not code:
            score = {
                "iter": iter_num,
                "task_id": task_id,
                "sample_id": sample_id,
                "reward_total": 0.0,
                "reward_per_scenario": {},
                "ran_successfully": False,
                "errors": ["Generation failed"],
            }
        else:
            task = task_map[task_id]
            score_result = score_sample(
                code,
                task,
                timeout=score_cfg["timeout_seconds"],
                max_steps=score_cfg["max_steps"],
            )
            score = {
                "iter": iter_num,
                "task_id": task_id,
                "sample_id": sample_id,
                **score_result,
            }

        all_scores.append(score)
        logger.info(
            f"  Task {task_id} sample {sample_id}: "
            f"reward={score['reward_total']:.1f} "
            f"success={score.get('ran_successfully', False)}"
        )

    out_path = str(rollout_dir / "scores.jsonl")
    save_jsonl(all_scores, out_path)

    # Summary
    rewards = [s["reward_total"] for s in all_scores]
    n_success = sum(1 for s in all_scores if s.get("ran_successfully", False))
    n_total = len(all_scores)
    mean_reward = sum(rewards) / n_total if n_total else 0.0
    max_reward = max(rewards) if rewards else 0.0

    logger.info(
        f"Scored {n_total} samples: "
        f"{n_success} ran successfully, "
        f"mean reward={mean_reward:.1f}, "
        f"max reward={max_reward:.1f}"
    )

    # Append to progress log
    from src.progress import append_train_progress
    append_train_progress(
        iter_num=iter_num,
        n_samples=n_total,
        n_success=n_success,
        mean_reward=mean_reward,
        max_reward=max_reward,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
