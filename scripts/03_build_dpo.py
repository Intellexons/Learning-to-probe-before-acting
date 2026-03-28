"""Build DPO preference pairs from scored samples.

Usage: python scripts/03_build_dpo.py --iter 0 --config configs/loop.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import (
    load_jsonl,
    save_jsonl,
    merge_samples_and_scores,
    build_dpo_pairs,
)
from src.prompts import load_template, render_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", default="configs/loop.yaml")
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    iter_num = args.iter
    dpo_cfg = cfg["dpo_pairs"]

    rollouts_base = Path(cfg["paths"]["rollouts_dir"])
    if args.run_name:
        rollouts_base = rollouts_base / args.run_name
    rollout_dir = rollouts_base / f"iter_{iter_num:03d}"
    template = load_template("configs/prompt_template.txt")

    # Build prompt map for train tasks
    train_tasks = load_jsonl(cfg["paths"]["tasks_train"])
    prompt_map = {}
    for task in train_tasks:
        prompt_map[task["task_id"]] = render_prompt(template, task)

    # Merge samples and scores
    merged = merge_samples_and_scores(
        str(rollout_dir / "samples.jsonl"),
        str(rollout_dir / "scores.jsonl"),
    )

    # Build DPO pairs
    pairs = build_dpo_pairs(
        merged,
        prompt_map=prompt_map,
        min_chosen_reward=dpo_cfg["min_chosen_reward"],
        pairs_per_task=dpo_cfg["pairs_per_task"],
    )

    # Save train pairs
    dpo_dir = Path(cfg["paths"]["dpo_dir"])
    if args.run_name:
        dpo_dir = dpo_dir / args.run_name
    dpo_dir.mkdir(parents=True, exist_ok=True)
    train_path = str(dpo_dir / f"iter_{iter_num:03d}_train.jsonl")
    save_jsonl(pairs, train_path)
    logger.info(f"Saved {len(pairs)} DPO train pairs to {train_path}")

    # Build eval pairs (if eval scores exist)
    eval_rollout_dir = rollout_dir  # eval uses same iter dir
    eval_samples_path = str(eval_rollout_dir / "eval_samples.jsonl")
    eval_scores_path = str(eval_rollout_dir / "eval_scores.jsonl")

    if Path(eval_samples_path).exists() and Path(eval_scores_path).exists():
        eval_tasks = load_jsonl(cfg["paths"]["tasks_eval"])
        eval_prompt_map = {}
        for task in eval_tasks:
            eval_prompt_map[task["task_id"]] = render_prompt(template, task)

        eval_merged = merge_samples_and_scores(eval_samples_path, eval_scores_path)
        eval_pairs = build_dpo_pairs(
            eval_merged,
            prompt_map=eval_prompt_map,
            min_chosen_reward=dpo_cfg["min_chosen_reward"],
            pairs_per_task=dpo_cfg["pairs_per_task"],
        )
        eval_path = str(dpo_dir / f"iter_{iter_num:03d}_eval.jsonl")
        save_jsonl(eval_pairs, eval_path)
        logger.info(f"Saved {len(eval_pairs)} DPO eval pairs to {eval_path}")


if __name__ == "__main__":
    main()
