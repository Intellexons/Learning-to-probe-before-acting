"""Evaluate current model on held-out eval tasks.

Usage: python scripts/08_eval.py --iter 1 --config configs/loop.yaml
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ollama_client import OllamaClient
from src.prompts import load_template, render_prompt
from src.runner import score_sample
from src.dataset import load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_EVAL_SAMPLES = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", default="configs/loop.yaml")
    parser.add_argument("--ollama_model", default=None)
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    iter_num = args.iter
    gen_cfg = cfg["generation"]
    score_cfg = cfg["scoring"]

    # Determine Ollama model
    if args.ollama_model:
        ollama_model = args.ollama_model
    elif iter_num == 0:
        ollama_model = cfg["base_model"]["ollama"]
    else:
        prefix = cfg["loop"]["ollama_model_prefix"]
        ollama_model = f"{prefix}:iter{iter_num:03d}"

    client = OllamaClient(
        base_url=gen_cfg["ollama_base_url"],
        schema_path="configs/schema.json",
    )

    if not client.check_health():
        logger.error("Ollama is not running.")
        sys.exit(1)

    template = load_template("configs/prompt_template.txt")
    eval_tasks = load_jsonl(cfg["paths"]["tasks_eval"])

    results = {}
    all_samples = []
    all_scores = []

    for task in eval_tasks:
        task_id = task["task_id"]
        prompt = render_prompt(template, task)
        logger.info(f"Eval task {task_id}: generating {N_EVAL_SAMPLES} samples")

        samples = client.generate_batch(
            model=ollama_model,
            prompt=prompt,
            n_samples=N_EVAL_SAMPLES,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_tokens"],
        )

        best_reward = 0.0
        task_rewards = []

        for i, sample in enumerate(samples):
            all_samples.append({
                "iter": iter_num,
                "task_id": task_id,
                "sample_id": i,
                "prompt": prompt,
                "code": sample["code"],
                "generation_success": sample["success"],
                "generation_error": sample["error"],
            })

            if sample["success"] and sample["code"]:
                score_result = score_sample(
                    sample["code"],
                    task,
                    timeout=score_cfg["timeout_seconds"],
                    max_steps=score_cfg["max_steps"],
                )
                reward = score_result["reward_total"]
            else:
                reward = 0.0
                score_result = {
                    "reward_total": 0.0,
                    "reward_per_scenario": {},
                    "ran_successfully": False,
                    "errors": ["Generation failed"],
                }

            all_scores.append({
                "iter": iter_num,
                "task_id": task_id,
                "sample_id": i,
                **score_result,
            })

            task_rewards.append(reward)
            best_reward = max(best_reward, reward)

        results[task_id] = {
            "best_reward": best_reward,
            "mean_reward": sum(task_rewards) / len(task_rewards) if task_rewards else 0.0,
            "n_samples": len(task_rewards),
        }

    # Summary
    all_bests = [r["best_reward"] for r in results.values()]
    summary = {
        "iter": iter_num,
        "ollama_model": ollama_model,
        "per_task": results,
        "mean_best_reward": sum(all_bests) / len(all_bests) if all_bests else 0.0,
        "n_eval_tasks": len(eval_tasks),
    }

    # Save
    rollouts_base = Path(cfg["paths"]["rollouts_dir"])
    if args.run_name:
        rollouts_base = rollouts_base / args.run_name
    out_dir = rollouts_base / f"iter_{iter_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(all_samples, str(out_dir / "eval_samples.jsonl"))
    save_jsonl(all_scores, str(out_dir / "eval_scores.jsonl"))

    with open(str(out_dir / "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Eval summary: mean best reward = {summary['mean_best_reward']:.1f}")
    for tid, r in results.items():
        logger.info(f"  {tid}: best={r['best_reward']:.1f} mean={r['mean_reward']:.1f}")

    # Append to progress log
    n_success = sum(
        1 for s in all_scores if s.get("ran_successfully", False)
    )
    all_rewards = [s["reward_total"] for s in all_scores]
    from src.progress import append_eval_progress
    append_eval_progress(
        iter_num=iter_num,
        model_name=ollama_model,
        n_samples=len(all_scores),
        n_success=n_success,
        mean_reward=sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        mean_best_reward=summary["mean_best_reward"],
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
