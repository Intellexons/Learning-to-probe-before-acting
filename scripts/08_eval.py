"""Evaluate the current model on the held-out tasks.

The model acts through tool calls (`src.agent_loop`). With one reply
(eval.n_turns = 1) the only tool is `final_run`: the model writes the
controller from the brief. With N replies it also has `probe_run`, whose
printed output comes back as the tool response. The loop is
`src.multi_turn.run_rollout`, built on the `execute_turn` training uses, so
eval and training see the same messages.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hf_generator import generate_reply, load_model_for_generation  # noqa: E402
from src.multi_turn import run_rollout  # noqa: E402
from src.prompts import load_template, render_prompt  # noqa: E402
from src.runner import (  # noqa: E402
    ERROR_PENALTY_PER_SCENARIO,
    error_floor_result,
    score_sample,
    validate_code,
)
from src.dataset import load_jsonl, save_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_EVAL_SAMPLES = 4


def run_n_turn_rollout(
    model,
    tokenizer,
    template: str,
    task: dict,
    n_turns: int,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    scoring_timeout: int,
    scoring_max_steps: int,
) -> dict:
    """One sample of one task through the agent loop, with `n_turns` replies.

    Returns a dict the surrounding loop consumes:

      {
        "code":          the controller of the final run ("" if the task ended
                         without a valid one),
        "success":       True iff there was a final run and its controller
                         passes `validate_code`,
        "error":         None on success, otherwise a short reason,
        "raw_response":  the model's last reply,
        "transcript":    the full chat, for the debug log,
        "actions":       what each reply was (`src.multi_turn.ACTION_*`).
      }
    """
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")

    def generate(messages, tools):
        return generate_reply(model, tokenizer, messages, tools=tools, temperature=temperature,
                              top_p=top_p, max_tokens=max_tokens)

    brief = render_prompt(template, task, total_turns=n_turns)
    roll = run_rollout(generate, brief, task, n_turns,
                       timeout_policy=scoring_timeout, max_steps=scoring_max_steps)
    out = {
        "raw_response": roll["transcript"][-1]["content"],   # a rollout ends on a reply
        "transcript": roll["transcript"],
        "actions": [t.action for t in roll["turns"]],
    }
    if roll["final_run"]:
        ok, err = validate_code(roll["code"])
        if ok:
            return {**out, "code": roll["code"], "success": True, "error": None}
        error = f"the controller of the final run is not valid: {err}"
    else:
        error = roll["turns"][-1].text
    return {**out, "code": "", "success": False, "error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    iter_num = args.iter
    gen_cfg = cfg["generation"]
    score_cfg = cfg["scoring"]
    eval_cfg = cfg.get("eval", {}) or {}
    n_turns = int(eval_cfg.get("n_turns", 1))

    # Load model with all adapters merged up to this iteration
    base_model_path = str(Path(cfg["paths"]["hf_base_dir"]) / "iter_000")
    model, tokenizer = load_model_for_generation(
        base_model_path=base_model_path,
        lora_base=cfg["paths"]["lora_dir"],
        iter_num=iter_num,
        run_name=args.run_name,
    )

    model_label = f"hf_iter_{iter_num:03d}"
    logger.info(
        f"Evaluating model with {iter_num} merged adapters "
        f"(n_turns={n_turns})"
    )

    template = load_template("configs/prompt_template.txt")
    eval_tasks = load_jsonl(cfg["paths"]["tasks_eval"])

    results = {}
    all_samples = []
    all_scores = []

    for task in eval_tasks:
        task_id = task["task_id"]
        logger.info(f"Eval task {task_id}: rolling out {N_EVAL_SAMPLES} samples × {n_turns} turn(s)")

        samples = []
        for sample_id in range(N_EVAL_SAMPLES):
            logger.info(f"  Sample {sample_id+1}/{N_EVAL_SAMPLES}")
            samples.append(run_n_turn_rollout(
                model=model,
                tokenizer=tokenizer,
                template=template,
                task=task,
                n_turns=n_turns,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                max_tokens=gen_cfg["max_tokens"],
                scoring_timeout=score_cfg["timeout_seconds"],
                scoring_max_steps=score_cfg["max_steps"],
            ))

        best_reward = 0.0
        task_rewards = []

        for i, sample in enumerate(samples):
            # Used to log, irrelevant to scoring.
            transcript = sample.get("transcript", [])
            all_samples.append({
                "iter": iter_num,
                "task_id": task_id,
                "sample_id": i,
                "n_turns": n_turns,
                "code": sample["code"],
                "generation_success": sample["success"],
                "generation_error": sample["error"],
                "transcript": transcript,
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
                # No valid final policy → match the training error floor
                # (sum of -200 across the task's scenarios) so the eval
                # metric tracks the same scoring convention training does.
                n_scenarios = max(len(task.get("scenarios", [])), 1)
                reward = ERROR_PENALTY_PER_SCENARIO * n_scenarios
                score_result = error_floor_result(
                    task, [sample.get("error") or "no valid final policy"],
                )

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
        "model": model_label,
        "n_turns": n_turns,
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
    # Task-outcome metrics over all (sample × scenario) pairs: the env's own
    # success test, fast success (within FAST_FACTOR of the time-optimal
    # batch) and normalised time. The denominator is the number of scenario
    # attempts, not the number of samples — `n_success` above is
    # rollout-level ("the code ran").
    n_task_success = sum(s.get("n_scenarios_success", 0) for s in all_scores)
    n_fast = sum(s.get("n_scenarios_fast", 0) for s in all_scores)
    norm_times = [
        v for s in all_scores for v in s.get("norm_time_per_scenario", {}).values()
    ]
    n_attempts = len(norm_times)
    mean_norm_time = sum(norm_times) / n_attempts if n_attempts else 0.0
    from src.progress import append_eval_progress
    append_eval_progress(
        iter_num=iter_num,
        model_name=model_label,
        n_samples=len(all_scores),
        n_success=n_success,
        mean_reward=sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        mean_best_reward=summary["mean_best_reward"],
        n_task_success=n_task_success,
        n_fast=n_fast,
        n_attempts=n_attempts,
        mean_norm_time=mean_norm_time,
        run_name=args.run_name,
    )
    logger.info(
        f"Eval summary: task success = {n_task_success}/{n_attempts} "
        f"({100 * n_task_success / max(n_attempts, 1):.1f}%) "
        f"| fast = {n_fast}/{n_attempts} ({100 * n_fast / max(n_attempts, 1):.1f}%) "
        f"| mean normalised time = {mean_norm_time:.3f} "
        f"| ran-without-crash = {n_success}/{len(all_scores)} "
        f"({100 * n_success / max(len(all_scores), 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
