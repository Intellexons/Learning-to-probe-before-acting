"""Evaluate the current model on the held-out tasks.

Single-turn (eval.n_turns = 1): the model writes the controller from the brief.
Multi-turn (eval.n_turns = N): every turn but the last is a probe run, executed
in the sandbox with its stdout returned to the model; the last turn is the
controller, which is scored. Turn kinds are positional, exactly as in training
(`src/multi_turn.execute_turn`), so eval and training see the same messages.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hf_generator import (  # noqa: E402
    generate_batch,
    generate_code,
    generate_code_from_messages,
    load_model_for_generation,
)
from src.multi_turn import execute_turn  # noqa: E402
from src.prompts import compose_turn_message, load_template, render_prompt  # noqa: E402
from src.runner import (  # noqa: E402
    ERROR_PENALTY_PER_SCENARIO,
    error_floor_result,
    extract_code,
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
    """Run an `n_turns`-turn rollout for one task; never short-circuit.

    Returns a dict shaped like the existing single-turn `generate_code`
    result so the surrounding loop can consume it identically:

      {
        "code":          the final turn's controller (or "" if the rollout
                         ended without a valid one),
        "success":       True iff the final turn is a valid controller,
        "error":         None on success, otherwise a short reason string,
        "raw_response":  raw text of the FINAL turn (matches single-turn
                         semantics; the full transcript is available via
                         `transcript` for callers that need it),
        "transcript":    [{"role": ..., "content": ...}, ...] — the full
                         chat history the model saw, for debug logging.
      }

    The turn position decides what each completion is: every turn but the
    last is probe code (executed via `execute_turn`, stdout returned to the
    model), the last is the controller. The controller must pass
    `validate_code`; otherwise the rollout is rejected, matching the rule
    training enforces in `src/multi_turn.execute_turn` (`is_last_turn=True`).
    """
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")

    # Single-turn path is preserved separately to keep the surface area
    # of legacy DPO-loop runs unchanged.
    if n_turns == 1:
        prompt = render_prompt(template, task, turn_idx=1, total_turns=1)
        sample = generate_code(
            model, tokenizer, prompt,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        return {
            **sample,
            "transcript": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": sample.get("raw_response", "")},
            ],
        }

    # Multi-turn path. Build the chat history incrementally; never short-circuit.
    transcript: list[dict] = []
    final_code = ""
    final_raw = ""

    # The single scenario the probe-run env is built from (same hidden
    # parameters as the final run, probe batch). Tasks are single-scenario,
    # so this matches what the trainer's `execute_turn` resolves.
    scenario = (task.get("scenarios") or [{}])[0]

    for turn_idx in range(1, n_turns + 1):
        is_last = (turn_idx == n_turns)

        if turn_idx == 1:
            prompt_t1 = render_prompt(template, task, turn_idx=1, total_turns=n_turns)
            transcript.append({"role": "user", "content": prompt_t1})
            sample = generate_code(
                model, tokenizer, prompt_t1,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            )
        else:
            sample = generate_code_from_messages(
                model, tokenizer, transcript,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            )

        if not sample["success"]:
            logger.warning(f"  turn {turn_idx} generation failed: {sample.get('error')}")
            return {
                "code": "",
                "success": False,
                "error": f"turn {turn_idx}: {sample.get('error')}",
                "raw_response": sample.get("raw_response", ""),
                "transcript": transcript,
            }

        raw = sample["raw_response"]
        final_raw = raw
        transcript.append({"role": "assistant", "content": raw})

        if is_last:
            final_code = extract_code(raw)
            break

        # Probe turn: dispatch via the same execute_turn used in training so
        # the tool-message text is byte-identical to the training rollout.
        tool_result = execute_turn(
            raw, task,
            is_last_turn=False,
            scenario=scenario,
            timeout_policy=scoring_timeout,
            max_steps=scoring_max_steps,
        )
        next_prompt = render_prompt(template, task, turn_idx=turn_idx + 1, total_turns=n_turns)
        transcript.append({"role": "user", "content": compose_turn_message(next_prompt, tool_result.text)})

    ok, err = validate_code(final_code)
    if ok:
        return {
            "code": final_code,
            "success": True,
            "error": None,
            "raw_response": final_raw,
            "transcript": transcript,
        }
    return {
        "code": "",
        "success": False,
        "error": f"final turn (turn {n_turns}) is not a valid controller: {err}",
        "raw_response": final_raw,
        "transcript": transcript,
    }

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

        if n_turns == 1:
            # Legacy fast path: same shape as before so DPO loops are unaffected.
            prompt = render_prompt(template, task)
            samples = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                n_samples=N_EVAL_SAMPLES,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                max_tokens=gen_cfg["max_tokens"],
            )
            samples = [
                {**s, "transcript": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": s.get("raw_response", "")},
                ]}
                for s in samples
            ]
        else:
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
