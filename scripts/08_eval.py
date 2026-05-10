"""Evaluate current model on held-out eval tasks.

Usage: python scripts/08_eval.py --iter 1 --config configs/loop.yaml

Supports two modes via `cfg["eval"]["n_turns"]` (default 1):
  - n_turns=1: single-turn rollout (legacy DPO-loop behaviour).
  - n_turns>=2: full N-turn rollout matching the training protocol —
    intermediate `# MODE: main` turns are executed in the sandbox and
    their stdout fed back as a tool message; intermediate `# MODE: policy`
    turns are scored too (their reward returned as the tool message) but
    do NOT short-circuit the rollout. The policy that gets evaluated for
    the W&B reward curve is the LAST `# MODE: policy` block emitted across
    the whole rollout — typically the final-turn one. If the final turn
    is not policy (i.e. the model emits `main` or invalid as its last
    turn) the trajectory is rejected at the error floor, matching the
    training-side `is_last_turn=True` rule in `src/multi_turn.execute_turn`.

Why "last policy" and not "first policy": the trained model is being
optimised to refine its draft across turns, so an early commit is not
representative of what the model produced at the end of an N-turn
rollout. This intentionally diverges from `eval_frozen_baseline.py`,
which short-circuits on turn-1 policy because the frozen baseline
isn't trained to use later turns.
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
from src.prompts import load_template, render_prompt  # noqa: E402
from src.runner import (  # noqa: E402
    ERROR_PENALTY_PER_SCENARIO,
    extract_code_with_mode,
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
        "code":          last `# MODE: policy` block emitted (or "" if
                         the rollout ended without a valid final policy),
        "success":       True iff the final turn emitted a valid policy,
        "error":         None on success, otherwise a short reason string,
        "raw_response":  raw text of the FINAL turn (matches single-turn
                         semantics; the full transcript is available via
                         `transcript` for callers that need it),
        "transcript":    [{"role": ..., "content": ...}, ...] — the full
                         chat history the model saw, for debug logging.
      }

    Selection rule: among all turns 1..N, the LAST turn whose extracted
    mode is "policy" (and whose code passes `validate_code`) wins. If
    the final turn (turn N) is not a valid policy, the rollout is
    rejected even if an earlier turn emitted one — matching the
    "final turn must be `# MODE: policy`" protocol rule that training
    enforces in `src/multi_turn.execute_turn` (`is_last_turn=True`).
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

    # Multi-turn path. Build chat history incrementally; track last
    # valid policy across all turns; never short-circuit.
    transcript: list[dict] = []
    last_policy_code = ""
    final_mode = "invalid"
    final_raw = ""

    # The single scenario the calibration env should target. Post-PR #49
    # tasks are single-scenario, so this picks up the matching (seed, κ)
    # automatically. The byte-identical-env invariant from issue #47
    # carries over to eval here.
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

        mode, code = extract_code_with_mode(raw)
        final_mode = mode

        if mode == "policy":
            is_valid, _ = validate_code(code)
            if is_valid:
                # Track but never stop. Even if turn 1 is policy and we're
                # in a 2-turn rollout, we keep going to give the model the
                # chance to refine in turn 2.
                last_policy_code = code

        if is_last:
            break

        # Intermediate turn: dispatch via the same execute_turn used in
        # training so the tool-message text is byte-identical to the
        # training rollout. For mode==main: stdout. For mode==policy
        # (intermediate, non-final): scored; reward returned as text.
        # For mode==invalid: rejection text.
        tool_result = execute_turn(
            raw, task,
            is_last_turn=False,
            scenario=scenario,
            timeout_policy=scoring_timeout,
            max_steps=scoring_max_steps,
        )

        # Render the next-turn prompt and prepend it to the tool message,
        # matching the convention used by eval_frozen_baseline.py:198-203
        # and BranchedGRPOTrainer post-PR-#50.
        next_prompt = render_prompt(
            template, task, turn_idx=turn_idx + 1, total_turns=n_turns,
        )
        user_content = (
            f"{next_prompt}\n\n[Tool output from turn {turn_idx}]\n{tool_result.text}"
        )
        transcript.append({"role": "user", "content": user_content})

    # Decide acceptance. Final turn must be a valid policy; otherwise the
    # protocol rejects the trajectory even if an earlier turn produced
    # one. last_policy_code is the LAST valid policy emitted across the
    # rollout, which is the same thing as the final-turn policy when
    # that turn is policy.
    if final_mode == "policy" and last_policy_code:
        return {
            "code": last_policy_code,
            "success": True,
            "error": None,
            "raw_response": final_raw,
            "transcript": transcript,
        }
    return {
        "code": "",
        "success": False,
        "error": f"final turn (turn {n_turns}) emitted mode={final_mode!r}, not policy",
        "raw_response": final_raw,
        "transcript": transcript,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", default="configs/loop.yaml")
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
                score_result = {
                    "reward_total": reward,
                    "reward_per_scenario": {
                        s["scenario_id"]: ERROR_PENALTY_PER_SCENARIO
                        for s in task.get("scenarios", [])
                    },
                    "ran_successfully": False,
                    "errors": [sample.get("error") or "no valid final policy"],
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
    from src.progress import append_eval_progress
    append_eval_progress(
        iter_num=iter_num,
        model_name=model_label,
        n_samples=len(all_scores),
        n_success=n_success,
        mean_reward=sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        mean_best_reward=summary["mean_best_reward"],
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
