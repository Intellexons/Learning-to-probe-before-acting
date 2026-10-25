"""Frozen-model references on the crystallizer: Stage 1 (one turn) and Stage 2 (two turns).

No training. Stage 1 (--turns 1): the model writes `policy(state)` in a single
turn with no probe run. Stage 2 (--turns 2): the model gets a probe run
(`# MODE: main` against the probe-run env: same hidden limit, different batch,
limit masked) and then writes the final policy. The two-turn protocol is the
training/eval one: the final turn must be a policy block, and an early policy
does not end the rollout.

For every task in the task file (default: the held-out eval set) N samples
are drawn and scored with the same `score_sample` the trainer uses. Per task
and overall the script reports success, fast success (within 1.25x the
time-optimal batch), normalised time (oracle steps / own steps, 0 on failure)
and reward, next to two hand-written references computed in-process: the
blind cautious controller (rides at 0.077, safe for every limit in the range)
and the oracle (tracks the true limit). For Stage 2 the turn-1 mode, the
turn-1 code and the tool output are kept per sample, so the record shows what
the frozen model does with the probe run.

usage:
  uv run python scripts/eval_frozen_baseline.py --turns 1 --n-samples 8
  uv run python scripts/eval_frozen_baseline.py --turns 2 --n-samples 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "seed_tasks"))

from src.dataset import load_jsonl, save_jsonl  # noqa: E402
from src.hf_generator import (  # noqa: E402
    generate_code,
    generate_code_from_messages,
    load_model_for_generation,
)
from src.multi_turn import execute_turn  # noqa: E402
from src.prompts import EXAMPLE_STYLES, load_template, render_prompt  # noqa: E402
from src.runner import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    FAST_FACTOR,
    POLICY_DEFAULT_TIMEOUT,
    error_floor_result,
    extract_code_with_mode,
    score_sample,
    validate_code,
)
from crystallizer_env import HiddenParams  # noqa: E402
from crystallizer_utils import cautious_policy, make_tracking_policy, run_episode  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hand-written references (in-process, no model)
# ---------------------------------------------------------------------------


def _reference(policy, scenario: dict) -> dict:
    hidden = HiddenParams(sigma_max=scenario["hidden"]["sigma_max"])
    res = run_episode(policy, seed=scenario["seed"], hidden=hidden)
    oracle = scenario["oracle_steps"]
    fast = bool(res["success"] and res["steps"] <= FAST_FACTOR * oracle)
    norm = oracle / res["steps"] if res["success"] and res["steps"] else 0.0
    return {"success": bool(res["success"]), "fast": fast, "norm_time": norm,
            "reward": res["reward"], "steps": res["steps"]}


def reference_rows(task: dict) -> dict:
    sc = task["scenarios"][0]
    smax = sc["hidden"]["sigma_max"]
    return {
        "blind": _reference(cautious_policy, sc),
        "oracle": _reference(make_tracking_policy(smax * (1.0 - 1e-9)), sc),
    }


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


def one_turn_rollout(model, tokenizer, template: str, task: dict, *,
                     temperature: float, top_p: float, max_tokens: int,
                     example: str = "full") -> dict:
    """Stage 1: a single turn, `policy(state)` straight from the task prompt."""
    prompt = render_prompt(template, task, turn_idx=1, total_turns=1, example=example)
    sample = generate_code(model, tokenizer, prompt, temperature=temperature,
                           top_p=top_p, max_tokens=max_tokens)
    raw = sample.get("raw_response", "")
    transcript = [{"role": "user", "content": prompt},
                  {"role": "assistant", "content": raw}]
    if not sample["success"]:
        return {"code": "", "success": False, "error": sample.get("error"),
                "transcript": transcript}
    code = sample["code"]
    ok, err = validate_code(code)
    return {"code": code if ok else "", "success": ok, "error": None if ok else err,
            "transcript": transcript}


def two_turn_rollout(model, tokenizer, template: str, task: dict, *,
                     temperature: float, top_p: float, max_tokens: int,
                     timeout: int, max_steps: int, example: str = "full") -> dict:
    """Stage 2: probe run (turn 1, unscored), then the final run's policy (turn 2).

    Mirrors `08_eval.run_n_turn_rollout` for n_turns=2 and keeps the turn-1
    details: mode, code and the tool output the model read before turn 2.
    """
    scenario = task["scenarios"][0]
    prompt_t1 = render_prompt(template, task, turn_idx=1, total_turns=2, example=example)
    t1 = generate_code(model, tokenizer, prompt_t1, temperature=temperature,
                       top_p=top_p, max_tokens=max_tokens)
    t1_raw = t1.get("raw_response", "")
    transcript = [{"role": "user", "content": prompt_t1},
                  {"role": "assistant", "content": t1_raw}]
    if not t1["success"]:
        return {"code": "", "success": False, "error": f"turn 1: {t1.get('error')}",
                "t1_mode": "none", "t1_code": "", "tool_output": "", "t2_mode": "none",
                "transcript": transcript}

    t1_mode, t1_code = extract_code_with_mode(t1_raw)
    tool = execute_turn(t1_raw, task, is_last_turn=False, scenario=scenario,
                        timeout_policy=timeout, max_steps=max_steps)
    prompt_t2 = render_prompt(template, task, turn_idx=2, total_turns=2, example=example)
    transcript.append({"role": "user",
                       "content": f"{prompt_t2}\n\n[Tool output from turn 1]\n{tool.text}"})
    t2 = generate_code_from_messages(model, tokenizer, transcript, temperature=temperature,
                                     top_p=top_p, max_tokens=max_tokens)
    t2_raw = t2.get("raw_response", "")
    transcript.append({"role": "assistant", "content": t2_raw})
    t2_mode, t2_code = extract_code_with_mode(t2_raw)
    ok = bool(t2["success"]) and t2_mode == "policy" and validate_code(t2_code)[0]
    if ok:
        error = None
    elif not t2["success"]:
        error = f"turn 2: {t2.get('error')}"
    elif t2_mode != "policy":
        error = f"final turn emitted mode={t2_mode!r}, not policy"
    else:
        error = f"final policy invalid: {validate_code(t2_code)[1]}"
    return {"code": t2_code if ok else "", "success": ok, "error": error,
            "t1_mode": t1_mode, "t1_code": t1_code, "tool_output": tool.text,
            "t2_mode": t2_mode, "transcript": transcript}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, default=1, choices=[1, 2],
                        help="1 = Stage 1 (one turn, no probe run), 2 = Stage 2 (probe run, then final run)")
    parser.add_argument("--model-path", default="models/hf_base/iter_000")
    parser.add_argument("--task-file", default="scripts/seed_tasks/crystallizer_eval.jsonl")
    parser.add_argument("--n-samples", type=int, default=8, help="rollouts per task")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=POLICY_DEFAULT_TIMEOUT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--example", default="full", choices=list(EXAMPLE_STYLES),
                        help="what the prompt's multi-turn format example shows (see src/prompts.py)")
    parser.add_argument("--out-dir", default=None,
                        help="default data/frozen/stage<turns>[_example-<style>]")
    args = parser.parse_args()

    stage = args.turns
    suffix = "" if args.example == "full" else f"_example-{args.example}"
    out_dir = Path(args.out_dir or f"data/frozen/stage{stage}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.task_file)
    template = load_template("configs/prompt_template.txt")

    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_for_generation(
        base_model_path=args.model_path, lora_base=None, iter_num=0,
    )

    per_task = []
    samples_out = []
    for task in tasks:
        tid = task["task_id"]
        sc = task["scenarios"][0]
        sid = sc["scenario_id"]
        refs = reference_rows(task)
        rows = []
        for i in range(args.n_samples):
            logger.info(f"{tid}: sample {i + 1}/{args.n_samples} ({stage}-turn)")
            if stage == 1:
                roll = one_turn_rollout(model, tokenizer, template, task,
                                        temperature=args.temperature, top_p=args.top_p,
                                        max_tokens=args.max_tokens, example=args.example)
            else:
                roll = two_turn_rollout(model, tokenizer, template, task,
                                        temperature=args.temperature, top_p=args.top_p,
                                        max_tokens=args.max_tokens, timeout=args.timeout,
                                        max_steps=args.max_steps, example=args.example)
            if roll["success"] and roll["code"]:
                score = score_sample(roll["code"], task, timeout=args.timeout,
                                     max_steps=args.max_steps)
            else:
                score = error_floor_result(task, [roll.get("error") or "no valid final policy"])
            row = {
                "task_id": tid, "sample_id": i, "sigma_max": sc["hidden"]["sigma_max"],
                "generated": bool(roll["success"]), "generation_error": roll.get("error"),
                "ran": bool(score["ran_successfully"]),
                "success": bool(score["success_per_scenario"].get(sid, False)),
                "fast": bool(score["fast_per_scenario"].get(sid, False)),
                "norm_time": float(score["norm_time_per_scenario"].get(sid, 0.0)),
                "reward": float(score["reward_per_scenario"].get(sid, score["reward_total"])),
                "steps": int(score["steps_per_scenario"].get(sid, 0)),
                "errors": score["errors"],
                "code": roll["code"],
            }
            if stage == 2:
                row.update({k: roll.get(k) for k in ("t1_mode", "t1_code", "tool_output", "t2_mode")})
            rows.append(row)
            samples_out.append({**row, "transcript": roll.get("transcript", [])})
            logger.info(f"  success={row['success']} fast={row['fast']} "
                        f"norm_time={row['norm_time']:.3f} reward={row['reward']:.1f}"
                        + (f" t1_mode={row.get('t1_mode')}" if stage == 2 else ""))

        summary = {
            "task_id": tid, "sigma_max": sc["hidden"]["sigma_max"],
            "oracle_steps": sc["oracle_steps"],
            "n_samples": len(rows),
            "generated_rate": _mean(r["generated"] for r in rows),
            "ran_rate": _mean(r["ran"] for r in rows),
            "success_rate": _mean(r["success"] for r in rows),
            "fast_rate": _mean(r["fast"] for r in rows),
            "mean_norm_time": _mean(r["norm_time"] for r in rows),
            "mean_reward": _mean(r["reward"] for r in rows),
            "blind": refs["blind"], "oracle": refs["oracle"],
        }
        if stage == 2:
            summary["t1_modes"] = dict(Counter(r.get("t1_mode") for r in rows))
        per_task.append(summary)

    overall = {
        "stage": stage, "turns": stage, "model": args.model_path, "example": args.example,
        "task_file": args.task_file, "n_tasks": len(tasks), "n_samples_per_task": args.n_samples,
        "generated_rate": _mean(t["generated_rate"] for t in per_task),
        "ran_rate": _mean(t["ran_rate"] for t in per_task),
        "success_rate": _mean(t["success_rate"] for t in per_task),
        "fast_rate": _mean(t["fast_rate"] for t in per_task),
        "mean_norm_time": _mean(t["mean_norm_time"] for t in per_task),
        "mean_reward": _mean(t["mean_reward"] for t in per_task),
        "blind": {
            "fast_rate": _mean(t["blind"]["fast"] for t in per_task),
            "mean_norm_time": _mean(t["blind"]["norm_time"] for t in per_task),
            "mean_reward": _mean(t["blind"]["reward"] for t in per_task),
        },
        "oracle": {
            "fast_rate": _mean(t["oracle"]["fast"] for t in per_task),
            "mean_norm_time": _mean(t["oracle"]["norm_time"] for t in per_task),
            "mean_reward": _mean(t["oracle"]["reward"] for t in per_task),
        },
        "per_task": per_task,
    }
    if stage == 2:
        overall["t1_modes"] = dict(sum((Counter(t["t1_modes"]) for t in per_task), Counter()))

    save_jsonl(samples_out, str(out_dir / f"stage{stage}_samples.jsonl"))
    with open(out_dir / f"stage{stage}_summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Table
    print(f"\n=== Stage {stage}: frozen model, {stage} turn(s), prompt example '{args.example}', "
          f"{len(tasks)} tasks x {args.n_samples} samples ===")
    head = (f"{'task':24s} {'smax':>6s} {'oracle':>6s} | {'blind':>5s} {'norm':>5s} | "
            f"{'ran':>5s} {'succ':>5s} {'fast':>5s} {'norm':>5s} {'reward':>7s}")
    if stage == 2:
        head += " | t1 modes"
    print(head)
    for t in sorted(per_task, key=lambda t: t["sigma_max"]):
        line = (f"{t['task_id']:24s} {t['sigma_max']:6.3f} {t['oracle_steps']:6d} | "
                f"{100 * t['blind']['fast']:4.0f}% {t['blind']['norm_time']:5.2f} | "
                f"{100 * t['ran_rate']:4.0f}% {100 * t['success_rate']:4.0f}% "
                f"{100 * t['fast_rate']:4.0f}% {t['mean_norm_time']:5.2f} {t['mean_reward']:7.1f}")
        if stage == 2:
            line += " | " + ", ".join(f"{k}={v}" for k, v in sorted(t["t1_modes"].items()))
        print(line)
    print(f"\noverall: ran {100 * overall['ran_rate']:.1f}%  success {100 * overall['success_rate']:.1f}%  "
          f"fast {100 * overall['fast_rate']:.1f}%  norm-time {overall['mean_norm_time']:.3f}  "
          f"reward {overall['mean_reward']:.1f}")
    print(f"blind:   fast {100 * overall['blind']['fast_rate']:.1f}%  "
          f"norm-time {overall['blind']['mean_norm_time']:.3f}  reward {overall['blind']['mean_reward']:.1f}")
    print(f"oracle:  fast {100 * overall['oracle']['fast_rate']:.1f}%  "
          f"norm-time {overall['oracle']['mean_norm_time']:.3f}  reward {overall['oracle']['mean_reward']:.1f}")
    if stage == 2:
        print("turn-1 modes: " + ", ".join(f"{k}={v}" for k, v in sorted(overall["t1_modes"].items())))
    print(f"\nSaved {out_dir / f'stage{stage}_summary.json'} and {out_dir / f'stage{stage}_samples.jsonl'}")


if __name__ == "__main__":
    main()
