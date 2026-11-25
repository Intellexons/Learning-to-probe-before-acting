"""Frozen-model references on the crystallizer: Stage 1 (one reply) and Stage 2 (two replies).

No training. The model acts as a tool-using agent (`src.agent_loop`). Stage 1
(--turns 1): one reply and no tool; the reply is the final answer, the
controller written from the brief. Stage 2 (--turns 2): two replies and one
tool, `probe_run`; a first reply that calls it is a probe run (its code runs
on the probe batch: same hidden limit, different batch, limit masked; what it
prints comes back as the tool response), and a reply without a tool call is
the final answer. The loop is `src.multi_turn.run_rollout`, built on the
`execute_turn` training uses.

For every task in the task file (default: the held-out eval set) N samples
are drawn. Per task and overall the script reports success, fast success
(within 1.25x the time-optimal batch), normalised time (oracle steps / own
steps, 0 on failure) and reward, next to two hand-written references computed
in-process: the blind cautious controller (rides at 0.077, safe for every
limit in the range) and the oracle (knows the limit and holds the end-of-step
sigma at it). Every sample keeps what each reply was (`actions`), the code of
the first call, the tool response and the full transcript, so the record
shows what the frozen model does with the probe run.

usage:
  uv run python scripts/eval_frozen_baseline.py --turns 1 --n-samples 8
  uv run python scripts/eval_frozen_baseline.py --turns 2 --n-samples 8
  uv run python scripts/eval_frozen_baseline.py --turns 4 --n-samples 8   # Stage 2 with four replies
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "seed_tasks"))

from src.dataset import load_jsonl, save_jsonl  # noqa: E402
from src.hf_generator import generate_reply, load_model_for_generation  # noqa: E402
from src.multi_turn import ACTION_PROBE, run_rollout  # noqa: E402
from src.provenance import provenance  # noqa: E402
from src.prompts import load_template, render_prompt  # noqa: E402
from src.runner import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    FAST_FACTOR,
    POLICY_DEFAULT_TIMEOUT,
    validate_code,
)
from crystallizer_env import HiddenParams  # noqa: E402
from crystallizer_utils import run_cautious_episode, run_oracle_episode  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hand-written references (in-process, no model)
# ---------------------------------------------------------------------------


def _metrics(res: dict, scenario: dict) -> dict:
    oracle = scenario["oracle_steps"]
    fast = bool(res["success"] and res["steps"] <= FAST_FACTOR * oracle)
    norm = oracle / res["steps"] if res["success"] and res["steps"] else 0.0
    return {"success": bool(res["success"]), "fast": fast, "norm_time": norm,
            "reward": res["reward"], "steps": res["steps"]}


def reference_rows(task: dict) -> dict:
    sc = task["scenarios"][0]
    hidden = HiddenParams(sigma_max=sc["hidden"]["sigma_max"])
    return {
        "blind": _metrics(run_cautious_episode(sc["seed"], hidden), sc),
        "oracle": _metrics(run_oracle_episode(sc["seed"], hidden), sc),
    }


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


def rollout(model, tokenizer, template: str, task: dict, total_turns: int, *,
            temperature: float, top_p: float, max_tokens: int,
            timeout: int, max_steps: int, show_replies_left: bool = False) -> dict:
    """One sample: the agent loop from the brief to the end of the task.

    Returns what the row needs: the score of the task, the controller of the
    final run ("" if there was none or it failed validation), what each reply
    was, and the first reply's code and tool response.
    """
    def generate(messages, tools):
        return generate_reply(model, tokenizer, messages, tools=tools, temperature=temperature,
                              top_p=top_p, max_tokens=max_tokens)

    brief = render_prompt(template, task, total_turns=total_turns)
    roll = run_rollout(generate, brief, task, total_turns, show_replies_left=show_replies_left,
                       timeout_policy=timeout, timeout_main=timeout, max_steps=max_steps)
    turns = roll["turns"]
    first = turns[0]
    if roll["final_run"]:
        valid, verr = validate_code(roll["code"])
        error = None if valid else f"controller invalid: {verr}"
    else:
        valid, error = False, turns[-1].text
    return {
        "score": roll["score"],
        "code": roll["code"] if valid else "",
        "success": valid, "error": error,
        "actions": [t.action for t in turns],
        "call_errors": [c.error for t in turns for c in t.calls if c.error],
        # the first reply was a probe run whose output came back to the model
        "probed": first.action == ACTION_PROBE,
        "t1_code": first.code,
        "tool_output": "" if first.ended else first.text,
        "transcript": roll["transcript"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Names the interaction protocol in every sample and summary, so a reader of
# the result files can tell these runs from the code-block protocol before them.
PROTOCOL = "agent-loop/final-answer"


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def probe_estimate_error(tool_output: str, sigma_max: float):
    """How far the closest plausible level in the probe output is from the true
    limit: a diagnostic of what the probe found. Numbers in [0.03, 0.5] count as
    candidate levels, at any precision (a model's probe prints unrounded
    floats); None if the output holds none."""
    nums = [float(x) for x in re.findall(r"(?<![\d.\-])0\.\d{2,}(?!\d)", tool_output or "")]
    nums = [x for x in nums if 0.03 <= x <= 0.5]
    return min(abs(x - sigma_max) for x in nums) if nums else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, default=1,
                        help="the number of replies the model has: 1 = Stage 1 (the final answer only), "
                             "2 = Stage 2 (a probe run is possible); more than 2 is Stage 2 with a longer budget")
    parser.add_argument("--model-path", default="models/hf_base/iter_000")
    parser.add_argument("--task-file", default="scripts/seed_tasks/crystallizer_eval.jsonl")
    parser.add_argument("--n-samples", type=int, default=8, help="rollouts per task")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--show-replies-left", action="store_true",
                        help="end every tool response with a line that states how many replies the model has left")
    parser.add_argument("--no-earlier-batches", action="store_true",
                        help="drop the 'What you know from earlier batches' section from the brief (ablation)")
    parser.add_argument("--timeout", type=int, default=POLICY_DEFAULT_TIMEOUT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--out-dir", default=None, help="default data/frozen/stage<turns>")
    args = parser.parse_args()
    prov = provenance(sys.argv)

    if args.turns < 1:
        parser.error("--turns must be at least 1")
    turns = args.turns
    stage = 1 if turns == 1 else 2   # the stage is whether a probe run is on offer; `turns` is the reply budget
    out_dir = Path(args.out_dir or f"data/frozen/stage{stage}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.task_file)
    if args.no_earlier_batches:
        for t in tasks:
            t.pop("earlier_batches", None)
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
            logger.info(f"{tid}: sample {i + 1}/{args.n_samples} ({turns} repl{'y' if turns == 1 else 'ies'})")
            roll = rollout(model, tokenizer, template, task, turns,
                           temperature=args.temperature, top_p=args.top_p,
                           max_tokens=args.max_tokens, timeout=args.timeout,
                           max_steps=args.max_steps, show_replies_left=args.show_replies_left)
            score = roll["score"]
            row = {
                "task_id": tid, "sample_id": i, "sigma_max": sc["hidden"]["sigma_max"],
                "protocol": PROTOCOL, "first_action": roll["actions"][0], "actions": roll["actions"],
                "call_errors": roll["call_errors"],
                "generated": bool(roll["success"]), "generation_error": roll.get("error"),
                "ran": bool(score["ran_successfully"]),
                "success": bool(score["success_per_scenario"].get(sid, False)),
                "fast": bool(score["fast_per_scenario"].get(sid, False)),
                "norm_time": float(score["norm_time_per_scenario"].get(sid, 0.0)),
                "reward": float(score["reward_per_scenario"].get(sid, score["reward_total"])),
                "steps": int(score["steps_per_scenario"].get(sid, 0)),
                "peak_sigma": score.get("peak_sigma_per_scenario", {}).get(sid),
                "errors": score["errors"],
                "code": roll["code"],
            }
            row["ride_ratio"] = (row["peak_sigma"] / sc["hidden"]["sigma_max"]
                                 if row["peak_sigma"] is not None else None)
            if stage == 2:
                row.update({k: roll.get(k) for k in ("probed", "t1_code", "tool_output")})
                row["estimate_error"] = (probe_estimate_error(roll.get("tool_output"), sc["hidden"]["sigma_max"])
                                         if roll.get("probed") else None)
            rows.append(row)
            samples_out.append({**row, "transcript": roll.get("transcript", [])})
            logger.info(f"  {'+'.join(row['actions'])}: success={row['success']} fast={row['fast']} "
                        f"norm_time={row['norm_time']:.3f} reward={row['reward']:.1f}")

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
            "mean_ride_ratio": _mean_or_none(r["ride_ratio"] for r in rows),
            "blind": refs["blind"], "oracle": refs["oracle"],
        }
        if stage == 2:
            probed = [r for r in rows if r.get("probed")]
            with_est = [r for r in probed if r.get("estimate_error") is not None]
            summary.update({
                "probe_rate": _mean(bool(r.get("probed")) for r in rows),
                "estimate_rate": len(with_est) / len(probed) if probed else 0.0,
                "mean_estimate_error": _mean_or_none(r["estimate_error"] for r in with_est),
            })
        per_task.append(summary)

    overall = {
        "provenance": prov,
        "stage": stage, "turns": turns, "model": args.model_path,
        "task_file": args.task_file, "n_tasks": len(tasks), "n_samples_per_task": args.n_samples,
        "generated_rate": _mean(t["generated_rate"] for t in per_task),
        "ran_rate": _mean(t["ran_rate"] for t in per_task),
        "success_rate": _mean(t["success_rate"] for t in per_task),
        "fast_rate": _mean(t["fast_rate"] for t in per_task),
        "mean_norm_time": _mean(t["mean_norm_time"] for t in per_task),
        "mean_reward": _mean(t["mean_reward"] for t in per_task),
        "mean_ride_ratio": _mean_or_none(t["mean_ride_ratio"] for t in per_task),
        "earlier_batches_in_brief": not args.no_earlier_batches,
        "max_tokens": args.max_tokens,
        "protocol": PROTOCOL,
        "replies_left_shown": bool(args.show_replies_left),
        "first_action_counts": _count(r["first_action"] for r in samples_out),
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
        probed = [r for r in samples_out if r.get("probed")]
        with_est = [r for r in probed if r.get("estimate_error") is not None]
        overall.update({
            "probe_rate": _mean(bool(r.get("probed")) for r in samples_out),
            "estimate_rate": len(with_est) / len(probed) if probed else 0.0,
            "mean_estimate_error": _mean_or_none(r["estimate_error"] for r in with_est),
        })

    save_jsonl(samples_out, str(out_dir / f"stage{stage}_samples.jsonl"))
    with open(out_dir / f"stage{stage}_summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Table
    print(f"\n=== Stage {stage}: frozen model, {turns} repl{'y' if turns == 1 else 'ies'}, "
          f"{len(tasks)} tasks x {args.n_samples} samples ===")
    head = (f"{'task':24s} {'smax':>6s} {'oracle':>6s} | {'blind':>5s} {'norm':>5s} | "
            f"{'ran':>5s} {'succ':>5s} {'fast':>5s} {'norm':>5s} {'reward':>7s} {'ride':>5s}")
    if stage == 2:
        head += f" {'probe':>5s} {'est':>5s} {'err':>6s}"
    print(head)
    for t in sorted(per_task, key=lambda t: t["sigma_max"]):
        line = (f"{t['task_id']:24s} {t['sigma_max']:6.3f} {t['oracle_steps']:6d} | "
                f"{100 * t['blind']['fast']:4.0f}% {t['blind']['norm_time']:5.2f} | "
                f"{100 * t['ran_rate']:4.0f}% {100 * t['success_rate']:4.0f}% "
                f"{100 * t['fast_rate']:4.0f}% {t['mean_norm_time']:5.2f} {t['mean_reward']:7.1f} "
                f"{(t['mean_ride_ratio'] if t['mean_ride_ratio'] is not None else float('nan')):5.2f}")
        if stage == 2:
            err = t["mean_estimate_error"]
            line += (f" {100 * t['probe_rate']:4.0f}% {100 * t['estimate_rate']:4.0f}% "
                     f"{(err if err is not None else float('nan')):6.3f}")
        print(line)
    print(f"\noverall: ran {100 * overall['ran_rate']:.1f}%  success {100 * overall['success_rate']:.1f}%  "
          f"fast {100 * overall['fast_rate']:.1f}%  norm-time {overall['mean_norm_time']:.3f}  "
          f"reward {overall['mean_reward']:.1f}  ride {(overall['mean_ride_ratio'] if overall['mean_ride_ratio'] is not None else float('nan')):.2f}")
    print(f"first reply: {overall['first_action_counts']}")
    if stage == 2:
        err = overall["mean_estimate_error"]
        print(f"probe:   rate {100 * overall['probe_rate']:.1f}%  with an estimate {100 * overall['estimate_rate']:.1f}%  "
              f"mean |estimate - limit| {(err if err is not None else float('nan')):.3f}")
    print(f"blind:   fast {100 * overall['blind']['fast_rate']:.1f}%  "
          f"norm-time {overall['blind']['mean_norm_time']:.3f}  reward {overall['blind']['mean_reward']:.1f}")
    print(f"oracle:  fast {100 * overall['oracle']['fast_rate']:.1f}%  "
          f"norm-time {overall['oracle']['mean_norm_time']:.3f}  reward {overall['oracle']['mean_reward']:.1f}")
    print(f"\nSaved {out_dir / f'stage{stage}_summary.json'} and {out_dir / f'stage{stage}_samples.jsonl'}")


if __name__ == "__main__":
    main()
