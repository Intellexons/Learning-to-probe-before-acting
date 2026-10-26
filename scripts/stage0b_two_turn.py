"""Stage 0b: a hand-written probe run and final run, from the brief alone, through the real path.

The proof that the task is solvable with what the model is told. The probe code
uses only what the brief offers (`env.reset()`, `env.step(a)`, the `done` flag,
`env.failure_reason` and the `sigma` reading); the controller uses only the
state and may keep memory, as the brief allows. Both go through
`execute_turn`, i.e. the sandbox subprocess with the probe-run env (hidden
limit masked, probe batch different from the scoring batch) and `score_sample`
for the final run, exactly as training and eval do. Pre-registered gate: fast
success >= 95%.

usage:
  uv run python scripts/stage0b_two_turn.py                      # train + eval task files
  uv run python scripts/stage0b_two_turn.py --n-random 200       # plus 200 fresh batches/limits
  uv run python scripts/stage0b_two_turn.py --task-file scripts/seed_tasks/crystallizer_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "seed_tasks"))

from src.dataset import load_jsonl  # noqa: E402
from src.multi_turn import execute_turn  # noqa: E402
from src.runner import DEFAULT_MAX_STEPS, FAST_FACTOR, extract_code, score_sample  # noqa: E402
from build_crystallizer_tasks import task_entry  # noqa: E402
from crystallizer_env import HiddenParams  # noqa: E402
from crystallizer_utils import cautious_policy, run_episode  # noqa: E402

DEFAULT_TASK_FILES = [
    "scripts/seed_tasks/crystallizer_train.jsonl",
    "scripts/seed_tasks/crystallizer_eval.jsonl",
]
RANDOM_SEED_BASE = 1_000
GATE_FAST_MIN = 0.95

# The probe run: find the shower limit by crossing it, on three batches. Readings only.
PROBE_SOURCE = '''
# Probe run: find the level of sigma at which this crystallizer showers, by crossing it.
lo, hi = 0.0, 0.40          # sigma_max is above lo (a reading seen safe) and below hi (a reading at a shower)
G = 0.016                   # cautious guess at how much sigma rises per full-cooling step
for npass in range(3):
    s, _ = env.reset()
    if npass == 0:
        while True:         # first batch: cool at the jacket limit until the shower
            s, r, done, _, _ = env.step(-1.0)
            if done:
                break
            lo = max(lo, s["sigma"])
        if env.failure_reason == "shower":
            hi = min(hi, s["sigma"])
        continue
    target, inc = lo, (hi - lo) / 8.0      # later batches: go to the last safe level, then creep up
    while True:
        a = max(-1.0, min(0.0, -(target - s["sigma"]) / G))
        s, r, done, _, _ = env.step(a)
        if done:
            break
        lo = max(lo, s["sigma"])
        if s["sigma"] >= target - inc / 2:
            target = min(target + inc, hi)
    if env.failure_reason == "shower":
        hi = min(hi, s["sigma"])
print(f"SIGMA_MAX_LOW={lo:.5f} SIGMA_MAX_HIGH={hi:.5f}")
'''

# The final run: hold the sigma reading just under the level the probe found.
CONTROLLER_TEMPLATE = '''
SETPOINT = {sigma_hat:.5f} - 0.003
G = 0.014                       # about how much sigma rises per full-cooling step
mem = {{"sigma_prev": None, "a_prev": 0.0, "eaten": 0.0}}
def policy(state):
    sigma, T, T_target = state["sigma"], state["T"], state["T_target"]
    if mem["sigma_prev"] is not None:
        # what the crystals ate last step: what cooling should have added, minus what we got
        eaten = mem["sigma_prev"] - mem["a_prev"] * G - sigma
        mem["eaten"] = 0.5 * mem["eaten"] + 0.5 * max(0.0, eaten)
    if T <= T_target:
        a = 0.0
    else:
        a = max(-1.0, min(0.0, -(SETPOINT - sigma + mem["eaten"]) / G))
        a = max(a, -(T - T_target) / 0.5)      # do not overshoot the target temperature
    mem["sigma_prev"], mem["a_prev"] = sigma, a
    return a
'''


def _fence(src: str) -> str:
    return "```python\n" + src.strip("\n") + "\n```"


def run_task(task: dict) -> dict:
    sc = task["scenarios"][0]
    sid = sc["scenario_id"]
    smax = sc["hidden"]["sigma_max"]
    oracle = sc["oracle_steps"]
    t0 = time.time()
    row = {"task_id": task["task_id"], "sigma_max": smax, "oracle_steps": oracle,
           "seed": sc["seed"], "probe_seed": sc["probe_seed"]}

    # Turn 1: the probe run through the real path.
    tr = execute_turn(_fence(PROBE_SOURCE), task, is_last_turn=False, max_steps=DEFAULT_MAX_STEPS)
    if "SIGMA_MAX_LOW=" not in tr.text:
        row.update({"probe_error": tr.text[:300], "lo": None, "success": False, "fast": False,
                    "norm_time": 0.0, "reward": None, "steps": 0, "errors": ["probe run failed"]})
    else:
        lo = float(tr.text.split("SIGMA_MAX_LOW=")[1].split()[0])
        # Turn 2: the final run, scored exactly as the trainer scores it.
        code = extract_code(_fence(CONTROLLER_TEMPLATE.format(sigma_hat=lo)))
        res = score_sample(code, task, max_steps=DEFAULT_MAX_STEPS)
        row.update({
            "probe_error": None, "lo": lo, "probe_abs_error": smax - lo,
            "success": bool(res["success_per_scenario"][sid]),
            "fast": bool(res["fast_per_scenario"][sid]),
            "norm_time": float(res["norm_time_per_scenario"][sid]),
            "reward": float(res["reward_per_scenario"][sid]),
            "steps": int(res["steps_per_scenario"][sid]),
            "errors": res["errors"],
        })

    # Blind reference on the same batch, in-process.
    blind = run_episode(cautious_policy, seed=sc["seed"], hidden=HiddenParams(sigma_max=smax))
    row["blind_fast"] = bool(blind["success"] and blind["steps"] <= FAST_FACTOR * oracle)
    row["blind_norm_time"] = oracle / blind["steps"] if blind["success"] and blind["steps"] else 0.0
    row["blind_reward"] = blind["reward"]
    row["seconds"] = time.time() - t0
    return row


def _summarise(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["probe_error"] is None]
    return {
        "n": n,
        "probe_failures": n - len(ok),
        "success_rate": sum(r["success"] for r in rows) / n,
        "fast_rate": sum(r["fast"] for r in rows) / n,
        "mean_norm_time": statistics.fmean(r["norm_time"] for r in rows),
        "min_norm_time": min(r["norm_time"] for r in rows),
        "mean_reward": statistics.fmean(r["reward"] for r in rows if r["reward"] is not None) if ok else None,
        "max_probe_abs_error": max((r["probe_abs_error"] for r in ok), default=None),
        "blind_fast_rate": sum(r["blind_fast"] for r in rows) / n,
        "blind_mean_norm_time": statistics.fmean(r["blind_norm_time"] for r in rows),
        "blind_mean_reward": statistics.fmean(r["blind_reward"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-file", action="append", default=None,
                        help="task file(s) to run; default: the train and eval files")
    parser.add_argument("--n-random", type=int, default=0,
                        help="also build this many fresh tasks (seeds from 1000) with the task builder's rule")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None, help="JSON output (default data/stage0b/two_turn_<date>.json)")
    args = parser.parse_args()

    tasks: list[dict] = []
    for f in args.task_file or DEFAULT_TASK_FILES:
        tasks.extend(load_jsonl(f))
    for i in range(args.n_random):
        tasks.append(task_entry("stage0b", RANDOM_SEED_BASE + i))
    print(f"Stage 0b: {len(tasks)} tasks ({len(tasks) - args.n_random} from files, {args.n_random} fresh), "
          f"{args.workers} workers")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_task, tasks))
    elapsed = time.time() - t0

    rows.sort(key=lambda r: r["sigma_max"])
    k = len(rows) // 3
    groups = [("sigma_max low third", rows[:k]), ("middle third", rows[k:2 * k]), ("high third", rows[2 * k:])]
    overall = _summarise(rows)

    print(f"\n{'group':22s} {'n':>4s} {'smax range':>14s} | {'succ':>6s} {'fast':>6s} {'norm-t':>7s} {'min':>6s} "
          f"{'reward':>7s} {'probe err':>9s} | {'blind fast':>10s} {'blind norm-t':>12s} {'blind rew':>9s}")
    for name, g in groups + [("overall", rows)]:
        if not g:
            continue
        s = _summarise(g)
        print(f"{name:22s} {s['n']:4d} {g[0]['sigma_max']:6.3f}-{g[-1]['sigma_max']:6.3f} | "
              f"{100 * s['success_rate']:5.1f}% {100 * s['fast_rate']:5.1f}% {s['mean_norm_time']:7.3f} "
              f"{s['min_norm_time']:6.3f} {s['mean_reward'] if s['mean_reward'] is not None else float('nan'):7.1f} "
              f"{s['max_probe_abs_error'] if s['max_probe_abs_error'] is not None else float('nan'):9.4f} | "
              f"{100 * s['blind_fast_rate']:9.1f}% {s['blind_mean_norm_time']:12.3f} {s['blind_mean_reward']:9.1f}")
    verdict = "PASS" if overall["fast_rate"] >= GATE_FAST_MIN and overall["probe_failures"] == 0 else "FAIL"
    print(f"\nGate: fast success >= {100 * GATE_FAST_MIN:.0f}% through the real path: "
          f"{100 * overall['fast_rate']:.1f}% ({overall['probe_failures']} probe failures) -> {verdict}")
    print(f"wall time {elapsed:.1f}s, {elapsed / max(len(rows), 1):.2f}s per task")

    out = Path(args.out or f"data/stage0b/two_turn_{date.today().isoformat()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"overall": overall, "groups": {n: _summarise(g) for n, g in groups if g},
                   "gate_fast_min": GATE_FAST_MIN, "verdict": verdict, "rows": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
