"""Build the crystallizer task files from crystallizer_env.py.

Writes crystallizer_train.jsonl (24 tasks), crystallizer_eval.jsonl (12 held-out
tasks) and crystallizer_smoke_train.jsonl (2 tasks for pipeline smoke runs).

One task = one batch with its own hidden metastable limit, as a single-element
`scenarios` list. The branched-rollout estimator averages G_2 leaves under the
same first turn, so everything the env does must be a deterministic function
of the task: the batch seed, the probe batch seed and the hidden limit are all
written into the task file. Each scenario carries

  seed          the batch the final run (the one that is scored) is run on
  probe_seed    a different batch for the probe run; same hidden limit
  hidden        {"sigma_max": ...}; the runner builds HiddenParams(**hidden)
  oracle_steps  steps the time-optimal controller needs on `seed` when it knows
                the limit (it holds the end-of-step sigma at sigma_max); the
                runner's fast-success and normalised-time metrics are relative to it

The env source is embedded verbatim in every task's env_code so the runner can
execute it in its sandbox; the model never sees it. What the model sees is the
brief, rendered by src/prompts.py from the task's `intro`, `goal`, `state_spec`,
`action_spec`, `unobserved`, `probe_spec` and `final_spec`: the goal and the
physics in words, what is observed, what is not, and what the probe run is for.
No equation, no source. Run this whenever the env source or the text changes.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from crystallizer_env import CrystallizerEnv, EnvConfig, HiddenParams  # noqa: E402
from crystallizer_utils import run_oracle_episode  # noqa: E402

ENV_SRC = (HERE / "crystallizer_env.py").read_text()
CFG = EnvConfig()
SIGMA_MAX_RANGE = (0.08, 0.25)          # CrystallizerEnv.sample_hidden
STEP_DEGC = CFG.max_rate * CFG.dt / 60.0

INTRO = (
    "You are the control engineer for a seeded batch cooling crystallizer: a stirred tank of "
    "hot solution holding a known mass of small seed crystals, cooled through a jacket whose "
    "rate you set. Your job is a Python controller for one batch."
)

GOAL = (
    "Cool the batch to its target temperature as fast as possible without ruining it.\n\n"
    "The liquid holds less dissolved material as it cools. Whatever it cannot hold is the "
    "supersaturation, the reading sigma: sigma = 0.10 means ten percent more is dissolved than "
    "the liquid can hold at the current temperature. Supersaturation makes the seed crystals "
    "grow, and growth uses it up, so at constant temperature sigma falls back towards zero; the "
    "more crystal surface there is, the faster. If sigma rises too far the liquid showers: a "
    "cloud of tiny new crystals forms at once and the batch is ruined. Heating does the opposite "
    "of cooling: it dissolves the seeds, and if they shrink below "
    f"{CFG.dissolved_frac:.0%} of their starting size the batch is lost as well.\n\n"
    f"Success: T <= T_target + {CFG.T_tol:g} degC with sigma at most {CFG.sigma_tol:g}. The episode "
    f"ends at success, at a shower, when the seeds are lost, or after {CFG.max_steps} steps "
    f"({CFG.max_steps * CFG.dt / 60:.0f} min). Reward: -{CFG.time_weight:g} per step, "
    f"+{CFG.success_bonus:g} on success, -{CFG.fail_penalty:g} if the batch is ruined."
)

STATE_SPEC = (
    "`state` is a dict with keys: t (elapsed steps), T (solution temperature, degC), "
    "c (dissolved material, kg per kg solvent), L (mean crystal size, m), T_target (target "
    "temperature, degC), steps_left (steps remaining), sigma (relative supersaturation, "
    "dimensionless)."
)

ACTION_SPEC = (
    f"`action` is a float in [-1, 1]: the jacket rate as a fraction of its {CFG.max_rate:g} degC/min "
    "limit; negative cools, positive heats; the env clips it to [-1, 1]. One step is "
    f"{CFG.dt:g} s, so full cooling lowers T by {STEP_DEGC:g} degC per step."
)

UNOBSERVED = (
    "sigma_max, the level of sigma at which this batch showers. It is not in the state, and "
    "nothing in the readings depends on it until it is crossed. It is fixed for this task and "
    f"lies between {SIGMA_MAX_RANGE[0]:g} and {SIGMA_MAX_RANGE[1]:g}. How fast you can safely cool "
    "depends on it."
)

PROBE_SPEC = (
    "the same crystallizer, with the same sigma_max, on a different batch (its own start "
    "temperature, target and seed loading)"
)
FINAL_SPEC = "the same sigma_max as your probe run, a different batch"

TRAIN_SEEDS = list(range(24))
EVAL_SEEDS = list(range(10_000, 10_012))
# Smoke tasks: one limit near each end of the hidden range.
SMOKE = [(0, 0.09), (1, 0.24)]

PROBE_SEED_OFFSET = 100_000
HIDDEN_SALT = 7_919


def draw_sigma_max(seed: int) -> float:
    """The hidden limit for a task, a deterministic function of its seed."""
    rng = random.Random(HIDDEN_SALT * seed + 17)
    return round(CrystallizerEnv.sample_hidden(rng).sigma_max, 4)


def oracle_steps(seed: int, sigma_max: float) -> int:
    """Steps the time-optimal controller needs on this batch when it knows the limit."""
    res = run_oracle_episode(seed, HiddenParams(sigma_max=sigma_max))
    if not res["success"]:
        raise RuntimeError(f"oracle failed on seed={seed} sigma_max={sigma_max}: {res}")
    return int(res["steps"])


def task_entry(task_id_prefix: str, seed: int, sigma_max: float | None = None) -> dict:
    """One task = one batch with one hidden limit. See module docstring."""
    smax = draw_sigma_max(seed) if sigma_max is None else float(sigma_max)
    return {
        "task_id": f"{task_id_prefix}_s{seed}",
        "env_class": "CrystallizerEnv",
        "env_code": ENV_SRC,
        "intro": INTRO,
        "goal": GOAL,
        "state_spec": STATE_SPEC,
        "action_spec": ACTION_SPEC,
        "unobserved": UNOBSERVED,
        "probe_spec": PROBE_SPEC,
        "final_spec": FINAL_SPEC,
        "scenarios": [
            {
                "scenario_id": f"s{seed}",
                "seed": seed,
                "probe_seed": seed + PROBE_SEED_OFFSET,
                "hidden": {"sigma_max": smax},
                "oracle_steps": oracle_steps(seed, smax),
            },
        ],
    }


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def main() -> None:
    train_entries = [task_entry("crystallizer_train", seed) for seed in TRAIN_SEEDS]
    eval_entries = [task_entry("crystallizer_eval", seed) for seed in EVAL_SEEDS]
    smoke_entries = [task_entry("crystallizer_smoke", seed, smax) for seed, smax in SMOKE]

    for name, entries in [
        ("crystallizer_train.jsonl", train_entries),
        ("crystallizer_eval.jsonl", eval_entries),
        ("crystallizer_smoke_train.jsonl", smoke_entries),
    ]:
        out = HERE / name
        _write_jsonl(out, entries)
        limits = sorted(e["scenarios"][0]["hidden"]["sigma_max"] for e in entries)
        print(f"Wrote {out} ({len(entries)} tasks, 1 scenario each, "
              f"sigma_max {limits[0]:.4f}..{limits[-1]:.4f})")


if __name__ == "__main__":
    main()
