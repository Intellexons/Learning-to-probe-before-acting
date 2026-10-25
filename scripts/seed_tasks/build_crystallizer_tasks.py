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
  oracle_steps  steps the time-optimal controller (tracking at sigma_max) needs
                on `seed`; the runner's fast-success and normalised-time metrics
                are relative to it

The env source is embedded verbatim in every task's env_code so the prompt
template can paste it into the model's context. Run this whenever the env
source or the task text changes.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from crystallizer_env import CrystallizerEnv, EnvConfig, HiddenParams  # noqa: E402
from crystallizer_utils import make_tracking_policy, run_episode  # noqa: E402

ENV_SRC = (HERE / "crystallizer_env.py").read_text()
CFG = EnvConfig()

STATE_SPEC = (
    "state is a dict with keys: "
    "t (elapsed steps), T (solution temperature, degC), "
    "c (dissolved solute, kg per kg solvent), L (mean crystal size, m), "
    "T_target (target temperature, degC), steps_left (steps remaining)."
)
ACTION_SPEC = (
    "action is a float in [-1, 1]: the commanded jacket rate as a fraction of its "
    "1.5 degC/min limit (negative cools, positive heats). The env clips it to [-1, 1]. "
    "One step is 20 s."
)
GOAL = (
    "Cool the seeded batch to the target temperature as fast as possible without "
    "ruining it. Success: T <= T_target + 0.25 degC and the solution is equilibrated, "
    "i.e. the relative supersaturation c / c*(T) - 1 is at most 0.02, where the "
    "solubility is c*(T) = 0.20 + 0.006 T + 0.0001 T^2. Cooling raises the "
    "supersaturation; the seed crystals grow and consume it. If the supersaturation "
    "ever exceeds a hidden metastable limit sigma_max, a shower of new nuclei forms "
    "and the batch is ruined (episode over). sigma_max is fixed for this task, lies "
    "between 0.08 and 0.25, and does not appear in the state. Reward: -0.1 per step, "
    "+120 on success, -80 on failure. The episode ends after at most 600 steps (200 min)."
)

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
    """Steps the time-optimal controller needs on this batch."""
    res = run_episode(
        make_tracking_policy(sigma_max * (1.0 - 1e-9)),
        seed=seed,
        hidden=HiddenParams(sigma_max=sigma_max),
    )
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
        "state_spec": STATE_SPEC,
        "action_spec": ACTION_SPEC,
        "goal": GOAL,
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
