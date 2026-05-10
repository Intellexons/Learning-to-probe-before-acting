"""Build slosh_train.jsonl / slosh_eval.jsonl from slosh_tank_env.py.

Each output file contains one task entry per (seed, kappa) pair, with a
single-element `scenarios` list. The branched-rollout training scheme
estimates V_1^(i) by averaging G_2 leaves at the same prompt; for that
estimator to be unbiased the env must be byte-identical across leaves and
between turn-1 calibration and turn-2 scoring. Multi-scenario tasks
violated that — see issue #47.

Embeds the env source verbatim into each task's env_code field so the prompt
template can paste it into the model's context. Run whenever the env source
changes.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ENV_SRC = (HERE / "slosh_tank_env.py").read_text()

STATE_SPEC = (
    "state is a dict with keys: "
    "t (elapsed steps), x (cart position, m), v (cart velocity, m/s), "
    "theta (slosh angle, rad), omega (slosh angular velocity, rad/s), "
    "goal (target cart position, m), steps_left (steps remaining)."
)
ACTION_SPEC = (
    "action is a float — commanded cart acceleration in m/s². "
    "The env clips it to [-1, 1] before applying."
)
GOAL = (
    "Move the cart to the goal position and settle the liquid. "
    "Success requires all four conditions simultaneously: "
    "|x - goal| < 0.12 m, |v| < 0.12 m/s, |theta| < 0.07 rad, |omega| < 0.18 rad/s. "
    "A hidden actuator gain κ ∈ {0.65, 0.80, 1.00} scales every command you send — "
    "you do not observe κ directly. Episode length is 105 steps (6.3 s)."
)

TRAIN_SEEDS = list(range(8))
EVAL_SEEDS = list(range(10_000, 10_004))
KAPPAS = [0.65, 0.80, 1.00]


def task_entry(task_id_prefix: str, seed: int, kappa: float) -> dict:
    """One task = one (seed, kappa) scenario. See module docstring for why."""
    return {
        "task_id": f"{task_id_prefix}_s{seed}_k{kappa:.2f}",
        "env_class": "SloshTankEnv",
        "env_code": ENV_SRC,
        "state_spec": STATE_SPEC,
        "action_spec": ACTION_SPEC,
        "goal": GOAL,
        "scenarios": [
            {"scenario_id": f"s{seed}_k{kappa:.2f}", "seed": seed, "kappa": kappa},
        ],
    }


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def main() -> None:
    train_entries = [
        task_entry("slosh_tank_train", seed, kappa)
        for seed in TRAIN_SEEDS
        for kappa in KAPPAS
    ]
    eval_entries = [
        task_entry("slosh_tank_eval", seed, kappa)
        for seed in EVAL_SEEDS
        for kappa in KAPPAS
    ]

    out_train = HERE / "slosh_train.jsonl"
    out_eval = HERE / "slosh_eval.jsonl"
    _write_jsonl(out_train, train_entries)
    _write_jsonl(out_eval, eval_entries)
    print(f"Wrote {out_train} ({len(train_entries)} tasks, 1 scenario each)")
    print(f"Wrote {out_eval}  ({len(eval_entries)} tasks, 1 scenario each)")


if __name__ == "__main__":
    main()
