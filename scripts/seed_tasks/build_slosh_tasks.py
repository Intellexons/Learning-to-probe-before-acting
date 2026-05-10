"""Build slosh_train.jsonl / slosh_eval.jsonl from slosh_tank_env.py.

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


def task_entry(task_id: str, seeds: list[int]) -> dict:
    return {
        "task_id": task_id,
        "env_class": "SloshTankEnv",
        "env_code": ENV_SRC,
        "state_spec": STATE_SPEC,
        "action_spec": ACTION_SPEC,
        "goal": GOAL,
        "scenarios": [
            {"scenario_id": f"s{seed}_k{kappa:.2f}", "seed": seed, "kappa": kappa}
            for seed in seeds
            for kappa in KAPPAS
        ],
    }


def main() -> None:
    train = task_entry("slosh_tank_train", TRAIN_SEEDS)
    eval_ = task_entry("slosh_tank_eval", EVAL_SEEDS)

    out_train = HERE / "slosh_train.jsonl"
    out_eval  = HERE / "slosh_eval.jsonl"
    out_train.write_text(json.dumps(train) + "\n")
    out_eval.write_text(json.dumps(eval_) + "\n")
    print(f"Wrote {out_train} ({len(TRAIN_SEEDS)} scenarios)")
    print(f"Wrote {out_eval}  ({len(EVAL_SEEDS)} scenarios)")


if __name__ == "__main__":
    main()
