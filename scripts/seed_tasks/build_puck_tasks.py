"""Build puck_train.jsonl / puck_eval.jsonl from puck_tilted_plane_env.py.

Embeds the env source verbatim into each task's env_code field so the prompt
template can paste it into the model's context. Run whenever the env source
changes.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ENV_SRC = (HERE / "puck_tilted_plane_env.py").read_text()

STATE_SPEC = (
    "state is a dict with keys px, py (puck position, meters), "
    "vx, vy (puck velocity, m/s), gx, gy (goal position, meters). "
    "Arena is bounded to [-6, 6] on both axes; hitting a wall zeroes "
    "the corresponding velocity component."
)
ACTION_SPEC = (
    "action is a 2-element list or tuple [ax, ay] of floats — the 2D thrust "
    "vector. It will be norm-clipped to the env's a_max before being applied. "
    "a_max, the tilt vector, and the drag coefficient mu are NOT in the state; "
    "read the env code to see their ranges, or infer them from the trajectory."
)
GOAL = (
    "Drive the puck to within 0.25 m of (gx, gy) before the episode ends "
    "(max 120 steps). The plane is tilted and linear drag applies, so "
    "thrusting straight at the goal is usually wrong — you have to cancel "
    "the tilt drift and damp velocity. Per-step reward is -dt * distance-to-"
    "goal and there is a +50 bonus on reaching the goal."
)

TRAIN_SEEDS = list(range(8))
EVAL_SEEDS = list(range(10_000, 10_004))


def task_entry(task_id: str, seeds: list[int]) -> dict:
    return {
        "task_id": task_id,
        "env_class": "PuckTiltedPlaneEnv",
        "env_code": ENV_SRC,
        "state_spec": STATE_SPEC,
        "action_spec": ACTION_SPEC,
        "goal": GOAL,
        "scenarios": [
            {"scenario_id": f"s{i}", "seed": seed}
            for i, seed in enumerate(seeds)
        ],
    }


def main() -> None:
    train = task_entry("puck_tilted_plane_train", TRAIN_SEEDS)
    eval_ = task_entry("puck_tilted_plane_eval", EVAL_SEEDS)

    (HERE / "puck_train.jsonl").write_text(json.dumps(train) + "\n")
    (HERE / "puck_eval.jsonl").write_text(json.dumps(eval_) + "\n")
    print(f"Wrote puck_train.jsonl ({len(TRAIN_SEEDS)} scenarios)")
    print(f"Wrote puck_eval.jsonl  ({len(EVAL_SEEDS)} scenarios)")


if __name__ == "__main__":
    main()
