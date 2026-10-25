"""Locks in the layout of the crystallizer task files.

One task = one batch with one hidden limit, as a single-element `scenarios`
list. The branched-rollout estimator V̂_1^(i) = mean over G_2 leaves of
R_{i,j} needs the env to be a deterministic function of the task across the
G_1 × G_2 leaves of a prompt, and the probe run must see the same hidden
limit as the final run. So the scenario pins the batch seed, the probe batch
seed, the hidden limit and the oracle time, and every task embeds the same
env source.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "scripts" / "seed_tasks"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEED_DIR))

TRAIN = SEED_DIR / "crystallizer_train.jsonl"
EVAL = SEED_DIR / "crystallizer_eval.jsonl"
SMOKE = SEED_DIR / "crystallizer_smoke_train.jsonl"
ALL_FILES = [TRAIN, EVAL, SMOKE]


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_single_scenario_per_task(path: Path):
    for task in _load(path):
        assert len(task["scenarios"]) == 1, (
            f"task {task['task_id']} has {len(task['scenarios'])} scenarios; expected 1"
        )


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_scenario_fields(path: Path):
    """Batch seed, a different probe batch seed, the hidden limit as a dict
    the runner can splat into HiddenParams, and the oracle time."""
    for task in _load(path):
        sc = task["scenarios"][0]
        assert task["env_class"] == "CrystallizerEnv"
        assert set(sc) >= {"scenario_id", "seed", "probe_seed", "hidden", "oracle_steps"}
        assert sc["probe_seed"] != sc["seed"]
        assert set(sc["hidden"]) == {"sigma_max"}
        assert 0.08 <= sc["hidden"]["sigma_max"] <= 0.25
        assert 0 < sc["oracle_steps"] <= 600


def test_train_covers_24_distinct_batches():
    tasks = _load(TRAIN)
    assert len(tasks) == 24
    assert {t["scenarios"][0]["seed"] for t in tasks} == set(range(24))
    limits = [t["scenarios"][0]["hidden"]["sigma_max"] for t in tasks]
    assert len(set(limits)) == len(limits), "hidden limits should differ between tasks"


def test_eval_batches_are_held_out():
    train_seeds = {t["scenarios"][0]["seed"] for t in _load(TRAIN)}
    eval_tasks = _load(EVAL)
    assert len(eval_tasks) == 12
    assert not ({t["scenarios"][0]["seed"] for t in eval_tasks} & train_seeds)


def test_smoke_tasks_sit_near_both_ends_of_the_range():
    limits = sorted(t["scenarios"][0]["hidden"]["sigma_max"] for t in _load(SMOKE))
    assert len(limits) == 2
    assert limits[0] < 0.10 and limits[1] > 0.23


def test_env_code_is_the_shipped_env_source():
    """Every task must embed the exact env source — divergent env_code
    across tasks would mean the model is trained against subtly different
    envs depending on which task entry is sampled."""
    src = (SEED_DIR / "crystallizer_env.py").read_text()
    for path in ALL_FILES:
        for task in _load(path):
            assert task["env_code"] == src, f"{task['task_id']} embeds a different env source"


def test_oracle_steps_match_the_tracking_controller():
    """`oracle_steps` is what the time-optimal controller needs on the
    scoring batch; the fast-success metric is relative to it."""
    from crystallizer_env import HiddenParams
    from crystallizer_utils import make_tracking_policy, run_episode

    for task in _load(EVAL)[:3]:
        sc = task["scenarios"][0]
        smax = sc["hidden"]["sigma_max"]
        res = run_episode(make_tracking_policy(smax * (1 - 1e-9)), seed=sc["seed"],
                          hidden=HiddenParams(sigma_max=smax))
        assert res["success"]
        assert res["steps"] == sc["oracle_steps"]


def test_probe_scenario_shares_the_hidden_limit_with_the_scoring_scenario():
    """What `execute_turn` resolves for `# MODE: main` is the task's single
    scenario, so the probe run and the final run share the hidden limit while
    running on different batches."""
    task = _load(TRAIN)[0]
    cal = (task.get("scenarios") or [{}])[0]  # the resolution execute_turn uses
    scoring = task["scenarios"][0]
    assert cal["hidden"] == scoring["hidden"]
    assert cal["probe_seed"] != scoring["seed"]


def test_prompt_text_names_the_hidden_limit_and_its_range():
    task = _load(TRAIN)[0]
    assert "sigma_max" in task["goal"]
    assert "0.08" in task["goal"] and "0.25" in task["goal"]
    for key in ("T", "c", "L", "T_target", "steps_left"):
        assert key in task["state_spec"]
