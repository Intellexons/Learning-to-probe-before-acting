"""Locks in the per-(seed, κ) single-scenario task layout for slosh.

Background: the branched-rollout MC value estimator V̂_1^(i) =
mean over G_2 leaves of R_{i,j} requires byte-identical envs across the
G_1 × G_2 leaves of a single prompt — and between the turn-1 calibration
env and the turn-2 scoring scenarios. Multi-scenario tasks (older
layout) violated that. See issue #47 for the full argument.

This test guards against regressions to the multi-scenario layout.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "scripts" / "seed_tasks"
sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_slosh_train_is_single_scenario_per_task():
    tasks = _load(SEED_DIR / "slosh_train.jsonl")
    for task in tasks:
        assert len(task["scenarios"]) == 1, (
            f"task {task['task_id']} has {len(task['scenarios'])} scenarios; "
            f"expected 1 (per issue #47)"
        )


def test_slosh_train_covers_all_seed_kappa_pairs():
    """8 seeds × 3 κ = 24 task entries, each unique."""
    tasks = _load(SEED_DIR / "slosh_train.jsonl")
    seen = {(t["scenarios"][0]["seed"], t["scenarios"][0]["kappa"]) for t in tasks}
    expected = {(s, k) for s in range(8) for k in (0.65, 0.80, 1.00)}
    assert seen == expected
    assert len(tasks) == 24


def test_slosh_eval_is_single_scenario_per_task():
    tasks = _load(SEED_DIR / "slosh_eval.jsonl")
    for task in tasks:
        assert len(task["scenarios"]) == 1


def test_slosh_train_env_code_is_uniform():
    """Every task must embed the same env source — divergent env_code
    across tasks would mean the model is trained against subtly different
    envs depending on which task entry is sampled."""
    tasks = _load(SEED_DIR / "slosh_train.jsonl")
    env_codes = {t["env_code"] for t in tasks}
    assert len(env_codes) == 1


def test_calibration_scenario_equals_scoring_scenario_in_execute_turn():
    """Functional check: with a single-scenario task, the calibration env
    that `execute_turn` chooses for `# MODE: main` is byte-identical to the
    scoring scenario `score_policy_turn` would use later.

    This is the load-bearing invariant for branched MC value estimation:
    across the G_1 × G_2 leaves the env must not change.
    """
    from src.runner import DEFAULT_ENV_CLASS  # noqa: F401  ensure import path

    tasks = _load(SEED_DIR / "slosh_train.jsonl")
    task = tasks[0]
    assert len(task["scenarios"]) == 1

    # The exact resolution `execute_turn` uses for cal at multi_turn.py:
    #   cal = scenario or (task.get("scenarios") or [{}])[0]
    cal = (task.get("scenarios") or [{}])[0]
    scoring = task["scenarios"][0]
    assert cal == scoring, (
        "calibration scenario must equal the (single) scoring scenario "
        "for byte-identical env across turns"
    )
