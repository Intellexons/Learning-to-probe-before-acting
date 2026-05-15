"""Tests for the 4-condition `settled` metric.

`settled` is the env's task-outcome success — distinct from the existing
`ran_successfully` (which only says "policy code didn't raise"). The
metric flows from the env's `is_success()` method through the runner
subprocess → `score_sample` → `eval_scores.jsonl` → eval.csv → W&B.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.progress import EVAL_FIELDS, append_eval_progress  # noqa: E402
from src.runner import score_sample  # noqa: E402


# Env that succeeds when action > 0, fails otherwise. `is_success()` is the
# canonical SloshTank shape — a method on the env instance that reads its
# current state. Single-step episode so the test is fast.
_ENV_WITH_SUCCESS = '''
class HiddenParams:
    def __init__(self, actuator_gain=1.0):
        self.actuator_gain = actuator_gain


class _SuccessEnv:
    def __init__(self, seed=0, hidden=None):
        self._win = False

    def reset(self):
        return {"x": 0.0}, {}

    def step(self, action):
        self._win = float(action) > 0
        return {"x": float(action)}, 1.0 if self._win else -1.0, True, False, {}

    def is_success(self):
        return self._win
'''

# Env with no `is_success()` method. Runner template should report
# `settled=None`, and `score_sample` should fall back to False at
# aggregation (unknown is not success).
_ENV_NO_SUCCESS = '''
class HiddenParams:
    def __init__(self, actuator_gain=1.0):
        self.actuator_gain = actuator_gain


class _NoSuccessEnv:
    def __init__(self, seed=0, hidden=None):
        pass

    def reset(self):
        return {"x": 0.0}, {}

    def step(self, action):
        return {"x": float(action)}, 0.0, True, False, {}
'''


def _task(env_class: str, env_src: str, kappas: list[float]) -> dict:
    return {
        "task_id": "t",
        "env_class": env_class,
        "env_code": env_src,
        "scenarios": [
            {"scenario_id": f"s{i}", "seed": i, "kappa": k}
            for i, k in enumerate(kappas)
        ],
    }


def _winning_policy() -> str:
    return "def policy(state):\n    return 1.0\n"


def _losing_policy() -> str:
    return "def policy(state):\n    return -1.0\n"


def test_settled_true_when_env_reports_success():
    """Win condition (action > 0) → env.is_success() True → settled=True."""
    result = score_sample(_winning_policy(), _task("_SuccessEnv", _ENV_WITH_SUCCESS, [1.0]))
    assert result["settled_per_scenario"] == {"s0": True}
    assert result["n_scenarios_settled"] == 1
    assert result["ran_successfully"] is True


def test_settled_false_when_env_reports_failure():
    """Lose condition → env.is_success() False → settled=False; code still
    ran cleanly so `ran_successfully` stays True. This is the discriminator
    between the two metrics."""
    result = score_sample(_losing_policy(), _task("_SuccessEnv", _ENV_WITH_SUCCESS, [1.0]))
    assert result["settled_per_scenario"] == {"s0": False}
    assert result["n_scenarios_settled"] == 0
    assert result["ran_successfully"] is True


def test_settled_aggregates_across_scenarios():
    """Three scenarios, all winnable with action=1.0 → 3/3 settled."""
    result = score_sample(_winning_policy(), _task("_SuccessEnv", _ENV_WITH_SUCCESS, [0.65, 0.80, 1.00]))
    assert sum(result["settled_per_scenario"].values()) == 3
    assert result["n_scenarios_settled"] == 3


def test_settled_false_when_env_lacks_is_success():
    """Env without is_success() should not be reported as settled."""
    result = score_sample(_winning_policy(), _task("_NoSuccessEnv", _ENV_NO_SUCCESS, [1.0]))
    assert result["settled_per_scenario"] == {"s0": False}
    assert result["n_scenarios_settled"] == 0


def test_settled_false_on_validation_failure():
    """Invalid code → settled=False for every scenario, no subprocess run."""
    bad_code = "import os\n"  # disallowed import
    result = score_sample(bad_code, _task("_SuccessEnv", _ENV_WITH_SUCCESS, [0.65, 0.80]))
    assert result["settled_per_scenario"] == {"s0": False, "s1": False}
    assert result["n_scenarios_settled"] == 0
    assert result["ran_successfully"] is False


def test_append_eval_progress_writes_settle_columns(tmp_path):
    """The new settle columns land in eval.csv with the right header."""
    import src.progress as p
    original_dir = p.PROGRESS_DIR
    p.PROGRESS_DIR = str(tmp_path)
    try:
        append_eval_progress(
            iter_num=1, model_name="m", n_samples=4, n_success=4,
            mean_reward=-10.0, mean_best_reward=5.0,
            n_settled=2, n_settle_attempts=8,
            run_name="r",
        )
        eval_csv = tmp_path / "r" / "eval.csv"
        with open(eval_csv) as f:
            rows = list(csv.DictReader(f))
        assert "n_settled" in EVAL_FIELDS
        assert "settle_rate" in EVAL_FIELDS
        assert rows[0]["n_settled"] == "2"
        assert rows[0]["settle_rate"] == "25.00%"
    finally:
        p.PROGRESS_DIR = original_dir


def test_append_eval_progress_zero_attempts_does_not_div_by_zero(tmp_path):
    """Defensive: an empty eval shouldn't crash."""
    import src.progress as p
    original_dir = p.PROGRESS_DIR
    p.PROGRESS_DIR = str(tmp_path)
    try:
        append_eval_progress(
            iter_num=0, model_name="m", n_samples=0, n_success=0,
            mean_reward=0.0, mean_best_reward=0.0,
            n_settled=0, n_settle_attempts=0,
            run_name="r",
        )
        eval_csv = tmp_path / "r" / "eval.csv"
        with open(eval_csv) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["settle_rate"] == "0%"
    finally:
        p.PROGRESS_DIR = original_dir
