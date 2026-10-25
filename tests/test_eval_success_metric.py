"""Tests for the task-outcome metrics: success, fast success, normalised time.

`success` is the env's own `is_success()`, distinct from `ran_successfully`
(the policy code did not raise). `fast` is success within FAST_FACTOR of the
scenario's `oracle_steps`; `norm_time` is oracle_steps / own steps, 0 on
failure. All three flow from the runner subprocess through `score_sample`
to eval.csv and W&B.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.progress import EVAL_FIELDS, append_eval_progress  # noqa: E402
from src.runner import FAST_FACTOR, error_floor_result, score_sample  # noqa: E402


# Env that runs for `action` steps (rounded, at least 1) and succeeds iff the
# action was positive, so a policy chooses both the outcome and the time.
_STEPS_ENV = '''
class HiddenParams:
    def __init__(self, **kw):
        self.kw = kw


class _StepsEnv:
    def __init__(self, seed=0, hidden=None):
        self._n = None
        self._t = 0
        self._win = False

    def reset(self):
        return {"t": 0.0}, {}

    def step(self, action):
        if self._n is None:
            self._n = max(1, int(round(float(action))))
            self._win = float(action) > 0
        self._t += 1
        done = self._t >= self._n
        return {"t": float(self._t)}, 1.0 if self._win else -1.0, done, False, {}

    def is_success(self):
        return self._win
'''

# Env with no `is_success()` method: the runner reports success=None and
# `score_sample` counts that as not-a-success.
_NO_SUCCESS_ENV = '''
class HiddenParams:
    def __init__(self, **kw):
        self.kw = kw


class _NoSuccessEnv:
    def __init__(self, seed=0, hidden=None):
        pass

    def reset(self):
        return {"x": 0.0}, {}

    def step(self, action):
        return {"x": float(action)}, 0.0, True, False, {}
'''

# Env whose reward is its hidden parameter: shows the scenario's `hidden`
# dict reaches HiddenParams(**hidden) in the subprocess.
_HIDDEN_ECHO_ENV = '''
class HiddenParams:
    def __init__(self, **kw):
        self.kw = kw


class _EchoEnv:
    def __init__(self, seed=0, hidden=None):
        self.hidden = hidden

    def reset(self):
        return {"x": 0.0}, {}

    def step(self, action):
        return {"x": 0.0}, float(self.hidden.kw["limit"]), True, False, {}

    def is_success(self):
        return True
'''


def _task(env_class: str, env_src: str, scenarios: list[dict]) -> dict:
    return {"task_id": "t", "env_class": env_class, "env_code": env_src, "scenarios": scenarios}


def _policy(action: float) -> str:
    return f"def policy(state):\n    return {action!r}\n"


def test_success_and_fast_when_within_oracle_time():
    task = _task("_StepsEnv", _STEPS_ENV, [{"scenario_id": "s0", "seed": 0, "oracle_steps": 4}])
    result = score_sample(_policy(4.0), task)
    assert result["success_per_scenario"] == {"s0": True}
    assert result["fast_per_scenario"] == {"s0": True}
    assert result["norm_time_per_scenario"]["s0"] == pytest.approx(1.0)
    assert result["steps_per_scenario"] == {"s0": 4}
    assert result["n_scenarios_success"] == 1 and result["n_scenarios_fast"] == 1
    assert result["mean_norm_time"] == pytest.approx(1.0)
    assert result["ran_successfully"] is True


def test_success_but_slow_is_not_fast():
    """Twice the oracle time: success yes, fast no, normalised time 0.5."""
    task = _task("_StepsEnv", _STEPS_ENV, [{"scenario_id": "s0", "seed": 0, "oracle_steps": 4}])
    result = score_sample(_policy(8.0), task)
    assert result["success_per_scenario"] == {"s0": True}
    assert result["fast_per_scenario"] == {"s0": False}
    assert result["norm_time_per_scenario"]["s0"] == pytest.approx(0.5)
    assert 8 > FAST_FACTOR * 4


def test_failure_zeroes_the_time_metrics_but_code_still_ran():
    task = _task("_StepsEnv", _STEPS_ENV, [{"scenario_id": "s0", "seed": 0, "oracle_steps": 4}])
    result = score_sample(_policy(-1.0), task)
    assert result["success_per_scenario"] == {"s0": False}
    assert result["fast_per_scenario"] == {"s0": False}
    assert result["norm_time_per_scenario"] == {"s0": 0.0}
    assert result["ran_successfully"] is True


def test_scenario_without_oracle_time_falls_back_to_success():
    task = _task("_StepsEnv", _STEPS_ENV, [{"scenario_id": "s0", "seed": 0}])
    result = score_sample(_policy(7.0), task)
    assert result["fast_per_scenario"] == {"s0": True}
    assert result["norm_time_per_scenario"]["s0"] == pytest.approx(1.0)


def test_aggregates_across_scenarios():
    scenarios = [{"scenario_id": f"s{i}", "seed": i, "oracle_steps": 4} for i in range(3)]
    result = score_sample(_policy(5.0), _task("_StepsEnv", _STEPS_ENV, scenarios))
    assert result["n_scenarios_success"] == 3
    assert result["n_scenarios_fast"] == 3  # 5 <= 1.25 * 4
    assert result["mean_norm_time"] == pytest.approx(0.8)


def test_env_without_is_success_is_not_a_success():
    task = _task("_NoSuccessEnv", _NO_SUCCESS_ENV, [{"scenario_id": "s0", "seed": 0}])
    result = score_sample(_policy(1.0), task)
    assert result["success_per_scenario"] == {"s0": False}
    assert result["fast_per_scenario"] == {"s0": False}
    assert result["ran_successfully"] is True


def test_hidden_dict_reaches_the_env():
    task = _task("_EchoEnv", _HIDDEN_ECHO_ENV,
                 [{"scenario_id": "s0", "seed": 0, "hidden": {"limit": 0.3}}])
    result = score_sample(_policy(0.0), task)
    assert result["reward_per_scenario"]["s0"] == pytest.approx(0.3)


def test_validation_failure_is_the_error_floor_for_every_scenario():
    """Invalid code: no subprocess run, every metric at its floor."""
    scenarios = [{"scenario_id": "s0", "seed": 0, "oracle_steps": 4},
                 {"scenario_id": "s1", "seed": 1, "oracle_steps": 4}]
    task = _task("_StepsEnv", _STEPS_ENV, scenarios)
    result = score_sample("import os\n", task)  # disallowed import
    assert result["success_per_scenario"] == {"s0": False, "s1": False}
    assert result["fast_per_scenario"] == {"s0": False, "s1": False}
    assert result["norm_time_per_scenario"] == {"s0": 0.0, "s1": 0.0}
    assert result["n_scenarios_success"] == 0 and result["n_scenarios_fast"] == 0
    assert result["ran_successfully"] is False
    # Same shape as the stub eval uses for a rollout with no valid final policy.
    assert set(result) == set(error_floor_result(task, ["x"]))


def test_append_eval_progress_writes_task_outcome_columns(tmp_path):
    import src.progress as p
    original_dir = p.PROGRESS_DIR
    p.PROGRESS_DIR = str(tmp_path)
    try:
        append_eval_progress(
            iter_num=1, model_name="m", n_samples=4, n_success=4,
            mean_reward=-10.0, mean_best_reward=5.0,
            n_task_success=6, n_fast=2, n_attempts=8, mean_norm_time=0.4375,
            run_name="r",
        )
        with open(tmp_path / "r" / "eval.csv") as f:
            rows = list(csv.DictReader(f))
        for col in ("n_task_success", "task_success_rate", "n_fast", "fast_rate", "mean_norm_time"):
            assert col in EVAL_FIELDS
        assert rows[0]["n_task_success"] == "6"
        assert rows[0]["task_success_rate"] == "75.00%"
        assert rows[0]["n_fast"] == "2"
        assert rows[0]["fast_rate"] == "25.00%"
        assert rows[0]["mean_norm_time"] == "0.438"
    finally:
        p.PROGRESS_DIR = original_dir


def test_append_eval_progress_zero_attempts_does_not_div_by_zero(tmp_path):
    import src.progress as p
    original_dir = p.PROGRESS_DIR
    p.PROGRESS_DIR = str(tmp_path)
    try:
        append_eval_progress(
            iter_num=0, model_name="m", n_samples=0, n_success=0,
            mean_reward=0.0, mean_best_reward=0.0,
            n_task_success=0, n_fast=0, n_attempts=0, mean_norm_time=0.0,
            run_name="r",
        )
        with open(tmp_path / "r" / "eval.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["task_success_rate"] == "0%"
        assert rows[0]["fast_rate"] == "0%"
    finally:
        p.PROGRESS_DIR = original_dir
