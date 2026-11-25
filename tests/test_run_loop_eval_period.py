"""The evaluation period of the training loop (`loop.eval_every`)."""
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# `09_run_loop.py` starts with a digit, so it is loaded by spec.
_spec = importlib.util.spec_from_file_location("run_loop_09", REPO_ROOT / "scripts" / "09_run_loop.py")
_loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loop)
should_evaluate = _loop.should_evaluate


def test_a_period_of_one_evaluates_after_every_iteration():
    assert all(should_evaluate(k, 1) for k in range(23))


def test_a_period_of_four_evaluates_after_every_fourth_iteration():
    assert [k for k in range(12) if should_evaluate(k, 4)] == [3, 7, 11]


def test_a_period_longer_than_the_run_never_evaluates():
    assert not any(should_evaluate(k, 1000) for k in range(23))


def test_a_period_below_one_is_rejected():
    with pytest.raises(ValueError):
        should_evaluate(0, 0)


def test_the_branched_run_does_not_evaluate_in_the_loop():
    """The full-run config sets the period past its own length."""
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "loop_grpo_branched_crystallizer.yaml").read_text())
    period, iterations = cfg["loop"]["eval_every"], cfg["loop"]["iterations"]
    assert not any(should_evaluate(k, period) for k in range(iterations))
