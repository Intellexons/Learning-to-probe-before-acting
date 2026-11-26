"""The aggregation behind the training-record tables (`scripts/train_render.py`) and the
first-reply table of the checkpoint evaluations (`scripts/frozen_render.py`)."""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_render = _load("train_render")
frozen_render = _load("frozen_render")


def _step(calls: int, floor: float, reward: float, seconds: float, gib: float) -> dict:
    return {"branched/generate_calls": str(calls), "branched/frac_floor": str(floor), "reward": str(reward),
            "step_time": str(seconds), "mem/peak_step_gib": str(gib)}


def test_an_iteration_counts_the_first_replies_that_were_followed_by_a_second():
    """One generate call makes the first replies; each one that does not end the task costs one more."""
    steps = [_step(5, 0.5, -200.0, 100.0, 17.0), _step(1, 0.0, -50.0, 40.0, 14.0)]
    row = train_render.iteration_row(steps, g1=4)
    assert row["first_replies"] == 8 and row["went_on"] == 4
    assert row["floor"] == pytest.approx(0.25) and row["reward"] == pytest.approx(-125.0)
    assert row["step_seconds"] == pytest.approx(70.0) and row["peak_gib"] == 17.0


def test_pooled_iterations_weigh_their_means_by_steps():
    a = train_render.iteration_row([_step(5, 1.0, -300.0, 100.0, 17.0)], g1=4)
    b = train_render.iteration_row([_step(1, 0.0, 0.0, 10.0, 12.0)] * 3, g1=4)
    both = train_render.pool([a, b])
    assert both["steps"] == 4 and both["went_on"] == 4 and both["first_replies"] == 16
    assert both["reward"] == pytest.approx(-75.0) and both["floor"] == pytest.approx(0.25)
    assert both["seconds"] == pytest.approx(130.0) and both["peak_gib"] == 17.0


def _sample(actions: list[str], reward: float, success: bool = False) -> dict:
    return {"actions": actions, "reward": reward, "success": success, "fast": False}


def test_the_first_reply_table_gives_each_kind_its_mean_outcome_and_the_gap():
    run = {"rows": [_sample(["final_run"], -60.0), _sample(["final_run"], -40.0, True),
                    _sample(["probe_run", "final_run"], -80.0), _sample(["probe_run", "probe_run"], -300.0),
                    _sample(["invalid", "final_run"], -100.0)]}
    answer, probe, invalid, gap = frozen_render.first_reply_rows(run)
    assert (answer["n"], probe["n"], invalid["n"]) == (2, 2, 1)
    assert answer["reward"] == pytest.approx(-50.0) and answer["success"] == pytest.approx(0.5)
    assert probe["reward"] == pytest.approx(-190.0) and probe["final"] == pytest.approx(0.5)
    assert gap["reward"] == pytest.approx(-140.0)


def test_the_gap_is_left_out_when_a_kind_has_no_sample():
    run = {"rows": [_sample(["final_run"], -60.0)]}
    assert frozen_render.first_reply_rows(run)[-1]["reward"] is None


def test_a_checkpoint_is_named_in_the_model_column():
    assert frozen_render.model_label({"model": "models/hf_base/iter_000"}) == "iter_000"
    assert frozen_render.model_label({"model": "models/hf_base/iter_000",
                                      "checkpoint": {"n_adapters": 3, "lora_dir": "/x"}}) == "checkpoint 3"
