"""The eval-side contract of `08_eval.py:run_n_turn_rollout`.

Mocks the generator so the rollout can be exercised on CPU. What a reply is
and what follows it is covered by `test_multi_turn.py`; here we lock down
what the eval script makes of a rollout:

  - a final run ends the task on any reply, and its controller is the code
    the script scores;
  - a rollout whose replies run out without a final run, or whose controller
    fails validation, is reported as not successful with the reason.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.agent_loop import TOOL_FINAL, TOOL_PROBE, render_tool_call  # noqa: E402
from src.prompts import load_template  # noqa: E402

TEMPLATE = load_template(str(REPO_ROOT / "configs" / "prompt_template.txt"))

# `08_eval.py` starts with a digit so a normal import won't work; load
# it as a module by spec.
_EVAL_PATH = REPO_ROOT / "scripts" / "08_eval.py"
_spec = importlib.util.spec_from_file_location("eval_08", _EVAL_PATH)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
run_n_turn_rollout = _eval_mod.run_n_turn_rollout


# A minimal env source the runner can `score_sample` against. Single
# scenario keeps eval/training byte-identical (post-PR #49).
_TINY_ENV_SRC = '''
import math, random


class HiddenParams:
    def __init__(self, **hidden):
        self.hidden = hidden


class TinyEnv:
    def __init__(self, seed=0, hidden=None):
        rng = random.Random(seed)
        self.gx = rng.uniform(-1, 1)
        self.steps = 0

    def reset(self):
        return {"px": 0.0, "gx": self.gx}, {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= 5
        return {"px": float(action[0]) * 0.1, "gx": self.gx}, -1.0, done, False, {}
'''


def _tiny_task() -> dict:
    return {
        "task_id": "tiny",
        "env_class": "TinyEnv",
        "env_code": _TINY_ENV_SRC,
        "state_spec": "- px (float)\n- gx (float)",
        "action_spec": "(ax, ay) — 2-tuple",
        "goal": "test goal",
        "scenarios": [{"scenario_id": "s0", "seed": 0, "hidden": {}}],
    }


class _ScriptedGenerator:
    """Drop-in stand-in for `generate_reply`: returns the scripted replies in
    turn. The rollout does not care what the chat looks like, only what the
    next reply says."""

    def __init__(self, completions: list[str]):
        self._queue = list(completions)
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return self._queue.pop(0) if self._queue else ""


def _patch_generators(monkeypatch, completions: list[str]):
    gen = _ScriptedGenerator(completions)
    monkeypatch.setattr(_eval_mod, "generate_reply", gen)
    return gen


# The controllers below return `(0.5, 0.0)` so the tiny env completes its
# 5-step horizon: enough for the scoring path to return a reward.
_POLICY_A = render_tool_call(TOOL_FINAL, "def policy(state):\n    return (0.5, 0.0)  # marker A")
_POLICY_B = render_tool_call(TOOL_FINAL, "def policy(state):\n    return (0.5, 0.0)  # marker B")
_MAIN_BLOCK = render_tool_call(TOOL_PROBE, 'print("probe")')
_INVALID_BLOCK = """no tool call here, model output garbage"""
_NOT_A_CONTROLLER = render_tool_call(TOOL_FINAL, 'print("no policy defined")')


def test_a_controller_in_turn_one_ends_the_rollout(monkeypatch):
    """A final run in the first reply ends the task there; the second scripted
    reply is never asked for."""
    gen = _patch_generators(monkeypatch, [_POLICY_A, _POLICY_B])
    out = run_n_turn_rollout(
        model=None, tokenizer=None, template=TEMPLATE,
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is True
    assert "marker A" in out["code"] and "marker B" not in out["code"]
    assert gen.calls == 1
    assert len(out["transcript"]) == 2  # the brief and the one reply


def test_main_then_policy_uses_final_policy(monkeypatch):
    """The expected happy path: probe in turn 1, policy in turn 2."""
    _patch_generators(monkeypatch, [_MAIN_BLOCK, _POLICY_B])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template=TEMPLATE,
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is True
    assert "marker B" in out["code"]
    assert out["actions"] == ["probe_run", "final_run"]
    assert [m["role"] for m in out["transcript"]] == ["user", "assistant", "tool", "assistant"]


def test_probe_then_probe_is_rejected(monkeypatch):
    """The last reply has to hold the final run: two probe runs in a row leave
    the rollout without one."""
    _patch_generators(monkeypatch, [_MAIN_BLOCK, _MAIN_BLOCK])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template=TEMPLATE,
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is False
    assert out["code"] == ""
    assert "no final run" in out["error"]


def test_final_turn_invalid_is_rejected(monkeypatch):
    """The second reply holds no tool call → no final run, whatever the first was."""
    _patch_generators(monkeypatch, [_MAIN_BLOCK, _INVALID_BLOCK])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template=TEMPLATE,
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is False
    assert "no final run" in out["error"]


def test_final_run_with_an_invalid_controller_is_rejected(monkeypatch):
    _patch_generators(monkeypatch, [_NOT_A_CONTROLLER])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template=TEMPLATE,
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is False and out["code"] == ""
    assert "is not valid" in out["error"]


def test_one_reply_is_the_final_run(monkeypatch):
    """With one reply the model has only `final_run`."""
    _patch_generators(monkeypatch, [_POLICY_A])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template=TEMPLATE,
        task=_tiny_task(),
        n_turns=1,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is True
    assert "marker A" in out["code"]
    assert out["actions"] == ["final_run"]


def test_n_turns_zero_raises():
    with pytest.raises(ValueError):
        run_n_turn_rollout(
            model=None, tokenizer=None, template="x",
            task=_tiny_task(),
            n_turns=0,
            temperature=0.0, top_p=1.0, max_tokens=128,
            scoring_timeout=10, scoring_max_steps=10,
        )
