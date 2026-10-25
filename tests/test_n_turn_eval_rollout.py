"""Last-policy-wins semantics for `08_eval.py:run_n_turn_rollout`.

Mocks the generator so the rollout logic can be exercised on CPU. The
training side is already covered by `test_multi_turn.py`; here we lock
down the eval-specific contract:

  - Never short-circuit on intermediate `# MODE: policy`.
  - Among the policies emitted, the LAST one wins.
  - If the final turn doesn't emit a valid policy, reject (error floor)
    even when an earlier turn did — matching the training-side
    `is_last_turn=True` rule.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
    """Drop-in stand-in for generate_code / generate_code_from_messages.

    Returns successive scripted completions; the rollout doesn't care
    what the prompt looks like, only what the next assistant turn says.
    Mimics `generate_code` by running `extract_code` on the raw text so
    the single-turn legacy path receives a populated `code` field.
    """

    def __init__(self, completions: list[str]):
        self._queue = list(completions)
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        from src.hf_generator import extract_code
        self.calls += 1
        if not self._queue:
            return {
                "code": "",
                "success": False,
                "error": "scripted generator exhausted",
                "raw_response": "",
            }
        text = self._queue.pop(0)
        return {
            "code": extract_code(text),
            "success": True,
            "error": None,
            "raw_response": text,
        }


def _patch_generators(monkeypatch, completions: list[str]):
    gen = _ScriptedGenerator(completions)
    monkeypatch.setattr(_eval_mod, "generate_code", gen)
    monkeypatch.setattr(_eval_mod, "generate_code_from_messages", gen)
    return gen


# All of the rollout's downstream actions need an `env.step(...)` to
# succeed, but our scripted assistant turns include only valid syntax;
# `validate_code` passes and `score_sample` runs `policy(state)` against
# `TinyEnv`. The policies below all return `(0.5, 0.0)` so the env
# completes its 5-step horizon — sufficient for the scoring path to
# return a non-None reward and exercise the eval's accept/reject branch.
_POLICY_A = """```python
# MODE: policy
def policy(state):
    return (0.5, 0.0)  # marker A
```"""

_POLICY_B = """```python
# MODE: policy
def policy(state):
    return (0.5, 0.0)  # marker B
```"""

_MAIN_BLOCK = """```python
# MODE: main
print("probe")
```"""

_INVALID_BLOCK = """no fenced block here, model output garbage"""


def test_two_turns_both_policy_picks_the_second(monkeypatch):
    """User-stated invariant: 'we don't stop at the first policy.'"""
    _patch_generators(monkeypatch, [_POLICY_A, _POLICY_B])
    out = run_n_turn_rollout(
        model=None, tokenizer=None, template="{multi_turn_protocol}{state_spec}{action_spec}{goal}{env_section}{multi_turn_example}{requirements}{response_format}",  # template body unused — render_prompt sees a dummy task
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is True
    assert "marker B" in out["code"], (
        "expected the SECOND policy to win, but got first-turn code"
    )
    assert "marker A" not in out["code"]


def test_main_then_policy_uses_final_policy(monkeypatch):
    """The expected happy path: probe in turn 1, policy in turn 2."""
    _patch_generators(monkeypatch, [_MAIN_BLOCK, _POLICY_B])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template="{multi_turn_protocol}{state_spec}{action_spec}{goal}{env_section}{multi_turn_example}{requirements}{response_format}",
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is True
    assert "marker B" in out["code"]


def test_policy_then_main_is_rejected(monkeypatch):
    """Final turn must be policy (training-side rule). Even though turn 1
    emitted a valid policy, the rollout is rejected because the final
    turn was main."""
    _patch_generators(monkeypatch, [_POLICY_A, _MAIN_BLOCK])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template="{multi_turn_protocol}{state_spec}{action_spec}{goal}{env_section}{multi_turn_example}{requirements}{response_format}",
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is False
    assert out["code"] == ""
    assert "main" in out["error"]


def test_final_turn_invalid_is_rejected(monkeypatch):
    """Turn 2 emits no recognized fence → reject, regardless of turn 1."""
    _patch_generators(monkeypatch, [_POLICY_A, _INVALID_BLOCK])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template="{multi_turn_protocol}{state_spec}{action_spec}{goal}{env_section}{multi_turn_example}{requirements}{response_format}",
        task=_tiny_task(),
        n_turns=2,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    assert out["success"] is False
    assert "invalid" in out["error"]


def test_n_turns_one_path_unchanged(monkeypatch):
    """Single-turn legacy path still produces the expected success
    result (back-compat for DPO-loop callers)."""
    _patch_generators(monkeypatch, [_POLICY_A])
    out = run_n_turn_rollout(
        model=None, tokenizer=None,
        template="{multi_turn_protocol}{state_spec}{action_spec}{goal}{env_section}{multi_turn_example}{requirements}{response_format}",
        task=_tiny_task(),
        n_turns=1,
        temperature=0.0, top_p=1.0, max_tokens=128,
        scoring_timeout=10, scoring_max_steps=10,
    )
    # generate_code returns the scripted text in raw_response and the
    # code-extracting `extract_code` runs separately; for n_turns=1 the
    # rollout passes through the result unmodified, so success follows
    # whatever the underlying generate_code returned.
    assert out["success"] is True
    # The legacy path's `code` is whatever extract_code() pulled. It
    # contains the policy body since the script wraps it in a fence.
    assert "marker A" in out["code"] or "def policy" in out["code"]


def test_n_turns_zero_raises():
    with pytest.raises(ValueError):
        run_n_turn_rollout(
            model=None, tokenizer=None, template="x",
            task=_tiny_task(),
            n_turns=0,
            temperature=0.0, top_p=1.0, max_tokens=128,
            scoring_timeout=10, scoring_max_steps=10,
        )
