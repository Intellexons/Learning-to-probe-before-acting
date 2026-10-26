"""The brief: what the model is told, and nothing else.

The prompt is a briefing (goal and physics in words, what is observed, what is
not, what the probe run is for, how the final run is scored, the format). No
environment source, no format example, no mode directives. Turn kinds are
positional, so turn 1 of a two-turn rollout asks for probe code and the last
turn asks for the controller.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompts import compose_turn_message, load_template, render_prompt  # noqa: E402
from src.runner import MAIN_DEFAULT_TIMEOUT, MAIN_STDOUT_TRUNCATE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "configs" / "prompt_template.txt"


def _toy_task() -> dict:
    return {
        "env_code": "class FakeEnv: pass",
        "intro": "You run a toy cart.",
        "goal": "drive the cart to the goal.",
        "state_spec": "- px (float): x position",
        "action_spec": "(ax, ay) — 2-tuple of floats",
        "unobserved": "the goal position wobbles.",
        "probe_spec": "the same cart, a different start",
        "final_spec": "the same cart as your probe run",
    }


def _render(**kw) -> str:
    return render_prompt(load_template(str(TEMPLATE_PATH)), _toy_task(), **kw)


def test_single_turn_is_the_brief_asking_for_the_controller():
    rendered = _render()
    assert rendered.startswith("You run a toy cart.")
    assert "## Goal" in rendered and "drive the cart to the goal." in rendered
    assert "## What you observe" in rendered and "- px (float): x position" in rendered
    assert "(ax, ay) — 2-tuple of floats" in rendered
    assert "## What you do not observe" in rendered and "the goal position wobbles." in rendered
    assert "## The final run (this message)" in rendered
    assert "## Your probe run" not in rendered
    assert rendered.rstrip().endswith("the controller, defining `def policy(state)`.")


def test_default_args_match_explicit_single_turn():
    assert _render() == _render(turn_idx=1, total_turns=1)


def test_the_env_source_is_not_in_the_prompt():
    for kw in ({}, {"turn_idx": 1, "total_turns": 2}, {"turn_idx": 2, "total_turns": 2}):
        assert "class FakeEnv" not in _render(**kw)


def test_no_mode_directives_or_example_anywhere():
    for kw in ({}, {"turn_idx": 1, "total_turns": 2}, {"turn_idx": 2, "total_turns": 2}):
        rendered = _render(**kw)
        assert "# MODE" not in rendered and "## Example" not in rendered


def test_two_turn_first_turn_asks_for_probe_code_and_states_the_interface():
    rendered = _render(turn_idx=1, total_turns=2)
    assert "## Your probe run (this message)" in rendered
    assert "the same cart, a different start" in rendered
    for line in ("env.reset()", "env.step(action)", "env.failure_reason", "env.showered, env.done"):
        assert line in rendered
    assert f"within {MAIN_DEFAULT_TIMEOUT} seconds" in rendered
    assert f"beyond {MAIN_STDOUT_TRUNCATE} characters" in rendered
    assert "## The final run (the last message)" in rendered
    assert rendered.rstrip().endswith("the probe code.")


def test_two_turn_final_turn_is_the_short_instruction():
    rendered = _render(turn_idx=2, total_turns=2)
    assert rendered.startswith("Now the final run.")
    assert "def policy(state)" in rendered
    assert "the same cart as your probe run" in rendered
    assert "## Goal" not in rendered  # the brief is already in the chat history


def test_compose_turn_message_puts_the_output_first():
    msg = compose_turn_message(_render(turn_idx=2, total_turns=2), "LIMIT=0.18")
    assert msg.startswith("Output of your probe run:\nLIMIT=0.18\n\n")
    assert msg.endswith(_render(turn_idx=2, total_turns=2))


def test_three_turns_middle_turn_offers_another_probe():
    rendered = _render(turn_idx=2, total_turns=3)
    assert "another probe" in rendered and "the probe code." in rendered
    assert "def policy" not in rendered
    first = _render(turn_idx=1, total_turns=3)
    assert "## Your probe run (this message and the next 1)" in first


def test_turn_idx_out_of_range_raises():
    with pytest.raises(ValueError):
        _render(turn_idx=3, total_turns=2)
    with pytest.raises(ValueError):
        _render(turn_idx=0, total_turns=2)
    with pytest.raises(ValueError):
        _render(turn_idx=1, total_turns=0)


def test_optional_fields_have_defaults():
    task = {"goal": "g", "state_spec": "s", "action_spec": "a"}
    rendered = render_prompt(load_template(str(TEMPLATE_PATH)), task, turn_idx=1, total_turns=2)
    assert "## What you do not observe" not in rendered
    assert "the same environment, with the same hidden parameters" in rendered
    assert rendered.startswith("Write a Python controller for the process below.")
