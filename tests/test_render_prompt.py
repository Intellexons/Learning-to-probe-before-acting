"""The brief: what the model is told, and nothing else.

The brief states what exists: the goal and the physics in words, what is
observed, what is not, what a probe run is, what the final run is, and how
many replies the model has. It carries no environment source, no example, and
no sentence about what to do first.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompts import load_template, render_prompt  # noqa: E402
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
        "earlier_batches": "Run A: step 1 x=0.1\nstep 2 x=0.3 crashed",
        "probe_spec": "the same cart, a different start",
    }


def _render(**kw) -> str:
    return render_prompt(load_template(str(TEMPLATE_PATH)), _toy_task(), **kw)


def test_one_reply_is_the_brief_with_the_final_run_only():
    rendered = _render()
    assert rendered.startswith("You run a toy cart.")
    assert "## Goal" in rendered and "drive the cart to the goal." in rendered
    assert "## What you observe" in rendered and "- px (float): x position" in rendered
    assert "(ax, ay) — 2-tuple of floats" in rendered
    assert "## What you do not observe" in rendered and "the goal position wobbles." in rendered
    assert "## What you know from earlier batches" in rendered and "step 2 x=0.3 crashed" in rendered
    assert "## The final run" in rendered
    assert "Your final answer is the controller: a Python code block that defines `def policy(state)`." in rendered
    assert "## The probe run" not in rendered and "probe_run" not in rendered and "env" not in rendered.split("## The final run")[1]
    assert "tool" not in rendered
    assert rendered.rstrip().endswith("You have one reply: your final answer.")


def test_default_is_one_reply():
    assert _render() == _render(total_turns=1)


def test_the_env_source_is_not_in_the_brief():
    for n in (1, 2, 3):
        assert "class FakeEnv" not in _render(total_turns=n)


def test_no_mode_directives_example_or_reply_format():
    for n in (1, 2):
        rendered = _render(total_turns=n)
        assert "# MODE" not in rendered and "## Example" not in rendered
        assert "## Format" not in rendered and "Reply with" not in rendered


def test_two_replies_describe_the_probe_run_and_the_final_run():
    rendered = _render(total_turns=2)
    assert "## The probe run" in rendered
    assert "`probe_run` runs your code on the same cart, a different start." in rendered
    for line in ("env.reset()", "env.step(action)", "env.failure_reason", "env.showered, env.done"):
        assert line in rendered
    # the flags are attributes of env, not state keys: the sentence that stops the KeyError
    assert "`done` is returned by `env.step`; `env.done`, `env.showered` and `env.failure_reason` are attributes of `env`, not keys of `state`." in rendered
    assert f"as fit in {MAIN_DEFAULT_TIMEOUT} seconds" in rendered
    assert f"up to {MAIN_STDOUT_TRUNCATE} characters" in rendered
    assert "## The final run" in rendered and "It has no access to `env`." in rendered
    assert "## What you know from earlier batches" in rendered
    assert rendered.rstrip().endswith(
        "You have 2 replies. A reply that calls `probe_run` is a probe run; a reply without a tool call "
        "is your final answer. The task ends with your final answer, or when your replies run out.")


def test_the_brief_says_nothing_about_what_to_do_first():
    """The tools exist; whether and when to call them is the behaviour under
    study, so the sections that describe them must not steer."""
    rendered = _render(total_turns=2)
    tail = rendered[rendered.index("## The probe run"):].lower()
    for phrase in ("first", "before", "you should", "you must", "you may", "optional", "then ", "afterwards", "start by"):
        assert phrase not in tail.replace("from its first step", ""), f"the brief steers: {phrase!r}"


def test_more_replies_change_only_the_count():
    assert _render(total_turns=3) == _render(total_turns=2).replace("You have 2 replies", "You have 3 replies")


def test_zero_replies_raises():
    with pytest.raises(ValueError):
        _render(total_turns=0)


def test_optional_fields_have_defaults():
    task = {"goal": "g", "state_spec": "s", "action_spec": "a"}
    rendered = render_prompt(load_template(str(TEMPLATE_PATH)), task, total_turns=2)
    assert "## What you do not observe" not in rendered
    assert "## What you know from earlier batches" not in rendered
    assert "the same environment, with the same hidden parameters" in rendered
    assert rendered.startswith("Write a Python controller for the process below.")
