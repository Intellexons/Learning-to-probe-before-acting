"""Tests for the multi-turn-aware render_prompt path.

Goals:
1. Default args produce a prompt byte-identical to the pre-multi-turn
   single-turn prompt, so existing single-turn pipelines (probe, generate,
   train, eval) are not behaviorally affected.
2. total_turns >= 2 produces a prompt with a Multi-Turn Protocol section
   that mentions the turn counter, the mode-fence rules, and the
   last-turn-must-be-policy clause.
3. Edge cases: turn_idx out of range raises; final-turn rendering
   includes the FINAL-turn marker.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompts import load_template, render_prompt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "configs" / "prompt_template.txt"


def _toy_task() -> dict:
    return {
        "env_code": "class FakeEnv: pass",
        "state_spec": "- px (float): x position",
        "action_spec": "(ax, ay) — 2-tuple of floats",
        "goal": "drive the puck to the goal.",
    }


def test_default_args_match_single_turn_layout():
    """No `## Multi-Turn Protocol` section should appear when total_turns=1."""
    template = load_template(str(TEMPLATE_PATH))
    rendered = render_prompt(template, _toy_task())
    assert "## Multi-Turn Protocol" not in rendered
    # Single-turn must keep the original "no fences" instruction so the
    # existing extract_code path (raw def policy) keeps working.
    assert "Do NOT include markdown fences" in rendered


def test_default_args_match_explicit_single_turn():
    """render_prompt() == render_prompt(total_turns=1, turn_idx=1)."""
    template = load_template(str(TEMPLATE_PATH))
    a = render_prompt(template, _toy_task())
    b = render_prompt(template, _toy_task(), turn_idx=1, total_turns=1)
    assert a == b


def test_multi_turn_first_turn_prompt():
    template = load_template(str(TEMPLATE_PATH))
    rendered = render_prompt(template, _toy_task(), turn_idx=1, total_turns=2)
    assert "## Multi-Turn Protocol" in rendered
    assert "Turn 1 of 2" in rendered
    assert "1 turn remains" in rendered
    # First turn (not last) — not the FINAL marker
    assert "FINAL turn" not in rendered
    # Both mode directives available on a non-final turn
    assert "# MODE: policy" in rendered
    assert "# MODE: main" in rendered
    # last-turn-must-be-policy clause is always visible in the protocol
    assert "final turn (turn 2) MUST" in rendered


def test_multi_turn_final_turn_prompt():
    template = load_template(str(TEMPLATE_PATH))
    rendered = render_prompt(template, _toy_task(), turn_idx=2, total_turns=2)
    assert "FINAL turn" in rendered
    assert "Turn 2 of 2" in rendered
    # Final turn requires policy block specifically — the response_format
    # narrows to policy-only.
    final_section = rendered.split("## Response Format", 1)[1]
    assert "# MODE: policy" in final_section
    assert "# MODE: main" not in final_section


def test_multi_turn_three_turns_middle():
    template = load_template(str(TEMPLATE_PATH))
    rendered = render_prompt(template, _toy_task(), turn_idx=2, total_turns=3)
    assert "Turn 2 of 3" in rendered
    assert "1 turn remains" in rendered
    assert "FINAL turn" not in rendered


def test_multi_turn_three_turns_first():
    template = load_template(str(TEMPLATE_PATH))
    rendered = render_prompt(template, _toy_task(), turn_idx=1, total_turns=3)
    assert "Turn 1 of 3" in rendered
    assert "2 turns remain" in rendered


def test_turn_idx_out_of_range_raises():
    template = load_template(str(TEMPLATE_PATH))
    with pytest.raises(ValueError):
        render_prompt(template, _toy_task(), turn_idx=3, total_turns=2)
    with pytest.raises(ValueError):
        render_prompt(template, _toy_task(), turn_idx=0, total_turns=2)


def test_task_fields_are_substituted():
    template = load_template(str(TEMPLATE_PATH))
    rendered = render_prompt(template, _toy_task())
    assert "class FakeEnv: pass" in rendered
    assert "drive the puck to the goal." in rendered
    assert "(ax, ay) — 2-tuple of floats" in rendered
