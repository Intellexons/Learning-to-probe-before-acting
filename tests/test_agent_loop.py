"""The agent loop's interface: the tool, the parser for the model's calls, and
the reader for the controller in a final answer.

The model calls `probe_run` in its own chat format; a reply without a tool
call is its final answer. These tests pin what counts as a call that can be
acted on, what does not and why, how the controller is read from an answer,
and that what the renderers write reads back as itself.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent_loop import (  # noqa: E402
    PROBE_TOOL,
    TOOL_PROBE,
    controller_from_answer,
    parse_tool_calls,
    render_final_answer,
    render_tool_call,
    tool_message,
    tools_for,
)

CONTROLLER = "def policy(state):\n    return 0.0"
CODE = 'state, info = env.reset()\nprint("sigma", state["sigma"])'


def test_more_than_one_reply_offers_the_probe_run_one_reply_offers_no_tool():
    assert tools_for(2) == [PROBE_TOOL]
    assert tools_for(3) == [PROBE_TOOL]
    assert tools_for(1) is None
    with pytest.raises(ValueError):
        tools_for(0)


def test_the_tool_definition_says_what_exists_and_nothing_about_order():
    fn = PROBE_TOOL["function"]
    assert PROBE_TOOL["type"] == "function" and fn["parameters"]["required"] == ["code"]
    assert fn["name"] == TOOL_PROBE == "probe_run"
    text = json.dumps(PROBE_TOOL).lower()
    for word in ("first", "before", "should", "must", "then"):
        assert word not in text, f"the definition steers ({word!r})"


def test_a_rendered_call_reads_back_as_itself():
    (call,) = parse_tool_calls(render_tool_call(TOOL_PROBE, CODE))
    assert (call.name, call.code, call.error) == (TOOL_PROBE, CODE, None)


def test_text_around_a_call_is_ignored():
    text = "Let me look first.\n" + render_tool_call(TOOL_PROBE, CODE) + "\nThat should do."
    (call,) = parse_tool_calls(text)
    assert call.name == TOOL_PROBE and call.error is None


def test_calls_come_back_in_the_order_written():
    text = render_tool_call(TOOL_PROBE, "print(1)") + "\n" + render_tool_call(TOOL_PROBE, "print(2)")
    assert [c.code for c in parse_tool_calls(text)] == ["print(1)", "print(2)"]


def test_a_code_string_with_real_line_breaks_is_accepted():
    text = '<tool_call>\n{"name": "probe_run", "arguments": {"code": "x = 1\nprint(x)"}}\n</tool_call>'
    (call,) = parse_tool_calls(text)
    assert call.error is None and call.code == "x = 1\nprint(x)"


def test_arguments_given_as_a_json_string_are_accepted():
    body = json.dumps({"name": TOOL_PROBE, "arguments": json.dumps({"code": "print(1)"})})
    (call,) = parse_tool_calls(f"<tool_call>\n{body}\n</tool_call>")
    assert call.error is None and call.code == "print(1)"


def test_a_markdown_fence_inside_the_code_string_is_dropped():
    (call,) = parse_tool_calls(render_tool_call(TOOL_PROBE, "```python\nprint(1)\n```"))
    assert call.error is None and call.code == "print(1)"


def test_no_call_is_an_empty_list():
    """A reply without a tool call is the final answer."""
    assert parse_tool_calls("") == []
    assert parse_tool_calls(None) == []
    assert parse_tool_calls(render_final_answer(CONTROLLER)) == []


def test_a_call_cut_off_by_the_token_budget_cannot_be_acted_on():
    text = '<tool_call>\n{"name": "probe_run", "arguments": {"code": "state, info = env.reset()\\nfor'
    (call,) = parse_tool_calls(text)
    assert call.name == TOOL_PROBE and call.code is None
    assert "cut off" in call.error


def test_a_complete_call_before_a_cut_off_one_is_kept():
    text = render_tool_call(TOOL_PROBE, "print(1)") + '\n<tool_call>\n{"name": "probe_run", "argu'
    first, second = parse_tool_calls(text)
    assert first.error is None and first.name == TOOL_PROBE
    assert second.name == TOOL_PROBE and "cut off" in second.error


@pytest.mark.parametrize("body, reason", [
    ('{"name": "probe_run", "arguments": {"code": "print("hi")"}}', "not valid JSON"),
    ('{"name": "final_run", "arguments": {"code": "def policy(state):\\n    return 0.0"}}', "no tool named 'final_run'"),
    ('{"arguments": {"code": "print(1)"}}', 'no "name"'),
    ('{"name": "run_python", "arguments": {"code": "print(1)"}}', "no tool named 'run_python'"),
    ('{"name": "probe_run", "arguments": {}}', "probe_run needs"),
    ('{"name": "probe_run", "arguments": {"code": "   "}}', "probe_run needs"),
    ('{"name": "probe_run"}', "probe_run needs"),
    ('[1, 2]', 'no "name"'),
])
def test_calls_that_cannot_be_acted_on_say_why(body, reason):
    (call,) = parse_tool_calls(f"<tool_call>\n{body}\n</tool_call>")
    assert call.code is None and reason in call.error


def test_the_tool_name_is_kept_when_the_json_is_broken():
    (call,) = parse_tool_calls('<tool_call>\n{"name": "probe_run", "arguments": {"code": "print("hi")"}}\n</tool_call>')
    assert call.name == TOOL_PROBE


def test_tool_message_has_the_tool_role():
    assert tool_message("LIMIT=0.18") == {"role": "tool", "content": "LIMIT=0.18"}


# ---------------------------------------------------------------------------
# The controller in a final answer
# ---------------------------------------------------------------------------


def test_a_rendered_answer_reads_back_as_the_controller():
    assert controller_from_answer(render_final_answer(CONTROLLER)) == CONTROLLER


def test_prose_around_the_code_block_is_ignored():
    text = "Here is my controller.\n\n" + render_final_answer(CONTROLLER) + "\n\nIt holds the batch steady."
    assert controller_from_answer(text) == CONTROLLER


def test_with_several_blocks_the_controller_is_the_last_that_defines_policy():
    sketch = "```python\nrate = 0.5  # degC per step at full cooling\n```"
    first = render_final_answer("def policy(state):\n    return -1.0  # draft")
    text = sketch + "\nA draft:\n" + first + "\nThe controller:\n" + render_final_answer(CONTROLLER)
    assert controller_from_answer(text) == CONTROLLER


def test_a_block_cut_off_by_the_token_budget_is_read_as_far_as_it_goes():
    text = "My controller:\n```python\ndef policy(state):\n    sigma = state['sigma']\n    if sigma >"
    assert controller_from_answer(text).startswith("def policy(state):") and controller_from_answer(text).endswith("if sigma >")


def test_bare_code_and_answers_without_code():
    assert controller_from_answer(CONTROLLER) == CONTROLLER
    assert controller_from_answer("I need more information before I can answer.") == "I need more information before I can answer."
    assert controller_from_answer("") == ""
    assert controller_from_answer(None) == ""
