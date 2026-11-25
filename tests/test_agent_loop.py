"""The agent loop's interface: the two tools and the parser for the model's calls.

The model acts through tool calls in its own chat format. These tests pin what
counts as a call that can be acted on, what does not and why, and that a call
written by `render_tool_call` reads back as itself.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent_loop import (  # noqa: E402
    FINAL_TOOL,
    PROBE_TOOL,
    TOOL_FINAL,
    TOOL_PROBE,
    parse_tool_calls,
    render_tool_call,
    tool_message,
    tools_for,
)

CODE = 'state, info = env.reset()\nprint("sigma", state["sigma"])'


def test_two_replies_offer_both_tools_one_reply_only_the_final_run():
    assert tools_for(2) == [PROBE_TOOL, FINAL_TOOL]
    assert tools_for(3) == [PROBE_TOOL, FINAL_TOOL]
    assert tools_for(1) == [FINAL_TOOL]
    with pytest.raises(ValueError):
        tools_for(0)


def test_tool_definitions_say_what_exists_and_nothing_about_order():
    for tool in (PROBE_TOOL, FINAL_TOOL):
        fn = tool["function"]
        assert tool["type"] == "function" and fn["parameters"]["required"] == ["code"]
        text = json.dumps(tool).lower()
        for word in ("first", "before", "should", "must", "then"):
            assert word not in text, f"{fn['name']}: the definition steers ({word!r})"
    assert PROBE_TOOL["function"]["name"] == TOOL_PROBE == "probe_run"
    assert FINAL_TOOL["function"]["name"] == TOOL_FINAL == "final_run"


def test_a_rendered_call_reads_back_as_itself():
    for name in (TOOL_PROBE, TOOL_FINAL):
        (call,) = parse_tool_calls(render_tool_call(name, CODE))
        assert (call.name, call.code, call.error) == (name, CODE, None)


def test_text_around_a_call_is_ignored():
    text = "Let me look first.\n" + render_tool_call(TOOL_PROBE, CODE) + "\nThat should do."
    (call,) = parse_tool_calls(text)
    assert call.name == TOOL_PROBE and call.error is None


def test_calls_come_back_in_the_order_written():
    text = render_tool_call(TOOL_PROBE, "print(1)") + "\n" + render_tool_call(TOOL_FINAL, "def policy(state):\n    return 0.0")
    assert [c.name for c in parse_tool_calls(text)] == [TOOL_PROBE, TOOL_FINAL]


def test_a_code_string_with_real_line_breaks_is_accepted():
    text = '<tool_call>\n{"name": "probe_run", "arguments": {"code": "x = 1\nprint(x)"}}\n</tool_call>'
    (call,) = parse_tool_calls(text)
    assert call.error is None and call.code == "x = 1\nprint(x)"


def test_arguments_given_as_a_json_string_are_accepted():
    body = json.dumps({"name": TOOL_FINAL, "arguments": json.dumps({"code": "def policy(state):\n    return 0.0"})})
    (call,) = parse_tool_calls(f"<tool_call>\n{body}\n</tool_call>")
    assert call.error is None and call.code.startswith("def policy")


def test_a_markdown_fence_inside_the_code_string_is_dropped():
    (call,) = parse_tool_calls(render_tool_call(TOOL_PROBE, "```python\nprint(1)\n```"))
    assert call.error is None and call.code == "print(1)"


def test_no_call_is_an_empty_list():
    assert parse_tool_calls("") == []
    assert parse_tool_calls(None) == []
    assert parse_tool_calls("```python\ndef policy(state):\n    return 0.0\n```") == []


def test_a_call_cut_off_by_the_token_budget_cannot_be_acted_on():
    text = '<tool_call>\n{"name": "final_run", "arguments": {"code": "def policy(state):\\n    return'
    (call,) = parse_tool_calls(text)
    assert call.name == TOOL_FINAL and call.code is None
    assert "cut off" in call.error


def test_a_complete_call_before_a_cut_off_one_is_kept():
    text = render_tool_call(TOOL_PROBE, "print(1)") + '\n<tool_call>\n{"name": "final_run", "argu'
    first, second = parse_tool_calls(text)
    assert first.error is None and first.name == TOOL_PROBE
    assert second.name == TOOL_FINAL and "cut off" in second.error


@pytest.mark.parametrize("body, reason", [
    ('{"name": "final_run", "arguments": {"code": "print("hi")"}}', "not valid JSON"),
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
    (call,) = parse_tool_calls('<tool_call>\n{"name": "final_run", "arguments": {"code": "print("hi")"}}\n</tool_call>')
    assert call.name == TOOL_FINAL


def test_tool_message_has_the_tool_role():
    assert tool_message("LIMIT=0.18") == {"role": "tool", "content": "LIMIT=0.18"}
