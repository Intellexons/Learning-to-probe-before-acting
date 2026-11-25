"""The agent loop: how the model acts.

The model works as a tool-using agent does, in its own chat format:

  - It has one tool, `probe_run(code)`: the code runs on a probe batch, where
    the object `env` is available, and what it prints comes back as the tool
    response.
  - A reply without a tool call is its final answer. The controller in it,
    `def policy(state)` in a Python code block, runs on the scored batch: the
    final run. The task ends.

This is the loop every tool runtime runs: act on the tool calls of a reply and
hand back the results; a reply that calls nothing is the answer. Nothing in
the brief says whether to probe or when: the tool exists, and what the model
does with it is the behaviour under study.

(A first version, 2026-11-25, made the final run a second tool, `final_run`.
With only that tool on offer the frozen model wrote its controller as an
ordinary answer with a code block in 9 of 12 replies and called nothing, so
the final answer became what it is for the model: the reply itself.)

This module holds the tool definition, the parser for the model's calls, the
reader for the controller in a final answer, and renderers that write a call
and an answer the way the model would. Acting on a reply is
`src.multi_turn.execute_turn`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.runner import extract_code

TOOL_PROBE = "probe_run"

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_PROBE,
        "description": "Run Python code on a probe batch; the object `env` is available to it. "
                       "Returns what the code prints.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code."}},
            "required": ["code"],
        },
    },
}


def tools_for(total_turns: int) -> list[dict] | None:
    """The tools the model is offered: the probe run when there is a reply to
    spend on it. With a single reply there is no tool, and the chat template
    writes no tool block: the reply is the final answer."""
    if total_turns < 1:
        raise ValueError(f"total_turns must be >= 1, got {total_turns}")
    return [PROBE_TOOL] if total_turns > 1 else None


@dataclass(frozen=True)
class ToolCall:
    """One tool call found in a reply. `error` is None when the call can be
    acted on; otherwise it says why not, and `name` holds the tool name if one
    could be read."""
    name: str | None
    code: str | None
    error: str | None = None


_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')
_FENCE_RE = re.compile(r"```\w*[ \t]*\n(.*?)```(?!\w)", re.DOTALL)


def _parse_call(body: str) -> ToolCall:
    body = body.strip()
    seen = _NAME_RE.search(body)
    name = seen.group(1) if seen else None
    try:
        # strict=False accepts a code string written with real line breaks
        obj = json.loads(body, strict=False)
    except json.JSONDecodeError as e:
        return ToolCall(name, None, f"the call is not valid JSON ({e.msg}, character {e.pos})")
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return ToolCall(None, None, 'the call has no "name"')
    name = obj["name"]
    if name != TOOL_PROBE:
        return ToolCall(name, None, f"there is no tool named {name!r}")
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False)
        except json.JSONDecodeError:
            args = None
    code = args.get("code") if isinstance(args, dict) else None
    if not isinstance(code, str) or not code.strip():
        return ToolCall(name, None, f'{name} needs "arguments": {{"code": "<python source>"}}')
    # a code string that carries its own Markdown fence is still just code
    return ToolCall(name, extract_code(code), None)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """The tool calls of a reply, in the order they are written. A call the
    token budget cut before `</tool_call>` is returned as one that cannot be
    acted on. A reply without any call gives an empty list: it is the final
    answer."""
    text = text or ""
    calls = []
    end = 0
    for m in _CALL_RE.finditer(text):
        calls.append(_parse_call(m.group(1)))
        end = m.end()
    rest = text[end:]
    if "<tool_call>" in rest:
        seen = _NAME_RE.search(rest)
        calls.append(ToolCall(seen.group(1) if seen else None, None,
                              "the call was cut off before </tool_call>"))
    return calls


def controller_from_answer(text: str) -> str:
    """The controller in a final answer.

    The answer is prose around a Python code block. If it holds several
    blocks, the controller is the last one that defines `policy`; if none
    does, or the answer is one block (closed or cut off by the token budget)
    or bare code, `runner.extract_code` reads it as before. What comes back
    is validated and scored by the runner, so an answer without a controller
    scores the error floor there.
    """
    text = text or ""
    with_policy = [b for b in _FENCE_RE.findall(text) if re.search(r"^def policy\b", b, re.MULTILINE)]
    if with_policy:
        return with_policy[-1].strip()
    return extract_code(text)


def render_tool_call(name: str, code: str) -> str:
    """A tool call as the model writes one (and as the chat template renders
    one). Used by hand-written solutions and tests, so they go through the
    same parser as the model's replies."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": {"code": code}}) + "\n</tool_call>"


def render_final_answer(code: str) -> str:
    """A final answer as the model writes one: the controller in a Python
    code block. Used by hand-written solutions and tests."""
    return "```python\n" + code.strip("\n") + "\n```"


def tool_message(text: str) -> dict:
    """The chat message that carries a tool response back to the model."""
    return {"role": "tool", "content": text}
