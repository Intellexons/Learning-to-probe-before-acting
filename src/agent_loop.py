"""The agent loop: how the model acts.

The model acts through tool calls in its own chat format, as any tool-using
agent does. There are two tools:

  probe_run(code)   runs the code on a probe batch, where the object `env` is
                    available; what the code prints comes back as the tool
                    response.
  final_run(code)   runs the controller, `def policy(state)`, on the scored
                    batch; the task ends.

Nothing in the brief says which to call or in what order: the tools exist, and
what the model does with them is the behaviour under study. The runtime does
what a tool runtime does: it acts on the calls of a reply in the order they
are written, the task ends at the first final run, and a call it cannot act on
gets a response that says why.

This module holds the tool definitions, the parser for the model's calls and
the renderer that writes a call the way the model would. Executing a call is
`src.multi_turn.execute_turn`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.runner import extract_code

TOOL_PROBE = "probe_run"
TOOL_FINAL = "final_run"


def _tool(name: str, description: str, code_description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": code_description}},
                "required": ["code"],
            },
        },
    }


PROBE_TOOL = _tool(
    TOOL_PROBE,
    "Run Python code on a probe batch; the object `env` is available to it. Returns what the code prints.",
    "Python source code.",
)
FINAL_TOOL = _tool(
    TOOL_FINAL,
    "Run the controller on the scored batch. This ends the task.",
    "Python source code that defines `def policy(state)`.",
)


def tools_for(total_turns: int) -> list[dict]:
    """The tools the model is offered. With a single reply there is no probe
    run to offer: the reply is the final run."""
    if total_turns < 1:
        raise ValueError(f"total_turns must be >= 1, got {total_turns}")
    return [PROBE_TOOL, FINAL_TOOL] if total_turns > 1 else [FINAL_TOOL]


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
    if name not in (TOOL_PROBE, TOOL_FINAL):
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
    acted on. A reply without any call gives an empty list."""
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


def render_tool_call(name: str, code: str) -> str:
    """A tool call as the model writes one (and as the chat template renders
    one). Used by hand-written solutions and tests, so they go through the
    same parser as the model's replies."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": {"code": code}}) + "\n</tool_call>"


def tool_message(text: str) -> dict:
    """The chat message that carries a tool response back to the model."""
    return {"role": "tool", "content": text}


def system_message_with_tools(tokenizer, tools: list[dict]) -> dict:
    """The system message the chat template writes when it is given `tools`,
    as an explicit message.

    For callers that hand a chat to code that renders it without tools (TRL's
    stock GRPOTrainer): a chat that starts with this message renders to the
    same text as the chat rendered with the tools, which is checked here.
    """
    probe = [{"role": "user", "content": "x"}]
    with_tools = tokenizer.apply_chat_template(probe, tools=tools, tokenize=False, add_generation_prompt=True)
    start, end = "<|im_start|>system\n", "<|im_end|>\n"
    if not with_tools.startswith(start) or end not in with_tools:
        raise RuntimeError("the chat template does not open with a ChatML system message")
    message = {"role": "system", "content": with_tools[len(start):with_tools.index(end)]}
    plain = tokenizer.apply_chat_template([message] + probe, tokenize=False, add_generation_prompt=True)
    if plain != with_tools:
        raise RuntimeError("the explicit system message does not reproduce the template's tool block")
    return message
