"""The tokens between two replies, as the trainer conditions on them.

`prompt_ids + turn1_ids + between_reply_ids(...)` has to be exactly the token
prefix the chat template gives for the chat up to the second reply: it is
what the second reply is sampled from and what the loss conditions on. Needs
the model's tokenizer on disk; skipped where it is not.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
TOKENIZER_DIR = REPO_ROOT / "models" / "hf_base" / "iter_000"
pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(),
                                reason="the model's tokenizer is not on disk")

from src.agent_loop import TOOL_PROBE, render_tool_call, tool_message, tools_for  # noqa: E402

PROMPT = [{"role": "user", "content": "the brief"}]
REPLY = "I will look.\n" + render_tool_call(TOOL_PROBE, 'state, info = env.reset()\nprint("sigma", state["sigma"])')


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def _ids(tok, messages, tools):
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    return tok(text).input_ids


@pytest.mark.parametrize("following", [
    (tool_message("LIMIT=0.18\n"),),
    (tool_message("one\n"), tool_message("[tool call not acted on: there is no tool named 'x']")),
])
@pytest.mark.parametrize("model_ended_its_reply", [True, False])
def test_prefix_of_the_second_reply_is_rebuilt_exactly(tok, following, model_ended_its_reply):
    from src.branched_trainer import between_reply_ids
    tools = tools_for(2)
    prompt_ids = _ids(tok, PROMPT, tools)
    turn1_ids = tok(REPLY, add_special_tokens=False).input_ids
    if model_ended_its_reply:
        turn1_ids = turn1_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    between = between_reply_ids(tok, tools, PROMPT, REPLY, turn1_ids, following)
    expected = _ids(tok, PROMPT + [{"role": "assistant", "content": REPLY}] + list(following), tools)
    assert prompt_ids + turn1_ids + between == expected
    # a reply the token budget cut gets its end-of-turn marker from the template, not from the model
    assert (between[0] == tok.convert_tokens_to_ids("<|im_end|>")) is (not model_ended_its_reply)


def test_the_tool_block_is_in_the_prompt_and_calls_survive_decoding(tok):
    text = tok.apply_chat_template(PROMPT, tools=tools_for(2), tokenize=False, add_generation_prompt=True)
    assert '"name": "probe_run"' in text and "final_run" not in text
    # with one reply there is no tool, and the template writes no tool block
    one = tok.apply_chat_template(PROMPT, tools=tools_for(1), tokenize=False, add_generation_prompt=True)
    assert "# Tools" not in one and "probe_run" not in one
    call = render_tool_call(TOOL_PROBE, 'print("sigma", state["sigma"])')
    assert tok.decode(tok(call, add_special_tokens=False).input_ids, skip_special_tokens=True) == call
