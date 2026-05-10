from pathlib import Path


_SINGLE_TURN_RESPONSE = (
    "Respond ONLY with valid Python code. Do NOT include markdown fences, "
    "explanations, or any text besides the code itself."
)


def _build_multi_turn_protocol(turn_idx: int, total_turns: int) -> str:
    """Build the protocol section that explains the multi-turn contract.

    Returns an empty string for single-turn (total_turns == 1) so the prompt
    layout is byte-identical to the pre-multi-turn template — existing
    single-turn runs and probes are unaffected.
    """
    if total_turns <= 1:
        return ""

    if turn_idx < 1 or turn_idx > total_turns:
        raise ValueError(
            f"turn_idx={turn_idx} out of range for total_turns={total_turns}"
        )

    remaining_after = total_turns - turn_idx
    if remaining_after == 0:
        budget_note = "**FINAL turn — must be `policy`.**"
    elif remaining_after == 1:
        budget_note = "1 turn remains after this."
    else:
        budget_note = f"{remaining_after} turns remain after this."

    return f"""
## Multi-Turn Protocol

Turn {turn_idx} of {total_turns}. {budget_note}

Each turn must output ONE fenced code block tagged either:
- ```` ```policy ```` — defines `def policy(state)`. Tool returns total_reward across all scenarios.
- ```` ```main ```` — arbitrary Python with no env access. Tool returns stdout. Use for derivations, not probing.

The final turn (turn {total_turns}) MUST be ```` ```policy ```` — a `main` final turn is rejected at the error floor. Any turn missing a recognized fence tag is also rejected.
"""


def _build_response_format(turn_idx: int, total_turns: int) -> str:
    """Build the `## Response Format` section that closes the prompt.

    Single-turn keeps the original "no fences" instruction so `extract_code`'s
    existing path (raw def policy + python-tagged fence) keeps working.
    Multi-turn uses tagged fences so the runner can dispatch on mode.
    """
    if total_turns <= 1:
        return _SINGLE_TURN_RESPONSE

    is_last = (turn_idx == total_turns)
    if is_last:
        body = (
            "Output exactly one fenced code block tagged ```` ```policy ````.\n"
            "Inside the fence, define `def policy(state)`."
        )
    else:
        body = (
            "Output exactly one fenced code block tagged either ```` ```policy ```` "
            "(define `def policy(state)`) or ```` ```main ```` (arbitrary code; "
            "stdout is returned to you).\n"
            "Do not output any text outside the single fenced block."
        )
    return f"## Response Format\n\n{body}"


def _build_requirements(turn_idx: int, total_turns: int) -> str:
    """Build the Requirements bullet list.

    Single-turn and the final turn of a multi-turn sequence both require a
    def policy. Non-final turns can output either main or policy, so the
    requirement to define a policy function is conditional rather than absolute
    — otherwise the model skips the scratchpad turn and jumps straight to code.
    """
    common = (
        "- Only import `math` and `random`. No global variables or state between calls.\n"
        "- Only read the state keys listed above."
    )
    if total_turns <= 1 or turn_idx == total_turns:
        return (
            "- Define exactly one function `def policy(state)` returning a valid action.\n"
            + common
        )
    # Non-final turn: model may choose main or policy
    return (
        "- If outputting `````policy`````: define `def policy(state)` returning a valid action.\n"
        "- If outputting `````main`````: use `print()` to surface derivations — "
        "stdout is returned to you next turn. No `def policy` needed here.\n"
        + common
    )


def load_template(template_path: str = "configs/prompt_template.txt") -> str:
    return Path(template_path).read_text()


def render_prompt(
    template: str,
    task: dict,
    turn_idx: int = 1,
    total_turns: int = 1,
) -> str:
    """Fill in template with task fields and (optionally) multi-turn protocol.

    Single-turn callers can omit `turn_idx` / `total_turns` and the rendered
    output matches the pre-multi-turn prompt exactly.
    """
    return template.format(
        env_code=task["env_code"],
        state_spec=task["state_spec"],
        action_spec=task["action_spec"],
        goal=task["goal"],
        multi_turn_protocol=_build_multi_turn_protocol(turn_idx, total_turns),
        requirements=_build_requirements(turn_idx, total_turns),
        response_format=_build_response_format(turn_idx, total_turns),
    )


def format_dpo_example(
    prompt_text: str,
    chosen_code: str,
    rejected_code: str,
) -> dict:
    """Build a single DPO training example in TRL conversational format.

    TRL DPOTrainer with conversational format will apply the chat template
    (ChatML for Qwen2.5) automatically via tokenizer.apply_chat_template().
    """
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "chosen": [{"role": "assistant", "content": chosen_code}],
        "rejected": [{"role": "assistant", "content": rejected_code}],
    }
