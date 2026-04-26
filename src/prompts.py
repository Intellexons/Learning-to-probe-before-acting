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
        budget_note = (
            "**This is the FINAL turn.** Your output MUST be a `policy` block — "
            "a `main` block on the final turn is rejected and scored at the error floor."
        )
    elif remaining_after == 1:
        budget_note = "You will have **1 more turn** after this one."
    else:
        budget_note = f"You will have **{remaining_after} more turns** after this one."

    return f"""
## Multi-Turn Protocol

You have **{total_turns} turns total** in this coding session. This is **turn {turn_idx} of {total_turns}**.
{budget_note}

Each turn, output exactly one fenced code block. The fence's language tag declares the mode:

- ```` ```policy ```` — define `def policy(state)`. After your turn, the runner executes the policy across all scenarios and the next user message reports the total reward.
- ```` ```main ```` — arbitrary Python (math, derivations, intermediate variables you want to print). Captured stdout is returned to you. `main` does **NOT** have access to the env class or any scenario data — use it for reasoning and derivations, not for empirical probing.

**The last turn (turn {total_turns}) MUST be a `policy` turn.** A `main` last turn is rejected and scored at the error floor.

A turn that does not declare a recognized mode (no fence, or a fence with an unknown tag) is also rejected at the error floor.
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
