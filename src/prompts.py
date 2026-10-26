"""Prompt rendering: the task brief.

The model is briefed the way an engineer would be: the goal and the physics in
words, what the readings are, what cannot be observed, what the probe run is
for, how the final run is scored, and the reply format. No environment source,
no format example, no mode directives: the probe run is turn 1 by construction
and the final run is the last turn, so the runner reads the turn position.

Task fields used: `intro`, `goal`, `state_spec`, `action_spec`, `unobserved`,
`probe_spec`, `final_spec` (see scripts/seed_tasks/build_crystallizer_tasks.py).
"""
from pathlib import Path

from src.runner import MAIN_DEFAULT_TIMEOUT, MAIN_STDOUT_TRUNCATE

# What the controller may and may not do. Stated once in the brief and again in
# the final-turn message.
CONTROLLER_RULES = (
    "It is called once per step with the state dict and returns the action. It "
    "may keep state between calls (module-level variables are fine); it runs on "
    "one batch. Import only `math` and `random`. Read only the state keys listed above."
)

_DEFAULT_PROBE_SPEC = "the same environment, with the same hidden parameters, on a different episode"
_DEFAULT_FINAL_SPEC = "the same hidden parameters as your probe run, a different episode"


def _probe_section(task: dict, total_turns: int) -> str:
    n_probe = total_turns - 1
    when = "this message" if n_probe == 1 else f"this message and the next {n_probe - 1}"
    probe_spec = task.get("probe_spec", _DEFAULT_PROBE_SPEC)
    return f"""## Your probe run ({when})

Before the scored batch you get a probe run on {probe_spec}. It is the object `env`, already constructed:

    state, info = env.reset()                                # starts a fresh probe batch
    state, reward, done, truncated, info = env.step(action)  # one step; `state` has the keys above
    env.failure_reason                                       # None, "shower", "seeds_dissolved" or "timeout"
    env.showered, env.done                                   # True once the batch has showered / has ended

Drive it however you like, as many batches as you want, within {MAIN_DEFAULT_TIMEOUT} seconds of compute. Print what you learn: only your printed output reaches the next step, and output beyond {MAIN_STDOUT_TRUNCATE} characters is cut off. The controller you write afterwards has no access to `env`, to this code or to its variables.
"""


def _final_section(total_turns: int) -> str:
    if total_turns == 1:
        when, lead = "this message", "Write `def policy(state)`, the controller."
    else:
        when = "the last message"
        lead = "You will receive your printed output and then write `def policy(state)`, the controller."
    return f"""## The final run ({when})

{lead} It runs on the scored batch from its first step. {CONTROLLER_RULES}
"""


def _format_section(total_turns: int) -> str:
    what = "the controller, defining `def policy(state)`" if total_turns == 1 else "the probe code"
    return f"Reply with one Python code block and nothing else: {what}."


def _followup(task: dict, turn_idx: int, total_turns: int) -> str:
    """The instructions for a turn after the first (the probe output is prepended by
    `compose_turn_message`)."""
    if turn_idx < total_turns:
        return (
            "You may run another probe on a fresh `env` (same hidden parameters, a new batch; "
            "nothing from your previous code carries over). Reply with one Python code block "
            "and nothing else: the probe code."
        )
    final_spec = task.get("final_spec", _DEFAULT_FINAL_SPEC)
    return (
        "Now the final run. Reply with one Python code block and nothing else: the controller, "
        f"defining `def policy(state)`. It runs on the scored batch ({final_spec}) from its first "
        f"step and has no access to `env` or to your probe code. {CONTROLLER_RULES}"
    )


def load_template(template_path: str = "configs/prompt_template.txt") -> str:
    return Path(template_path).read_text()


def render_prompt(template: str, task: dict, turn_idx: int = 1, total_turns: int = 1) -> str:
    """The user message for one turn.

    Turn 1 is the brief (the template filled with the task's text). Later turns
    are the short instruction that follows the probe output; callers prepend
    the output with `compose_turn_message`. Single-turn callers can omit
    `turn_idx` and `total_turns`.
    """
    if total_turns < 1 or turn_idx < 1 or turn_idx > total_turns:
        raise ValueError(f"turn_idx={turn_idx} out of range for total_turns={total_turns}")
    if turn_idx > 1:
        return _followup(task, turn_idx, total_turns)
    unobserved = (task.get("unobserved") or "").strip()
    unobserved_section = f"## What you do not observe\n\n{unobserved}\n\n" if unobserved else ""
    turn_section = (_probe_section(task, total_turns) + "\n" if total_turns > 1 else "") + _final_section(total_turns)
    return template.format(
        intro=(task.get("intro") or "Write a Python controller for the process below.").strip(),
        goal=task["goal"].strip(),
        state_spec=task["state_spec"].strip(),
        action_spec=task["action_spec"].strip(),
        unobserved_section=unobserved_section,
        turn_section=turn_section,
        format=_format_section(total_turns),
    )


def compose_turn_message(turn_prompt: str, tool_text: str) -> str:
    """The user message after a probe run: what it printed, then this turn's instructions."""
    return f"Output of your probe run:\n{tool_text}\n\n{turn_prompt}"


def format_dpo_example(
    prompt_text: str,
    chosen_code: str,
    rejected_code: str,
) -> dict:
    """Build a single DPO training example in TRL conversational format.

    TRL DPOTrainer with conversational format will apply the chat template
    (ChatML for Qwen) automatically via tokenizer.apply_chat_template().
    """
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "chosen": [{"role": "assistant", "content": chosen_code}],
        "rejected": [{"role": "assistant", "content": rejected_code}],
    }
