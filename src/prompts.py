"""Prompt rendering: the task brief.

The model is briefed the way an engineer would be: the goal and the physics in
words, what the readings are, what cannot be observed, what a probe run is,
what the final run is, and how it acts. No environment source, no example, no
directive: the brief states what exists and says nothing about what to do
with it or in what order.

The model acts through two tools, `probe_run` and `final_run`
(`src.agent_loop`). Their definitions reach the model through the chat
template's own tool block; the brief describes what each runs on. With a
single reply there is no probe run, and neither the tool nor its section is
there. What the model knows from earlier batches (quench logs of other
batches) is part of the brief, as the batch record would be.

Task fields used: `intro`, `goal`, `state_spec`, `action_spec`, `unobserved`,
`earlier_batches`, `probe_spec`
(see scripts/seed_tasks/build_crystallizer_tasks.py).
"""
from pathlib import Path

from src.agent_loop import TOOL_FINAL, TOOL_PROBE
from src.runner import MAIN_DEFAULT_TIMEOUT, MAIN_STDOUT_TRUNCATE

# What the controller may and may not do.
CONTROLLER_RULES = (
    "It is called once per step with the state dict and returns the action. It "
    "may keep state between calls (module-level variables are fine); it runs on "
    "one batch. Import only `math` and `random`. Read only the state keys listed above."
)

_DEFAULT_PROBE_SPEC = "the same environment, with the same hidden parameters, on a different episode"


def _probe_section(task: dict) -> str:
    probe_spec = task.get("probe_spec", _DEFAULT_PROBE_SPEC)
    return f"""## The probe run

`{TOOL_PROBE}` runs your code on {probe_spec}. The object `env` is already constructed there:

    state, info = env.reset()                                # starts a fresh probe batch
    state, reward, done, truncated, info = env.step(action)  # one step; `state` has the keys above
    env.failure_reason                                       # None, "shower", "seeds_dissolved" or "timeout"
    env.showered, env.done                                   # True once the batch has showered / has ended

`done` is returned by `env.step`; `env.done`, `env.showered` and `env.failure_reason` are attributes of `env`, not keys of `state`. The code may run as many probe batches as fit in {MAIN_DEFAULT_TIMEOUT} seconds of compute. What it prints is returned to you, up to {MAIN_STDOUT_TRUNCATE} characters; nothing else carries over, neither the code nor its variables.
"""


def _final_section(total_turns: int) -> str:
    no_env = " It has no access to `env`." if total_turns > 1 else ""
    return f"""## The final run

`{TOOL_FINAL}` takes the controller, `def policy(state)`, and runs it on the scored batch from its first step; the task ends there. {CONTROLLER_RULES}{no_env}
"""


def _act_section(total_turns: int) -> str:
    if total_turns == 1:
        return f"You act through a tool call: `{TOOL_FINAL}`. You have one reply."
    return (f"You act through tool calls: `{TOOL_PROBE}` and `{TOOL_FINAL}`. You have {total_turns} replies; "
            "the task ends with the final run or when the replies run out.")


def load_template(template_path: str = "configs/prompt_template.txt") -> str:
    return Path(template_path).read_text()


def render_prompt(template: str, task: dict, total_turns: int = 1) -> str:
    """The brief: the one user message that opens a task.

    `total_turns` is the number of replies the model has. With one reply the
    brief describes the final run only; with more it also describes the probe
    run. What follows a reply is not a prompt but the tool response
    (`src.multi_turn.execute_turn`).
    """
    if total_turns < 1:
        raise ValueError(f"total_turns must be >= 1, got {total_turns}")
    unobserved = (task.get("unobserved") or "").strip()
    unobserved_section = f"## What you do not observe\n\n{unobserved}\n\n" if unobserved else ""
    earlier = (task.get("earlier_batches") or "").strip()
    if earlier:
        unobserved_section += f"## What you know from earlier batches\n\n{earlier}\n\n"
    run_sections = (_probe_section(task) + "\n" if total_turns > 1 else "") + _final_section(total_turns)
    return template.format(
        intro=(task.get("intro") or "Write a Python controller for the process below.").strip(),
        goal=task["goal"].strip(),
        state_spec=task["state_spec"].strip(),
        action_spec=task["action_spec"].strip(),
        unobserved_section=unobserved_section,
        run_sections=run_sections,
        act=_act_section(total_turns),
    )


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
