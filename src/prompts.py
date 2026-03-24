from pathlib import Path


def load_template(template_path: str = "configs/prompt_template.txt") -> str:
    return Path(template_path).read_text()


def render_prompt(template: str, task: dict) -> str:
    """Fill in template with task fields."""
    return template.format(
        env_code=task["env_code"],
        state_spec=task["state_spec"],
        action_spec=task["action_spec"],
        goal=task["goal"],
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
