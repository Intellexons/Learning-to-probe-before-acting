"""Config validation for training pipelines."""

VALID_TRAINERS = {"dpo", "grpo", "grpo_branched"}

SHARED_TRAINING_KEYS = {
    "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
    "num_train_epochs", "warmup_ratio", "bf16", "gradient_checkpointing", "logging_steps",
}

DPO_TRAINING_KEYS = {"beta", "max_length", "max_prompt_length", "loss_type"}

DPO_REQUIRED_SECTIONS = {"generation", "scoring", "dpo_pairs", "training", "lora"}
GRPO_REQUIRED_SECTIONS = {"grpo", "scoring", "training", "lora"}
# Branched-rollout adds a `branched` section with g1/g2/max_turn_tokens.
# `grpo` is still required (TRL still reads epsilon/beta/loss_type/etc.) but
# `num_generations` and `max_completion_length` are derived from g1*g2 and
# 2*max_turn_tokens by the training script, so they are NOT required here.
GRPO_BRANCHED_REQUIRED_SECTIONS = {"grpo", "branched", "scoring", "training", "lora"}
GRPO_BRANCHED_REQUIRED_KEYS = {"g1", "g2"}

GRPO_REQUIRED_KEYS = {"num_generations", "max_completion_length"}


def validate_config(cfg: dict):
    """Validate config consistency. Raises ValueError on misconfiguration."""
    trainer = cfg.get("loop", {}).get("trainer", "dpo")

    if trainer not in VALID_TRAINERS:
        raise ValueError(
            f"Unknown trainer '{trainer}'. Must be one of: {', '.join(sorted(VALID_TRAINERS))}"
        )

    if trainer == "dpo":
        missing = DPO_REQUIRED_SECTIONS - set(cfg.keys())
        if missing:
            raise ValueError(f"trainer is 'dpo' but config is missing sections: {', '.join(sorted(missing))}")

        training = cfg.get("training", {})
        missing_keys = (SHARED_TRAINING_KEYS | DPO_TRAINING_KEYS) - set(training.keys())
        if missing_keys:
            raise ValueError(
                f"trainer is 'dpo' but 'training' section is missing keys: {', '.join(sorted(missing_keys))}"
            )

    elif trainer == "grpo":
        missing = GRPO_REQUIRED_SECTIONS - set(cfg.keys())
        if missing:
            raise ValueError(f"trainer is 'grpo' but config is missing sections: {', '.join(sorted(missing))}")

        training = cfg.get("training", {})
        missing_keys = SHARED_TRAINING_KEYS - set(training.keys())
        if missing_keys:
            raise ValueError(
                f"trainer is 'grpo' but 'training' section is missing keys: {', '.join(sorted(missing_keys))}"
            )

        grpo = cfg.get("grpo", {})
        missing_grpo = GRPO_REQUIRED_KEYS - set(grpo.keys())
        if missing_grpo:
            raise ValueError(
                f"trainer is 'grpo' but 'grpo' section is missing keys: {', '.join(sorted(missing_grpo))}"
            )

    elif trainer == "grpo_branched":
        missing = GRPO_BRANCHED_REQUIRED_SECTIONS - set(cfg.keys())
        if missing:
            raise ValueError(
                f"trainer is 'grpo_branched' but config is missing sections: {', '.join(sorted(missing))}"
            )

        training = cfg.get("training", {})
        missing_keys = SHARED_TRAINING_KEYS - set(training.keys())
        if missing_keys:
            raise ValueError(
                f"trainer is 'grpo_branched' but 'training' section is missing keys: {', '.join(sorted(missing_keys))}"
            )

        branched = cfg.get("branched", {})
        missing_branched = GRPO_BRANCHED_REQUIRED_KEYS - set(branched.keys())
        if missing_branched:
            raise ValueError(
                f"trainer is 'grpo_branched' but 'branched' section is missing keys: "
                f"{', '.join(sorted(missing_branched))}"
            )
