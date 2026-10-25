"""Tests for `src.config_schema`.

Two roles:
  1. Every YAML config in `configs/` must validate. If this
     suite goes red after touching the schema, either the schema is too
     strict or the config is genuinely broken.
  2. Dead-config patterns we already know about must fail validation —
     specifically `scale_rewards` in a branched-GRPO config, which is the
     exact bug that motivated this work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config_schema import (  # noqa: E402
    BranchedGRPOConfig,
    BranchedGRPOSection,
    BranchedSection,
    DPOConfig,
    FlatGRPOConfig,
    FlatGRPOSection,
    validate_config,
)


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _minimal_dpo_cfg() -> dict:
    """Smallest config that validates as DPOConfig. The DPO pipeline ships no
    YAML (it is not part of the current study), so the dispatch tests build
    one inline."""
    return {
        "base_model": {"hf": "Qwen/Qwen3-4B-Instruct-2507"},
        "paths": {
            "hf_base_dir": "models/hf_base",
            "lora_dir": "models/lora",
            "rollouts_dir": "data/rollouts",
            "dpo_dir": "data/dpo",
            "tasks_train": "scripts/seed_tasks/crystallizer_train.jsonl",
            "tasks_eval": "scripts/seed_tasks/crystallizer_eval.jsonl",
        },
        "loop": {"iterations": 1, "trainer": "dpo"},
        "generation": {"n_samples_per_task": 2, "temperature": 0.8, "top_p": 0.95, "max_tokens": 400},
        "scoring": {"timeout_seconds": 30, "max_steps": 500},
        "dpo_pairs": {"min_chosen_reward": 1.0, "pairs_per_task": 1},
        "training": {
            "learning_rate": 5e-6,
            "per_device_train_batch_size": 2,
            "num_train_epochs": 1,
            "max_length": 4096,
            "max_prompt_length": 3072,
        },
        "lora": {"target_modules": ["q_proj"], "r": 8, "alpha": 16, "dropout": 0.05, "task_type": "CAUSAL_LM"},
    }


# ---------------------------------------------------------------------------
# Happy path: every shipped YAML config validates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", [p.name for p in CONFIGS_DIR.glob("*.yaml")])
def test_existing_config_validates(config_name: str) -> None:
    """Every YAML under `configs/` must pass schema validation."""
    cfg = yaml.safe_load((CONFIGS_DIR / config_name).read_text())
    validate_config(cfg)


# ---------------------------------------------------------------------------
# Trainer dispatch
# ---------------------------------------------------------------------------


def test_dispatches_to_dpo_when_trainer_dpo() -> None:
    assert isinstance(validate_config(_minimal_dpo_cfg()), DPOConfig)


def test_dispatches_to_flat_grpo() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_crystallizer.yaml").read_text())
    assert isinstance(validate_config(cfg), FlatGRPOConfig)


def test_dispatches_to_branched_grpo() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_branched_crystallizer.yaml").read_text())
    assert isinstance(validate_config(cfg), BranchedGRPOConfig)


def test_default_trainer_is_dpo() -> None:
    """A config that omits `loop.trainer` routes to DPO (the historical default)."""
    cfg = _minimal_dpo_cfg()
    del cfg["loop"]["trainer"]
    assert isinstance(validate_config(cfg), DPOConfig)


def test_unknown_trainer_raises() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_crystallizer.yaml").read_text())
    cfg["loop"]["trainer"] = "not_a_real_trainer"
    with pytest.raises((ValueError, ValidationError)):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Dead-config detection: the motivating bug
# ---------------------------------------------------------------------------


def test_scale_rewards_in_branched_grpo_section_is_rejected() -> None:
    """The motivating bug: `scale_rewards` sat in branched configs for months
    while the branched trainer ignored it. The schema must reject it."""
    with pytest.raises(ValidationError) as exc:
        BranchedGRPOSection.model_validate({
            "loss_type": "dapo",
            "scale_rewards": "batch",  # dead — branched trainer doesn't honor it
        })
    msg = str(exc.value)
    assert "scale_rewards" in msg
    assert "Extra inputs" in msg or "extra" in msg.lower()


def test_scale_rewards_in_flat_grpo_section_is_accepted() -> None:
    """Same key, different trainer: flat GRPO routes through stock TRL which
    DOES honor `scale_rewards`. The schema must accept it here."""
    section = FlatGRPOSection.model_validate({
        "num_generations": 8,
        "max_completion_length": 600,
        "scale_rewards": "batch",
    })
    assert section.scale_rewards == "batch"


def test_unknown_key_in_branched_section_rejected() -> None:
    """Property: dead-config protection generalises beyond `scale_rewards`."""
    with pytest.raises(ValidationError) as exc:
        BranchedSection.model_validate({
            "g1": 4, "g2": 4,
            "this_knob_does_not_exist": 1,
        })
    assert "this_knob_does_not_exist" in str(exc.value)


def test_unknown_top_level_section_rejected() -> None:
    """Top-level extras forbidden — keeps the config dict in sync with the
    schema as the surface area changes."""
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_branched_crystallizer.yaml").read_text())
    cfg["new_section_that_nothing_reads"] = {"foo": "bar"}
    with pytest.raises(ValidationError) as exc:
        validate_config(cfg)
    assert "new_section_that_nothing_reads" in str(exc.value)


# ---------------------------------------------------------------------------
# Trainer-specific section invariants
# ---------------------------------------------------------------------------


def test_branched_grpo_loss_type_restricted_to_additive() -> None:
    """The two-pass loss decomposition only handles additive losses
    (`BranchedGRPOTrainer._assert_supported_config`). The schema's Literal
    surfaces that constraint at config-load time instead of trainer-startup
    time."""
    # dapo and cispo are fine
    BranchedGRPOSection.model_validate({"loss_type": "dapo"})
    BranchedGRPOSection.model_validate({"loss_type": "cispo"})
    # but `grpo` (per-row averaged) is not — would break the decomposition
    with pytest.raises(ValidationError):
        BranchedGRPOSection.model_validate({"loss_type": "grpo"})


def test_branched_section_requires_g1_g2() -> None:
    with pytest.raises(ValidationError):
        BranchedSection.model_validate({"max_turn_tokens": 400})


def test_loop_iterations_must_be_positive() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_crystallizer.yaml").read_text())
    cfg["loop"]["iterations"] = 0
    with pytest.raises(ValidationError):
        validate_config(cfg)


def test_lora_dropout_must_be_in_unit_interval() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_crystallizer.yaml").read_text())
    cfg["lora"]["dropout"] = 1.5
    with pytest.raises(ValidationError):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Missing-section diagnostics (replaces the old hand-rolled validator)
# ---------------------------------------------------------------------------


def test_branched_config_missing_branched_section() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_branched_crystallizer.yaml").read_text())
    del cfg["branched"]
    with pytest.raises(ValidationError) as exc:
        validate_config(cfg)
    assert "branched" in str(exc.value)


def test_dpo_config_missing_dpo_pairs_section() -> None:
    cfg = _minimal_dpo_cfg()
    del cfg["dpo_pairs"]
    with pytest.raises(ValidationError) as exc:
        validate_config(cfg)
    assert "dpo_pairs" in str(exc.value)


# ---------------------------------------------------------------------------
# Optional `prompt` section
# ---------------------------------------------------------------------------


def test_prompt_example_section_is_optional_and_validated() -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "loop_grpo_branched_crystallizer.yaml").read_text())
    assert validate_config(cfg).prompt is None
    cfg["prompt"] = {"example": "format"}
    assert validate_config(cfg).prompt.example == "format"
    cfg["prompt"] = {"example": "bogus"}
    with pytest.raises(ValidationError):
        validate_config(cfg)
    cfg["prompt"] = {"exmaple": "none"}  # misspelt key must not pass silently
    with pytest.raises(ValidationError):
        validate_config(cfg)
