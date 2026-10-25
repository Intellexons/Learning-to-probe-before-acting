"""Pydantic schemas for training-pipeline YAML configs.

The point of these schemas is to make dead-config impossible. The case that
motivated this: `scale_rewards: "batch"` sat in the branched-trainer YAMLs
for months while the branched trainer overrode the only TRL code path that
reads `scale_rewards` — so the field had no effect, and nobody noticed.
With these schemas, the branched config's `grpo` section has no
`scale_rewards` field, and `model_config = ConfigDict(extra="forbid")`
makes its presence a validation error.

Strictness policy:

- **Strict** (`extra="forbid"`): every section that's coupled to a specific
  trainer's code path. Any unknown key is an error. These are the sections
  where dead config is most likely to cause silent bugs.
    - `grpo` (`FlatGRPOSection` / `BranchedGRPOSection` — different sets)
    - `branched`
    - `dpo_pairs`
    - `training` (`DPOTrainingSection` / `GRPOTrainingSection`)

- **Permissive** (`extra="allow"`): sections that are user-customisable or
  read by many code paths, where forbidding extras would create friction
  without much benefit.
    - `paths`, `base_model`, `loop`, `lora`, `scoring`, `generation`, `eval`

Trainer dispatch: `validate_config()` reads `loop.trainer` and routes to
`DPOConfig`, `FlatGRPOConfig`, or `BranchedGRPOConfig`. Each top-level
config enumerates its own required sections.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared "permissive" sections — extra keys allowed
# ---------------------------------------------------------------------------


class _PermissiveBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseModelSection(_PermissiveBase):
    hf: str


class LoopSection(_PermissiveBase):
    iterations: int = Field(gt=0)
    # Default "dpo" matches the existing validator (smoke_test.yaml omits the
    # field and gets dispatched to DPO).
    trainer: Literal["dpo", "grpo", "grpo_branched"] = "dpo"


class PathsSection(_PermissiveBase):
    hf_base_dir: str
    lora_dir: str
    rollouts_dir: str
    tasks_train: str
    tasks_eval: str
    # Only used by DPO; absent in GRPO configs.
    dpo_dir: str | None = None


class LoRASection(_PermissiveBase):
    target_modules: list[str]
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    task_type: str


class ScoringSection(_PermissiveBase):
    timeout_seconds: int = Field(gt=0)
    max_steps: int = Field(gt=0)


class GenerationSection(_PermissiveBase):
    n_samples_per_task: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_tokens: int = Field(gt=0)


class EvalSection(_PermissiveBase):
    n_turns: int = Field(gt=0)


class PromptSection(_StrictBase):
    """`prompt` section: what the multi-turn format example shows. Strict,
    so a misspelt key cannot silently leave the default in place."""

    example: Literal["full", "format", "none"] = "full"


# ---------------------------------------------------------------------------
# Strict trainer-coupled sections
# ---------------------------------------------------------------------------


class DPOPairsSection(_StrictBase):
    min_chosen_reward: float
    pairs_per_task: int = Field(gt=0)


# `loss_type` values accepted by TRL's stock GRPOTrainer. The branched
# trainer's two-pass decomposition only handles the additive losses (dapo,
# cispo) — see `BranchedGRPOTrainer._assert_supported_config`.
_FlatGRPOLossType = Literal["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo"]
_BranchedGRPOLossType = Literal["dapo", "cispo"]
_ScaleRewardsMode = Literal["batch", "group", "none"]


class FlatGRPOSection(_StrictBase):
    """`grpo` section for the flat-GRPO trainer (scripts/04_train_grpo_lora.py).

    Mirrors the fields read in 04_train_grpo_lora.py + the TRL `GRPOConfig`
    knobs the script forwards. Adding a new knob to the script means adding
    it here too — that's the friction we're trading for dead-config safety.
    """

    num_generations: int = Field(gt=0)
    max_completion_length: int = Field(gt=0)
    num_iterations: int = Field(default=1, gt=0)
    epsilon: float = 0.2
    epsilon_high: float | None = None
    beta: float = 0.0
    loss_type: _FlatGRPOLossType = "dapo"
    scale_rewards: _ScaleRewardsMode | None = None
    temperature: float = Field(default=0.8, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    mask_truncated_completions: bool = False
    log_completions: bool = False


class BranchedGRPOSection(_StrictBase):
    """`grpo` section for the branched trainer (scripts/05_train_grpo_branched.py).

    Subset of FlatGRPOSection. Critical: `scale_rewards` is NOT here — TRL
    only applies it inside the stock `_generate_and_score_completions` that
    the branched trainer overrides, so it has no effect on branched runs.
    `num_generations` and `max_completion_length` are also absent: the
    branched script derives them from g1*g2 and 2*max_turn_tokens.
    """

    num_iterations: int = Field(default=1, gt=0)
    epsilon: float = 0.2
    epsilon_high: float | None = None
    beta: float = 0.0
    loss_type: _BranchedGRPOLossType = "dapo"
    temperature: float = Field(default=0.8, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    mask_truncated_completions: bool = False
    log_completions: bool = False


class BranchedSection(_StrictBase):
    g1: int = Field(gt=0)
    g2: int = Field(gt=0)
    max_turn_tokens: int = Field(default=400, gt=0)
    loss_micro_batch_size: int | None = Field(default=None, gt=0)


class _SharedTrainingFields(_StrictBase):
    """Keys both DPO and GRPO training sections share."""

    learning_rate: float = Field(gt=0.0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    num_train_epochs: float = Field(gt=0)  # float so values like 0.5 work
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = Field(default=1, gt=0)
    report_to: str | None = None
    wandb_project: str | None = None


class DPOTrainingSection(_SharedTrainingFields):
    """`training` section for the DPO trainer."""

    beta: float = Field(default=0.1, ge=0.0)
    max_length: int = Field(gt=0)
    max_prompt_length: int = Field(gt=0)
    loss_type: str = "sigmoid"
    # Colab/CPU-friendly DPOTrainer knob.
    precompute_ref_log_probs: bool = False


class GRPOTrainingSection(_SharedTrainingFields):
    """`training` section for flat and branched GRPO trainers."""


# ---------------------------------------------------------------------------
# Top-level configs (one per trainer)
# ---------------------------------------------------------------------------


class _TopLevelBase(_StrictBase):
    """Top level forbids extras too: if you add a new section to a YAML, the
    schema should know about it (or you should ask whether it's dead config)."""

    base_model: BaseModelSection
    paths: PathsSection
    loop: LoopSection
    scoring: ScoringSection
    lora: LoRASection
    # Optional: omitted means the "full" example (the original prompt).
    prompt: PromptSection | None = None


class DPOConfig(_TopLevelBase):
    generation: GenerationSection
    dpo_pairs: DPOPairsSection
    training: DPOTrainingSection


class FlatGRPOConfig(_TopLevelBase):
    grpo: FlatGRPOSection
    training: GRPOTrainingSection
    # Optional in the flat-GRPO smoke (configs/smoke_test_grpo.yaml omits it).
    generation: GenerationSection | None = None


class BranchedGRPOConfig(_TopLevelBase):
    branched: BranchedSection
    grpo: BranchedGRPOSection
    training: GRPOTrainingSection
    generation: GenerationSection
    eval: EvalSection | None = None


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------


_TRAINER_TO_CONFIG: dict[str, type[_TopLevelBase]] = {
    "dpo": DPOConfig,
    "grpo": FlatGRPOConfig,
    "grpo_branched": BranchedGRPOConfig,
}


def validate_config(cfg: dict[str, Any]) -> _TopLevelBase:
    """Validate a raw YAML dict against the right schema for its `loop.trainer`.

    Returns the validated pydantic model. Existing callers that only need
    side-effect validation (raise on error) can ignore the return value.
    """
    trainer = cfg.get("loop", {}).get("trainer", "dpo")
    config_cls = _TRAINER_TO_CONFIG.get(trainer)
    if config_cls is None:
        raise ValueError(
            f"Unknown trainer '{trainer}'. Must be one of: "
            f"{', '.join(sorted(_TRAINER_TO_CONFIG))}"
        )
    return config_cls.model_validate(cfg)
