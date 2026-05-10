"""Train LoRA adapter with branched-rollout multi-turn GRPO (N=2, VinePPO-style).

Usage: python scripts/05_train_grpo_branched.py --iter 0 --config configs/loop_grpo_branched.yaml
"""
import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.branched_trainer import BranchedGRPOTrainer
from src.dataset import load_jsonl
from src.diversity import DiversityCallback
from src.prompts import load_template, render_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# Slack tokens added on top of `2 * max_turn_tokens` to accommodate the
# chat-template overhead (role tags) and the tool-result message between
# turns. 256 is comfortable for ChatML + a typical tool result; bump if
# we start seeing turn-2 truncation.
COMPLETION_LENGTH_SLACK = 256
TOTAL_TURNS_FOR_BRANCHED = 2  # this trainer is N=2 by construction


def _placeholder_reward_fn(completions, **kwargs):
    """No-op reward function. Required by GRPOTrainer's __init__ but never
    called — BranchedGRPOTrainer computes rewards inside its own override of
    _generate_and_score_completions, bypassing _calculate_rewards entirely.
    """
    return [0.0] * len(completions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", default="configs/loop_grpo_branched.yaml")
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    from src.config import validate_config
    validate_config(cfg)

    iter_num = args.iter
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    grpo_cfg = cfg["grpo"]
    branched_cfg = cfg["branched"]
    scoring_cfg = cfg["scoring"]

    base_model_path = str(Path(cfg["paths"]["hf_base_dir"]) / "iter_000")
    lora_base = Path(cfg["paths"]["lora_dir"])
    if args.run_name:
        lora_base = lora_base / args.run_name
    output_path = str(lora_base / f"iter_{iter_num:03d}")

    if not Path(base_model_path).exists():
        logger.error(
            f"Base model not found at {base_model_path}. "
            f"Run 'bash scripts/00_setup_env.sh' to download it."
        )
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompt dataset. Prompts use the multi-turn template (total_turns=2).
    tasks = load_jsonl(cfg["paths"]["tasks_train"])
    if not tasks:
        logger.error("No training tasks found.")
        sys.exit(1)

    template = load_template("configs/prompt_template.txt")
    task_map = {}
    prompt_turn2_map = {}
    dataset_rows = []
    for task in tasks:
        tid = task["task_id"]
        prompt_text = render_prompt(
            template, task, turn_idx=1, total_turns=TOTAL_TURNS_FOR_BRANCHED,
        )
        prompt_turn2_map[tid] = render_prompt(
            template, task,
            turn_idx=TOTAL_TURNS_FOR_BRANCHED,
            total_turns=TOTAL_TURNS_FOR_BRANCHED,
        )
        task_map[tid] = task
        dataset_rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "task_id": tid,
        })

    train_dataset = Dataset.from_list(dataset_rows)
    logger.info(f"Training on {len(train_dataset)} task prompts (multi-turn N=2)")

    g1 = branched_cfg["g1"]
    g2 = branched_cfg["g2"]
    max_turn_tokens = branched_cfg.get("max_turn_tokens", 400)
    logger.info(f"Branched rollout: g1={g1}, g2={g2}, max_turn_tokens={max_turn_tokens}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        bias="none",
    )

    use_mps = torch.backends.mps.is_available()

    pdtbs = train_cfg["per_device_train_batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    epochs = train_cfg["num_train_epochs"]
    prompts_per_step = max(1, (pdtbs * grad_accum) // (g1 * g2))
    max_steps = max(1, math.ceil(len(train_dataset) * epochs / prompts_per_step))
    logger.info(
        f"Computed max_steps={max_steps} "
        f"({len(train_dataset)} prompts × {epochs} epochs / {prompts_per_step} prompts/step)"
    )

    training_args = GRPOConfig(
        output_dir=output_path,
        num_generations=g1 * g2,
        num_iterations=grpo_cfg.get("num_iterations", 1),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high"),
        beta=grpo_cfg.get("beta", 0.0),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        scale_rewards=grpo_cfg.get("scale_rewards", "batch"),
        max_completion_length=2 * max_turn_tokens + COMPLETION_LENGTH_SLACK,
        # Default matches the eval pipeline (`scripts/eval_frozen_baseline.py`,
        # `--temperature` default = 0.8). Production YAMLs override this
        # explicitly; the fallback only kicks in if a config omits the key.
        temperature=grpo_cfg.get("temperature", 0.8),
        top_p=grpo_cfg.get("top_p", 1.0),
        mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", False),
        log_completions=grpo_cfg.get("log_completions", False),
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=pdtbs,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=not use_mps and train_cfg.get("bf16", False),
        fp16=use_mps or train_cfg.get("fp16", False),
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=not use_mps,
        use_cpu=False,
    )

    dtype = torch.float16 if use_mps else torch.bfloat16
    logger.info("Loading original base model (iter_000)...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)

    for i in range(iter_num):
        prev_lora = str(lora_base / f"iter_{i:03d}")
        logger.info(f"  Merging previous adapter iter_{i:03d}...")
        model = PeftModel.from_pretrained(model, prev_lora)
        model = model.merge_and_unload()

    trainer = BranchedGRPOTrainer(
        model=model,
        reward_funcs=_placeholder_reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        g1=g1,
        g2=g2,
        task_map=task_map,
        prompt_turn2_map=prompt_turn2_map,
        max_turn_tokens=max_turn_tokens,
        scoring_timeout=scoring_cfg["timeout_seconds"],
        scoring_max_steps=scoring_cfg["max_steps"],
    )

    diversity_cb = DiversityCallback()
    diversity_cb.trainer = trainer
    trainer.add_callback(diversity_cb)

    logger.info("Starting branched GRPO training...")
    trainer.train()

    import json
    log_dir = Path("data/progress") / (args.run_name or "default")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"trainer_log_iter_{iter_num:03d}.json"
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f)
    logger.info(f"Wrote trainer log history to {log_path}")

    logger.info(f"Saving adapter to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
