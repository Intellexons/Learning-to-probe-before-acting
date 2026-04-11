"""Train LoRA adapter with GRPO on task prompts.

Usage: python scripts/04_train_grpo_lora.py --iter 0 --config configs/loop.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import load_jsonl
from src.prompts import load_template, render_prompt
from src.reward import make_reward_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", default="configs/loop.yaml")
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    from src.config import validate_config
    validate_config(cfg)

    iter_num = args.iter
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    grpo_cfg = cfg["grpo"]
    scoring_cfg = cfg["scoring"]

    base_model_path = str(Path(cfg["paths"]["hf_base_dir"]) / "iter_000")
    lora_base = Path(cfg["paths"]["lora_dir"])
    if args.run_name:
        lora_base = lora_base / args.run_name
    output_path = str(lora_base / f"iter_{iter_num:03d}")

    # Check base model exists
    if not Path(base_model_path).exists():
        logger.error(
            f"Base model not found at {base_model_path}. "
            f"Run 'bash scripts/00_setup_env.sh' to download it."
        )
        sys.exit(1)

    logger.info(f"Original base: {base_model_path}")
    logger.info(f"Will merge adapters 0..{iter_num - 1} before training adapter {iter_num}")
    logger.info(f"Output: {output_path}")

    # Load tokenizer from original base
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompt dataset from training tasks
    tasks = load_jsonl(cfg["paths"]["tasks_train"])
    if not tasks:
        logger.error("No training tasks found. Skipping training.")
        sys.exit(1)

    template = load_template("configs/prompt_template.txt")
    task_map = {}
    dataset_rows = []
    for task in tasks:
        tid = task["task_id"]
        prompt_text = render_prompt(template, task)
        task_map[tid] = task
        dataset_rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "task_id": tid,
        })

    train_dataset = Dataset.from_list(dataset_rows)
    logger.info(f"Training on {len(train_dataset)} task prompts")

    # Build reward function
    reward_fn = make_reward_fn(task_map, scoring_cfg)

    # LoRA config
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        bias="none",
    )

    # Detect device — MPS needs special handling (no pin_memory, fp16 instead of bf16)
    use_mps = torch.backends.mps.is_available()
    if use_mps:
        logger.info("MPS backend detected — using fp16, disabling pin_memory")

    # GRPO training config
    training_args = GRPOConfig(
        output_dir=output_path,
        # GRPO-specific params
        num_generations=grpo_cfg["num_generations"],
        num_iterations=grpo_cfg.get("num_iterations", 1),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high"),
        beta=grpo_cfg.get("beta", 0.0),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
        max_completion_length=grpo_cfg["max_completion_length"],
        temperature=grpo_cfg.get("temperature", 1.0),
        top_p=grpo_cfg.get("top_p", 1.0),
        mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", False),
        log_completions=grpo_cfg.get("log_completions", False),
        # Shared training params
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=not use_mps and train_cfg.get("bf16", False),
        fp16=use_mps or train_cfg.get("fp16", False),
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy="no",
        remove_unused_columns=False,
        report_to=train_cfg.get("report_to", "none"),
        dataloader_pin_memory=not use_mps,
        use_cpu=False,
    )

    # Load original base model and reconstruct by merging previous adapters
    dtype = torch.float16 if use_mps else torch.bfloat16
    logger.info("Loading original base model (iter_000)...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)

    # Merge all previous adapters (0..iter_num-1) to reconstruct current state
    for i in range(iter_num):
        prev_lora = str(lora_base / f"iter_{i:03d}")
        logger.info(f"  Merging previous adapter iter_{i:03d}...")
        model = PeftModel.from_pretrained(model, prev_lora)
        model = model.merge_and_unload()

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    logger.info(f"Saving adapter to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
