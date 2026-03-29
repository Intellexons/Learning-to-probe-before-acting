"""Train LoRA adapter with DPO on preference pairs.

Usage: python scripts/04_train_dpo_lora.py --iter 0 --config configs/loop.yaml
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
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import load_jsonl

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

    base_model_path = str(Path(cfg["paths"]["hf_base_dir"]) / "iter_000")
    dpo_base = Path(cfg["paths"]["dpo_dir"])
    lora_base = Path(cfg["paths"]["lora_dir"])
    if args.run_name:
        dpo_base = dpo_base / args.run_name
        lora_base = lora_base / args.run_name
    dpo_data_path = str(dpo_base / f"iter_{iter_num:03d}_train.jsonl")
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
    logger.info(f"DPO data: {dpo_data_path}")
    logger.info(f"Output: {output_path}")

    # Load tokenizer from original base
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (TRL conversational format)
    raw_data = load_jsonl(dpo_data_path)
    if not raw_data:
        logger.error("No DPO training pairs found. Skipping training.")
        sys.exit(1)

    train_dataset = Dataset.from_list(raw_data)
    logger.info(f"Training on {len(train_dataset)} preference pairs")

    # Optionally load eval dataset
    eval_data_path = str(dpo_base / f"iter_{iter_num:03d}_eval.jsonl")
    eval_dataset = None
    if Path(eval_data_path).exists():
        eval_data = load_jsonl(eval_data_path)
        if eval_data:
            eval_dataset = Dataset.from_list(eval_data)
            logger.info(f"Eval set: {len(eval_dataset)} pairs")

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

    # DPO training config
    training_args = DPOConfig(
        output_dir=output_path,
        beta=train_cfg["beta"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_length=train_cfg["max_length"],
        max_prompt_length=train_cfg["max_prompt_length"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=not use_mps and train_cfg["bf16"],
        fp16=use_mps,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        loss_type=train_cfg["loss_type"],
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
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
    # ref_model=None -> TRL uses the frozen base (non-LoRA) weights as reference
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    logger.info(f"Saving adapter to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
