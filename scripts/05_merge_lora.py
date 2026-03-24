"""Merge LoRA adapters into base model weights.

Reconstructs the model by loading the original base (iter_000) and
sequentially merging all adapters 0..iter, so we never need to store
full merged HF weights permanently.

Usage: python scripts/05_merge_lora.py --iter 0 --config configs/loop.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", default="configs/loop.yaml")
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    iter_num = args.iter

    base_path = str(Path(cfg["paths"]["hf_base_dir"]) / "iter_000")
    # Temp merged output goes under run_name to avoid conflicts
    hf_base = Path(cfg["paths"]["hf_base_dir"])
    if args.run_name:
        hf_base = hf_base / args.run_name
    output_path = str(hf_base / f"iter_{iter_num + 1:03d}")

    logger.info(f"Original base model: {base_path}")
    logger.info(f"Merging adapters 0..{iter_num} sequentially")
    logger.info(f"Output (merged): {output_path}")

    # Load original base on CPU in float32 for numerically safe merging
    logger.info("Loading original base model (iter_000) in float32...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Sequentially merge all adapters from 0 through iter_num
    lora_base = Path(cfg["paths"]["lora_dir"])
    if args.run_name:
        lora_base = lora_base / args.run_name
    for i in range(iter_num + 1):
        lora_path = str(lora_base / f"iter_{i:03d}")
        if not Path(lora_path).exists():
            logger.error(f"Adapter not found: {lora_path}")
            sys.exit(1)
        logger.info(f"  Applying adapter iter_{i:03d}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # Copy tokenizer from original base
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete.")


if __name__ == "__main__":
    main()
