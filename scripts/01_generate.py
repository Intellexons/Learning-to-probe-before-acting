"""Generate N code samples per task using HuggingFace model directly.

Usage: python scripts/01_generate.py --iter 0 --config <config.yaml>
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hf_generator import load_model_for_generation, generate_batch
from src.prompts import load_template, render_prompt
from src.dataset import load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    iter_num = args.iter
    gen_cfg = cfg["generation"]

    # Load model with all previous adapters merged
    base_model_path = str(Path(cfg["paths"]["hf_base_dir"]) / "iter_000")
    model, tokenizer = load_model_for_generation(
        base_model_path=base_model_path,
        lora_base=cfg["paths"]["lora_dir"],
        iter_num=iter_num,
        run_name=args.run_name,
    )

    logger.info(f"Generating with base + {iter_num} merged adapters")
    template = load_template("configs/prompt_template.txt")
    tasks = load_jsonl(cfg["paths"]["tasks_train"])

    all_samples = []
    for task in tasks:
        task_id = task["task_id"]
        prompt = render_prompt(template, task)
        logger.info(f"Task {task_id}: generating {gen_cfg['n_samples_per_task']} samples")

        results = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n_samples=gen_cfg["n_samples_per_task"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_tokens"],
        )

        for i, result in enumerate(results):
            sample = {
                "iter": iter_num,
                "task_id": task_id,
                "sample_id": i,
                "prompt": prompt,
                "code": result["code"],
                "generation_success": result["success"],
                "generation_error": result["error"],
            }
            all_samples.append(sample)

    # Save
    rollouts_base = Path(cfg["paths"]["rollouts_dir"])
    if args.run_name:
        rollouts_base = rollouts_base / args.run_name
    out_dir = rollouts_base / f"iter_{iter_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "samples.jsonl")
    save_jsonl(all_samples, out_path)

    n_success = sum(1 for s in all_samples if s["generation_success"])
    logger.info(
        f"Saved {len(all_samples)} samples ({n_success} successful) to {out_path}"
    )


if __name__ == "__main__":
    main()
