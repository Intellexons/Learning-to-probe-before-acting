"""Generate N code samples per task using Ollama.

Usage: python scripts/01_generate.py --iter 0 --config configs/loop.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ollama_client import OllamaClient
from src.prompts import load_template, render_prompt
from src.dataset import load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--config", default="configs/loop.yaml")
    parser.add_argument("--ollama_model", default=None, help="Override Ollama model name")
    parser.add_argument("--run_name", default=None, help="Run name for output directories")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    iter_num = args.iter

    # Determine Ollama model
    if args.ollama_model:
        ollama_model = args.ollama_model
    elif iter_num == 0:
        ollama_model = cfg["base_model"]["ollama"]
    else:
        prefix = cfg["loop"]["ollama_model_prefix"]
        ollama_model = f"{prefix}:iter{iter_num:03d}"

    gen_cfg = cfg["generation"]
    client = OllamaClient(
        base_url=gen_cfg["ollama_base_url"],
        schema_path="configs/schema.json",
    )

    if not client.check_health():
        logger.error("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    logger.info(f"Using model: {ollama_model}")
    template = load_template("configs/prompt_template.txt")
    tasks = load_jsonl(cfg["paths"]["tasks_train"])

    all_samples = []
    for task in tasks:
        task_id = task["task_id"]
        prompt = render_prompt(template, task)
        logger.info(f"Task {task_id}: generating {gen_cfg['n_samples_per_task']} samples")

        results = client.generate_batch(
            model=ollama_model,
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
