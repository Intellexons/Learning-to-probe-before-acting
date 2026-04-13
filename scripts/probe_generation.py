"""Probe: what does the model actually generate, and where does it get lost?

Runs the full generation → extraction → validation → scoring pipeline on the
puck training task (or any task JSONL pointed at by the config), using the
exact same code paths as GRPO's reward function, and prints every step for
every sample. Read the output top-to-bottom to see whether the model is
writing reasonable code or nonsense, whether `extract_code` is preserving it,
whether `validate_code` is rejecting it, and what the per-scenario scores
look like.

Usage on a GPU VM (base model, no adapters):
  uv run python scripts/probe_generation.py --config configs/loop_grpo.yaml --iter 0 --n 4

With merged adapters from a previous run:
  uv run python scripts/probe_generation.py \
      --config configs/loop_grpo.yaml \
      --iter 10 --run_name puck_grpo_run_3 --n 4

This does NOT train anything and does NOT write to W&B. It only reads weights
and prints. Safe to run against an idle VM.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import load_jsonl
from src.hf_generator import extract_code, generate_code, load_model_for_generation
from src.prompts import load_template, render_prompt
from src.runner import score_sample, validate_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _hline(char: str = "-", n: int = 78) -> str:
    return char * n


def _show_block(label: str, body: str, max_chars: int = 4000) -> None:
    print(_hline("="))
    print(label)
    print(_hline("="))
    if len(body) > max_chars:
        print(body[:max_chars])
        print(f"... [truncated, {len(body) - max_chars} more chars]")
    else:
        print(body)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config (for paths + generation params)")
    parser.add_argument("--iter", type=int, default=0, help="Merge adapters 0..iter-1 (0 = base only)")
    parser.add_argument("--run_name", default=None, help="Run name scoping for adapter lookup")
    parser.add_argument("--n", type=int, default=4, help="Number of completions to generate")
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Which training task to probe (default: first task)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Load first (or N-th) training task
    tasks = load_jsonl(cfg["paths"]["tasks_train"])
    if not tasks:
        raise SystemExit("No tasks found in tasks_train")
    task = tasks[args.task_index]
    print(f"Probing task: {task['task_id']} (env_class={task.get('env_class')})")
    print(f"Scenarios: {len(task['scenarios'])}")

    # Build the prompt exactly as the GRPO trainer would
    template = load_template("configs/prompt_template.txt")
    prompt_text = render_prompt(template, task)
    _show_block(f"RENDERED PROMPT ({len(prompt_text)} chars)", prompt_text)

    # Load the model the same way generation step does: base + merged adapters
    base_model_path = str(Path(cfg["paths"]["hf_base_dir"]) / "iter_000")
    lora_base = cfg["paths"].get("lora_dir", "models/lora")
    model, tokenizer = load_model_for_generation(
        base_model_path=base_model_path,
        lora_base=lora_base,
        iter_num=args.iter,
        run_name=args.run_name,
    )

    # Generation params — match the GRPO config for apples-to-apples
    grpo_cfg = cfg.get("grpo", {})
    temperature = grpo_cfg.get("temperature", 0.8)
    top_p = grpo_cfg.get("top_p", 0.95)
    max_tokens = grpo_cfg.get("max_completion_length", 600)

    print()
    print(f"Generating {args.n} completions at temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    print()

    scoring_cfg = cfg["scoring"]
    summary_rows = []

    for i in range(args.n):
        print()
        print(_hline("#"))
        print(f"# SAMPLE {i + 1}/{args.n}")
        print(_hline("#"))

        result = generate_code(
            model, tokenizer, prompt_text,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        raw = result.get("raw_response", "")
        _show_block("RAW MODEL OUTPUT", raw)

        if not result["success"]:
            print(f"generate_code reported failure: {result.get('error')}")
            summary_rows.append((i, "gen_fail", 0.0, False, result.get("error")))
            continue

        code = result["code"]
        _show_block("EXTRACTED CODE (after extract_code)", code)

        is_valid, err_msg = validate_code(code)
        print(f"validate_code: is_valid={is_valid}  error={err_msg!r}")
        print()

        score = score_sample(
            code, task,
            timeout=scoring_cfg["timeout_seconds"],
            max_steps=scoring_cfg["max_steps"],
        )
        print(f"score_sample reward_total={score['reward_total']:.2f} "
              f"ran_successfully={score['ran_successfully']}")
        print(f"  per_scenario: {score['reward_per_scenario']}")
        if score["errors"]:
            print(f"  errors: {score['errors']}")

        summary_rows.append((
            i,
            "valid" if is_valid else "invalid",
            score["reward_total"],
            score["ran_successfully"],
            (score["errors"][:1] or [None])[0],
        ))

    # Summary table
    print()
    print(_hline("="))
    print("SUMMARY")
    print(_hline("="))
    print(f"{'#':>3} {'status':>8} {'reward':>12} {'ran_ok':>7}  first_error")
    print(_hline())
    for row in summary_rows:
        i, status, rew, ok, err = row
        err_s = (err[:60] if err else "-")
        print(f"{i:>3} {status:>8} {rew:>12.2f} {str(ok):>7}  {err_s}")


if __name__ == "__main__":
    main()
