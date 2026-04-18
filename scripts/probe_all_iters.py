"""Run probe_generation.py for every available iteration of a training run.

Discovers adapter checkpoints under models/lora/{run_name}/iter_NNN/ and runs
the probe for each one, saving results to data/probe/ as usual.

Usage:
  uv run python scripts/probe_all_iters.py --run_name puck_grpo_run_5 --n 20
  uv run python scripts/probe_all_iters.py --run_name puck_grpo_run_5 --n 20 --iters 0 2 4
  uv run python scripts/probe_all_iters.py --run_name puck_grpo_run_5 --n 20 --skip-existing
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def find_iters(run_name: str, lora_dir: str) -> list[int]:
    run_path = Path(lora_dir) / run_name
    if not run_path.exists():
        raise SystemExit(f"Run directory not found: {run_path}")
    iters = sorted(
        int(p.name.split("_")[1])
        for p in run_path.iterdir()
        if p.is_dir() and p.name.startswith("iter_")
    )
    return iters


def probe_dir_exists(run_name: str, iter_num: int, temperature: float, probe_base: str) -> bool:
    temp_str = f"{temperature:.2f}".replace(".", "p")
    name = f"{run_name}_iter{iter_num:03d}_t{temp_str}"
    summary = Path(probe_base) / name / "summary.csv"
    return summary.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/loop_grpo.yaml")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--n", type=int, default=20, help="Samples per iteration")
    parser.add_argument("--iters", type=int, nargs="+", default=None,
                        help="Specific iterations to probe (default: all found)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip iterations where probe output already exists")
    parser.add_argument("--output-dir", default="data/probe")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    lora_dir = cfg["paths"].get("lora_dir", "models/lora")
    temperature = cfg.get("grpo", {}).get("temperature", 0.8)

    iters = args.iters if args.iters is not None else find_iters(args.run_name, lora_dir)
    # Always include iter 0 (base model)
    if 0 not in iters:
        iters = [0] + iters

    print(f"Run: {args.run_name}")
    print(f"Iterations to probe: {iters}")
    print(f"Samples per iter: {args.n}")
    print()

    for iter_num in iters:
        if args.skip_existing and probe_dir_exists(args.run_name, iter_num, temperature, args.output_dir):
            print(f"[iter {iter_num:03d}] Already exists — skipping (--skip-existing)")
            continue

        print(f"{'=' * 60}")
        print(f"[iter {iter_num:03d}] Running probe...")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable, "scripts/probe_generation.py",
            "--config", args.config,
            "--iter", str(iter_num),
            "--run_name", args.run_name,
            "--n", str(args.n),
            "--output-dir", args.output_dir,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[iter {iter_num:03d}] WARNING: probe exited with code {result.returncode}")

    print()
    print("All done. Results in:", args.output_dir)


if __name__ == "__main__":
    main()
