"""Reproduce the mode-collapse analysis from the 2026-05-16 findings entry.

Reads the eval rollouts from a `gcp_fetch_artifacts.sh`-style download
directory and prints:

  1. Per-iter `# MODE: main` usage rate (paper-relevant probe modality).
  2. Per-iter settle count (reward > 0 → 4-condition success bonus paid).
  3. Per-iter best and mean reward.

Usage:
    python docs/reproducibility/16f43b0_mode_collapse.py \\
        ~/Downloads/lora4vr_dedup-prod-l4-mb4/dedup-prod-l4-mb4

The path is the directory containing `iter_NNN/eval_samples.jsonl` and
`iter_NNN/eval_scores.jsonl`. The fetch script produces this layout.

Output (for run `dedup-prod-l4-mb4` at commit 16f43b0):

    eval_iter   n_main/N    settle/N    max_r     mean_r
        1        0/48        0/48      -10.8      -47.6
        2        0/48        0/48       -9.6      -32.0
        3        0/48        0/48       -9.5      -26.2
        4        0/48        0/48       -9.5      -15.0
        5        0/48        0/48      -10.7      -17.3
        6        0/48        1/48     +110.0      -13.7

The `n_main/N` column is the count of samples whose turn-1 assistant
message begins with `# MODE: main` (case-insensitive). In this run all
288 samples emit `# MODE: policy` for turn 1 — none probe.

For comparison, the Arm B frozen-baseline rollout from the 2026-05-10
findings entry shows 60/60 = 100% `# MODE: main` usage on the same
prompt template. So one iteration of GRPO collapsed the model from
100% probe-mode to 0%.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_MODE_MAIN_PATTERN = re.compile(r"#\s*MODE:\s*main", re.IGNORECASE)


def _turn1_text(sample: dict) -> str:
    """Return the first assistant message in the rollout transcript."""
    for msg in sample.get("transcript", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def main(run_dir: Path) -> None:
    print(f"{'eval_iter':<11}{'n_main/N':<13}{'settle/N':<13}{'max_r':<10}{'mean_r':<10}")
    for iter_dir in sorted(run_dir.glob("iter_*")):
        scores_path = iter_dir / "eval_scores.jsonl"
        samples_path = iter_dir / "eval_samples.jsonl"
        if not (scores_path.exists() and samples_path.exists()):
            continue

        scores = [json.loads(line) for line in scores_path.open()]
        samples = [json.loads(line) for line in samples_path.open()]

        rewards = [s["reward_total"] for s in scores]
        n_settled = sum(1 for r in rewards if r > 0)
        n_main = sum(1 for s in samples if _MODE_MAIN_PATTERN.search(_turn1_text(s)))
        N = len(scores)

        iter_num = int(iter_dir.name.split("_")[1])
        print(
            f"{iter_num:<11}{f'{n_main}/{N}':<13}{f'{n_settled}/{N}':<13}"
            f"{max(rewards):+9.1f} {sum(rewards) / N:+9.1f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <path-to-fetched-run-dir>")
    main(Path(sys.argv[1]).expanduser())
