"""Progress tracking — appends one row per iteration to CSV files.

Files (per run):
  data/progress/{run_name}/train.csv  — train generation + scoring stats per iteration
  data/progress/{run_name}/eval.csv   — eval stats per iteration

These give you a single-file view of whether the model is improving.
"""
import csv
from datetime import datetime
from pathlib import Path

PROGRESS_DIR = "data/progress"

TRAIN_FIELDS = [
    "timestamp", "iter", "n_samples", "n_success", "success_rate",
    "mean_reward", "max_reward",
]
EVAL_FIELDS = [
    "timestamp", "iter", "model", "n_samples", "n_success", "success_rate",
    "mean_reward", "mean_best_reward",
]


def _get_csv_paths(run_name: str | None) -> tuple[str, str]:
    """Return (train_csv, eval_csv) paths for the given run."""
    if run_name:
        base = Path(PROGRESS_DIR) / run_name
    else:
        base = Path(PROGRESS_DIR)
    return str(base / "train.csv"), str(base / "eval.csv")


def _append_row(path: str, fields: list[str], row: dict):
    """Append a row to a CSV, writing header if file is new."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_train_progress(
    iter_num: int,
    n_samples: int,
    n_success: int,
    mean_reward: float,
    max_reward: float,
    run_name: str | None = None,
):
    train_csv, _ = _get_csv_paths(run_name)
    _append_row(train_csv, TRAIN_FIELDS, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iter": iter_num,
        "n_samples": n_samples,
        "n_success": n_success,
        "success_rate": f"{n_success / n_samples:.2%}" if n_samples else "0%",
        "mean_reward": f"{mean_reward:.1f}",
        "max_reward": f"{max_reward:.1f}",
    })


def append_eval_progress(
    iter_num: int,
    model_name: str,
    n_samples: int,
    n_success: int,
    mean_reward: float,
    mean_best_reward: float,
    run_name: str | None = None,
):
    _, eval_csv = _get_csv_paths(run_name)
    _append_row(eval_csv, EVAL_FIELDS, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iter": iter_num,
        "model": model_name,
        "n_samples": n_samples,
        "n_success": n_success,
        "success_rate": f"{n_success / n_samples:.2%}" if n_samples else "0%",
        "mean_reward": f"{mean_reward:.1f}",
        "mean_best_reward": f"{mean_best_reward:.1f}",
    })


def print_progress_table(run_name: str | None = None):
    """Print both CSVs as formatted tables to stdout."""
    train_csv, eval_csv = _get_csv_paths(run_name)
    if run_name:
        print(f"\nRun: {run_name}")
    for label, path in [("TRAIN", train_csv), ("EVAL", eval_csv)]:
        p = Path(path)
        if not p.exists():
            print(f"\n{label}: no data yet")
            continue
        print(f"\n{'=' * 70}")
        print(f" {label} PROGRESS")
        print(f"{'=' * 70}")
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print("  (empty)")
            continue
        # Print header
        fields = list(rows[0].keys())
        header = " | ".join(f"{f:>15}" for f in fields)
        print(header)
        print("-" * len(header))
        for row in rows:
            print(" | ".join(f"{row[f]:>15}" for f in fields))
