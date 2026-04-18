"""Analyze and visualize probe results across training iterations.

Loads summary CSVs from data/probe/{run_name}_iter*/ and produces a
multi-panel figure covering reward distribution, crash rate, diversity
collapse, and per-scenario consistency.

Usage:
  uv run python scripts/analyze_probe.py --run_name puck_grpo_run_5
  uv run python scripts/analyze_probe.py --run_name puck_grpo_run_5 --output data/probe/analysis.png
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


ERROR_REWARD = -200.0  # per-scenario floor for crashes


def load_run(probe_base: Path, run_name: str) -> dict[int, dict]:
    """Return {iter_num: {rewards, crashed, codes, scenario_rewards}} sorted by iter."""
    pattern = re.compile(rf"^{re.escape(run_name)}_iter(\d+)_")
    runs = {}
    for d in sorted(probe_base.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        summary_path = d / "summary.csv"
        scenario_path = d / "per_scenario_rewards.csv"
        samples_path = d / "samples.json"
        if not summary_path.exists():
            continue

        rewards, crashed = [], []
        with summary_path.open() as f:
            for row in csv.DictReader(f):
                rewards.append(float(row["reward_total"]) if row["reward_total"] else float("nan"))
                crashed.append(row["ran_successfully"].strip().lower() != "true")

        # Per-scenario rewards (list of dicts scenario_id -> reward)
        scenario_rewards = []
        if scenario_path.exists():
            with scenario_path.open() as f:
                reader = csv.DictReader(f)
                scenario_cols = [c for c in reader.fieldnames if c not in ("sample", "status", "reward_total")]
                for row in reader:
                    sr = {c: float(row[c]) if row[c] else float("nan") for c in scenario_cols}
                    scenario_rewards.append(sr)

        # Policy code for diversity (hash each valid policy file)
        codes = []
        if samples_path.exists():
            samples = json.loads(samples_path.read_text())
            for s in samples:
                code = s.get("code") or ""
                codes.append(hashlib.md5(code.strip().encode()).hexdigest() if code.strip() else None)
        else:
            # Fall back to .py files
            for py in sorted(d.glob("sample_*.py")):
                codes.append(hashlib.md5(py.read_text().strip().encode()).hexdigest())

        runs[iter_num] = {
            "rewards": np.array(rewards),
            "crashed": np.array(crashed, dtype=bool),
            "codes": codes,
            "scenario_rewards": scenario_rewards,
        }
    return dict(sorted(runs.items()))


def unique_policy_count(codes: list) -> int:
    return len({c for c in codes if c is not None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--probe-dir", default="data/probe")
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/probe/{run_name}_analysis.png)")
    args = parser.parse_args()

    probe_base = Path(args.probe_dir)
    runs = load_run(probe_base, args.run_name)

    if not runs:
        raise SystemExit(f"No probe results found for '{args.run_name}' in {probe_base}")

    iters = sorted(runs.keys())
    print(f"Found {len(iters)} iterations: {iters}")

    # ------------------------------------------------------------------ #
    # Derived metrics per iteration
    # ------------------------------------------------------------------ #
    crash_rates, mean_rewards, best_rewards, reward_stds, unique_counts = [], [], [], [], []
    runner_rewards_per_iter = []  # only samples that ran successfully

    for k in iters:
        d = runs[k]
        r = d["rewards"]
        ok = ~d["crashed"]
        crash_rates.append(d["crashed"].mean())
        mean_rewards.append(np.nanmean(r))
        best_rewards.append(np.nanmax(r[ok]) if ok.any() else float("nan"))
        reward_stds.append(np.nanstd(r[ok]) if ok.any() else 0.0)
        unique_counts.append(unique_policy_count(d["codes"]))
        runner_rewards_per_iter.append(r[ok] if ok.any() else np.array([]))

    # ------------------------------------------------------------------ #
    # Figure layout: 2 rows × 3 cols
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{args.run_name} — policy probe analysis", fontsize=13, fontweight="bold")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(iters)))

    # ---- (0,0) Crash rate per iteration --------------------------------
    ax = axes[0, 0]
    bars = ax.bar(iters, [r * 100 for r in crash_rates], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Crash rate (%)")
    ax.set_title("Crash rate")
    ax.set_xticks(iters)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    for bar, v in zip(bars, crash_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8)

    # ---- (0,1) Reward distribution — violin per iteration --------------
    ax = axes[0, 1]
    data_for_violin = [r for r in runner_rewards_per_iter if len(r) > 0]
    iter_labels_violin = [k for k, r in zip(iters, runner_rewards_per_iter) if len(r) > 0]
    if data_for_violin:
        parts = ax.violinplot(data_for_violin, positions=iter_labels_violin,
                              showmedians=True, showextrema=True, widths=0.6)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total reward")
    ax.set_title("Reward distribution (runners only)")
    ax.set_xticks(iters)

    # ---- (0,2) Best vs mean reward per iteration -----------------------
    ax = axes[0, 2]
    ax.plot(iters, mean_rewards, "o-", color="steelblue", label="Mean (all samples)", linewidth=2)
    ax.plot(iters, best_rewards, "s--", color="seagreen", label="Best (runners)", linewidth=2)
    ax.fill_between(
        iters,
        [m - s for m, s in zip(mean_rewards, reward_stds)],
        [m + s for m, s in zip(mean_rewards, reward_stds)],
        alpha=0.15, color="steelblue", label="±1 std (runners)"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total reward")
    ax.set_title("Reward trajectory")
    ax.set_xticks(iters)
    ax.legend(fontsize=8)

    # ---- (1,0) Unique policies — diversity / collapse ------------------
    ax = axes[1, 0]
    n_samples = max(len(runs[k]["codes"]) for k in iters)
    ax.plot(iters, unique_counts, "o-", color="darkorange", linewidth=2, markersize=7)
    ax.axhline(n_samples, color="gray", linestyle=":", linewidth=1, label=f"Max ({n_samples})")
    ax.axhline(1, color="red", linestyle=":", linewidth=1, label="Collapsed (1)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Unique policies")
    ax.set_title("Output diversity (unique policy hashes)")
    ax.set_xticks(iters)
    ax.set_ylim(0, n_samples + 1)
    ax.legend(fontsize=8)

    # ---- (1,1) Per-scenario reward heatmap -----------------------------
    ax = axes[1, 1]
    # Build matrix: rows = iters, cols = scenarios, value = mean reward for that scenario
    all_scenario_ids = []
    for k in iters:
        for sr in runs[k]["scenario_rewards"]:
            for sid in sr:
                if sid not in all_scenario_ids:
                    all_scenario_ids.append(sid)
    matrix = np.full((len(iters), len(all_scenario_ids)), float("nan"))
    for i, k in enumerate(iters):
        srs = runs[k]["scenario_rewards"]
        if srs:
            for j, sid in enumerate(all_scenario_ids):
                vals = [sr[sid] for sr in srs if sid in sr and not np.isnan(sr[sid])]
                matrix[i, j] = np.mean(vals) if vals else float("nan")
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(matrix), vmax=min(0, np.nanmax(matrix)))
    ax.set_xticks(range(len(all_scenario_ids)))
    ax.set_xticklabels(all_scenario_ids, fontsize=8)
    ax.set_yticks(range(len(iters)))
    ax.set_yticklabels([f"iter {k}" for k in iters], fontsize=8)
    ax.set_title("Mean reward per scenario × iteration")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- (1,2) Reward std — spread within iteration --------------------
    ax = axes[1, 2]
    ax.bar(iters, reward_stds, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward std (runners)")
    ax.set_title("Within-iteration reward spread\n(0 = fully collapsed)")
    ax.set_xticks(iters)
    for i, (x, v) in enumerate(zip(iters, reward_stds)):
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = Path(args.output) if args.output else probe_base / f"{args.run_name}_analysis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
