"""Render the record of a training run into LaTeX fragments.

A training run (`jobs/hpi/train.sbatch`, `scripts/09_run_loop.py`) leaves, per
iteration, the trainer's per-step record `progress/trainer_log_iter_NNN.json`,
next to its config and its log. From these this script writes

  record.tex      one row per iteration: how many first replies were followed
                  by a second reply (probe runs, and the rare tool call that
                  could not be acted on), the share of conversations that
                  ended at the error floor, the mean reward, the time per
                  step and the peak GPU memory;
  phases.tex      the same quantities over groups of iterations, for the text;
  timing.tex      (with --timing) the steps of a timing run, one row each;
  provenance.tex  macros: commit, job, steps, iterations, hours.

Every number comes from the result files; every fragment is headed by a
comment naming them, the commit and the job.

  uv run python scripts/train_render.py --run docs/training/<run>/train \\
      --timing docs/train_timing/<job dir> --out docs/training/<run>/tables
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

# The groups of iterations of phases.tex (1-based, inclusive): the first five one by one,
# then three stretches.
PHASES = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 15), (16, 10**6))


def steps_of(log_file: Path) -> list[dict]:
    """The per-step entries of one iteration's trainer record."""
    return [e for e in json.loads(log_file.read_text()) if "step_time" in e]


def iteration_row(steps: list[dict], g1: int) -> dict:
    """One iteration, from its steps. A step is one task: `g1` first replies, and one more
    generate call for every first reply that did not end the task, so `generate_calls - 1`
    counts the first replies that were followed by a second one."""
    f = lambda key: [float(e[key]) for e in steps]  # noqa: E731
    n = len(steps)
    return {
        "steps": n,
        "first_replies": g1 * n,
        "went_on": round(sum(f("branched/generate_calls")) - n),
        "floor": sum(f("branched/frac_floor")) / n,
        "reward": sum(f("reward")) / n,
        "step_seconds": sum(f("step_time")) / n,
        "seconds": sum(f("step_time")),
        "peak_gib": max(f("mem/peak_step_gib")),
    }


def pool(rows: list[dict]) -> dict:
    """Several iterations as one row: counts and seconds add up, means are weighted by steps."""
    n = sum(r["steps"] for r in rows)
    w = lambda key: sum(r[key] * r["steps"] for r in rows) / n  # noqa: E731
    return {"steps": n, "first_replies": sum(r["first_replies"] for r in rows),
            "went_on": sum(r["went_on"] for r in rows), "floor": w("floor"), "reward": w("reward"),
            "step_seconds": w("step_seconds"), "seconds": sum(r["seconds"] for r in rows),
            "peak_gib": max(r["peak_gib"] for r in rows)}


def load_run(run_dir: Path) -> dict:
    cfg_files = sorted(run_dir.glob("*.yaml"))
    logs = sorted(run_dir.glob("*.out"))
    files = sorted((run_dir / "progress").glob("trainer_log_iter_*.json"))
    if not cfg_files or not files:
        raise FileNotFoundError(f"{run_dir}: need the config (*.yaml) and progress/trainer_log_iter_*.json")
    cfg = yaml.safe_load(cfg_files[0].read_text())
    head = {"commit": "unknown", "job": "none", "seconds": None}
    if logs:
        text = logs[0].read_text(errors="replace")
        m = re.search(r"^job (\d+) on \S+, start \S+, commit ([0-9a-f]+)", text, re.M)
        if m:
            head["job"], head["commit"] = m.group(1), m.group(2)
        m = re.search(r"^done \S+, (\d+)s", text, re.M)
        if m:
            head["seconds"] = int(m.group(1))
    g1 = int(cfg.get("branched", {}).get("g1", 1))
    rows = [iteration_row(steps_of(f), g1) for f in files]
    return {"dir": run_dir, "config": cfg_files[0], "cfg": cfg, "g1": g1, "g2": int(cfg.get("branched", {}).get("g2", 1)),
            "rows": rows, "log": logs[0] if logs else None, **head}


def num(x: float, d: int = 1) -> str:
    s = f"{abs(x):.{d}f}"
    return f"$-${s}" if x < 0 and float(s) != 0 else s


def record_line(label: str, r: dict) -> str:
    return (f"{label} & {r['steps']} & {r['went_on']} of {r['first_replies']} & {100 * r['floor']:.0f}\\% & "
            f"{num(r['reward'])} & {r['step_seconds']:.0f} & {r['peak_gib']:.1f} \\\\")


RECORD_HEAD = ("Iteration & Steps & First replies followed by a second & At the error floor & Reward & "
               "Seconds per step & Peak memory (GiB) \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run's record: config, progress/, log")
    ap.add_argument("--timing", default=None, help="the result directory of a timing run (jobs/hpi/train_timing.sbatch)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--macro-prefix", default="TR")
    args = ap.parse_args()
    run = load_run(Path(args.run))
    timing = load_run_timing(Path(args.timing)) if args.timing else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = ("% GENERATED by scripts/train_render.py. Do not edit; re-run the renderer.\n"
              f"% run: {run['dir']}, code commit {run['commit']}, Slurm job {run['job']}, config {run['config'].name}, "
              f"{len(run['rows'])} iterations, {sum(r['steps'] for r in run['rows'])} steps\n")
    if timing:
        header += (f"% timing run: {timing['dir']}, code commit {timing['commit']}, Slurm job {timing['job']}, "
                   f"{len(timing['steps'])} steps\n")

    def write(name: str, lines: list[str]) -> None:
        (out / name).write_text(header + "\n".join(lines) + "\n")

    rows = run["rows"]
    lines = ["\\begin{tabular}{@{}rrrrrrr@{}}", "\\toprule", RECORD_HEAD, "\\midrule"]
    lines += [record_line(str(i + 1), r) for i, r in enumerate(rows)]
    lines += ["\\midrule", record_line("all", pool(rows)), "\\bottomrule", "\\end{tabular}"]
    write("record.tex", lines)

    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", RECORD_HEAD.replace("Iteration &", "Iterations &"), "\\midrule"]
    for lo, hi in PHASES:
        part = rows[lo - 1:hi]
        if not part:
            continue
        last = min(hi, len(rows))
        lines.append(record_line(str(lo) if lo == last else f"{lo}--{last}", pool(part)))
    lines += ["\\midrule", record_line("all", pool(rows)), "\\bottomrule", "\\end{tabular}"]
    write("phases.tex", lines)

    n_fragments = 2
    if timing:
        lines = ["\\begin{tabular}{@{}rrrrr@{}}", "\\toprule",
                 "Step & Conversations & Seconds, rollout and update & Peak memory (GiB) & At the error floor \\\\", "\\midrule"]
        for i, e in enumerate(timing["steps"]):
            lines.append(f"{i + 1} & {timing['g1'] * timing['g2']} & {float(e['step_time']):.0f} & "
                         f"{float(e['mem/peak_step_gib']):.1f} & {100 * float(e['branched/frac_floor']):.0f}\\% \\\\")
        lines += ["\\midrule",
                  f"\\multicolumn{{5}}{{@{{}}l}}{{training time {timing['train_runtime']:.0f} s for {len(timing['steps'])} steps: "
                  f"{timing['train_runtime'] / len(timing['steps']):.0f} s per step, logging included}} \\\\",
                  "\\bottomrule", "\\end{tabular}"]
        write("timing.tex", lines)
        n_fragments += 1

    P = args.macro_prefix
    total = pool(rows)
    best = min(range(len(rows)), key=lambda i: -rows[i]["reward"])
    macros = [
        f"\\newcommand{{\\{P}commit}}{{{run['commit'][:7]}}}", f"\\newcommand{{\\{P}job}}{{{run['job']}}}",
        f"\\newcommand{{\\{P}iterations}}{{{len(rows)}}}", f"\\newcommand{{\\{P}steps}}{{{total['steps']}}}",
        f"\\newcommand{{\\{P}stepsperiteration}}{{{rows[0]['steps']}}}",
        f"\\newcommand{{\\{P}gone}}{{{run['g1']}}}", f"\\newcommand{{\\{P}gtwo}}{{{run['g2']}}}",
        f"\\newcommand{{\\{P}conversations}}{{{run['g1'] * run['g2']}}}",
        f"\\newcommand{{\\{P}maxturntokens}}{{{run['cfg'].get('branched', {}).get('max_turn_tokens')}}}",
        f"\\newcommand{{\\{P}rank}}{{{run['cfg'].get('lora', {}).get('r')}}}",
        f"\\newcommand{{\\{P}bestiteration}}{{{best + 1}}}", f"\\newcommand{{\\{P}bestreward}}{{{num(rows[best]['reward'])}}}",
        f"\\newcommand{{\\{P}lastreward}}{{{num(rows[-1]['reward'])}}}",
    ]
    if run["seconds"] is not None:
        macros.append(f"\\newcommand{{\\{P}hours}}{{{run['seconds'] / 3600:.1f}}}")
    if timing:
        macros += [f"\\newcommand{{\\{P}timingcommit}}{{{timing['commit'][:7]}}}",
                   f"\\newcommand{{\\{P}timingjob}}{{{timing['job']}}}"]
    write("provenance.tex", macros)
    print(f"wrote {n_fragments + 1} fragments to {out}")


def load_run_timing(d: Path) -> dict:
    """A timing run: one trainer record, its config and its log."""
    files = sorted(d.glob("trainer_log_iter_*.json"))
    cfg_files = sorted(d.glob("*.yaml"))
    logs = sorted(d.glob("*.out"))
    if not files or not cfg_files:
        raise FileNotFoundError(f"{d}: need trainer_log_iter_*.json and the config (*.yaml)")
    entries = json.loads(files[0].read_text())
    cfg = yaml.safe_load(cfg_files[0].read_text())
    head = {"commit": "unknown", "job": "none"}
    if logs:
        m = re.search(r"^job (\d+) on \S+, start \S+, commit ([0-9a-f]+)", logs[0].read_text(errors="replace"), re.M)
        if m:
            head["job"], head["commit"] = m.group(1), m.group(2)
    runtime = next(float(e["train_runtime"]) for e in entries if "train_runtime" in e)
    return {"dir": d, "steps": [e for e in entries if "step_time" in e], "train_runtime": runtime,
            "g1": int(cfg["branched"]["g1"]), "g2": int(cfg["branched"]["g2"]), **head}


if __name__ == "__main__":
    main()
