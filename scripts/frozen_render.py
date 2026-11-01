"""Render frozen-model runs (Stage 1: one turn; Stage 2: probe run + controller)
into LaTeX fragments: one comparison table across runs, a per-third table, a
diagnostics table from the samples, and provenance macros per run.

Each run directory holds stage<N>_summary.json (with its provenance block),
stage<N>_samples.jsonl and the job log. Every fragment is headed by a comment
naming the result files, commits and jobs; the document cites nothing visibly.

  uv run python scripts/frozen_render.py --out docs/stage12/tables \\
      stage1-4b=docs/stage12/stage1-4b_<job> stage2-4b=docs/stage12/stage2-4b_<job> ...
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

THIRDS = ("low", "middle", "high")
RANGE = (0.08, 0.25)


def third_of(smax: float) -> str:
    k = min(2, max(0, int((smax - RANGE[0]) / (RANGE[1] - RANGE[0]) * 3)))
    return THIRDS[k]


def tex(s) -> str:
    """Escape a name for LaTeX text."""
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")


def pct(x) -> str:
    return "--" if x is None else f"{100 * x:.1f}\\%"


def num(x, d=1) -> str:
    return "--" if x is None else f"{x:.{d}f}".replace("-", "$-$")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def load(run_dir: Path) -> dict:
    summaries = sorted(run_dir.glob("stage*_summary.json"))
    samples = sorted(run_dir.glob("stage*_samples.jsonl"))
    if not summaries or not samples:
        raise FileNotFoundError(f"{run_dir}: need stage<N>_summary.json and stage<N>_samples.jsonl")
    S = json.loads(summaries[0].read_text())
    rows = [json.loads(l) for l in samples[0].read_text().splitlines() if l.strip()]
    logs = sorted(run_dir.glob("*.out"))
    prov = S.get("provenance", {})
    return {"summary": S, "rows": rows, "dir": run_dir, "log": logs[0] if logs else None,
            "commit": prov.get("commit", "unknown"), "job": prov.get("slurm", {}).get("job_id", "none"),
            "stage": S.get("stage", S.get("turns"))}


def outcome(r: dict) -> str:
    if not r.get("ran"):
        return "error"
    if r.get("success"):
        return "success"
    if r.get("steps", 0) >= 600:
        return "timeout"
    return "ruined"


def cut_off(r: dict) -> bool:
    """The last assistant message opens a code fence it never closes."""
    t = r.get("transcript") or []
    msgs = [m["content"] for m in t if m.get("role") == "assistant"]
    if not msgs:
        return False
    last = msgs[-1]
    return last.count("```") % 2 == 1


def diagnostics(run: dict) -> dict:
    rows, S = run["rows"], run["summary"]
    n = len(rows)
    d = {"n": n, "generated": mean(bool(r["generated"]) for r in rows),
         "cut_off": mean(cut_off(r) for r in rows),
         "outcomes": {k: sum(outcome(r) == k for r in rows) for k in ("success", "ruined", "timeout", "error")},
         "fast": mean(bool(r["fast"]) for r in rows), "reward": mean(r["reward"] for r in rows),
         "norm_time": mean(r["norm_time"] for r in rows),
         "ride_ratio": mean(r.get("ride_ratio") for r in rows),
         "steps_ruined": statistics.median([r["steps"] for r in rows if outcome(r) == "ruined"]) if any(outcome(r) == "ruined" for r in rows) else None}
    if run["stage"] == 2:
        probed = [r for r in rows if r.get("probed")]
        d["probe_rate"] = len(probed) / n if n else None
        d["probe_crashed"] = mean(("Traceback" in (r.get("tool_output") or "") or "Error" in (r.get("tool_output") or "")) for r in probed) if probed else None
        d["probe_cut_off"] = mean(((r.get("transcript") or [{}])[1].get("content", "").count("```") % 2 == 1) for r in probed) if probed else None
        ests = [r.get("estimate_error") for r in probed if r.get("estimate_error") is not None]
        d["estimate_within_0.01"] = (sum(e <= 0.01 for e in ests) / len(probed)) if probed else None
        d["estimate_rate"] = (len(ests) / len(probed)) if probed else None
        d["estimate_error_median"] = statistics.median(ests) if ests else None
    return d


def by_third(run: dict) -> dict:
    out = {}
    for th in THIRDS:
        rs = [r for r in run["rows"] if third_of(r["sigma_max"]) == th]
        out[th] = {"n": len(rs), "fast": mean(bool(r["fast"]) for r in rs), "success": mean(bool(r["success"]) for r in rs),
                   "reward": mean(r["reward"] for r in rs), "norm_time": mean(r["norm_time"] for r in rs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=dir, in the order the tables should list them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--macro-prefix", default="SF")
    args = ap.parse_args()
    runs = []
    for spec in args.runs:
        label, _, d = spec.partition("=")
        runs.append((label, load(Path(d))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = "% GENERATED by scripts/frozen_render.py. Do not edit; re-run the renderer.\n" + "".join(
        f"% {label}: {r['dir']}, code commit {r['commit']}, Slurm job {r['job']}, model {r['summary'].get('model')}, "
        f"{r['summary'].get('n_tasks')} tasks x {r['summary'].get('n_samples_per_task')} samples, max_tokens {r['summary'].get('max_tokens')}\n"
        for label, r in runs)

    def write(name: str, lines: list[str]) -> None:
        (out / name).write_text(header + "\n".join(lines) + "\n")

    P = args.macro_prefix
    macros = []
    for label, r in runs:
        words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
                 "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
        key = "".join(words.get(ch, ch) for ch in label if ch.isalnum())   # macro names: letters only
        macros += [f"\\newcommand{{\\{P}{key}commit}}{{{r['commit'][:7]}}}", f"\\newcommand{{\\{P}{key}job}}{{{r['job']}}}",
                   f"\\newcommand{{\\{P}{key}model}}{{{tex(str(r['summary'].get('model', '')).split('/')[-1])}}}"]
    first = runs[0][1]["summary"]
    macros += [f"\\newcommand{{\\{P}ntasks}}{{{first.get('n_tasks')}}}", f"\\newcommand{{\\{P}nsamples}}{{{first.get('n_samples_per_task')}}}",
               f"\\newcommand{{\\{P}maxtokens}}{{{first.get('max_tokens')}}}"]
    write("provenance.tex", macros)

    # ---- the comparison: one row per run, then the references from the first run -----------
    lines = ["\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule",
             "Run & Model & Valid & Success & Fast & Norm.\\ time & Reward & Ride ratio & Probe rate & Est.\\ error \\\\", "\\midrule"]
    for label, r in runs:
        S, d = r["summary"], diagnostics(r)
        lines.append(f"{tex(label)} & {tex(str(S.get('model', '')).split('/')[-1])} & {pct(d['generated'])} & {pct(S.get('success_rate'))} & "
                     f"{pct(S.get('fast_rate'))} & {num(S.get('mean_norm_time'), 3)} & {num(S.get('mean_reward'))} & "
                     f"{num(d['ride_ratio'], 2)} & {pct(S.get('probe_rate')) if r['stage'] == 2 else '--'} & "
                     f"{num(d.get('estimate_error_median'), 4) if r['stage'] == 2 else '--'} \\\\")
    lines.append("\\midrule")
    for key, label in (("blind", "cautious blind reference"), ("oracle", "oracle")):
        ref = first.get(key, {})
        lines.append(f"{label} & -- & -- & -- & {pct(ref.get('fast_rate'))} & {num(ref.get('mean_norm_time'), 3)} & {num(ref.get('mean_reward'))} & -- & -- & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("comparison.tex", lines)

    # ---- outcomes and failure diagnostics ---------------------------------------------------
    lines = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
             "Run & $n$ & Cut off & Success & Ruined & Timeout & Error & Probe crashed & Probe cut off & Est.\\ within 0.01 \\\\", "\\midrule"]
    for label, r in runs:
        d = diagnostics(r)
        o = d["outcomes"]
        lines.append(f"{tex(label)} & {d['n']} & {pct(d['cut_off'])} & {o['success']} & {o['ruined']} & {o['timeout']} & {o['error']} & "
                     f"{pct(d.get('probe_crashed')) if r['stage'] == 2 else '--'} & {pct(d.get('probe_cut_off')) if r['stage'] == 2 else '--'} & "
                     f"{pct(d.get('estimate_within_0.01')) if r['stage'] == 2 else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("diagnostics.tex", lines)

    # ---- by third of the limit's range -------------------------------------------------------
    lines = ["\\begin{tabular}{@{}ll" + "rrr" * len(runs) + "@{}}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{tex(label)}}}" for label, _ in runs) + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(runs))),
             "Third & $n$ & " + " & ".join("Fast & Succ. & Reward" for _ in runs) + " \\\\", "\\midrule"]
    bt = [by_third(r) for _, r in runs]
    for th in THIRDS:
        cells = " & ".join(f"{pct(b[th]['fast'])} & {pct(b[th]['success'])} & {num(b[th]['reward'])}" for b in bt)
        lines.append(f"{th} & {bt[0][th]['n']} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("by_third.tex", lines)
    print(f"wrote 4 fragments to {out} for {len(runs)} runs")


if __name__ == "__main__":
    main()
