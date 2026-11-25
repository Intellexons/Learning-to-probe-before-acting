"""Render frozen-model runs (Stage 1: one turn; Stage 2: probe run + controller)
into LaTeX fragments: one comparison table across runs, a per-third table, a
diagnostics table from the samples, an error-kind table (why the samples that
did not run failed, and three code patterns counted over all samples), a table
of what the first reply of each Stage 2 sample is (the replies counted as
probe code, and where the controllers mention the probe environment), and
provenance macros per run. With --tokenizer, one more: the length of the first
replies in tokens against a list of budgets, next to the token counts of the
hand-written probe code and controller of Stage 0.

Each run directory holds stage<N>_summary.json (with its provenance block),
stage<N>_samples.jsonl and the job log. Every fragment is headed by a comment
naming the result files, commits and jobs; the document cites nothing visibly.

  uv run python scripts/frozen_render.py --out docs/stage12/tables \\
      stage1-4b=docs/stage12/stage1-4b_<job> stage2-4b=docs/stage12/stage2-4b_<job> ...
  uv run python scripts/frozen_render.py --out docs/stage12/tables_budget \\
      --tokenizer models/hf_base/iter_000/tokenizer.json --macro-prefix SFT ...
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
from pathlib import Path

THIRDS = ("low", "middle", "high")
RANGE = (0.08, 0.25)
TOKEN_BUDGETS = (750, 1000, 1250, 1500, 2000)


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


def last_reply(r: dict) -> str:
    msgs = [m["content"] for m in (r.get("transcript") or []) if m.get("role") == "assistant"]
    return msgs[-1] if msgs else ""


def first_reply(r: dict) -> str:
    msgs = [m["content"] for m in (r.get("transcript") or []) if m.get("role") == "assistant"]
    return msgs[0] if msgs else ""


def diagnostics(run: dict) -> dict:
    rows, S = run["rows"], run["summary"]
    n = len(rows)
    d = {"n": n, "generated": mean(bool(r["generated"]) for r in rows),
         "cut_off": mean(cut_off(r) for r in rows),
         "reply_chars_median": statistics.median(len(last_reply(r)) for r in rows) if rows else None,
         "outcomes": {k: sum(outcome(r) == k for r in rows) for k in ("success", "ruined", "timeout", "error")},
         "fast": mean(bool(r["fast"]) for r in rows), "reward": mean(r["reward"] for r in rows),
         "norm_time": mean(r["norm_time"] for r in rows),
         "ride_ratio": mean(r.get("ride_ratio") for r in rows),
         "steps_ruined": statistics.median([r["steps"] for r in rows if outcome(r) == "ruined"]) if any(outcome(r) == "ruined" for r in rows) else None}
    if run["stage"] == 2:
        probed = [r for r in rows if r.get("probed")]
        d["probe_rate"] = len(probed) / n if n else None
        d["probe_crashed"] = mean(("Traceback" in (r.get("tool_output") or "") or "Error" in (r.get("tool_output") or "")) for r in probed) if probed else None
        d["probe_cut_off"] = mean((first_reply(r).count("```") % 2 == 1) for r in probed) if probed else None
        d["probe_chars_range"] = (min(len(first_reply(r)) for r in probed), max(len(first_reply(r)) for r in probed)) if probed else None
        ests = [r.get("estimate_error") for r in probed if r.get("estimate_error") is not None]
        d["estimate_within_0.01"] = (sum(e <= 0.01 for e in ests) / len(probed)) if probed else None
        d["estimate_rate"] = (len(ests) / len(probed)) if probed else None
        d["estimate_error_median"] = statistics.median(ests) if ests else None
    return d


ERROR_KINDS = (
    "syntax error (reply cut off or malformed)",
    "forbidden name at validation",
    "reads state['done'], ['showered'] or ['failure_reason']",
    "calls env in the controller",
    "undefined name (NameError)",
    "missing state key (KeyError)",
    "local used before assignment",
    "sandbox output parse error (the bug fixed on 2026-11-01)",
    "other runtime error",
)
FLAG_KEYS = ("done", "showered", "failure_reason")


def error_kind(r: dict) -> str | None:
    """Why a sample that did not run failed, in a few fixed kinds; None if it ran."""
    import re
    if r.get("ran"):
        return None
    if not r.get("generated"):
        g = str(r.get("generation_error") or "")
        if "Syntax error" in g:
            return ERROR_KINDS[0]
        if "Forbidden" in g:
            return ERROR_KINDS[1]
        return ERROR_KINDS[8]
    e = str((r.get("errors") or [""])[0])
    e = re.sub(r"^s\d+: ", "", e)
    if "Output parse error" in e:
        return ERROR_KINDS[7]
    if re.search(r"name 'env' is not defined|local variable 'env'", e):
        return ERROR_KINDS[3]
    if re.fullmatch(r"policy\(\) error: '(%s)'" % "|".join(FLAG_KEYS), e):
        return ERROR_KINDS[2]
    if "is not defined" in e:
        return ERROR_KINDS[4]
    if re.fullmatch(r"policy\(\) error: '[^']*'", e):
        return ERROR_KINDS[5]
    if "cannot access local variable" in e:
        return ERROR_KINDS[6]
    return ERROR_KINDS[8]


CODE_PATTERNS = (
    ("controllers that call env.reset or env.step", r"\benv\.(reset|step)\b"),
    ("controllers that read state['done'], ['showered'] or ['failure_reason']",
     r"state\[\s*['\"](done|showered|failure_reason)['\"]\s*\]"),
    ("controllers that print", r"\bprint\("),
)


def error_table(run: dict) -> dict:
    """Counts per error kind over the samples that did not run, and per code
    pattern over all samples (the controller code, whether it ran or not)."""
    import re
    rows = run["rows"]
    kinds = {k: 0 for k in ERROR_KINDS}
    for r in rows:
        k = error_kind(r)
        if k is not None:
            kinds[k] += 1
    patterns = {name: sum(bool(re.search(rx, r.get("code") or "")) for r in rows) for name, rx in CODE_PATTERNS}
    return {"kinds": kinds, "patterns": patterns, "n": len(rows), "errors": sum(kinds.values())}


def by_third(run: dict) -> dict:
    out = {}
    for th in THIRDS:
        rs = [r for r in run["rows"] if third_of(r["sigma_max"]) == th]
        out[th] = {"n": len(rs), "fast": mean(bool(r["fast"]) for r in rs), "success": mean(bool(r["success"]) for r in rs),
                   "reward": mean(r["reward"] for r in rs), "norm_time": mean(r["norm_time"] for r in rs)}
    return out


ENV_CALL = r"\benv\.(reset|step)\b"
FIRST_REPLY_ROWS = (
    ("probe", "first replies counted as probe code (the probe rate's numerator)"),
    ("probe_cut_off", "\\quad cut off by the token budget"),
    ("probe_rejected", "\\quad rejected by the sandbox before running"),
    ("probe_def_policy", "\\quad hold a top-level \\texttt{def policy} (a controller that did not parse)"),
    ("probe_env_call", "\\quad call \\texttt{env.reset} or \\texttt{env.step} outside comments"),
    ("probe_printed", "\\quad ran and printed output"),
    None,
    ("controller", "first replies taken as the controller"),
    ("ctrl_env", "\\quad mention \\texttt{env.reset} or \\texttt{env.step}"),
    ("ctrl_env_policy", "\\qquad inside \\texttt{policy}"),
    ("ctrl_env_top", "\\qquad at top level"),
    ("ctrl_env_other", "\\qquad in another function"),
    ("ctrl_env_comment", "\\qquad in comments only"),
)


def env_call_site(code: str) -> str:
    """Where a controller's code calls env.reset or env.step: 'policy', 'top'
    (a top-level statement), 'other' (another function or class), or 'comment'
    (the names appear in the text but not in the parsed code). One site per
    controller, in that order of precedence."""
    def calls(node: ast.AST) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr in ("reset", "step")
                   and isinstance(n.value, ast.Name) and n.value.id == "env" for n in ast.walk(node))
    try:
        body = ast.parse(code).body
    except SyntaxError:
        return "comment"
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if any(isinstance(n, defs) and n.name == "policy" and calls(n) for n in body):
        return "policy"
    if any(not isinstance(n, defs) and calls(n) for n in body):
        return "top"
    if any(isinstance(n, defs) and calls(n) for n in body):
        return "other"
    return "comment"


def first_reply_table(run: dict) -> dict:
    """What the first reply of a Stage 2 sample is. A reply is counted as probe
    code when runner.defines_policy is false for it, which is also the case for
    a controller that does not parse; the rows separate the two."""
    rows = run["rows"]
    probed = [r for r in rows if r.get("probed")]
    ctrl = [r for r in rows if r.get("probed") is False]
    d = {"n": len(rows), "probe": len(probed), "controller": len(ctrl)}

    def uncommented(code: str) -> str:
        return "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))

    tool = lambda r: (r.get("tool_output") or "").strip()
    d["probe_cut_off"] = sum(first_reply(r).count("```") % 2 == 1 for r in probed)
    d["probe_rejected"] = sum(tool(r).startswith("[main rejected") for r in probed)
    d["probe_def_policy"] = sum(bool(re.search(r"(?m)^def policy\s*\(", r.get("t1_code") or "")) for r in probed)
    d["probe_env_call"] = sum(bool(re.search(ENV_CALL, uncommented(r.get("t1_code") or ""))) for r in probed)
    d["probe_printed"] = sum(bool(tool(r)) and not tool(r).startswith("[main rejected") and tool(r) != "[empty output]" for r in probed)
    sites = [env_call_site(r.get("t1_code") or "") for r in ctrl if re.search(ENV_CALL, r.get("t1_code") or "")]
    d["ctrl_env"] = len(sites)
    for key, site in (("ctrl_env_policy", "policy"), ("ctrl_env_top", "top"), ("ctrl_env_other", "other"), ("ctrl_env_comment", "comment")):
        d[key] = sites.count(site)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=dir, in the order the tables should list them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--macro-prefix", default="SF")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json of the model; adds reply_tokens.tex (first-reply lengths in tokens)")
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
    lines = ["\\begin{tabular}{@{}lrrrrrrrrrrr@{}}", "\\toprule",
             "Run & $n$ & Reply chars & Cut off & Success & Ruined & Timeout & Error & Probe chars & Probe crashed & Probe cut off & Est.\\ within 0.01 \\\\", "\\midrule"]
    for label, r in runs:
        d = diagnostics(r)
        o = d["outcomes"]
        pr = d.get("probe_chars_range")
        lines.append(f"{tex(label)} & {d['n']} & {d['reply_chars_median']:.0f} & {pct(d['cut_off'])} & {o['success']} & {o['ruined']} & {o['timeout']} & {o['error']} & "
                     f"{(f'{pr[0]}--{pr[1]}' if pr else '--') if r['stage'] == 2 else '--'} & "
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

    # ---- why the samples that did not run failed; code patterns over all samples ----------
    et = [error_table(r) for _, r in runs]
    lines = ["\\begin{tabular}{@{}l" + "r" * len(runs) + "@{}}", "\\toprule",
             " & " + " & ".join(tex(label) for label, _ in runs) + " \\\\", "\\midrule",
             "samples that did not run & " + " & ".join(str(e["errors"]) for e in et) + " \\\\"]
    for k in ERROR_KINDS:
        lines.append(f"\\quad {tex(k)} & " + " & ".join(str(e["kinds"][k]) for e in et) + " \\\\")
    lines.append("\\midrule")
    for name, _ in CODE_PATTERNS:
        lines.append(f"{tex(name)}, of {et[0]['n']} & " + " & ".join(str(e["patterns"][name]) for e in et) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("errors.tex", lines)
    # ---- what the first reply of a Stage 2 sample is ---------------------------------------
    two = [(label, r) for label, r in runs if r["stage"] == 2]
    n_fragments = 5
    if two:
        ft = [first_reply_table(r) for _, r in two]
        lines = ["\\begin{tabular}{@{}l" + "r" * len(two) + "@{}}", "\\toprule",
                 " & " + " & ".join(tex(label) for label, _ in two) + " \\\\", "\\midrule",
                 "Stage 2 samples & " + " & ".join(str(f["n"]) for f in ft) + " \\\\", "\\midrule"]
        for row in FIRST_REPLY_ROWS:
            if row is None:
                lines.append("\\midrule")
                continue
            key, name = row
            lines.append(f"{name} & " + " & ".join(str(f[key]) for f in ft) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("first_replies.tex", lines)
        n_fragments = 6
    # ---- first-reply lengths in tokens, against a list of budgets --------------------------
    if args.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)

        def ntok(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False).ids)

        lines = [f"% token counts: {args.tokenizer}, no special tokens; the first assistant reply of every sample.",
                 "% A reply fits a budget when its text is shorter than the budget (the end-of-turn token takes one).",
                 "% For a run whose own budget is at or below a column's, the column is empty: the cut hides it.",
                 "\\begin{tabular}{@{}lr" + "r" * (2 + len(TOKEN_BUDGETS)) + "@{}}", "\\toprule",
                 " & & & & \\multicolumn{%d}{c}{First replies that fit in} \\\\" % len(TOKEN_BUDGETS),
                 "\\cmidrule(l){5-%d}" % (4 + len(TOKEN_BUDGETS)),
                 "Run & Budget & Median & Longest & " + " & ".join(str(b) for b in TOKEN_BUDGETS) + " \\\\", "\\midrule"]
        for label, r in runs:
            budget = r["summary"].get("max_tokens")
            n = [ntok(first_reply(x)) for x in r["rows"] if first_reply(x)]
            cells = [pct(mean(v < b for v in n)) if (budget is None or b < budget) else "--" for b in TOKEN_BUDGETS]
            lines.append(f"{tex(label)} & {budget} & {statistics.median(n):.0f} & {max(n)} & " + " & ".join(cells) + " \\\\")
        # the hand-written solution of Stage 0, as it is sent through the sandbox
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from stage0_solution import GATE_VARIANT, PROBE_VARIANTS, controller_source, fence
        span = 2 + len(TOKEN_BUDGETS)
        lines += ["\\midrule",
                  f"Stage~0 hand-written probe code ({tex(GATE_VARIANT)}) & -- & \\multicolumn{{{span}}}{{l}}{{{ntok(fence(PROBE_VARIANTS[GATE_VARIANT]))} tokens}} \\\\",
                  f"Stage~0 hand-written controller & -- & \\multicolumn{{{span}}}{{l}}{{{ntok(fence(controller_source(sum(RANGE) / 2)))} tokens}} \\\\",
                  "\\bottomrule", "\\end{tabular}"]
        write("reply_tokens.tex", lines)
        n_fragments += 1
    print(f"wrote {n_fragments} fragments to {out} for {len(runs)} runs")


if __name__ == "__main__":
    main()
