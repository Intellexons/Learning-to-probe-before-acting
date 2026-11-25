"""Render frozen-model runs (Stage 1: one turn; Stage 2: probe run + controller)
into LaTeX fragments: one comparison table across runs, a per-third table, a
diagnostics table from the samples, an error-kind table (why the samples that
did not run failed, and three code patterns counted over all samples), a table
of what the first reply of each Stage 2 sample is (the replies counted as
probe code, and where the controllers mention the probe environment), and
provenance macros per run. With --tokenizer, one more: the length of the first
replies in tokens against a list of budgets, next to the token counts of the
hand-written probe code and controller of Stage 0.

Runs come in two protocols and the renderer reads both. In the code-block
protocol (runs up to 2026-11-01) a reply was one code block, read by its
contents; its first-reply table is first_replies.tex. In the agent loop (runs
whose samples carry a `"protocol"` that starts with `agent-loop`) the model
calls the tool `probe_run` and a reply without a tool call is its final
answer; the first-reply table is first_actions.tex, sequences.tex counts the
samples by the sequence of their replies, by_path.tex sets the samples that
answered straight away beside those that probed first, probe_outcomes.tex
counts every probe run by how it ended, after_first_probe.tex sets how the
first probe run ended against what the next reply was, probe_fate.tex follows
the probes that found the limit to the controller and the final run,
answered.tex splits each run into the samples that ended without a final run
and the rest, and the error table gains the ways a task can end without a
final run. The first version of the loop
(`"protocol": "agent-loop"`, superseded the day it ran) had a second tool,
`final_run`; its samples render through the same code, with two more rows.
The fragments of a folder that holds only code-block runs come out as before.

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


def is_agent_loop(r: dict) -> bool:
    return str(r.get("protocol") or "").startswith("agent-loop")


def has_final_tool(r: dict) -> bool:
    """The first version of the agent loop, in which the final run was a tool."""
    return r.get("protocol") == "agent-loop"


def reply_cut_off(text: str, agent_loop: bool) -> bool:
    """The token budget closed the reply mid-way: a code fence never closed,
    or in the agent loop a tool call without its end tag."""
    if agent_loop and text.count("<tool_call>") > text.count("</tool_call>"):
        return True
    return text.count("```") % 2 == 1


def cut_off(r: dict) -> bool:
    """The last assistant message was cut off by the token budget."""
    t = r.get("transcript") or []
    msgs = [m["content"] for m in t if m.get("role") == "assistant"]
    if not msgs:
        return False
    return reply_cut_off(msgs[-1], is_agent_loop(r))


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
        d["probe_cut_off"] = mean(reply_cut_off(first_reply(r), is_agent_loop(r)) for r in probed) if probed else None
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
# The agent loop: a task can end without a final run, and a controller that was
# handed in can fail validation. Listed before the runtime kinds they precede.
AGENT_ERROR_KINDS = (
    "no final run: the tool call was cut off by the token budget",
    "no final run: the tool call is not valid JSON",
    "no final run: no tool call in the reply",          # first version of the loop only
    "no final run: the last reply is a probe run",
    "no final run: other (unknown tool, no code)",
    "the controller does not parse",
    "the controller defines no policy",
    "the final answer was cut off by the token budget",
)


def agent_error_kind(r: dict) -> str | None:
    """As `error_kind`, for a sample of the agent loop."""
    import re
    if r.get("ran"):
        return None
    if not r.get("generated"):
        g = str(r.get("generation_error") or "")
        if g.startswith("controller invalid"):
            if "Syntax error" in g:
                return AGENT_ERROR_KINDS[7] if (cut_off(r) and not has_final_tool(r)) else AGENT_ERROR_KINDS[5]
            if "No 'def policy" in g:
                return AGENT_ERROR_KINDS[6]
            return ERROR_KINDS[1]
        last = (r.get("actions") or [""])[-1]
        if last == "probe_run":
            return AGENT_ERROR_KINDS[3]
        if last == "none":
            return AGENT_ERROR_KINDS[2]
        why = " ".join(r.get("call_errors") or [])
        if "cut off" in why and cut_off(r):
            return AGENT_ERROR_KINDS[0]
        if "not valid JSON" in why:
            return AGENT_ERROR_KINDS[1]
        return AGENT_ERROR_KINDS[4]
    e = str((r.get("errors") or [""])[0])
    e = re.sub(r"^s\d+: ", "", e)
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


def kinds_for(runs: list[dict]) -> tuple[str, ...]:
    """The rows of the error table: the code-block kinds for code-block runs,
    the agent-loop kinds for agent-loop runs, both for a mixed table."""
    agent = [bool(r["rows"]) and is_agent_loop(r["rows"][0]) for r in runs]
    if not any(agent):
        return ERROR_KINDS
    runtime = tuple(k for i, k in enumerate(ERROR_KINDS) if i not in (0, 7))
    final_tool = any(bool(r["rows"]) and has_final_tool(r["rows"][0]) for r in runs)
    # "no tool call" is a failure only where the final run was a tool; elsewhere it is the final answer
    loop_kinds = tuple(k for i, k in enumerate(AGENT_ERROR_KINDS) if final_tool or i != 2)
    if all(agent):
        return loop_kinds + runtime
    return ERROR_KINDS + loop_kinds


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


def error_table(run: dict, kind_names: tuple[str, ...] = ERROR_KINDS) -> dict:
    """Counts per error kind over the samples that did not run, and per code
    pattern over all samples (the controller code, whether it ran or not)."""
    import re
    rows = run["rows"]
    kinds = {k: 0 for k in kind_names}
    for r in rows:
        k = agent_error_kind(r) if is_agent_loop(r) else error_kind(r)
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


# The rows of first_actions.tex. A row marked "final tool" is shown only when
# a run of the first version of the loop is in the table.
FIRST_ACTION_ROWS = (
    ("final", "a final run (the controller straight away)", False),
    ("probe", "a probe run, whose output came back", False),
    ("probe_clean", "\\quad the probe code ran without an error", False),
    ("probe_near", "\\quad the output holds a number within 0.01 of the limit", False),
    ("probe_then_final", "\\quad the task went on to a final run", False),
    ("probe_final", "a probe run and a final run in one reply (the probe never read)", True),
    ("invalid", "a tool call that could not be acted on", False),
    ("invalid_cut", "\\quad cut off by the token budget", False),
    ("invalid_json", "\\quad not valid JSON", False),
    ("invalid_other", "\\quad other (unknown tool, no code)", False),
    ("none", "no tool call", True),
    ("none_fenced_policy", "\\quad a fenced code block that defines \\texttt{policy}", True),
    None,
    ("no_final", "samples that ended without a final run", False),
    ("budget_mentions", "samples in which some reply mentions the reply budget", False),
)
FINAL_ACTIONS = ("final_run", "probe_run+final_run")


def first_action_table(run: dict) -> dict:
    """What the first reply of a sample is under the agent loop (`first_action`:
    the final answer, a probe run, or a call that could not be acted on), and
    how the samples that probed went on."""
    rows = run["rows"]
    two = run["stage"] == 2

    def n(pred) -> int:
        return sum(bool(pred(r)) for r in rows)

    act = lambda r: r.get("first_action")
    probed = [r for r in rows if act(r) == "probe_run"]
    invalid = [r for r in rows if act(r) == "invalid"]
    why = lambda r: " ".join(r.get("call_errors") or [])
    cut = lambda r: "cut off" in why(r) and reply_cut_off(first_reply(r), True)
    d = {"n": len(rows), "final": n(lambda r: act(r) == "final_run"),
         "probe": len(probed) if two else None,
         "probe_clean": sum("[probe run rejected" not in (r.get("tool_output") or "")
                            and "[probe run error" not in (r.get("tool_output") or "") for r in probed) if two else None,
         "probe_near": sum(r.get("estimate_error") is not None and r["estimate_error"] <= 0.01 for r in probed) if two else None,
         "probe_then_final": sum((r.get("actions") or [""])[-1] in FINAL_ACTIONS for r in probed) if two else None,
         "probe_final": n(lambda r: act(r) == "probe_run+final_run") if two else None,
         "invalid": len(invalid),
         "invalid_cut": sum(cut(r) for r in invalid),
         "invalid_json": sum("not valid JSON" in why(r) and not cut(r) for r in invalid),
         "none": n(lambda r: act(r) == "none"),
         "none_fenced_policy": n(lambda r: act(r) == "none" and re.search(r"```[^`]*def policy\b", first_reply(r), re.DOTALL)),
         "no_final": n(lambda r: (r.get("actions") or [""])[-1] not in FINAL_ACTIONS),
         "budget_mentions": n(mentions_budget)}
    d["invalid_other"] = d["invalid"] - d["invalid_cut"] - d["invalid_json"]
    return d


REPLIES_LEFT_LINE = re.compile(r"\n*\[\d+ repl(?:y|ies) left\]\s*$")


def tool_text(content: str) -> str:
    """A tool response without the closing line that states the replies left
    (runs made with --show-replies-left carry it)."""
    return REPLIES_LEFT_LINE.sub("", content)


def probe_outputs(r: dict) -> list[str]:
    """The tool responses of a sample's probe runs, in order, read from the
    transcript (a response to a call that could not be acted on is not one)."""
    return [tool_text(m["content"]) for m in (r.get("transcript") or [])
            if m.get("role") == "tool" and not m["content"].startswith("[tool call not acted on")]


def estimate_error_of(text: str, sigma_max: float) -> float | None:
    """How far the closest plausible level in a probe's output is from the true
    limit; the reading of `eval_frozen_baseline.probe_estimate_error`, kept in
    step with it, so that every probe run of a sample can be read, not only
    the first."""
    nums = [float(x) for x in re.findall(r"(?<![\d.\-])0\.\d{2,}(?!\d)", text or "")]
    nums = [x for x in nums if 0.03 <= x <= 0.5]
    return min(abs(x - sigma_max) for x in nums) if nums else None


def some_probe_near(r: dict, tol: float = 0.01) -> bool:
    """Some probe run of the sample printed a number within `tol` of the limit."""
    errs = [estimate_error_of(t, r["sigma_max"]) for t in probe_outputs(r)]
    return any(e is not None and e <= tol for e in errs)


def probe_runs(r: dict) -> list[tuple[str, str]]:
    """The probe runs of a sample as (code, tool response) pairs, in order:
    the calls of each reply that could be acted on, matched with the tool
    responses that follow the reply. A probe run in the last reply was not
    executed and has no response, so it is not in the list."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.agent_loop import parse_tool_calls
    t = r.get("transcript") or []
    pairs = []
    for i, m in enumerate(t):
        if m.get("role") != "assistant":
            continue
        codes = [c.code for c in parse_tool_calls(m["content"]) if c.error is None]
        outs = []
        for nxt in t[i + 1:]:
            if nxt.get("role") != "tool":
                break
            if not nxt["content"].startswith("[tool call not acted on"):
                outs.append(tool_text(nxt["content"]))
        pairs.extend(zip(codes, outs))
    return pairs


BUDGET_MENTION = re.compile(
    r"\b(repl(y|ies) (left|remaining)|(last|final|second|third|fourth|one more) reply|remaining repl"
    r"|only (one|two|three|four|\d) repl|I have (one|two|three|four|\d) repl)", re.IGNORECASE)


def mentions_budget(r: dict) -> bool:
    """Some reply of the sample speaks of how many replies it has or has left."""
    return any(BUDGET_MENTION.search(m["content"]) for m in (r.get("transcript") or []) if m.get("role") == "assistant")


PROBE_OUTCOMES = (
    "ran and printed a number within 0.01 of the limit",
    "ran and printed, no number near the limit",
    "ran and printed nothing",
    "stopped on an end flag read from the wrong place (KeyError)",
    "stopped on another error",
    "ran out of its 30 seconds",
    "rejected before running (syntax error, forbidden name)",
)


def probe_outcome(text: str, sigma_max: float) -> str:
    """How one probe run ended, from its tool response."""
    if text.startswith("[probe run rejected"):
        return PROBE_OUTCOMES[6]
    if "[probe run error: Timeout" in text:
        return PROBE_OUTCOMES[5]
    if "[probe run error" in text:
        flags = "|".join(FLAG_KEYS)
        return PROBE_OUTCOMES[3] if re.search(r"KeyError: '(%s)'" % flags, text) else PROBE_OUTCOMES[4]
    if text.strip() in ("", "[empty output]"):
        return PROBE_OUTCOMES[2]
    e = estimate_error_of(text, sigma_max)
    return PROBE_OUTCOMES[0] if (e is not None and e <= 0.01) else PROBE_OUTCOMES[1]


SILENT_NO_PRINT = "silent_no_print"


def probe_outcome_table(run: dict) -> dict:
    """Every probe run of every sample, counted by how it ended; of the runs
    that printed nothing, how many hold no call to `print` at all."""
    counts = {k: 0 for k in PROBE_OUTCOMES}
    counts[SILENT_NO_PRINT] = 0
    for r in run["rows"]:
        for code, text in probe_runs(r):
            k = probe_outcome(text, r["sigma_max"])
            counts[k] += 1
            if k == PROBE_OUTCOMES[2] and not re.search(r"\bprint\s*\(", code or ""):
                counts[SILENT_NO_PRINT] += 1
    return counts


def after_first_probe_table(run: dict) -> dict:
    """How the first probe run of a sample ended, against what the reply after
    it was: per outcome, the samples and how many of them then gave the final
    answer (the others called the tool again)."""
    out = {k: [0, 0] for k in PROBE_OUTCOMES}
    for r in run["rows"]:
        if r.get("first_action") != "probe_run":
            continue
        pairs = probe_runs(r)
        if not pairs:
            continue
        k = probe_outcome(pairs[0][1], r["sigma_max"])
        out[k][0] += 1
        if (r.get("actions") or [])[1:2] == ["final_run"]:
            out[k][1] += 1
    return out


def controller_numbers(code: str) -> list[float]:
    """The numeric literals of a controller's code."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    return [float(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]


def controller_holds_limit(r: dict, tol: float = 0.01) -> bool:
    """The controller's code holds a number within `tol` of the true limit."""
    return any(0.03 <= x <= 0.5 and abs(x - r["sigma_max"]) <= tol for x in controller_numbers(r.get("code") or ""))


def held_number(r: dict) -> float | None:
    """The number in the controller's code that lies closest to the true limit."""
    xs = [x for x in controller_numbers(r.get("code") or "") if 0.03 <= x <= 0.5]
    return min(xs, key=lambda x: abs(x - r["sigma_max"])) if xs else None


def showered(r: dict) -> bool:
    """The final run's reading went over the limit."""
    return r.get("peak_sigma") is not None and r["peak_sigma"] > r["sigma_max"]


def _final_run_counts(rs: list[dict]) -> dict:
    return {"n": len(rs), "success": sum(bool(r["success"]) for r in rs), "fast": sum(bool(r["fast"]) for r in rs),
            "ruined": sum(outcome(r) == "ruined" for r in rs),
            "showered": sum(outcome(r) == "ruined" and showered(r) for r in rs),
            # of the showers: the number the controller holds is above the true limit
            "showered_above": sum(outcome(r) == "ruined" and showered(r) and (held_number(r) or 0.0) > r["sigma_max"] for r in rs),
            "other": sum(outcome(r) in ("timeout", "error") for r in rs)}


def probe_fate_table(run: dict) -> dict:
    """What became of the probes that found the limit: from every probe run to
    the samples in which one printed the limit, to the controllers that carry
    the number, to how their final runs ended; and the same for the samples
    that probed without finding it."""
    rows = run["rows"]
    runs_ = [(r, code, text) for r in rows for code, text in probe_runs(r)]
    errs = [estimate_error_of(t, r["sigma_max"]) for r, _, t in runs_]
    clean = [probe_outcome(t, r["sigma_max"]) == PROBE_OUTCOMES[0] for r, _, t in runs_]
    near = [e is not None and e <= 0.01 for e in errs]
    d = {"runs": len(runs_), "runs_near": sum(near),
         # the same count as probe_outcomes.tex, and the rest: the number was printed, then the code stopped on an error
         "runs_near_clean": sum(clean), "runs_near_then_error": sum(n and not c for n, c in zip(near, clean))}
    probed = [r for r in rows if probe_outputs(r)]
    found = [r for r in probed if some_probe_near(r)]
    final = lambda r: (r.get("actions") or [""])[-1] in FINAL_ACTIONS
    found_final = [r for r in found if final(r)]
    d.update({"samples_found": len(found), "found_no_final": len(found) - len(found_final),
              "found_final": len(found_final),
              "carries": _final_run_counts([r for r in found_final if controller_holds_limit(r)]),
              "not_carries": _final_run_counts([r for r in found_final if not controller_holds_limit(r)])})
    missed_final = [r for r in probed if not some_probe_near(r) and final(r)]
    d["missed"] = _final_run_counts(missed_final)
    d["missed_carries"] = sum(controller_holds_limit(r) for r in missed_final)
    return d


def answered_row(run: dict) -> dict:
    """A run split into the samples that ended without a final run, which score
    the error floor, and the samples that gave one."""
    rows = run["rows"]
    yes = [r for r in rows if (r.get("actions") or [""])[-1] in FINAL_ACTIONS]
    return {"n": len(rows), "no_final": len(rows) - len(yes), "reward_all": mean(r["reward"] for r in rows),
            "reward_answered": mean(r["reward"] for r in yes), "success_answered": mean(bool(r["success"]) for r in yes),
            "fast_answered": mean(bool(r["fast"]) for r in yes)}


SEQUENCE_WORDS = {"final_run": "final", "probe_run": "probe", "probe_run+final_run": "probe+final",
                  "invalid": "invalid call", "none": "no call"}
MAX_SEQUENCE_ROWS = 12


def sequence_of(r: dict) -> str:
    """The replies of a sample as a short sequence, e.g. "probe, probe, final"."""
    return ", ".join(SEQUENCE_WORDS.get(a, a) for a in (r.get("actions") or []))


def sequence_table(runs: list[dict]) -> tuple[list[str], list[dict]]:
    """Counts of samples per sequence of replies, one dict per run, and the
    sequences ordered by how many samples took them over all runs. Past
    MAX_SEQUENCE_ROWS the rest are summed as "other"."""
    counts = []
    total: dict[str, int] = {}
    for run in runs:
        c: dict[str, int] = {}
        for r in run["rows"]:
            k = sequence_of(r)
            c[k] = c.get(k, 0) + 1
            total[k] = total.get(k, 0) + 1
        counts.append(c)
    order = sorted(total, key=lambda k: (-total[k], k))
    shown, rest = order[:MAX_SEQUENCE_ROWS], order[MAX_SEQUENCE_ROWS:]
    if rest:
        for c in counts:
            c["other"] = sum(c.get(k, 0) for k in rest)
        shown = shown + ["other"]
    return shown, counts


PATHS = (
    ("answered straight away", lambda r: (r.get("actions") or [""])[0] == "final_run"),
    ("probed, then a final run", lambda r: (r.get("actions") or [""])[0] != "final_run"
                                          and (r.get("actions") or [""])[-1] in FINAL_ACTIONS),
    ("ended without a final run", lambda r: (r.get("actions") or [""])[-1] not in FINAL_ACTIONS),
)


def path_rows(run: dict) -> list[dict]:
    """The samples of a run split by the path they took, with the outcome of each part."""
    out = []
    for name, pred in PATHS:
        rs = [r for r in run["rows"] if pred(r)]
        out.append({"path": name, "n": len(rs),
                    "ran": mean(bool(r["ran"]) for r in rs), "success": mean(bool(r["success"]) for r in rs),
                    "fast": mean(bool(r["fast"]) for r in rs), "reward": mean(r["reward"] for r in rs),
                    "ride_ratio": mean(r.get("ride_ratio") for r in rs),
                    "near": sum(some_probe_near(r) for r in rs) if not name.startswith("answered") else None})
    return out


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
    kind_names = kinds_for([r for _, r in runs])
    et = [error_table(r, kind_names) for _, r in runs]
    lines = ["\\begin{tabular}{@{}l" + "r" * len(runs) + "@{}}", "\\toprule",
             " & " + " & ".join(tex(label) for label, _ in runs) + " \\\\", "\\midrule",
             "samples that did not run & " + " & ".join(str(e["errors"]) for e in et) + " \\\\"]
    for k in kind_names:
        lines.append(f"\\quad {tex(k)} & " + " & ".join(str(e["kinds"][k]) for e in et) + " \\\\")
    lines.append("\\midrule")
    for name, _ in CODE_PATTERNS:
        lines.append(f"{tex(name)}, of {et[0]['n']} & " + " & ".join(str(e["patterns"][name]) for e in et) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("errors.tex", lines)
    # ---- what the first reply of a Stage 2 sample is ---------------------------------------
    def agent(r: dict) -> bool:
        return bool(r["rows"]) and is_agent_loop(r["rows"][0])

    n_fragments = 5
    loop_runs = [(label, r) for label, r in runs if agent(r)]
    if loop_runs:
        ft = [first_action_table(r) for _, r in loop_runs]
        lines = ["\\begin{tabular}{@{}l" + "r" * len(loop_runs) + "@{}}", "\\toprule",
                 " & " + " & ".join(tex(label) for label, _ in loop_runs) + " \\\\", "\\midrule",
                 "samples & " + " & ".join(str(f["n"]) for f in ft) + " \\\\", "\\midrule",
                 "\\multicolumn{%d}{@{}l}{the first reply is} \\\\" % (1 + len(loop_runs))]
        final_tool = any(has_final_tool(r["rows"][0]) for _, r in loop_runs)
        for row in FIRST_ACTION_ROWS:
            if row is None:
                lines.append("\\midrule")
                continue
            key, name, only_final_tool = row
            if only_final_tool and not final_tool:
                continue
            indent = "" if key in ("no_final", "budget_mentions") else "\\quad "
            lines.append(f"{indent}{name} & " + " & ".join("--" if f[key] is None else str(f[key]) for f in ft) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("first_actions.tex", lines)
        n_fragments += 1

        # the sequence of replies each sample took
        shown, counts = sequence_table([r for _, r in loop_runs])
        lines = ["\\begin{tabular}{@{}l" + "r" * len(loop_runs) + "@{}}", "\\toprule",
                 "Replies of the sample, in order & " + " & ".join(tex(label) for label, _ in loop_runs) + " \\\\", "\\midrule"]
        for k in shown:
            lines.append(f"{tex(k)} & " + " & ".join(str(c.get(k, 0)) for c in counts) + " \\\\")
        lines += ["\\midrule", "samples & " + " & ".join(str(len(r["rows"])) for _, r in loop_runs) + " \\\\",
                  "\\bottomrule", "\\end{tabular}"]
        write("sequences.tex", lines)
        n_fragments += 1

        # answered straight away against probed first
        lines = ["\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
                 "Run & Path & $n$ & Controller ran & Success & Fast & Reward & Ride ratio & Est.\\ within 0.01 \\\\", "\\midrule"]
        for i, (label, r) in enumerate(loop_runs):
            if i:
                lines.append("\\addlinespace")
            for j, row in enumerate(path_rows(r)):
                lines.append(f"{tex(label) if j == 0 else ''} & {row['path']} & {row['n']} & {pct(row['ran'])} & {pct(row['success'])} & "
                             f"{pct(row['fast'])} & {num(row['reward'])} & {num(row['ride_ratio'], 2)} & "
                             f"{'--' if row['near'] is None else row['near']} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("by_path.tex", lines)
        n_fragments += 1

        # every probe run, by how it ended
        probing = [(label, r) for label, r in loop_runs if r["stage"] == 2]
        if probing:
            pt = [probe_outcome_table(r) for _, r in probing]
            lines = ["\\begin{tabular}{@{}l" + "r" * len(probing) + "@{}}", "\\toprule",
                     " & " + " & ".join(tex(label) for label, _ in probing) + " \\\\", "\\midrule",
                     "probe runs, over all replies & " + " & ".join(str(sum(c[k] for k in PROBE_OUTCOMES)) for c in pt) + " \\\\"]
            for k in PROBE_OUTCOMES:
                lines.append(f"\\quad {tex(k)} & " + " & ".join(str(c[k]) for c in pt) + " \\\\")
                if k == PROBE_OUTCOMES[2]:
                    lines.append("\\qquad of these, the code holds no \\texttt{print} & "
                                 + " & ".join(str(c[SILENT_NO_PRINT]) for c in pt) + " \\\\")
            lines += ["\\bottomrule", "\\end{tabular}"]
            write("probe_outcomes.tex", lines)
            n_fragments += 1

            # how the first probe ended, against what the next reply was
            at = [after_first_probe_table(r) for _, r in probing]
            lines = ["\\begin{tabular}{@{}l" + "rr" * len(probing) + "@{}}", "\\toprule",
                     " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{tex(label)}}}" for label, _ in probing) + " \\\\",
                     "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(probing))),
                     "The first probe run & " + " & ".join("$n$ & then final" for _ in probing) + " \\\\", "\\midrule"]
            for k in PROBE_OUTCOMES:
                lines.append(f"{tex(k)} & " + " & ".join(f"{a[k][0]} & {a[k][1]}" for a in at) + " \\\\")
            lines += ["\\midrule", "all first probe runs & " + " & ".join(
                f"{sum(v[0] for v in a.values())} & {sum(v[1] for v in a.values())}" for a in at) + " \\\\",
                "\\bottomrule", "\\end{tabular}"]
            write("after_first_probe.tex", lines)
            n_fragments += 1

            # what became of the probes that found the limit
            ft = [probe_fate_table(r) for _, r in probing]

            def row(name: str, get) -> str:
                return f"{name} & " + " & ".join(str(get(f)) for f in ft) + " \\\\"

            def group(prefix: str, key: str, above: bool = False) -> list[str]:
                out = [row(f"{prefix}the final run succeeded", lambda f: f[key]["success"]),
                       row(f"{prefix}\\quad and was fast", lambda f: f[key]["fast"]),
                       row(f"{prefix}the batch was ruined", lambda f: f[key]["ruined"]),
                       row(f"{prefix}\\quad by a shower: the reading went over the limit", lambda f: f[key]["showered"])]
                if above:
                    out.append(row(f"{prefix}\\qquad the number in the controller is above the limit",
                                   lambda f: f[key]["showered_above"]))
                return out + [row(f"{prefix}timed out or did not run", lambda f: f[key]["other"])]

            lines = ["\\begin{tabular}{@{}l" + "r" * len(probing) + "@{}}", "\\toprule",
                     " & " + " & ".join(tex(label) for label, _ in probing) + " \\\\", "\\midrule",
                     row("probe runs executed", lambda f: f["runs"]),
                     row("\\quad whose output holds a number within 0.01 of the limit", lambda f: f["runs_near"]),
                     row("\\qquad the code ran to its end", lambda f: f["runs_near_clean"]),
                     row("\\qquad the code stopped on an error after printing it", lambda f: f["runs_near_then_error"]),
                     "\\midrule",
                     row("samples in which a probe run printed such a number", lambda f: f["samples_found"]),
                     row("\\quad ran out of replies before a final answer", lambda f: f["found_no_final"]),
                     row("\\quad went on to a final run", lambda f: f["found_final"]),
                     row("\\qquad the controller holds a number within 0.01 of the limit", lambda f: f["carries"]["n"]),
                     *group("\\qquad\\quad ", "carries", above=True),
                     row("\\qquad the controller holds no such number", lambda f: f["not_carries"]["n"]),
                     *group("\\qquad\\quad ", "not_carries"),
                     "\\midrule",
                     row("samples that probed, found no such number, and gave a final run", lambda f: f["missed"]["n"]),
                     row("\\quad the controller holds a number within 0.01 of the limit all the same", lambda f: f["missed_carries"]),
                     *group("\\quad ", "missed"),
                     "\\bottomrule", "\\end{tabular}"]
            write("probe_fate.tex", lines)
            n_fragments += 1

        # the samples that ended without a final run, and the rest
        ar = [answered_row(r) for _, r in loop_runs]
        lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
                 " & & & & \\multicolumn{3}{c}{The samples that gave a final run} \\\\", "\\cmidrule(l){5-7}",
                 "Run & Samples & No final run & Reward, all & Reward & Success & Fast \\\\", "\\midrule"]
        for (label, _), a in zip(loop_runs, ar):
            lines.append(f"{tex(label)} & {a['n']} & {a['no_final']} & {num(a['reward_all'])} & {num(a['reward_answered'])} & "
                         f"{pct(a['success_answered'])} & {pct(a['fast_answered'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("answered.tex", lines)
        n_fragments += 1
    two = [(label, r) for label, r in runs if r["stage"] == 2 and not agent(r)]
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
        n_fragments += 1
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
        if all(agent(r) for _, r in runs):
            # as the replies the agent loop reads: a probe_run call, then the final answer
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from src.agent_loop import TOOL_PROBE, render_final_answer, render_tool_call
            probe_reply = render_tool_call(TOOL_PROBE, PROBE_VARIANTS[GATE_VARIANT].strip("\n"))
            final_reply = render_final_answer(controller_source(sum(RANGE) / 2))
            form = " as a tool call"
        else:
            probe_reply = fence(PROBE_VARIANTS[GATE_VARIANT])
            final_reply = fence(controller_source(sum(RANGE) / 2))
            form = ""
        span = 2 + len(TOKEN_BUDGETS)
        lines += ["\\midrule",
                  f"Stage~0 hand-written probe code ({tex(GATE_VARIANT)}){form} & -- & \\multicolumn{{{span}}}{{l}}{{{ntok(probe_reply)} tokens}} \\\\",
                  f"Stage~0 hand-written controller & -- & \\multicolumn{{{span}}}{{l}}{{{ntok(final_reply)} tokens}} \\\\",
                  "\\bottomrule", "\\end{tabular}"]
        write("reply_tokens.tex", lines)
        n_fragments += 1
    print(f"wrote {n_fragments} fragments to {out} for {len(runs)} runs")


if __name__ == "__main__":
    main()
