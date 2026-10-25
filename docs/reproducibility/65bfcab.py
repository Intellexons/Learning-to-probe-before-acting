"""Stage 2 example ablation (2026-10-26): every number in the diary entry, from the compact
sample files in docs/stage2.

    uv run python docs/reproducibility/65bfcab.py
    uv run python docs/reproducibility/65bfcab.py full=<dir> format=<dir> none=<dir>   # raw result dirs

The three runs differ only in the prompt's multi-turn format example (config key
prompt.example): full = the original example (reset the env, print the state), format = the
same two blocks with a comment in place of the probe code, none = no example section.
Rows are counts out of 96 samples (12 held-out batches x 8 samples) unless marked.
"""
import json
import re
import statistics
import sys
from pathlib import Path

DEFAULT = {
    "full": "docs/stage2/stage2_samples_example-full_2026-10-25.jsonl",
    "format": "docs/stage2/stage2_samples_example-format_2026-10-25.jsonl",
    "none": "docs/stage2/stage2_samples_example-none_2026-10-25.jsonl",
}


def load(path):
    p = Path(path)
    if p.is_dir():
        p = p / "stage2_samples.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def t1(r):
    return r.get("t1_code") or ""


def own_env(r):
    return bool(re.search(r"CrystallizerEnv\s*\(", t1(r)))


def steps_env(r):
    return "env.step" in t1(r)


def to_shower(r):
    """Steps an environment in a loop and checks the end flag: the shape of a real experiment."""
    c = t1(r)
    return steps_env(r) and bool(re.search(r"\b(for|while)\b", c)) and bool(
        re.search(r"if\s+done|while\s+not\s+done|break|showered|is_unsafe|failure_reason", c))


def says_find(r):
    return bool(re.search(r"(find|estimat|determin|infer|identify|measure|detect)\w*\s+(the\s+)?"
                          r"(hidden|metastable|sigma_max|limit|shower)", t1(r), re.I))


def pastes_probe_numbers(r):
    nums = set(re.findall(r"\d+\.\d{3,}", r.get("tool_output") or ""))
    return any(n[:6] in (r.get("code") or "") for n in nums if len(n) >= 6)


ROWS = [
    ("took the probe run (turn 1 is main)", lambda rs: sum(r.get("t1_mode") == "main" for r in rs)),
    ("turn 1 rejected (no recognised block)", lambda rs: sum(r.get("t1_mode") == "invalid" for r in rs)),
    ("valid final policy", lambda rs: sum(bool(r.get("generated")) for r in rs)),
    ("probe copies the example: reset+print, no step", lambda rs: sum(r.get("t1_mode") == "main" and not steps_env(r) for r in rs)),
    ("probe announces it will find the limit", lambda rs: sum(says_find(r) for r in rs)),
    ("probe steps an env in a loop and checks the end flag", lambda rs: sum(to_shower(r) for r in rs)),
    ("    on the given env", lambda rs: sum(to_shower(r) and not own_env(r) for r in rs)),
    ("    on a fresh env of its own", lambda rs: sum(to_shower(r) and own_env(r) for r in rs)),
    ("probe constructs a fresh CrystallizerEnv", lambda rs: sum(own_env(r) for r in rs)),
    ("    with a limit the model chose (HiddenParams(sigma_max=0.x))", lambda rs: sum(bool(re.search(r"HiddenParams\(\s*sigma_max\s*=\s*0\.\d", t1(r))) for r in rs)),
    ("    and prints the limit it set", lambda rs: sum(own_env(r) and bool(re.search(r"hidden\.sigma_max|print\(.*sigma_max", t1(r))) for r in rs)),
    ("probe heats (+1) while talking about cooling", lambda rs: sum(bool(re.search(r"env\.step\(\s*\+?1(\.0)?\s*\)|action\s*=\s*\+?1\.0\b", t1(r))) and bool(re.search(r"cool", t1(r), re.I)) for r in rs)),
    ("probe failed at execution", lambda rs: sum((r.get("tool_output") or "").startswith("[main error") for r in rs)),
    ("    of which `import env`", lambda rs: sum("No module named 'env'" in (r.get("tool_output") or "") for r in rs)),
    ("tool output shows a shower", lambda rs: sum(bool(re.search(r"showered\W*(True|=\s*1)|done\W*True|shower", r.get("tool_output") or "", re.I)) for r in rs)),
    ("final policy keeps the constant 0.25", lambda rs: sum("0.25" in (r.get("code") or "") for r in rs)),
    ("final policy with another sigma setpoint", lambda rs: sum(bool(re.search(r"(sigma_max|SIGMA_MAX|target_sigma|sigma_target|sigma_safe|SIGMA_SAFE|sigma_limit)\s*=\s*0\.(?!25)\d+", r.get("code") or "")) for r in rs)),
    ("final policy pastes probe-batch numbers", lambda rs: sum(pastes_probe_numbers(r) for r in rs)),
    ("success", lambda rs: sum(bool(r.get("success")) for r in rs)),
    ("fast success", lambda rs: sum(bool(r.get("fast")) for r in rs)),
    ("normalised time (mean)", lambda rs: f"{statistics.mean(r.get('norm_time') or 0.0 for r in rs):.3f}"),
    ("reward (mean)", lambda rs: f"{statistics.mean(r['reward'] for r in rs):.1f}"),
    ("probe length, median lines of code", lambda rs: statistics.median(len(t1(r).splitlines()) for r in rs)),
]


def main(argv):
    runs = dict(a.split("=", 1) for a in argv) if argv else DEFAULT
    data = {k: load(v) for k, v in runs.items()}
    print(f"{'out of 96 samples':64s}" + "".join(f"{k:>9s}" for k in data))
    for name, fn in ROWS:
        print(f"{name:64s}" + "".join(f"{fn(rs)!s:>9s}" for rs in data.values()))


if __name__ == "__main__":
    main(sys.argv[1:])
