"""Stage 0: the environment gate. Hand-written controllers only; no model is run.

Question: does knowing the hidden limit sigma_max buy anything that a controller
without it cannot get from the readings of its own batch? Three gates, fixed
before the run:
  G1  the informed side reaches >= 90% fast success at every anchor
  G2  the best blind controller is >= 40 points of fast success below the informed side
  G3  the hand-written probe run + controller, from the brief alone, through the real
      execution path, reaches >= 95% fast success at every anchor with no probe failure

Protocol (docs/stage0_gate.tex is the full account):
  families   five hand-written controller families (track, linear, power, hybrid, pwl),
             1 to 10 knobs in the unit cube, mapped to physical ranges in make_controller
  sides      blind: one controller per family, tuned across limits drawn by seed;
             informed: one controller per family per anchor, tuned at that one limit
  search     300 random knob vectors; the best 6 hill-climb for 120 rounds; objective
             the mean env reward over the 48 search batches
  scoring    200 fresh batches never searched on: success, fast success (within
             FAST_FACTOR of the oracle's steps), normalised time (oracle steps / own)
  limits     a batch's limit is a deterministic function of its seed, uniform within one
             third of the range (build_crystallizer_tasks.draw_sigma_max), or an anchor
  oracle     ride_action at the limit (crystallizer_utils); the task files use the same
  batches    construct the env, reset() once, as the runner does (crystallizer_utils)
  G3         scripts/stage0_solution.py through execute_turn and score_sample, the sandbox
             path training and eval use, on the fresh batches over the range and at anchors
  references the oracle, the cautious blind controller, the searched blind controller and
             the probe + controller, on the held-out evaluation tasks (crystallizer_eval.jsonl)

Every output carries src.provenance (commit, command, host, Slurm job).
  uv run python scripts/stage0_gate.py --out results/stage0 --workers 32
  uv run python scripts/stage0_gate.py --quick --out /tmp/stage0_quick     # pipeline smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "seed_tasks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.multi_turn import execute_turn  # noqa: E402
from src.provenance import provenance  # noqa: E402
from src.runner import DEFAULT_MAX_STEPS, FAST_FACTOR, extract_code, score_sample  # noqa: E402
from build_crystallizer_tasks import N_STRATA, draw_sigma_max, stratum_bounds, task_entry  # noqa: E402
from crystallizer_env import EnvConfig, HiddenParams  # noqa: E402
from crystallizer_utils import (  # noqa: E402
    SIGMA_MAX_RANGE, ride_action, run_cautious_episode, run_episode, run_oracle_episode,
)
from stage0_solution import (  # noqa: E402
    CONTROLLER_MARGIN, GATE_VARIANT, PROBE_VARIANTS, controller_source, fence, parse_bracket,
)

CFG = EnvConfig()
STEP_DEGC = CFG.max_rate * CFG.dt / 60.0
ANCHORS = (0.08, 0.165, 0.25)
SEARCH_SEEDS = list(range(20_000, 20_048))      # tuned on; never scored on
FRESH_SEEDS = list(range(30_000, 30_200))       # scored on; never tuned on
EVAL_FILE = REPO / "scripts" / "seed_tasks" / "crystallizer_eval.jsonl"
BUDGET = {"n_random": 300, "topk": 6, "n_climb": 120}
QUICK_BUDGET = {"n_random": 6, "topk": 2, "n_climb": 2}
GATES = {"g1_informed_fast_min": 0.90, "g2_gap_min": 0.40, "g3_fast_min": 0.95}
FAMILIES = {"track": 1, "linear": 1, "power": 2, "hybrid": 3, "pwl": 10}
THIRDS = ("low", "middle", "high")


# ----------------------------------------------------------------------------------------
# batches and their limits
# ----------------------------------------------------------------------------------------

def stratified_pairs(seeds: list[int]) -> list[tuple[int, float]]:
    """(seed, limit) with the limit drawn in the third given by the seed's index."""
    return [(s, draw_sigma_max(s, i % N_STRATA)) for i, s in enumerate(seeds)]


def anchor_pairs(seeds: list[int], anchor: float) -> list[tuple[int, float]]:
    return [(s, anchor) for s in seeds]


def third_of(sigma_max: float) -> str:
    lo, hi = SIGMA_MAX_RANGE
    k = min(N_STRATA - 1, int((sigma_max - lo) / (hi - lo) * N_STRATA))
    return THIRDS[max(0, k)]


# ----------------------------------------------------------------------------------------
# controller families: controller(state, env) -> action, knobs p in the unit cube
# ----------------------------------------------------------------------------------------

def _clip(a: float) -> float:
    return max(-1.0, min(1.0, a))


def physical_params(family: str, p: list[float]) -> dict:
    if family == "track":
        return {"level": 0.02 + 0.33 * p[0]}
    if family == "linear":
        return {"rate_degC_per_min": 0.02 + 1.48 * p[0]}
    if family == "power":
        return {"tau_min": 10.0 + 190.0 * p[0], "exponent": 0.5 + 3.5 * p[1]}
    if family == "hybrid":
        return {"tau_min": 10.0 + 190.0 * p[0], "exponent": 0.5 + 3.5 * p[1], "cap": 0.02 + 0.33 * p[2]}
    if family == "pwl":
        return {"knot_times": sorted(p[:4]), "levels": list(p[4:10])}
    raise ValueError(family)


def make_controller(family: str, p: list[float]):
    """A controller of `family` with knobs `p`. Every family computes a desired temperature
    for the end of the next step and returns the jacket fraction that gets there, clipped,
    except `track`, which asks the rider for the most cooling that keeps the reported sigma
    at its level, and `hybrid`, which caps its profile with the rider."""
    q = physical_params(family, p)
    mem: dict = {}

    def t_min(state):
        return state["t"] * CFG.dt / 60.0

    def to_action(T_des, state):
        return _clip((max(T_des, state["T_target"]) - state["T"]) / STEP_DEGC)

    def profile_T(state, tau, n):
        T0 = mem.setdefault("T0", state["T"])
        return T0 - (T0 - state["T_target"]) * min(1.0, t_min(state) / tau) ** n

    if family == "track":
        level = q["level"]

        def ctl(state, env):
            return ride_action(env, level)
    elif family == "linear":
        rate = q["rate_degC_per_min"]

        def ctl(state, env):
            T0 = mem.setdefault("T0", state["T"])
            return to_action(T0 - rate * t_min(state), state)
    elif family == "power":
        tau, n = q["tau_min"], q["exponent"]

        def ctl(state, env):
            return to_action(profile_T(state, tau, n), state)
    elif family == "hybrid":
        tau, n, cap = q["tau_min"], q["exponent"], q["cap"]

        def ctl(state, env):
            return max(to_action(profile_T(state, tau, n), state), ride_action(env, cap))
    elif family == "pwl":
        knots = [0.0] + q["knot_times"] + [1.0]
        vals = q["levels"]

        def ctl(state, env):
            T0 = mem.setdefault("T0", state["T"])
            x = state["t"] / CFG.max_steps
            for i in range(5):
                if x <= knots[i + 1]:
                    w = (x - knots[i]) / max(knots[i + 1] - knots[i], 1e-9)
                    f = vals[i] + (vals[i + 1] - vals[i]) * w
                    break
            else:
                f = vals[5]
            return to_action(state["T_target"] + f * (T0 - state["T_target"]), state)
    else:
        raise ValueError(family)
    return ctl


# ----------------------------------------------------------------------------------------
# episodes and their scores
# ----------------------------------------------------------------------------------------

def episode_row(res: dict, seed: int, sigma_max: float, oracle: int) -> dict:
    fast = bool(res["success"] and res["steps"] <= FAST_FACTOR * oracle)
    return {"seed": seed, "sigma_max": sigma_max, "third": third_of(sigma_max), "oracle_steps": oracle,
            "steps": res["steps"], "success": bool(res["success"]), "fast": fast,
            "norm_time": (oracle / res["steps"]) if res["success"] and res["steps"] else 0.0,
            "reward": res["reward"], "failure_reason": res["failure_reason"]}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    reasons = {}
    for r in rows:
        key = r["failure_reason"] or "success"
        reasons[key] = reasons.get(key, 0) + 1
    return {"n": n,
            "reward": statistics.fmean(r["reward"] for r in rows),
            "success": sum(r["success"] for r in rows) / n,
            "fast": sum(r["fast"] for r in rows) / n,
            "norm_time": statistics.fmean(r["norm_time"] for r in rows),
            "min_norm_time": min(r["norm_time"] for r in rows),
            "steps": statistics.fmean(r["steps"] for r in rows),
            "outcomes": reasons}


def by_third(rows: list[dict]) -> dict:
    return {t: summarize([r for r in rows if r["third"] == t]) for t in THIRDS}


def _oracle_job(args):
    seed, smax = args
    res = run_oracle_episode(seed, HiddenParams(sigma_max=smax))
    if not res["success"]:
        raise RuntimeError(f"oracle failed on seed {seed} sigma_max {smax}: {res}")
    return (seed, smax), res


def _episode_job(args):
    family, p, seed, smax, oracle = args
    res = run_episode(make_controller(family, p), seed, HiddenParams(sigma_max=smax), policy_sees_env=True)
    return episode_row(res, seed, smax, oracle)


def evaluate_many(pool, family: str, cands: list[list[float]], triples: list[tuple], workers: int) -> list[list[dict]]:
    """Rows of every candidate on every (seed, limit, oracle_steps) triple, in order."""
    jobs = [(family, p, s, m, o) for p in cands for (s, m, o) in triples]
    chunk = max(1, len(jobs) // (8 * workers))
    rows = pool.map(_episode_job, jobs, chunksize=chunk)
    n = len(triples)
    return [rows[i * n:(i + 1) * n] for i in range(len(cands))]


def mean_reward(rows: list[dict]) -> float:
    return statistics.fmean(r["reward"] for r in rows)


# ----------------------------------------------------------------------------------------
# search: random sample, then hill-climb the best few
# ----------------------------------------------------------------------------------------

def search_family(pool, family: str, triples: list[tuple], rng: random.Random, budget: dict, workers: int):
    d = FAMILIES[family]
    cands = [[rng.random() for _ in range(d)] for _ in range(budget["n_random"])]
    vals = [mean_reward(rows) for rows in evaluate_many(pool, family, cands, triples, workers)]
    order = sorted(range(len(cands)), key=lambda i: -vals[i])[:budget["topk"]]
    pop = [cands[i][:] for i in order]
    val = [vals[i] for i in order]
    n_climb = budget["n_climb"]
    for it in range(n_climb):
        scale = 0.15 * (1 - it / n_climb) + 0.01
        props = [[min(1.0, max(0.0, x + rng.gauss(0, scale))) for x in pp] for pp in pop]
        pv = [mean_reward(rows) for rows in evaluate_many(pool, family, props, triples, workers)]
        for k in range(len(pop)):
            if pv[k] > val[k]:
                pop[k], val[k] = props[k], pv[k]
    best = max(range(len(pop)), key=lambda k: val[k])
    return pop[best], val[best], {"n_evaluations": budget["n_random"] + budget["topk"] * n_climb}


# ----------------------------------------------------------------------------------------
# G3: the hand-written probe and controller through the real execution path
# ----------------------------------------------------------------------------------------

def run_two_turn(task: dict, variant: str) -> dict:
    """Probe run through execute_turn (sandbox, probe batch, limit masked), then the
    controller through score_sample, exactly as training and eval execute code."""
    sc = task["scenarios"][0]
    smax = sc["hidden"]["sigma_max"]
    oracle = sc["oracle_steps"]
    row = {"task_id": task["task_id"], "seed": sc["seed"], "probe_seed": sc["probe_seed"],
           "sigma_max": smax, "third": third_of(smax), "oracle_steps": oracle, "variant": variant}
    tr = execute_turn(fence(PROBE_VARIANTS[variant]), task, is_last_turn=False, max_steps=DEFAULT_MAX_STEPS)
    bracket = parse_bracket(tr.text)
    if bracket is None:
        row.update({"probe_failed": True, "probe_text": tr.text[:400], "lo": None, "hi": None,
                    "probe_error": None, "bracket_width": None, "steps": 0, "success": False,
                    "fast": False, "norm_time": 0.0, "reward": None, "failure_reason": "probe_failed"})
        return row
    lo, hi = bracket
    code = extract_code(fence(controller_source(lo)))
    res = score_sample(code, task, max_steps=DEFAULT_MAX_STEPS)
    sid = sc["scenario_id"]
    row.update({"probe_failed": False, "lo": lo, "hi": hi, "probe_error": smax - lo,
                "bracket_width": hi - lo, "steps": int(res["steps_per_scenario"][sid]),
                "success": bool(res["success_per_scenario"][sid]), "fast": bool(res["fast_per_scenario"][sid]),
                "norm_time": float(res["norm_time_per_scenario"][sid]),
                "reward": float(res["reward_per_scenario"][sid]),
                "failure_reason": None if res["success_per_scenario"][sid] else "see errors",
                "errors": res["errors"]})
    return row


def summarize_two_turn(rows: list[dict]) -> dict:
    s = summarize([{**r, "reward": r["reward"] if r["reward"] is not None else -300.0} for r in rows])
    ok = [r for r in rows if not r["probe_failed"]]
    errs = sorted(abs(r["probe_error"]) for r in ok)
    s["probe_failures"] = len(rows) - len(ok)
    s["probe_error"] = {"median": statistics.median(errs), "p90": errs[int(0.9 * (len(errs) - 1))],
                        "max": errs[-1]} if errs else None
    s["bracket_width_max"] = max((r["bracket_width"] for r in ok), default=None)
    return s


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Log:
    def __init__(self, path: Path):
        self.path = path
        self.t0 = time.monotonic()
        self.path.write_text("")

    def __call__(self, msg: str = ""):
        line = f"[{time.monotonic() - self.t0:7.1f}s] {msg}" if msg else ""
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def fmt(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "n=0"
    return (f"reward {s['reward']:7.1f}  success {100 * s['success']:5.1f}%  fast {100 * s['fast']:5.1f}%  "
            f"norm-time {s['norm_time']:.3f}  steps {s['steps']:6.1f}  outcomes {s['outcomes']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--search-seed", type=int, default=0, help="seed of the search's random numbers")
    ap.add_argument("--quick", action="store_true", help="tiny budget and few batches: a pipeline smoke test")
    args = ap.parse_args()
    prov = provenance(sys.argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "stage0_summary.txt")
    budget = QUICK_BUDGET if args.quick else BUDGET
    search_seeds = SEARCH_SEEDS[:6] if args.quick else SEARCH_SEEDS
    fresh_seeds = FRESH_SEEDS[:12] if args.quick else FRESH_SEEDS
    eval_tasks = load_jsonl(EVAL_FILE)
    if args.quick:
        eval_tasks = eval_tasks[:6]
    rng = random.Random(args.search_seed)
    workers = max(1, args.workers)
    t_start = time.time()

    log(f"Stage 0 gate: commit {prov['commit'][:7]} ({prov['commit_source']}"
        f"{', dirty tree' if prov['dirty_tree'] else ''}), host {prov['hostname']}, "
        f"slurm {prov['slurm'].get('job_id', '-')}, workers {workers}, quick={args.quick}")
    log(f"budget {budget}, search batches {len(search_seeds)}, fresh batches {len(fresh_seeds)}, "
        f"held-out tasks {len(eval_tasks)}, anchors {ANCHORS}, gates {GATES}")

    config = {"env": asdict(CFG), "fast_factor": FAST_FACTOR, "anchors": list(ANCHORS),
              "sigma_max_range": list(SIGMA_MAX_RANGE), "n_strata": N_STRATA,
              "search_seeds": [search_seeds[0], search_seeds[-1], len(search_seeds)],
              "fresh_seeds": [fresh_seeds[0], fresh_seeds[-1], len(fresh_seeds)],
              "budget": budget, "search_seed": args.search_seed, "gates": GATES,
              "families": FAMILIES, "controller_margin": CONTROLLER_MARGIN,
              "probe_variants": list(PROBE_VARIANTS), "gate_variant": GATE_VARIANT,
              "inputs": {"crystallizer_eval.jsonl": sha256(EVAL_FILE), "n_eval_tasks": len(eval_tasks)}}

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        # Oracle steps for every batch that will be scored, once.
        pairs_search_range = stratified_pairs(search_seeds)
        pairs_fresh_range = stratified_pairs(fresh_seeds)
        all_pairs = set(pairs_search_range) | set(pairs_fresh_range)
        for a in ANCHORS:
            all_pairs |= set(anchor_pairs(search_seeds, a)) | set(anchor_pairs(fresh_seeds, a))
        oracle = {}
        for key, res in pool.map(_oracle_job, sorted(all_pairs), chunksize=8):
            oracle[key] = res["steps"]
        log(f"oracle: {len(oracle)} (batch, limit) pairs")

        def triples(pairs):
            return [(s, m, oracle[(s, m)]) for s, m in pairs]

        # ---- search: blind side, one controller per family across the range ----------------
        search = {"blind": {}, "informed": {a: {} for a in ANCHORS}}
        rows_store: dict = {"blind": {}, "informed": {}, "g3": {}, "references": None}
        log("blind side: one controller per family, tuned across the range")
        for fam in FAMILIES:
            t = time.monotonic()
            p, v, info = search_family(pool, fam, triples(pairs_search_range), rng, budget, workers)
            rows = evaluate_many(pool, fam, [p], triples(pairs_fresh_range), workers)[0]
            at_anchors = {}
            for a in ANCHORS:
                at_anchors[a] = summarize(evaluate_many(pool, fam, [p], triples(anchor_pairs(fresh_seeds, a)), workers)[0])
            search["blind"][fam] = {"params": p, "physical": physical_params(fam, p), "search_reward": v,
                                    "eval": summarize(rows), "eval_by_third": by_third(rows),
                                    "eval_at_anchors": at_anchors, "seconds": time.monotonic() - t, **info}
            rows_store["blind"][fam] = rows
            log(f"  blind {fam:7s} {fmt(summarize(rows))}  physical {physical_params(fam, p)}  [{time.monotonic() - t:.0f}s]")

        # ---- search: informed side, one controller per family per anchor ----------------
        log("informed side: one controller per family per anchor, tuned at that limit")
        for a in ANCHORS:
            rows_store["informed"][a] = {}
            for fam in FAMILIES:
                t = time.monotonic()
                p, v, info = search_family(pool, fam, triples(anchor_pairs(search_seeds, a)), rng, budget, workers)
                rows = evaluate_many(pool, fam, [p], triples(anchor_pairs(fresh_seeds, a)), workers)[0]
                search["informed"][a][fam] = {"params": p, "physical": physical_params(fam, p), "search_reward": v,
                                              "eval": summarize(rows), "seconds": time.monotonic() - t, **info}
                rows_store["informed"][a][fam] = rows
                log(f"  informed sigma_max={a:.3f} {fam:7s} {fmt(summarize(rows))}  physical {physical_params(fam, p)}  [{time.monotonic() - t:.0f}s]")

        # ---- the cautious blind reference on the same fresh batches -------------------------
        log("cautious blind reference: rides a hair under the bottom of the range")
        caut_rows = []
        for (s, m, o) in triples(pairs_fresh_range):
            caut_rows.append(episode_row(run_cautious_episode(s, HiddenParams(sigma_max=m)), s, m, o))
        caut_anchor = {}
        for a in ANCHORS:
            caut_anchor[a] = summarize([episode_row(run_cautious_episode(s, HiddenParams(sigma_max=m)), s, m, o)
                                        for (s, m, o) in triples(anchor_pairs(fresh_seeds, a))])
        cautious = {"eval": summarize(caut_rows), "eval_by_third": by_third(caut_rows), "eval_at_anchors": caut_anchor}
        rows_store["cautious"] = caut_rows
        log(f"  cautious {fmt(cautious['eval'])}")

    # ---- verdicts G1, G2 -------------------------------------------------------------------
    best_blind_fam = max(search["blind"], key=lambda f: search["blind"][f]["eval"]["reward"])
    best_blind = search["blind"][best_blind_fam]
    best_informed = {}
    for a in ANCHORS:
        fam = max(search["informed"][a], key=lambda f: search["informed"][a][f]["eval"]["reward"])
        best_informed[a] = {"family": fam, **search["informed"][a][fam]["eval"]}
    informed_mean_fast = statistics.fmean(best_informed[a]["fast"] for a in ANCHORS)
    g1 = all(best_informed[a]["fast"] >= GATES["g1_informed_fast_min"] for a in ANCHORS)
    gap = informed_mean_fast - best_blind["eval"]["fast"]
    g2 = gap >= GATES["g2_gap_min"]
    verdicts = {"best_blind_family": best_blind_fam, "best_blind_physical": best_blind["physical"],
                "blind_fast_range": best_blind["eval"]["fast"], "blind_success_range": best_blind["eval"]["success"],
                "blind_fast_at_anchors": {a: best_blind["eval_at_anchors"][a]["fast"] for a in ANCHORS},
                "best_informed": best_informed, "informed_mean_fast": informed_mean_fast,
                "gap_pp": 100 * gap, "G1": g1, "G2": g2}
    log(f"best blind family {best_blind_fam} {best_blind['physical']}: fast {100 * best_blind['eval']['fast']:.1f}% over the range, "
        f"at anchors {[round(100 * best_blind['eval_at_anchors'][a]['fast']) for a in ANCHORS]}")
    log(f"best informed at anchors: " + ", ".join(f"{a:.3f}: {best_informed[a]['family']} fast {100 * best_informed[a]['fast']:.1f}%" for a in ANCHORS))
    log(f"G1 informed fast >= {100 * GATES['g1_informed_fast_min']:.0f}% at every anchor: {'PASS' if g1 else 'FAIL'}")
    log(f"G2 gap {100 * gap:+.0f} pp (need >= {100 * GATES['g2_gap_min']:.0f}): {'PASS' if g2 else 'FAIL'}")

    # ---- G3: probe + controller through the real path ---------------------------------------
    log("G3: hand-written probe run and controller through the sandbox")
    fresh_tasks = [task_entry("stage0_fresh", s, sigma_max=m) for s, m in pairs_fresh_range]
    anchor_tasks = {a: [task_entry("stage0_anchor", s, sigma_max=a) for s in fresh_seeds] for a in ANCHORS}
    g3 = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for variant in PROBE_VARIANTS:
            t = time.monotonic()
            rows = list(ex.map(lambda tk, v=variant: run_two_turn(tk, v), fresh_tasks))
            g3[variant] = {"range": summarize_two_turn(rows), "range_by_third": {k: summarize_two_turn(v) for k, v in
                           ((th, [r for r in rows if r["third"] == th]) for th in THIRDS) if v}}
            rows_store["g3"][variant] = {"range": rows}
            if variant == GATE_VARIANT:
                g3[variant]["anchors"] = {}
                for a in ANCHORS:
                    arows = list(ex.map(lambda tk, v=variant: run_two_turn(tk, v), anchor_tasks[a]))
                    g3[variant]["anchors"][a] = summarize_two_turn(arows)
                    rows_store["g3"][variant][f"anchor_{a}"] = arows
            g3[variant]["seconds"] = time.monotonic() - t
            s = g3[variant]["range"]
            log(f"  {variant:11s} {fmt(s)}  probe failures {s['probe_failures']}  probe error {s['probe_error']}  [{time.monotonic() - t:.0f}s]")
        gate = g3[GATE_VARIANT]
        g3_pass = (all(gate["anchors"][a]["fast"] >= GATES["g3_fast_min"] for a in ANCHORS)
                   and gate["range"]["probe_failures"] == 0
                   and all(gate["anchors"][a]["probe_failures"] == 0 for a in ANCHORS))
        verdicts["G3"] = g3_pass
        verdicts["g3_fast_at_anchors"] = {a: gate["anchors"][a]["fast"] for a in ANCHORS}
        log(f"G3 {GATE_VARIANT} fast >= {100 * GATES['g3_fast_min']:.0f}% at every anchor, no probe failure: "
            f"{[round(100 * gate['anchors'][a]['fast'], 1) for a in ANCHORS]} -> {'PASS' if g3_pass else 'FAIL'}")

        # ---- references on the held-out evaluation tasks --------------------------------------
        log(f"references on the {len(eval_tasks)} held-out tasks")
        ref_rows = []
        probe_rows = list(ex.map(lambda tk: run_two_turn(tk, GATE_VARIANT), eval_tasks))
        blind_ctl_args = (best_blind_fam, best_blind["params"])
        for task, pr in zip(eval_tasks, probe_rows):
            sc = task["scenarios"][0]
            smax, seed, o = sc["hidden"]["sigma_max"], sc["seed"], sc["oracle_steps"]
            hidden = HiddenParams(sigma_max=smax)
            orc = episode_row(run_oracle_episode(seed, hidden), seed, smax, o)
            caut = episode_row(run_cautious_episode(seed, hidden), seed, smax, o)
            blind = episode_row(run_episode(make_controller(*blind_ctl_args), seed, hidden, policy_sees_env=True), seed, smax, o)
            ref_rows.append({"task_id": task["task_id"], "seed": seed, "sigma_max": smax, "third": third_of(smax),
                             "oracle": orc, "cautious": caut, "blind_searched": blind, "probe_controller": pr})
    refs = {"per_task": ref_rows}
    for name in ("oracle", "cautious", "blind_searched"):
        rows = [r[name] for r in ref_rows]
        refs[name] = {"overall": summarize(rows), "by_third": by_third(rows)}
    rows = [r["probe_controller"] for r in ref_rows]
    refs["probe_controller"] = {"overall": summarize_two_turn(rows),
                                "by_third": {th: summarize_two_turn([r for r in rows if r["third"] == th]) for th in THIRDS}}
    for name in ("oracle", "cautious", "blind_searched", "probe_controller"):
        log(f"  {name:16s} {fmt(refs[name]['overall'])}")
    rows_store["references"] = ref_rows

    result = {"provenance": prov, "config": config, "search": {
                  "blind": search["blind"],
                  "informed": {str(a): v for a, v in search["informed"].items()}},
              "cautious": {"eval": cautious["eval"], "eval_by_third": cautious["eval_by_third"],
                           "eval_at_anchors": {str(a): v for a, v in cautious["eval_at_anchors"].items()}},
              "verdicts": {k: ({str(a): x for a, x in v.items()} if isinstance(v, dict) and all(isinstance(a, float) for a in v) else v)
                           for k, v in verdicts.items()},
              "g3": {variant: {**{k: v for k, v in d.items() if k != "anchors"},
                               **({"anchors": {str(a): x for a, x in d["anchors"].items()}} if "anchors" in d else {})}
                     for variant, d in g3.items()},
              "references": refs, "seconds": time.time() - t_start}
    (out / "stage0_results.json").write_text(json.dumps(result, indent=1))
    (out / "stage0_rows.json").write_text(json.dumps(
        {"provenance": prov,
         "blind": rows_store["blind"], "cautious": rows_store["cautious"],
         "informed": {str(a): v for a, v in rows_store["informed"].items()},
         "g3": rows_store["g3"], "references": rows_store["references"]}))
    log(f"wrote {out / 'stage0_results.json'} and stage0_rows.json; total {time.time() - t_start:.0f}s")
    log(f"VERDICT G1 {'PASS' if g1 else 'FAIL'}  G2 {'PASS' if g2 else 'FAIL'}  G3 {'PASS' if g3_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
