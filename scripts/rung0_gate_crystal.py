"""Rung 0 validity gate for the seeded cooling crystallizer.

Question: does knowing the hidden metastable limit (sigma_max) buy anything that a
controller without it cannot get from the final run's own readings?

Protocol:
  * blind side     one controller for every hidden value
  * informed side  one controller per hidden value
  * identical controller families, identical optimizer budget
    (random sample in the unit cube, then hill-climbing from the top-k)
  * searched on seeds 0-47, reported on 200 fresh seeds 1000-1199
  * optimizer objective on both sides: mean env reward, i.e. what RL would maximize

Scores per episode:
  success        finished without a shower inside the horizon
  fast success   success AND steps <= 1.25 x the oracle's steps for that (seed, sigma_max)
  norm time      oracle steps / own steps, 0 on failure
The oracle is the one-step-lookahead controller that rides exactly at sigma_max.

Pre-registered gates (decided before any search was run):
  G1  informed fast-success >= 90% at every anchor value of sigma_max
  G2  blind fast-success at least 40 pp below informed over the hidden range
  G3  hand-written probe run + final run >= 95% fast-success at every anchor

Hidden-value configs:  "range:0.08-0.25"  (continuous; anchors low / mid / high are reported)
                       "set:0.08,0.12,0.18,0.25"  (discrete)

Usage:
  python scripts/rung0_gate_crystal.py search --configs range:0.08-0.25 set:0.08,0.12,0.18,0.25 --out data/rung0/crystal.json
  python scripts/rung0_gate_crystal.py two-turn --configs range:0.08-0.25 [--show-source]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "seed_tasks"))
from crystallizer_env import (  # noqa: E402
    CrystallizerEnv, EnvConfig, HiddenParams, saturation_temperature, solubility,
)

CFG = EnvConfig()
ENV_KW: dict = {}      # EnvConfig overrides (e.g. time_weight) set from the CLI; copied into workers
SEARCH_SEEDS = list(range(48))
EVAL_SEEDS = list(range(1000, 1200))
GATE_INFORMED_MIN = 0.90
GATE_GAP_MIN = 0.40
GATE_TWO_TURN_MIN = 0.95
FAST_FACTOR = 1.25

FAMILIES = {"track": 1, "linear": 1, "power": 2, "pwl": 10, "hybrid": 3}


# ----------------------------------------------------------------------------------------
# hidden-value configs
# ----------------------------------------------------------------------------------------

def parse_config(spec: str) -> dict:
    kind, _, body = spec.partition(":")
    if kind == "range":
        lo, hi = (float(x) for x in body.split("-"))
        return {"spec": spec, "kind": "range", "lo": lo, "hi": hi,
                "anchors": [lo, 0.5 * (lo + hi), hi]}
    if kind == "set":
        vals = [float(x) for x in body.split(",")]
        return {"spec": spec, "kind": "set", "values": vals, "anchors": vals}
    raise ValueError(f"bad config spec {spec!r}")


def hidden_draw(cfg: dict, seed: int) -> float:
    """The hidden value attached to a seed when the whole range is in play."""
    if cfg["kind"] == "range":
        return random.Random(10_007 * seed + 17).uniform(cfg["lo"], cfg["hi"])
    return cfg["values"][seed % len(cfg["values"])]


# ----------------------------------------------------------------------------------------
# controller families  (every family outputs a desired temperature; actions follow)
# ----------------------------------------------------------------------------------------

def _to_action(T_des: float, T: float) -> float:
    a = (T_des - T) / (CFG.max_rate * CFG.dt / 60.0)
    return max(-1.0, min(1.0, a))


def _track_T(state: dict, sigma_t: float) -> float:
    """Temperature that puts the pre-growth supersaturation exactly at sigma_t."""
    T_des = saturation_temperature(state["c"] / (1.0 + sigma_t), CFG)
    return max(T_des, state["T_target"])


def make_policy(family: str, p: list[float]):
    """p is in the unit cube; mapped to physical parameters here."""
    mem: dict = {}

    def t_min(state):
        return state["t"] * CFG.dt / 60.0

    def profile_T(state, tau_min, power):
        T0 = mem.setdefault("T0", state["T"])
        frac = min(1.0, t_min(state) / max(tau_min, 1e-6)) ** power
        return T0 - (T0 - state["T_target"]) * frac

    if family == "track":
        sigma_t = 0.02 + 0.33 * p[0]

        def pol(state):
            return _to_action(_track_T(state, sigma_t), state["T"])
    elif family == "linear":
        rate = 0.02 + 1.48 * p[0]

        def pol(state):
            T0 = mem.setdefault("T0", state["T"])
            return _to_action(max(state["T_target"], T0 - rate * t_min(state)), state["T"])
    elif family == "power":
        tau, power = 10.0 + 190.0 * p[0], 0.5 + 3.5 * p[1]

        def pol(state):
            return _to_action(profile_T(state, tau, power), state["T"])
    elif family == "hybrid":
        tau, power, cap = 10.0 + 190.0 * p[0], 0.5 + 3.5 * p[1], 0.02 + 0.33 * p[2]

        def pol(state):
            T_prof = profile_T(state, tau, power)
            return _to_action(max(T_prof, _track_T(state, cap)), state["T"])
    elif family == "pwl":
        knots_t = [0.0] + sorted(p[:4]) + [1.0]
        vals = list(p[4:10])

        def pol(state):
            T0 = mem.setdefault("T0", state["T"])
            x = state["t"] / CFG.max_steps
            for i in range(5):
                if x <= knots_t[i + 1]:
                    w = (x - knots_t[i]) / max(knots_t[i + 1] - knots_t[i], 1e-9)
                    f = vals[i] + (vals[i + 1] - vals[i]) * w
                    break
            else:
                f = vals[5]
            return _to_action(state["T_target"] + f * (T0 - state["T_target"]), state["T"])
    else:
        raise ValueError(family)
    return pol


# ----------------------------------------------------------------------------------------
# episodes
# ----------------------------------------------------------------------------------------

def _init_worker(env_kw: dict):
    ENV_KW.clear()
    ENV_KW.update(env_kw)


def run_episode(policy, seed: int, sigma_max: float) -> dict:
    env = CrystallizerEnv(seed=seed, hidden=HiddenParams(sigma_max=sigma_max), config=EnvConfig(**ENV_KW))
    state = env.observe()
    total = 0.0
    while True:
        try:
            a = policy(state)
        except Exception:
            return {"reward": -200.0, "success": False, "steps": env.t, "reason": "policy_exception"}
        state, r, done, _, _ = env.step(a)
        total += r
        if done:
            break
    return {"reward": total, "success": bool(env.is_success()), "steps": env.t,
            "reason": env.failure_reason}


_ORACLE: dict = {}


def oracle_steps(seed: int, sigma_max: float) -> int:
    key = (seed, round(sigma_max, 12))
    if key not in _ORACLE:
        res = run_episode(make_policy("track", [(sigma_max * (1 - 1e-9) - 0.02) / 0.33]), seed, sigma_max)
        if not res["success"]:
            raise RuntimeError(f"oracle failed at seed {seed} sigma_max {sigma_max}: {res}")
        _ORACLE[key] = res["steps"]
    return _ORACLE[key]


def evaluate(family: str, p: list[float], pairs: list[tuple[int, float]]) -> dict:
    """Mean metrics of one controller over (seed, sigma_max) pairs."""
    rew = succ = fast = norm = 0.0
    for seed, smax in pairs:
        res = run_episode(make_policy(family, p), seed, smax)
        o = oracle_steps(seed, smax)
        rew += res["reward"]
        succ += res["success"]
        fast += res["success"] and res["steps"] <= FAST_FACTOR * o
        norm += (o / res["steps"]) if res["success"] else 0.0
    n = len(pairs)
    return {"reward": rew / n, "success": succ / n, "fast": fast / n, "norm_time": norm / n}


# ----------------------------------------------------------------------------------------
# search  (random sample + hill climb, parallel over candidates)
# ----------------------------------------------------------------------------------------

def _eval_job(args):
    family, p, pairs = args
    return evaluate(family, p, pairs)["reward"]


def search_family(family: str, pairs: list[tuple[int, float]], pool, rng: random.Random,
                  n_random: int, topk: int, n_climb: int) -> tuple[list[float], float]:
    d = FAMILIES[family]
    cands = [[rng.random() for _ in range(d)] for _ in range(n_random)]
    vals = pool.map(_eval_job, [(family, p, pairs) for p in cands], chunksize=4)
    order = sorted(range(n_random), key=lambda i: -vals[i])[:topk]
    pop = [cands[i][:] for i in order]
    val = [vals[i] for i in order]
    for it in range(n_climb):
        scale = 0.15 * (1 - it / n_climb) + 0.01
        props = [[min(1.0, max(0.0, x + rng.gauss(0, scale))) for x in pp] for pp in pop]
        pv = pool.map(_eval_job, [(family, q, pairs) for q in props], chunksize=1)
        for k in range(topk):
            if pv[k] > val[k]:
                pop[k], val[k] = props[k], pv[k]
    best = max(range(topk), key=lambda k: val[k])
    return pop[best], val[best]


def anchor_pairs(cfg: dict, value: float, seeds: list[int]) -> list[tuple[int, float]]:
    return [(s, value) for s in seeds]


def range_pairs(cfg: dict, seeds: list[int]) -> list[tuple[int, float]]:
    return [(s, hidden_draw(cfg, s)) for s in seeds]


def cmd_search(args):
    ctx = get_context("spawn")
    rng = random.Random(0)
    out = {"configs": [], "budget": {"n_random": args.n_random, "topk": args.topk, "n_climb": args.n_climb}}
    t_start = time.time()
    if args.time_weight is not None:
        ENV_KW["time_weight"] = args.time_weight
    out["env_overrides"] = dict(ENV_KW)
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(dict(ENV_KW),)) as pool:
        for spec in args.configs:
            cfg = parse_config(spec)
            print(f"\n=== config {spec}  env overrides {ENV_KW or 'none'} ===", flush=True)
            search_blind = range_pairs(cfg, SEARCH_SEEDS)
            eval_range = range_pairs(cfg, EVAL_SEEDS)
            report = {"spec": spec, "anchors": cfg["anchors"], "blind": {}, "informed": {}}

            # blind: one controller per family for the whole hidden set
            for fam in FAMILIES:
                p, v = search_family(fam, search_blind, pool, rng, args.n_random, args.topk, args.n_climb)
                row = {"params": p, "search_reward": v, "range": evaluate(fam, p, eval_range)}
                for a in cfg["anchors"]:
                    row[f"anchor_{a:.3f}"] = evaluate(fam, p, anchor_pairs(cfg, a, EVAL_SEEDS))
                report["blind"][fam] = row
                r = row["range"]
                print(f"  blind    {fam:7s} reward {r['reward']:7.1f}  success {r['success']*100:5.1f}%  "
                      f"fast {r['fast']*100:5.1f}%  norm-time {r['norm_time']:.3f}   "
                      f"[{time.time()-t_start:5.0f}s]", flush=True)

            # informed: one controller per family per anchor value
            for a in cfg["anchors"]:
                report["informed"][f"{a:.3f}"] = {}
                for fam in FAMILIES:
                    p, v = search_family(fam, anchor_pairs(cfg, a, SEARCH_SEEDS), pool, rng,
                                         args.n_random, args.topk, args.n_climb)
                    ev = evaluate(fam, p, anchor_pairs(cfg, a, EVAL_SEEDS))
                    report["informed"][f"{a:.3f}"][fam] = {"params": p, "search_reward": v, "eval": ev}
                    print(f"  informed sigma_max={a:.3f} {fam:7s} reward {ev['reward']:7.1f}  "
                          f"success {ev['success']*100:5.1f}%  fast {ev['fast']*100:5.1f}%  "
                          f"norm-time {ev['norm_time']:.3f}   [{time.time()-t_start:5.0f}s]", flush=True)

            summarize(report)
            out["configs"].append(report)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(out, indent=1))


def summarize(report: dict):
    """Best blind (by mean reward over the range) vs best informed per anchor; gate verdicts."""
    best_blind_fam = max(report["blind"], key=lambda f: report["blind"][f]["range"]["reward"])
    bb = report["blind"][best_blind_fam]
    print(f"\n  best blind family by reward: {best_blind_fam}")
    print(f"  {'sigma_max':>10s} {'blind fast':>11s} {'inf fast':>9s} {'gap':>6s} | "
          f"{'blind norm-t':>12s} {'inf norm-t':>10s} | {'blind reward':>12s} {'inf reward':>10s}")
    g1 = True
    inf_fast_over_anchors = []
    for a in report["anchors"]:
        key = f"{a:.3f}"
        inf_fam = max(report["informed"][key], key=lambda f: report["informed"][key][f]["eval"]["reward"])
        inf = report["informed"][key][inf_fam]["eval"]
        bl = bb[f"anchor_{a:.3f}"]
        g1 &= inf["fast"] >= GATE_INFORMED_MIN
        inf_fast_over_anchors.append(inf["fast"])
        print(f"  {a:10.3f} {bl['fast']*100:10.1f}% {inf['fast']*100:8.1f}% {100*(inf['fast']-bl['fast']):+5.0f} | "
              f"{bl['norm_time']:12.3f} {inf['norm_time']:10.3f} | {bl['reward']:12.1f} {inf['reward']:10.1f}   "
              f"(informed family: {inf_fam})")
    rng_fast = bb["range"]["fast"]
    inf_mean_fast = sum(inf_fast_over_anchors) / len(inf_fast_over_anchors)
    gap = inf_mean_fast - rng_fast
    print(f"  over the range: blind fast {rng_fast*100:.1f}%, blind norm-time {bb['range']['norm_time']:.3f}, "
          f"blind reward {bb['range']['reward']:.1f}")
    print(f"  G1 informed fast >= {GATE_INFORMED_MIN*100:.0f}% at every anchor: {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 fast-success gap {gap*100:+.0f} pp (need >= {GATE_GAP_MIN*100:.0f}): {'PASS' if gap >= GATE_GAP_MIN else 'FAIL'}")
    report["verdict"] = {"best_blind_family": best_blind_fam, "G1": g1, "G2_gap_pp": 100 * gap,
                         "G2": gap >= GATE_GAP_MIN}


# ----------------------------------------------------------------------------------------
# two-turn hand-written solution  (probe run as the model would write it, then final run)
# ----------------------------------------------------------------------------------------

PROBE_SOURCE = '''
# MODE: main
# Probe run: find the shower limit by crossing it. Three passes, each one batch.
cfg = env.config
def sat_T(conc):
    return saturation_temperature(conc, cfg)
def step_to_sigma(sig):
    """one step toward the temperature that puts the pre-growth supersaturation at sig"""
    T_des = max(sat_T(s["c"] / (1.0 + sig)), s["T_target"])
    return max(-1.0, min(1.0, (T_des - s["T"]) / (cfg.max_rate * cfg.dt / 60.0)))

lo, hi = 0.0, 0.40                       # bracket on sigma_max: safe below lo, shower above hi
for npass in range(3):
    s, _ = env.reset()
    target = lo                          # go quickly to the last safe level ...
    inc = (hi - lo) / 8.0                # ... then creep up in eighths of the bracket
    while True:
        c_prev = s["c"]
        a = step_to_sigma(target) if npass == 0 and lo == 0.0 else step_to_sigma(target)
        if npass == 0 and lo == 0.0:
            a = -1.0                     # first pass: cool at the jacket limit
        s, r, done, _, _ = env.step(a)
        sig_pre = c_prev / solubility(s["T"], cfg) - 1.0   # the value the shower test used
        if done:
            if env.is_success():
                lo = max(lo, sig_pre); break
            hi = min(hi, sig_pre); break
        lo = max(lo, sig_pre)
        if abs(sig_pre - target) < 1e-4 or npass == 0:
            target = min(target + inc, hi)
print(f"SIGMA_MAX_LOW={lo:.5f} SIGMA_MAX_HIGH={hi:.5f}")
'''

PROBE_ONE_PASS_SOURCE = """
# MODE: main
# Probe run, one batch: cool at the jacket limit until the shower; the last safe level is the estimate.
cfg = env.config
s, _ = env.reset()
lo, hi = 0.0, 0.40
while True:
    c_prev = s["c"]
    s, r, done, _, _ = env.step(-1.0)
    sig_pre = c_prev / solubility(s["T"], cfg) - 1.0
    if done:
        if not env.is_success():
            hi = sig_pre
        break
    lo = max(lo, sig_pre)
print(f"SIGMA_MAX_LOW={lo:.5f} SIGMA_MAX_HIGH={hi:.5f}")
"""

PROBE_CREEP_TEMPLATE = """
# MODE: main
# Probe run, one batch: raise the supersaturation target by STEP every step until the shower.
STEP = {step}
cfg = env.config
s, _ = env.reset()
lo, hi, target = 0.0, 0.40, STEP
while True:
    c_prev = s["c"]
    T_des = max(saturation_temperature(c_prev / (1.0 + target), cfg), s["T_target"])
    a = max(-1.0, min(1.0, (T_des - s["T"]) / (cfg.max_rate * cfg.dt / 60.0)))
    s, r, done, _, _ = env.step(a)
    sig_pre = c_prev / solubility(s["T"], cfg) - 1.0
    if done:
        if not env.is_success():
            hi = sig_pre
        break
    lo = max(lo, sig_pre)
    target += STEP
print(f"SIGMA_MAX_LOW={{lo:.5f}} SIGMA_MAX_HIGH={{hi:.5f}}")
"""


def probe_source(variant: str, creep_step: float) -> str:
    if variant == "three-pass":
        return PROBE_SOURCE
    if variant == "one-pass":
        return PROBE_ONE_PASS_SOURCE
    if variant == "creep":
        return PROBE_CREEP_TEMPLATE.format(step=creep_step)
    raise ValueError(variant)


FINAL_TEMPLATE = '''
# MODE: policy
SIGMA_HAT = {sigma_hat:.5f}
MARGIN = 0.02
def policy(state):
    cfg = EnvConfig()
    T_des = saturation_temperature(state["c"] / (1.0 + SIGMA_HAT * (1.0 - MARGIN)), cfg)
    T_des = max(T_des, state["T_target"])
    a = (T_des - state["T"]) / (cfg.max_rate * cfg.dt / 60.0)
    return max(-1.0, min(1.0, a))
'''


def run_probe(seed: int, sigma_max: float, source: str = None) -> tuple[float, float]:
    """Execute a probe-run source against a fresh env exactly as the runner would (same namespace)."""
    env = CrystallizerEnv(seed=seed, hidden=HiddenParams(sigma_max=sigma_max), config=EnvConfig(**ENV_KW))
    ns = {"env": env, "solubility": solubility, "saturation_temperature": saturation_temperature,
          "EnvConfig": EnvConfig}
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(source if source is not None else PROBE_SOURCE, ns)
    out = buf.getvalue()
    lo = float(out.split("SIGMA_MAX_LOW=")[1].split()[0])
    hi = float(out.split("SIGMA_MAX_HIGH=")[1].split()[0])
    return lo, hi


def final_policy_from_source(sigma_hat: float):
    ns = {"EnvConfig": EnvConfig, "saturation_temperature": saturation_temperature}
    exec(FINAL_TEMPLATE.format(sigma_hat=sigma_hat), ns)
    return ns["policy"]


def _two_turn_job(args):
    seed, smax, variant, creep_step = args
    lo, hi = run_probe(seed, smax, probe_source(variant, creep_step))
    pol = final_policy_from_source(lo)
    res = run_episode(pol, seed, smax)
    o = oracle_steps(seed, smax)
    return {"seed": seed, "sigma_max": smax, "lo": lo, "hi": hi, "success": res["success"],
            "fast": res["success"] and res["steps"] <= FAST_FACTOR * o,
            "norm_time": (o / res["steps"]) if res["success"] else 0.0, "reward": res["reward"]}


def cmd_two_turn(args):
    if args.show_source:
        print(probe_source(args.probe, args.creep_step))
        print(FINAL_TEMPLATE)
    ctx = get_context("spawn")
    if args.time_weight is not None:
        ENV_KW["time_weight"] = args.time_weight
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(dict(ENV_KW),)) as pool:
        for spec in args.configs:
            cfg = parse_config(spec)
            print(f"\n=== two-turn, config {spec}  probe {args.probe}"
                  f"{' step ' + str(args.creep_step) if args.probe == 'creep' else ''}"
                  f"  env overrides {ENV_KW or 'none'} ===")
            all_ok = True
            for a in cfg["anchors"] + ["range"]:
                pairs = range_pairs(cfg, EVAL_SEEDS) if a == "range" else anchor_pairs(cfg, a, EVAL_SEEDS)
                rows = pool.map(_two_turn_job, [(sd, sm, args.probe, args.creep_step) for sd, sm in pairs], chunksize=4)
                n = len(rows)
                fast = sum(r["fast"] for r in rows) / n
                succ = sum(r["success"] for r in rows) / n
                nt = sum(r["norm_time"] for r in rows) / n
                err = max(abs(r["lo"] - r["sigma_max"]) for r in rows)
                width = max(r["hi"] - r["lo"] for r in rows)
                label = f"sigma_max={a:.3f}" if a != "range" else "whole range  "
                print(f"  {label}: success {succ*100:5.1f}%  fast {fast*100:5.1f}%  norm-time {nt:.3f}  "
                      f"probe error <= {err:.4f}  bracket width <= {width:.4f}")
                if a != "range":
                    all_ok &= fast >= GATE_TWO_TURN_MIN
            print(f"  G3 hand-written two-turn fast-success >= {GATE_TWO_TURN_MIN*100:.0f}% at every anchor: "
                  f"{'PASS' if all_ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--configs", nargs="+", default=["range:0.08-0.25"])
    s.add_argument("--n-random", type=int, default=300)
    s.add_argument("--topk", type=int, default=6)
    s.add_argument("--n-climb", type=int, default=120)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--out", default="")
    s.add_argument("--time-weight", type=float, default=None, help="override EnvConfig.time_weight (sensitivity check)")
    s.set_defaults(func=cmd_search)
    t = sub.add_parser("two-turn")
    t.add_argument("--configs", nargs="+", default=["range:0.08-0.25"])
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t.add_argument("--show-source", action="store_true")
    t.add_argument("--probe", choices=["three-pass", "one-pass", "creep"], default="three-pass",
                   help="probe-run budget: three batches (default), one batch at the jacket limit, "
                        "or one batch creeping the supersaturation target up by --creep-step per step")
    t.add_argument("--creep-step", type=float, default=0.005)
    t.add_argument("--time-weight", type=float, default=None, help="override EnvConfig.time_weight (sensitivity check)")
    t.set_defaults(func=cmd_two_turn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
