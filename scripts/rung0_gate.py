#!/usr/bin/env python3
"""Rung 0 gate: does the slosh task NEED the turn-1 lab?

Setting under test (the proposed redesign, nothing in the env is modified):
  * The scored controller (`policy(state)`, written in turn 2) observes only
    (t, x, v, goal, steps_left). The slosh angle theta and its rate omega are
    hidden from it: the lab has a sensor on the liquid, the deployed cart only
    has an encoder.
  * The hidden parameter is the slosh length L (sets the slosh period
    Tn = 2*pi*sqrt(L/g)). The actuator gain kappa is fixed at 1.0.
  * Because theta never enters the cart update (see SloshTankEnv.step), the
    scored controller's inputs are independent of L. No controller, stateful or
    not, can identify L in-episode. The only blind strategy is robustness.

This script measures what robustness costs, adversarially:
  blind     one controller for every L in the set (robust objective)
  informed  one controller per L (upper bound on what a perfect turn-1
            measurement could buy)
Both sides search the SAME controller classes with the SAME budget on the SAME
search seeds and are reported on fresh seeds.

Pre-registered gate (per configuration):
  G1  informed >= 90% success at every L in the set
  G2  informed_mean - blind_mean >= 40 percentage points
  G3  (two-turn subcommand) a hand-written turn-1 measurement + turn-2
      controller reaches >= 95% on fresh seeds

Subcommands:
  bench      episodes per second on this machine (to size the search)
  search     G1/G2 over a grid of (L set, step budget) configurations
  two-turn   G3 for one configuration

Usage:
  python scripts/rung0_gate.py bench
  python scripts/rung0_gate.py search --workers 11 --out data/rung0/search.json
  python scripts/rung0_gate.py two-turn --L 2 3 4 --steps 90
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "seed_tasks"))
from slosh_tank_env import EnvConfig, HiddenParams, SloshTankEnv  # noqa: E402

DT = EnvConfig().dt
G = EnvConfig().gravity
KAPPA = 1.0
FIELD_KEYS = ("t", "x", "v", "goal", "steps_left")
CLASSES = ("bang", "free", "pd", "shaped_pd", "ff_fb")
FREE_K = 8  # segments per half in the free open-loop profile

SEARCH_SEEDS = list(range(48))
EVAL_SEEDS = list(range(1000, 1200))
LAB_SEEDS = list(range(5000, 5016))

GATE_INFORMED_MIN = 0.90
GATE_GAP_MIN = 0.40
GATE_TWO_TURN_MIN = 0.95


def Tn(L: float) -> float:
    return 2.0 * math.pi * math.sqrt(L / G)


# ----------------------------------------------------------------------------
# Input shapers: list of (delay_seconds, weight), weights sum to 1
# ----------------------------------------------------------------------------

def shaper(kind: int, td1: float, td2: float) -> List[Tuple[float, float]]:
    def zv(T):
        return [(0.0, 0.5), (T / 2, 0.5)]

    def zvd(T):
        return [(0.0, 0.25), (T / 2, 0.5), (T, 0.25)]

    def zvdd(T):
        return [(0.0, 0.125), (T / 2, 0.375), (T, 0.375), (1.5 * T, 0.125)]

    def ei(T, V=0.05):
        return [(0.0, (1 + V) / 4), (T / 2, (1 - V) / 2), (T, (1 + V) / 4)]

    def conv(s1, s2):
        return [(t1 + t2, w1 * w2) for t1, w1 in s1 for t2, w2 in s2]

    table = [
        [(0.0, 1.0)],
        zv(td1),
        zvd(td1),
        zvdd(td1),
        ei(td1),
        conv(zv(td1), zv(td2)),
        conv(zvd(td1), zv(td2)),
    ]
    return table[kind]


SHAPER_NAMES = ["none", "ZV", "ZVD", "ZVDD", "EI", "ZVxZV", "ZVDxZV"]


# ----------------------------------------------------------------------------
# Open-loop move sequences (discrete time, exact landing when unclipped)
# ----------------------------------------------------------------------------

def _apply_shaper(seq: List[float], imps) -> List[float]:
    out = [0.0] * (len(seq) + max(int(round(t / DT)) for t, _ in imps))
    for tau, w in imps:
        n = int(round(tau / DT))
        for i, a in enumerate(seq):
            out[i + n] += w * a
    return out


def _simulate_cart(seq: Sequence[float]) -> Tuple[float, float, List[float], List[float]]:
    """Cart recursion exactly as in SloshTankEnv.step with kappa=1."""
    x = v = 0.0
    xs, vs = [], []
    for a in seq:
        v += DT * a
        x += DT * v
        xs.append(x)
        vs.append(v)
    return x, v, xs, vs


def _finish_sequence(unscaled: List[float], d: float, sg: float):
    """Scale an antisymmetric sequence so the cart lands exactly at distance d
    (zero final velocity is guaranteed by antisymmetry). Returns the clipped
    action sequence plus reference x/v arrays, or None if degenerate."""
    disp, _, _, _ = _simulate_cart(unscaled)
    if disp <= 1e-6:
        return None
    s = sg * d / disp
    seq = [max(-1.0, min(1.0, s * a)) for a in unscaled]
    _, _, xs, vs = _simulate_cart(seq)
    return seq, xs, vs


def bang_sequence(A: float, d: float, sg: float, imps):
    n1 = max(1, int(round(math.sqrt(d / A) / DT)))
    base = [A] * n1 + [-A] * n1
    return _finish_sequence(_apply_shaper(base, imps), d, sg)


def free_sequence(beta: float, coeffs: Sequence[float], d: float, sg: float, imps):
    H = beta * math.sqrt(d)
    K = len(coeffs)
    nseg = max(1, int(round(H / K / DT)))
    full = list(coeffs) + [-c for c in reversed(coeffs)]
    base = []
    for c in full:
        base.extend([c] * nseg)
    return _finish_sequence(_apply_shaper(base, imps), d, sg)


# ----------------------------------------------------------------------------
# Controller classes. Each decoder maps a normalized vector u in [0,1]^n to a
# *factory*: factory() returns a fresh policy for one episode (so stateful
# classes get clean memory). Policies receive only FIELD_KEYS.
# ----------------------------------------------------------------------------

def _kind(u):
    return min(6, int(u * 7))


def _td(u):
    return 1.0 + 5.0 * u  # 1 .. 6 s


def _kp(u):
    return 10 ** (-1.3 + 2.0 * u)  # 0.05 .. 5


def _kd(u):
    return 10 ** (-1.0 + 1.7 * u)  # 0.1 .. 5


def _amp(u):
    return 0.02 + 0.98 * u


def _A(u):
    return 0.1 + 0.9 * u


def _clip1(a):
    return max(-1.0, min(1.0, a))


def describe(cls: str, u: Sequence[float]) -> str:
    if cls == "bang":
        return (f"A={_A(u[0]):.2f} shaper={SHAPER_NAMES[_kind(u[1])]} Td1={_td(u[2]):.2f} "
                f"Td2={_td(u[3]):.2f} kp={_kp(u[4]):.2f} kd={_kd(u[5]):.2f} apd={_amp(u[6]):.2f}")
    if cls == "free":
        c = ",".join(f"{2 * x - 1:+.2f}" for x in u[1:1 + FREE_K])
        return (f"beta={0.6 + 2.4 * u[0]:.2f} c=[{c}] shaper={SHAPER_NAMES[_kind(u[1 + FREE_K])]} "
                f"Td1={_td(u[2 + FREE_K]):.2f} Td2={_td(u[3 + FREE_K]):.2f} kp={_kp(u[4 + FREE_K]):.2f} "
                f"kd={_kd(u[5 + FREE_K]):.2f} apd={_amp(u[6 + FREE_K]):.2f}")
    if cls == "pd":
        return f"kp={_kp(u[0]):.2f} kd={_kd(u[1]):.2f} amax={_amp(u[2]):.2f}"
    if cls == "shaped_pd":
        return (f"kp={_kp(u[0]):.2f} kd={_kd(u[1]):.2f} amax={_amp(u[2]):.2f} "
                f"shaper={SHAPER_NAMES[_kind(u[3])]} Td1={_td(u[4]):.2f} Td2={_td(u[5]):.2f}")
    if cls == "ff_fb":
        return (f"A={_A(u[0]):.2f} shaper={SHAPER_NAMES[_kind(u[1])]} Td1={_td(u[2]):.2f} "
                f"Td2={_td(u[3]):.2f} kp={_kp(u[4]):.2f} kd={_kd(u[5]):.2f}")
    raise ValueError(cls)


DIMS = {"bang": 7, "free": 7 + FREE_K, "pd": 3, "shaped_pd": 6, "ff_fb": 6}


def make_factory(cls: str, u: Sequence[float]) -> Callable[[], Callable[[dict], float]]:
    if cls == "bang":
        A, imps = _A(u[0]), shaper(_kind(u[1]), _td(u[2]), _td(u[3]))
        kp, kd, apd = _kp(u[4]), _kd(u[5]), _amp(u[6])

        def factory():
            mem: Dict = {}

            def policy(s):
                if "seq" not in mem:
                    g = s["goal"]
                    mem["seq"] = bang_sequence(A, abs(g), 1.0 if g > 0 else -1.0, imps)
                r = mem["seq"]
                t = int(s["t"])
                if r is not None and t < len(r[0]):
                    return r[0][t]
                return max(-apd, min(apd, kp * (s["goal"] - s["x"]) - kd * s["v"]))
            return policy
        return factory

    if cls == "free":
        beta = 0.6 + 2.4 * u[0]
        coeffs = [2 * x - 1 for x in u[1:1 + FREE_K]]
        imps = shaper(_kind(u[1 + FREE_K]), _td(u[2 + FREE_K]), _td(u[3 + FREE_K]))
        kp, kd, apd = _kp(u[4 + FREE_K]), _kd(u[5 + FREE_K]), _amp(u[6 + FREE_K])

        def factory():
            mem: Dict = {}

            def policy(s):
                if "seq" not in mem:
                    g = s["goal"]
                    mem["seq"] = free_sequence(beta, coeffs, abs(g), 1.0 if g > 0 else -1.0, imps)
                r = mem["seq"]
                t = int(s["t"])
                if r is not None and t < len(r[0]):
                    return r[0][t]
                return max(-apd, min(apd, kp * (s["goal"] - s["x"]) - kd * s["v"]))
            return policy
        return factory

    if cls == "pd":
        kp, kd, amax = _kp(u[0]), _kd(u[1]), _amp(u[2])

        def policy(s):
            return max(-amax, min(amax, kp * (s["goal"] - s["x"]) - kd * s["v"]))
        return lambda: policy

    if cls == "shaped_pd":
        kp, kd, amax = _kp(u[0]), _kd(u[1]), _amp(u[2])
        imps = [(int(round(t / DT)), w) for t, w in shaper(_kind(u[3]), _td(u[4]), _td(u[5]))]

        def factory():
            hist: List[float] = []

            def policy(s):
                hist.append(max(-amax, min(amax, kp * (s["goal"] - s["x"]) - kd * s["v"])))
                a = 0.0
                for n, w in imps:
                    i = len(hist) - 1 - n
                    if i >= 0:
                        a += w * hist[i]
                return _clip1(a)
            return policy
        return factory

    if cls == "ff_fb":
        A, imps = _A(u[0]), shaper(_kind(u[1]), _td(u[2]), _td(u[3]))
        kp, kd = _kp(u[4]), _kd(u[5])

        def factory():
            mem: Dict = {}

            def policy(s):
                if "seq" not in mem:
                    g = s["goal"]
                    mem["seq"] = bang_sequence(A, abs(g), 1.0 if g > 0 else -1.0, imps)
                r = mem["seq"]
                t = int(s["t"])
                if r is not None and t < len(r[0]):
                    seq, xs, vs = r
                    return _clip1(seq[t] + kp * (xs[t] - s["x"]) + kd * (vs[t] - s["v"]))
                return _clip1(kp * (s["goal"] - s["x"]) - kd * s["v"])
            return policy
        return factory

    raise ValueError(cls)


# ----------------------------------------------------------------------------
# Physically seeded starting points (inverse decoders), so the search does not
# depend on random luck to find textbook designs on either side.
# ----------------------------------------------------------------------------

def _inv_A(A):
    return (A - 0.1) / 0.9


def _inv_kind(k):
    return (k + 0.5) / 7


def _inv_td(T):
    return min(1.0, max(0.0, (T - 1.0) / 5.0))


def _inv_kp(k):
    return min(1.0, max(0.0, (math.log10(k) + 1.3) / 2.0))


def _inv_kd(k):
    return min(1.0, max(0.0, (math.log10(k) + 1.0) / 1.7))


def _inv_amp(a):
    return min(1.0, max(0.0, (a - 0.02) / 0.98))


def seeded_vectors(cls: str, Ls: Sequence[float]) -> List[List[float]]:
    periods = sorted({Tn(L) for L in Ls} | {sum(Tn(L) for L in Ls) / len(Ls)})
    out: List[List[float]] = []
    if cls in ("bang", "ff_fb"):
        for A, kind, T in itertools.product((0.4, 0.7, 1.0), (1, 2, 4), periods):
            v = [_inv_A(A), _inv_kind(kind), _inv_td(T), _inv_td(T), _inv_kp(0.5), _inv_kd(1.0)]
            if cls == "bang":
                v.append(_inv_amp(0.3))
            out.append(v)
        for A, T1, T2 in itertools.product((0.5, 0.8), periods, periods):
            if T1 < T2:
                v = [_inv_A(A), _inv_kind(5), _inv_td(T1), _inv_td(T2), _inv_kp(0.5), _inv_kd(1.0)]
                if cls == "bang":
                    v.append(_inv_amp(0.3))
                out.append(v)
    elif cls == "free":
        for beta, kind, T in itertools.product((1.0, 1.5, 2.2), (0, 1, 2), periods):
            out.append([(beta - 0.6) / 2.4] + [1.0] * FREE_K
                       + [_inv_kind(kind), _inv_td(T), _inv_td(T), _inv_kp(0.5), _inv_kd(1.0), _inv_amp(0.3)])
        for beta in (1.5, 2.2, 3.0):  # smooth ramp up / hold / ramp down
            ramp = [0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25][:FREE_K]
            out.append([(beta - 0.6) / 2.4] + [(c + 1) / 2 for c in ramp]
                       + [_inv_kind(0), 0.5, 0.5, _inv_kp(0.5), _inv_kd(1.0), _inv_amp(0.3)])
    elif cls == "pd":
        for kp, zeta, amax in itertools.product((0.1, 0.3, 1.0), (0.7, 1.0, 1.5), (0.3, 0.6, 1.0)):
            out.append([_inv_kp(kp), _inv_kd(2 * zeta * math.sqrt(kp)), _inv_amp(amax)])
    elif cls == "shaped_pd":
        for kp, kind, T in itertools.product((0.1, 0.3, 1.0), (1, 2, 4), periods):
            out.append([_inv_kp(kp), _inv_kd(2 * math.sqrt(kp)), _inv_amp(0.6), _inv_kind(kind), _inv_td(T), _inv_td(T)])
    return out


# ----------------------------------------------------------------------------
# Evaluation with the hidden (field) observation
# ----------------------------------------------------------------------------

def run_field_episode(policy: Callable[[dict], float], L: float, N: int, seed: int) -> Tuple[bool, float, int]:
    cfg = EnvConfig(slosh_length=L, max_steps=N)
    env = SloshTankEnv(seed=seed, hidden=HiddenParams(actuator_gain=KAPPA), config=cfg)
    s = env.observe()
    total = 0.0
    for _ in range(N):
        try:
            a = policy({k: s[k] for k in FIELD_KEYS})
        except Exception:
            return False, -200.0, env.t
        s, r, done, _, _ = env.step(a)
        total += r
        if done:
            break
    return bool(env.is_success()), total, env.t


def evaluate(factory, pairs: Sequence[Tuple[float, int]], N: int) -> Tuple[float, float]:
    """pairs: (L, seed). Returns (success_rate, mean_reward)."""
    ok = 0
    rew = 0.0
    for L, seed in pairs:
        s, r, _ = run_field_episode(factory(), L, N, seed)
        ok += s
        rew += r
    n = len(pairs)
    return ok / n, rew / n


def objective_value(succ: float, rew: float) -> float:
    # success dominates; reward only breaks ties / gives a signal when all fail
    return succ + 1e-4 * rew


# ----------------------------------------------------------------------------
# Search: random sample + hill-climbing from the top-k, in normalized space
# ----------------------------------------------------------------------------

def optimize(objective, dim: int, rng: random.Random, n_random: int, topk: int, n_climb: int,
             seeds_init: Sequence[Sequence[float]] = ()):
    pop = [list(x) for x in seeds_init] + [[rng.random() for _ in range(dim)] for _ in range(n_random)]
    scored = sorted(((objective(x), x) for x in pop), key=lambda p: -p[0])
    best = scored[0]
    for f, x in scored[:topk]:
        cur_f, cur_x, sigma = f, x, 0.15
        for _ in range(n_climb):
            y = [min(1.0, max(0.0, xi + rng.gauss(0.0, sigma))) for xi in cur_x]
            fy = objective(y)
            if fy >= cur_f:
                cur_f, cur_x = fy, y
            else:
                sigma = max(0.02, sigma * 0.97)
            if cur_f > best[0]:
                best = (cur_f, cur_x)
    return best


def search_pairs(Ls: Sequence[float], continuous: bool) -> Dict[str, List[Tuple[float, int]]]:
    """Blind search pairs: every (L, seed) for discrete sets; for a continuous
    range, one L per seed drawn uniformly (deterministic)."""
    if continuous:
        rng = random.Random(12345)
        lo, hi = min(Ls), max(Ls)
        return {"blind": [(rng.uniform(lo, hi), s) for s in SEARCH_SEEDS]}
    return {"blind": [(L, s) for L in Ls for s in SEARCH_SEEDS]}


def eval_pairs(Ls: Sequence[float], continuous: bool) -> List[Tuple[float, int]]:
    if continuous:
        rng = random.Random(54321)
        lo, hi = min(Ls), max(Ls)
        return [(rng.uniform(lo, hi), s) for s in EVAL_SEEDS]
    return [(L, s) for L in Ls for s in EVAL_SEEDS]


def run_job(job: dict) -> dict:
    """One (config, side, class[, L]) search. Pure function of its inputs."""
    cls, N, Ls = job["cls"], job["N"], job["Ls"]
    key = f"{cls}|{job['side']}|{Ls}|{N}|{job.get('L')}|{job['continuous']}"
    rng = random.Random(key)
    if job["side"] == "blind":
        pairs = search_pairs(Ls, job["continuous"])["blind"]
        seeds_init = seeded_vectors(cls, Ls)
    else:
        pairs = [(job["L"], s) for s in SEARCH_SEEDS]
        seeds_init = seeded_vectors(cls, [job["L"]])

    def objective(u):
        succ, rew = evaluate(make_factory(cls, u), pairs, N)
        return objective_value(succ, rew)

    t0 = time.time()
    f, u = optimize(objective, DIMS[cls], rng, job["n_random"], job["topk"], job["n_climb"], seeds_init)
    return {**job, "search_obj": f, "u": u, "seconds": time.time() - t0}


def run_eval(job: dict) -> dict:
    factory = make_factory(job["cls"], job["u"])
    out = {}
    for L in job["Ls"]:
        succ, rew = evaluate(factory, [(L, s) for s in EVAL_SEEDS], job["N"])
        out[str(L)] = {"success": succ, "reward": rew}
    if job["continuous"]:
        succ, rew = evaluate(factory, eval_pairs(job["Ls"], True), job["N"])
        out["range"] = {"success": succ, "reward": rew}
    return {**job, "eval": out}


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------

def cmd_bench(args):
    fac = make_factory("bang", [0.6, _inv_kind(1), _inv_td(3.5), 0.5, 0.5, 0.5, 0.3])
    n = 400
    t0 = time.time()
    for s in range(n):
        run_field_episode(fac(), 3.0, 105, s)
    dt = time.time() - t0
    print(f"{n / dt:.0f} episodes/s on one core ({dt / n * 1e3:.2f} ms/episode, 105-step budget)")


def parse_configs(spec: List[str]) -> List[dict]:
    """'2,3,4@90' -> discrete set; '2-4@90' -> continuous range with anchors."""
    out = []
    for s in spec:
        ls, n = s.split("@")
        if "-" in ls:
            lo, hi = (float(x) for x in ls.split("-"))
            out.append({"Ls": [lo, (lo + hi) / 2, hi], "N": int(n), "continuous": True})
        else:
            out.append({"Ls": [float(x) for x in ls.split(",")], "N": int(n), "continuous": False})
    return out


DEFAULT_CONFIGS = ["2,3,4@90", "2,3,4@105", "2,3,4@120",
                   "3,4,5@90", "3,4,5@105", "3,4,5@120",
                   "2.5,3.5,4.5@90", "2.5,3.5,4.5@105", "2.5,3.5,4.5@120"]


def cmd_search(args):
    configs = parse_configs(args.configs)
    budget = {"n_random": args.n_random, "topk": args.topk, "n_climb": args.n_climb}
    jobs = []
    for c in configs:
        for cls in CLASSES:
            jobs.append({**c, **budget, "cls": cls, "side": "blind", "L": None})
            for L in c["Ls"]:
                jobs.append({**c, **budget, "cls": cls, "side": "informed", "L": L})
    # longest jobs first for better packing
    jobs.sort(key=lambda j: (j["side"] != "blind", -DIMS[j["cls"]]))
    print(f"{len(jobs)} search jobs on {args.workers} workers "
          f"({len(configs)} configs x {len(CLASSES)} classes x (1 blind + per-L informed))", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
        # pick best class per (config, side[, L]) by SEARCH score, then report on fresh seeds
        picks = {}
        for r in results:
            k = (tuple(r["Ls"]), r["N"], r["continuous"], r["side"], r["L"])
            if k not in picks or r["search_obj"] > picks[k]["search_obj"]:
                picks[k] = r
        evals = list(pool.imap_unordered(run_eval, list(picks.values())))
    print(f"search + eval done in {time.time() - t0:.0f}s\n")

    # ---- report
    rows = []
    for c in configs:
        ck = (tuple(c["Ls"]), c["N"], c["continuous"])
        blind = next(e for e in evals if (tuple(e["Ls"]), e["N"], e["continuous"], e["side"]) == ck + ("blind",))
        inf = {e["L"]: e for e in evals if (tuple(e["Ls"]), e["N"], e["continuous"], e["side"]) == ck + ("informed",)}
        b = [blind["eval"][str(L)]["success"] for L in c["Ls"]]
        i = [inf[L]["eval"][str(L)]["success"] for L in c["Ls"]]
        bm, im = sum(b) / len(b), sum(i) / len(i)
        g1, g2 = min(i) >= GATE_INFORMED_MIN, (im - bm) >= GATE_GAP_MIN
        rows.append({"config": c, "blind": {"cls": blind["cls"], "params": describe(blind["cls"], blind["u"]),
                                            "per_L": b, "mean": bm,
                                            "range": blind["eval"].get("range", {}).get("success")},
                     "informed": {"per_L": i, "mean": im,
                                  "cls": {str(L): inf[L]["cls"] for L in c["Ls"]},
                                  "params": {str(L): describe(inf[L]["cls"], inf[L]["u"]) for L in c["Ls"]}},
                     "gap": im - bm, "G1": g1, "G2": g2})
    pct = lambda xs: " / ".join(f"{100 * x:3.0f}" for x in xs)  # noqa: E731
    print("| L set (m) | steps | blind per L | blind mean | informed per L | informed mean | gap | G1 | G2 | blind winner |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["config"]
        lset = (f"U[{c['Ls'][0]:g},{c['Ls'][-1]:g}]" if c["continuous"] else "{" + ",".join(f"{L:g}" for L in c["Ls"]) + "}")
        print(f"| {lset} | {c['N']} | {pct(r['blind']['per_L'])} | {100 * r['blind']['mean']:.0f} | "
              f"{pct(r['informed']['per_L'])} | {100 * r['informed']['mean']:.0f} | {100 * r['gap']:+.0f} | "
              f"{'pass' if r['G1'] else 'FAIL'} | {'pass' if r['G2'] else 'FAIL'} | {r['blind']['cls']}: {r['blind']['params']} |")
    print("\nInformed winners:")
    for r in rows:
        c = r["config"]
        for L in c["Ls"]:
            print(f"  {c['Ls']}@{c['N']} L={L:g}: {r['informed']['cls'][str(L)]}: {r['informed']['params'][str(L)]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"budget": budget, "search_seeds": SEARCH_SEEDS, "eval_seeds": EVAL_SEEDS,
                       "gate": {"informed_min": GATE_INFORMED_MIN, "gap_min": GATE_GAP_MIN},
                       "rows": rows, "all_search_results": results}, f, indent=1)
        print(f"\nwrote {args.out}")


# ----------------------------------------------------------------------------
# two-turn: what the model has to write, by hand
# ----------------------------------------------------------------------------

TURN2_TEMPLATE = '''
import math
TN = {tn:.4f}      # slosh period measured in the lab (turn 1)
A = {A:.3f}        # move acceleration chosen in the lab (turn 1)
DT = {dt}
KP, KD, APD = 0.5, 1.0, 0.3
_mem = {{}}

def _plan(goal):
    d, sg = abs(goal), (1.0 if goal > 0 else -1.0)
    n1 = max(1, int(round(math.sqrt(d / A) / DT)))
    base = [A] * n1 + [-A] * n1                 # bang-bang
    n_half = int(round(TN / 2 / DT))
    seq = [0.0] * (len(base) + n_half)
    for i, a in enumerate(base):                # ZV shaper: two half-strength copies, half a period apart
        seq[i] += 0.5 * a
        seq[i + n_half] += 0.5 * a
    x = v = 0.0                                 # rescale so the cart lands exactly on the goal
    for a in seq:
        v += DT * a; x += DT * v
    s = sg * d / x
    return [max(-1.0, min(1.0, s * a)) for a in seq]

def policy(state):
    if "seq" not in _mem:
        _mem["seq"] = _plan(state["goal"])
    t = int(state["t"])
    if t < len(_mem["seq"]):
        return _mem["seq"][t]
    a = KP * (state["goal"] - state["x"]) - KD * state["v"]
    return max(-APD, min(APD, a))
'''


def lab_measure_period(env: SloshTankEnv) -> Optional[float]:
    """Turn-1 code, as the model would write it against the LAB env (full
    observation): short pulse, then read the slosh period off theta's zero
    crossings."""
    s, _ = env.reset()
    ts, th = [], []
    for i in range(env.config.max_steps):
        a = 0.5 if i < 4 else 0.0
        s, _, done, _, _ = env.step(a)
        ts.append(s["t"] * env.config.dt)
        th.append(s["theta"])
        if done:
            break
    crossings = []
    for i in range(5, len(th)):
        if th[i - 1] == 0.0 or (th[i - 1] < 0) != (th[i] < 0):
            frac = th[i - 1] / (th[i - 1] - th[i]) if th[i - 1] != th[i] else 0.0
            crossings.append(ts[i - 1] + frac * (ts[i] - ts[i - 1]))
    if len(crossings) < 2:
        return None
    halves = [b - a for a, b in zip(crossings, crossings[1:])]
    return 2.0 * sum(halves) / len(halves)


def lab_pick_A(L: float, N: int, tn_est: float) -> float:
    """Turn-1 tuning on the lab env over lab goals (never the scoring goals)."""
    best = ((-1.0, -1e9), 0.5)
    for A in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ns = {}
        exec(TURN2_TEMPLATE.format(tn=tn_est, A=A, dt=DT), ns)
        hits, rew = 0, 0.0
        for seed in LAB_SEEDS:
            ns["_mem"].clear()
            ok, r, _ = run_field_episode(ns["policy"], L, N, seed)
            hits += ok
            rew += r
        score = (hits, rew)  # most successes, then highest total reward (faster moves)
        if score > best[0]:
            best = (score, A)
    return best[1]


def two_turn_one_L(job: dict) -> dict:
    L, N = job["L"], job["N"]
    lab = SloshTankEnv(seed=LAB_SEEDS[0], hidden=HiddenParams(actuator_gain=KAPPA),
                       config=EnvConfig(slosh_length=L, max_steps=N))
    tn_est = lab_measure_period(lab)
    if tn_est is None:
        return {**job, "tn_true": Tn(L), "tn_est": None, "A": None, "success": 0.0}
    A = lab_pick_A(L, N, tn_est)
    ns = {}
    exec(TURN2_TEMPLATE.format(tn=tn_est, A=A, dt=DT), ns)
    hits = 0
    for seed in EVAL_SEEDS:
        ns["_mem"].clear()  # fresh episode: plan is recomputed from the observed goal
        ok, _, _ = run_field_episode(ns["policy"], L, N, seed)
        hits += ok
    return {**job, "tn_true": Tn(L), "tn_est": tn_est, "A": A, "success": hits / len(EVAL_SEEDS)}


def cmd_two_turn(args):
    jobs = [{"L": L, "N": args.steps} for L in args.L]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(two_turn_one_L, jobs)
    print(f"two-turn hand-written solution, {args.steps} steps, {len(EVAL_SEEDS)} fresh seeds per L")
    print("| L (m) | Tn true (s) | Tn measured in lab (s) | A chosen in lab | success % | G3 |")
    print("|---|---|---|---|---|---|")
    for r in res:
        est = f"{r['tn_est']:.3f}" if r["tn_est"] else "n/a"
        print(f"| {r['L']:g} | {r['tn_true']:.3f} | {est} | {r['A']} | {100 * r['success']:.1f} | "
              f"{'pass' if r['success'] >= GATE_TWO_TURN_MIN else 'FAIL'} |")
    print("\nturn-2 controller source (constants filled in by turn 1):")
    print(TURN2_TEMPLATE.format(tn=res[0]["tn_est"] or 0.0, A=res[0]["A"] or 0.0, dt=DT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bench")
    s = sub.add_parser("search")
    s.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="'2,3,4@90' discrete L set at 90 steps; '2-4@90' continuous range")
    s.add_argument("--n-random", type=int, default=160)
    s.add_argument("--topk", type=int, default=4)
    s.add_argument("--n-climb", type=int, default=40)
    s.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    s.add_argument("--out", default="data/rung0/search.json")
    t = sub.add_parser("two-turn")
    t.add_argument("--L", type=float, nargs="+", required=True)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = p.parse_args()
    {"bench": cmd_bench, "search": cmd_search, "two-turn": cmd_two_turn}[args.cmd](args)


if __name__ == "__main__":
    main()
