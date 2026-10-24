#!/usr/bin/env python3
"""Rung 0 gate: does the slosh task NEED the turn-1 lab?

The question, made testable with hand-written controllers and no model:
is there information that the scored `policy(state)` (written in turn 2)
cannot get in-episode, that the turn-1 lab can measure, and that changes
success by a large margin? Two sides search the SAME controller classes with
the SAME budget on the SAME search seeds and are reported on fresh seeds:

  blind     one controller for every hidden value in the set (robust objective)
  informed  one controller per hidden value (upper bound on what a perfect
            turn-1 measurement could buy)

Variants (all evaluated against the unmodified SloshTankEnv, or a thin
subclass for the fixed-horizon rule):
  --obs full   scored controller sees (t, x, v, goal, steps_left); theta/omega hidden
  --obs dead   scored controller sees (t, goal, steps_left) only: dead reckoning
  --fixed-horizon   success is judged at the final step; the episode does not
                    end at the first success (closes the sequential-hedging
                    loophole for scale-like hidden parameters)
  hidden parameters: actuator gain kappa and/or slosh length L, discrete sets
  or continuous ranges, e.g. "k=0.65,0.8,1.0;L=2,3,4@105" or "k=0.6-1.0;L=0.7@105"

Pre-registered gate (per configuration):
  G1  informed >= 90% success at every hidden value
  G2  informed_mean - blind_mean >= 40 percentage points
  G3  (two-turn subcommand) a hand-written turn-1 measurement + turn-2
      controller reaches >= 95% on fresh seeds

Usage:
  python scripts/rung0_gate.py bench
  python scripts/rung0_gate.py search --obs full --configs "2,3,4@90" ...
  python scripts/rung0_gate.py search --obs dead --fixed-horizon --configs "k=0.65,0.8,1.0;L=0.7@105"
  python scripts/rung0_gate.py two-turn --obs dead --fixed-horizon --config "k=0.65,0.8,1.0;L=0.7@105"
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
OBS_KEYS = {"full": ("t", "x", "v", "goal", "steps_left"), "dead": ("t", "goal", "steps_left")}
FREE_K = 8  # segments per half in the free open-loop profile

SEARCH_SEEDS = list(range(48))
EVAL_SEEDS = list(range(1000, 1200))
LAB_SEEDS = list(range(5000, 5016))

GATE_INFORMED_MIN = 0.90
GATE_GAP_MIN = 0.40
GATE_TWO_TURN_MIN = 0.95

Hidden = Tuple[float, float]  # (kappa, L)

# Per-process settings; set by the main process and by every worker from the job dict.
SETTINGS = {"obs": "full", "fixed_horizon": False}


def apply_settings(d: dict):
    SETTINGS["obs"] = d["obs"]
    SETTINGS["fixed_horizon"] = d["fixed_horizon"]


def field_keys():
    return OBS_KEYS[SETTINGS["obs"]]


def has_feedback():
    return "x" in field_keys()


def Tn(L: float) -> float:
    return 2.0 * math.pi * math.sqrt(L / G)


class FixedHorizonSloshEnv(SloshTankEnv):
    """Same dynamics and reward, but success is judged at the final step only:
    the episode does not end at the first success, and the success bonus is
    paid only if the final state is a success."""

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        c = self.config
        if self.done and not self.is_unsafe() and self.t < c.max_steps:
            reward -= c.success_bonus  # undo the early success: keep going
            self.done = False
            done = False
        return obs, reward, done, trunc, info


def make_env(hidden: Hidden, N: int, seed: int):
    kappa, L = hidden
    cls = FixedHorizonSloshEnv if SETTINGS["fixed_horizon"] else SloshTankEnv
    return cls(seed=seed, hidden=HiddenParams(actuator_gain=kappa), config=EnvConfig(slosh_length=L, max_steps=N))


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

    table = [[(0.0, 1.0)], zv(td1), zvd(td1), zvdd(td1), ei(td1), conv(zv(td1), zv(td2)), conv(zvd(td1), zv(td2))]
    return table[kind]


SHAPER_NAMES = ["none", "ZV", "ZVD", "ZVDD", "EI", "ZVxZV", "ZVDxZV"]


# ----------------------------------------------------------------------------
# Open-loop move sequences (discrete time; exact landing under the assumed
# gain khat when unclipped)
# ----------------------------------------------------------------------------

def _apply_shaper(seq: List[float], imps) -> List[float]:
    out = [0.0] * (len(seq) + max(int(round(t / DT)) for t, _ in imps))
    for tau, w in imps:
        n = int(round(tau / DT))
        for i, a in enumerate(seq):
            out[i + n] += w * a
    return out


def _simulate_cart(seq: Sequence[float], khat: float) -> Tuple[float, float, List[float], List[float]]:
    """Cart recursion exactly as in SloshTankEnv.step, with gain khat."""
    x = v = 0.0
    xs, vs = [], []
    for a in seq:
        v += DT * khat * a
        x += DT * v
        xs.append(x)
        vs.append(v)
    return x, v, xs, vs


def _finish_sequence(unscaled: List[float], d: float, sg: float, khat: float):
    """Scale an antisymmetric sequence so the khat-model lands exactly at
    signed distance sg*d (zero final velocity by antisymmetry). Returns
    (clipped actions, reference x, reference v) or None if degenerate."""
    disp, _, _, _ = _simulate_cart(unscaled, khat)
    if disp <= 1e-6:
        return None
    s = sg * d / disp
    seq = [max(-1.0, min(1.0, s * a)) for a in unscaled]
    _, _, xs, vs = _simulate_cart(seq, khat)
    return seq, xs, vs


def bang_sequence(A: float, d: float, sg: float, imps, khat: float):
    n1 = max(1, int(round(math.sqrt(d / A) / DT)))
    return _finish_sequence(_apply_shaper([A] * n1 + [-A] * n1, imps), d, sg, khat)


def free_sequence(beta: float, coeffs: Sequence[float], d: float, sg: float, imps, khat: float):
    H = beta * math.sqrt(d)
    nseg = max(1, int(round(H / len(coeffs) / DT)))
    full = list(coeffs) + [-c for c in reversed(coeffs)]
    base = []
    for c in full:
        base.extend([c] * nseg)
    return _finish_sequence(_apply_shaper(base, imps), d, sg, khat)


def multistop_sequence(A: float, stops: Sequence[float], dwell: int, imps, khat: float):
    """Back-to-back shaped bang-bang moves through planned signed positions
    `stops` (under the khat model), with `dwell` idle steps between moves.
    Under early termination this lets a blind plan test several gain
    hypotheses in sequence; under a fixed horizon only the last stop counts."""
    seq: List[float] = []
    pos = 0.0
    for p in stops:
        d = abs(p - pos)
        if d < 1e-6:
            continue
        r = bang_sequence(A, d, 1.0 if p > pos else -1.0, imps, khat)
        if r is None:
            return None
        seq.extend(r[0])
        seq.extend([0.0] * dwell)
        pos = p
    return seq


# ----------------------------------------------------------------------------
# Controller classes. Each decoder maps a normalized vector u in [0,1]^n to a
# *factory*: factory() returns a fresh policy for one episode (stateful
# classes get clean memory). Policies receive only the field keys.
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


def _khat(u):
    return 0.5 + 0.7 * u  # 0.5 .. 1.2


def _clip1(a):
    return max(-1.0, min(1.0, a))


FEEDBACK_CLASSES = ("pd", "shaped_pd", "ff_fb")
OPEN_LOOP_CLASSES = ("bang", "free", "multistop")
DIMS = {"bang": 8, "free": 8 + FREE_K, "multistop": 9, "pd": 3, "shaped_pd": 6, "ff_fb": 7}


def active_classes():
    return OPEN_LOOP_CLASSES + (FEEDBACK_CLASSES if has_feedback() else ())


def describe(cls: str, u: Sequence[float]) -> str:
    if cls == "bang":
        return (f"khat={_khat(u[0]):.2f} A={_A(u[1]):.2f} shaper={SHAPER_NAMES[_kind(u[2])]} Td1={_td(u[3]):.2f} "
                f"Td2={_td(u[4]):.2f} kp={_kp(u[5]):.2f} kd={_kd(u[6]):.2f} apd={_amp(u[7]):.2f}")
    if cls == "free":
        c = ",".join(f"{2 * x - 1:+.2f}" for x in u[2:2 + FREE_K])
        k = 2 + FREE_K
        return (f"khat={_khat(u[0]):.2f} beta={0.6 + 2.4 * u[1]:.2f} c=[{c}] shaper={SHAPER_NAMES[_kind(u[k])]} "
                f"Td1={_td(u[k + 1]):.2f} Td2={_td(u[k + 2]):.2f} kp={_kp(u[k + 3]):.2f} "
                f"kd={_kd(u[k + 4]):.2f} apd={_amp(u[k + 5]):.2f}")
    if cls == "multistop":
        n = 1 + min(2, int(u[5] * 3))
        return (f"khat={_khat(u[0]):.2f} A={_A(u[1]):.2f} shaper={SHAPER_NAMES[_kind(u[2])]} Td1={_td(u[3]):.2f} "
                f"Td2={_td(u[4]):.2f} stops={n} r2={1.0 + 0.6 * u[6]:.2f} r3={1.0 + 0.6 * u[7]:.2f} "
                f"dwell={int(u[8] * 11)}")
    if cls == "pd":
        return f"kp={_kp(u[0]):.2f} kd={_kd(u[1]):.2f} amax={_amp(u[2]):.2f}"
    if cls == "shaped_pd":
        return (f"kp={_kp(u[0]):.2f} kd={_kd(u[1]):.2f} amax={_amp(u[2]):.2f} "
                f"shaper={SHAPER_NAMES[_kind(u[3])]} Td1={_td(u[4]):.2f} Td2={_td(u[5]):.2f}")
    if cls == "ff_fb":
        return (f"khat={_khat(u[0]):.2f} A={_A(u[1]):.2f} shaper={SHAPER_NAMES[_kind(u[2])]} Td1={_td(u[3]):.2f} "
                f"Td2={_td(u[4]):.2f} kp={_kp(u[5]):.2f} kd={_kd(u[6]):.2f}")
    raise ValueError(cls)


def _terminal(kp, kd, apd):
    """What an open-loop class does after its sequence ends."""
    if has_feedback():
        return lambda s: max(-apd, min(apd, kp * (s["goal"] - s["x"]) - kd * s["v"]))
    return lambda s: 0.0


def _sequence_policy(plan: Callable[[float], Optional[list]], after: Callable[[dict], float]):
    def factory():
        mem: Dict = {}

        def policy(s):
            if "seq" not in mem:
                mem["seq"] = plan(s["goal"])
            seq = mem["seq"]
            t = int(s["t"])
            if seq is not None and t < len(seq):
                return seq[t]
            return after(s)
        return policy
    return factory


def make_factory(cls: str, u: Sequence[float]) -> Callable[[], Callable[[dict], float]]:
    if cls == "bang":
        khat, A, imps = _khat(u[0]), _A(u[1]), shaper(_kind(u[2]), _td(u[3]), _td(u[4]))
        after = _terminal(_kp(u[5]), _kd(u[6]), _amp(u[7]))

        def plan(g):
            r = bang_sequence(A, abs(g), 1.0 if g > 0 else -1.0, imps, khat)
            return None if r is None else r[0]
        return _sequence_policy(plan, after)

    if cls == "free":
        khat, beta = _khat(u[0]), 0.6 + 2.4 * u[1]
        coeffs = [2 * x - 1 for x in u[2:2 + FREE_K]]
        k = 2 + FREE_K
        imps = shaper(_kind(u[k]), _td(u[k + 1]), _td(u[k + 2]))
        after = _terminal(_kp(u[k + 3]), _kd(u[k + 4]), _amp(u[k + 5]))

        def plan(g):
            r = free_sequence(beta, coeffs, abs(g), 1.0 if g > 0 else -1.0, imps, khat)
            return None if r is None else r[0]
        return _sequence_policy(plan, after)

    if cls == "multistop":
        khat, A, imps = _khat(u[0]), _A(u[1]), shaper(_kind(u[2]), _td(u[3]), _td(u[4]))
        n = 1 + min(2, int(u[5] * 3))
        ratios = sorted([1.0, 1.0 + 0.6 * u[6], 1.0 + 0.6 * u[7]])[:n]
        dwell = int(u[8] * 11)

        def plan(g):
            return multistop_sequence(A, [r * g for r in ratios], dwell, imps, khat)
        return _sequence_policy(plan, lambda s: 0.0)

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
        khat, A, imps = _khat(u[0]), _A(u[1]), shaper(_kind(u[2]), _td(u[3]), _td(u[4]))
        kp, kd = _kp(u[5]), _kd(u[6])

        def factory():
            mem: Dict = {}

            def policy(s):
                if "seq" not in mem:
                    g = s["goal"]
                    mem["seq"] = bang_sequence(A, abs(g), 1.0 if g > 0 else -1.0, imps, khat)
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
# Physically seeded starting points (inverse decoders), so neither side
# depends on random luck to find textbook designs.
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


def _inv_khat(k):
    return min(1.0, max(0.0, (k - 0.5) / 0.7))


def seeded_vectors(cls: str, hiddens: Sequence[Hidden]) -> List[List[float]]:
    kappas = sorted({k for k, _ in hiddens} | {sum(k for k, _ in hiddens) / len(hiddens)})
    periods = sorted({Tn(L) for _, L in hiddens} | {sum(Tn(L) for _, L in hiddens) / len(hiddens)})
    out: List[List[float]] = []
    if cls in ("bang", "ff_fb"):
        for kh, A, kind, T in itertools.product(kappas, (0.4, 0.7, 1.0), (1, 2, 4), periods):
            v = [_inv_khat(kh), _inv_A(A), _inv_kind(kind), _inv_td(T), _inv_td(T), _inv_kp(0.5), _inv_kd(1.0)]
            if cls == "bang":
                v.append(_inv_amp(0.3))
            out.append(v)
        for kh, A, T1, T2 in itertools.product(kappas, (0.5, 0.8), periods, periods):
            if T1 < T2:
                v = [_inv_khat(kh), _inv_A(A), _inv_kind(5), _inv_td(T1), _inv_td(T2), _inv_kp(0.5), _inv_kd(1.0)]
                if cls == "bang":
                    v.append(_inv_amp(0.3))
                out.append(v)
    elif cls == "free":
        tail = lambda kind, T: [_inv_kind(kind), _inv_td(T), _inv_td(T), _inv_kp(0.5), _inv_kd(1.0), _inv_amp(0.3)]  # noqa: E731
        for kh, beta, kind, T in itertools.product(kappas, (1.0, 1.5, 2.2), (0, 1, 2), periods):
            out.append([_inv_khat(kh), (beta - 0.6) / 2.4] + [1.0] * FREE_K + tail(kind, T))
        ramp = [0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25][:FREE_K]
        for kh, beta in itertools.product(kappas, (1.5, 2.2, 3.0)):
            out.append([_inv_khat(kh), (beta - 0.6) / 2.4] + [(c + 1) / 2 for c in ramp] + tail(0, periods[0]))
    elif cls == "multistop":
        ks = sorted({k for k, _ in hiddens})
        base = max(ks)
        # stops at d/k for each hypothesis, planned under khat = max kappa
        ratios = sorted(base / k for k in ks)
        r2 = ratios[1] if len(ratios) > 1 else 1.3
        r3 = ratios[2] if len(ratios) > 2 else 1.6
        for A, kind, T, dwell in itertools.product((0.5, 0.8, 1.0), (1, 2), periods, (0, 5)):
            out.append([_inv_khat(base), _inv_A(A), _inv_kind(kind), _inv_td(T), _inv_td(T),
                        0.9, (r2 - 1.0) / 0.6, (r3 - 1.0) / 0.6, dwell / 11])
    elif cls == "pd":
        for kp, zeta, amax in itertools.product((0.1, 0.3, 1.0), (0.7, 1.0, 1.5), (0.3, 0.6, 1.0)):
            out.append([_inv_kp(kp), _inv_kd(2 * zeta * math.sqrt(kp)), _inv_amp(amax)])
    elif cls == "shaped_pd":
        for kp, kind, T in itertools.product((0.1, 0.3, 1.0), (1, 2, 4), periods):
            out.append([_inv_kp(kp), _inv_kd(2 * math.sqrt(kp)), _inv_amp(0.6), _inv_kind(kind), _inv_td(T), _inv_td(T)])
    return [[min(1.0, max(0.0, x)) for x in v] for v in out]


# ----------------------------------------------------------------------------
# Evaluation with the hidden (field) observation
# ----------------------------------------------------------------------------

def run_field_episode(policy: Callable[[dict], float], hidden: Hidden, N: int, seed: int) -> Tuple[bool, float, int]:
    env = make_env(hidden, N, seed)
    keys = field_keys()
    s = env.observe()
    total = 0.0
    for _ in range(N):
        try:
            a = policy({k: s[k] for k in keys})
        except Exception:
            return False, -200.0, env.t
        s, r, done, _, _ = env.step(a)
        total += r
        if done:
            break
    return bool(env.is_success()), total, env.t


def evaluate(factory, pairs: Sequence[Tuple[Hidden, int]], N: int) -> Tuple[float, float]:
    ok = 0
    rew = 0.0
    for hidden, seed in pairs:
        s, r, _ = run_field_episode(factory(), hidden, N, seed)
        ok += s
        rew += r
    return ok / len(pairs), rew / len(pairs)


def objective_value(succ: float, rew: float) -> float:
    return succ + 1e-4 * rew  # success dominates; reward breaks ties / guides when all fail


# ----------------------------------------------------------------------------
# Search: random sample + hill-climbing from the top-k, in normalized space
# ----------------------------------------------------------------------------

def optimize(objective, dim, rng, n_random, topk, n_climb, seeds_init=()):
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


# ----------------------------------------------------------------------------
# Configurations: "k=0.65,0.8,1.0;L=2,3,4@105", "k=0.6-1.0;L=0.7@105", legacy "2,3,4@90"
# ----------------------------------------------------------------------------

def parse_config(spec: str) -> dict:
    left, n = spec.split("@")
    dims = {"k": ([1.0], False), "L": ([EnvConfig().slosh_length], False)}
    for part in left.split(";"):
        if "=" in part:
            name, vals = part.split("=")
        else:
            name, vals = "L", part  # legacy: L list only
        if "-" in vals:
            lo, hi = (float(x) for x in vals.split("-"))
            dims[name] = ([lo, (lo + hi) / 2, hi], True)
        else:
            dims[name] = ([float(x) for x in vals.split(",")], False)
    (ks, kc), (Ls, Lc) = dims["k"], dims["L"]
    return {"spec": spec, "kappas": ks, "Ls": Ls, "k_cont": kc, "L_cont": Lc, "N": int(n),
            "hiddens": [(k, L) for k in ks for L in Ls]}


def _draw(rng, vals, cont):
    return rng.uniform(min(vals), max(vals)) if cont else rng.choice(vals)


def blind_search_pairs(c: dict) -> List[Tuple[Hidden, int]]:
    if c["k_cont"] or c["L_cont"]:
        rng = random.Random(12345)
        return [((_draw(rng, c["kappas"], c["k_cont"]), _draw(rng, c["Ls"], c["L_cont"])), s) for s in SEARCH_SEEDS]
    return [(h, s) for h in c["hiddens"] for s in SEARCH_SEEDS]


def range_eval_pairs(c: dict) -> List[Tuple[Hidden, int]]:
    rng = random.Random(54321)
    return [((_draw(rng, c["kappas"], c["k_cont"]), _draw(rng, c["Ls"], c["L_cont"])), s) for s in EVAL_SEEDS]


def hidden_label(h: Hidden, c: dict) -> str:
    parts = []
    if len(c["kappas"]) > 1:
        parts.append(f"k={h[0]:g}")
    if len(c["Ls"]) > 1:
        parts.append(f"L={h[1]:g}")
    return ",".join(parts) or f"k={h[0]:g},L={h[1]:g}"


def run_job(job: dict) -> dict:
    apply_settings(job)
    c, cls = job["config"], job["cls"]
    rng = random.Random(f"{cls}|{job['side']}|{c['spec']}|{job.get('hidden')}|{job['obs']}|{job['fixed_horizon']}")
    if job["side"] == "blind":
        pairs = blind_search_pairs(c)
        seeds_init = seeded_vectors(cls, c["hiddens"])
    else:
        pairs = [(tuple(job["hidden"]), s) for s in SEARCH_SEEDS]
        seeds_init = seeded_vectors(cls, [tuple(job["hidden"])])

    def objective(u):
        succ, rew = evaluate(make_factory(cls, u), pairs, c["N"])
        return objective_value(succ, rew)

    t0 = time.time()
    f, u = optimize(objective, DIMS[cls], rng, job["n_random"], job["topk"], job["n_climb"], seeds_init)
    return {**job, "search_obj": f, "u": u, "seconds": time.time() - t0}


def run_eval(job: dict) -> dict:
    apply_settings(job)
    c = job["config"]
    factory = make_factory(job["cls"], job["u"])
    out = {}
    for h in c["hiddens"]:
        succ, rew = evaluate(factory, [(h, s) for s in EVAL_SEEDS], c["N"])
        out[hidden_label(h, c)] = {"success": succ, "reward": rew}
    if c["k_cont"] or c["L_cont"]:
        succ, rew = evaluate(factory, range_eval_pairs(c), c["N"])
        out["range"] = {"success": succ, "reward": rew}
    return {**job, "eval": out}


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------

def cmd_bench(args):
    apply_settings({"obs": args.obs, "fixed_horizon": args.fixed_horizon})
    fac = make_factory("bang", [_inv_khat(1.0), 0.6, _inv_kind(1), _inv_td(3.5), 0.5, 0.5, 0.5, 0.3])
    n = 400
    t0 = time.time()
    for s in range(n):
        run_field_episode(fac(), (1.0, 3.0), 105, s)
    dt = time.time() - t0
    print(f"{n / dt:.0f} episodes/s on one core ({dt / n * 1e3:.2f} ms/episode, 105-step budget)")


def cmd_search(args):
    apply_settings({"obs": args.obs, "fixed_horizon": args.fixed_horizon})
    configs = [parse_config(s) for s in args.configs]
    budget = {"n_random": args.n_random, "topk": args.topk, "n_climb": args.n_climb}
    common = {**budget, "obs": args.obs, "fixed_horizon": args.fixed_horizon}
    classes = active_classes()
    jobs = []
    for c in configs:
        for cls in classes:
            jobs.append({**common, "config": c, "cls": cls, "side": "blind", "hidden": None})
            for h in c["hiddens"]:
                jobs.append({**common, "config": c, "cls": cls, "side": "informed", "hidden": list(h)})
    jobs.sort(key=lambda j: (j["side"] != "blind", -DIMS[j["cls"]]))
    print(f"obs={args.obs} fixed_horizon={args.fixed_horizon} classes={list(classes)}")
    print(f"{len(jobs)} search jobs on {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            results.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
        picks = {}
        for r in results:  # best class per (config, side, hidden) by SEARCH score
            k = (r["config"]["spec"], r["side"], str(r["hidden"]))
            if k not in picks or r["search_obj"] > picks[k]["search_obj"]:
                picks[k] = r
        evals = list(pool.imap_unordered(run_eval, list(picks.values())))
    print(f"search + eval done in {time.time() - t0:.0f}s\n")

    rows = []
    for c in configs:
        blind = next(e for e in evals if e["config"]["spec"] == c["spec"] and e["side"] == "blind")
        inf = {str(list(h)): e for e in evals if e["config"]["spec"] == c["spec"] and e["side"] == "informed"
               for h in [tuple(e["hidden"])]}
        labels = [hidden_label(h, c) for h in c["hiddens"]]
        b = [blind["eval"][lab]["success"] for lab in labels]
        i = [inf[str(list(h))]["eval"][lab]["success"] for h, lab in zip(c["hiddens"], labels)]
        bm, im = sum(b) / len(b), sum(i) / len(i)
        rows.append({"config": c, "labels": labels,
                     "blind": {"cls": blind["cls"], "params": describe(blind["cls"], blind["u"]), "per_hidden": b,
                               "mean": bm, "range": blind["eval"].get("range", {}).get("success")},
                     "informed": {"per_hidden": i, "mean": im,
                                  "cls": [inf[str(list(h))]["cls"] for h in c["hiddens"]],
                                  "params": [describe(inf[str(list(h))]["cls"], inf[str(list(h))]["u"]) for h in c["hiddens"]]},
                     "gap": im - bm, "G1": min(i) >= GATE_INFORMED_MIN, "G2": (im - bm) >= GATE_GAP_MIN})
    pct = lambda xs: " / ".join(f"{100 * x:3.0f}" for x in xs)  # noqa: E731
    print("| hidden set | steps | blind per value | blind mean | informed per value | informed mean | gap | G1 | G2 | blind winner |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["config"]
        rng_note = f" (range {100 * r['blind']['range']:.0f})" if r["blind"]["range"] is not None else ""
        print(f"| {c['spec'].split('@')[0]} | {c['N']} | {pct(r['blind']['per_hidden'])}{rng_note} | {100 * r['blind']['mean']:.0f} | "
              f"{pct(r['informed']['per_hidden'])} | {100 * r['informed']['mean']:.0f} | {100 * r['gap']:+.0f} | "
              f"{'pass' if r['G1'] else 'FAIL'} | {'pass' if r['G2'] else 'FAIL'} | {r['blind']['cls']}: {r['blind']['params']} |")
    print("\nvalue order per row:", {r["config"]["spec"]: r["labels"] for r in rows})
    print("\nInformed winners:")
    for r in rows:
        for lab, cls, par in zip(r["labels"], r["informed"]["cls"], r["informed"]["params"]):
            print(f"  {r['config']['spec']} {lab}: {cls}: {par}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"settings": {"obs": args.obs, "fixed_horizon": args.fixed_horizon}, "budget": budget,
                       "search_seeds": SEARCH_SEEDS, "eval_seeds": EVAL_SEEDS,
                       "gate": {"informed_min": GATE_INFORMED_MIN, "gap_min": GATE_GAP_MIN},
                       "rows": rows, "all_search_results": results}, f, indent=1)
        print(f"\nwrote {args.out}")


# ----------------------------------------------------------------------------
# two-turn: what the model has to write, by hand
# ----------------------------------------------------------------------------

TURN2_TEMPLATE = '''
import math
KAPPA = {kappa:.4f}   # actuator gain measured in the lab (turn 1)
TN = {tn:.4f}         # slosh period measured in the lab (turn 1)
A = {A:.3f}           # move acceleration chosen in the lab (turn 1)
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
        v += DT * KAPPA * a; x += DT * v
    s = sg * d / x
    return [max(-1.0, min(1.0, s * a)) for a in seq]

def policy(state):
    if "seq" not in _mem:
        _mem["seq"] = _plan(state["goal"])
    t = int(state["t"])
    if t < len(_mem["seq"]):
        return _mem["seq"][t]
    if "x" not in state:                        # no position feedback: hold still
        return 0.0
    a = KP * (state["goal"] - state["x"]) - KD * state["v"]
    return max(-APD, min(APD, a))
'''


def lab_measure(env) -> Tuple[Optional[float], Optional[float]]:
    """Turn-1 code, as the model would write it against the LAB env (full
    observation): one full-throttle step gives the gain from the velocity it
    produces; a short pulse then gives the slosh period from theta's zero
    crossings."""
    env.reset()
    s, _, _, _, _ = env.step(1.0)
    kappa = s["v"] / (1.0 * env.config.dt)
    env.reset()
    ts, th = [], []
    for i in range(env.config.max_steps):
        s, _, done, _, _ = env.step(0.5 if i < 4 else 0.0)
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
        return kappa, None
    halves = [b - a for a, b in zip(crossings, crossings[1:])]
    return kappa, 2.0 * sum(halves) / len(halves)


def lab_pick_A(hidden: Hidden, N: int, kappa_est: float, tn_est: float) -> float:
    """Turn-1 tuning on the lab env over lab goals (never the scoring goals):
    most successes, then highest total reward (faster moves)."""
    best = ((-1.0, -1e9), 0.5)
    for A in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ns = {}
        exec(TURN2_TEMPLATE.format(kappa=kappa_est, tn=tn_est, A=A, dt=DT), ns)
        hits, rew = 0, 0.0
        for seed in LAB_SEEDS:
            ns["_mem"].clear()
            ok, r, _ = run_field_episode(ns["policy"], hidden, N, seed)
            hits += ok
            rew += r
        if (hits, rew) > best[0]:
            best = ((hits, rew), A)
    return best[1]


def two_turn_one(job: dict) -> dict:
    apply_settings(job)
    hidden, N = tuple(job["hidden"]), job["N"]
    lab = make_env(hidden, N, LAB_SEEDS[0])
    kappa_est, tn_est = lab_measure(lab)
    if tn_est is None:
        return {**job, "kappa_est": kappa_est, "tn_true": Tn(hidden[1]), "tn_est": None, "A": None, "success": 0.0}
    A = lab_pick_A(hidden, N, kappa_est, tn_est)
    ns = {}
    exec(TURN2_TEMPLATE.format(kappa=kappa_est, tn=tn_est, A=A, dt=DT), ns)
    hits = 0
    for seed in EVAL_SEEDS:
        ns["_mem"].clear()
        ok, _, _ = run_field_episode(ns["policy"], hidden, N, seed)
        hits += ok
    return {**job, "kappa_est": kappa_est, "tn_true": Tn(hidden[1]), "tn_est": tn_est, "A": A,
            "success": hits / len(EVAL_SEEDS)}


def cmd_two_turn(args):
    apply_settings({"obs": args.obs, "fixed_horizon": args.fixed_horizon})
    c = parse_config(args.config)
    jobs = [{"hidden": list(h), "N": c["N"], "obs": args.obs, "fixed_horizon": args.fixed_horizon} for h in c["hiddens"]]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(two_turn_one, jobs)
    print(f"two-turn hand-written solution: {args.config}, obs={args.obs}, fixed_horizon={args.fixed_horizon}, "
          f"{len(EVAL_SEEDS)} fresh seeds per value")
    print("| hidden (k, L) | kappa measured | Tn true (s) | Tn measured (s) | A chosen | success % | G3 |")
    print("|---|---|---|---|---|---|---|")
    for r in res:
        est = f"{r['tn_est']:.3f}" if r["tn_est"] else "n/a"
        print(f"| ({r['hidden'][0]:g}, {r['hidden'][1]:g}) | {r['kappa_est']:.4f} | {r['tn_true']:.3f} | {est} | {r['A']} | "
              f"{100 * r['success']:.1f} | {'pass' if r['success'] >= GATE_TWO_TURN_MIN else 'FAIL'} |")
    if args.show_source:
        print("\nturn-2 controller source (constants filled in by turn 1):")
        print(TURN2_TEMPLATE.format(kappa=res[0]["kappa_est"], tn=res[0]["tn_est"] or 0.0, A=res[0]["A"] or 0.0, dt=DT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs", choices=list(OBS_KEYS), default="full",
                   help="what the scored controller observes (default: full = t,x,v,goal,steps_left)")
    p.add_argument("--fixed-horizon", action="store_true", help="judge success at the final step only")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bench")
    s = sub.add_parser("search")
    s.add_argument("--configs", nargs="+", required=True,
                   help='e.g. "k=0.65,0.8,1.0;L=0.7@105", "k=0.6-1.0;L=2-4@105", legacy "2,3,4@90"')
    s.add_argument("--n-random", type=int, default=800)
    s.add_argument("--topk", type=int, default=8)
    s.add_argument("--n-climb", type=int, default=150)
    s.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    s.add_argument("--out", default="data/rung0/search.json")
    t = sub.add_parser("two-turn")
    t.add_argument("--config", required=True)
    t.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    t.add_argument("--show-source", action="store_true")
    args = p.parse_args()
    {"bench": cmd_bench, "search": cmd_search, "two-turn": cmd_two_turn}[args.cmd](args)


if __name__ == "__main__":
    main()
