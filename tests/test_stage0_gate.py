"""The pieces the Stage 0 gate is built from: the env's one-step preview, the
rider and the two references on it, the controller families, the stratified
limits, and the hand-written probe variants."""
import contextlib
import io
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "seed_tasks"))

from crystallizer_env import CrystallizerEnv, HiddenParams  # noqa: E402
from crystallizer_utils import (  # noqa: E402
    SIGMA_MAX_RANGE, make_rider, ride_action, run_cautious_episode, run_episode, run_oracle_episode,
)
from build_crystallizer_tasks import N_STRATA, draw_sigma_max, stratum_bounds  # noqa: E402
from stage0_gate import FAMILIES, make_controller, physical_params, summarize, third_of  # noqa: E402
from stage0_solution import PROBE_VARIANTS, controller_source, parse_bracket  # noqa: E402


def test_peek_is_exactly_the_sigma_the_step_reports():
    rng = random.Random(0)
    for seed in (0, 7, 10_000):
        env = CrystallizerEnv(seed=seed, hidden=HiddenParams(sigma_max=0.25))
        env.reset()
        for _ in range(40):
            a = rng.uniform(-1.0, 0.3)
            previewed = env.peek(a)
            state, _, done, _, _ = env.step(a)
            assert previewed == state["sigma"]
            if done:
                break


def test_rider_holds_the_reported_sigma_at_or_below_its_level():
    level = 0.12
    env = CrystallizerEnv(seed=5, hidden=HiddenParams(sigma_max=0.25))
    env.reset()
    peak = 0.0
    for _ in range(env.config.max_steps):
        state, _, done, _, _ = env.step(ride_action(env, level))
        peak = max(peak, state["sigma"])
        if done:
            break
    assert env.is_success()
    assert 0.9 * level < peak <= level


def test_oracle_and_cautious_references():
    for smax in (0.08, 0.165, 0.25):
        hidden = HiddenParams(sigma_max=smax)
        for seed in (0, 1, 10_000):
            o = run_oracle_episode(seed, hidden)
            c = run_cautious_episode(seed, hidden)
            assert o["success"] and c["success"]
            assert (o["steps"] == c["steps"]) if smax == SIGMA_MAX_RANGE[0] else (o["steps"] < c["steps"])


def test_every_family_runs_and_keeps_actions_in_range():
    rng = random.Random(1)
    for fam, d in FAMILIES.items():
        p = [rng.random() for _ in range(d)]
        physical_params(fam, p)
        ctl = make_controller(fam, p)
        env = CrystallizerEnv(seed=3, hidden=HiddenParams(sigma_max=0.2))
        state, _ = env.reset()
        for _ in range(20):
            a = ctl(dict(state), env)
            assert -1.0 <= a <= 1.0, (fam, a)
            state, _, done, _, _ = env.step(a)
            if done:
                break
        res = run_episode(make_controller(fam, p), seed=3, hidden=HiddenParams(sigma_max=0.2), policy_sees_env=True)
        assert res["steps"] > 0 and res["failure_reason"] != "policy_exception"


def test_informed_track_family_contains_the_oracle():
    """The track family's knob maps to a level; at the limit it is the oracle."""
    smax = 0.2
    p = [(smax * (1 - 1e-9) - 0.02) / 0.33]
    mine = run_episode(make_controller("track", p), seed=4, hidden=HiddenParams(sigma_max=smax), policy_sees_env=True)
    oracle = run_oracle_episode(4, HiddenParams(sigma_max=smax))
    assert mine["success"] and mine["steps"] == oracle["steps"]


def test_limits_are_stratified_by_thirds_and_deterministic():
    lo_all, hi_all = SIGMA_MAX_RANGE
    for i, seed in enumerate(range(20_000, 20_060)):
        k = i % N_STRATA
        lo, hi = stratum_bounds(k)
        v = draw_sigma_max(seed, k)
        assert lo <= v <= hi and lo_all <= v <= hi_all
        assert v == draw_sigma_max(seed, k)
        assert third_of(v) == ("low", "middle", "high")[k]
    assert lo_all <= draw_sigma_max(5, None) <= hi_all


def test_summarize_counts_outcomes():
    rows = [{"reward": 1.0, "success": True, "fast": True, "norm_time": 1.0, "steps": 10, "failure_reason": None},
            {"reward": -80.0, "success": False, "fast": False, "norm_time": 0.0, "steps": 5, "failure_reason": "shower"}]
    s = summarize(rows)
    assert s["n"] == 2 and s["fast"] == 0.5 and s["outcomes"] == {"success": 1, "shower": 1}


def test_parse_bracket():
    assert parse_bracket("SIGMA_MAX_LOW=0.18100 SIGMA_MAX_HIGH=0.18240\n") == (0.181, 0.1824)
    assert parse_bracket("Traceback ...") is None


def test_probe_variants_bracket_the_limit_in_process():
    """Each variant, run on the real env object (no sandbox), prints a bracket that
    contains the limit; the creep variants may leave the top end loose."""
    for smax in (0.1, 0.2):
        for name, src in PROBE_VARIANTS.items():
            env = CrystallizerEnv(seed=11, hidden=HiddenParams(sigma_max=smax))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(src, {"env": env})
            br = parse_bracket(buf.getvalue())
            assert br is not None, (name, buf.getvalue())
            lo, hi = br
            assert lo <= smax <= hi, (name, smax, br)


def test_controller_from_the_bracket_rides_and_succeeds():
    ns = {}
    exec(controller_source(0.2), ns)
    res = run_episode(ns["policy"], seed=12, hidden=HiddenParams(sigma_max=0.2))
    assert res["success"] and res["failure_reason"] is None
