"""Contract tests for the crystallizer env: Gymnasium-shaped reset/step, the
hidden limit leaves no trace in the observation, and the two reference
behaviours (ride the limit, cross the limit) do what the gate relies on."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "seed_tasks"))

from crystallizer_env import CrystallizerEnv, EnvConfig, HiddenParams, solubility  # noqa: E402
from crystallizer_utils import (  # noqa: E402
    make_rider, max_rate_probe_policy, ride_action, run_cautious_episode, run_episode, run_oracle_episode,
)

KEYS = ("t", "T", "c", "L", "T_target", "steps_left", "sigma")


def test_reset_returns_state_and_info_with_documented_keys():
    env = CrystallizerEnv(seed=0, hidden=HiddenParams(sigma_max=0.15))
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2
    state, info = out
    assert isinstance(state, dict) and isinstance(info, dict)
    assert set(state) == set(KEYS)
    assert abs(state["c"] - solubility(state["T"], env.config)) < 1e-12  # saturated start
    assert abs(state["sigma"]) < 1e-12
    state, _, _, _, _ = env.step(-1.0)
    assert abs(state["sigma"] - (state["c"] / solubility(state["T"], env.config) - 1.0)) < 1e-12


def test_step_returns_5_tuple_and_hidden_is_required():
    env = CrystallizerEnv(seed=0, hidden=HiddenParams(sigma_max=0.15))
    out = env.step(-0.5)
    assert isinstance(out, tuple) and len(out) == 5
    try:
        CrystallizerEnv(seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("constructing without hidden must raise")


def test_hidden_limit_leaves_no_trace_until_crossed():
    """Two batches with different limits, same actions: identical observations
    while neither has showered. This is the property the whole env rests on."""
    a = CrystallizerEnv(seed=3, hidden=HiddenParams(sigma_max=0.10))
    b = CrystallizerEnv(seed=3, hidden=HiddenParams(sigma_max=0.25))
    sa, sb = a.observe(), b.observe()
    for _ in range(a.config.max_steps):
        act = ride_action(a, 0.09)   # the same action for both, chosen on batch a
        sa, _, da, _, _ = a.step(act)
        sb, _, db, _, _ = b.step(act)
        assert sa == sb
        if da or db:
            break
    assert a.is_success() and b.is_success()


def test_riding_the_limit_succeeds_and_crossing_it_showers():
    for smax in (0.08, 0.165, 0.25):
        hidden = HiddenParams(sigma_max=smax)
        ok = run_episode(make_rider(smax * (1 - 1e-9)), seed=1, hidden=hidden, policy_sees_env=True)
        assert ok["success"] and ok["failure_reason"] is None
        bad = run_episode(max_rate_probe_policy, seed=1, hidden=hidden)
        assert not bad["success"] and bad["failure_reason"] == "shower"
        env = CrystallizerEnv(seed=1, hidden=hidden)
        env.step(1.0)
        assert not env.is_unsafe()  # heating a little is allowed


def test_shower_test_uses_the_reported_sigma():
    """The reading the state reports is the value compared with the limit: a
    batch whose readings never exceed the limit never showers, and the
    showering step's reading is above it."""
    smax = 0.15
    env = CrystallizerEnv(seed=2, hidden=HiddenParams(sigma_max=smax))
    state, _ = env.reset()
    while True:
        state, _, done, _, _ = env.step(-1.0)
        if done:
            break
    assert env.failure_reason == "shower"
    assert state["sigma"] > smax
    # the informed controller holds the reading at or below the limit and never showers
    env = CrystallizerEnv(seed=2, hidden=HiddenParams(sigma_max=smax))
    env.reset()
    from crystallizer_utils import oracle_action
    peak = 0.0
    for _ in range(env.config.max_steps):
        state, _, done, _, _ = env.step(oracle_action(env, smax))
        peak = max(peak, state["sigma"])
        if done:
            break
    assert env.is_success() and peak <= smax and peak > 0.9 * smax


def test_informed_controller_is_faster_than_the_cautious_one_above_the_bottom():
    for smax in (0.08, 0.165, 0.25):
        hidden = HiddenParams(sigma_max=smax)
        oracle = run_oracle_episode(1, hidden)
        cautious = run_cautious_episode(1, hidden)
        assert oracle["success"] and cautious["success"]
        if smax == 0.08:
            assert oracle["steps"] == cautious["steps"]
        else:
            assert oracle["steps"] < cautious["steps"]


def test_faster_limit_means_shorter_batch():
    steps = [run_episode(make_rider(s * (1 - 1e-9)), seed=0, hidden=HiddenParams(sigma_max=s),
                         policy_sees_env=True)["steps"] for s in (0.08, 0.165, 0.25)]
    assert steps[0] > steps[1] > steps[2]
    assert steps[0] < EnvConfig().max_steps  # slowest informed batch fits the horizon
