"""Contract tests for the crystallizer env: Gymnasium-shaped reset/step, the
hidden limit leaves no trace in the observation, and the two reference
behaviours (ride the limit, cross the limit) do what the gate relies on."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "seed_tasks"))

from crystallizer_env import CrystallizerEnv, EnvConfig, HiddenParams, solubility  # noqa: E402
from crystallizer_utils import make_tracking_policy, run_episode  # noqa: E402

KEYS = ("t", "T", "c", "L", "T_target", "steps_left")


def test_reset_returns_state_and_info_with_documented_keys():
    env = CrystallizerEnv(seed=0, hidden=HiddenParams(sigma_max=0.15))
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2
    state, info = out
    assert isinstance(state, dict) and isinstance(info, dict)
    assert set(state) == set(KEYS)
    assert abs(state["c"] - solubility(state["T"], env.config)) < 1e-12  # saturated start


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
    pol = make_tracking_policy(0.09)
    sa, sb = a.observe(), b.observe()
    for _ in range(a.config.max_steps):
        act = pol(sa)
        sa, _, da, _, _ = a.step(act)
        sb, _, db, _, _ = b.step(act)
        assert sa == sb
        if da or db:
            break
    assert a.is_success() and b.is_success()


def test_riding_the_limit_succeeds_and_crossing_it_showers():
    for smax in (0.08, 0.165, 0.25):
        hidden = HiddenParams(sigma_max=smax)
        ok = run_episode(make_tracking_policy(smax * (1 - 1e-9)), seed=1, hidden=hidden)
        assert ok["success"] and ok["failure_reason"] is None
        bad = run_episode(make_tracking_policy(smax * 1.05), seed=1, hidden=hidden)
        assert not bad["success"] and bad["failure_reason"] == "shower"
        env = CrystallizerEnv(seed=1, hidden=hidden)
        env.step(1.0)
        assert not env.is_unsafe()  # heating a little is allowed


def test_faster_limit_means_shorter_batch():
    steps = [run_episode(make_tracking_policy(s * (1 - 1e-9)), seed=0,
                         hidden=HiddenParams(sigma_max=s))["steps"] for s in (0.08, 0.165, 0.25)]
    assert steps[0] > steps[1] > steps[2]
    assert steps[0] < EnvConfig().max_steps  # slowest informed batch fits the horizon
