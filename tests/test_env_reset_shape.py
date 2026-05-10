"""Locks in the Gymnasium-compliant `reset() -> (state, info)` shape on
both seed envs. The base model's prior is `obs, info = env.reset()` and
deviating from it silently breaks every probe a model writes; this test
prevents a future refactor from regressing the signature."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "seed_tasks"))


def test_slosh_reset_returns_state_and_info():
    from slosh_tank_env import HiddenParams, SloshTankEnv

    env = SloshTankEnv(seed=0, hidden=HiddenParams(actuator_gain=0.8))
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2
    state, info = out
    assert isinstance(state, dict)
    assert isinstance(info, dict)
    # Sanity: state contains the documented keys
    for key in ("t", "x", "v", "theta", "omega", "goal", "steps_left"):
        assert key in state, f"missing state key: {key}"


def test_slosh_step_still_returns_5_tuple():
    """Reset's signature change must not have collateral effects on step."""
    from slosh_tank_env import HiddenParams, SloshTankEnv

    env = SloshTankEnv(seed=0, hidden=HiddenParams(actuator_gain=0.8))
    env.reset()
    out = env.step(0.1)
    assert isinstance(out, tuple) and len(out) == 5


def test_puck_reset_returns_state_and_info():
    from puck_tilted_plane_env import PuckTiltedPlaneEnv

    env = PuckTiltedPlaneEnv(seed=0)
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2
    state, info = out
    assert isinstance(state, dict)
    assert isinstance(info, dict)
    for key in ("px", "py", "vx", "vy", "gx", "gy"):
        assert key in state, f"missing state key: {key}"


def test_puck_step_still_returns_5_tuple():
    from puck_tilted_plane_env import PuckTiltedPlaneEnv

    env = PuckTiltedPlaneEnv(seed=0)
    env.reset()
    out = env.step((0.1, 0.0))
    assert isinstance(out, tuple) and len(out) == 5
