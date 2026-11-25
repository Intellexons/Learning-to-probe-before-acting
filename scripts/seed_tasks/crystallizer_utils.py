"""Offline rollout helpers and reference controllers for the crystallizer.

Not part of the environment core; never embedded in a prompt. Everything here
runs in-process with the environment object in hand, which is what the Stage 0
gate and the blind and oracle references need. A controller the briefed model
writes sees only `state`; the references here may also use `env.peek`, the
exact one-step lookahead, because they stand for the best a hand-written
controller can do with and without the hidden limit.

Batches: every helper constructs the env and then calls `reset()` once before
the first step, exactly as the runner's policy template does. `reset()` without
a seed continues the seeded stream, so the batch scored here is the batch the
runner scores for the same seed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from crystallizer_env import CrystallizerEnv, EnvConfig, HiddenParams, State

# A controller: either `policy(state)` as the model writes it, or
# `controller(state, env)` for in-process references that use the lookahead.
Policy = Callable[[State], float]
EnvPolicy = Callable[[State, CrystallizerEnv], float]

SIGMA_MAX_RANGE = (0.08, 0.25)      # the hidden limit's range; CrystallizerEnv.sample_hidden
RIDE_BELOW = 1.0 - 1e-9             # a rider holds sigma a hair under its level, never on it


def run_episode(
    policy: Callable,
    seed: int,
    hidden: Optional[HiddenParams] = None,
    config: Optional[EnvConfig] = None,
    return_trace: bool = False,
    policy_sees_env: bool = False,
) -> Dict:
    """Run one batch under `policy` and return its outcome.

    `policy_sees_env=True` calls `policy(state, env)` instead of `policy(state)`;
    it is for the in-process references and the Stage 0 families, which use the
    env's lookahead. A controller that raises scores the error floor of the
    runner (-300) on this batch, as the runner would."""
    env = CrystallizerEnv(seed=seed, hidden=hidden, config=config)
    state, _ = env.reset()
    total_reward = 0.0
    trace: List[Dict] = []
    for _ in range(env.config.max_steps):
        try:
            action = policy(dict(state), env) if policy_sees_env else policy(dict(state))
        except Exception:
            return {"reward": -300.0, "success": False, "failure_reason": "policy_exception",
                    "steps": env.t, "minutes": env.t * env.config.dt / 60.0,
                    "trace": trace if return_trace else None, "hidden": env.hidden}
        next_state, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        if return_trace:
            trace.append({"t": float(env.t), "T": float(env.T), "c": float(env.c),
                          "L": float(env.crystal_size()), "sigma": float(env.sigma()),
                          "rate_cmd": float(env.rate_cmd), "reward": float(reward)})
        state = next_state
        if terminated:
            break
    return {
        "reward": float(total_reward),
        "success": bool(env.is_success()),
        "failure_reason": env.failure_reason,
        "steps": int(env.t),
        "minutes": env.t * env.config.dt / 60.0,
        "trace": trace if return_trace else None,
        "hidden": env.hidden,
    }


# -----------------------------------------------------------------------------
# The rider: hold the end-of-step supersaturation at a level
# -----------------------------------------------------------------------------

def ride_action(env: CrystallizerEnv, level: float, iters: int = 30) -> float:
    """The most cooling such that the supersaturation the state reports after the step
    stays at or below `level`, found by bisection on `env.peek`, and never below the target
    temperature. With `level` = the hidden limit this is the time-optimal action; with a
    lower level it is the fastest controller that is safe for every limit above it."""
    cfg = env.config
    step_degC = cfg.max_rate * cfg.dt / 60.0
    a_floor = max(-1.0, min(0.0, (env.T_target - env.T) / step_degC))   # no overshoot
    if env.peek(a_floor) <= level:
        return a_floor
    lo, hi = a_floor, 0.0            # lo cools too hard, hi is safe (no cooling raises nothing)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if env.peek(mid) <= level:
            hi = mid
        else:
            lo = mid
    return hi


def make_rider(level: float) -> EnvPolicy:
    """`controller(state, env)` that rides at `level`: the tracking family of the gate."""
    def controller(state: State, env: CrystallizerEnv) -> float:
        return ride_action(env, level)
    return controller


def oracle_action(env: CrystallizerEnv, sigma_max: float) -> float:
    """The time-optimal action when the limit is known: ride a hair under it."""
    return ride_action(env, sigma_max * RIDE_BELOW)


def run_oracle_episode(seed: int, hidden: HiddenParams, config: Optional[EnvConfig] = None) -> Dict:
    """The informed reference: the time-optimal controller on the batch the runner scores."""
    return run_episode(make_rider(hidden.sigma_max * RIDE_BELOW), seed=seed, hidden=hidden,
                       config=config, policy_sees_env=True)


def cautious_controller(state: State, env: CrystallizerEnv) -> float:
    """The blind reference: rides a hair under the lowest limit in the hidden range, the
    fastest controller that never showers whatever the batch's limit is."""
    return ride_action(env, SIGMA_MAX_RANGE[0] * RIDE_BELOW)


def run_cautious_episode(seed: int, hidden: HiddenParams, config: Optional[EnvConfig] = None) -> Dict:
    return run_episode(cautious_controller, seed=seed, hidden=hidden, config=config,
                       policy_sees_env=True)


def max_rate_probe_policy(state: State) -> float:
    """Cools at the jacket limit. Showers in minutes on every batch; useful only in a probe run."""
    return -1.0


if __name__ == "__main__":
    for smax in (0.08, 0.165, 0.25):
        hidden = HiddenParams(sigma_max=smax)
        for name, run in (("cautious", run_cautious_episode), ("oracle", run_oracle_episode)):
            rows = [run(s, hidden) for s in range(20)]
            print(f"sigma_max={smax:.3f} {name:9s} reward {sum(r['reward'] for r in rows) / 20:7.1f}  "
                  f"success {sum(r['success'] for r in rows) / 20:.2f}  "
                  f"steps {sum(r['steps'] for r in rows) / 20:6.1f}")
