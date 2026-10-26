"""Offline rollout helpers and reference policies for the crystallizer.

Not part of the environment core — do not embed in LLM prompts.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

from crystallizer_env import CrystallizerEnv, EnvConfig, HiddenParams, State, saturation_temperature

Policy = Callable[[State], float]


def run_episode(
    policy: Policy,
    seed: int,
    hidden: Optional[HiddenParams] = None,
    config: Optional[EnvConfig] = None,
    return_trace: bool = False,
) -> Dict:
    # Mirror the runner: construct, then reset() once before the first step.
    # reset() without a seed continues the seeded RNG stream, so this is the
    # batch the runner's policy template runs on (not the constructor's).
    env = CrystallizerEnv(seed=seed, hidden=hidden, config=config)
    state, _ = env.reset()
    total_reward = 0.0
    trace: List[Dict] = []
    for _ in range(env.config.max_steps):
        try:
            action = policy(dict(state))
        except Exception:
            return {"reward": -200.0, "success": False, "failure_reason": "policy_exception",
                    "steps": env.t, "trace": trace if return_trace else None}
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


def score_policy(policy: Policy, seeds: List[int], hidden: HiddenParams,
                 config: Optional[EnvConfig] = None) -> Dict:
    rewards, successes, failures = [], 0, {}
    for seed in seeds:
        result = run_episode(policy, seed=seed, hidden=hidden, config=config)
        rewards.append(result["reward"])
        successes += int(result["success"])
        reason = result["failure_reason"] or "success"
        failures[reason] = failures.get(reason, 0) + 1
    n = max(1, len(seeds))
    return {"mean_reward": sum(rewards) / n, "success_rate": successes / n,
            "num_scenarios": float(n), **{f"count_{k}": float(v) for k, v in failures.items()}}


# -----------------------------------------------------------------------------
# Reference policies
# -----------------------------------------------------------------------------

def make_tracking_policy(sigma_target: float, config: Optional[EnvConfig] = None) -> Policy:
    """One-step lookahead with the solubility curve: move to the temperature that puts the
    supersaturation before growth at `sigma_target`, never below the target temperature. The
    end-of-step value the shower test uses is lower than that by the growth of the step, so this
    controller is safe at sigma_target = sigma_max and a little slower than `run_oracle_episode`.
    It needs the formula, so it is a reference, not something the briefed model can write."""
    cfg = config or EnvConfig()

    def policy(state: State) -> float:
        T_des = max(saturation_temperature(state["c"] / (1.0 + sigma_target), cfg), state["T_target"])
        a = (T_des - state["T"]) / (cfg.max_rate * cfg.dt / 60.0)
        return max(-1.0, min(1.0, a))
    return policy


def cautious_policy(state: State) -> float:
    """Blind reference: rides at the lowest limit in the hidden range (0.08) minus a margin."""
    return make_tracking_policy(0.077)(state)


def oracle_action(env: CrystallizerEnv, sigma_max: float) -> float:
    """The time-optimal action for the env's current state when the limit is known: the most
    cooling such that the end-of-step supersaturation (the tested value) stays at or below
    sigma_max, found by bisection on a copy of the env, and never below the target temperature."""
    cfg = env.config
    step_degC = cfg.max_rate * cfg.dt / 60.0
    a_floor = max(-1.0, min(0.0, (env.T_target - env.T) / step_degC))

    def safe(a: float) -> bool:
        e = copy.deepcopy(env)
        st, _, _, _, _ = e.step(a)
        return (not e.showered) and st["sigma"] <= sigma_max * (1.0 - 1e-9)

    if safe(a_floor):
        return a_floor
    lo, hi = a_floor, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if safe(mid):
            hi = mid
        else:
            lo = mid
    return hi


def run_oracle_episode(seed: int, hidden: HiddenParams, config: Optional[EnvConfig] = None) -> Dict:
    """The informed reference: the greedy time-optimal controller on the batch the runner scores
    (construct, then reset() once, as the runner does)."""
    env = CrystallizerEnv(seed=seed, hidden=hidden, config=config)
    env.reset()
    total = 0.0
    for _ in range(env.config.max_steps):
        _, reward, terminated, _, _ = env.step(oracle_action(env, hidden.sigma_max))
        total += reward
        if terminated:
            break
    return {"reward": float(total), "success": bool(env.is_success()),
            "failure_reason": env.failure_reason, "steps": int(env.t),
            "minutes": env.t * env.config.dt / 60.0}


def max_rate_probe_policy(state: State) -> float:
    """Cools at the jacket limit. Showers in minutes on every batch; useful only in a probe run."""
    return -1.0


if __name__ == "__main__":
    seeds = list(range(20))
    for smax in (0.08, 0.165, 0.25):
        hidden = HiddenParams(sigma_max=smax)
        for name, pol in [("cautious", cautious_policy), ("tracking", make_tracking_policy(smax * (1 - 1e-9))),
                          ("max_rate_probe", max_rate_probe_policy)]:
            print(f"sigma_max={smax:.3f} {name:15s}", score_policy(pol, seeds, hidden))
        oracle = [run_oracle_episode(s, hidden) for s in seeds]
        print(f"sigma_max={smax:.3f} {'oracle':15s}", {"mean_reward": sum(o["reward"] for o in oracle) / len(seeds),
              "success_rate": sum(o["success"] for o in oracle) / len(seeds),
              "mean_steps": sum(o["steps"] for o in oracle) / len(seeds)})
