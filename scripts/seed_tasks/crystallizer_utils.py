"""Offline rollout helpers and reference policies for the crystallizer.

Not part of the environment core — do not embed in LLM prompts.
"""
from __future__ import annotations

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
    env = CrystallizerEnv(seed=seed, hidden=hidden, config=config)
    state = env.observe()
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
    """Exact one-step lookahead: move to the temperature that puts the pre-growth
    supersaturation at `sigma_target`, never below the target temperature.
    With sigma_target = sigma_max this is the time-optimal oracle."""
    cfg = config or EnvConfig()

    def policy(state: State) -> float:
        T_des = max(saturation_temperature(state["c"] / (1.0 + sigma_target), cfg), state["T_target"])
        a = (T_des - state["T"]) / (cfg.max_rate * cfg.dt / 60.0)
        return max(-1.0, min(1.0, a))
    return policy


def cautious_policy(state: State) -> float:
    """Blind reference: rides at the lowest limit in the hidden range (0.08) minus a margin."""
    return make_tracking_policy(0.077)(state)


def max_rate_probe_policy(state: State) -> float:
    """Cools at the jacket limit. Showers in minutes on every batch; useful only in a probe run."""
    return -1.0


if __name__ == "__main__":
    seeds = list(range(20))
    for smax in (0.08, 0.165, 0.25):
        hidden = HiddenParams(sigma_max=smax)
        for name, pol in [("cautious", cautious_policy), ("oracle", make_tracking_policy(smax * (1 - 1e-9))),
                          ("max_rate_probe", max_rate_probe_policy)]:
            print(f"sigma_max={smax:.3f} {name:15s}", score_policy(pol, seeds, hidden))
