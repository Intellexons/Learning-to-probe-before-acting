"""Offline rollout helpers, scoring utilities, and reference policies for the slosh tank.

Not part of the environment core — do not embed in LLM prompts.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from slosh_tank_env import EnvConfig, HiddenParams, SloshTankEnv, State

Policy = Callable[[State], float]


# -----------------------------------------------------------------------------
# Rollout / scoring
# -----------------------------------------------------------------------------

def run_episode(
    policy: Policy,
    seed: int,
    hidden: Optional[HiddenParams] = None,
    config: Optional[EnvConfig] = None,
    return_trace: bool = False,
) -> Dict:
    env = SloshTankEnv(seed=seed, hidden=hidden, config=config)
    state = env.observe()
    total_reward = 0.0
    trace: List[Dict] = []

    for _ in range(env.config.max_steps):
        try:
            action = policy(dict(state))
        except Exception:
            return {
                "reward": -200.0,
                "success": False,
                "failure_reason": "policy_exception",
                "steps": env.t,
                "trace": trace if return_trace else None,
            }

        next_state, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        if return_trace:
            trace.append({
                "t": float(env.t),
                "x": float(env.x),
                "v": float(env.v),
                "theta": float(env.theta),
                "omega": float(env.omega),
                "goal": float(env.goal),
                "a_cmd": float(env.a_cmd),
                "a_true": float(env.a_true),
                "reward": float(reward),
            })

        state = next_state
        if terminated:
            break

    return {
        "reward": float(total_reward),
        "success": bool(env.is_success()),
        "failure_reason": env.failure_reason,
        "steps": int(env.t),
        "goal": float(env.goal),
        "trace": trace if return_trace else None,
        "hidden": env.hidden,
    }


def score_policy(
    policy: Policy,
    seeds: List[int],
    config: Optional[EnvConfig] = None,
) -> Dict:
    rewards, successes, failures = [], 0, {}
    for seed in seeds:
        result = run_episode(policy, seed=seed, config=config)
        rewards.append(result["reward"])
        successes += int(result["success"])
        reason = result["failure_reason"] or "success"
        failures[reason] = failures.get(reason, 0) + 1
    n = max(1, len(seeds))
    return {
        "mean_reward": sum(rewards) / n,
        "success_rate": successes / n,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "num_scenarios": float(n),
        **{f"count_{k}": float(v) for k, v in failures.items()},
    }


# -----------------------------------------------------------------------------
# Probe trace diagnostics
# -----------------------------------------------------------------------------

def summarize_trace(trace: List[Dict]) -> Dict:
    if not trace:
        return {}

    max_abs_theta = max(abs(r["theta"]) for r in trace)
    max_abs_omega = max(abs(r["omega"]) for r in trace)
    max_abs_x    = max(abs(r["x"])     for r in trace)
    final = trace[-1]

    first_cmd_t = first_response_t = None
    for r in trace:
        if first_cmd_t is None and abs(r["a_cmd"]) > 1e-6:
            first_cmd_t = int(r["t"])
        if first_cmd_t is not None and first_response_t is None and abs(r["a_true"]) > 1e-6:
            first_response_t = int(r["t"])
            break
    estimated_delay_steps = float(max(0, first_response_t - first_cmd_t)) if (first_cmd_t is not None and first_response_t is not None) else 0.0

    nonzero = [abs(r["a_true"]) / max(abs(r["a_cmd"]), 1e-6)
               for r in trace if abs(r["a_cmd"]) > 0.3 and abs(r["a_true"]) > 1e-6]
    estimated_actuator_gain = sum(nonzero) / len(nonzero) if nonzero else 0.0

    peak_t, peak_val = 0.0, 0.0
    for r in trace:
        if abs(r["theta"]) > abs(peak_val):
            peak_val, peak_t = r["theta"], r["t"]

    peaks: List[Tuple[float, float]] = []
    for i in range(1, len(trace) - 1):
        c = abs(trace[i]["theta"])
        if c >= abs(trace[i-1]["theta"]) and c >= abs(trace[i+1]["theta"]) and c > 0.01:
            peaks.append((trace[i]["t"], c))

    estimated_peak_spacing = peaks[1][0] - peaks[0][0] if len(peaks) >= 2 else 0.0
    decay_ratio = peaks[1][1] / max(peaks[0][1], 1e-6) if len(peaks) >= 2 else 0.0

    return {
        "max_abs_theta": float(max_abs_theta),
        "max_abs_omega": float(max_abs_omega),
        "max_abs_x": float(max_abs_x),
        "final_x": float(final["x"]),
        "final_v": float(final["v"]),
        "final_theta": float(final["theta"]),
        "final_omega": float(final["omega"]),
        "estimated_delay_steps": float(estimated_delay_steps),
        "estimated_actuator_gain": float(estimated_actuator_gain),
        "first_theta_peak_step": float(peak_t),
        "theta_peak_abs": float(abs(peak_val)),
        "estimated_peak_spacing_steps": float(estimated_peak_spacing),
        "theta_decay_ratio": float(decay_ratio),
    }


# -----------------------------------------------------------------------------
# Reference policies
# -----------------------------------------------------------------------------

def conservative_policy(state: State) -> float:
    x, v, goal = state["x"], state["v"], state["goal"]
    theta, omega = state["theta"], state["omega"]
    a = 0.55 * (goal - x) - 0.75 * v + 0.70 * theta + 0.25 * omega
    return max(-0.35, min(0.35, a))


def nominal_fast_policy(state: State) -> float:
    x, v, goal = state["x"], state["v"], state["goal"]
    theta, omega = state["theta"], state["omega"]
    a = 1.25 * (goal - x) - 1.15 * v + 1.30 * theta + 0.45 * omega
    return max(-1.0, min(1.0, a))


def pulse_probe_policy(state: State) -> float:
    t = int(state["t"])
    if t < 8:  return  0.20
    if t < 16: return  0.0
    if t < 24: return  0.55
    if t < 32: return -0.55
    return 0.0


if __name__ == "__main__":
    seeds = list(range(50))
    for name, pol in [
        ("conservative", conservative_policy),
        ("nominal_fast", nominal_fast_policy),
        ("pulse_probe",  pulse_probe_policy),
    ]:
        print(name, score_policy(pol, seeds))

    result = run_episode(pulse_probe_policy, seed=3, return_trace=True)
    print("probe_hidden_for_debug", result["hidden"])
    print("probe_summary", summarize_trace(result["trace"] or []))
