"""Compact slosh-tank mechanics environment for code-policy generation.

A cart carries liquid. The liquid is modeled as one damped slosh mode, like a
pendulum attached to the cart. The policy commands horizontal acceleration.
The hidden scenario parameters change actuator response and slosh dynamics.

Design goals:
- Small enough to include in an LLM prompt.
- Hidden unknowns are identifiable by a short first-turn probe.
- Final control requires moving to a target while settling the liquid.
- Conservative controllers are safe but slow; aggressive controllers excite slosh.

Policy API
----------
The user/model writes:

    def policy(state):
        return acceleration_command

where acceleration_command is a float. The environment clips it to [-1, 1].

Observed state:
    t, x, v, theta, omega, goal, steps_left

Hidden per-scenario parameters, not shown in state:
    actuator_gain, action_delay, actuator_deadband, slosh_length, damping

Turn-1/probe use
----------------
For rehearsal/probing, use run_episode(..., return_trace=True) and summarize_trace(...).
A useful first-turn probe is usually pulse/coast/reverse/coast, which reveals:
    actuator strength, action delay, deadband, slosh period, damping, spill risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import math
import random


State = Dict[str, float]
Policy = Callable[[State], float]


@dataclass(frozen=True)
class HiddenParams:
    """Scenario parameters hidden from the policy state."""

    actuator_gain: float = 1.0
    action_delay: int = 0
    actuator_deadband: float = 0.0
    slosh_length: float = 0.70
    damping: float = 0.06


@dataclass(frozen=True)
class EnvConfig:
    # Integration / episode length.
    dt: float = 0.06
    max_steps: int = 105

    # Mechanics.
    gravity: float = 9.81
    coupling: float = 1.00
    max_accel: float = 1.00

    # Track and target.
    track_limit: float = 3.0
    goal_min_abs: float = 1.4
    goal_max_abs: float = 2.2

    # Success thresholds on true state.
    pos_tol: float = 0.12
    vel_tol: float = 0.12
    theta_tol: float = 0.07
    omega_tol: float = 0.18

    # Safety.
    spill_theta: float = 0.42

    # Reward weights.
    success_bonus: float = 120.0
    fail_penalty: float = 80.0
    distance_weight: float = 0.06
    velocity_weight: float = 0.015
    slosh_weight: float = 0.35
    omega_weight: float = 0.025
    action_weight: float = 0.015
    time_weight: float = 0.03


class SloshTankEnv:
    """One-dimensional cart with one slosh mode.

    True state:
        x      cart position
        v      cart velocity
        theta  slosh angle / fluid surface displacement mode
        omega  slosh angular velocity

    Dynamics:
        a_true = actuator_fault(a_cmd)
        v     += dt * a_true
        x     += dt * v
        omega += dt * (-(g/L) * theta - damping * omega - coupling * a_true)
        theta += dt * omega

    The sign convention is intentionally simple: cart acceleration excites the
    slosh mode. A good controller must shape acceleration/braking and optionally
    damp theta/omega.
    """

    def __init__(
        self,
        seed: int = 0,
        hidden: Optional[HiddenParams] = None,
        config: Optional[EnvConfig] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.config = config or EnvConfig()
        self.hidden = hidden or self.sample_hidden(self.rng)
        self.reset(seed=seed)

    @staticmethod
    def sample_hidden(rng: random.Random) -> HiddenParams:
        """Sample a hidden scenario.

        Start curriculum suggestion:
        - First train/evaluate with only actuator_gain or only action_delay varying.
        - Then enable deadband, slosh_length, and damping variation.
        """
        # V1: actuator_gain only. Clean gap confirmed at max_steps=105 (see slosh_gain_sweep.png).
        return HiddenParams(
            actuator_gain=rng.choice([0.65, 0.80, 1.00]),
            action_delay=0,
            actuator_deadband=0.0,
            slosh_length=0.70,
            damping=0.06,
        )

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self.rng.seed(seed)

        c = self.config
        sign = -1.0 if self.rng.random() < 0.5 else 1.0
        self.goal = sign * self.rng.uniform(c.goal_min_abs, c.goal_max_abs)

        self.t = 0
        self.x = 0.0
        self.v = 0.0
        self.theta = self.rng.uniform(-0.015, 0.015)
        self.omega = 0.0
        self.done = False
        self.a_cmd = 0.0
        self.a_true = 0.0
        self.failure_reason: Optional[str] = None

        # Delay queue stores clipped command accelerations before actuator gain/deadband.
        delay_len = max(0, int(self.hidden.action_delay))
        self._action_queue: List[float] = [0.0 for _ in range(delay_len)]

        return self.observe()

    def observe(self) -> State:
        """Return the policy-visible state.

        In v1 this exposes clean x/v/theta/omega. Sensor faults can be added later
        by corrupting these values while leaving the true state internal.
        """
        return {
            "t": float(self.t),
            "x": float(self.x),
            "v": float(self.v),
            "theta": float(self.theta),
            "omega": float(self.omega),
            "goal": float(self.goal),
            "steps_left": float(self.config.max_steps - self.t),
        }

    def _clip_action(self, action: Any) -> float:
        try:
            a = float(action)
        except Exception:
            a = 0.0
        if not math.isfinite(a):
            a = 0.0
        return max(-self.config.max_accel, min(self.config.max_accel, a))

    def _apply_actuator_fault(self, a_cmd: float) -> float:
        h = self.hidden

        # Delay is applied to the command before gain/deadband.
        if h.action_delay > 0:
            self._action_queue.append(a_cmd)
            delayed = self._action_queue.pop(0)
        else:
            delayed = a_cmd

        # Deadband means small commands physically do nothing.
        if abs(delayed) < h.actuator_deadband:
            delayed = 0.0

        return h.actuator_gain * delayed

    def step(self, action: Any) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        if self.done:
            return self.observe(), 0.0, True, False, {}

        c = self.config
        h = self.hidden

        a_cmd = self._clip_action(action)
        a_true = self._apply_actuator_fault(a_cmd)

        # Semi-implicit Euler integration. Good enough for this toy control env.
        self.v += c.dt * a_true
        self.x += c.dt * self.v

        slosh_accel = -((c.gravity / h.slosh_length) * self.theta) - (h.damping * self.omega) - (c.coupling * a_true)
        self.omega += c.dt * slosh_accel
        self.theta += c.dt * self.omega

        self.t += 1

        dist = abs(self.x - self.goal)
        reward = 0.0
        reward -= c.distance_weight * dist
        reward -= c.velocity_weight * abs(self.v)
        reward -= c.slosh_weight * abs(self.theta)
        reward -= c.omega_weight * abs(self.omega)
        reward -= c.action_weight * (a_cmd * a_cmd)
        reward -= c.time_weight

        success = self.is_success()
        unsafe = self.is_unsafe()

        if success:
            reward += c.success_bonus
            self.failure_reason = None
        elif unsafe:
            reward -= c.fail_penalty
            self.failure_reason = "unsafe_or_spilled"
        elif self.t >= c.max_steps:
            self.failure_reason = "timeout"

        self.a_cmd = a_cmd
        self.a_true = a_true
        self.done = success or unsafe or (self.t >= c.max_steps)
        return self.observe(), reward, self.done, False, {}

    def is_success(self) -> bool:
        c = self.config
        return (
            abs(self.x - self.goal) <= c.pos_tol
            and abs(self.v) <= c.vel_tol
            and abs(self.theta) <= c.theta_tol
            and abs(self.omega) <= c.omega_tol
        )

    def is_unsafe(self) -> bool:
        c = self.config
        return abs(self.x) > c.track_limit or abs(self.theta) > c.spill_theta


# -----------------------------------------------------------------------------
# Rollout/scoring helpers
# -----------------------------------------------------------------------------


def run_episode(
    policy: Policy,
    seed: int,
    hidden: Optional[HiddenParams] = None,
    config: Optional[EnvConfig] = None,
    return_trace: bool = False,
) -> Dict[str, Any]:
    """Run one scenario and return score plus optional trace."""
    env = SloshTankEnv(seed=seed, hidden=hidden, config=config)
    state = env.observe()
    total_reward = 0.0
    trace: List[Dict[str, float]] = []

    for _ in range(env.config.max_steps):
        try:
            action = policy(dict(state))
        except Exception:
            # Policy errors are strongly bad but do not crash the evaluator.
            return {
                "reward": -200.0,
                "success": False,
                "failure_reason": "policy_exception",
                "steps": env.t,
                "trace": trace if return_trace else None,
            }

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if return_trace:
            row = {
                "t": float(env.t),
                "x": float(env.x),
                "v": float(env.v),
                "theta": float(env.theta),
                "omega": float(env.omega),
                "goal": float(env.goal),
                "a_cmd": float(env.a_cmd),
                "a_true": float(env.a_true),
                "reward": float(reward),
            }
            trace.append(row)

        state = next_state
        if terminated or truncated:
            break

    return {
        "reward": float(total_reward),
        "success": bool(env.is_success()),
        "failure_reason": env.failure_reason,
        "steps": int(env.t),
        "goal": float(env.goal),
        "trace": trace if return_trace else None,
        # Include hidden only for offline debugging/baseline sweeps. Do not put this
        # into the policy-visible state or normal LLM tool message.
        "hidden": env.hidden,
    }


def score_policy(
    policy: Policy,
    seeds: List[int],
    config: Optional[EnvConfig] = None,
) -> Dict[str, float]:
    """Score a policy across paired scenarios."""
    rewards = []
    successes = 0
    failures = {}

    for seed in seeds:
        result = run_episode(policy, seed=seed, config=config, return_trace=False)
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
# Probe summary helpers for first-turn diagnostics
# -----------------------------------------------------------------------------


def summarize_trace(trace: List[Dict[str, float]]) -> Dict[str, float]:
    """Extract simple diagnostics from a probe rollout.

    These are intended for the first-turn tool message. They are deliberately
    mechanical and interpretable, not hidden-parameter labels.
    """
    if not trace:
        return {}

    max_abs_theta = max(abs(r["theta"]) for r in trace)
    max_abs_omega = max(abs(r["omega"]) for r in trace)
    max_abs_x = max(abs(r["x"]) for r in trace)
    final = trace[-1]

    # Estimate command-response delay by finding first step where true accel differs
    # from zero after a nonzero command. This is simple and works for pulse probes.
    first_cmd_t = None
    first_response_t = None
    for r in trace:
        if first_cmd_t is None and abs(r["a_cmd"]) > 1e-6:
            first_cmd_t = int(r["t"])
        if first_cmd_t is not None and first_response_t is None and abs(r["a_true"]) > 1e-6:
            first_response_t = int(r["t"])
            break
    estimated_delay_steps = 0.0
    if first_cmd_t is not None and first_response_t is not None:
        estimated_delay_steps = float(max(0, first_response_t - first_cmd_t))

    # Estimate actuator response from early nonzero true accel values.
    nonzero = [abs(r["a_true"]) / max(abs(r["a_cmd"]), 1e-6) for r in trace if abs(r["a_cmd"]) > 0.3 and abs(r["a_true"]) > 1e-6]
    estimated_actuator_gain = sum(nonzero) / len(nonzero) if nonzero else 0.0

    # Peak timing for theta after excitation.
    peak_t = 0.0
    peak_val = 0.0
    for r in trace:
        if abs(r["theta"]) > abs(peak_val):
            peak_val = r["theta"]
            peak_t = r["t"]

    # Crude peak list for period/decay estimates.
    peaks: List[Tuple[float, float]] = []
    for i in range(1, len(trace) - 1):
        prev_abs = abs(trace[i - 1]["theta"])
        cur_abs = abs(trace[i]["theta"])
        next_abs = abs(trace[i + 1]["theta"])
        if cur_abs >= prev_abs and cur_abs >= next_abs and cur_abs > 0.01:
            peaks.append((trace[i]["t"], cur_abs))

    estimated_peak_spacing = 0.0
    decay_ratio = 0.0
    if len(peaks) >= 2:
        estimated_peak_spacing = peaks[1][0] - peaks[0][0]
        decay_ratio = peaks[1][1] / max(peaks[0][1], 1e-6)

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
    """Slow, safe-ish baseline. Should be robust but not high-performing."""
    x = state["x"]
    v = state["v"]
    goal = state["goal"]
    theta = state["theta"]
    omega = state["omega"]

    a = 0.55 * (goal - x) - 0.75 * v + 0.70 * theta + 0.25 * omega
    return max(-0.35, min(0.35, a))


def nominal_fast_policy(state: State) -> float:
    """Aggressive nominal controller. Good in easy cases, likely excites slosh."""
    x = state["x"]
    v = state["v"]
    goal = state["goal"]
    theta = state["theta"]
    omega = state["omega"]

    a = 1.25 * (goal - x) - 1.15 * v + 1.30 * theta + 0.45 * omega
    return max(-1.0, min(1.0, a))


def pulse_probe_policy(state: State) -> float:
    """Canonical first-turn diagnostic probe.

    Small pulse tests deadband. Larger pulse tests gain/delay and excites slosh.
    Reverse pulse tests timing/symmetry. Coast reveals period and damping.
    """
    t = int(state["t"])
    if t < 8:
        return 0.20
    if t < 16:
        return 0.0
    if t < 24:
        return 0.55
    if t < 32:
        return -0.55
    return 0.0


if __name__ == "__main__":
    seeds = list(range(50))
    for name, pol in [
        ("conservative", conservative_policy),
        ("nominal_fast", nominal_fast_policy),
        ("pulse_probe", pulse_probe_policy),
    ]:
        print(name, score_policy(pol, seeds))

    # Show one probe summary.
    result = run_episode(pulse_probe_policy, seed=3, return_trace=True)
    print("probe_hidden_for_debug", result["hidden"])
    print("probe_summary", summarize_trace(result["trace"] or []))
