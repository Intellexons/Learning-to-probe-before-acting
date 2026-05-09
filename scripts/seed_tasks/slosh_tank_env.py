"""Slosh-tank mechanics environment for code-policy generation.

A cart carries liquid modeled as a damped pendulum slosh mode. The policy
commands horizontal cart acceleration. Hidden scenario parameters change
actuator response and slosh dynamics.

Rollout helpers, scoring utilities, and reference policies live in
slosh_tank_utils.py — not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random


State = Dict[str, float]


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


