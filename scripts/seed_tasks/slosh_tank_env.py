from dataclasses import dataclass
from typing import Dict, Optional
import math
import random

State = Dict[str, float]


@dataclass(frozen=True)
class HiddenParams:
    actuator_gain: float


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 0.06
    max_steps: int = 105
    gravity: float = 9.81
    coupling: float = 1.00
    max_accel: float = 1.00
    slosh_length: float = 0.70
    damping: float = 0.06
    track_limit: float = 3.0
    goal_min_abs: float = 1.4
    goal_max_abs: float = 2.2
    pos_tol: float = 0.12
    vel_tol: float = 0.12
    theta_tol: float = 0.07
    omega_tol: float = 0.18
    spill_theta: float = 0.42
    success_bonus: float = 120.0
    fail_penalty: float = 80.0
    distance_weight: float = 0.06
    velocity_weight: float = 0.015
    slosh_weight: float = 0.35
    omega_weight: float = 0.025
    action_weight: float = 0.015
    time_weight: float = 0.03


class SloshTankEnv:
    def __init__(self, seed=0, hidden=None, config=None):
        self.rng = random.Random(seed)
        self.config = config or EnvConfig()
        self.hidden = hidden or self.sample_hidden(self.rng)
        self.reset(seed=seed)

    @staticmethod
    def sample_hidden(rng):
        return HiddenParams(actuator_gain=rng.choice([0.65, 0.80, 1.00]))

    def reset(self, seed=None):
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
        return self.observe()

    def observe(self):
        return {
            "t": float(self.t),
            "x": float(self.x),
            "v": float(self.v),
            "theta": float(self.theta),
            "omega": float(self.omega),
            "goal": float(self.goal),
            "steps_left": float(self.config.max_steps - self.t),
        }

    def step(self, action):
        if self.done:
            return self.observe(), 0.0, True, False, {}
        c = self.config
        try:
            a = float(action)
        except Exception:
            a = 0.0
        if not math.isfinite(a):
            a = 0.0
        self.a_cmd = max(-c.max_accel, min(c.max_accel, a))
        self.a_true = self.hidden.actuator_gain * self.a_cmd

        self.v += c.dt * self.a_true
        self.x += c.dt * self.v
        self.omega += c.dt * (-(c.gravity / c.slosh_length) * self.theta
                              - c.damping * self.omega
                              - c.coupling * self.a_true)
        self.theta += c.dt * self.omega
        self.t += 1

        dist = abs(self.x - self.goal)
        reward = -(c.distance_weight * dist
                   + c.velocity_weight * abs(self.v)
                   + c.slosh_weight * abs(self.theta)
                   + c.omega_weight * abs(self.omega)
                   + c.action_weight * self.a_cmd ** 2
                   + c.time_weight)

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

        self.done = success or unsafe or (self.t >= c.max_steps)
        return self.observe(), reward, self.done, False, {}

    def is_success(self):
        c = self.config
        return (abs(self.x - self.goal) <= c.pos_tol
                and abs(self.v) <= c.vel_tol
                and abs(self.theta) <= c.theta_tol
                and abs(self.omega) <= c.omega_tol)

    def is_unsafe(self):
        c = self.config
        return abs(self.x) > c.track_limit or abs(self.theta) > c.spill_theta
