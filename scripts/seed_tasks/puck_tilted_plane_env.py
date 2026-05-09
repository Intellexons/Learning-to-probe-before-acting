import math
import random


class PuckTiltedPlaneEnv:
    """Puck on a tilted frictioned plane. Drive it to a goal position.

    Per-scenario randomized (hidden from state — read this code to learn them):
      tilt_x, tilt_y : constant body-frame acceleration from the plane tilt
      mu             : linear drag coefficient
      a_max          : max thrust magnitude (action is norm-clipped to this)
      goal, start    : positions in a [-6, 6] arena

    A naive "thrust straight at the goal" policy loses — it has to cancel
    the tilt drift and lead the velocity.
    """

    def __init__(self, seed=0):
        rng = random.Random(seed)
        self.tilt_x = rng.uniform(-2.5, 2.5)
        self.tilt_y = rng.uniform(-2.5, 2.5)
        self.mu = rng.uniform(0.1, 0.8)
        tilt_mag = math.hypot(self.tilt_x, self.tilt_y)
        self.a_max = rng.uniform(tilt_mag * 1.05 + 0.3, tilt_mag * 1.05 + 1.0)
        self.gx = rng.uniform(-4.5, 4.5)
        self.gy = rng.uniform(-4.5, 4.5)
        while True:
            self.p0x = rng.uniform(-4.5, 4.5)
            self.p0y = rng.uniform(-4.5, 4.5)
            if math.hypot(self.p0x - self.gx, self.p0y - self.gy) >= 3.0:
                break
        self.v0x = rng.gauss(0.0, 0.5)
        self.v0y = rng.gauss(0.0, 0.5)
        self.dt = 0.05
        self.max_steps = 120
        self.bound = 6.0
        self.goal_radius = 0.25
        self.steps = 0
        self.px = self.py = self.vx = self.vy = 0.0

    def reset(self):
        self.px, self.py = self.p0x, self.p0y
        self.vx, self.vy = self.v0x, self.v0y
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return {
            "px": self.px, "py": self.py,
            "vx": self.vx, "vy": self.vy,
            "gx": self.gx, "gy": self.gy,
        }

    def step(self, action):
        ax = float(action[0])
        ay = float(action[1])
        norm = math.hypot(ax, ay)
        if norm > self.a_max:
            ax *= self.a_max / norm
            ay *= self.a_max / norm
        self.vx += (ax + self.tilt_x - self.mu * self.vx) * self.dt
        self.vy += (ay + self.tilt_y - self.mu * self.vy) * self.dt
        self.px += self.vx * self.dt
        self.py += self.vy * self.dt
        if self.px < -self.bound:
            self.px, self.vx = -self.bound, 0.0
        elif self.px > self.bound:
            self.px, self.vx = self.bound, 0.0
        if self.py < -self.bound:
            self.py, self.vy = -self.bound, 0.0
        elif self.py > self.bound:
            self.py, self.vy = self.bound, 0.0
        self.steps += 1
        dist = math.hypot(self.px - self.gx, self.py - self.gy)
        reward = -self.dt * dist
        terminated = dist < self.goal_radius or self.steps >= self.max_steps
        if dist < self.goal_radius:
            reward += 50.0
        return self.get_state(), reward, terminated, False, {}
