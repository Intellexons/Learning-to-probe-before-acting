"""Seeded batch cooling crystallizer with a hidden metastable limit.

A saturated solution (1 kg of solvent) holding a known mass of seed crystals
is cooled from T0 to a target temperature. Cooling makes the solution
supersaturated; the seeds grow and consume the excess. If the supersaturation
ever exceeds the hidden metastable limit, a shower of tiny new crystals
nucleates at once and the batch is ruined. The only control is the cooling
rate, limited by the jacket.

Physics (T in degC, c in kg solute per kg solvent, monodisperse seeds):
    solubility       c*(T) = sol_c0 + sol_c1*T + sol_c2*T**2
    supersaturation  sigma = c / c*(T) - 1
    growth           dL/dt = k_g * sigma**g          (dissolution with the same law if sigma < 0)
    mass balance     dc/dt = -rho * N * 3 * L**2 * dL/dt   (deposited mass goes onto the N seeds)
    shower           sigma > sigma_max  -> unsafe, episode over

Order inside one step: the jacket moves T by the commanded amount; sigma is
evaluated at the new T with the current c and compared with the hidden limit;
then the crystals grow for dt seconds. Everything is deterministic given the
seed, so a controller that knows sigma_max can predict the next sigma exactly.

sigma_max is the hidden parameter. It appears nowhere except in the shower
test, so the observation carries no information about it until it is crossed.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import math
import random

State = Dict[str, float]


@dataclass(frozen=True)
class HiddenParams:
    sigma_max: float          # relative supersaturation at which a shower of new nuclei forms


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 20.0                  # s per control step
    max_steps: int = 600              # 200 min
    # solubility c*(T) = sol_c0 + sol_c1*T + sol_c2*T^2  (kg solute / kg solvent, T in degC)
    sol_c0: float = 0.20
    sol_c1: float = 0.0060
    sol_c2: float = 0.00010
    # growth kinetics
    k_g: float = 1.5e-6               # m/s at sigma = 1
    g: float = 1.5                    # growth order
    rho: float = 2000.0               # crystal density, kg/m^3 (cubic shape factor 1)
    seed_size: float = 1.0e-4         # m, initial seed size
    # per-seed randomisation
    T0_min: float = 55.0
    T0_max: float = 65.0
    T_target_min: float = 18.0
    T_target_max: float = 25.0
    seed_loading_min: float = 0.03    # seed mass as a fraction of the theoretical yield
    seed_loading_max: float = 0.06
    # equipment
    max_rate: float = 1.5             # degC/min, jacket heating/cooling limit
    T_min: float = 15.0
    T_max: float = 70.0
    # success / failure
    T_tol: float = 0.25               # degC above the target still counts as "at target"
    sigma_tol: float = 0.02           # batch counts as equilibrated below this supersaturation
    dissolved_frac: float = 0.3       # seeds shrunk below this fraction of their size = lost
    success_bonus: float = 120.0
    fail_penalty: float = 80.0
    time_weight: float = 0.10         # per step


def solubility(T: float, c: EnvConfig) -> float:
    return c.sol_c0 + c.sol_c1 * T + c.sol_c2 * T * T


def saturation_temperature(conc: float, c: EnvConfig) -> float:
    """Inverse of solubility(): the temperature at which `conc` is exactly saturated."""
    # sol_c2*T^2 + sol_c1*T + (sol_c0 - conc) = 0, take the physical (larger) root
    disc = c.sol_c1 * c.sol_c1 - 4.0 * c.sol_c2 * (c.sol_c0 - conc)
    if disc < 0.0:
        return -c.sol_c1 / (2.0 * c.sol_c2)
    return (-c.sol_c1 + math.sqrt(disc)) / (2.0 * c.sol_c2)


class CrystallizerEnv:
    def __init__(self, seed=0, hidden=None, config=None):
        if hidden is None:
            raise ValueError(
                "hidden must be specified explicitly. "
                "Use HiddenParams(sigma_max=s) or CrystallizerEnv.sample_hidden(rng)."
            )
        self.rng = random.Random(seed)
        self.config = config or EnvConfig()
        self.hidden = hidden
        self.reset(seed=seed)

    @staticmethod
    def sample_hidden(rng):
        return HiddenParams(sigma_max=rng.uniform(0.08, 0.25))

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        c = self.config
        self.T0 = self.rng.uniform(c.T0_min, c.T0_max)
        self.T_target = self.rng.uniform(c.T_target_min, c.T_target_max)
        loading = self.rng.uniform(c.seed_loading_min, c.seed_loading_max)
        self.T = self.T0
        self.c = solubility(self.T0, c)                       # saturated start
        yield_mass = solubility(self.T0, c) - solubility(self.T_target, c)
        self.m_seed = loading * yield_mass
        self.N = self.m_seed / (c.rho * c.seed_size ** 3)     # number of seed crystals
        self.m_c = self.m_seed                                 # crystal mass on the seeds
        self.t = 0
        self.done = False
        self.showered = False
        self.rate_cmd = 0.0
        self.failure_reason: Optional[str] = None
        return self.observe(), {}

    # --- derived quantities -------------------------------------------------------------
    def crystal_size(self) -> float:
        return (max(self.m_c, 0.0) / (self.config.rho * self.N)) ** (1.0 / 3.0)

    def sigma(self) -> float:
        return self.c / solubility(self.T, self.config) - 1.0

    def observe(self):
        return {
            "t": float(self.t),
            "T": float(self.T),
            "c": float(self.c),
            "L": float(self.crystal_size()),
            "T_target": float(self.T_target),
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
        self.rate_cmd = max(-1.0, min(1.0, a))

        # 1. jacket moves the temperature (a < 0 cools, a > 0 heats)
        self.T = max(c.T_min, min(c.T_max, self.T + self.rate_cmd * c.max_rate * c.dt / 60.0))

        # 2. supersaturation at the new temperature, before any growth: the shower test
        sigma = self.sigma()
        if sigma > self.hidden.sigma_max:
            self.showered = True

        # 3. growth or dissolution for dt seconds (mass-conserving, cannot cross saturation)
        L = self.crystal_size()
        area_mass = c.rho * self.N * 3.0 * L * L            # d(m_c)/dL
        rate = c.k_g * abs(sigma) ** c.g
        excess = self.c - solubility(self.T, c)
        if sigma >= 0.0:
            dm = min(area_mass * rate * c.dt, max(excess, 0.0))
        else:
            dm = -min(area_mass * rate * c.dt, self.m_c, max(-excess, 0.0))
        self.m_c += dm
        self.c -= dm
        self.t += 1

        reward = -c.time_weight
        success = self.is_success()
        unsafe = self.is_unsafe()
        if success:
            reward += c.success_bonus
            self.failure_reason = None
        elif unsafe:
            reward -= c.fail_penalty
            self.failure_reason = "shower" if self.showered else "seeds_dissolved"
        elif self.t >= c.max_steps:
            self.failure_reason = "timeout"

        self.done = success or unsafe or (self.t >= c.max_steps)
        return self.observe(), reward, self.done, False, {}

    def is_success(self):
        c = self.config
        return (not self.showered
                and self.T <= self.T_target + c.T_tol
                and self.sigma() <= c.sigma_tol)

    def is_unsafe(self):
        c = self.config
        return self.showered or self.crystal_size() < c.dissolved_frac * c.seed_size
