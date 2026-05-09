"""Slosh gain sweep: blind vs oracle success rate vs kappa.

Config: max_steps=105, goal=[1.4,2.2]m, L=0.70m, theta_tol=0.07 (all defaults).
Sweep kappa from 0.40 to 1.00 in steps of 0.05.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "seed_tasks"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slosh_tank_env import EnvConfig, HiddenParams
from slosh_tank_utils import run_episode, nominal_fast_policy

CFG = EnvConfig()  # max_steps=105, goal=[1.4,2.2], L=0.70 via defaults

SEEDS = list(range(200))
KAPPAS = [round(0.40 + 0.02 * i, 2) for i in range(31)]  # 0.40..1.00


def make_oracle(kappa):
    def policy(state):
        x, v, goal = state["x"], state["v"], state["goal"]
        theta, omega = state["theta"], state["omega"]
        a = (1.25 * (goal - x) - 1.15 * v + 1.30 * theta + 0.45 * omega) / kappa
        return max(-1.0, min(1.0, a))
    return policy


blind_rates, oracle_rates = [], []

for kappa in KAPPAS:
    hidden = HiddenParams(actuator_gain=kappa)
    blind_ok = oracle_ok = 0
    for s in SEEDS:
        if run_episode(nominal_fast_policy, seed=s, hidden=hidden, config=CFG)["success"]:
            blind_ok += 1
        if run_episode(make_oracle(kappa), seed=s, hidden=hidden, config=CFG)["success"]:
            oracle_ok += 1
    blind_rates.append(100 * blind_ok / len(SEEDS))
    oracle_rates.append(100 * oracle_ok / len(SEEDS))
    print(f"kappa={kappa:.2f}  blind={blind_rates[-1]:.1f}%  oracle={oracle_rates[-1]:.1f}%")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(KAPPAS, blind_rates,  "o-", label="blind (nominal PD)", color="steelblue")
ax.plot(KAPPAS, oracle_rates, "s-", label="oracle (gains / κ)", color="darkorange")
ax.axhline(50, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("actuator gain κ")
ax.set_ylabel("success rate (%)")
ax.set_title("Slosh tank: blind vs oracle — max_steps=105, goal=[1.4,2.2]m, L=0.70m")
ax.legend()
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "docs", "slosh_gain_sweep.png")
fig.savefig(out, dpi=150)
print(f"\nSaved to {os.path.abspath(out)}")
