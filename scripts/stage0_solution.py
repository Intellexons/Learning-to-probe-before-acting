"""The hand-written probe run and controller, from the brief alone.

What a control engineer would write with exactly what the brief offers and
nothing else: the probe-run object `env` with `reset`, `step`, the `done` flag,
`failure_reason`, and the `sigma` reading; the controller sees only `state`
and may keep memory. No solubility curve, no concentration arithmetic, no
lookahead. These are the Stage 0 gate's G3 (the gap is closable by probing)
and the achievable probe-aware reference the model stages are read against.

Four probe variants ask how much probing is needed:
  three-pass   one batch at the jacket limit until the shower, then two batches
               that go to the last safe level and creep up in eighths of the bracket
  one-pass     one batch at the jacket limit until the shower; the last safe reading
               and the reading at the shower are the bracket
  creep-0.005  one batch that raises its sigma setpoint by 0.005 every step
  creep-0.001  the same at 0.001 per step
Each prints SIGMA_MAX_LOW (a reading seen safe) and SIGMA_MAX_HIGH (a reading
at a shower). The controller holds the reading a margin under SIGMA_MAX_LOW.
"""
from __future__ import annotations

PROBE_THREE_PASS = '''
# Probe run: find the level of sigma at which this crystallizer showers, by crossing it.
lo, hi = 0.0, 0.40    # sigma_max is above lo (a reading seen safe) and below hi (a reading at a shower)
G = 0.016             # cautious guess at how much sigma rises per full-cooling step
for npass in range(3):
    s, _ = env.reset()
    if npass == 0:
        while True:         # first batch: cool at the jacket limit until the shower
            s, r, done, _, _ = env.step(-1.0)
            if done:
                break
            lo = max(lo, s["sigma"])
        if env.failure_reason == "shower":
            hi = min(hi, s["sigma"])
        continue
    target, inc = lo, (hi - lo) / 8.0      # later batches: go to the last safe level, then creep up
    while True:
        a = max(-1.0, min(0.0, -(target - s["sigma"]) / G))
        s, r, done, _, _ = env.step(a)
        if done:
            break
        lo = max(lo, s["sigma"])
        if s["sigma"] >= target - inc / 2:
            target = min(target + inc, hi)
    if env.failure_reason == "shower":
        hi = min(hi, s["sigma"])
print(f"SIGMA_MAX_LOW={lo:.5f} SIGMA_MAX_HIGH={hi:.5f}")
'''

PROBE_ONE_PASS = '''
# Probe run, one batch: cool at the jacket limit until the shower.
lo, hi = 0.0, 0.40
s, _ = env.reset()
while True:
    s, r, done, _, _ = env.step(-1.0)
    if done:
        break
    lo = max(lo, s["sigma"])
if env.failure_reason == "shower":
    hi = min(hi, s["sigma"])
print(f"SIGMA_MAX_LOW={lo:.5f} SIGMA_MAX_HIGH={hi:.5f}")
'''

PROBE_CREEP_TEMPLATE = '''
# Probe run, one batch: raise the sigma setpoint by STEP every step until the shower.
STEP = {step}
G = 0.016
lo, hi, target = 0.0, 0.40, STEP
s, _ = env.reset()
while True:
    a = max(-1.0, min(0.0, -(target - s["sigma"]) / G))
    s, r, done, _, _ = env.step(a)
    if done:
        break
    lo = max(lo, s["sigma"])
    target += STEP
if env.failure_reason == "shower":
    hi = min(hi, s["sigma"])
print(f"SIGMA_MAX_LOW={{lo:.5f}} SIGMA_MAX_HIGH={{hi:.5f}}")
'''

PROBE_VARIANTS = {
    "three-pass": PROBE_THREE_PASS,
    "one-pass": PROBE_ONE_PASS,
    "creep-0.005": PROBE_CREEP_TEMPLATE.format(step=0.005),
    "creep-0.001": PROBE_CREEP_TEMPLATE.format(step=0.001),
}
GATE_VARIANT = "three-pass"

CONTROLLER_MARGIN = 0.003

# The final run: hold the sigma reading just under the level the probe found.
CONTROLLER_TEMPLATE = '''
SETPOINT = {sigma_hat:.5f} - {margin}
G = 0.014                       # about how much sigma rises per full-cooling step
mem = {{"sigma_prev": None, "a_prev": 0.0, "eaten": 0.0}}
def policy(state):
    sigma, T, T_target = state["sigma"], state["T"], state["T_target"]
    if mem["sigma_prev"] is not None:
        # what the crystals ate last step: what cooling should have added, minus what we got
        eaten = mem["sigma_prev"] - mem["a_prev"] * G - sigma
        mem["eaten"] = 0.5 * mem["eaten"] + 0.5 * max(0.0, eaten)
    if T <= T_target:
        a = 0.0
    else:
        a = max(-1.0, min(0.0, -(SETPOINT - sigma + mem["eaten"]) / G))
        a = max(a, -(T - T_target) / 0.5)      # do not overshoot the target temperature
    mem["sigma_prev"], mem["a_prev"] = sigma, a
    return a
'''


def controller_source(sigma_hat: float, margin: float = CONTROLLER_MARGIN) -> str:
    return CONTROLLER_TEMPLATE.format(sigma_hat=sigma_hat, margin=margin)


def fence(src: str) -> str:
    return "```python\n" + src.strip("\n") + "\n```"


def parse_bracket(text: str) -> tuple[float, float] | None:
    """The (low, high) the probe printed, or None if the probe run did not print it."""
    if "SIGMA_MAX_LOW=" not in text or "SIGMA_MAX_HIGH=" not in text:
        return None
    try:
        lo = float(text.split("SIGMA_MAX_LOW=")[1].split()[0])
        hi = float(text.split("SIGMA_MAX_HIGH=")[1].split()[0])
    except (IndexError, ValueError):
        return None
    return lo, hi
