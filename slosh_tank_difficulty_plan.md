# Slosh Tank Mechanics Environment: Difficulty Ladder and Implementation Plan

## 1. Purpose

The slosh tank environment is meant to be a compact mechanics benchmark for multi-turn code-policy generation.

The core task is:

> Move a cart carrying a sloshing liquid to a target quickly, while ending with low residual slosh, under hidden mechanical unknowns.

The hidden unknowns should be discoverable through a short first-turn diagnostic experiment, then used by the second-turn final policy.

The environment should **not** become a trivial “set the right recipe parameter” task. The final policy should still perform dynamic control: accelerate, brake, damp, and settle. The unknowns affect *how* to control, not simply which scalar parameter to choose.

---

## 2. Core Mechanics

Minimal state:

```python
state = {
    "t": t,
    "x": x,              # cart position
    "v": v,              # cart velocity
    "theta": theta,      # slosh angle / liquid displacement mode
    "omega": omega,      # slosh angular velocity
    "goal": goal_x,
    "steps_left": steps_left,
}
```

Action:

```python
a_cmd = policy(state)   # commanded cart acceleration, clipped to [-1, 1]
```

True dynamics:

```python
a_true = actuator_fault(a_cmd)

v += dt * a_true
x += dt * v

omega += dt * (-(g / L) * theta - damping * omega - coupling * a_true)
theta += dt * omega
```

The important coupling is:

```python
-coupling * a_true
```

Accelerating the cart excites liquid slosh. The controller must move fast while avoiding residual oscillation.

Success should require all of:

```python
abs(x - goal) < pos_tol
abs(v) < vel_tol
abs(theta) < theta_tol
abs(omega) < omega_tol
```

This prevents the task from collapsing into “reach target by any means.”

---

## 3. What the First Turn Should Examine

The first turn should not try to solve the task. It should run a **mechanical diagnostic experiment**.

The canonical probe is:

```python
def policy(state):
    t = int(state["t"])

    if t < 8:
        return 0.20      # small pulse: tests deadband
    elif t < 16:
        return 0.0       # coast
    elif t < 24:
        return 0.55      # larger pulse: tests gain/delay and excites slosh
    elif t < 32:
        return -0.55     # reverse pulse: tests timing/symmetry
    else:
        return 0.0       # coast: observe slosh period and damping
```

This one probe can reveal:

```text
1. Actuator response strength
2. Action delay
3. Deadband
4. Slosh period / natural frequency
5. Slosh damping
6. Spill risk
```

The tool message after the first turn should summarize telemetry such as:

```text
max_abs_theta
max_abs_omega
final_x
final_v
final_theta
final_omega
estimated_delay_steps
estimated_actuator_gain
first_theta_peak_step
estimated_peak_spacing_steps
theta_decay_ratio
spill / unsafe flag
```

The final policy can then adapt:

```text
weak actuator      -> stronger commands, earlier braking
action delay       -> lower gains, anticipate braking
deadband           -> avoid tiny damping corrections
long slosh period  -> slower accel/brake timing
low damping        -> actively damp and approach gently
sensor lag         -> predict current theta/omega or lower feedback gains
```

---

## 4. Difficulty Levels

### Level 1 — Actuator Gain

#### Unknown

The commanded acceleration is scaled before reaching the plant:

```python
a_true = actuator_gain * a_cmd
```

Example values:

```python
actuator_gain in [0.70, 0.85, 1.00, 1.15]
```

#### Why it is easy

A single pulse reveals actuator strength from the resulting velocity change.

#### First-turn examination

```python
def policy(state):
    t = int(state["t"])
    if t < 10:
        return 0.5
    return 0.0
```

#### What to measure

```text
delta_v after pulse
estimated_gain ≈ observed_delta_v / expected_delta_v
```

#### Final policy adaptation

```text
weak actuator   -> use larger commands and brake earlier
strong actuator -> use gentler commands and avoid overshoot
```

#### Implementation activation

In `sample_hidden`:

```python
return HiddenParams(
    actuator_gain=rng.choice([0.70, 0.85, 1.00, 1.15]),
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=0.06,
)
```

This is the easiest first experiment.

---

### Level 2 — Action Delay

#### Unknown

Commands are applied several steps later:

```python
a_applied[t] = a_cmd[t - delay]
```

Example values:

```python
action_delay in [0, 2, 4]
```

#### Why it is easy

Command switches happen at known times. The observed velocity slope changes later if delay is present.

#### First-turn examination

```python
def policy(state):
    t = int(state["t"])
    if t < 8:
        return 0.5
    elif t < 16:
        return -0.5
    return 0.0
```

#### What to measure

```text
time when command changes
time when cart acceleration / velocity slope changes
estimated_delay_steps
```

#### Final policy adaptation

```text
brake earlier
use lower gains
avoid rapid sign-changing feedback
prefer planned smooth motion profiles
```

#### Implementation activation

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=rng.choice([0, 2, 4]),
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=0.06,
)
```

This is probably the best single first unknown because it matters strongly for slosh timing.

---

### Level 3 — Actuator Deadband

#### Unknown

Small commands physically do nothing:

```python
if abs(a_cmd) < actuator_deadband:
    a_true = 0.0
else:
    a_true = actuator_gain * a_cmd
```

Example values:

```python
actuator_deadband in [0.00, 0.12, 0.20]
```

#### Why it is medium difficulty

You need at least two command magnitudes to distinguish deadband from weak actuator gain.

#### First-turn examination

```python
def policy(state):
    t = int(state["t"])
    if t < 8:
        return 0.15      # small pulse
    elif t < 16:
        return 0.0
    elif t < 24:
        return 0.45      # large pulse
    return 0.0
```

#### What to measure

```text
small pulse response
large pulse response
```

Diagnosis:

```text
small no response + large response -> deadband
both weak response                 -> low actuator gain
both normal response               -> no deadband / normal gain
```

#### Final policy adaptation

```text
avoid tiny corrective actions
use thresholded pulses
use bang-bang or pulse-width style damping
```

#### Implementation activation

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=rng.choice([0.00, 0.12, 0.20]),
    slosh_length=0.70,
    damping=0.06,
)
```

This is a good second or third unknown because it changes the qualitative controller style.

---

### Level 4 — Slosh Natural Frequency

#### Unknown

The liquid’s effective pendulum length changes:

```python
omega_dot = -(g / slosh_length) * theta - ...
```

Example values:

```python
slosh_length in [0.45, 0.70, 1.00]
```

Shorter length means faster slosh. Longer length means slower slosh.

#### Why it is medium difficulty

The controller has to excite the slosh and observe peak timing or zero crossings.

#### First-turn examination

```python
def policy(state):
    t = int(state["t"])
    if t < 10:
        return 0.45
    return 0.0
```

#### What to measure

```text
time of first theta peak
time between theta peaks
zero-crossing interval
estimated slosh period
```

#### Final policy adaptation

```text
time acceleration and braking to avoid exciting resonance
choose braking phase to cancel slosh
adapt anti-slosh feedback gains
```

#### Implementation activation

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=rng.choice([0.45, 0.70, 1.00]),
    damping=0.06,
)
```

This is the first truly slosh-specific unknown. Add this after actuator unknowns work.

---

### Level 5 — Slosh Damping

#### Unknown

Liquid oscillation decays quickly or slowly:

```python
omega_dot = ... - damping * omega
```

Example values:

```python
damping in [0.025, 0.060, 0.110]
```

#### Why it is medium difficulty

You need to observe at least two oscillation peaks or a long enough coast period.

#### First-turn examination

```python
def policy(state):
    t = int(state["t"])
    if t < 10:
        return 0.45
    return 0.0
```

Run long enough after the pulse for slosh to decay.

#### What to measure

```text
first_peak_abs_theta
second_peak_abs_theta
decay_ratio = second_peak / first_peak
```

#### Final policy adaptation

```text
low damping  -> move slower, actively damp, require gentle settling
high damping -> move faster, rely more on natural settling
```

#### Implementation activation

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=rng.choice([0.025, 0.060, 0.110]),
)
```

If episodes are very short, do not vary damping at first.

---

### Level 6 — Slosh Sensor Lag

#### Unknown

The observed slosh state is delayed:

```python
theta_obs[t] = theta_true[t - sensor_lag]
omega_obs[t] = omega_true[t - sensor_lag]
```

Example values:

```python
sensor_lag in [0, 2, 4]
```

#### Why it is medium-hard

The slosh signal still looks plausible, but the phase is wrong. A naive damping controller may inject energy instead of removing it.

#### First-turn examination

```python
def policy(state):
    t = int(state["t"])
    if t < 8:
        return 0.45
    elif t < 16:
        return -0.45
    return 0.0
```

#### What to measure

```text
phase relationship between command changes and observed theta response
whether observed theta peaks are late
consistency of omega with finite differences of theta
```

#### Final policy adaptation

```python
theta_est = theta_obs + lag_steps * dt * omega_obs
```

or:

```text
lower anti-slosh feedback gains
use smoother planned trajectories
avoid aggressive phase-sensitive damping
```

#### Implementation activation

You need to add fields to `HiddenParams`:

```python
slosh_sensor_lag: int = 0
```

Add a slosh observation queue in `reset`:

```python
self._theta_queue = [self.theta for _ in range(self.hidden.slosh_sensor_lag + 1)]
self._omega_queue = [self.omega for _ in range(self.hidden.slosh_sensor_lag + 1)]
```

Update queues after integration:

```python
self._theta_queue.append(self.theta)
self._omega_queue.append(self.omega)
theta_obs = self._theta_queue.pop(0)
omega_obs = self._omega_queue.pop(0)
```

Modify `observe`:

```python
return {
    ...
    "theta": theta_obs,
    "omega": omega_obs,
}
```

Activate:

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=0.06,
    slosh_sensor_lag=rng.choice([0, 2, 4]),
)
```

Do not start here. Add this after actuator and slosh-frequency variations are working.

---

### Level 7 — Position Sensor Bias

#### Unknown

The cart position is biased:

```python
x_obs = x_true + position_bias
```

Example values:

```python
position_bias in [-0.20, 0.0, 0.20]
```

#### Why it is hard

If the policy only sees biased position, the true target can become ambiguous unless the first-turn telemetry or reward gives enough information.

#### First-turn examination

Use a known pulse and compare position/velocity consistency:

```python
def policy(state):
    t = int(state["t"])
    if t < 10:
        return 0.4
    return 0.0
```

#### What to measure

```text
difference between x_obs change and integrated v_obs
final reported position versus actual score/telemetry
```

#### Final policy adaptation

```text
subtract estimated bias
use velocity and slosh cues more than absolute x
approach target conservatively
```

#### Implementation activation

Add to `HiddenParams`:

```python
position_bias: float = 0.0
```

Modify `observe`:

```python
"x": self.x + self.hidden.position_bias
```

Activate:

```python
position_bias=rng.choice([-0.20, 0.0, 0.20])
```

This can be unfair if the policy lacks enough information to infer the bias. Avoid in v1.

---

### Level 8 — Mixed Actuator Faults

#### Unknown

Multiple actuator effects happen together:

```text
actuator gain + delay + deadband
```

#### Why it is hard

The same behavior may have multiple possible explanations. For example, a weak response could be low gain, deadband, delay, or a combination.

#### First-turn examination

Use a multi-amplitude, sign-changing probe:

```python
def policy(state):
    t = int(state["t"])
    if t < 6:
        return 0.15
    elif t < 12:
        return 0.0
    elif t < 18:
        return 0.45
    elif t < 24:
        return -0.45
    return 0.0
```

#### What to measure

```text
small-pulse response
large-pulse response
response onset time
positive/negative symmetry
cart velocity change
```

#### Final policy adaptation

```text
gain scaling
earlier braking
thresholded commands
lower feedback gains
```

#### Implementation activation

```python
return HiddenParams(
    actuator_gain=rng.choice([0.70, 0.85, 1.00, 1.15]),
    action_delay=rng.choice([0, 2, 4]),
    actuator_deadband=rng.choice([0.00, 0.12, 0.20]),
    slosh_length=0.70,
    damping=0.06,
)
```

This is a good stage after all individual actuator faults have clean baseline gaps.

---

### Level 9 — Sensor Fault + Actuator Fault

#### Unknown

Both the control interface and observation interface are wrong:

```text
weak actuator + slosh sensor lag
deadband + noisy omega
position bias + action delay
```

#### Why it is hardest

Diagnostics become confounded. Did the tank respond late because the actuator is delayed, or because the sensor is lagged?

#### First-turn examination

Needs multiple channels:

```text
1. cart response to command       -> actuator diagnosis
2. slosh response to cart motion  -> slosh/sensor diagnosis
3. coast decay                    -> physical damping/frequency diagnosis
```

Probe:

```python
def policy(state):
    t = int(state["t"])
    if t < 8:
        return 0.4
    elif t < 16:
        return 0.0
    elif t < 24:
        return -0.4
    return 0.0
```

#### What to measure

```text
cart velocity response timing
theta peak timing relative to cart acceleration
theta decay during coast
consistency of omega with theta finite differences
```

#### Final policy adaptation

Fault-specific robust strategy:

```text
actuator delay      -> anticipate
sensor lag          -> predict current state
weak actuator       -> scale up commands
low damping         -> actively damp
deadband            -> threshold commands
```

#### Implementation activation

Enable a small number of predefined combinations, not arbitrary cross-product at first:

```python
fault_cases = [
    HiddenParams(actuator_gain=0.70, action_delay=0, actuator_deadband=0.0,
                 slosh_length=0.70, damping=0.06, slosh_sensor_lag=0),
    HiddenParams(actuator_gain=1.00, action_delay=4, actuator_deadband=0.0,
                 slosh_length=0.70, damping=0.06, slosh_sensor_lag=0),
    HiddenParams(actuator_gain=1.00, action_delay=0, actuator_deadband=0.0,
                 slosh_length=0.70, damping=0.06, slosh_sensor_lag=4),
    HiddenParams(actuator_gain=0.85, action_delay=2, actuator_deadband=0.12,
                 slosh_length=1.00, damping=0.025, slosh_sensor_lag=2),
]
return rng.choice(fault_cases)
```

Do not use this until simpler stages are validated.

---

## 5. Ranking Table

| Rank | Unknown | Difficulty | First-turn examination | Main measurement | Final policy change |
|---:|---|---|---|---|---|
| 1 | Actuator gain | Easy | Single pulse | Velocity change / estimated gain | Scale commands, brake earlier/later |
| 2 | Action delay | Easy | Sign-changing pulse | Response onset after command switch | Anticipate braking, lower gains |
| 3 | Deadband | Easy-medium | Small pulse + large pulse | Small response absent, large response present | Avoid tiny corrections, use pulses |
| 4 | Slosh frequency | Medium | Pulse then coast | Peak timing / zero crossings | Time accel/brake phases |
| 5 | Damping | Medium | Pulse then long coast | Peak decay ratio | Active damping vs fast motion |
| 6 | Slosh sensor lag | Medium-hard | Pulse/reverse pulse | Phase mismatch, omega consistency | Predict current theta, lower gains |
| 7 | Position bias | Hard | Motion consistency / telemetry | Position-vs-velocity inconsistency | Offset correction |
| 8 | Mixed actuator faults | Hard | Multi-amplitude pulse train | Gain + delay + deadband features | Combined compensation |
| 9 | Sensor + actuator faults | Hardest | Multi-channel diagnostic | Confounded timing/sensor evidence | Fault-specific robust strategy |

---

## 6. Best V1

### Recommendation

Start with:

```text
V1 = action_delay only
```

Use:

```python
action_delay in [0, 2, 4]
actuator_gain = 1.0
actuator_deadband = 0.0
slosh_length = 0.70
damping = 0.06
```

### Why action delay is the best V1

Action delay is the best first unknown because:

```text
1. It is easy to infer from a short probe.
2. It strongly affects slosh control.
3. It breaks naive high-gain feedback.
4. It forces anticipation instead of pure reaction.
5. It is physically plausible.
6. It should create visible differences between nominal, conservative, and fault-aware controllers.
```

### V1 first-turn probe

```python
def policy(state):
    t = int(state["t"])

    if t < 8:
        return 0.5
    elif t < 16:
        return -0.5
    else:
        return 0.0
```

### V1 tool feedback

Return:

```text
estimated_delay_steps
cart_delta_v_after_first_pulse
max_abs_theta
first_theta_peak_step
final_x
final_v
final_theta
final_omega
spill
```

### V1 final policy adaptations

```text
delay = 0:
    use normal anti-slosh feedback

delay = 2:
    brake earlier
    reduce feedback gains

delay = 4:
    use smoother planned motion
    avoid rapid sign changes
    start braking much earlier
```

### V1 implementation

In `sample_hidden`:

```python
@staticmethod
def sample_hidden(rng: random.Random) -> HiddenParams:
    return HiddenParams(
        actuator_gain=1.0,
        action_delay=rng.choice([0, 2, 4]),
        actuator_deadband=0.0,
        slosh_length=0.70,
        damping=0.06,
    )
```

This should be the first baseline-gap test.

---

## 7. Suggested Stages

### Stage A — Actuator-only curriculum

Purpose:

```text
Verify that first-turn probing helps identify control-interface faults.
```

Unknowns:

```text
A1. Actuator gain only
A2. Action delay only
A3. Deadband only
A4. Gain + delay + deadband
```

Canonical probe:

```python
def policy(state):
    t = int(state["t"])

    if t < 8:
        return 0.20
    elif t < 16:
        return 0.0
    elif t < 24:
        return 0.55
    elif t < 32:
        return -0.55
    else:
        return 0.0
```

Expected baseline behavior:

```text
nominal controller:
    good when no fault, poor with delay/deadband

conservative controller:
    safe but slow

diagnostic/fault-aware:
    better net reward, especially under delay/deadband
```

Activation example:

```python
return HiddenParams(
    actuator_gain=rng.choice([0.70, 0.85, 1.00, 1.15]),
    action_delay=rng.choice([0, 2, 4]),
    actuator_deadband=rng.choice([0.00, 0.12, 0.20]),
    slosh_length=0.70,
    damping=0.06,
)
```

Do not jump to full cross-product immediately. First test one unknown at a time.

---

### Stage B — Slosh-mode curriculum

Purpose:

```text
Make the unknown specifically about slosh timing and residual oscillation.
```

Unknowns:

```text
B1. Slosh frequency only
B2. Damping only
B3. Slosh frequency + damping
```

Probe:

```python
def policy(state):
    t = int(state["t"])

    if t < 10:
        return 0.45
    else:
        return 0.0
```

Expected behavior:

```text
wrong timing:
    reaches target but leaves residual slosh or spills

frequency-aware:
    times braking and damping better

low-damping-aware:
    slows down and actively damps before final approach
```

Activation example:

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=rng.choice([0.45, 0.70, 1.00]),
    damping=rng.choice([0.025, 0.060, 0.110]),
)
```

---

### Stage C — Sensor-fault curriculum

Purpose:

```text
Introduce observation-interface faults after the control-interface and slosh dynamics are working.
```

Unknowns:

```text
C1. Slosh sensor lag only
C2. Omega noise or missing omega
C3. Position bias
```

Probe:

```python
def policy(state):
    t = int(state["t"])

    if t < 8:
        return 0.45
    elif t < 16:
        return -0.45
    else:
        return 0.0
```

Expected behavior:

```text
nominal anti-slosh feedback:
    may inject energy under sensor lag

sensor-lag-aware:
    predicts current slosh state or lowers gains

position-bias-aware:
    estimates offset or approaches carefully
```

Implementation note:

Sensor faults require modifying `observe`, not the true dynamics.

Example:

```python
def observe(self):
    x_obs = self.x + self.hidden.position_bias
    theta_obs = delayed_theta_if_enabled
    omega_obs = delayed_omega_if_enabled

    return {
        "t": float(self.t),
        "x": x_obs,
        "v": float(self.v),
        "theta": theta_obs,
        "omega": omega_obs,
        "goal": float(self.goal),
        "steps_left": float(self.config.max_steps - self.t),
    }
```

---

### Stage D — Combined faults

Purpose:

```text
Test genuine diagnosis under confounded evidence.
```

Unknowns:

```text
D1. Gain + delay
D2. Delay + slosh frequency
D3. Deadband + low damping
D4. Weak actuator + sensor lag
D5. Small curated set of multi-fault cases
```

Probe:

```python
def policy(state):
    t = int(state["t"])

    if t < 6:
        return 0.15
    elif t < 12:
        return 0.0
    elif t < 18:
        return 0.45
    elif t < 24:
        return -0.45
    elif t < 55:
        return 0.0
    else:
        return 0.0
```

Activation should use curated cases instead of a huge random cross-product:

```python
fault_cases = [
    HiddenParams(actuator_gain=0.70, action_delay=0, actuator_deadband=0.0,
                 slosh_length=0.70, damping=0.06),
    HiddenParams(actuator_gain=1.00, action_delay=4, actuator_deadband=0.0,
                 slosh_length=0.70, damping=0.06),
    HiddenParams(actuator_gain=1.00, action_delay=0, actuator_deadband=0.20,
                 slosh_length=0.70, damping=0.06),
    HiddenParams(actuator_gain=0.85, action_delay=2, actuator_deadband=0.12,
                 slosh_length=1.00, damping=0.025),
]
return rng.choice(fault_cases)
```

Expected behavior:

```text
simple nominal controller:
    brittle

conservative controller:
    safe but slow

diagnostic/fault-aware:
    adapts strategy, best net reward
```

---

## 8. Baselines to Run Before RL

Do not start RL until the hand-written baselines show a real gap.

### Baseline 1 — Nominal Fast Controller

Assumes normal actuator and known slosh mode.

Expected:

```text
good in normal case
bad under action delay, sensor lag, deadband, or low damping
```

### Baseline 2 — Conservative Controller

Moves slowly with low acceleration.

Expected:

```text
safe across many unknowns
low reward because it is slow and often fails timeout/speed objective
```

### Baseline 3 — Oracle-Aware Controller

Gets hidden parameters directly for debugging only.

Expected:

```text
high reward
demonstrates achievable ceiling
```

### Baseline 4 — Probe-Then-Control Controller

Uses first-turn diagnostics or telemetry summary to choose controller settings.

Expected:

```text
near oracle on easy unknowns
above conservative and nominal
```

### Minimum acceptable gap

For the benchmark to be useful:

```text
conservative < diagnostic < oracle
nominal should fail under at least some faults
best blind should be meaningfully below probe-aware
```

A possible target:

```text
nominal fast:        high variance, many failures
conservative:        moderate success, low mean reward
probe-aware:         higher mean reward and success
oracle-aware:        highest reward
```

---

## 9. Tool Message Design

For the first-turn policy run, the tool message should not only return scalar reward.

Recommended tool feedback:

```text
probe_reward: ...
success: ...
failure_reason: ...
max_abs_theta: ...
max_abs_omega: ...
max_abs_x: ...
final_x: ...
final_v: ...
final_theta: ...
final_omega: ...
estimated_delay_steps: ...
estimated_actuator_gain: ...
first_theta_peak_step: ...
theta_peak_abs: ...
estimated_peak_spacing_steps: ...
theta_decay_ratio: ...
```

Do not return the actual hidden parameters in the normal LLM-facing tool message.

Hidden parameters can be returned only in offline debugging or oracle baselines.

---

## 10. Environment Code Activation Checklist

### To activate actuator gain only

```python
return HiddenParams(
    actuator_gain=rng.choice([0.70, 0.85, 1.00, 1.15]),
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=0.06,
)
```

### To activate action delay only

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=rng.choice([0, 2, 4]),
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=0.06,
)
```

### To activate deadband only

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=rng.choice([0.00, 0.12, 0.20]),
    slosh_length=0.70,
    damping=0.06,
)
```

### To activate slosh frequency only

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=rng.choice([0.45, 0.70, 1.00]),
    damping=0.06,
)
```

### To activate damping only

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=0,
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=rng.choice([0.025, 0.060, 0.110]),
)
```

### To activate v1 recommended

```python
return HiddenParams(
    actuator_gain=1.0,
    action_delay=rng.choice([0, 2, 4]),
    actuator_deadband=0.0,
    slosh_length=0.70,
    damping=0.06,
)
```

### To activate first combined actuator stage

```python
return HiddenParams(
    actuator_gain=rng.choice([0.70, 0.85, 1.00, 1.15]),
    action_delay=rng.choice([0, 2, 4]),
    actuator_deadband=rng.choice([0.00, 0.12, 0.20]),
    slosh_length=0.70,
    damping=0.06,
)
```

---

## 11. Recommended Development Order

### Step 1

Implement V1:

```text
action_delay only
```

Run baselines:

```text
nominal_fast
conservative
oracle_delay_aware
diagnostic_delay_aware
```

Check whether delay-aware policies outperform blind robust policies.

### Step 2

Add actuator gain only.

Check whether first-turn gain estimation improves reward.

### Step 3

Add deadband only.

Check whether thresholded/pulse controllers beat smooth small-correction controllers.

### Step 4

Add slosh frequency only.

Check whether period-aware braking improves final settling.

### Step 5

Add damping only.

Check whether low-damping cases require slower/active damping strategies.

### Step 6

Combine actuator unknowns:

```text
gain + delay + deadband
```

### Step 7

Combine actuator unknowns with slosh-mode unknowns:

```text
delay + slosh_length
deadband + low damping
weak actuator + long period
```

### Step 8

Only then add sensor faults:

```text
slosh sensor lag
position bias
omega noise
```

---

## 12. Summary Recommendation

Start simple.

Best V1:

```text
Slosh tank with hidden action delay in {0, 2, 4}
```

Use a first-turn sign-changing pulse to estimate delay. The second-turn controller should use that estimate to anticipate braking and reduce phase-sensitive feedback.

Then add:

```text
1. actuator gain
2. deadband
3. slosh frequency
4. damping
5. sensor lag
6. combinations
```

The final benchmark should reward:

```text
fast transport
low residual slosh
safe operation
fault-aware adaptation
```

The benchmark is successful only if:

```text
best blind robust controller < diagnostic/fault-aware controller < oracle-aware controller
```
