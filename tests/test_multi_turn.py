"""Tests for the branched-rollout pure-math + helper functions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.multi_turn import (  # noqa: E402
    assemble_per_token_advantages,
    assemble_tool_mask,
    compute_branched_advantages,
    error_floor_reward,
    execute_turn,
    format_tool_message,
    rollout_diagnostics,
)
from src.runner import ERROR_PENALTY_PER_SCENARIO  # noqa: E402


def test_compute_branched_advantages_uniform_grid_zeros_advantages():
    """When all rewards are equal, every advantage is 0."""
    rewards = [[5.0, 5.0], [5.0, 5.0]]
    v0, v1, a1, a2 = compute_branched_advantages(rewards)
    assert v0 == 5.0
    assert v1 == [5.0, 5.0]
    assert a1 == [[0.0, 0.0], [0.0, 0.0]]
    assert a2 == [[0.0, 0.0], [0.0, 0.0]]


def test_compute_branched_advantages_first_turn_winner():
    """First-turn variant 0 leads to high rewards, variant 1 to low ones.
    A_turn1 should be positive for variant 0 and negative for variant 1.
    A_turn2 should be ~0 within each row (continuations are similar)."""
    rewards = [[10.0, 11.0], [-9.0, -10.0]]
    v0, v1, a1, a2 = compute_branched_advantages(rewards)
    assert v0 == pytest.approx(0.5)
    assert v1[0] == pytest.approx(10.5)
    assert v1[1] == pytest.approx(-9.5)
    # turn-1 advantage discriminates first-turn variants
    assert a1[0][0] > 0
    assert a1[1][0] < 0
    # turn-2 advantage discriminates within a row
    assert a2[0][0] == pytest.approx(-0.5)  # 10 - 10.5
    assert a2[0][1] == pytest.approx(0.5)   # 11 - 10.5


def test_compute_branched_advantages_rejects_ragged_grid():
    with pytest.raises(ValueError):
        compute_branched_advantages([[1.0, 2.0], [3.0]])


def test_compute_branched_advantages_rejects_empty():
    with pytest.raises(ValueError):
        compute_branched_advantages([])


def test_assemble_per_token_advantages_layout():
    adv = assemble_per_token_advantages(3, 2, 4, a_turn1=1.5, a_turn2=-2.0)
    assert adv == [1.5, 1.5, 1.5, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0]


def test_assemble_tool_mask_layout():
    mask = assemble_tool_mask(3, 2, 4)
    assert mask == [1, 1, 1, 0, 0, 1, 1, 1, 1]


def test_rollout_diagnostics_first_turn_dominates():
    """Strong between-variant difference, low within-variant noise →
    high signal_ratio, A_turn1_std > 0."""
    rewards = [[10.0, 10.5, 9.5, 10.0], [-5.0, -5.5, -4.5, -5.0]]
    _, _, a_turn1, a_turn2 = compute_branched_advantages(rewards)
    diag = rollout_diagnostics(rewards, a_turn1, a_turn2)
    assert diag["v1_between_std"] > 5.0  # V_1 differs by ~15
    assert diag["v1_within_std_mean"] < 1.0  # rows are tight
    assert diag["v1_signal_ratio"] > 1.0  # first-turn dominates
    assert diag["a_turn1_std"] > 0  # turn-1 advantages discriminate


def test_rollout_diagnostics_within_variant_noise_dominates():
    """Same V_1 across variants but high within-variant spread →
    A_turn1_std == 0, A_turn2_std > 0."""
    rewards = [[5.0, 0.0, 10.0, -5.0], [5.0, 0.0, 10.0, -5.0]]
    _, _, a_turn1, a_turn2 = compute_branched_advantages(rewards)
    diag = rollout_diagnostics(rewards, a_turn1, a_turn2)
    assert diag["v1_between_std"] == pytest.approx(0.0)
    assert diag["v1_within_std_mean"] > 0
    assert diag["v1_signal_ratio"] == pytest.approx(0.0)
    assert diag["a_turn1_std"] == pytest.approx(0.0)  # NO first-turn signal
    assert diag["a_turn2_std"] > 0


def test_rollout_diagnostics_uniform_grid_is_silent():
    """All rewards equal → every diagnostic is 0; signal_ratio is inf
    (zero-over-zero, defined as inf to flag the degenerate case)."""
    rewards = [[5.0, 5.0], [5.0, 5.0]]
    _, _, a_turn1, a_turn2 = compute_branched_advantages(rewards)
    diag = rollout_diagnostics(rewards, a_turn1, a_turn2)
    assert diag["v1_between_std"] == 0.0
    assert diag["v1_within_std_mean"] == 0.0
    assert diag["a_turn1_std"] == 0.0
    assert diag["a_turn2_std"] == 0.0
    assert diag["v1_signal_ratio"] == float("inf")


def test_format_tool_message_role_user():
    msg = format_tool_message("[tool] hello")
    assert msg == {"role": "user", "content": "[tool] hello"}


def test_error_floor_reward_scales_with_scenarios():
    assert error_floor_reward(1) == ERROR_PENALTY_PER_SCENARIO
    assert error_floor_reward(8) == ERROR_PENALTY_PER_SCENARIO * 8


def _puck_task_one_scenario():
    """Minimal puck task — one scenario — for execute_turn integration tests."""
    env_code = '''
import math, random
class TinyEnv:
    def __init__(self, seed=0):
        rng = random.Random(seed)
        self.gx = rng.uniform(-1, 1)
        self.steps = 0
    def reset(self):
        return {"px": 0.0, "gx": self.gx}
    def step(self, action):
        self.steps += 1
        done = self.steps >= 5
        return {"px": float(action[0]) * 0.1, "gx": self.gx}, -1.0, done
'''
    return {
        "task_id": "tiny",
        "env_class": "TinyEnv",
        "env_code": env_code,
        "scenarios": [{"scenario_id": "s0", "seed": 0}],
    }


def test_execute_turn_policy_last_turn_returns_real_reward():
    task = _puck_task_one_scenario()
    completion = """Here goes:
```policy
def policy(state):
    return (1.0, 0.0)
```
"""
    tr = execute_turn(completion, task, is_last_turn=True, timeout_policy=10, max_steps=10)
    assert tr.mode == "policy"
    # Reward integrates -1.0 over 5 steps = -5.0 (no goal in tiny env)
    assert tr.reward == pytest.approx(-5.0, abs=0.01)
    assert "[tool] policy" in tr.text


def test_execute_turn_main_on_last_turn_is_rejected():
    task = _puck_task_one_scenario()
    completion = """```main
print('thinking')
```"""
    tr = execute_turn(completion, task, is_last_turn=True, timeout_policy=10, max_steps=10)
    assert tr.mode == "main"
    assert tr.reward == error_floor_reward(1)
    assert "rejected" in tr.text


def test_execute_turn_invalid_last_turn_is_rejected_at_floor():
    task = _puck_task_one_scenario()
    tr = execute_turn(
        "no fenced block here, just text",
        task,
        is_last_turn=True,
        timeout_policy=10,
        max_steps=10,
    )
    assert tr.mode == "invalid"
    assert tr.reward == error_floor_reward(1)


def test_execute_turn_main_on_intermediate_turn_returns_stdout_no_reward():
    task = _puck_task_one_scenario()
    completion = """```main
import math
print('answer:', math.sqrt(16))
```
"""
    tr = execute_turn(completion, task, is_last_turn=False, timeout_policy=10, max_steps=10)
    assert tr.mode == "main"
    assert tr.reward is None
    assert "answer:" in tr.text
    assert "4.0" in tr.text


def test_execute_turn_invalid_intermediate_turn_is_floor():
    """Even mid-trajectory, an invalid turn carries a floor penalty so the
    branched-rollout V̂_1 estimate reflects the bad turn-1."""
    task = _puck_task_one_scenario()
    tr = execute_turn(
        "junk text",
        task,
        is_last_turn=False,
        timeout_policy=10,
        max_steps=10,
    )
    assert tr.mode == "invalid"
    assert tr.reward == error_floor_reward(1)
