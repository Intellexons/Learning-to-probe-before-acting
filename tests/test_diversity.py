"""Tests for src.diversity metric and callback wiring."""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diversity import (  # noqa: E402
    DiversityCallback,
    _collapse_warning,
    _compute_metrics,
    _split_completions_by_turn,
)


def test_metrics_on_diverse_completions():
    completions = [
        "def policy(state):\n    return (1.0, 0.0)",
        "def policy(state):\n    return (state['gx'], state['gy'])",
        "def policy(state):\n    import math\n    return (math.sin(0.1), 0.5)",
        "def policy(state):\n    return (-0.5, 0.5) if state['vx'] > 0 else (0.5, 0)",
    ]
    m = _compute_metrics(completions, prefix="diversity")
    assert m["diversity/n_unique_completions"] == 4
    assert m["diversity/unique_ratio"] == 1.0
    assert m["diversity/edit_distance/mean"] > 0
    assert m["diversity/length/mean"] > 0
    assert not _collapse_warning(m, prefix="diversity")


def test_metrics_on_collapsed_completions():
    """All identical — the run-5 failure mode."""
    same = "def policy(state):\n    return (state['gx'] - state['px'], 0.0)"
    completions = [same] * 8
    m = _compute_metrics(completions, prefix="diversity")
    assert m["diversity/n_unique_completions"] == 1
    assert m["diversity/unique_ratio"] == 1 / 8
    assert m["diversity/edit_distance/mean"] == 0.0
    assert _collapse_warning(m, prefix="diversity")


def test_metrics_on_near_collapsed_completions():
    """Near-identical with cosmetic variation should still trip the warning."""
    base = "def policy(state):\n    return (state['gx'] - state['px'], 0.0)"
    completions = [base, base, base, base, base + "  # comment", base.replace("0.0", "0")]
    m = _compute_metrics(completions, prefix="diversity")
    assert m["diversity/n_unique_completions"] < len(completions)
    assert m["diversity/edit_distance/mean"] < 5.0
    assert _collapse_warning(m, prefix="diversity")


def test_callback_writes_into_logs():
    cb = DiversityCallback()

    class _StubTrainer:
        _logs = {
            "completion": deque(
                [
                    "def policy(state):\n    return (1.0, 0.0)",
                    "def policy(state):\n    return (-1.0, 0.0)",
                    "def policy(state):\n    return (0.0, 1.0)",
                    "def policy(state):\n    return (0.0, -1.0)",
                ]
            )
        }

    class _StubState:
        global_step = 7

    cb.trainer = _StubTrainer()
    logs = {"train/loss": 0.42}
    cb.on_log(args=None, state=_StubState(), control=None, logs=logs)
    assert logs["train/loss"] == 0.42  # original keys preserved
    assert logs["diversity/n_unique_completions"] == 4
    assert logs["diversity/unique_ratio"] == 1.0
    assert "diversity/edit_distance/mean" in logs
    assert logs["diversity/collapse_warning"] == 0.0


def test_callback_flags_collapse():
    cb = DiversityCallback()
    same = "def policy(state):\n    return (state['gx'], state['gy'])"

    class _StubTrainer:
        _logs = {"completion": deque([same] * 8)}

    class _StubState:
        global_step = 1

    cb.trainer = _StubTrainer()
    logs: dict[str, float] = {}
    cb.on_log(args=None, state=_StubState(), control=None, logs=logs)
    assert logs["diversity/collapse_warning"] == 1.0


def test_callback_emits_per_turn_metrics():
    """When turn_boundaries is stashed, per-turn metrics are also emitted."""
    cb = DiversityCallback()

    completions = [
        "TURN1_AAAA[tool] reward=10TURN2_VARIED_X",
        "TURN1_AAAA[tool] reward=10TURN2_VARIED_Y",
        "TURN1_AAAA[tool] reward=10TURN2_VARIED_Z",
        "TURN1_AAAA[tool] reward=10TURN2_VARIED_W",
    ]
    boundaries = [(9, len(c)) for c in completions]

    class _StubTrainer:
        _logs = {"completion": deque(completions)}
        _latest_rollout_diag = {"turn_boundaries": boundaries}

    class _StubState:
        global_step = 3

    cb.trainer = _StubTrainer()
    logs: dict[str, float] = {}
    cb.on_log(args=None, state=_StubState(), control=None, logs=logs)
    # Turn 1 collapsed (all "TURN1_AAAA")
    assert logs["diversity/turn_1/n_unique_completions"] == 1
    # Turn 2 varied
    assert logs["diversity/turn_2/n_unique_completions"] == 4
    # Global warning trips because turn 1 collapsed
    assert logs["diversity/collapse_warning"] == 1.0


def test_split_completions_by_turn_two_turns():
    completions = ["AAATOOLBBB", "CCCTOOLDDD"]
    boundaries = [(3, 10), (3, 10)]
    per_turn = _split_completions_by_turn(completions, boundaries)
    assert per_turn == [["AAA", "CCC"], ["TOOLBBB", "TOOLDDD"]]


def test_split_completions_by_turn_three_turns():
    """Dynamic-N support: tuple length implies number of turns."""
    completions = ["a1a2a3", "b1b2b3"]
    boundaries = [(2, 4, 6), (2, 4, 6)]
    per_turn = _split_completions_by_turn(completions, boundaries)
    assert per_turn == [["a1", "b1"], ["a2", "b2"], ["a3", "b3"]]


def test_split_completions_by_turn_empty():
    assert _split_completions_by_turn([], []) == []


def test_callback_handles_three_turn_boundaries():
    """Callback emits per-turn metrics for any N inferred from tuple length."""
    cb = DiversityCallback()
    completions = ["A1B1C1", "A2B2C2", "A3B3C3", "A4B4C4"]
    boundaries = [(2, 4, 6)] * 4

    class _StubTrainer:
        _logs = {"completion": deque(completions)}
        _latest_rollout_diag = {"turn_boundaries": boundaries}

    class _StubState:
        global_step = 0

    cb.trainer = _StubTrainer()
    logs: dict[str, float] = {}
    cb.on_log(args=None, state=_StubState(), control=None, logs=logs)
    assert logs["diversity/turn_1/n_unique_completions"] == 4
    assert logs["diversity/turn_2/n_unique_completions"] == 4
    assert logs["diversity/turn_3/n_unique_completions"] == 4


def test_callback_no_completions_is_noop():
    cb = DiversityCallback()

    class _StubTrainer:
        _logs = {"completion": deque()}

    class _StubState:
        global_step = 0

    cb.trainer = _StubTrainer()
    logs = {"train/loss": 0.1}
    cb.on_log(args=None, state=_StubState(), control=None, logs=logs)
    # Untouched
    assert logs == {"train/loss": 0.1}
