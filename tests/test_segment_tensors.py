"""Tests for the per-segment tensor builder used by the two-pass loss path.

Validates that `build_segment_tensors` produces the right shapes, padding,
and content alignment for the two `super()._compute_loss` calls in
BranchedGRPOTrainer.compute_loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.branched_trainer import (  # noqa: E402
    _Trajectory,
    _slice_row_dim,
    build_segment_tensors,
)


def _close(actual: torch.Tensor, expected: list) -> None:
    """Assert a float tensor matches `expected` (nested python list)."""
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-5, atol=1e-6,
    )


PAD = 99
DEVICE = torch.device("cpu")


def _make_traj(
    *,
    prompt_ids: list[int],
    turn1_ids: list[int],
    tool_ids: list[int],
    turn2_ids: list[int],
    a_turn1: float,
    a_turn2: float,
    i: int = 0,
    j: int = 0,
) -> _Trajectory:
    """Build a _Trajectory with just the fields the segment-tensor builder reads."""
    return _Trajectory(
        i=i, j=j,
        prompt_ids=prompt_ids,
        turn1_ids=turn1_ids,
        tool_ids=tool_ids,
        turn2_ids=turn2_ids,
        turn1_text="", turn2_text="",
        turn1_mode="policy", turn2_mode="policy",
        reward=0.0,
        a_turn1=a_turn1, a_turn2=a_turn2,
    )


def test_uniform_lengths_no_padding():
    """When all rows have identical segment lengths, no padding is added."""
    trajs = [
        _make_traj(prompt_ids=[1, 2, 3], turn1_ids=[10, 11], tool_ids=[20, 21],
                   turn2_ids=[30, 31], a_turn1=0.5, a_turn2=-0.3),
        _make_traj(prompt_ids=[1, 2, 3], turn1_ids=[12, 13], tool_ids=[22, 23],
                   turn2_ids=[32, 33], a_turn1=0.7, a_turn2=0.1),
    ]
    out = build_segment_tensors(trajs, DEVICE, PAD)

    assert out["turn1_completion_ids"].tolist() == [[10, 11], [12, 13]]
    assert out["turn1_completion_mask"].tolist() == [[1, 1], [1, 1]]
    _close(out["turn1_advantages"], [[0.5, 0.5], [0.7, 0.7]])

    # pass2 prompt = [prompt | turn1 | tool]
    assert out["pass2_prompt_ids"].tolist() == [
        [1, 2, 3, 10, 11, 20, 21],
        [1, 2, 3, 12, 13, 22, 23],
    ]
    assert out["pass2_prompt_mask"].tolist() == [[1] * 7, [1] * 7]

    assert out["turn2_completion_ids"].tolist() == [[30, 31], [32, 33]]
    assert out["turn2_completion_mask"].tolist() == [[1, 1], [1, 1]]
    _close(out["turn2_advantages"], [[-0.3, -0.3], [0.1, 0.1]])


def test_turn1_right_padding():
    """Shorter turn1 is right-padded; mask zeros out pad positions and the
    advantage tensor is zero on pad (so it can't leak into the loss)."""
    trajs = [
        _make_traj(prompt_ids=[1], turn1_ids=[10, 11, 12], tool_ids=[20],
                   turn2_ids=[30], a_turn1=0.5, a_turn2=0.0),
        _make_traj(prompt_ids=[1], turn1_ids=[13], tool_ids=[20],
                   turn2_ids=[30], a_turn1=0.9, a_turn2=0.0),
    ]
    out = build_segment_tensors(trajs, DEVICE, PAD)

    assert out["turn1_completion_ids"].tolist() == [[10, 11, 12], [13, PAD, PAD]]
    assert out["turn1_completion_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    # Advantage zeroed on pad positions (multiplied by mask).
    _close(out["turn1_advantages"], [[0.5, 0.5, 0.5], [0.9, 0.0, 0.0]])


def test_turn2_right_padding():
    """Shorter turn2 is right-padded similarly."""
    trajs = [
        _make_traj(prompt_ids=[1], turn1_ids=[10], tool_ids=[20],
                   turn2_ids=[30, 31, 32], a_turn1=0.0, a_turn2=-0.4),
        _make_traj(prompt_ids=[1], turn1_ids=[10], tool_ids=[20],
                   turn2_ids=[33], a_turn1=0.0, a_turn2=0.6),
    ]
    out = build_segment_tensors(trajs, DEVICE, PAD)

    assert out["turn2_completion_ids"].tolist() == [[30, 31, 32], [33, PAD, PAD]]
    assert out["turn2_completion_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    _close(out["turn2_advantages"], [[-0.4, -0.4, -0.4], [0.6, 0.0, 0.0]])


def test_pass2_prompt_left_padding():
    """The pass-2 prompt is [prompt | turn1 | tool] left-padded.

    Left-padding ensures every row's completion (turn2) starts at the same
    absolute position, which is what makes
    `logits_to_keep = T_turn2_max` pick the right tokens for every row in
    the parent's `_compute_loss`.
    """
    trajs = [
        # Row 0: long prefix (prompt=2 + turn1=2 + tool=2 = 6)
        _make_traj(prompt_ids=[1, 2], turn1_ids=[10, 11], tool_ids=[20, 21],
                   turn2_ids=[30], a_turn1=0.0, a_turn2=0.0),
        # Row 1: short prefix (prompt=1 + turn1=1 + tool=1 = 3)
        _make_traj(prompt_ids=[1], turn1_ids=[10], tool_ids=[20],
                   turn2_ids=[30], a_turn1=0.0, a_turn2=0.0),
    ]
    out = build_segment_tensors(trajs, DEVICE, PAD)

    # Row 0 prefix is 6 tokens, row 1 is 3. Common length is 6; row 1 left-
    # padded by 3.
    assert out["pass2_prompt_ids"].tolist() == [
        [1, 2, 10, 11, 20, 21],
        [PAD, PAD, PAD, 1, 10, 20],
    ]
    assert out["pass2_prompt_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]


def test_advantage_broadcasts_per_row():
    """Each row's scalar `a_turn1` / `a_turn2` is broadcast across that row's
    actual segment positions only — never leaks into other rows or pad."""
    trajs = [
        _make_traj(prompt_ids=[1], turn1_ids=[10, 11], tool_ids=[20],
                   turn2_ids=[30, 31], a_turn1=2.0, a_turn2=3.0),
        _make_traj(prompt_ids=[1], turn1_ids=[12, 13, 14], tool_ids=[20],
                   turn2_ids=[32, 33, 34], a_turn1=-2.0, a_turn2=-3.0),
    ]
    out = build_segment_tensors(trajs, DEVICE, PAD)

    # Turn1: row 0 has 2 turn1 tokens (padded to 3), row 1 has 3.
    _close(out["turn1_advantages"], [[2.0, 2.0, 0.0], [-2.0, -2.0, -2.0]])
    # Turn2 mirrors structure.
    _close(out["turn2_advantages"], [[3.0, 3.0, 0.0], [-3.0, -3.0, -3.0]])


def test_slice_row_dim_slices_batched_tensors():
    """`_slice_row_dim` slices every batched tensor along dim 0."""
    seg = {
        "prompt_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "completion_ids": torch.tensor([[10], [20], [30], [40]]),
        "advantages": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    }
    out = _slice_row_dim(seg, 1, 3)

    assert out["prompt_ids"].tolist() == [[3, 4], [5, 6]]
    assert out["completion_ids"].tolist() == [[20], [30]]
    assert out["advantages"].tolist() == [[2.0], [3.0]]


def test_slice_row_dim_preserves_scalar_normalizer():
    """0-dim tensors (e.g. num_items_in_batch) must pass through unchanged.

    This is the invariant that makes micro-batched losses sum to the
    single-pass loss: every chunk divides by the same global token count.
    """
    seg = {
        "completion_ids": torch.tensor([[10], [20], [30]]),
        "num_items_in_batch": torch.tensor(42),
        "scalar_float": torch.tensor(3.14),
    }
    out = _slice_row_dim(seg, 0, 2)

    assert out["completion_ids"].tolist() == [[10], [20]]
    assert out["num_items_in_batch"].item() == 42
    assert out["scalar_float"].item() == torch.tensor(3.14).item()


def test_slice_row_dim_passes_through_non_tensors():
    """Non-tensor values (None, ints, etc.) pass through unchanged."""
    seg = {
        "completion_ids": torch.tensor([[10], [20]]),
        "old_per_token_logps": None,
        "some_int": 7,
    }
    out = _slice_row_dim(seg, 0, 1)

    assert out["completion_ids"].tolist() == [[10]]
    assert out["old_per_token_logps"] is None
    assert out["some_int"] == 7


def test_dtypes_and_devices():
    """Token tensors are long, masks are long, advantages are float32."""
    trajs = [
        _make_traj(prompt_ids=[1], turn1_ids=[10], tool_ids=[20],
                   turn2_ids=[30], a_turn1=0.5, a_turn2=-0.5),
    ]
    out = build_segment_tensors(trajs, DEVICE, PAD)

    assert out["turn1_completion_ids"].dtype == torch.long
    assert out["turn1_completion_mask"].dtype == torch.long
    assert out["turn1_advantages"].dtype == torch.float32
    assert out["pass2_prompt_ids"].dtype == torch.long
    assert out["pass2_prompt_mask"].dtype == torch.long
    assert out["turn2_advantages"].dtype == torch.float32
    for t in out.values():
        assert t.device == DEVICE
