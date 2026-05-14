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
    _row_chunks,
    _slice_row_dim,
    _unique_by_task_id,
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


def test_unique_by_task_id_repeated_sampler_pattern():
    """RepeatSampler at num_generations=16 hands us 16 copies of one prompt;
    dedup must collapse that to one row."""
    inputs = [{"task_id": "t0", "prompt": [{"role": "user", "content": "p0"}]}] * 16
    out = _unique_by_task_id(inputs)
    assert len(out) == 1
    assert out[0]["task_id"] == "t0"


def test_unique_by_task_id_multi_prompt_step():
    """If pdtbs > num_generations, inputs contains several distinct prompts;
    dedup preserves one per task_id."""
    inputs = (
        [{"task_id": "t0", "prompt": "p0"}] * 4
        + [{"task_id": "t1", "prompt": "p1"}] * 4
        + [{"task_id": "t2", "prompt": "p2"}] * 4
    )
    out = _unique_by_task_id(inputs)
    assert [row["task_id"] for row in out] == ["t0", "t1", "t2"]


def test_unique_by_task_id_preserves_first_occurrence_order():
    """Order of unique task_ids matches their first appearance, not sort order.
    Matters when downstream metric logging assumes input order."""
    inputs = [
        {"task_id": "z", "prompt": "p_z"},
        {"task_id": "a", "prompt": "p_a"},
        {"task_id": "z", "prompt": "p_z"},
        {"task_id": "m", "prompt": "p_m"},
        {"task_id": "a", "prompt": "p_a"},
    ]
    out = _unique_by_task_id(inputs)
    assert [row["task_id"] for row in out] == ["z", "a", "m"]


def test_unique_by_task_id_single_row():
    inputs = [{"task_id": "t0", "prompt": "p0"}]
    assert _unique_by_task_id(inputs) == inputs


def test_unique_by_task_id_empty():
    assert _unique_by_task_id([]) == []


def test_row_chunks_none_returns_single_full_chunk():
    """`micro_batch_size=None` keeps pre-change behaviour: one [0, B) chunk."""
    assert _row_chunks(16, None) == [(0, 16)]
    assert _row_chunks(1, None) == [(0, 1)]


def test_row_chunks_size_at_or_above_rows_returns_single_chunk():
    """When mb >= rows, the bound caps to rows — equivalent to mb=None."""
    assert _row_chunks(4, 4) == [(0, 4)]
    assert _row_chunks(4, 100) == [(0, 4)]


def test_row_chunks_evenly_divisible():
    """Even split into equal-sized chunks."""
    assert _row_chunks(16, 4) == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert _row_chunks(8, 2) == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_row_chunks_uneven_last_chunk():
    """Last chunk is shorter when rows is not divisible by mb."""
    assert _row_chunks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert _row_chunks(5, 2) == [(0, 2), (2, 4), (4, 5)]


def test_row_chunks_size_one():
    """`mb=1` yields one chunk per row — maximal split."""
    assert _row_chunks(4, 1) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_row_chunks_zero_rows():
    """Empty input yields no chunks — defensive."""
    assert _row_chunks(0, None) == []
    assert _row_chunks(0, 4) == []


def test_row_chunks_covers_every_row_exactly_once():
    """Property check: across all chunks, indices cover [0, B) with no gaps
    or overlaps. The micro-batched gradient identity relies on this."""
    for rows in [1, 4, 5, 7, 16]:
        for mb in [None, 1, 2, 3, 4, rows, rows + 5]:
            chunks = _row_chunks(rows, mb)
            covered = [i for s, e in chunks for i in range(s, e)]
            assert covered == list(range(rows)), (
                f"rows={rows}, mb={mb} → {chunks} covers {covered}"
            )


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
