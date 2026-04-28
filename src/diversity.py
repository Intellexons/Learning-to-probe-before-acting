"""Per-step diversity metrics for GRPO training.

Run 5 collapsed silently to a single byte-identical policy and we did not
notice for a week because token-level entropy stayed healthy. This callback
inspects completions directly and surfaces unique-count, edit-distance, and
length statistics so collapse is visible in real time on the W&B dashboard.

When a multi-turn rollout_func stashes per-trajectory `_turn_boundaries` on
the trainer, the callback also emits per-turn variants of the same metrics
so we can tell whether any individual turn collapsed independently.
"""
from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from difflib import SequenceMatcher

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# Thresholds for the collapse-warning heuristic. First-pass values; revisit
# empirically once the callback has been logging for a few runs.
#  - unique_ratio < 0.5  : less than half the batch is distinct
#  - edit_distance/length < 0.05 (i.e. < 5% character-level difference on
#    average) : near-byte-identical content. The fraction-of-length form is
#    used instead of an absolute char count so short and long completions
#    are treated consistently.
COLLAPSE_UNIQUE_RATIO = 0.5
COLLAPSE_EDIT_FRACTION = 0.05

# Cap on pairwise edit-distance comparisons per batch. Pairwise scales as
# O(B^2); 200 is comfortable for B<=20 (full coverage) and bounds cost at
# larger B by sampling. The cap doesn't bias the *mean* much because we
# enumerate pairs in sorted order, but if we ever care, swap to random
# sampling.
MAX_PAIRWISE_COMPARISONS = 200


# ---------------------------------------------------------------------------
# Pure-function metrics
# ---------------------------------------------------------------------------


def _pstdev_safe(values: Sequence[float]) -> float:
    """Population stdev with len<2 guard."""
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _pairwise_edit_distances(
    strings: list[str],
    max_pairs: int = MAX_PAIRWISE_COMPARISONS,
) -> list[int]:
    """Pairwise Levenshtein-style distance via difflib (no extra dep).

    Caps total comparisons at `max_pairs` to keep this cheap on larger batches.
    `SequenceMatcher.ratio` gives a similarity score; we convert to an
    integer-valued distance via `len_avg * (1 - ratio)` rounded down. This is
    not strict Levenshtein but is monotone with it and good enough for "are
    completions essentially identical."
    """
    if len(strings) < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            pairs.append((i, j))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    distances: list[int] = []
    for i, j in pairs:
        a, b = strings[i], strings[j]
        if not a and not b:
            distances.append(0)
            continue
        ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
        avg_len = (len(a) + len(b)) / 2
        distances.append(int(round(avg_len * (1.0 - ratio))))
    return distances


def _compute_metrics(completions: list[str], prefix: str = "diversity") -> dict[str, float]:
    """Compute the diversity metric set for a list of strings."""
    cleaned = [c.strip() for c in completions]
    if not cleaned:
        return {}

    n_unique = len(set(cleaned))
    distances = _pairwise_edit_distances(cleaned)
    lengths = [len(c) for c in cleaned]

    return {
        f"{prefix}/n_unique_completions": float(n_unique),
        f"{prefix}/unique_ratio": n_unique / len(cleaned),
        f"{prefix}/length/mean": statistics.fmean(lengths),
        f"{prefix}/length/std": _pstdev_safe(lengths),
        f"{prefix}/edit_distance/mean": statistics.fmean(distances) if distances else 0.0,
        f"{prefix}/edit_distance/std": _pstdev_safe(distances),
    }


def _collapse_warning(metrics: dict[str, float], prefix: str) -> bool:
    """True if the prefix's metrics indicate a collapsed batch."""
    n_unique = metrics.get(f"{prefix}/n_unique_completions")
    unique_ratio = metrics.get(f"{prefix}/unique_ratio")
    edit_mean = metrics.get(f"{prefix}/edit_distance/mean")
    length_mean = metrics.get(f"{prefix}/length/mean")
    if n_unique is not None and n_unique <= 1:
        return True
    if unique_ratio is not None and unique_ratio < COLLAPSE_UNIQUE_RATIO:
        return True
    if (
        edit_mean is not None
        and length_mean is not None
        and length_mean > 0
        and (edit_mean / length_mean) < COLLAPSE_EDIT_FRACTION
    ):
        return True
    return False


def _split_completions_by_turn(
    completions: list[str],
    boundaries: list[Sequence[int]],
) -> list[list[str]]:
    """Slice each completion text into per-turn segments using char offsets.

    `boundaries[t]` is a sequence of cumulative char-end positions for
    completion t — boundaries[t][k] is where turn k+1 ends. Turn k spans
    `[boundaries[t][k-1], boundaries[t][k])`, with turn 1 starting at 0.

    Returns a list-of-lists indexed by turn, each holding the per-completion
    text slice for that turn. The N=2 case has `boundaries[t] = (turn1_end,
    turn2_end)` and produces `[[turn1 texts...], [turn2 texts...]]`.
    """
    if not boundaries:
        return []
    n_turns = len(boundaries[0])
    per_turn: list[list[str]] = [[] for _ in range(n_turns)]
    for text, ends in zip(completions, boundaries):
        starts = (0,) + tuple(ends[:-1])
        for k, (start, end) in enumerate(zip(starts, ends)):
            per_turn[k].append(text[start:end])
    return per_turn


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------


class DiversityCallback(TrainerCallback):
    """TrainerCallback that emits completion-level diversity metrics on each log.

    The callback needs a backreference to the trainer to read `trainer._logs`
    (TRL's GRPOTrainer keeps a ring buffer of recent completions there). Set
    it after constructing the trainer:

        cb = DiversityCallback()
        cb.trainer = trainer
        trainer.add_callback(cb)

    On each `on_log` event the callback reads the most recent batch of
    completions, computes diversity metrics, and stuffs them into the
    `logs` dict (which transformers.Trainer pushes into `state.log_history`).
    The orchestrator forwards log_history to W&B, so the metrics appear on
    the dashboard alongside `train/reward`, `train/loss`, etc.

    Multi-turn support: if `trainer._latest_rollout_diag` exposes a
    `turn_boundaries` field — a list of cumulative-char-end tuples aligned
    with the completion list — per-turn variants are also emitted. The
    number of turns is inferred from the tuple length, so this scales
    beyond N=2 with no code change.
    """

    def __init__(self) -> None:
        self.trainer = None  # set by caller after trainer construction

    def _read_completions(self) -> list[str]:
        if self.trainer is None:
            return []
        logs = getattr(self.trainer, "_logs", None)
        if not logs:
            return []
        completions = logs.get("completion")
        if completions is None:
            return []
        return list(completions)

    def _read_turn_boundaries(self) -> list[Sequence[int]] | None:
        """Per-trajectory cumulative char offsets (one tuple per completion).

        Set by a custom rollout on the trainer for multi-turn runs. Tuple
        length implies the number of turns: `(turn1_end,)` for N=1,
        `(turn1_end, turn2_end)` for N=2, etc. Tool-result tokens between
        turns are excluded by the rollout when it computes these offsets.
        """
        if self.trainer is None:
            return None
        diag = getattr(self.trainer, "_latest_rollout_diag", None)
        if not diag:
            return None
        return diag.get("turn_boundaries")

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if logs is None:
            return
        completions = self._read_completions()
        if not completions:
            return

        global_metrics = _compute_metrics(completions, prefix="diversity")
        logs.update(global_metrics)

        warnings_raised: list[str] = []
        if _collapse_warning(global_metrics, prefix="diversity"):
            warnings_raised.append("global")

        boundaries = self._read_turn_boundaries()
        if boundaries and len(boundaries) == len(completions):
            per_turn_texts = _split_completions_by_turn(completions, boundaries)
            for turn_idx, turn_texts in enumerate(per_turn_texts, start=1):
                prefix = f"diversity/turn_{turn_idx}"
                turn_metrics = _compute_metrics(turn_texts, prefix=prefix)
                logs.update(turn_metrics)
                if _collapse_warning(turn_metrics, prefix=prefix):
                    warnings_raised.append(f"turn_{turn_idx}")

        logs["diversity/collapse_warning"] = 1.0 if warnings_raised else 0.0
        if warnings_raised:
            logger.warning(
                "Diversity collapse detected at step %s: scopes=%s, "
                "n_unique=%s/%d, unique_ratio=%.2f, edit_distance/mean=%.2f",
                state.global_step,
                ",".join(warnings_raised),
                int(global_metrics["diversity/n_unique_completions"]),
                len(completions),
                global_metrics["diversity/unique_ratio"],
                global_metrics["diversity/edit_distance/mean"],
            )
