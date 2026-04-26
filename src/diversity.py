"""Per-step diversity metrics for GRPO training.

Run 5 collapsed silently to a single byte-identical policy and we did not
notice for a week because token-level entropy stayed healthy. This callback
inspects completions directly and surfaces unique-count, edit-distance, and
length statistics so collapse is visible in real time on the W&B dashboard.

When a multi-turn rollout_func stashes per-trajectory `_turn_boundaries` on
the trainer, the callback also emits per-turn variants of the same metrics
so we can tell whether turn 1 or turn 2 collapsed independently.
"""
from __future__ import annotations

import logging
import statistics
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


def _pairwise_edit_distances(strings: list[str], max_pairs: int = 200) -> list[int]:
    """Pairwise Levenshtein-style distance via difflib (no extra dep).

    Caps total comparisons at `max_pairs` to keep this cheap on larger batches.
    `SequenceMatcher.ratio` gives a similarity score; we convert to an
    integer-valued distance via `len_avg * (1 - ratio)` rounded down. This is
    not strict Levenshtein but is monotone with it and good enough for "are
    completions essentially identical."
    """
    n = len(strings)
    if n < 2:
        return []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    distances = []
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
    n = len(cleaned)
    if n == 0:
        return {}

    n_unique = len(set(cleaned))
    distances = _pairwise_edit_distances(cleaned)
    lengths = [len(c) for c in cleaned]

    metrics: dict[str, float] = {
        f"{prefix}/n_unique_completions": float(n_unique),
        f"{prefix}/unique_ratio": n_unique / n,
        f"{prefix}/length/mean": statistics.fmean(lengths) if lengths else 0.0,
        f"{prefix}/length/std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
    }
    if distances:
        metrics[f"{prefix}/edit_distance/mean"] = statistics.fmean(distances)
        metrics[f"{prefix}/edit_distance/std"] = (
            statistics.pstdev(distances) if len(distances) > 1 else 0.0
        )
    else:
        metrics[f"{prefix}/edit_distance/mean"] = 0.0
        metrics[f"{prefix}/edit_distance/std"] = 0.0
    return metrics


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
    `turn_boundaries` field (list of (turn1_end, turn2_end) char offsets
    aligned with the completion list), per-turn variants are also emitted.
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

    def _read_turn_boundaries(self) -> list[tuple[int, int]] | None:
        """Optional per-trajectory char offsets: (turn1_end_idx, turn2_end_idx).

        Set by a custom rollout_func on the trainer for multi-turn runs. Each
        tuple is (where turn 1 text ends, where turn 2 text ends) in the
        decoded completion string. The slice [0:turn1_end] is turn 1, and
        [turn1_end:turn2_end] is turn 2 (the tool-result block in between
        belongs to neither and is excluded).
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

        warnings_raised = []
        if _collapse_warning(global_metrics, prefix="diversity"):
            warnings_raised.append("global")

        boundaries = self._read_turn_boundaries()
        if boundaries and len(boundaries) == len(completions):
            turn1_texts: list[str] = []
            turn2_texts: list[str] = []
            for text, (t1_end, t2_end) in zip(completions, boundaries):
                turn1_texts.append(text[:t1_end])
                turn2_texts.append(text[t1_end:t2_end])
            for turn_idx, turn_texts in [(1, turn1_texts), (2, turn2_texts)]:
                turn_metrics = _compute_metrics(
                    turn_texts, prefix=f"diversity/turn_{turn_idx}"
                )
                logs.update(turn_metrics)
                if _collapse_warning(turn_metrics, prefix=f"diversity/turn_{turn_idx}"):
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
