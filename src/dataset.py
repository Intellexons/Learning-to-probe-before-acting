import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def merge_samples_and_scores(
    samples_path: str,
    scores_path: str,
) -> list[dict]:
    """Join samples and scores by (task_id, sample_id)."""
    samples = load_jsonl(samples_path)
    scores = load_jsonl(scores_path)

    score_map = {}
    for s in scores:
        key = (s["task_id"], s["sample_id"])
        score_map[key] = s

    merged = []
    for sample in samples:
        key = (sample["task_id"], sample["sample_id"])
        score = score_map.get(key, {})
        merged.append({**sample, **score})
    return merged


def build_dpo_pairs(
    merged: list[dict],
    prompt_map: dict[str, str],
    min_chosen_reward: float = 1.0,
    pairs_per_task: int = 2,
) -> list[dict]:
    """Build DPO preference pairs from scored samples.

    Args:
        merged: list of dicts with task_id, sample_id, code, reward_total, prompt
        prompt_map: {task_id: rendered_prompt_text}
        min_chosen_reward: skip tasks where best reward is below this
        pairs_per_task: max pairs per task

    Returns:
        list of dicts in TRL conversational format
    """
    from src.prompts import format_dpo_example

    # Group by task_id
    by_task: dict[str, list[dict]] = {}
    for item in merged:
        tid = item["task_id"]
        by_task.setdefault(tid, []).append(item)

    pairs = []
    for task_id, items in by_task.items():
        # Sort by reward descending
        items.sort(key=lambda x: x.get("reward_total", 0.0), reverse=True)

        if len(items) < 2:
            logger.warning(f"Task {task_id}: fewer than 2 samples, skipping")
            continue

        best = items[0]
        best_reward = best.get("reward_total", 0.0)

        if best_reward <= min_chosen_reward:
            logger.warning(
                f"Task {task_id}: best reward {best_reward:.2f} <= threshold "
                f"{min_chosen_reward}, skipping"
            )
            continue

        prompt_text = prompt_map.get(task_id, best.get("prompt", ""))
        created = 0

        # Pair: best vs median
        median_idx = len(items) // 2
        if median_idx > 0 and created < pairs_per_task:
            median = items[median_idx]
            pair = format_dpo_example(prompt_text, best["code"], median["code"])
            pairs.append(pair)
            created += 1

        # Pair: best vs worst
        if created < pairs_per_task and len(items) > 2:
            worst = items[-1]
            pair = format_dpo_example(prompt_text, best["code"], worst["code"])
            pairs.append(pair)
            created += 1

    logger.info(f"Built {len(pairs)} DPO pairs from {len(by_task)} tasks")
    return pairs
