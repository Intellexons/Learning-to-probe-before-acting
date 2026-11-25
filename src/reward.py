"""Reward function for GRPO training.

Bridges TRL's GRPOTrainer reward interface to the existing scoring pipeline.
Single-turn training gives the model one reply, so the reply has to hold the
final run: it is read by the same `execute_turn` the agent loop uses, and a
reply without a `final_run` call scores the error floor.
"""
import logging

from src.multi_turn import execute_turn

logger = logging.getLogger(__name__)


def make_reward_fn(task_map: dict, scoring_cfg: dict):
    """Create a reward function compatible with TRL's GRPOTrainer.

    Args:
        task_map: {task_id: task_dict} for looking up env_code/scenarios.
        scoring_cfg: {"timeout_seconds": int, "max_steps": int}.

    Returns:
        Callable with signature (completions, task_id, **kwargs) -> list[float].
        The returned function exposes a `.stats` list that records
        (reward, ran_successfully) tuples for every completion it scores,
        so callers can aggregate per-iteration progress after training.
    """
    timeout = scoring_cfg["timeout_seconds"]
    max_steps = scoring_cfg["max_steps"]

    def reward_fn(completions, task_id, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            # Extract text from conversational format
            if isinstance(completion, list):
                text = completion[0]["content"] if completion else ""
            else:
                text = completion

            tid = task_id[i] if i < len(task_id) else None
            if tid is None or tid not in task_map:
                logger.warning(f"Unknown task_id {tid}, assigning reward 0.0")
                rewards.append(0.0)
                reward_fn.stats.append((0.0, False))
                continue

            task = task_map[tid]
            result = execute_turn(text, task, is_last_turn=True,
                                  timeout_policy=timeout, max_steps=max_steps).score
            rewards.append(result["reward_total"])
            reward_fn.stats.append(
                (result["reward_total"], bool(result.get("ran_successfully", False)))
            )

        return rewards

    reward_fn.stats = []
    return reward_fn
