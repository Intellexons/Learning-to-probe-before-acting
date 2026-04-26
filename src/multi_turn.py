"""Branched-rollout (VinePPO-style) helpers for N=2 multi-turn GRPO.

The training-loop trick designed in the 2026-04-26 conversation:

  For each prompt, generate G_1 first-turn variants. For each first-turn
  variant i, execute the tool (run policy or capture main stdout) to get
  the tool-result message, then generate G_2 second-turn continuations.
  Each second-turn continuation must be a `policy` block — anything else
  is rejected and scored at the error floor. The final reward of the
  trajectory (i, j) is the policy reward of turn 2.

  Empirical values (no learned critic, all MC):
    V̂_0     = grand mean over all G_1 G_2 final rewards (initial-state baseline)
    V̂_1^(i) = mean over the G_2 continuations of first-turn variant i

  Per-turn advantages broadcast to per-token advantages of shape (B, T):
    A^(1)_i   = V̂_1^(i) - V̂_0     applied to turn-1 tokens of trajectory (i, j)
    A^(2)_{i,j} = R_{i,j} - V̂_1^(i)  applied to turn-2 tokens

  Tool-result tokens (the user-role message TRL inserts between turn 1 and
  turn 2) are masked out via tool_mask so they do not contribute to the loss.

This module holds the pure functions for that math + the chat-message helpers
that aren't tied to the trainer subclass. The trainer-side I/O (model.generate
calls, tokenization, padding) lives in `src/branched_trainer.py`.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.runner import (
    ERROR_PENALTY_PER_SCENARIO,
    extract_code_with_mode,
    score_main_turn,
    score_policy_turn,
)


# Reward floor for an invalid last turn (mode != "policy"). Matches the
# per-scenario error-floor convention so a fully-rejected last turn scores
# the same as a fully-errored policy. The trainer multiplies by n_scenarios
# at construction time so the magnitude tracks task size.
def error_floor_reward(n_scenarios: int) -> float:
    return ERROR_PENALTY_PER_SCENARIO * n_scenarios


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a single turn's code.

    `mode` is the parsed code-fence tag ("policy", "main", or "invalid").
    `reward` is the policy-reward total (only meaningful when mode=="policy").
    `text` is the body of the user-role message that gets fed back to the
    model as the next-turn prompt context.
    """
    mode: str
    reward: float | None
    text: str


def execute_turn(
    raw_completion: str,
    task: dict,
    *,
    is_last_turn: bool,
    timeout_policy: int = 30,
    timeout_main: int = 10,
    max_steps: int = 500,
) -> ToolResult:
    """Run the code from one turn's completion. Returns a ToolResult.

    On the **last turn**, only `policy` mode is allowed — `main` and
    `invalid` are both scored at the error floor and surfaced as a
    rejection message in the chat. On non-last turns, all three modes are
    valid: policy yields a reward, main yields stdout, invalid yields a
    rejection message AND a floor-reward penalty so the model still gets
    a learning signal not to emit malformed turns mid-trajectory (the
    advantage at turn 1 will reflect that).

    For non-last `policy` and `main` turns the returned `reward` is None
    (it has no direct meaning — turn-1 credit comes from the branched
    V̂_1 estimate, not from the immediate tool result).
    """
    n_scenarios = len(task.get("scenarios", []))
    floor = error_floor_reward(max(n_scenarios, 1))

    mode, code = extract_code_with_mode(raw_completion)

    if is_last_turn:
        if mode != "policy":
            return ToolResult(
                mode=mode,
                reward=floor,
                text=(
                    f"[tool] turn rejected: final turn must be a `policy` block "
                    f"(got mode={mode!r}). Trajectory scored at error floor."
                ),
            )
        score = score_policy_turn(
            code, task, timeout=timeout_policy, max_steps=max_steps
        )
        return ToolResult(
            mode="policy",
            reward=score["reward_total"],
            text=(
                f"[tool] policy executed across {n_scenarios} scenarios. "
                f"total_reward = {score['reward_total']:.2f}"
            ),
        )

    # Non-final turn: every mode produces a tool message; only invalid
    # carries a forward reward signal (a floor penalty that propagates
    # through the V̂_1 estimate and turn-1 advantage).
    if mode == "policy":
        score = score_policy_turn(
            code, task, timeout=timeout_policy, max_steps=max_steps
        )
        return ToolResult(
            mode="policy",
            reward=None,
            text=(
                f"[tool] policy executed across {n_scenarios} scenarios. "
                f"total_reward = {score['reward_total']:.2f}"
            ),
        )
    if mode == "main":
        out = score_main_turn(code, timeout=timeout_main)
        return ToolResult(
            mode="main",
            reward=None,
            text=f"[tool] main() executed. stdout:\n{out['stdout']}",
        )
    return ToolResult(
        mode="invalid",
        reward=floor,
        text=(
            "[tool] turn rejected: code did not declare a recognized mode "
            "(expected ```policy or ```main fence)."
        ),
    )


def compute_branched_advantages(
    rewards: list[list[float]],
) -> tuple[float, list[float], list[list[float]], list[list[float]]]:
    """Given a (G_1, G_2) reward grid, return the empirical values and
    per-turn advantages.

    Returns:
      V0:  scalar — grand mean reward (initial-state value estimate)
      V1:  list of G_1 floats — mean reward across G_2 continuations of
           each first-turn variant (value-after-turn-1 estimate)
      A_turn1: G_1 × G_2 list of floats — V1[i] - V0   (broadcast to all j)
      A_turn2: G_1 × G_2 list of floats — rewards[i][j] - V1[i]

    No std-normalization here — the trainer applies any DAPO-style scaling
    on the final assembled (B, T) tensor if `scale_rewards` is configured.
    """
    if not rewards or not rewards[0]:
        raise ValueError("Empty reward grid")
    g1 = len(rewards)
    g2 = len(rewards[0])
    if any(len(row) != g2 for row in rewards):
        raise ValueError(f"Reward grid is ragged: rows have unequal length")

    flat = [r for row in rewards for r in row]
    v0 = statistics.fmean(flat)
    v1 = [statistics.fmean(row) for row in rewards]
    a_turn1 = [[v1[i] - v0 for _ in range(g2)] for i in range(g1)]
    a_turn2 = [[rewards[i][j] - v1[i] for j in range(g2)] for i in range(g1)]
    return v0, v1, a_turn1, a_turn2


def assemble_per_token_advantages(
    turn1_len: int,
    tool_len: int,
    turn2_len: int,
    a_turn1: float,
    a_turn2: float,
) -> list[float]:
    """Build the (T,) per-token advantage vector for one trajectory.

    Layout: [turn1_tokens (a_turn1)] + [tool_tokens (0.0)] + [turn2_tokens (a_turn2)].
    Tool tokens are zeroed AND additionally masked via the tool_mask in the
    GRPO loss, so the 0.0 advantage here is just defensive — the mask is the
    primary mechanism preventing tool tokens from contributing to the loss.
    """
    return (
        [a_turn1] * turn1_len
        + [0.0] * tool_len
        + [a_turn2] * turn2_len
    )


def assemble_tool_mask(
    turn1_len: int,
    tool_len: int,
    turn2_len: int,
) -> list[int]:
    """Build the (T,) tool_mask for one trajectory.

    1 = model-generated token (loss applies), 0 = tool-result token (skip).
    Matches TRL's convention at `grpo_trainer.py:1473-1475` — the field is
    treated as "model tokens (1) vs external tokens (0)" inside _compute_loss.
    """
    return (
        [1] * turn1_len
        + [0] * tool_len
        + [1] * turn2_len
    )


def format_tool_message(text: str) -> dict:
    """Wrap a tool-result text into a chat message.

    We use role="user" (not "tool") because Qwen2.5's chat template doesn't
    have a dedicated tool-role pathway and we don't want to register an
    actual tool-calling schema. The `[tool]` static prefix on the content
    (added by `execute_turn` already) gives the model a stylistic marker
    so it can recognize tool output without needing a schema change.
    """
    return {"role": "user", "content": text}
