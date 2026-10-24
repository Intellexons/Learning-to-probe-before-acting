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
    DEFAULT_ENV_CLASS,
    DEFAULT_MAX_STEPS,
    ERROR_PENALTY_PER_SCENARIO,
    MAIN_DEFAULT_TIMEOUT,
    POLICY_DEFAULT_TIMEOUT,
    extract_code_with_mode,
    score_main_turn,
    score_policy_turn,
)

# Mode constants — defined here once so callers don't sprinkle string literals.
MODE_POLICY = "policy"
MODE_MAIN = "main"
MODE_INVALID = "invalid"


# ---------------------------------------------------------------------------
# Reward floor + tool-message formatters
# ---------------------------------------------------------------------------


def error_floor_reward(n_scenarios: int) -> float:
    """Reward floor for an invalid last turn (mode != "policy") or a fully
    errored policy. Matches the per-scenario error-floor convention so a
    fully-rejected last turn scores the same as a fully-errored policy."""
    return ERROR_PENALTY_PER_SCENARIO * n_scenarios


def _policy_tool_text(reward_total: float, n_scenarios: int) -> str:
    return (
        f"Turn-1 policy was scored across {n_scenarios} scenarios. "
        f"total_reward = {reward_total:.2f}"
    )


def _main_tool_text(stdout: str) -> str:
    """Body of the tool-result message for an intermediate `# MODE: main` turn.

    Returns the captured stdout verbatim (or "[empty output]" if blank), with
    no extra prefix — eval (`scripts/eval_frozen_baseline.py`) wraps the body
    with `[Tool output from turn 1]\n...` at message-construction time and
    training does the same in `BranchedGRPOTrainer._branched_rollout_one_prompt`.
    """
    return stdout or "[empty output]"


def _final_turn_rejected_text(mode: str) -> str:
    return (
        f"Final-turn rejected: final turn must start with `# MODE: policy` "
        f"(got mode={mode!r}). Trajectory scored at error floor."
    )


# When turn 1 emits no recognized mode directive, eval surfaces this exact
# string (`scripts/eval_frozen_baseline.py:195`); training mirrors it.
_INTERMEDIATE_INVALID_TEXT = "[turn 1 produced no recognized code block]"


# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------


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


def _score_policy(
    code: str, task: dict, timeout: int, max_steps: int,
) -> tuple[float, str]:
    """Score a policy turn → (reward_total, tool_text)."""
    n_scenarios = len(task.get("scenarios", []))
    score = score_policy_turn(code, task, timeout=timeout, max_steps=max_steps)
    return score["reward_total"], _policy_tool_text(score["reward_total"], n_scenarios)


def execute_turn(
    raw_completion: str,
    task: dict,
    *,
    is_last_turn: bool,
    scenario: dict | None = None,
    timeout_policy: int = POLICY_DEFAULT_TIMEOUT,
    timeout_main: int = MAIN_DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ToolResult:
    """Run the code from one turn's completion and return a ToolResult.

    On the **last turn**, only `policy` mode is allowed — `main` and
    `invalid` are scored at the error floor and surfaced as a rejection
    message in the chat. On non-last turns, all three modes are valid:
    policy yields a reward, main yields stdout, invalid yields a rejection
    message AND a floor-reward penalty so the model still gets a learning
    signal not to emit malformed turns mid-trajectory (the advantage at
    turn 1 will reflect that).

    For non-final `policy` and `main` turns the returned `reward` is None
    (it has no direct meaning — turn-1 credit comes from the branched
    V̂_1 estimate, not from the immediate tool result).
    """
    floor = error_floor_reward(max(len(task.get("scenarios", [])), 1))
    mode, code = extract_code_with_mode(raw_completion)

    if is_last_turn:
        if mode != MODE_POLICY:
            return ToolResult(mode=mode, reward=floor, text=_final_turn_rejected_text(mode))
        reward, text = _score_policy(code, task, timeout_policy, max_steps)
        return ToolResult(mode=MODE_POLICY, reward=reward, text=text)

    # Non-final turn — every mode produces a tool message; only invalid
    # carries a forward reward signal (a floor penalty that propagates
    # through the V̂_1 estimate and turn-1 advantage).
    if mode == MODE_POLICY:
        _, text = _score_policy(code, task, timeout_policy, max_steps)
        return ToolResult(mode=MODE_POLICY, reward=None, text=text)
    if mode == MODE_MAIN:
        cal = scenario or (task.get("scenarios") or [{}])[0]
        out = score_main_turn(
            code,
            env_code=task.get("env_code", ""),
            kappa=cal.get("kappa", 1.0),
            seed=cal.get("seed", 0),
            env_class=task.get("env_class", DEFAULT_ENV_CLASS),
            timeout=timeout_main,
        )
        return ToolResult(mode=MODE_MAIN, reward=None, text=_main_tool_text(out["stdout"]))
    return ToolResult(mode=MODE_INVALID, reward=floor, text=_INTERMEDIATE_INVALID_TEXT)


# ---------------------------------------------------------------------------
# Per-turn advantage math (pure functions on the (G_1, G_2) reward grid)
# ---------------------------------------------------------------------------


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

    No std-normalization applied. TRL's `scale_rewards` only takes effect
    inside the stock `_generate_and_score_completions`, which the branched
    trainer overrides — so for branched runs, advantages reach the loss in
    the form returned here.
    """
    if not rewards or not rewards[0]:
        raise ValueError("Empty reward grid")
    g2 = len(rewards[0])
    if any(len(row) != g2 for row in rewards):
        raise ValueError("Reward grid is ragged: rows have unequal length")

    flat = [r for row in rewards for r in row]
    v0 = statistics.fmean(flat)
    v1 = [statistics.fmean(row) for row in rewards]
    a_turn1 = [[v1[i] - v0] * g2 for i in range(len(rewards))]
    a_turn2 = [[r - v1[i] for r in row] for i, row in enumerate(rewards)]
    return v0, v1, a_turn1, a_turn2


def rollout_diagnostics(
    rewards: list[list[float]],
    a_turn1: list[list[float]],
    a_turn2: list[list[float]],
) -> dict[str, float]:
    """Per-prompt diagnostics that inform G_1 vs G_2 tuning.

    Given the (G_1, G_2) reward grid and the per-turn advantage grids
    produced by `compute_branched_advantages`, return:

      v1_between_std    : std of V_1[i] across the G_1 first-turn variants.
                          High → turn-1 strategies discriminate; G_1 is
                          carrying signal. Low → first-turn choice barely
                          matters and the tree is wasted.
      v1_within_std_mean: mean over i of std(R[i, :]) — the noise inside
                          each first-turn variant. High → V_1[i] estimate
                          is shaky; G_2 is too small. Low → G_2 is overkill.
      v1_signal_ratio   : v1_between_std / v1_within_std_mean. > 1 means
                          first-turn variation dominates (favor larger G_1
                          at fixed budget); < 1 means within-variant noise
                          dominates (favor larger G_2).
      a_turn1_std       : std of A_turn1 across all G_1*G_2 trajectories
                          for this prompt. Std=0 means the tree gave no
                          first-turn ranking and we're effectively running
                          flat GRPO — branched at this G_1 buys nothing.
      a_turn2_std       : std of A_turn2 across all trajectories.

    All values are floats; std-of-one-sample is reported as 0.0 (vs
    raising) so a single-prompt step still logs cleanly.
    """
    v1 = [statistics.fmean(row) for row in rewards]
    v1_between_std = statistics.pstdev(v1) if len(v1) > 1 else 0.0
    within_stds = [
        statistics.pstdev(row) if len(row) > 1 else 0.0 for row in rewards
    ]
    v1_within_std_mean = statistics.fmean(within_stds) if within_stds else 0.0
    signal_ratio = (
        v1_between_std / v1_within_std_mean
        if v1_within_std_mean > 0 else float("inf")
    )

    flat_a1 = [a for row in a_turn1 for a in row]
    flat_a2 = [a for row in a_turn2 for a in row]
    a1_std = statistics.pstdev(flat_a1) if len(flat_a1) > 1 else 0.0
    a2_std = statistics.pstdev(flat_a2) if len(flat_a2) > 1 else 0.0

    return {
        "v1_between_std": v1_between_std,
        "v1_within_std_mean": v1_within_std_mean,
        "v1_signal_ratio": signal_ratio,
        "a_turn1_std": a1_std,
        "a_turn2_std": a2_std,
    }


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
    return [a_turn1] * turn1_len + [0.0] * tool_len + [a_turn2] * turn2_len


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
    return [1] * turn1_len + [0] * tool_len + [1] * turn2_len


# ---------------------------------------------------------------------------
# Chat-message wrapper
# ---------------------------------------------------------------------------


def format_tool_message(text: str) -> dict:
    """Wrap a tool-result text into a chat message.

    We use role="user" (not "tool") because Qwen's chat template doesn't
    have a dedicated tool-role pathway and we don't want to register an
    actual tool-calling schema. The `[tool]` static prefix on the content
    (added by `execute_turn` already) gives the model a stylistic marker
    so it can recognize tool output without needing a schema change.
    """
    return {"role": "user", "content": text}
