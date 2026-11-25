"""Branched-rollout (VinePPO-style) helpers for N=2 multi-turn GRPO.

The training-loop trick designed in the 2026-04-26 conversation:

  For each prompt, generate G_1 first replies. For each first reply i, act on
  it (`execute_turn`): a reply without a tool call is the final answer and
  ends the task there, a probe run returns what it printed as the tool
  response. Then generate G_2 second replies after each first reply that did
  not end the task. What a reply is follows from whether it calls the tool
  (`src.agent_loop`), not from its position or the code in it. The final
  reward of the trajectory (i, j) is the reward of its final run.

  Empirical values (no learned critic, all MC):
    V̂_0     = grand mean over all G_1 G_2 final rewards (initial-state baseline)
    V̂_1^(i) = mean over the G_2 continuations of first-turn variant i

  Per-turn advantages broadcast to per-token advantages of shape (B, T):
    A^(1)_i   = V̂_1^(i) - V̂_0     applied to turn-1 tokens of trajectory (i, j)
    A^(2)_{i,j} = R_{i,j} - V̂_1^(i)  applied to turn-2 tokens

  Tool-response tokens (the messages between turn 1 and turn 2) are masked
  out via tool_mask so they do not contribute to the loss.

This module holds the pure functions for that math + the chat-message helpers
that aren't tied to the trainer subclass. The trainer-side I/O (model.generate
calls, tokenization, padding) lives in `src/branched_trainer.py`.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.agent_loop import (
    ToolCall,
    controller_from_answer,
    parse_tool_calls,
    tool_message,
    tools_for,
)
from src.runner import (
    DEFAULT_ENV_CLASS,
    DEFAULT_MAX_STEPS,
    ERROR_PENALTY_PER_SCENARIO,
    MAIN_DEFAULT_TIMEOUT,
    POLICY_DEFAULT_TIMEOUT,
    error_floor_result,
    score_main_turn,
    score_policy_turn,
)

# What a reply was. A reply without a tool call is the final answer: the
# final run. A reply with tool calls is a probe run if at least one of them
# could be acted on.
ACTION_FINAL = "final_run"
ACTION_PROBE = "probe_run"
ACTION_INVALID = "invalid"   # tool calls, none of which could be acted on

# Turn kinds as the trainer records them: "policy" is a final run (scored, the
# task ended), "main" a probe run (what it printed comes back), "invalid" a
# reply whose calls could not be acted on.
MODE_POLICY = "policy"
MODE_MAIN = "main"
MODE_INVALID = "invalid"


# ---------------------------------------------------------------------------
# Reward floor + tool-message formatters
# ---------------------------------------------------------------------------


def error_floor_reward(n_scenarios: int) -> float:
    """Reward floor for a task that ended without a final run, or whose
    controller did not run. Matches the per-scenario error-floor convention,
    so the two score the same."""
    return ERROR_PENALTY_PER_SCENARIO * n_scenarios


def _policy_tool_text(reward_total: float, n_scenarios: int) -> str:
    return f"Final run scored across {n_scenarios} scenario(s): total_reward = {reward_total:.2f}"


def _main_tool_text(stdout: str) -> str:
    """The tool response of a probe run: what it printed, verbatim, or
    "[empty output]" if it printed nothing."""
    return stdout or "[empty output]"


# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """What came of one reply.

    `action` is what the reply was (ACTION_*), `mode` the same in the
    trainer's words. `ended` is True when the task is over: the reply was the
    final answer, or it was the last reply and was not. `reward` is the total
    reward when the task ended (the error floor if there was no final run),
    None otherwise. `score` is the full score dict behind that reward.
    `messages` are the chat messages that follow the reply when the task goes
    on: one tool response per call. `text` is their content joined, for logs
    and for callers that read the probe output. `code` is the controller of
    the final run, or else the code of the first probe run.
    """
    mode: str
    reward: float | None
    text: str
    ended: bool = False
    action: str = ""
    messages: tuple[dict, ...] = ()
    code: str = ""
    score: dict | None = None
    calls: tuple[ToolCall, ...] = ()


def replies_left_text(k: int) -> str:
    """The line that states how many replies the model has left. A fact about
    the state of the task, like `steps_left` in the environment's state; it
    says nothing about what to do with them."""
    return f"[{k} repl{'y' if k == 1 else 'ies'} left]"


_NO_FINAL_RUN = {
    ACTION_PROBE: "the last reply called probe_run",
    ACTION_INVALID: "the last reply held a tool call that could not be acted on",
}


def execute_turn(
    raw_completion: str,
    task: dict,
    *,
    is_last_turn: bool,
    scenario: dict | None = None,
    replies_left: int | None = None,
    timeout_policy: int = POLICY_DEFAULT_TIMEOUT,
    timeout_main: int = MAIN_DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ToolResult:
    """Act on one reply and return a ToolResult.

    A reply **without a tool call** is the final answer, on any turn: the
    controller in it (`agent_loop.controller_from_answer`) is scored as the
    final run and the task ends. An answer without a controller, a controller
    that fails validation and one that errors at runtime all score the error
    floor (see `score_sample`).

    A reply **with tool calls** is acted on call by call, in the order they
    are written: each `probe_run` is executed in the sandbox against the probe
    batch and what it printed is its tool response; a call that cannot be
    acted on gets a response that says why. `reward` is None then: credit for
    the first turn comes from the branched estimate, not from the tool
    response.

    The **last reply** has to be the final answer. If it calls a tool
    instead, the task ends without a final run and scores the error floor;
    nothing is executed.

    With `replies_left` given, the last tool response of the reply ends with
    a line that states the number (`replies_left_text`). `text` stays what
    the calls returned, without that line.
    """
    calls = tuple(parse_tool_calls(raw_completion))
    n_scenarios = len(task.get("scenarios", []))

    if not calls:
        code = controller_from_answer(raw_completion)
        score = score_policy_turn(code, task, timeout=timeout_policy, max_steps=max_steps)
        return ToolResult(
            mode=MODE_POLICY, reward=score["reward_total"],
            text=_policy_tool_text(score["reward_total"], n_scenarios), ended=True,
            action=ACTION_FINAL, code=code, score=score, calls=calls,
        )

    probes = [c for c in calls if c.error is None]
    action, mode = (ACTION_PROBE, MODE_MAIN) if probes else (ACTION_INVALID, MODE_INVALID)
    code = probes[0].code if probes else ""

    if is_last_turn:
        why = _NO_FINAL_RUN[action]
        score = error_floor_result(task, [f"no final run: {why}"])
        return ToolResult(mode=mode, reward=score["reward_total"], text=f"[no final run: {why}]",
                          ended=True, action=action, code=code, score=score, calls=calls)

    # The probe run uses the task's (single) scenario: same hidden parameters
    # as the final run, on the scenario's probe batch.
    cal = scenario or (task.get("scenarios") or [{}])[0]
    messages: list[dict] = []
    for c in calls:
        if c.error is not None:
            messages.append(tool_message(f"[tool call not acted on: {c.error}]"))
            continue
        out = score_main_turn(
            c.code,
            env_code=task.get("env_code", ""),
            scenario=cal,
            env_class=task.get("env_class", DEFAULT_ENV_CLASS),
            timeout=timeout_main,
        )
        messages.append(tool_message(_main_tool_text(out["stdout"])))
    text = "\n".join(m["content"] for m in messages)
    if replies_left is not None:
        last = messages[-1]
        messages[-1] = {**last, "content": last["content"].rstrip("\n") + "\n\n" + replies_left_text(replies_left)}
    return ToolResult(mode=mode, reward=None, text=text,
                      ended=False, action=action, messages=tuple(messages), code=code, calls=calls)


def run_rollout(
    generate,
    brief: str,
    task: dict,
    total_turns: int,
    *,
    show_replies_left: bool = False,
    timeout_policy: int = POLICY_DEFAULT_TIMEOUT,
    timeout_main: int = MAIN_DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict:
    """One task, from the brief to the end: the loop evaluation runs.

    `generate(messages, tools)` returns the model's next reply as text
    (`tools` is None when the model has a single reply). The loop gives the
    model the brief and the tool, acts on each reply with `execute_turn`, and
    stops when the task has ended, which the last reply always brings about.
    Returns the transcript, the ToolResult of every reply, and the score of
    the task. With `show_replies_left`, every tool response tells the model
    how many replies it has left.
    """
    tools = tools_for(total_turns)
    messages: list[dict] = [{"role": "user", "content": brief}]
    turns: list[ToolResult] = []
    for turn in range(1, total_turns + 1):
        raw = generate(messages, tools)
        messages.append({"role": "assistant", "content": raw})
        res = execute_turn(raw, task, is_last_turn=(turn == total_turns),
                           replies_left=(total_turns - turn) if show_replies_left else None,
                           timeout_policy=timeout_policy, timeout_main=timeout_main, max_steps=max_steps)
        turns.append(res)
        if res.ended:
            break
        messages.extend(res.messages)
    last = turns[-1]
    return {
        "transcript": messages,
        "turns": turns,
        "score": last.score,
        "final_run": last.mode == MODE_POLICY,
        "code": last.code if last.mode == MODE_POLICY else "",
    }


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
