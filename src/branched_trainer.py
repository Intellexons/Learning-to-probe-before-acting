"""Branched-rollout (VinePPO-style) GRPO trainer for N=2 multi-turn coding.

Subclasses TRL 0.28 `GRPOTrainer` and overrides three things:
  - `_generate_and_score_completions` — branched MC rollouts with per-turn
    credit assignment (G_1 first-turn variants × G_2 continuations each).
  - `compute_loss` / `training_step` — two-pass per-turn loss decomposition
    so peak memory becomes max(pass_turn1, pass_turn2) instead of their sum
    (the single-pass over [prompt|turn1|tool|turn2] OOMs on a 22 GiB L4
    because the LM-head logits gradient is `(B, T_full, V) ≈ 4 GiB`).

Why a subclass and not `rollout_func`: TRL 0.28's `rollout_func` parameter is
only invoked through the vLLM generation path (`vllm_generation.py:556, 620`).
The regular HF generate path bypasses it entirely (`grpo_trainer.py:1268`).
We're on a single-L4 setup with HF generate, so the only way to inject custom
rollout logic is by overriding the trainer.

Two-pass loss: pass-1 forwards `[prompt | turn1]` with `logits_to_keep=T_turn1`
and computes the DAPO loss over turn-1 advantages; backward; pass-2 forwards
`[prompt | turn1 | tool | turn2]` with `logits_to_keep=T_turn2` and computes
DAPO over turn-2 advantages; backward. Both passes share the same
`num_items_in_batch` normalizer, so loss1 + loss2 equals the single-pass
loss exactly. The trick that lets us reuse the parent's `_compute_loss`
unchanged: pack `[prompt | turn1 | tool]` into the "prompt" field for pass-2
and turn2 alone into the "completion" field; the parent then computes
`logits_to_keep = completion.size(1) = T_turn2_max` automatically.

Things this override deliberately does NOT support — it asserts on entry:
  - vLLM generation
  - multi-modal / image inputs
  - multi-objective rewards
  - Liger loss path
  - importance-sampling correction (vLLM-only concept)
  - per-row averaging losses (grpo / bnpo / dr_grpo / luspo) — only dapo
    and cispo decompose additively
  - non-token-level importance sampling
  - off-policy mask threshold, top-entropy quantile, beta != 0 — each would
    need per-segment plumbing that hasn't been written yet
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
from accelerate.utils import gather_object
from trl import GRPOTrainer
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import pad, use_adapter

from src.prompts import compose_turn_message
from src.multi_turn import (
    ToolResult,
    assemble_per_token_advantages,
    assemble_tool_mask,
    compute_branched_advantages,
    execute_turn,
    format_tool_message,
    rollout_diagnostics,
)
from src.runner import ERROR_PENALTY_PER_SCENARIO

logger = logging.getLogger(__name__)


# Padding values for `pad()` calls. Documented as named constants because the
# 0/1 distinction matters: `tool_mask` pads to 1 (treat padding as a model
# token, masked out separately by `completion_mask`) while `completion_mask`
# pads to 0 (skip padding tokens entirely in the loss).
_PAD_VALUE_TOOL_MASK = 1
_PAD_VALUE_COMPLETION_MASK = 0
_PAD_VALUE_ADVANTAGE = 0.0
# Tolerance when comparing per-trajectory rewards to the error floor for
# the `frac_floor` logging metric.
_FLOOR_COMPARISON_EPS = 1e-6


# ---------------------------------------------------------------------------
# Per-trajectory record produced during rollout
# ---------------------------------------------------------------------------


@dataclass
class _Trajectory:
    """One branch (i, j) of the G_1 × G_2 tree for a single prompt."""
    i: int
    j: int
    prompt_ids: list[int]
    turn1_ids: list[int]
    tool_ids: list[int]
    turn2_ids: list[int]
    turn1_text: str
    turn2_text: str
    turn1_mode: str
    turn2_mode: str
    reward: float
    a_turn1: float = 0.0  # scalar advantage applied to all turn-1 tokens
    a_turn2: float = 0.0  # scalar advantage applied to all turn-2 tokens
    advantages: list[float] | None = None
    tool_mask: list[int] | None = None

    @property
    def completion_ids(self) -> list[int]:
        return self.turn1_ids + self.tool_ids + self.turn2_ids

    @property
    def turn1_end_char(self) -> int:
        return len(self.turn1_text)

    @property
    def turn2_end_char(self) -> int:
        return len(self.turn1_text) + len(self.turn2_text)

    def segment_lengths(self) -> tuple[int, int, int]:
        return len(self.turn1_ids), len(self.tool_ids), len(self.turn2_ids)


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------


class BranchedGRPOTrainer(GRPOTrainer):
    """GRPO trainer that does branched 2-turn rollouts and per-turn advantages.

    Construction matches GRPOTrainer plus four extra kwargs:
      - g1: number of first-turn variants per prompt
      - g2: number of second-turn continuations per first-turn variant
      - task_map: {task_id: task_dict}, lookup needed to score policy turns
        against the env. Identical shape to the task_map used by `make_reward_fn`.
      - max_turn_tokens: per-turn generation length cap. Total trajectory
        length is roughly `2 * max_turn_tokens + |tool_msg|`.

    Example:

        trainer = BranchedGRPOTrainer(
            model=model, args=cfg, train_dataset=ds, processing_class=tok,
            peft_config=peft_cfg,
            reward_funcs=lambda **_: [0.0],  # placeholder, never called
            g1=4, g2=4, task_map=task_map, max_turn_tokens=400,
        )
    """

    def __init__(
        self,
        *args,
        g1: int,
        g2: int,
        task_map: dict[str, dict],
        prompt_turn2_map: dict[str, str] | None = None,
        max_turn_tokens: int = 400,
        scoring_timeout: int = 30,
        scoring_max_steps: int = 600,
        loss_micro_batch_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if g1 < 1 or g2 < 1:
            raise ValueError(f"Need g1>=1 and g2>=1, got {g1=}, {g2=}")
        if loss_micro_batch_size is not None and loss_micro_batch_size < 1:
            raise ValueError(
                f"loss_micro_batch_size must be >= 1 or None, "
                f"got {loss_micro_batch_size}"
            )
        self.g1 = g1
        self.g2 = g2
        self.task_map = task_map
        # Inner micro-batch along the row dim within each per-turn pass.
        # `None` keeps the original behaviour (one forward+backward per turn
        # over all rows). Setting e.g. 4 splits the 16-row segment into 4
        # micro-batches of 4 rows each, calling super()._compute_loss four
        # times and backwarding each; gradients accumulate into the same
        # params. Because num_items_in_batch is the global token count
        # (passed unchanged), the per-micro-batch loss values still divide
        # by the right denominator and sum to the single-pass loss exactly
        # (modulo dropout-induced stochastic ordering, which the existing
        # per-turn split already accepts). The motivation is to shrink the
        # `(B, K, |V|)` LM-head gradient tensor — the dominant backward
        # term per Appendix A of the paper — by reducing B.
        self.loss_micro_batch_size = loss_micro_batch_size
        # Per-task pre-rendered turn-2 prompt body. The eval pipeline
        # (`scripts/eval_frozen_baseline.py:198-203`) re-renders the turn-2
        # prompt and prepends it to the tool-result message; training matches
        # that contract so the model's chat history during rollouts is
        # byte-identical to what was measured at frozen baseline.
        # Empty dict => skip the prepend (legacy behaviour).
        self.prompt_turn2_map = prompt_turn2_map or {}
        self.max_turn_tokens = max_turn_tokens
        self.scoring_timeout = scoring_timeout
        self.scoring_max_steps = scoring_max_steps

        # The trainer's `num_generations` is set in GRPOConfig; override it
        # here so internal book-keeping (group sizes, log buffers) sees the
        # actual rollout count per prompt.
        if self.num_generations != g1 * g2:
            logger.warning(
                "Overriding num_generations from %d to g1*g2=%d to match "
                "the branched rollout fan-out.",
                self.num_generations, g1 * g2,
            )
            self.num_generations = g1 * g2

        # Diagnostics dict consumed by DiversityCallback. Reset each step.
        self._latest_rollout_diag: dict[str, Any] = {}
        # Counts the underlying `model.generate` calls. Per branched prompt:
        # 1 (turn-1 fan-out via num_return_sequences) + G_1 (one per
        # first-turn-prefix continuation). Reset per `_generate_and_score_*`.
        self._generate_call_count = 0

    # ------------------------------------------------------------------
    # Generation primitives
    # ------------------------------------------------------------------

    def _generate_one_turn(
        self,
        prefix_messages: list[dict],
        n_samples: int,
    ) -> list[tuple[list[int], list[int], str]]:
        """Generate `n_samples` continuations for a chat prefix.

        Returns a list of (prompt_ids, new_token_ids, decoded_text) triples.
        prompt_ids is the full tokenized prefix (same for every sample);
        new_token_ids is each sample's new tokens only.
        """
        tokenizer = self.processing_class
        device = self.accelerator.device
        prompt_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_ids = inputs.input_ids[0].tolist()
        prompt_len = inputs.input_ids.shape[1]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Temporarily disable gradient checkpointing so the model can use its
        # KV cache during decode. Without this, transformers forces
        # use_cache=False whenever gradient_checkpointing is enabled and every
        # generated token re-runs full attention over all prior tokens (O(N²)
        # decode instead of O(N)). Restored automatically on context exit.
        with torch.no_grad(), disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_turn_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=pad_id,
                num_return_sequences=n_samples,
                use_cache=True,
            )
        self._generate_call_count += 1

        results = []
        for i in range(n_samples):
            new_ids = out[i][prompt_len:].tolist()
            # HF generate left-pads sequences in the batch to longest; trim.
            while new_ids and new_ids[-1] == pad_id:
                new_ids.pop()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append((prompt_ids, new_ids, text))
        return results

    def _tokenize_tool_message(self, tool_msg: dict) -> list[int]:
        """Tokenize the user-role tool-result message on its own.

        This is an approximation — the chat template adds role-tokens around
        each message, so tokenize-the-difference would be more accurate. For
        Qwen's ChatML this is close enough. If we ever switch templates,
        revisit.
        """
        tokenizer = self.processing_class
        text = tokenizer.apply_chat_template(
            [tool_msg], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(text, add_special_tokens=False).input_ids

    # ------------------------------------------------------------------
    # Branched rollout phases (one prompt at a time)
    # ------------------------------------------------------------------

    def _execute_first_turn(self, raw_text: str, task: dict) -> ToolResult:
        return execute_turn(
            raw_text, task,
            is_last_turn=False,
            timeout_policy=self.scoring_timeout,
            max_steps=self.scoring_max_steps,
        )

    def _execute_last_turn(self, raw_text: str, task: dict) -> ToolResult:
        return execute_turn(
            raw_text, task,
            is_last_turn=True,
            timeout_policy=self.scoring_timeout,
            max_steps=self.scoring_max_steps,
        )

    def _branched_rollout_one_prompt(
        self,
        prompt_messages: list[dict],
        task: dict,
    ) -> tuple[list[_Trajectory], float, dict[str, float]]:
        """Run the full G_1×G_2 tree for one prompt.

        Returns (trajectories, V_0, diag) — V_0 is the grand-mean reward and
        `diag` is the per-prompt diagnostics dict from `rollout_diagnostics`
        (V_1 between/within variance, A_turn{1,2} std, signal ratio). Each
        trajectory has its per-token `advantages` and `tool_mask` populated.
        """
        # Stdout-only phase-timing logs. Each inner step processes one prompt
        # (prompts_per_step=1) so these line up 1:1 with TRL's `i/24` bar and
        # close the "what's happening mid-step" gap. Not pushed to W&B.
        step = int(getattr(self.state, "global_step", -1))
        tid = task.get("task_id", "?")
        t_start = time.monotonic()

        # Phase A — generate G_1 first-turn variants
        t_phase = time.monotonic()
        turn1_outputs = self._generate_one_turn(prompt_messages, self.g1)
        logger.info(
            f"[step {step} {tid}] phase A turn-1 gen (G1={self.g1}): "
            f"{time.monotonic() - t_phase:.1f}s"
        )

        # Phase B — execute each first-turn variant in the sandbox
        t_phase = time.monotonic()
        turn1_results = [self._execute_first_turn(text, task) for _, _, text in turn1_outputs]
        logger.info(
            f"[step {step} {tid}] phase B turn-1 sandbox: "
            f"{time.monotonic() - t_phase:.1f}s"
        )

        # Phase C — for each first-turn variant, generate G_2 continuations
        # and score each as a final policy turn. Records flow into trajectories.
        t_phase = time.monotonic()
        rewards_grid: list[list[float]] = []
        trajectories: list[_Trajectory] = []
        # The turn-2 user message is the probe output followed by the final-turn
        # instructions, composed by the same helper eval uses
        # (`src.prompts.compose_turn_message`), so training and eval see
        # byte-identical messages. With an empty prompt_turn2 (legacy callers
        # that don't provide it), fall back to the body only.
        prompt_turn2 = self.prompt_turn2_map.get(task["task_id"], "")
        for i, ((p_ids, t1_ids, t1_text), tr1) in enumerate(zip(turn1_outputs, turn1_results)):
            if prompt_turn2:
                user_content = compose_turn_message(prompt_turn2, tr1.text)
            else:
                user_content = tr1.text
            tool_msg = format_tool_message(user_content)
            tool_ids = self._tokenize_tool_message(tool_msg)
            prefix_messages = (
                list(prompt_messages)
                + [{"role": "assistant", "content": t1_text}]
                + [tool_msg]
            )
            turn2_outputs = self._generate_one_turn(prefix_messages, self.g2)

            row_rewards: list[float] = []
            for j, (_, t2_ids, t2_text) in enumerate(turn2_outputs):
                tr2 = self._execute_last_turn(t2_text, task)
                reward = tr2.reward if tr2.reward is not None else 0.0
                row_rewards.append(reward)
                trajectories.append(_Trajectory(
                    i=i, j=j,
                    prompt_ids=p_ids,
                    turn1_ids=t1_ids,
                    tool_ids=list(tool_ids),
                    turn2_ids=t2_ids,
                    turn1_text=t1_text,
                    turn2_text=t2_text,
                    turn1_mode=tr1.mode,
                    turn2_mode=tr2.mode,
                    reward=reward,
                ))
            rewards_grid.append(row_rewards)
        logger.info(
            f"[step {step} {tid}] phase C turn-2 gen+score "
            f"(G1×G2={self.g1 * self.g2} trajectories): {time.monotonic() - t_phase:.1f}s"
        )

        # Phase D — compute V_0, V_1, per-token advantages, and diagnostics.
        t_phase = time.monotonic()
        v0, _v1, a_turn1, a_turn2 = compute_branched_advantages(rewards_grid)
        for traj in trajectories:
            t1_len, tool_len, t2_len = traj.segment_lengths()
            traj.a_turn1 = a_turn1[traj.i][traj.j]
            traj.a_turn2 = a_turn2[traj.i][traj.j]
            traj.advantages = assemble_per_token_advantages(
                t1_len, tool_len, t2_len, traj.a_turn1, traj.a_turn2,
            )
            traj.tool_mask = assemble_tool_mask(t1_len, tool_len, t2_len)
        diag = rollout_diagnostics(rewards_grid, a_turn1, a_turn2)
        logger.info(
            f"[step {step} {tid}] phase D advantages: {time.monotonic() - t_phase:.2f}s "
            f"| rollout total {time.monotonic() - t_start:.1f}s | V0={v0:.3f}"
        )
        return trajectories, v0, diag

    # ------------------------------------------------------------------
    # Override _generate_and_score_completions
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self._assert_supported_config(inputs)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Reset the generate-call counter so its post-step value reflects
        # only this rollout. Logged below as `branched/generate_calls`.
        self._generate_call_count = 0

        # TRL's RepeatSampler replicates each unique prompt `num_generations`
        # times so flat GRPO can sample `num_generations` independent
        # completions per prompt. Our branched rollout already produces
        # `g1*g2` trajectories from one call, so the replicas are redundant:
        # without this dedup we'd run `num_generations` full branched trees
        # per unique prompt (16x at production sizing) and pay the same
        # multiplier in backward memory and wall-clock. The paper's
        # Appendix A B=16 budget assumes one tree per step.
        unique_inputs = _unique_by_task_id(inputs)

        # Branched rollout per unique prompt — produces g1*g2 trajectories.
        all_trajectories: list[_Trajectory] = []
        for inp in unique_inputs:
            task = self.task_map[inp["task_id"]]
            traj_list, v0, diag = self._branched_rollout_one_prompt(inp["prompt"], task)
            all_trajectories.extend(traj_list)
            self._metrics[mode]["branched/V0"].append(v0)
            for key, value in diag.items():
                self._metrics[mode][f"branched/{key}"].append(value)

        # Stash diagnostics for DiversityCallback to read on next on_log.
        self._latest_rollout_diag = {
            "turn_boundaries": [
                (t.turn1_end_char, t.turn2_end_char) for t in all_trajectories
            ]
        }

        # Convert lists of token IDs to padded tensors. Mirrors the parent's
        # block at grpo_trainer.py:1568-1599.
        prompt_ids, prompt_mask = self._build_prompt_tensors(all_trajectories, device)
        completion_ids, completion_mask = self._build_completion_tensors(
            all_trajectories, device
        )
        tool_mask = self._build_tool_mask_tensor(all_trajectories, device)
        advantages = self._build_advantages_tensor(all_trajectories, device)
        rewards = torch.tensor(
            [t.reward for t in all_trajectories], device=device, dtype=torch.float32
        )

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        # Compute old / ref logps. Mirrors the parent's branch at
        # grpo_trainer.py:1630-1713 minus the vLLM/IS code path.
        old_per_token_logps, ref_per_token_logps = self._compute_logps(
            prompt_completion_ids, attention_mask, logits_to_keep, batch_size,
        )

        # Per-segment tensors for the two-pass loss decomposition (see
        # `compute_loss`). Each pass calls `super()._compute_loss` with one
        # segment's prompt+completion, so the LM-head logits gradient at
        # backward materializes only for that segment's tokens.
        segment_tensors = self._build_segment_tensors(all_trajectories, device)

        self._update_logging_buffers(
            prompt_ids, completion_ids, advantages, rewards,
            mode=mode, inputs=unique_inputs,
        )

        # Compute count for the amortization-vs-flat sanity check. Flat
        # GRPO with G=g1*g2 and N=2 turns would do 2*g1*g2 generate calls
        # per prompt; branched does 1 + g1. Logged so we can verify the
        # claimed compute savings actually materialize at runtime.
        self._metrics[mode]["branched/generate_calls"].append(
            float(self._generate_call_count)
        )

        # `num_items_in_batch` is required by DAPO loss for the per-token-loss
        # normalizer — number of valid (model-generated, non-tool) tokens.
        valid_token_count = (completion_mask * tool_mask).sum()
        num_items_in_batch = self.accelerator.gather(valid_token_count).sum()

        output: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "tool_mask": tool_mask,
            **segment_tensors,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    # ------------------------------------------------------------------
    # Helpers used by the override
    # ------------------------------------------------------------------

    def _assert_supported_config(self, inputs: list[dict]) -> None:
        """Fail loudly on configurations this subclass doesn't handle.

        These are the cases the parent class supports but we deliberately
        haven't ported. Hitting one of these silently would produce wrong
        gradients or skipped rollouts, so assert up front.

        The two-pass loss decomposition (see `compute_loss`) further restricts
        us to additive loss types (dapo / cispo) so loss1 + loss2 equals the
        single-pass loss. Per-row averaging losses (grpo / bnpo / dr_grpo /
        luspo) don't decompose this way without a different normalizer scheme.
        """
        assert not self.use_vllm, "BranchedGRPOTrainer does not support vLLM"
        assert getattr(self, "use_liger_loss", False) is False, (
            "Liger loss path not supported in BranchedGRPOTrainer"
        )
        assert "images" not in inputs[0] and "image" not in inputs[0], (
            "Image inputs not supported"
        )
        assert len(self.reward_func_names) <= 1, "Multi-objective rewards not supported"
        assert self.loss_type in ("dapo", "cispo"), (
            f"BranchedGRPOTrainer two-pass decomposition requires an additive "
            f"loss_type (dapo or cispo); got {self.loss_type!r}. Other types "
            f"average per-row, so summing per-segment losses would double-count "
            f"the per-row averaging."
        )
        assert self.importance_sampling_level == "token", (
            f"Two-pass decomposition only supports token-level importance "
            f"sampling; got {self.importance_sampling_level!r}."
        )
        assert getattr(self, "off_policy_mask_threshold", None) is None, (
            "off_policy_mask_threshold not supported (would need per-segment "
            "sequence-level KL)"
        )
        assert self.top_entropy_quantile >= 1.0, (
            "top_entropy_quantile < 1 not supported (entropy mask is computed "
            "per-pass; would need a single-pass joint pass)"
        )
        assert self.beta == 0.0, (
            "BranchedGRPOTrainer two-pass requires beta=0 (no KL term). "
            "Supporting beta != 0 needs per-segment ref_per_token_logps, "
            "which would be a follow-up."
        )
        assert self.num_iterations == 1, (
            f"BranchedGRPOTrainer two-pass requires num_iterations=1; got "
            f"{self.num_iterations}. With more iterations, TRL plumbs "
            f"`old_per_token_logps` into inputs (computed once per generation "
            f"and reused across iterations). The two-pass split would need to "
            f"slice that tensor per segment with the same right-aligned layout "
            f"as the segment forwards — not yet implemented."
        )

    def _build_prompt_tensors(
        self, trajectories: list[_Trajectory], device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = [torch.tensor(t.prompt_ids, device=device) for t in trajectories]
        prompt_mask = [torch.ones_like(x, dtype=torch.long) for x in prompt_ids]
        return (
            pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left"),
            pad(prompt_mask, padding_value=0, padding_side="left"),
        )

    def _build_completion_tensors(
        self, trajectories: list[_Trajectory], device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [torch.tensor(t.completion_ids, device=device) for t in trajectories]
        mask = [torch.ones_like(x, dtype=torch.long) for x in ids]
        return (
            pad(ids, padding_value=self.pad_token_id, padding_side="right"),
            pad(mask, padding_value=_PAD_VALUE_COMPLETION_MASK, padding_side="right"),
        )

    def _build_tool_mask_tensor(
        self, trajectories: list[_Trajectory], device: torch.device,
    ) -> torch.Tensor:
        masks = [torch.tensor(t.tool_mask, device=device) for t in trajectories]
        return pad(masks, padding_value=_PAD_VALUE_TOOL_MASK, padding_side="right")

    def _build_advantages_tensor(
        self, trajectories: list[_Trajectory], device: torch.device,
    ) -> torch.Tensor:
        rows = [
            torch.tensor(t.advantages, device=device, dtype=torch.float32)
            for t in trajectories
        ]
        return pad(rows, padding_value=_PAD_VALUE_ADVANTAGE, padding_side="right")

    def _build_segment_tensors(
        self, trajectories: list[_Trajectory], device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Build per-segment tensors used by the two-pass `compute_loss`."""
        return build_segment_tensors(trajectories, device, self.pad_token_id)

    # ------------------------------------------------------------------
    # Two-pass loss (Option C: per-turn loss decomposition)
    # ------------------------------------------------------------------
    #
    # The parent class's _compute_loss does one forward over
    # [prompt | turn1 | tool | turn2] with logits_to_keep = T_completion_max,
    # which materializes a (B, T_completion_max, V) logits gradient at
    # backward. With branched-N=2 on L4 this hits ~4 GiB and OOMs.
    #
    # Decomposition: since loss_type=dapo computes
    #     loss = (per_token_loss * mask).sum() / num_items_in_batch
    # and mask is zero outside (turn1 ∪ turn2), the loss splits exactly:
    #     loss_total = loss_turn1 + loss_turn2
    # where each segment uses the SAME num_items_in_batch normalizer (count
    # of valid turn1+turn2 tokens across the batch). We compute and backward
    # them sequentially; gradients accumulate into the same params, but peak
    # memory is max(pass1, pass2) instead of their sum because pass-1's graph
    # is freed before pass-2's forward starts.

    def _segment_inputs(self, inputs: dict[str, Any], segment: str) -> dict[str, Any]:
        """Build a per-segment inputs dict to feed `super()._compute_loss`.

        The trick is that the segment's "prompt" and "completion" are aliased
        to fields the parent expects (prompt_ids / completion_ids / etc.), so
        we don't have to touch the parent's loss code.
        """
        if segment == "turn1":
            return {
                "prompt_ids": inputs["prompt_ids"],
                "prompt_mask": inputs["prompt_mask"],
                "completion_ids": inputs["turn1_completion_ids"],
                "completion_mask": inputs["turn1_completion_mask"],
                "advantages": inputs["turn1_advantages"],
                "num_items_in_batch": inputs["num_items_in_batch"],
            }
        if segment == "turn2":
            return {
                "prompt_ids": inputs["pass2_prompt_ids"],
                "prompt_mask": inputs["pass2_prompt_mask"],
                "completion_ids": inputs["turn2_completion_ids"],
                "completion_mask": inputs["turn2_completion_mask"],
                "advantages": inputs["turn2_advantages"],
                "num_items_in_batch": inputs["num_items_in_batch"],
            }
        raise ValueError(f"unknown segment: {segment!r}")

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """Two-pass loss: forward+backward for turn-1, then for turn-2.

        Backward is run inside this method, not by the outer `training_step`.
        See `training_step` below — we override it to skip the duplicate
        outer `accelerator.backward(loss)`.

        When `loss_micro_batch_size` is set, each per-turn pass is further
        split along the row (B) dim into sub-batches; the LM-head gradient
        peak shrinks linearly with sub-batch size. The denominator
        (`num_items_in_batch`) is the global model-token count and is
        passed unchanged to every sub-call, so the sum of sub-batch losses
        equals the single-pass loss exactly.
        """
        if return_outputs:
            raise ValueError(
                "BranchedGRPOTrainer.compute_loss does not support return_outputs"
            )
        # Each backward is scaled by 1/grad_accum_steps just like HF
        # Trainer.training_step does for the single-pass case, so the two
        # accumulated gradients sum to (loss1+loss2)/grad_accum_steps —
        # equivalent to a single backward on the combined scaled loss.
        scale = 1.0 / max(1, self.current_gradient_accumulation_steps)

        # Memory instrumentation: per-segment peak allocated bytes. The
        # paper's appendix Table 2 predicts (b) single-pass ~7.8 GiB vs
        # (c) per-turn ~3.9 GiB at production sizing — these counters
        # let us verify that empirically. Reset before each segment so
        # peaks are not contaminated by the previous segment's freed
        # tensors still sitting in the high-water mark.
        mem_log = torch.cuda.is_available()
        if mem_log:
            torch.cuda.reset_peak_memory_stats()

        loss1_value = self._run_segment_pass(model, inputs, "turn1", scale)
        if mem_log:
            peak1 = torch.cuda.max_memory_allocated() / 1024**3
            self._metrics["train"]["mem/peak_turn1_gib"].append(peak1)
            torch.cuda.reset_peak_memory_stats()

        loss2_value = self._run_segment_pass(model, inputs, "turn2", scale)
        if mem_log:
            peak2 = torch.cuda.max_memory_allocated() / 1024**3
            self._metrics["train"]["mem/peak_turn2_gib"].append(peak2)
            self._metrics["train"]["mem/peak_step_gib"].append(max(peak1, peak2))

        # Return the combined SCALED loss so HF Trainer's bookkeeping (which
        # logs the returned loss as the step's training loss) sees the same
        # value it would have for a single-pass run.
        return (loss1_value + loss2_value) * scale

    def _run_segment_pass(
        self,
        model,
        inputs: dict[str, Any],
        segment: str,
        scale: float,
    ) -> torch.Tensor:
        """Compute and backward one per-turn pass, optionally micro-batched.

        Returns the unscaled accumulated loss as a detached scalar.

        Logging caveat: super()._compute_loss appends to `self._metrics`
        (entropy, clip_ratio/*, kl) once per call. With micro-batching, those
        lists get one entry per chunk instead of one per turn, and TRL's
        `log()` averages them with equal *chunk* weight rather than equal
        *token* weight. For roughly equal-length rows the bias is small; if
        precise per-token-weighted metrics matter, derive them from raw
        completions instead of `self._metrics`.
        """
        seg_inputs = self._segment_inputs(inputs, segment)
        rows = seg_inputs["completion_ids"].size(0)

        accumulated: torch.Tensor | None = None
        for start, end in _row_chunks(rows, self.loss_micro_batch_size):
            chunk = _slice_row_dim(seg_inputs, start, end)
            chunk_loss = super()._compute_loss(model, chunk)
            self.accelerator.backward(chunk_loss * scale)
            detached = chunk_loss.detach()
            del chunk_loss
            accumulated = detached if accumulated is None else accumulated + detached

        # `accumulated` is None only when rows == 0, which would mean the
        # rollout produced no trajectories — impossible given the
        # _generate_and_score_completions contract, so an assert is fine.
        assert accumulated is not None
        return accumulated

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Two-pass training step — backward already done inside `compute_loss`.

        Mirrors `transformers.Trainer.training_step` (and TRL's wrapper around
        it) but skips the outer `accelerator.backward(loss)`, which would
        otherwise try to backward a detached tensor and error.
        """
        time_before = time.perf_counter()

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                # compute_loss runs both segment forwards + backwards and
                # returns the (already-scaled) combined loss.
                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch,
                )

            del inputs
            # No outer backward — done inside compute_loss.

        # Mirror TRL's per-step bookkeeping (see trl GRPOTrainer.training_step).
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0

        return loss.detach()

    def _compute_logps(
        self,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute (old_per_token_logps, ref_per_token_logps).

        Either may be None if the corresponding code path doesn't apply
        (no off-policy gap; beta == 0).
        """
        with torch.no_grad(), disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size=batch_size,
                )
            else:
                old_per_token_logps = None

            ref_per_token_logps = self._compute_ref_logps(
                prompt_completion_ids, attention_mask, logits_to_keep, batch_size,
            )

        return old_per_token_logps, ref_per_token_logps

    def _compute_ref_logps(
        self,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.beta == 0.0:
            return None
        if self.ref_model is not None:
            logps, _ = self._get_per_token_logps_and_entropies(
                self.ref_model, prompt_completion_ids, attention_mask,
                logits_to_keep, batch_size=batch_size,
            )
            return logps
        # PEFT path: disable adapter to expose the base model as the ref.
        model = self.accelerator.unwrap_model(self.model)
        ref_name = "ref" if "ref" in getattr(model, "peft_config", {}) else None
        with use_adapter(model, adapter_name=ref_name):
            logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask,
                logits_to_keep, batch_size=batch_size,
            )
        return logps

    def _update_logging_buffers(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        *,
        mode: str,
        inputs: list[dict],
    ) -> None:
        """Populate `_logs` (used by the completions table and DiversityCallback)
        and `_metrics` (used by the trainer's `log()` method)."""
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(advantages.detach().cpu().mean(dim=1).tolist())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        floor = self._error_floor_for_inputs(inputs)
        frac_floor = (rewards <= floor + _FLOOR_COMPARISON_EPS).float().mean().item()
        self._metrics[mode]["branched/frac_floor"].append(frac_floor)

    def _error_floor_for_inputs(self, inputs: list[dict]) -> float:
        """Per-trajectory error floor — used to log `frac_floor`. Assumes all
        prompts in one step share a task (true in our 1-prompt setup; if we
        later batch multiple distinct tasks per step, switch to a per-row floor).
        """
        task = self.task_map[inputs[0]["task_id"]]
        return ERROR_PENALTY_PER_SCENARIO * len(task.get("scenarios", [1]))


def _unique_by_task_id(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate rows from TRL's RepeatSampler expansion.

    TRL's `RepeatSampler` replicates each unique prompt
    `num_generations = g1*g2` times in `inputs` so that flat GRPO can sample
    `num_generations` independent completions per prompt. The branched
    rollout produces `g1*g2` trajectories from one call to
    `_branched_rollout_one_prompt`, so the replicas are redundant — without
    dedup, each step would do `num_generations` full branched trees of the
    same prompt (16x at production sizing).

    Preserves order. Two inputs sharing a `task_id` are assumed to have the
    same `prompt` field, which holds by construction in
    `scripts/05_train_grpo_branched.py:91-105` (one dataset row per task_id).
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for inp in inputs:
        tid = inp["task_id"]
        if tid in seen:
            continue
        seen.add(tid)
        unique.append(inp)
    return unique


def _row_chunks(
    num_rows: int, micro_batch_size: int | None,
) -> list[tuple[int, int]]:
    """Yield (start, end) row-index pairs covering [0, num_rows) in chunks.

    `micro_batch_size=None` (or any value >= num_rows) returns a single
    [0, num_rows) chunk — i.e. the no-op pre-change behaviour. A value of
    1 returns one chunk per row (maximally low memory, highest launch
    overhead). The last chunk may be shorter when num_rows is not divisible.
    """
    if num_rows <= 0:
        return []
    mb = micro_batch_size or num_rows
    mb = max(1, min(mb, num_rows))
    return [(start, min(start + mb, num_rows)) for start in range(0, num_rows, mb)]


def _slice_row_dim(seg_inputs: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Slice every batched tensor in a segment-inputs dict along dim 0.

    Scalar (0-dim) tensors like `num_items_in_batch` are passed through
    unchanged — that scalar is the *global* model-token count and must
    stay constant across micro-batches so the per-micro-batch losses sum
    to the single-pass loss.
    """
    sliced: dict[str, Any] = {}
    for key, value in seg_inputs.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def build_segment_tensors(
    trajectories: list[_Trajectory],
    device: torch.device,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Build per-segment tensors used by the two-pass `compute_loss`.

    The trick: we treat each pass as a separate (prompt, completion) pair so
    we can reuse `super()._compute_loss` unchanged. For pass 2 the "prompt"
    is the full prefix [prompt | turn1 | tool] and the "completion" is just
    turn2 — that way the parent's `logits_to_keep = completion_ids.size(1)`
    becomes T_turn2_max instead of T_completion_max, which is what saves the
    memory.

    Returns a dict of tensors:
      turn1_completion_ids   (B, T_turn1_max)  right-padded turn1 tokens
      turn1_completion_mask  (B, T_turn1_max)  1 on actual, 0 on pad
      turn1_advantages       (B, T_turn1_max)  A^(1)[b] on actual, 0 on pad
      pass2_prompt_ids       (B, P2_max)       left-padded [prompt|turn1|tool]
      pass2_prompt_mask      (B, P2_max)
      turn2_completion_ids   (B, T_turn2_max)  right-padded turn2 tokens
      turn2_completion_mask  (B, T_turn2_max)
      turn2_advantages       (B, T_turn2_max)  A^(2)[b] on actual, 0 on pad
    """
    # --- Pass 1: completion = turn1 ---
    t1_ids = [torch.tensor(t.turn1_ids, device=device) for t in trajectories]
    t1_masks = [torch.ones_like(x, dtype=torch.long) for x in t1_ids]
    turn1_completion_ids = pad(t1_ids, padding_value=pad_id, padding_side="right")
    turn1_completion_mask = pad(t1_masks, padding_value=0, padding_side="right")
    a1_per_row = torch.tensor(
        [t.a_turn1 for t in trajectories], device=device, dtype=torch.float32,
    ).unsqueeze(1)
    turn1_advantages = a1_per_row * turn1_completion_mask.to(torch.float32)

    # --- Pass 2: prompt = [prompt | turn1 | tool], completion = turn2 ---
    p2_prompt_lists = [
        torch.tensor(t.prompt_ids + t.turn1_ids + t.tool_ids, device=device)
        for t in trajectories
    ]
    p2_prompt_masks = [torch.ones_like(x, dtype=torch.long) for x in p2_prompt_lists]
    pass2_prompt_ids = pad(p2_prompt_lists, padding_value=pad_id, padding_side="left")
    pass2_prompt_mask = pad(p2_prompt_masks, padding_value=0, padding_side="left")

    t2_ids = [torch.tensor(t.turn2_ids, device=device) for t in trajectories]
    t2_masks = [torch.ones_like(x, dtype=torch.long) for x in t2_ids]
    turn2_completion_ids = pad(t2_ids, padding_value=pad_id, padding_side="right")
    turn2_completion_mask = pad(t2_masks, padding_value=0, padding_side="right")
    a2_per_row = torch.tensor(
        [t.a_turn2 for t in trajectories], device=device, dtype=torch.float32,
    ).unsqueeze(1)
    turn2_advantages = a2_per_row * turn2_completion_mask.to(torch.float32)

    return {
        "turn1_completion_ids": turn1_completion_ids,
        "turn1_completion_mask": turn1_completion_mask,
        "turn1_advantages": turn1_advantages,
        "pass2_prompt_ids": pass2_prompt_ids,
        "pass2_prompt_mask": pass2_prompt_mask,
        "turn2_completion_ids": turn2_completion_ids,
        "turn2_completion_mask": turn2_completion_mask,
        "turn2_advantages": turn2_advantages,
    }


def _pstdev_safe(values: list[float]) -> float:
    """Population stdev with len<2 guard — `statistics.pstdev` errors on empty."""
    return statistics.pstdev(values) if len(values) > 1 else 0.0
