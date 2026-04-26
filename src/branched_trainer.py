"""Branched-rollout (VinePPO-style) GRPO trainer for N=2 multi-turn coding.

Subclasses TRL 0.28 `GRPOTrainer` and overrides `_generate_and_score_completions`
to do branched MC rollouts with per-turn credit assignment.

Why a subclass and not `rollout_func`: TRL 0.28's `rollout_func` parameter is
only invoked through the vLLM generation path (`vllm_generation.py:556, 620`).
The regular HF generate path bypasses it entirely (`grpo_trainer.py:1268`).
We're on a single-L4 setup with HF generate, so the only way to inject custom
rollout logic is by overriding the trainer.

Scope of this override (compared to the base GRPO flow):
  - branched generation (G_1 first-turn variants × G_2 continuations)
  - per-turn (B, T) advantages: A_turn1 = V̂_1[i] - V̂_0, A_turn2 = R[i,j] - V̂_1[i]
  - tool_mask plumbed through to `_compute_loss` (already supported upstream)

Things this override deliberately does NOT support — it asserts on entry and
falls back to the parent class otherwise:
  - vLLM generation
  - multi-modal / image inputs
  - multi-objective rewards
  - Liger loss path
  - importance-sampling correction (since it's a vLLM-only concept)

Beta=0 (no KL) and DAPO loss are supported. The reference-model logp branch
is wired but not exercised in our default config.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any

import torch
from accelerate.utils import gather_object
from trl import GRPOTrainer
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import pad, use_adapter

from src.multi_turn import (
    ToolResult,
    assemble_per_token_advantages,
    assemble_tool_mask,
    compute_branched_advantages,
    execute_turn,
    format_tool_message,
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
        max_turn_tokens: int = 400,
        scoring_timeout: int = 30,
        scoring_max_steps: int = 500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if g1 < 1 or g2 < 1:
            raise ValueError(f"Need g1>=1 and g2>=1, got {g1=}, {g2=}")
        self.g1 = g1
        self.g2 = g2
        self.task_map = task_map
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

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_turn_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=pad_id,
                num_return_sequences=n_samples,
            )

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
        Qwen2.5's ChatML this is close enough. If we ever switch templates,
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
    ) -> tuple[list[_Trajectory], float, list[float]]:
        """Run the full G_1×G_2 tree for one prompt.

        Returns (trajectories, V_0, V_1) — V_0 is the grand-mean reward and
        V_1 is the per-first-turn-variant mean. Each trajectory has its
        per-token `advantages` and `tool_mask` populated.
        """
        # Phase A — generate G_1 first-turn variants
        turn1_outputs = self._generate_one_turn(prompt_messages, self.g1)

        # Phase B — execute each first-turn variant in the sandbox
        turn1_results = [self._execute_first_turn(text, task) for _, _, text in turn1_outputs]

        # Phase C — for each first-turn variant, generate G_2 continuations
        # and score each as a final policy turn. Records flow into trajectories.
        rewards_grid: list[list[float]] = []
        trajectories: list[_Trajectory] = []
        for i, ((p_ids, t1_ids, t1_text), tr1) in enumerate(zip(turn1_outputs, turn1_results)):
            tool_msg = format_tool_message(tr1.text)
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

        # Phase D — compute V_0, V_1, and per-token advantages.
        v0, v1, a_turn1, a_turn2 = compute_branched_advantages(rewards_grid)
        for traj in trajectories:
            t1_len, tool_len, t2_len = traj.segment_lengths()
            traj.advantages = assemble_per_token_advantages(
                t1_len, tool_len, t2_len,
                a_turn1[traj.i][traj.j],
                a_turn2[traj.i][traj.j],
            )
            traj.tool_mask = assemble_tool_mask(t1_len, tool_len, t2_len)
        return trajectories, v0, v1

    # ------------------------------------------------------------------
    # Override _generate_and_score_completions
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self._assert_supported_config(inputs)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]

        # Branched rollout per prompt — produces g1*g2 trajectories per prompt.
        all_trajectories: list[_Trajectory] = []
        for inp, prompt_messages in zip(inputs, prompts):
            task = self.task_map[inp["task_id"]]
            traj_list, v0, v1 = self._branched_rollout_one_prompt(prompt_messages, task)
            all_trajectories.extend(traj_list)
            self._metrics[mode]["branched/V0"].append(v0)
            self._metrics[mode]["branched/V1_std"].append(_pstdev_safe(v1))

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

        self._update_logging_buffers(
            prompt_ids, completion_ids, advantages, rewards,
            mode=mode, inputs=inputs,
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
        """
        assert not self.use_vllm, "BranchedGRPOTrainer does not support vLLM"
        assert getattr(self, "use_liger_loss", False) is False, (
            "Liger loss path not supported in BranchedGRPOTrainer"
        )
        assert "images" not in inputs[0] and "image" not in inputs[0], (
            "Image inputs not supported"
        )
        assert len(self.reward_func_names) <= 1, "Multi-objective rewards not supported"

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


def _pstdev_safe(values: list[float]) -> float:
    """Population stdev with len<2 guard — `statistics.pstdev` errors on empty."""
    return statistics.pstdev(values) if len(values) > 1 else 0.0
