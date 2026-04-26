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
from contextlib import nullcontext
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

logger = logging.getLogger(__name__)


class BranchedGRPOTrainer(GRPOTrainer):
    """GRPO trainer that does branched 2-turn rollouts and per-turn advantages.

    Construction matches GRPOTrainer plus three extra kwargs:
      - g1: int, number of first-turn variants per prompt
      - g2: int, number of second-turn continuations per first-turn variant
      - task_map: dict[str, dict], lookup from task_id (passed in dataset rows)
        to the full task dict — needed to score policy turns against the env.
        Identical shape to the task_map used by `make_reward_fn`.
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

        # The trainer's `num_generations` is set in GRPOConfig; we override
        # it here so internal book-keeping (group sizes, log buffers) sees
        # the actual rollout count per prompt.
        if self.num_generations != g1 * g2:
            logger.warning(
                "Overriding num_generations from %d to g1*g2=%d to match "
                "the branched rollout fan-out.",
                self.num_generations,
                g1 * g2,
            )
            self.num_generations = g1 * g2

        # Diagnostics dict consumed by DiversityCallback. Reset each step.
        self._latest_rollout_diag: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Generation: build the (G_1 × G_2) tree per prompt
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

        gen_kwargs = dict(
            max_new_tokens=self.max_turn_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            num_return_sequences=n_samples,
        )

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for i in range(n_samples):
            new_ids = out[i][prompt_len:].tolist()
            # Trim trailing pad tokens (HF generate left-pads to longest)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            while new_ids and new_ids[-1] == pad_id:
                new_ids.pop()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append((prompt_ids, new_ids, text))
        return results

    def _branched_rollout_one_prompt(self, prompt_messages: list[dict], task: dict):
        """Run the full G_1×G_2 tree for one prompt.

        Returns a flat list of `g1*g2` trajectory dicts. Each trajectory has:
          prompt_ids:    list[int]  — tokens of the original first-turn prompt
          completion_ids: list[int] — turn1 + tool_msg + turn2 tokens
          tool_mask:     list[int]  — 1 for assistant tokens, 0 for tool tokens
          advantages:    list[float] — per-token advantage (T,)
          reward:        float      — final reward (turn-2 policy score, or floor)
          turn1_text:    str        — raw turn-1 completion (for diversity callback)
          turn2_text:    str        — raw turn-2 completion
          turn1_end_char, turn2_end_char: char offsets into the decoded
              completion text (for the per-turn diversity metrics)
        """
        from src.prompts import load_template, render_prompt

        # Phase A: G_1 first-turn generations
        turn1_outputs = self._generate_one_turn(prompt_messages, self.g1)

        # Phase B: execute each turn 1 → tool result + maybe a forward reward
        turn1_tool_results: list[ToolResult] = []
        for _, _, text in turn1_outputs:
            tr = execute_turn(
                text,
                task,
                is_last_turn=False,
                timeout_policy=self.scoring_timeout,
                timeout_main=10,
                max_steps=self.scoring_max_steps,
            )
            turn1_tool_results.append(tr)

        # Phase C: for each turn-1 variant, generate G_2 turn-2 continuations
        # and score each as a final policy turn.
        rewards_grid: list[list[float]] = []
        all_traj: list[dict] = []
        for i, ((p_ids, t1_new_ids, t1_text), tr1) in enumerate(
            zip(turn1_outputs, turn1_tool_results)
        ):
            tool_msg = format_tool_message(tr1.text)
            prefix_messages = (
                list(prompt_messages)
                + [{"role": "assistant", "content": t1_text}]
                + [tool_msg]
            )
            turn2_outputs = self._generate_one_turn(prefix_messages, self.g2)

            row_rewards: list[float] = []
            for j, (p2_ids, t2_new_ids, t2_text) in enumerate(turn2_outputs):
                tr2 = execute_turn(
                    t2_text,
                    task,
                    is_last_turn=True,
                    timeout_policy=self.scoring_timeout,
                    timeout_main=10,
                    max_steps=self.scoring_max_steps,
                )
                # Final reward of the trajectory (i, j) — turn-2 policy score
                # OR floor if turn 2 was main/invalid.
                R = tr2.reward if tr2.reward is not None else 0.0
                row_rewards.append(R)

                # Token-level segmentation. We need the token counts for
                # turn1, tool, turn2 to build (T,) advantages and tool_mask.
                # Compute by re-tokenizing the chat template fragments.
                tokenizer = self.processing_class
                # Tool-message tokens: tokenize the user-role chat block on
                # its own. This is an approximation — the chat template adds
                # role-tokens around each message, so tokenize-the-difference
                # would be more accurate. For Qwen2.5's ChatML this is close
                # enough. If we ever switch templates, revisit.
                tool_text = tokenizer.apply_chat_template(
                    [tool_msg], tokenize=False, add_generation_prompt=False
                )
                tool_ids = tokenizer(tool_text, add_special_tokens=False).input_ids
                turn1_len = len(t1_new_ids)
                tool_len = len(tool_ids)
                turn2_len = len(t2_new_ids)

                # Concatenate tokens for the full completion.
                completion_ids = t1_new_ids + list(tool_ids) + t2_new_ids

                all_traj.append({
                    "i": i,
                    "j": j,
                    "prompt_ids": p_ids,
                    "completion_ids": completion_ids,
                    "turn1_len": turn1_len,
                    "tool_len": tool_len,
                    "turn2_len": turn2_len,
                    "reward": R,
                    "turn1_text": t1_text,
                    "turn2_text": t2_text,
                    "turn1_mode": tr1.mode,
                    "turn2_mode": tr2.mode,
                })
            rewards_grid.append(row_rewards)

        # Phase D: per-turn advantages from the (G_1, G_2) reward grid.
        v0, v1, a_turn1, a_turn2 = compute_branched_advantages(rewards_grid)

        for traj in all_traj:
            i, j = traj["i"], traj["j"]
            traj["advantages"] = assemble_per_token_advantages(
                traj["turn1_len"],
                traj["tool_len"],
                traj["turn2_len"],
                a_turn1[i][j],
                a_turn2[i][j],
            )
            traj["tool_mask"] = assemble_tool_mask(
                traj["turn1_len"],
                traj["tool_len"],
                traj["turn2_len"],
            )
            # Char offsets for the diversity callback's per-turn metrics.
            # The decoded completion text is t1_text + tool_text + t2_text;
            # store the character indices marking turn boundaries.
            traj["turn1_end_char"] = len(traj["turn1_text"])
            traj["turn2_end_char"] = (
                len(traj["turn1_text"]) + len(traj["turn2_text"])
            )

        return all_traj, v0, v1

    # ------------------------------------------------------------------
    # The override: replaces parent _generate_and_score_completions
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Unsupported feature gates — asserted up front so we fail loudly
        # rather than silently producing wrong gradients.
        assert not self.use_vllm, "BranchedGRPOTrainer does not support vLLM"
        assert getattr(self, "use_liger_loss", False) is False, (
            "Liger loss path not supported in BranchedGRPOTrainer"
        )
        assert "images" not in inputs[0] and "image" not in inputs[0], (
            "Image inputs not supported"
        )
        assert len(self.reward_func_names) <= 1, (
            "Multi-objective rewards not supported"
        )

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]

        # Branched rollout per prompt — produces g1*g2 trajectories per prompt.
        all_trajectories: list[dict] = []
        diag_turn_boundaries: list[tuple[int, int]] = []
        for inp, prompt_messages in zip(inputs, prompts):
            task_id = inp.get("task_id")
            task = self.task_map[task_id]
            traj_list, v0, v1 = self._branched_rollout_one_prompt(
                prompt_messages, task
            )
            for t in traj_list:
                all_trajectories.append(t)
                diag_turn_boundaries.append(
                    (t["turn1_end_char"], t["turn2_end_char"])
                )
            self._metrics[mode]["branched/V0"].append(v0)
            self._metrics[mode]["branched/V1_std"].append(
                _std_or_zero(v1)
            )

        # Stash diagnostics for DiversityCallback to read on next on_log.
        self._latest_rollout_diag = {"turn_boundaries": diag_turn_boundaries}

        # Convert lists of token IDs to padded tensors. Mirrors parent's
        # block at grpo_trainer.py:1568-1599.
        prompt_ids_list = [t["prompt_ids"] for t in all_trajectories]
        completion_ids_list = [t["completion_ids"] for t in all_trajectories]
        tool_mask_list = [t["tool_mask"] for t in all_trajectories]
        advantages_list = [t["advantages"] for t in all_trajectories]
        rewards_list = [t["reward"] for t in all_trajectories]

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
        tool_mask = pad(tool_mask, padding_value=1, padding_side="right")

        # (B, T) per-token advantages — pad to completion_ids' shape.
        advantages = [torch.tensor(a, device=device, dtype=torch.float32) for a in advantages_list]
        advantages = pad(advantages, padding_value=0.0, padding_side="right")

        rewards_tensor = torch.tensor(rewards_list, device=device, dtype=torch.float32)

        # Concatenate prompt + completion for the model forward pass.
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        # Compute old / ref logps. Mirrors the parent's branch at
        # grpo_trainer.py:1630-1713 but without the vLLM/IS code path.
        with torch.no_grad(), disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    ref_name = "ref" if "ref" in getattr(model, "peft_config", {}) else None
                    with use_adapter(model, adapter_name=ref_name):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                        )
            else:
                ref_per_token_logps = None

        # Logging buffers — TRL's `log()` method reads these for the
        # completions table and the diversity callback reads `_logs`.
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(advantages.detach().cpu().mean(dim=1).tolist())

        self._metrics[mode]["reward"].append(rewards_tensor.mean().item())
        self._metrics[mode]["reward_std"].append(rewards_tensor.std().item())
        # Fraction of trajectories scored at the floor (turn-2 rejected or env error).
        floor = float(self._error_floor_for_task(inputs))
        frac_floor = (rewards_tensor <= floor + 1e-6).float().mean().item()
        self._metrics[mode]["branched/frac_floor"].append(frac_floor)

        # `num_items_in_batch` is required by DAPO loss for the per-token-loss
        # normalizer. It's the number of valid (model-generated, non-tool,
        # non-truncated) completion tokens, gathered across processes.
        valid_token_count = (completion_mask * tool_mask).sum()
        num_items_in_batch = self.accelerator.gather(valid_token_count).sum()

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,  # (B, T) — TRL handles via the dim==1 check
            "num_items_in_batch": num_items_in_batch,
            "tool_mask": tool_mask,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def _error_floor_for_task(self, inputs):
        """Lookup the per-trajectory error floor (used for the frac_floor metric)."""
        from src.runner import ERROR_PENALTY_PER_SCENARIO
        # Pick any input — all prompts in one step share the task in our setup.
        task = self.task_map[inputs[0]["task_id"]]
        return ERROR_PENALTY_PER_SCENARIO * len(task.get("scenarios", [1]))


def _std_or_zero(values):
    if len(values) < 2:
        return 0.0
    import statistics
    return statistics.pstdev(values)
