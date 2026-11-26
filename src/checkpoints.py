"""The saved weights of a training run: which adapters make up a checkpoint.

The training loop (`scripts/09_run_loop.py`) saves one LoRA adapter per
iteration under `<lora dir>/iter_000`, `iter_001`, ... Each adapter is trained
on top of the base model with all earlier adapters merged in. The model after
k iterations, "checkpoint k", is therefore the base model with the adapters
`iter_000` ... `iter_{k-1}` merged in that order; checkpoint 0 is the frozen
base model. `src.hf_generator.load_model_for_generation` does the merging;
this module says which adapters a checkpoint needs, whether they are all on
disk, and what their weights hash to, so that a result can name the weights
it was computed with.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
# The trainer saves the tokenizer after the adapter (scripts/05_train_grpo_branched.py,
# scripts/04_train_grpo_lora.py), so its presence says the adapter was written out in full.
SAVED_LAST = "tokenizer_config.json"


def adapter_dir(lora_dir: str | Path, index: int) -> Path:
    """Where iteration `index` (0-based) of a run saves its adapter."""
    return Path(lora_dir) / f"iter_{index:03d}"


def adapter_is_complete(path: str | Path) -> bool:
    path = Path(path)
    return all((path / name).is_file() for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS, SAVED_LAST))


def available_checkpoints(lora_dir: str | Path) -> int:
    """The largest k for which checkpoint k is on disk in full (0 if there is no adapter yet)."""
    k = 0
    while adapter_is_complete(adapter_dir(lora_dir, k)):
        k += 1
    return k


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_adapters(lora_dir: str | Path, n_adapters: int) -> list[dict]:
    """The adapters of checkpoint `n_adapters`, in merge order, each with the hash of its weights.

    Raises if the checkpoint is not on disk in full: an evaluation of a
    checkpoint whose last adapter is still being trained would be an
    evaluation of something else.
    """
    if n_adapters < 0:
        raise ValueError(f"n_adapters must be >= 0, got {n_adapters}")
    adapters = []
    for i in range(n_adapters):
        path = adapter_dir(lora_dir, i)
        if not adapter_is_complete(path):
            raise FileNotFoundError(
                f"checkpoint {n_adapters} of {lora_dir} needs the adapters iter_000 to "
                f"iter_{n_adapters - 1:03d}; {path} is missing or not written out in full "
                f"({available_checkpoints(lora_dir)} iteration(s) are complete)"
            )
        adapters.append({"name": path.name, "sha256": _sha256(path / ADAPTER_WEIGHTS)})
    return adapters
