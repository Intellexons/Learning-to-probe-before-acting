"""Which adapters make up a checkpoint of a training run (`src.checkpoints`)."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.checkpoints import (  # noqa: E402
    ADAPTER_CONFIG, ADAPTER_WEIGHTS, SAVED_LAST,
    adapter_dir, available_checkpoints, checkpoint_adapters,
)


def _save_adapter(lora_dir: Path, index: int, weights: bytes, complete: bool = True) -> None:
    """What the trainer leaves for iteration `index`; without `complete`, the state in the
    middle of saving: the adapter is there, the tokenizer (saved last) is not yet."""
    path = adapter_dir(lora_dir, index)
    path.mkdir(parents=True)
    (path / ADAPTER_CONFIG).write_text("{}")
    (path / ADAPTER_WEIGHTS).write_bytes(weights)
    if complete:
        (path / SAVED_LAST).write_text("{}")


def test_checkpoint_zero_is_the_frozen_model(tmp_path):
    assert checkpoint_adapters(tmp_path, 0) == []
    assert available_checkpoints(tmp_path) == 0


def test_checkpoint_k_is_the_first_k_adapters_in_order(tmp_path):
    for i in range(3):
        _save_adapter(tmp_path, i, f"weights {i}".encode())
    adapters = checkpoint_adapters(tmp_path, 2)
    assert [a["name"] for a in adapters] == ["iter_000", "iter_001"]
    assert adapters[1]["sha256"] == hashlib.sha256(b"weights 1").hexdigest()
    assert available_checkpoints(tmp_path) == 3


def test_a_checkpoint_past_the_last_saved_adapter_is_an_error(tmp_path):
    _save_adapter(tmp_path, 0, b"w")
    with pytest.raises(FileNotFoundError, match="1 iteration"):
        checkpoint_adapters(tmp_path, 2)


def test_an_adapter_still_being_saved_does_not_count(tmp_path):
    _save_adapter(tmp_path, 0, b"w")
    _save_adapter(tmp_path, 1, b"w", complete=False)
    assert available_checkpoints(tmp_path) == 1
    with pytest.raises(FileNotFoundError):
        checkpoint_adapters(tmp_path, 2)


def test_a_gap_ends_the_run_of_checkpoints(tmp_path):
    _save_adapter(tmp_path, 0, b"w")
    _save_adapter(tmp_path, 2, b"w")
    assert available_checkpoints(tmp_path) == 1


def test_a_negative_checkpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        checkpoint_adapters(tmp_path, -1)


def test_the_evaluation_refuses_a_checkpoint_that_is_not_there(tmp_path):
    """The script stops before it loads a model: the error names what is missing."""
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_frozen_baseline.py"), "--turns", "2",
         "--lora-dir", str(tmp_path), "--n-adapters", "1", "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )
    assert out.returncode != 0
    assert "iter_000 is missing or not written out in full" in out.stderr


def test_the_evaluation_needs_the_run_for_a_checkpoint(tmp_path):
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_frozen_baseline.py"), "--n-adapters", "1"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )
    assert out.returncode != 0 and "--n-adapters needs --lora-dir" in out.stderr
