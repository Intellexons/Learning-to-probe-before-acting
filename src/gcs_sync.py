"""Sync per-run training artifacts to/from Google Cloud Storage.

Enabled by setting `paths.gcs_bucket` in the training config. When unset,
all functions in this module are no-ops, so local runs are unaffected.

The caller passes the list of local subtrees to sync. `09_run_loop.py`
builds this list from `cfg["paths"]` and the active trainer, so nothing in
this module is trainer-specific. Layout on GCS, for bucket `my-bucket`,
run `foo`, subtree `models/lora`:

    gs://my-bucket/runs/foo/models/lora/foo/iter_000/...

i.e. `gs://<bucket>/runs/<run>/<subtree>/<run>/...`.
"""
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def subtrees_for_config(cfg: dict) -> list[str]:
    """Derive the list of per-run local subtrees to sync from a training
    config. Skips `dpo_dir` entirely when the trainer is not `dpo`, so GRPO
    runs don't issue pointless rsyncs against an empty DPO directory.

    `data/progress` is hardcoded to match `src.progress.PROGRESS_DIR` since
    it's not currently configurable via `paths`.
    """
    paths = cfg.get("paths", {})
    trainer = cfg.get("loop", {}).get("trainer", "dpo")
    subtrees = [
        paths.get("lora_dir", "models/lora"),
        paths.get("rollouts_dir", "data/rollouts"),
    ]
    if trainer == "dpo":
        subtrees.append(paths.get("dpo_dir", "data/dpo"))
    subtrees.append("data/progress")
    return subtrees


def _gs_prefix(bucket: str, run_name: str) -> str:
    return f"gs://{bucket.strip('/')}/runs/{run_name}"


def _require_gsutil():
    if shutil.which("gsutil") is None:
        raise RuntimeError(
            "paths.gcs_bucket is set but `gsutil` is not on PATH. "
            "Install the Google Cloud SDK or unset paths.gcs_bucket."
        )


def _rsync(src: str, dst: str, *, allow_missing_src: bool = False):
    """Run `gsutil -m rsync -r src dst`, skipping silently when src is a
    missing local directory (first-iteration case)."""
    if allow_missing_src and not Path(src).exists():
        return
    cmd = ["gsutil", "-m", "rsync", "-r", src, dst]
    logger.info("gcs_sync: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def push_run(bucket: str | None, run_name: str, subtrees: list[str]):
    """Push per-run artifacts in `subtrees` to GCS. No-op if bucket is falsy.

    Call after each iteration. gsutil rsync is diff-based, so repeated calls
    only upload the delta.
    """
    if not bucket:
        return
    _require_gsutil()
    prefix = _gs_prefix(bucket, run_name)
    for subtree in subtrees:
        local = f"{subtree}/{run_name}"
        remote = f"{prefix}/{subtree}/{run_name}"
        _rsync(local, remote, allow_missing_src=True)


def pull_run(bucket: str | None, run_name: str, subtrees: list[str]):
    """Pull per-run artifacts in `subtrees` from GCS into the local tree.
    No-op if bucket is falsy.

    Call before resuming a run (e.g. when --start_iter > 0 on a fresh VM).
    Missing remote subtrees are tolerated — if nothing exists yet on GCS the
    local tree is simply left empty.
    """
    if not bucket:
        return
    _require_gsutil()
    prefix = _gs_prefix(bucket, run_name)
    for subtree in subtrees:
        local = f"{subtree}/{run_name}"
        remote = f"{prefix}/{subtree}/{run_name}"
        Path(local).mkdir(parents=True, exist_ok=True)
        # Remote may not exist on first run — `gsutil rsync` from a
        # nonexistent source returns non-zero, so probe first with `ls`.
        probe = subprocess.run(
            ["gsutil", "ls", remote],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            logger.info("gcs_sync: nothing to pull at %s", remote)
            continue
        _rsync(remote, local)
