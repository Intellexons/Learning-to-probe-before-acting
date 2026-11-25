"""Provenance for every result file: which code produced it, where, when, how.

Every script that writes a result embeds `provenance(sys.argv)` in it, and the
documents cite the commit, the job and the log from that record, so a number in
a table can be traced to the code that produced it. The commit comes from git
when the script runs inside the repository, otherwise from the environment
variable LORA4VR_GIT_COMMIT, which a cluster job receives from
`sbatch --export=ALL,LORA4VR_GIT_COMMIT=<hash>` (the cluster copy of the project
has no .git). Without either the script refuses to run: an untraceable result
is not a result.
"""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(argv: list[str] | None = None) -> dict:
    """The record to embed in a result: commit, tree state, command, host, time, Slurm job."""
    commit = _git("rev-parse", "HEAD")
    source = "git"
    if not commit:
        commit = os.environ.get("LORA4VR_GIT_COMMIT")
        source = "LORA4VR_GIT_COMMIT"
    if not commit:
        raise RuntimeError(
            "no git commit for provenance: run inside the repository, or set "
            "LORA4VR_GIT_COMMIT=<hash> (cluster jobs: sbatch --export=ALL,LORA4VR_GIT_COMMIT=...)"
        )
    status = _git("status", "--porcelain")
    dirty = None if status is None else bool(status.strip())
    slurm = {k[len("SLURM_"):].lower(): v for k, v in os.environ.items()
             if k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST",
                      "SLURM_CPUS_PER_TASK", "SLURM_SUBMIT_DIR")}
    return {
        "commit": commit,
        "commit_source": source,
        "dirty_tree": dirty,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "command": " ".join(argv if argv is not None else sys.argv),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slurm": slurm,
    }


def short(commit: str) -> str:
    return commit[:7]
