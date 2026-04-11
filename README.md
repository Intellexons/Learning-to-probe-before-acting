# Policy DPO Loop

Iterative post-training loop that teaches an LLM to write better policy code through self-play. The model generates Python `policy(state) -> action` functions, executes them in an environment, and uses DPO (Direct Preference Optimization) with LoRA to improve from its own outputs — repeating the cycle with the merged model each iteration.

**Base model:** Qwen2.5-Coder 3B Instruct (configurable in `configs/loop.yaml`)

## How it works

```
For each iteration k:
  1. Generate code candidates    — loads base model + adapters 0..k-1 on GPU
  2. Score by running in env     — sandboxed subprocess execution
  3. Build DPO preference pairs  — best vs median/worst by reward
  4. Train LoRA adapter k        — loads base + adapters 0..k-1, trains new adapter
  5. Evaluate                    — loads base + adapters 0..k, scores on held-out tasks
```

Generation and evaluation load the base model and merge all previous LoRA adapters in GPU memory (bfloat16), so there is no need for disk-based merging or GGUF conversion.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- GPU with 16 GB+ VRAM (T4, L4, A100, or Apple Silicon with MPS)
- Enough disk space for model weights (~6 GB for 3B base + LoRA adapters)

## Setup

Run the one-time setup script. It installs Python dependencies, downloads HF weights, and creates the directory structure:

```bash
bash scripts/00_setup_env.sh
```

## Quick start

Run the full loop end-to-end:

```bash
uv run python scripts/09_run_loop.py --config configs/loop.yaml
```

This is the main entry point. It runs all steps sequentially for each iteration and prints a progress table after each one.

## Scripts

### Main entry points

#### `scripts/00_setup_env.sh` — One-time setup

Installs dependencies, downloads HF weights to `models/hf_base/iter_000/`, and creates the directory structure.

```bash
bash scripts/00_setup_env.sh
```

#### `scripts/09_run_loop.py` — Main orchestrator

Runs the full training loop for the number of iterations specified in the config. This is the script you should use.

```bash
# Run with defaults (auto-generated run name, start from iteration 0)
uv run python scripts/09_run_loop.py --config configs/loop.yaml

# Name your experiment
uv run python scripts/09_run_loop.py --config configs/loop.yaml --run_name my_experiment

# Resume from a specific iteration
uv run python scripts/09_run_loop.py --config configs/loop.yaml --run_name my_experiment --start_iter 3
```

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config (default: `configs/loop.yaml`) |
| `--run_name` | Name for the run. Outputs go under `data/rollouts/<run_name>/`, `models/lora/<run_name>/`, etc. Auto-generated from timestamp if omitted. |
| `--start_iter` | Iteration to resume from (default: `0`). Useful if a run was interrupted. |

---

### Loop steps (called by the orchestrator)

#### Step 1: `scripts/01_generate.py` — Generate code samples

Loads the base HF model, merges all previous LoRA adapters in GPU memory, and generates N code candidates per task using `model.generate()`.

```bash
uv run python scripts/01_generate.py --iter 0 --config configs/loop.yaml
```

| Flag | Description |
|------|-------------|
| `--iter` | Iteration number (required). At iter 0, uses the base model. At iter N, merges adapters 0..N-1. |
| `--config` | YAML config path |
| `--run_name` | Run name for output directory scoping |

**Output:** `data/rollouts/[<run_name>/]iter_000/samples.jsonl`

#### Step 2: `scripts/02_score.py` — Score generated samples

Executes each generated policy in the environment across all scenarios and records rewards. Each policy runs in a sandboxed subprocess with a timeout. Dangerous imports (os, subprocess, etc.) are blocked via static analysis.

```bash
uv run python scripts/02_score.py --iter 0 --config configs/loop.yaml
```

| Flag | Description |
|------|-------------|
| `--iter` | Iteration number (required) |
| `--config` | YAML config path |
| `--run_name` | Run name for output directory scoping |

**Output:** `data/rollouts/[<run_name>/]iter_000/scores.jsonl`

Also appends a row to the train progress CSV (`data/progress/<run_name>/train.csv`).

#### Step 3: `scripts/03_build_dpo.py` — Build DPO preference pairs

Takes the scored samples and builds (chosen, rejected) pairs for DPO training. For each task, the highest-reward sample is paired against the median and worst samples.

```bash
uv run python scripts/03_build_dpo.py --iter 0 --config configs/loop.yaml
```

| Flag | Description |
|------|-------------|
| `--iter` | Iteration number (required) |
| `--config` | YAML config path |
| `--run_name` | Run name for output directory scoping |

**Output:** `data/dpo/[<run_name>/]iter_000_train.jsonl` (and `_eval.jsonl` if eval scores exist)

#### Step 4: `scripts/04_train_dpo_lora.py` — Train LoRA adapter with DPO

Loads the original base model, replays all previous LoRA adapters on top (merge-on-the-fly), then trains a new LoRA adapter using TRL's DPOTrainer. Automatically detects MPS (Apple Silicon) and adjusts settings.

```bash
uv run python scripts/04_train_dpo_lora.py --iter 0 --config configs/loop.yaml
```

| Flag | Description |
|------|-------------|
| `--iter` | Iteration number (required) |
| `--config` | YAML config path |
| `--run_name` | Run name for output directory scoping |

**Output:** `models/lora/[<run_name>/]iter_000/` (adapter weights + tokenizer)

#### Step 5: `scripts/08_eval.py` — Evaluate on held-out tasks

Loads the model with all adapters merged, generates a small number of samples (4 by default) on eval tasks, scores them, and writes a summary with per-task best and mean rewards.

```bash
uv run python scripts/08_eval.py --iter 1 --config configs/loop.yaml
```

| Flag | Description |
|------|-------------|
| `--iter` | Iteration number (required). Merges adapters 0..iter-1. |
| `--config` | YAML config path |
| `--run_name` | Run name for output directory scoping |

**Output:** `data/rollouts/[<run_name>/]iter_001/eval_summary.json`

Also appends a row to the eval progress CSV (`data/progress/<run_name>/eval.csv`).

## Project structure

```
configs/
  loop.yaml              # All hyperparameters and paths
  prompt_template.txt    # Prompt template with placeholders
data/
  tasks/
    train.jsonl          # Training tasks (env code + scenarios)
    eval.jsonl           # Held-out eval tasks
  rollouts/              # Generated samples + scores per iteration
  dpo/                   # DPO preference pairs per iteration
  progress/              # CSV logs tracking reward over iterations
models/
  hf_base/iter_000/      # Original HF weights (downloaded during setup)
  lora/                  # LoRA adapters per iteration
src/
  hf_generator.py        # HuggingFace model loading + generation
  prompts.py             # Prompt template rendering + DPO formatting
  dataset.py             # JSONL I/O, sample/score merging, DPO pair building
  runner.py              # Sandboxed policy execution + scoring
  progress.py            # CSV progress tracking
```

## Configuration

All settings live in `configs/loop.yaml`. Key options:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `loop` | `iterations` | 20 | Number of generate-train-evaluate cycles |
| `generation` | `n_samples_per_task` | 16 | Code candidates per task per iteration |
| `generation` | `temperature` | 0.8 | Sampling temperature |
| `lora` | `r` | 16 | LoRA rank |
| `lora` | `target_modules` | q/k/v/o_proj | Attention layers to adapt |
| `training` | `num_train_epochs` | 4 | DPO training epochs per iteration |
| `training` | `beta` | 0.1 | DPO beta (preference strength) |

## Task format

Each line in `data/tasks/train.jsonl` is a JSON object:

```json
{
  "task_id": "cartpole_001",
  "env_class": "CartPoleEnv",
  "env_code": "import math\n...",
  "state_spec": "state is a dict with keys: ...",
  "action_spec": "action is an int: 0 or 1",
  "goal": "Keep the pole balanced...",
  "scenarios": [
    {"scenario_id": "s0", "seed": 0},
    {"scenario_id": "s1", "seed": 1}
  ]
}
```

The environment code is included directly in the prompt so the model can read it and write a policy against it.

## Training on GCP

End-to-end playbook for running a training loop on a Google Cloud GPU VM.
Sized for a single NVIDIA L4 (24 GB) running the full `configs/loop.yaml`
profile; swap the machine type for other GPUs.

### Prerequisites

1. **gcloud SDK** installed locally (`gcloud --version`) and authenticated
   against the project you want to bill:
   ```bash
   gcloud auth login
   gcloud config set project <your-project-id>
   gcloud config set compute/region europe-west3
   gcloud config set compute/zone europe-west3-a
   ```
2. **Billing account** linked to the project, and **Compute Engine API**
   enabled (`gcloud services enable compute.googleapis.com`).
3. **GPU quota.** New personal projects have `GPUS_ALL_REGIONS = 0` by
   default, which blocks all GPU VMs regardless of regional quotas.
   Request at least `1` at
   [Console → IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas)
   (filter `GPUs (all regions)`). Approval can take minutes to a few days.
4. **(Optional) Spot VMs.** To use preemptible pricing, also raise
   `Preemptible CPUs` in your region to at least `8` — separate request.

### 1. Create the VM

One-shot create for 1×L4 on-demand in `europe-west3-a`, running the
prep-only startup script from this repo:

```bash
gcloud compute instances create lora4vr-train-1 \
  --zone=europe-west3-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu121 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=scripts/gcp_startup.sh
```

Notes:
- `--maintenance-policy=TERMINATE` is **required** for any GPU VM.
- `--scopes=cloud-platform` lets the VM talk to GCS for the optional
  checkpoint sync (see below). Harmless if you don't use GCS.
- For a private repo, override the clone URL via metadata:
  `--metadata=LORA4VR_REPO_URL=https://...`, or skip cloning and
  `gcloud compute scp` the working copy up manually after step 2.
- For Spot (once `Preemptible CPUs` quota is raised), add:
  `--provisioning-model=SPOT --instance-termination-action=STOP`

### 2. Watch the bootstrap

The startup script logs to the GCE serial console and to
`/var/log/lora4vr-startup.log` on the VM. Tail the serial console from
your workstation:

```bash
gcloud compute instances get-serial-port-output lora4vr-train-1 \
  --zone=europe-west3-a | tail -50
```

Bootstrap is complete when you see `lora4vr startup script complete` or
when `/var/run/lora4vr-ready` exists on the VM. Expect 3–8 minutes on a
fresh VM (most of it is the 6 GB HF model download).

### 3. SSH in and train

```bash
gcloud compute ssh lora4vr-train-1 --zone=europe-west3-a
```

On the VM:
```bash
cd /opt/lora4vr
uv run python scripts/09_run_loop.py \
  --config configs/loop.yaml \
  --run_name my_run
```

Leave the SSH session open to watch the run, or detach with `tmux` /
`nohup` if you want to disconnect.

### 4. Get artifacts back to your workstation

When training finishes, copy the outputs down before deleting the VM:

```bash
gcloud compute scp --recurse --zone=europe-west3-a \
  lora4vr-train-1:/opt/lora4vr/models/lora \
  lora4vr-train-1:/opt/lora4vr/data/progress \
  lora4vr-train-1:/opt/lora4vr/data/rollouts \
  ~/Downloads/lora4vr_results/
```

Egress to your workstation is ~$0.12/GB, so a full ~1–2 GB run is
roughly $0.15–0.25.

### 5. Tear down

**Delete the VM** — this stops all billing for compute, GPU, and the
boot disk:

```bash
gcloud compute instances delete lora4vr-train-1 --zone=europe-west3-a
```

⚠️ `gcloud compute instances stop` halts the GPU/CPU charges but keeps
the boot disk billing active (~€4/month for a 100 GB pd-balanced disk).
For a one-shot run, always prefer `delete`.

### Optional: GCS checkpoint sync

For multi-hour runs, Spot/preemptible VMs, or anything you want to
resume after a reboot, enable sync by adding one key to the config:

```yaml
paths:
  gcs_bucket: my-bucket-name
```

When set, `09_run_loop.py` calls `gsutil rsync` after each iteration to
push the per-run subtrees — `lora_dir`, `rollouts_dir`, `data/progress/`,
plus `dpo_dir` when `loop.trainer` is `dpo` — to
`gs://<bucket>/runs/<run>/`. On resume (`--start_iter > 0`), it pulls
the same tree back down first. When the key is unset or empty, sync is
a no-op and local runs are unaffected.

Create the bucket **in the same region as the VM** so the rsync traffic
is free (cross-region egress is $0.08/GB):

```bash
gsutil mb -l europe-west3 gs://<your-unique-bucket-name>/
```

Requires `gsutil` on PATH (preinstalled on Deep Learning VM images).

### Resume an interrupted run

If the VM was preempted / restarted mid-training, create a fresh VM the
same way and resume from the last completed iteration:

```bash
# On the new VM
cd /opt/lora4vr
uv run python scripts/09_run_loop.py \
  --config configs/loop.yaml \
  --run_name my_run \
  --start_iter 7          # whatever iteration was last pushed to GCS
```

Resume only works when `paths.gcs_bucket` is set in the config — the
orchestrator pulls iters `0..start_iter-1` from GCS before starting.

### Cost sanity-check

| Item | Rate (europe-west3, on-demand) | Per full 20-iter run |
|---|---|---|
| VM (`g2-standard-8` + 1×L4) | ~€0.70/hr | ~€3–6 for 4–8 hr |
| Boot disk (100 GB pd-balanced) | ~€0.14/day | ~€0.05 while running |
| GCS storage (Standard, ~2 GB) | ~€0.05/month | ~€0.01/month |
| Egress GCS→workstation | $0.12/GB | ~€0.15–0.25 once |

Spot brings the VM rate down ~70% (~€0.21/hr) — worth the extra quota
request if you plan to run more than a few experiments.

## Tracking progress

Each iteration appends to CSV files under `data/progress/<run_name>/`:

- **train.csv** — success rate, mean/max reward from generation+scoring
- **eval.csv** — eval task performance per iteration

The loop prints a formatted table after each iteration. You can also print it manually:

```python
from src.progress import print_progress_table
print_progress_table(run_name="my_experiment")
```
