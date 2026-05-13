# Branched Multi-Turn GRPO for Code-Policy Generation

Iterative post-training loop that teaches an LLM to write better Python control policies on a single 24 GB GPU. The model emits a `def policy(state)` body over two chat turns — a rehearsal turn (probe the env or draft a policy) and a committed-policy turn — and gets gradient-based credit from a branched Monte Carlo rollout with per-turn advantages and a per-turn loss decomposition.

See `docs/paper.tex` for the full method write-up and `docs/findings.tex` for the dated experiment log.

**Base model:** `Qwen/Qwen3-4B-Instruct-2507` (overridable via `LORA4VR_BASE_MODEL`).
**Trainer:** `BranchedGRPOTrainer` (`src/branched_trainer.py`) — TRL 0.28 `GRPOTrainer` subclass with two overrides: branched rollout in `_generate_and_score_completions` and a two-pass turn-decomposed loss in `compute_loss` / `training_step`.

## How it works

Each outer iteration runs one training subprocess (`scripts/05_train_grpo_branched.py`) and one eval pass (`scripts/08_eval.py`):

```
For each outer iteration k:
  1. Train         — load base + merge adapters 0..k-1, fresh LoRA on top,
                     run max_steps inner GRPO updates; save adapter k
  2. Evaluate      — load base + merge adapters 0..k, score on held-out tasks
```

Inside the trainer, one inner step does the following for each prompt $s$ in the batch:

- **Phase A — turn-1 fan-out.** Sample $G_1$ turn-1 variants $y^{(1)}_i \sim \pi(\cdot \mid s)$ in one `model.generate` call.
- **Phase B — turn-1 sandbox.** Execute each $y^{(1)}_i$ in the sandbox; the tool returns a message $m_i$ (truncated stdout for `main`, or the policy's scenario reward for `policy`).
- **Phase C — turn-2 fan-out and score.** For each $i$, sample $G_2$ continuations $y^{(2)}_{i,j} \sim \pi(\cdot \mid s, y^{(1)}_i, m_i)$ in one `model.generate` call ($G_1$ such calls total). Score each leaf to obtain $R_{i,j}$.
- **Phase D — empirical values and per-turn advantages.** Compute the conditional means

$$\widehat V_0 = \frac{1}{G_1 G_2}\sum_{i,j} R_{i,j}, \qquad \widehat V_1^{(i)} = \frac{1}{G_2}\sum_{j} R_{i,j},$$

and broadcast each to the tokens of its turn:

$$A^{(1)}_i = \widehat V_1^{(i)} - \widehat V_0 \quad \text{on } y^{(1)}_i, \qquad A^{(2)}_{i,j} = R_{i,j} - \widehat V_1^{(i)} \quad \text{on } y^{(2)}_{i,j}.$$

The turn-2 advantage uses the conditional mean given the turn-1 prefix, so a poor turn-2 sample drawn from a strong turn-1 is recognized as poor instead of inheriting its prefix's above-average outcome.

The loss runs in **two passes** with a shared global token normalizer $N_{\text{tok}}$:

- Pass 1: forward $[s \mid y^{(1)}]$, backward $\mathcal{L}_{\text{turn1}}$.
- Pass 2: forward $[s \mid y^{(1)} \mid m \mid y^{(2)}]$ with `logits_to_keep = T_{\text{turn2}}`, backward $\mathcal{L}_{\text{turn2}}$.

Because the DAPO surrogate is token-summed (not per-row averaged), $\mathcal{L}_{\text{turn1}} + \mathcal{L}_{\text{turn2}}$ equals the single-pass loss *exactly*. Peak LM-head saved-activation memory drops from $O(\sum_n T_n)$ to $O(\max_n T_n)$ — the trick that lets $N=2$ branched fit on a 24 GiB GPU at $G_1 = G_2 = 4$, $T = 400$.

Phase-boundary timings are logged to stdout (keyed by `[step k task_id]`) so you can watch progress during long inner steps. See `docs/paper.tex` §3 for the derivation.

### Why the multi-turn protocol

The first turn is either a `main` scratchpad — arbitrary code that probes the env via stdout — or a draft `policy`. The tool returns either truncated stdout or the policy's scenario reward. The second turn must emit a `policy`, which gets scored and produces the trajectory's only scalar reward. This gives the model an explicit affordance for **system-identification probes** when hidden per-scenario parameters (e.g., the slosh-tank's actuator gain) are not in the state. Branched MC on turn-1 prefixes plus per-turn advantages is what lets the gradient distinguish "good probe, bad final policy" from "bad probe, lucky final policy".

### Configuration restrictions for the loss decomposition

The two-pass per-turn loss is exact under: token-summed normalizer (`loss_type ∈ {dapo, cispo}`), `beta = 0`, `num_iterations = 1`, token-level importance sampling. Per-row averaged variants (vanilla GRPO, BNPO, Dr.GRPO) don't decompose this way without different normalizer plumbing; the trainer asserts on entry.

## Prerequisites

- Python 3.11+ (project pins to 3.14 via `uv`)
- [uv](https://docs.astral.sh/uv/)
- A GPU. The production `configs/loop_grpo_branched_slosh.yaml` profile (`G_1=G_2=4`, `max_turn_tokens=400`, bf16) targets **22 GiB free** — comfortably an A100-40GB. L4 (24 GB) fits the smaller smoke config (`configs/smoke_test_slosh.yaml`, `G_1=G_2=2`, `max_turn_tokens=200`) with the older 3B base; whether Qwen3-4B fits on L4 even at smoke sizing is being characterized — see `docs/findings.tex`.
- Disk: ~8 GB for the Qwen3-4B base + a few hundred MB per adapter.

## Setup

```bash
bash scripts/00_setup_env.sh
```

Installs deps, downloads the HF base model to `models/hf_base/iter_000/`, and creates the directory structure. Override the base model with `LORA4VR_BASE_MODEL=org/name` (default `Qwen/Qwen3-4B-Instruct-2507`).

## Quick start

Smoke first — single outer iter, ~15-25 min on an A100:

```bash
uv run python scripts/09_run_loop.py \
  --config configs/smoke_test_slosh.yaml \
  --run_name smoke_001
```

Production slosh run — 23 outer iters, 2 inner steps each, end-of-iter eval:

```bash
uv run python scripts/09_run_loop.py \
  --config configs/loop_grpo_branched_slosh.yaml \
  --run_name slosh_grpo_run_1
```

**Other runtimes:**
- **GCP GPU VM** — see [Training on GCP](#training-on-gcp). Fire-and-forget train via `gcp_find_gpu_and_train.sh` with auto-fetch and teardown.

## Scripts

### Entry points

| Script | Purpose |
|---|---|
| `00_setup_env.sh` | One-time bootstrap: deps, HF base download, dir layout |
| `09_run_loop.py` | Orchestrator: outer loop, train + eval per iter, W&B orchestrator run |
| `05_train_grpo_branched.py` | Single outer iter's training subprocess (current trainer) |
| `08_eval.py` | Single outer iter's eval subprocess (multi-turn rollouts) |
| `eval_frozen_baseline.py` | Standalone N-turn eval against a frozen / merged-adapter checkpoint |

The orchestrator handles all step chaining; you only invoke `09_run_loop.py` directly. The other scripts run as its subprocesses.

#### `scripts/09_run_loop.py` flags

| Flag | Description |
|---|---|
| `--config` | YAML config path (required) |
| `--run_name` | Run name. Outputs go under `models/lora/<run_name>/`, `data/progress/<run_name>/`, `data/rollouts/<run_name>/` |
| `--start_iter` | Iteration to resume from (default `0`). Useful after a crash. |

### Legacy scripts (DPO + single-turn GRPO)

`01_generate.py`, `02_score.py`, `03_build_dpo.py`, `04_train_dpo_lora.py`, `04_train_grpo_lora.py` are the older DPO and single-turn GRPO paths used for earlier experiments (puck environment, Qwen2.5-Coder-3B). They still work but the active path is `05_train_grpo_branched.py`. Configs `loop.yaml` / `loop_grpo.yaml` / `smoke_test*.yaml` correspond to those older pipelines.

## Project structure

```
configs/
  loop_grpo_branched_slosh.yaml   # production: slosh env, branched multi-turn GRPO
  smoke_test_slosh.yaml           # smoke: 1 outer iter, G_1=G_2=2
  loop_grpo_branched.yaml         # older: branched multi-turn GRPO on puck env
  loop_grpo.yaml / loop.yaml      # legacy single-turn GRPO / DPO
  smoke_test_grpo_branched.yaml   # legacy branched smoke (puck)
  prompt_template.txt
data/
  progress/<run>/                 # per-run CSVs and trainer_log_iter_*.json
  rollouts/<run>/                 # rollout JSONs per outer iter
docs/
  paper.tex                       # method write-up
  findings.tex                    # dated experiment log
  methodology.tex
models/
  hf_base/iter_000/               # HF base weights (downloaded by 00_setup_env.sh)
  lora/<run>/iter_NNN/            # LoRA adapter per outer iter
scripts/
  00_setup_env.sh                 # bootstrap
  05_train_grpo_branched.py       # current trainer subprocess
  08_eval.py                      # current eval subprocess
  09_run_loop.py                  # orchestrator
  01..04_*.py                     # legacy DPO + single-turn GRPO
  eval_frozen_baseline.py         # standalone N-turn eval
  gcp_*.sh                        # VM lifecycle + train/eval wrappers (see below)
  seed_tasks/
    slosh_train.jsonl             # 24 per-(seed, κ) slosh training tasks
    slosh_eval.jsonl              # 12 held-out slosh eval tasks
    slosh_smoke_train.jsonl       # 2-row subset for smoke runs
    slosh_tank_env.py             # slosh-tank env source
    build_slosh_tasks.py          # regenerates the jsonls from the env
    puck_*.jsonl, puck_tilted_plane_env.py    # earlier puck env (legacy)
src/
  branched_trainer.py             # current trainer (TRL 0.28 GRPOTrainer subclass)
  multi_turn.py                   # rollout/turn execution + per-turn advantages
  runner.py                       # sandboxed policy/main subprocess execution
  prompts.py                      # multi-turn chat template rendering
  diversity.py                    # DiversityCallback (W&B diagnostics)
  dataset.py, config.py, progress.py, hf_generator.py, gcs_sync.py
tests/                            # unit tests for trainer math, prompts, runner
```

## Configuration

Key knobs in `configs/loop_grpo_branched_slosh.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `loop` | `iterations` | `23` | Outer cycles (train + eval per cycle) |
| `loop` | `trainer` | `grpo_branched` | Selects `BranchedGRPOTrainer` via `09_run_loop.py` |
| `branched` | `g1`, `g2` | `4`, `4` | Tree shape; `G_1 G_2` leaves per prompt |
| `branched` | `max_turn_tokens` | `400` | Per-turn generation cap (trajectory ≈ `2·max_turn_tokens + tool_msg`) |
| `grpo` | `loss_type` | `dapo` | Token-summed clipped surrogate; required for the two-pass decomposition |
| `grpo` | `beta`, `num_iterations` | `0`, `1` | Required values for the two-pass decomposition |
| `grpo` | `epsilon`, `epsilon_high` | `0.2`, `0.28` | Asymmetric DAPO clip |
| `grpo` | `scale_rewards` | `batch` | Reward normalization on the assembled (B, T) tensor |
| `training` | `per_device_train_batch_size` | `16` | = `g1 * g2` (one prompt's worth of trajectories per inner step) |
| `training` | `num_train_epochs` | `1` | Passes over the train task set per outer iter |
| `eval` | `n_turns` | `2` | Eval rollouts match the training protocol's turn count |
| `lora` | `r`, `alpha` | `16`, `32` | LoRA rank + scaling on q/k/v/o_proj |
| `training` | `report_to` | `wandb` | `wandb` or `none` ([W&B section](#optional-monitoring-with-weights--biases)) |
| `paths` | `gcs_bucket` | _(unset)_ | Optional GCS checkpoint sync ([details](#optional-gcs-checkpoint-sync)) |

## Task format

Tasks live in `scripts/seed_tasks/*.jsonl`. Each line is one task with one `scenarios` entry (per-(seed, κ) split — see [#49](https://github.com/nima-siboni/LoRA4VR/pull/49) — so the branched MC value estimator's invariant holds: all `G_2` continuations of a given turn-1 variant see byte-identical env state). Schema:

```json
{
  "task_id": "slosh_tank_train_s0_k0.65",
  "env_class": "SloshTankEnv",
  "env_code": "from dataclasses import dataclass\n...",
  "state_spec": "state is a dict with keys: t, x, v, theta, omega, goal, steps_left",
  "action_spec": "action is a float — commanded cart acceleration in m/s²",
  "goal": "Move the cart to the goal position and settle the liquid...",
  "scenarios": [
    {"scenario_id": "s0_k0.65", "seed": 0, "kappa": 0.65}
  ]
}
```

The full env source is embedded in `env_code` so the prompt template can paste it into the model's context — the model must read the dynamics, not pattern-match an environment name.

`build_slosh_tasks.py` regenerates the jsonls when the env source changes.

## Training on GCP

End-to-end playbook for running on a Google Cloud GPU VM.

### Quick automation (recommended)

The whole lifecycle is wrapped by a small set of scripts. The shortest path is:

```bash
# Train: walk GPU zones until capacity is found, then chain into gcp_train.sh
bash scripts/gcp_find_gpu_and_train.sh \
  configs/loop_grpo_branched_slosh.yaml slosh_grpo_run_1 --wait

# Eval against a specific run's iter_NNN adapter on a fresh L4
bash scripts/gcp_find_gpu_and_eval.sh slosh_grpo_run_1 --wait -- --iter 22

# Manual teardown if you didn't use --wait
bash scripts/gcp_destroy_vm.sh --yes
```

**`gcp_find_gpu_and_train.sh`** walks a preset list of zones, attempting `gcloud compute instances create` in each until one has capacity (stockouts surface as `ZONE_RESOURCE_POOL_EXHAUSTED`). On the first successful zone it chains into `gcp_train.sh` which uploads the repo, runs `00_setup_env.sh`, starts the training in a detached tmux session, and (with `--wait`) blocks locally polling for a `/tmp/lora4vr-<run>.done` marker — then auto-fetches artifacts and destroys the VM. Non-stockout failures (e.g. quota errors) abort the search to avoid leaving partial VMs behind.

`--accelerator` selects the GPU class:

```
l4         (default)  g2-standard-8     1× nvidia-l4              24 GiB
a100-40gb              a2-highgpu-1g     1× nvidia-tesla-a100      40 GiB
a100-80gb              a2-ultragpu-1g    1× nvidia-a100-80gb       80 GiB
h100                   a3-highgpu-1g     1× nvidia-h100-80gb       80 GiB
```

Override the zone preset with `LORA4VR_ZONE_LIST="europe-west4-b europe-west1-c ..."`.

### Prerequisites

1. **gcloud SDK** authenticated against the billing project:
   ```bash
   gcloud auth login
   gcloud config set project <your-project-id>
   ```
2. **Compute Engine API** enabled (`gcloud services enable compute.googleapis.com`).
3. **GPU quota.** Fresh projects start with `GPUS_ALL_REGIONS = 0` (or 1 by default in some cases) and `NVIDIA_*_GPUS = 0` per region. Request at least `1` for the GPU class you intend to use at [Console → IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas). Approval can take hours to days; for new projects the standard denial reason is insufficient billing history.
4. **`.env`** in the repo root with `WANDB_API_KEY=...` if you want W&B logging.

### The underlying steps

If you'd rather drive things by hand, the `gcp_*.sh` wrappers are individually scriptable:

- `gcp_launch_vm.sh` — creates the VM, uploads the repo, runs `gcp_startup.sh` (which seeds the dir layout and downloads the HF base model).
- `gcp_train.sh <config> <run_name> [--wait]` — kicks off training in a detached tmux session, captures the orchestrator's real exit code via `${PIPESTATUS[0]}`, and (with `--wait`) auto-fetches + destroys on completion.
- `gcp_fetch_artifacts.sh <run_name>` — pulls `models/lora/<run>`, `data/progress/<run>`, `data/rollouts/<run>` from the VM to `~/Downloads/lora4vr_<run>/`.
- `gcp_destroy_vm.sh [--yes]` — deletes the VM and its boot disk.
- `gcp_eval.sh`, `gcp_find_gpu_and_eval.sh`, `gcp_fetch_eval.sh` — same pattern for eval-only runs.

Default zone for VM lifecycle scripts is `europe-west3-a` (override via `LORA4VR_ZONE`). The find-gpu wrappers ignore this and walk their own preset list (`europe-west4`, `us-central1`, etc., depending on `--accelerator`).

### Optional: monitoring with Weights & Biases

Two W&B runs are created per training session:
- The **inner trainer** logs per-inner-step metrics (loss, KL, rewards, branched diagnostics like `branched/V0`, `branched/v1_signal_ratio`, `branched/generate_calls`) if `training.report_to: wandb`.
- The **orchestrator** opens `<run_name>-orchestrator` and replays each outer iter's CSV row (train success rate, mean/max reward, eval row) once that iter finishes — giving an outer-iter view alongside the inner-step view.

One-time setup on the VM:

```bash
uv run wandb login   # paste API key from https://wandb.ai/authorize
```

The `gcp_train.sh` wrapper picks up `WANDB_API_KEY` from your local `.env` and forwards it inline so it never lands on disk.

### Optional: GCS checkpoint sync

For long runs on Spot VMs, set `paths.gcs_bucket` in the config to a bucket in the same region as the VM. `09_run_loop.py` then `gsutil rsync`'s `models/lora/<run>/`, `data/progress/<run>/`, and `data/rollouts/<run>/` to `gs://<bucket>/runs/<run>/` after each outer iter, and pulls them back down on `--start_iter > 0`.

```yaml
paths:
  gcs_bucket: my-bucket-name
```

Same-region rsync is free; cross-region egress is $0.08/GB.

### Cost sanity-check

| Item | Rate (europe-west4, on-demand) | Per 4-8 hr run |
|---|---|---|
| L4 VM (`g2-standard-8` + 1×L4) | ~$0.70/hr | $3–6 |
| A100-40GB VM (`a2-highgpu-1g`) | ~$3.67/hr | $15–30 |
| Boot disk (100 GB pd-balanced) | ~$0.14/day | ~$0.05 |
| Egress to workstation | $0.12/GB | $0.10–0.25 once |

Spot variants drop the VM rate ~70% — worth the extra `Preemptible CPUs` quota request if you'll run many experiments.

## Tracking progress

Per-run files under `data/progress/<run_name>/`:

- `train.csv` — per-outer-iter train rollout stats
- `eval.csv` — per-outer-iter eval rollout stats
- `trainer_log_iter_NNN.json` — TRL's per-inner-step log history, replayed to the orchestrator W&B run at end-of-iter

The orchestrator prints a formatted table after each outer iter. To print manually:

```python
from src.progress import print_progress_table
print_progress_table(run_name="my_experiment")
```

## References

- Paper: `docs/paper.tex`
- Experiment log: `docs/findings.tex`
- Methodology notes: `docs/methodology.tex`
