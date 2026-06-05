# Learning to probe before acting: a multi-turn RL approach

A small research codebase for teaching a 4B-parameter LLM to write Python
control policies, on a single 24 GB GPU.

The model doesn't just emit code in one shot. It interacts with a task over
two chat turns: a first turn it can use to probe the environment or sketch a
draft, and a final turn that must commit a `def policy(state)` body. Only the
final policy is executed and scored. Training has to figure out, from that
single scalar reward, whether the *probe* was useful or whether the final
policy happened to work for unrelated reasons. That credit-assignment problem
is what this project exists to study.

The accompanying paper is in `docs/paper.tex` ("Learning to Probe Before
Acting"). The dated experiment log is in `docs/findings.tex`.

## What it's training

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507` (override with
  `LORA4VR_BASE_MODEL`). Weights are frozen; only a LoRA adapter is trained.
- **Task family:** a custom slosh-tank control problem. A cart carries
  liquid (modelled as one damped pendulum mode) and the policy has to drive
  it to a goal *and* settle the slosh. A per-task hidden actuator gain
  $\kappa$ makes a blind controller fail at some levels and succeed at
  others — the gap that information-gathering turn 1 is meant to close. The
  environment source ships inside each prompt, so the model has to read the
  dynamics rather than recognize a name.
- **Loop:** a sequence of outer iterations. Each one trains a fresh LoRA on
  top of the merged previous adapters, then runs an eval pass on held-out
  scenarios.

## The idea, in plain language

Standard outcome-reward RL for LLMs samples one completion, scores it, and
hands that scalar back to every token it generated. That's a workable
approximation when the trajectory is just "one block of text". It's a worse
approximation here, where turn 1 (a probe) and turn 2 (the committed policy)
play different roles and only turn 2 is what gets executed.

Two ideas address this:

1. **Branched rollouts.** For each prompt we sample $G_1$ first-turn
   variants, and under each one we sample $G_2$ second-turn continuations.
   That gives a small tree per prompt. Averaging rewards across continuations
   from the *same* first-turn variant tells us how good that variant was;
   averaging across the whole tree gives the prompt's baseline. Per-turn
   advantages fall out of those two averages — turn-1 tokens are scored
   against the prompt baseline, turn-2 tokens against their own prefix's
   baseline. A bad final policy from a strong probe gets recognized as bad
   rather than inheriting the probe's reputation.
2. **Per-turn loss decomposition.** Because the underlying objective (DAPO's
   token-summed clipped surrogate) is additive over disjoint token subsets,
   we can run a separate forward + backward for each turn and still get the
   exact full-trajectory gradient. The point is memory: the LM head's
   activation tensor is what would otherwise OOM on a 24 GB card, and
   splitting it per turn roughly halves the peak.

Both pieces are in `src/branched_trainer.py`, a `BranchedGRPOTrainer` that
subclasses TRL 0.28's `GRPOTrainer` with two overrides — the branched
rollout and the two-pass loss.

For the full derivation, ablation arms, and slosh-tank specifics, see the
paper.

## Prerequisites

- Python 3.11+ (the project pins 3.14 via `uv`)
- [uv](https://docs.astral.sh/uv/)
- A GPU. The production config (`configs/loop_grpo_branched_slosh.yaml`,
  $G_1 = G_2 = 4$, 400 tokens per turn, bf16) targets ~22 GiB free —
  comfortably an A100-40GB. The smoke config (`configs/smoke_test_slosh.yaml`,
  $G_1 = G_2 = 2$, 200 tokens per turn) is sized for an L4 24 GB. Whether
  the production config also fits on L4 is being characterized in
  `docs/findings.tex`.
- About 8 GB of disk for the base model, plus a few hundred MB per adapter.

## Setup

```bash
bash scripts/00_setup_env.sh
```

Installs dependencies, downloads the HF base model to
`models/hf_base/iter_000/`, and creates the directory layout. Override the
base model with `LORA4VR_BASE_MODEL=org/name`.

## Quick start

Smoke run first — one outer iteration, roughly 15–25 minutes on an A100:

```bash
uv run python scripts/09_run_loop.py \
  --config configs/smoke_test_slosh.yaml \
  --run_name smoke_001
```

Production slosh run — 23 outer iterations, two inner steps each, eval at
the end of every iteration:

```bash
uv run python scripts/09_run_loop.py \
  --config configs/loop_grpo_branched_slosh.yaml \
  --run_name slosh_grpo_run_1
```

For running on a Google Cloud GPU VM with auto-fetch and auto-teardown, see
[Training on GCP](#training-on-gcp).

## Scripts

The orchestrator (`09_run_loop.py`) is the only entry point you invoke
directly. It chains training and eval subprocesses for each outer iteration.

| Script | Purpose |
|---|---|
| `00_setup_env.sh` | One-time bootstrap: dependencies, base model, directory layout |
| `09_run_loop.py` | Outer loop, train + eval per iteration, orchestrator W&B run |
| `05_train_grpo_branched.py` | One iteration's training subprocess (the current trainer) |
| `08_eval.py` | One iteration's eval subprocess (multi-turn rollouts) |
| `eval_frozen_baseline.py` | Standalone N-turn eval against a frozen or merged-adapter checkpoint |

`09_run_loop.py` flags:

| Flag | Description |
|---|---|
| `--config` | YAML config path (required) |
| `--run_name` | Output prefix under `models/lora/`, `data/progress/`, `data/rollouts/` |
| `--start_iter` | Iteration to resume from (default `0`); useful after a crash |

### Legacy scripts

`01_generate.py`, `02_score.py`, `03_build_dpo.py`, `04_train_dpo_lora.py`,
and `04_train_grpo_lora.py` are the older DPO and single-turn GRPO pipelines
from the puck-environment era. They still run, but the active path goes
through `05_train_grpo_branched.py`. The matching old configs are `loop.yaml`,
`loop_grpo.yaml`, and `smoke_test*.yaml`.

## Project structure

```
configs/
  loop_grpo_branched_slosh.yaml   # production: slosh env, branched multi-turn GRPO
  smoke_test_slosh.yaml           # smoke: 1 outer iter, G_1=G_2=2
  loop_grpo_branched.yaml         # older: branched multi-turn GRPO on the puck env
  loop_grpo.yaml / loop.yaml      # legacy single-turn GRPO / DPO
  smoke_test_grpo_branched.yaml   # legacy branched smoke (puck)
  prompt_template.txt
data/
  progress/<run>/                 # per-run CSVs and trainer_log_iter_*.json
  rollouts/<run>/                 # rollout JSONs per outer iteration
docs/
  paper.tex                       # method write-up
  findings.tex                    # dated experiment log
  methodology.tex
models/
  hf_base/iter_000/               # base weights downloaded by 00_setup_env.sh
  lora/<run>/iter_NNN/            # LoRA adapter per outer iteration
scripts/
  00_setup_env.sh                 # bootstrap
  05_train_grpo_branched.py       # current trainer subprocess
  08_eval.py                      # current eval subprocess
  09_run_loop.py                  # orchestrator
  01..04_*.py                     # legacy DPO + single-turn GRPO
  eval_frozen_baseline.py         # standalone N-turn eval
  gcp_*.sh                        # VM lifecycle + train/eval wrappers
  seed_tasks/
    slosh_train.jsonl             # 24 per-(seed, kappa) slosh training tasks
    slosh_eval.jsonl              # 12 held-out slosh eval tasks
    slosh_smoke_train.jsonl       # 2-row subset for smoke runs
    slosh_tank_env.py             # slosh-tank env source
    build_slosh_tasks.py          # regenerates the jsonls from the env
    puck_*.jsonl, puck_tilted_plane_env.py    # earlier puck env (legacy)
src/
  branched_trainer.py             # current trainer (TRL 0.28 GRPOTrainer subclass)
  multi_turn.py                   # rollout + per-turn advantages
  runner.py                       # sandboxed policy/main subprocess execution
  prompts.py                      # multi-turn chat template rendering
  diversity.py                    # DiversityCallback (W&B diagnostics)
  dataset.py, config.py, progress.py, hf_generator.py, gcs_sync.py
tests/                            # unit tests for trainer math, prompts, runner
```

## Configuration

The main config is `configs/loop_grpo_branched_slosh.yaml`. The knobs you're
most likely to touch:

| Section | Key | Default | Description |
|---|---|---|---|
| `loop` | `iterations` | `23` | Outer cycles (train + eval per cycle) |
| `loop` | `trainer` | `grpo_branched` | Selects `BranchedGRPOTrainer` |
| `branched` | `g1`, `g2` | `4`, `4` | Tree shape; `g1 * g2` leaves per prompt |
| `branched` | `max_turn_tokens` | `400` | Per-turn generation cap |
| `grpo` | `loss_type` | `dapo` | Token-summed surrogate; required for the per-turn decomposition |
| `grpo` | `beta`, `num_iterations` | `0`, `1` | Required for the per-turn decomposition |
| `grpo` | `epsilon`, `epsilon_high` | `0.2`, `0.28` | Asymmetric DAPO clip |
| `grpo` | `scale_rewards` | `batch` | Reward normalization |
| `training` | `per_device_train_batch_size` | `16` | = `g1 * g2`, one prompt's trajectories per inner step |
| `training` | `num_train_epochs` | `1` | Passes over the train task set per outer iter |
| `eval` | `n_turns` | `2` | Eval rollouts match the training protocol |
| `lora` | `r`, `alpha` | `16`, `32` | LoRA rank and scaling on q/k/v/o_proj |
| `training` | `report_to` | `wandb` | `wandb` or `none` |
| `paths` | `gcs_bucket` | _(unset)_ | Optional GCS checkpoint sync |

The per-turn loss decomposition is only exact under specific GRPO settings
(token-summed loss, `beta = 0`, `num_iterations = 1`); the trainer asserts
on entry rather than silently producing a different gradient. The paper's
§3 covers why.

## Tasks

Tasks live in `scripts/seed_tasks/*.jsonl`. One line per task, one scenario
per task — the per-(seed, kappa) split (see
[#49](https://github.com/nima-siboni/LoRA4VR/pull/49)) is what lets the
branched estimator's invariant hold: all $G_2$ continuations of a given
first-turn variant see byte-identical environment state.

```json
{
  "task_id": "slosh_tank_train_s0_k0.65",
  "env_class": "SloshTankEnv",
  "env_code": "from dataclasses import dataclass\n...",
  "state_spec": "state is a dict with keys: t, x, v, theta, omega, goal, steps_left",
  "action_spec": "action is a float — commanded cart acceleration in m/s^2",
  "goal": "Move the cart to the goal position and settle the liquid...",
  "scenarios": [
    {"scenario_id": "s0_k0.65", "seed": 0, "kappa": 0.65}
  ]
}
```

The full env source is embedded in `env_code` so the prompt template can
paste it into the model's context. `build_slosh_tasks.py` regenerates the
jsonls when the env source changes.

## Training on GCP

The whole VM lifecycle is wrapped by a few scripts. The shortest path:

```bash
# Train: walk GPU zones until one has capacity, then run training there
bash scripts/gcp_find_gpu_and_train.sh \
  configs/loop_grpo_branched_slosh.yaml slosh_grpo_run_1 --wait

# Eval against a specific iteration's adapter
bash scripts/gcp_find_gpu_and_eval.sh slosh_grpo_run_1 --wait -- --iter 22

# Manual teardown if you skipped --wait
bash scripts/gcp_destroy_vm.sh --yes
```

`gcp_find_gpu_and_train.sh` walks a preset zone list and tries
`gcloud compute instances create` in each one until something accepts. On
the first successful zone it chains into `gcp_train.sh`, which uploads the
repo, runs `00_setup_env.sh`, starts training in a detached tmux session,
and — with `--wait` — polls locally for a `/tmp/lora4vr-<run>.done` marker
before auto-fetching artifacts and destroying the VM. Non-stockout
failures (quota errors, for instance) abort the search rather than
leaving partial VMs behind.

`--accelerator` picks the GPU class:

```
l4         (default)  g2-standard-8     1× nvidia-l4              24 GiB
a100-40gb              a2-highgpu-1g     1× nvidia-tesla-a100      40 GiB
a100-80gb              a2-ultragpu-1g    1× nvidia-a100-80gb       80 GiB
h100                   a3-highgpu-1g     1× nvidia-h100-80gb       80 GiB
```

Override the zone preset with `LORA4VR_ZONE_LIST="europe-west4-b ..."`.

### Prerequisites

1. **gcloud SDK** authenticated against the billing project:
   ```bash
   gcloud auth login
   gcloud config set project <your-project-id>
   ```
2. **Compute Engine API** enabled
   (`gcloud services enable compute.googleapis.com`).
3. **GPU quota.** Fresh projects start with no GPU allowance per region.
   Request at least 1 GPU of the class you want at [Console → IAM & Admin
   → Quotas](https://console.cloud.google.com/iam-admin/quotas). Approval
   can take anywhere from hours to days.
4. **`.env`** in the repo root with `WANDB_API_KEY=...` if you want W&B
   logging.

### Driving the steps by hand

The wrappers are individually scriptable:

- `gcp_launch_vm.sh` — creates the VM, uploads the repo, runs
  `gcp_startup.sh` to seed the directory layout and download the base
  model.
- `gcp_train.sh <config> <run_name> [--wait]` — kicks off training in a
  detached tmux session, captures the real exit code via
  `${PIPESTATUS[0]}`, and (with `--wait`) auto-fetches plus destroys on
  completion.
- `gcp_fetch_artifacts.sh <run_name>` — pulls `models/lora/<run>`,
  `data/progress/<run>`, and `data/rollouts/<run>` from the VM to
  `~/Downloads/lora4vr_<run>/`.
- `gcp_destroy_vm.sh [--yes]` — deletes the VM and its boot disk.
- `gcp_eval.sh`, `gcp_find_gpu_and_eval.sh`, `gcp_fetch_eval.sh` — same
  pattern for eval-only runs.

The default zone for VM-lifecycle scripts is `europe-west3-a` (override
with `LORA4VR_ZONE`). The find-gpu wrappers ignore this and walk their
own preset list (`europe-west4`, `us-central1`, etc., depending on
`--accelerator`).

### Optional: monitoring with Weights & Biases

Two W&B runs are created per training session:

- The **inner trainer** logs per-inner-step metrics (loss, KL, rewards,
  and branched diagnostics like `branched/V0` and
  `branched/v1_signal_ratio`) when `training.report_to: wandb`.
- The **orchestrator** opens `<run_name>-orchestrator` and replays each
  outer iteration's CSV row (train success rate, mean and max reward,
  eval row) once that iteration finishes — giving an outer-iter view
  alongside the inner-step view.

One-time setup on the VM:

```bash
uv run wandb login   # paste API key from https://wandb.ai/authorize
```

`gcp_train.sh` picks up `WANDB_API_KEY` from your local `.env` and
forwards it inline so it never lands on disk.

### Optional: GCS checkpoint sync

For long runs on Spot VMs, set `paths.gcs_bucket` in the config to a
bucket in the same region as the VM. After each outer iteration,
`09_run_loop.py` rsyncs `models/lora/<run>/`, `data/progress/<run>/`,
and `data/rollouts/<run>/` to `gs://<bucket>/runs/<run>/`, and pulls
them back down on `--start_iter > 0`.

```yaml
paths:
  gcs_bucket: my-bucket-name
```

Same-region rsync is free; cross-region egress is $0.08/GB.

### Cost sanity check

| Item | Rate (europe-west4, on-demand) | Per 4–8 hr run |
|---|---|---|
| L4 VM (`g2-standard-8` + 1×L4) | ~$0.70/hr | $3–6 |
| A100-40GB VM (`a2-highgpu-1g`) | ~$3.67/hr | $15–30 |
| Boot disk (100 GB pd-balanced) | ~$0.14/day | ~$0.05 |
| Egress to workstation | $0.12/GB | $0.10–0.25 once |

Spot variants drop the VM rate by about 70% — worth a `Preemptible CPUs`
quota request if you're running many experiments.

## Tracking progress

Per-run files under `data/progress/<run_name>/`:

- `train.csv` — per-outer-iter train rollout stats
- `eval.csv` — per-outer-iter eval rollout stats
- `trainer_log_iter_NNN.json` — TRL's per-inner-step log history,
  replayed to the orchestrator W&B run at end-of-iteration

The orchestrator prints a formatted table after each outer iteration. To
print it manually:

```python
from src.progress import print_progress_table
print_progress_table(run_name="my_experiment")
```

## References

- Paper: `docs/paper.tex`
- Experiment log: `docs/findings.tex`
- Methodology notes: `docs/methodology.tex`
