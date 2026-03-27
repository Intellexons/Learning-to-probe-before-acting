# Policy DPO Loop

Iterative post-training loop that teaches an LLM to write better policy code through self-play. The model generates Python `policy(state) -> action` functions, executes them in an environment, and uses DPO (Direct Preference Optimization) with LoRA to improve from its own outputs — repeating the cycle with the merged model each iteration.

**Base model:** Qwen2.5-Coder 3B Instruct (configurable in `configs/loop.yaml`)

## How it works

```
For each iteration:
  1. Generate code candidates with HF model       (01_generate.py)
  2. Score them by running in the environment      (02_score.py)
  3. Build preference pairs from scores            (03_build_dpo.py)
  4. Train a LoRA adapter with DPO                 (04_train_dpo_lora.py)
  5. Evaluate on held-out tasks                    (08_eval.py)
```

Generation and evaluation load the base model and merge all previous LoRA adapters in GPU memory (bfloat16), so there is no need for disk-based merging, GGUF conversion, or Ollama.

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

### `scripts/09_run_loop.py` — Main orchestrator

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

### `scripts/00_setup_env.sh` — One-time setup

Installs dependencies, downloads HF weights to `models/hf_base/iter_000/`, and creates the directory structure.

```bash
bash scripts/00_setup_env.sh
```

---

### `scripts/01_generate.py` — Generate code samples

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

---

### `scripts/02_score.py` — Score generated samples

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

---

### `scripts/03_build_dpo.py` — Build DPO preference pairs

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

---

### `scripts/04_train_dpo_lora.py` — Train LoRA adapter with DPO

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

---

### `scripts/05_merge_lora.py` — Merge LoRA into base weights (standalone utility)

Loads the original base model and sequentially applies all LoRA adapters (0 through current iteration) to produce a fully merged checkpoint. Done in float32 on CPU for numerical stability. Not called by the main loop — useful for exporting a final model.

```bash
uv run python scripts/05_merge_lora.py --iter 0 --config configs/loop.yaml
```

| Flag | Description |
|------|-------------|
| `--iter` | Iteration number (required) |
| `--config` | YAML config path |
| `--run_name` | Run name for output directory scoping |

**Output:** `models/hf_base/[<run_name>/]iter_001/`

---

### `scripts/06_convert_to_gguf.sh` — Convert to GGUF (standalone utility)

Converts merged HF weights to GGUF format using llama.cpp, then quantizes (default: Q8_0). Not called by the main loop — useful for exporting to Ollama.

```bash
bash scripts/06_convert_to_gguf.sh <iter_num> [quantization] [run_name]
```

---

### `scripts/07_ollama_create.sh` — Register model in Ollama (standalone utility)

Creates an Ollama model from the GGUF file with the ChatML template baked in. Not called by the main loop.

```bash
bash scripts/07_ollama_create.sh <iter_num> [quantization] [run_name]
```

---

### `scripts/08_eval.py` — Evaluate on held-out tasks

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

## Tracking progress

Each iteration appends to CSV files under `data/progress/<run_name>/`:

- **train.csv** — success rate, mean/max reward from generation+scoring
- **eval.csv** — eval task performance per iteration

The loop prints a formatted table after each iteration. You can also print it manually:

```python
from src.progress import print_progress_table
print_progress_table(run_name="my_experiment")
```
