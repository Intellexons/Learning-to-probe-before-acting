"""Arm A/B — frozen Qwen2.5-Coder-3B baseline on slosh task.

Arm A (--turns 1): generates N one-turn policy completions, evaluates at
κ ∈ {0.65, 0.80, 1.00} over EVAL_SEEDS. Compares to oracle and blind curves.

Arm B (--turns 2): same but generates a two-turn (main + policy) rollout.
"""
import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_SEED_TASKS_DIR = Path(__file__).resolve().parent / "seed_tasks"
sys.path.insert(0, str(_SEED_TASKS_DIR))

from src.hf_generator import load_model_for_generation, generate_code, generate_code_from_messages
from src.prompts import load_template, render_prompt
from src.runner import validate_code, run_main_in_subprocess, extract_code_with_mode, ERROR_PENALTY_PER_SCENARIO
from src.dataset import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

KAPPAS = [0.65, 0.80, 1.00]
MAX_STEPS = 105
EPISODE_TIMEOUT = 15

_RUNNER_TMPL = """\
import json, math, random
{env_code}
{policy_code}
def _run(seed, kappa, max_steps):
    hidden = HiddenParams(actuator_gain=kappa)
    env = SloshTankEnv(seed=seed, hidden=hidden)
    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        try:
            action = policy(state)
        except Exception as e:
            return {{"reward": total_reward, "success": False, "error": str(e)}}
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    return {{"reward": float(total_reward), "success": env.is_success(), "error": None}}
print(json.dumps(_run({seed}, {kappa!r}, {max_steps})))
"""


def _run_at_kappa(policy_code: str, env_code: str, seed: int, kappa: float) -> dict:
    script = _RUNNER_TMPL.format(
        env_code=env_code,
        policy_code=policy_code,
        seed=seed,
        kappa=kappa,
        max_steps=MAX_STEPS,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    _err = lambda msg: {"success": False, "reward": ERROR_PENALTY_PER_SCENARIO, "error": msg}
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=EPISODE_TIMEOUT,
        )
        if proc.returncode != 0:
            return _err(proc.stderr[-300:])
        try:
            result = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, ValueError) as e:
            return _err(f"output parse error: {e}")
        # Policy crashed mid-episode — apply same penalty as GRPO scorer
        if result.get("error"):
            result["reward"] = ERROR_PENALTY_PER_SCENARIO
        return result
    except subprocess.TimeoutExpired:
        return _err("timeout")
    except Exception as e:
        return _err(str(e))
    finally:
        Path(tmp).unlink(missing_ok=True)


def eval_policy(policy_code: str, env_code: str, seeds: list[int]) -> dict:
    """Return per-kappa success_rate and mean_reward."""
    results = {}
    for kappa in KAPPAS:
        successes, rewards = 0, []
        for seed in seeds:
            r = _run_at_kappa(policy_code, env_code, seed, kappa)
            successes += int(r.get("success", False))
            rewards.append(r.get("reward", 0.0))
        results[kappa] = {
            "success_rate": successes / len(seeds),
            "mean_reward": sum(rewards) / len(rewards),
        }
    return results


def compute_reference_rates(seeds: list[int]) -> tuple[dict, dict]:
    """Return (blind_rates, oracle_rates) at each kappa using reference policies."""
    from slosh_tank_env import HiddenParams
    from slosh_tank_utils import run_episode, nominal_fast_policy

    def make_oracle(k: float):
        def pol(state):
            x, v, goal = state["x"], state["v"], state["goal"]
            theta, omega = state["theta"], state["omega"]
            a = (1.25 * (goal - x) - 1.15 * v + 1.30 * theta + 0.45 * omega) / k
            return max(-1.0, min(1.0, a))
        return pol

    blind_rates, oracle_rates = {}, {}
    for kappa in KAPPAS:
        hidden = HiddenParams(actuator_gain=kappa)
        oracle_pol = make_oracle(kappa)
        b_ok = o_ok = 0
        for seed in seeds:
            if run_episode(nominal_fast_policy, seed=seed, hidden=hidden)["success"]:
                b_ok += 1
            if run_episode(oracle_pol, seed=seed, hidden=hidden)["success"]:
                o_ok += 1
        blind_rates[kappa] = b_ok / len(seeds)
        oracle_rates[kappa] = o_ok / len(seeds)
    return blind_rates, oracle_rates


def generate_one_shot(model, tokenizer, prompt: str, temperature: float, max_tokens: int) -> dict:
    """Generate a single-turn policy. Returns dict with code, success, error."""
    sample = generate_code(model, tokenizer, prompt, temperature=temperature, max_tokens=max_tokens)
    if not sample["success"]:
        return sample
    # Single-turn: model may respond raw or with python fence — extract code
    code = sample["code"]
    is_valid, err = validate_code(code)
    if not is_valid:
        return {"code": code, "success": False, "error": err}
    return {"code": code, "success": True, "error": None}


def generate_two_turn(
    model, tokenizer,
    task: dict, template: str,
    kappa: float,
    calibration_seed: int,
    temperature: float, max_tokens: int,
) -> dict:
    """Generate a two-turn (main + policy) rollout calibrated to a specific kappa.

    Turn 1 (main): model receives the task prompt with env source and a
    pre-instantiated `env` object (same kappa/seed as evaluation) to run
    calibration experiments. Stdout is fed back as the tool result.
    Turn 2 (policy): model writes a policy informed by the calibration.
    """
    prompt_t1 = render_prompt(template, task, turn_idx=1, total_turns=2)
    t1 = generate_code(model, tokenizer, prompt_t1, temperature=temperature, max_tokens=max_tokens)
    if not t1["success"]:
        return {"code": "", "success": False, "error": f"turn1: {t1['error']}", "t1_raw": t1["raw_response"]}

    t1_mode, t1_code = extract_code_with_mode(t1["raw_response"])
    if t1_mode == "policy":
        is_valid, err = validate_code(t1_code)
        if is_valid:
            return {"code": t1_code, "success": True, "error": None, "t1_raw": t1["raw_response"], "t1_mode": "policy"}

    if t1_mode == "main":
        main_result = run_main_in_subprocess(
            t1_code,
            env_code=task["env_code"],
            kappa=kappa,
            seed=calibration_seed,
            env_class=task.get("env_class", "SloshTankEnv"),
        )
        tool_output = main_result["stdout"] or "[empty output]"
    else:
        tool_output = "[turn 1 produced no recognized code block]"

    prompt_t2 = render_prompt(template, task, turn_idx=2, total_turns=2)
    t2_messages = [
        {"role": "user", "content": prompt_t1},
        {"role": "assistant", "content": t1["raw_response"]},
        {"role": "user", "content": f"{prompt_t2}\n\n[Tool output from turn 1]\n{tool_output}"},
    ]
    t2 = generate_code_from_messages(model, tokenizer, t2_messages, temperature=temperature, max_tokens=max_tokens)
    if not t2["success"]:
        return {"code": "", "success": False, "error": f"turn2: {t2['error']}", "t1_raw": t1["raw_response"], "t1_mode": t1_mode}

    _, t2_code = extract_code_with_mode(t2["raw_response"])
    if not t2_code:
        t2_code = t2["code"]
    is_valid, err = validate_code(t2_code)
    if not is_valid:
        return {"code": t2_code, "success": False, "error": err, "t1_raw": t1["raw_response"], "t1_mode": t1_mode}
    return {"code": t2_code, "success": True, "error": None, "t1_raw": t1["raw_response"], "t1_mode": t1_mode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/hf_base/iter_000")
    parser.add_argument("--task-file", default="scripts/seed_tasks/slosh_train.jsonl")
    parser.add_argument("--turns", type=int, default=1, choices=[1, 2],
                        help="1 = arm A (one-shot), 2 = arm B (two-turn)")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    arm_label = "A" if args.turns == 1 else "B"
    out_dir = Path(args.out_dir or f"results/arm_{arm_label.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_seeds = list(range(args.n_seeds))

    tasks = load_jsonl(args.task_file)
    task = tasks[0]
    env_code = task["env_code"]
    template = load_template("configs/prompt_template.txt")

    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_for_generation(
        base_model_path=args.model_path,
        lora_base=None,
        iter_num=0,
    )

    # Compute reference rates once
    logger.info(f"Computing blind/oracle reference rates over {args.n_seeds} seeds...")
    blind_rates, oracle_rates = compute_reference_rates(eval_seeds)
    for kappa in KAPPAS:
        logger.info(f"  κ={kappa:.2f}  blind={blind_rates[kappa]*100:.1f}%  oracle={oracle_rates[kappa]*100:.1f}%")

    # For arm B: generate n_samples per kappa, each calibrated to that kappa.
    # For arm A: generate n_samples once and evaluate across all kappas.
    kappa_successes = {k: [] for k in KAPPAS}
    kappa_rewards = {k: [] for k in KAPPAS}
    per_sample_results = []

    if args.turns == 1:
        prompt = render_prompt(template, task, turn_idx=1, total_turns=1)
        samples = []
        for i in range(args.n_samples):
            logger.info(f"Generating sample {i+1}/{args.n_samples} (arm A)")
            sample = generate_one_shot(model, tokenizer, prompt, args.temperature, args.max_tokens)
            samples.append(sample)
            if not sample["success"]:
                logger.warning(f"  Sample {i+1} failed: {sample['error']}")

        for i, sample in enumerate(samples):
            if not sample["success"] or not sample.get("code"):
                for kappa in KAPPAS:
                    kappa_successes[kappa].append(0.0)
                    kappa_rewards[kappa].append(0.0)
                per_sample_results.append({
                    "sample_id": i, "kappa": None, "valid": False,
                    "error": sample.get("error"), "t1_raw": "", "per_kappa": {},
                })
                continue
            logger.info(f"Evaluating sample {i+1}/{args.n_samples} × {len(KAPPAS)} κ × {args.n_seeds} seeds")
            per_kappa = eval_policy(sample["code"], env_code, eval_seeds)
            for kappa in KAPPAS:
                kappa_successes[kappa].append(per_kappa[kappa]["success_rate"])
                kappa_rewards[kappa].append(per_kappa[kappa]["mean_reward"])
            per_sample_results.append({
                "sample_id": i, "kappa": None, "valid": True,
                "code": sample["code"], "t1_raw": "",
                "per_kappa": {str(k): v for k, v in per_kappa.items()},
            })
            logger.info("  " + "  ".join(
                f"κ={k:.2f}:{per_kappa[k]['success_rate']*100:.1f}%" for k in KAPPAS))
    else:
        # Arm B: one generation batch per kappa, calibrated to that kappa
        sample_idx = 0
        for kappa in KAPPAS:
            for i in range(args.n_samples):
                logger.info(f"Generating κ={kappa:.2f} sample {i+1}/{args.n_samples} (arm B, cal_seed={i})")
                sample = generate_two_turn(
                    model, tokenizer, task, template,
                    kappa=kappa, calibration_seed=i,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
                if not sample["success"]:
                    logger.warning(f"  Failed: {sample['error']}")
                    kappa_successes[kappa].append(0.0)
                    kappa_rewards[kappa].append(0.0)
                    per_sample_results.append({
                        "sample_id": sample_idx, "kappa": kappa, "valid": False,
                        "error": sample.get("error"),
                        "t1_raw": sample.get("t1_raw", ""),
                        "t1_mode": sample.get("t1_mode", ""),
                        "per_kappa": {},
                    })
                else:
                    result = _run_at_kappa(sample["code"], env_code, seed=0, kappa=kappa)
                    # Evaluate across all n_seeds
                    successes, rewards = 0, []
                    for seed in eval_seeds:
                        r = _run_at_kappa(sample["code"], env_code, seed=seed, kappa=kappa)
                        successes += int(r.get("success", False))
                        rewards.append(r.get("reward", 0.0))
                    sr = successes / len(eval_seeds)
                    mr = sum(rewards) / len(rewards)
                    kappa_successes[kappa].append(sr)
                    kappa_rewards[kappa].append(mr)
                    per_sample_results.append({
                        "sample_id": sample_idx, "kappa": kappa, "valid": True,
                        "code": sample["code"],
                        "t1_raw": sample.get("t1_raw", ""),
                        "t1_mode": sample.get("t1_mode", ""),
                        "success_rate": sr, "mean_reward": mr,
                        "per_kappa": {str(kappa): {"success_rate": sr, "mean_reward": mr}},
                    })
                    logger.info(f"  κ={kappa:.2f}: success={sr*100:.1f}%  reward={mr:.1f}")
                sample_idx += 1

    qwen_rates = {
        k: (sum(kappa_successes[k]) / len(kappa_successes[k]) if kappa_successes[k] else 0.0)
        for k in KAPPAS
    }

    summary = {
        "arm": arm_label,
        "turns": args.turns,
        "model": args.model_path,
        "n_samples": args.n_samples,
        "n_seeds": args.n_seeds,
        "qwen": {str(k): v for k, v in qwen_rates.items()},
        "blind_pd": {str(k): v for k, v in blind_rates.items()},
        "oracle_pd": {str(k): v for k, v in oracle_rates.items()},
        "per_sample": per_sample_results,
    }
    summary_path = out_dir / f"arm_{arm_label.lower()}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {summary_path}")

    # Print table
    label = f"Qwen {args.turns}-shot (frozen)"
    print(f"\n=== Arm {arm_label}: {label} ===")
    print(f"{'κ':>6}  {label:>20}  {'Blind PD':>10}  {'Oracle PD':>10}")
    for kappa in KAPPAS:
        print(
            f"{kappa:>6.2f}  {qwen_rates[kappa]*100:>19.1f}%  "
            f"{blind_rates[kappa]*100:>9.1f}%  {oracle_rates[kappa]*100:>9.1f}%"
        )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(KAPPAS, [oracle_rates[k] * 100 for k in KAPPAS],
            "-", color="darkorange", label="oracle (gains / κ)")
    ax.plot(KAPPAS, [blind_rates[k] * 100 for k in KAPPAS],
            "-", color="steelblue", label="blind (nominal PD)")
    ax.plot(KAPPAS, [qwen_rates[k] * 100 for k in KAPPAS],
            "-o", color="forestgreen", label=label)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("actuator gain κ")
    ax.set_ylabel("success rate (%)")
    ax.set_title(f"Arm {arm_label}: {label} vs baselines")
    ax.legend()
    ax.set_ylim(-5, 105)
    ax.set_xticks(KAPPAS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / f"arm_{arm_label.lower()}_success_rate.png"
    fig.savefig(str(plot_path), dpi=150)
    logger.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
