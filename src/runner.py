import ast
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DANGEROUS_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket",
    "http", "urllib", "ftplib", "smtplib", "ctypes", "signal",
    "multiprocessing", "threading", "importlib", "builtins",
    "pickle", "shelve", "marshal",
}

# Per-scenario penalty for a sample that fails validation or errors at runtime.
# Must be lower than the worst-possible-but-running trajectory in any supported
# env, so GRPO's advantage ranks "broken code" strictly below "code that runs
# and does badly." Otherwise errored samples score 0.0 (no reward accumulated)
# and end up *higher* than running-but-failing samples (which integrate negative
# distance penalties), and GRPO rewards the model for producing errors.
ERROR_PENALTY_PER_SCENARIO = -200.0


def validate_code(code: str) -> tuple[bool, str]:
    """Static validation of generated policy code."""
    # Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check that def policy( exists
    has_policy = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "policy":
            has_policy = True
            break
    if not has_policy:
        return False, "No 'def policy(...)' function found"

    # Check for dangerous imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in DANGEROUS_MODULES:
                    return False, f"Dangerous import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split(".")[0]
                if top_module in DANGEROUS_MODULES:
                    return False, f"Dangerous import from: {node.module}"

    return True, ""


_RUNNER_TEMPLATE = '''\
import json
import sys
import math
import random

# --- Environment code ---
{env_code}

# --- Policy code ---
{policy_code}

# --- Runner ---
def run_episode(seed, max_steps):
    env = {env_class}(seed=seed)
    state = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(max_steps):
        try:
            action = policy(state)
        except Exception as e:
            return {{"reward": total_reward, "steps": steps, "error": f"policy() error: {{e}}"}}
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    return {{"reward": total_reward, "steps": steps, "error": None}}

seed = {seed}
max_steps = {max_steps}
result = run_episode(seed, max_steps)
print(json.dumps(result))
'''


def run_policy_in_subprocess(
    code: str,
    env_code: str,
    scenario: dict,
    timeout: int = 30,
    max_steps: int = 500,
    env_class: str = "CartPoleEnv",
) -> dict:
    """Execute policy in a subprocess with timeout."""
    script = _RUNNER_TEMPLATE.format(
        env_code=env_code,
        policy_code=code,
        seed=scenario.get("seed", 0),
        max_steps=max_steps,
        env_class=env_class,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[-500:]
            return {"reward": 0.0, "steps": 0, "error": f"Process error: {stderr}"}

        output = result.stdout.strip()
        return json.loads(output)

    except subprocess.TimeoutExpired:
        return {"reward": 0.0, "steps": 0, "error": "Timeout"}
    except (json.JSONDecodeError, ValueError) as e:
        return {"reward": 0.0, "steps": 0, "error": f"Output parse error: {e}"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def score_sample(
    code: str,
    task: dict,
    timeout: int = 30,
    max_steps: int = 500,
) -> dict:
    """Score a single code sample across all scenarios in the task.

    Failed validation and per-scenario runtime errors are scored at
    `ERROR_PENALTY_PER_SCENARIO` so that erroring out is always strictly
    worse than running and doing badly. See the constant's docstring above.
    """
    scenarios = task["scenarios"]
    n_scenarios = len(scenarios)

    is_valid, error_msg = validate_code(code)
    if not is_valid:
        return {
            "reward_total": ERROR_PENALTY_PER_SCENARIO * n_scenarios,
            "reward_per_scenario": {
                s["scenario_id"]: ERROR_PENALTY_PER_SCENARIO for s in scenarios
            },
            "errors": [error_msg],
            "ran_successfully": False,
        }

    reward_per_scenario = {}
    errors = []
    ran_any = False

    for scenario in scenarios:
        sid = scenario["scenario_id"]
        result = run_policy_in_subprocess(
            code, task["env_code"], scenario, timeout=timeout, max_steps=max_steps,
            env_class=task.get("env_class", "CartPoleEnv"),
        )
        if result["error"]:
            reward_per_scenario[sid] = ERROR_PENALTY_PER_SCENARIO
            errors.append(f"{sid}: {result['error']}")
        else:
            reward_per_scenario[sid] = result["reward"]
            ran_any = True

    reward_total = sum(reward_per_scenario.values())
    return {
        "reward_total": reward_total,
        "reward_per_scenario": reward_per_scenario,
        "errors": errors,
        "ran_successfully": ran_any,
    }
