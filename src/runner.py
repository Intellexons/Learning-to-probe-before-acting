import ast
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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

# Cap on bytes of stdout returned from a `main` turn. Prevents a runaway
# print-loop from blowing the chat context.
MAIN_STDOUT_TRUNCATE = 2000

# Default subprocess timeouts (seconds). policy turns may run a full
# horizon of env steps; main turns are pure scratchpad math and should
# return quickly.
POLICY_DEFAULT_TIMEOUT = 30
MAIN_DEFAULT_TIMEOUT = 10

# How many trailing bytes of stderr to surface in error messages.
STDERR_TAIL_BYTES = 500

# Default max env steps per scenario rollout. Matches scoring config defaults.
DEFAULT_MAX_STEPS = 500
DEFAULT_ENV_CLASS = "CartPoleEnv"


# Recognized fence language tags for multi-turn dispatch. Anything else (or
# no fence at all) is treated as `mode = "invalid"` and rejected at the floor.
_FENCE_RE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Code parsing + static validation
# ---------------------------------------------------------------------------


def extract_code_with_mode(text: str) -> tuple[str, str]:
    """Parse a model completion into (mode, code).

    Modes:
      - "policy": fenced block tagged ```policy``` — code defines `def policy(state)`
      - "main":   fenced block tagged ```main```   — arbitrary scratchpad code
      - "invalid": no recognized fence (untagged, wrong tag, missing fence)

    The first matching fenced block wins. If multiple fences are present, only
    the first is considered — the runner does NOT execute multiple code blocks
    per turn.
    """
    if not text:
        return "invalid", ""
    match = _FENCE_RE.search(text)
    if match is None:
        return "invalid", ""
    tag = (match.group(1) or "").lower()
    code = (match.group(2) or "").strip()
    if tag in ("policy", "main"):
        return tag, code
    return "invalid", ""


def _check_dangerous_imports(tree: ast.AST) -> str | None:
    """Return an error message if any dangerous module is imported, else None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in DANGEROUS_MODULES:
                    return f"Dangerous import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in DANGEROUS_MODULES:
                return f"Dangerous import from: {node.module}"
    return None


def _has_function_def(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in ast.walk(tree)
    )


def validate_code(code: str) -> tuple[bool, str]:
    """Static validation of generated policy code.

    Requires `def policy(...)` to exist, and rejects any dangerous imports.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    if not _has_function_def(tree, "policy"):
        return False, "No 'def policy(...)' function found"

    bad = _check_dangerous_imports(tree)
    if bad is not None:
        return False, bad

    return True, ""


def validate_main_code(code: str) -> tuple[bool, str]:
    """Static validation for `main` turn code.

    Same dangerous-import allowlist as policy, but `def policy` is NOT
    required (the whole point of main is arbitrary scratchpad).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    bad = _check_dangerous_imports(tree)
    if bad is not None:
        return False, bad

    return True, ""


# ---------------------------------------------------------------------------
# Sandboxed subprocess execution
# ---------------------------------------------------------------------------


_RUNNER_TEMPLATE_POLICY = '''\
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


# `main` turn template: NO env class, NO scenario loop. The model uses this
# for derivations / scratchpad math / printing intermediate variables it
# wants to read in turn 2. Per the 2026-04-25 design decision, exposing the
# env class to main() would let the LLM substitute empirical probing of
# hidden parameters for analytical reasoning, which defeats the
# partial-observability point. Reversing this is a one-line change.
_RUNNER_TEMPLATE_MAIN = '''\
import math
import random

# --- Main code (model's scratchpad) ---
{main_code}
'''


@dataclass(frozen=True)
class _SubprocessOutcome:
    """Normalized result of running a script in a subprocess.

    `timed_out=True` means the process exceeded its timeout (and was killed);
    in that case stdout/stderr are empty and returncode is undefined. For
    completed processes, returncode is the actual exit code.
    """
    timed_out: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def stderr_tail(self) -> str:
        return (self.stderr or "").strip()[-STDERR_TAIL_BYTES:]


def _run_script_in_subprocess(script: str, timeout: int) -> _SubprocessOutcome:
    """Write a script to a temp file, run it under the current Python with
    a timeout, capture stdout/stderr. Always cleans up the temp file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return _SubprocessOutcome(
            timed_out=False,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
    except subprocess.TimeoutExpired:
        return _SubprocessOutcome(
            timed_out=True, returncode=-1, stdout="", stderr=""
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_policy_in_subprocess(
    code: str,
    env_code: str,
    scenario: dict,
    timeout: int = POLICY_DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
    env_class: str = DEFAULT_ENV_CLASS,
) -> dict:
    """Execute policy in a subprocess with timeout."""
    script = _RUNNER_TEMPLATE_POLICY.format(
        env_code=env_code,
        policy_code=code,
        seed=scenario.get("seed", 0),
        max_steps=max_steps,
        env_class=env_class,
    )
    outcome = _run_script_in_subprocess(script, timeout=timeout)

    if outcome.timed_out:
        return {"reward": 0.0, "steps": 0, "error": "Timeout"}
    if outcome.returncode != 0:
        return {
            "reward": 0.0, "steps": 0,
            "error": f"Process error: {outcome.stderr_tail()}",
        }
    try:
        return json.loads(outcome.stdout.strip())
    except (json.JSONDecodeError, ValueError) as e:
        return {"reward": 0.0, "steps": 0, "error": f"Output parse error: {e}"}


def run_main_in_subprocess(code: str, timeout: int = MAIN_DEFAULT_TIMEOUT) -> dict:
    """Execute a `main` turn in the sandbox and capture stdout.

    Returns: {"stdout": str, "error": str | None}. Stdout is truncated to
    `MAIN_STDOUT_TRUNCATE` bytes — the captured text is the model's "tool
    result" and an unbounded print loop would blow the chat context.
    """
    script = _RUNNER_TEMPLATE_MAIN.format(main_code=code)
    outcome = _run_script_in_subprocess(script, timeout=timeout)

    if outcome.timed_out:
        return {"stdout": "", "error": "Timeout"}
    stdout = outcome.stdout[:MAIN_STDOUT_TRUNCATE]
    if outcome.returncode != 0:
        return {"stdout": stdout, "error": f"Process error: {outcome.stderr_tail()}"}
    return {"stdout": stdout, "error": None}


# ---------------------------------------------------------------------------
# High-level scoring entry points
# ---------------------------------------------------------------------------


def score_sample(
    code: str,
    task: dict,
    timeout: int = POLICY_DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
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

    reward_per_scenario: dict = {}
    errors: list[str] = []
    ran_any = False
    env_class = task.get("env_class", DEFAULT_ENV_CLASS)
    env_code = task["env_code"]

    for scenario in scenarios:
        sid = scenario["scenario_id"]
        result = run_policy_in_subprocess(
            code, env_code, scenario,
            timeout=timeout, max_steps=max_steps, env_class=env_class,
        )
        if result["error"]:
            reward_per_scenario[sid] = ERROR_PENALTY_PER_SCENARIO
            errors.append(f"{sid}: {result['error']}")
        else:
            reward_per_scenario[sid] = result["reward"]
            ran_any = True

    return {
        "reward_total": sum(reward_per_scenario.values()),
        "reward_per_scenario": reward_per_scenario,
        "errors": errors,
        "ran_successfully": ran_any,
    }


# Alias for clarity in multi-turn code paths. `score_policy_turn` is the same
# operation as `score_sample` — both score a `policy(state)` body across all
# of a task's scenarios. Kept as a separate name so callers reading a
# branched-rollout function can see at a glance which mode is being scored.
score_policy_turn = score_sample


def score_main_turn(code: str, timeout: int = MAIN_DEFAULT_TIMEOUT) -> dict:
    """Score a `main` turn — execute the code in the sandbox, return stdout.

    Unlike `score_policy_turn`, this does NOT produce a numeric reward. The
    "value" of a main turn is the captured stdout text returned to the model
    in the next user message. Per-trajectory reward credit for a main turn
    comes from the branched-rollout MC estimator: turn-1's value V̂₁ is the
    mean reward over the G₂ continuations, regardless of whether turn 1 was
    `policy` or `main`. So the only thing this function must report is the
    text the model gets to read next turn.

    A validation failure or runtime error returns an "[error: ...]" stdout
    rather than raising, so the chat history stays well-formed.
    """
    is_valid, error_msg = validate_main_code(code)
    if not is_valid:
        return {"stdout": f"[main rejected: {error_msg}]", "error": error_msg}
    result = run_main_in_subprocess(code, timeout=timeout)
    if result["error"]:
        return {
            "stdout": f"{result['stdout']}\n[main error: {result['error']}]".strip(),
            "error": result["error"],
        }
    return {"stdout": result["stdout"], "error": None}
