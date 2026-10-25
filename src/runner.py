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
# and end up *higher* than running-but-failing samples, and GRPO rewards the
# model for producing errors. The crystallizer's worst running trajectory is
# -0.1 * 600 - 80 = -140.
ERROR_PENALTY_PER_SCENARIO = -200.0

# Cap on bytes of stdout returned from a `main` turn. Prevents a runaway
# print-loop from blowing the chat context.
MAIN_STDOUT_TRUNCATE = 2000

# Default subprocess timeouts (seconds). A final run may take the full
# 600-step horizon; a probe run may run a few full batches of its own.
POLICY_DEFAULT_TIMEOUT = 30
MAIN_DEFAULT_TIMEOUT = 10

# How many trailing bytes of stderr to surface in error messages.
STDERR_TAIL_BYTES = 500

# Default max env steps per scenario rollout. Matches the crystallizer's own
# horizon (EnvConfig.max_steps = 600) and the scoring config defaults.
DEFAULT_MAX_STEPS = 600
DEFAULT_ENV_CLASS = "CrystallizerEnv"

# A final run counts as "fast" when it succeeds within this factor of the
# time-optimal batch time stored on the scenario (`oracle_steps`).
FAST_FACTOR = 1.25


# Fenced-block matcher for multi-turn dispatch. Anything that doesn't yield
# a recognized mode is treated as `mode = "invalid"` and rejected at the floor.
_FENCE_RE = re.compile(r"```(\w+)?\s*\n(.*?)```(?!\w)", re.DOTALL)
_MODE_DIRECTIVE_RE = re.compile(
    r"^\s*#\s*MODE\s*:\s*(policy|main)\b", re.IGNORECASE
)
# Code-language tags we accept as a generic Python fence; mode is then read
# from a `# MODE: policy|main` directive on the first non-blank line.
_PY_TAGS = {"python", "py", ""}


def extract_code_with_mode(text: str) -> tuple[str, str]:
    """Parse a model completion into (mode, code).

    Modes:
      - "policy":  block whose first non-blank line is `# MODE: policy`
                   (or, for backward compat, a fence tagged ```policy```)
      - "main":    block whose first non-blank line is `# MODE: main`
                   (or, for backward compat, a fence tagged ```main```)
      - "invalid": no recognized fence, or a python/untagged fence without
                   a mode directive on its first non-blank line.

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
    if tag in _PY_TAGS:
        for line in code.splitlines():
            if not line.strip():
                continue
            m = _MODE_DIRECTIVE_RE.match(line)
            if m:
                return m.group(1).lower(), code
            break
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
    env = {env_class}(seed=seed, hidden=HiddenParams(**{hidden!r}))
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(max_steps):
        try:
            action = policy(state)
        except Exception as e:
            return {{"reward": total_reward, "steps": steps, "success": None, "error": f"policy() error: {{e}}"}}
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            break
    # Task outcome, distinct from "code ran without crashing". Envs that
    # define `is_success()` report a real bool; envs without it report None
    # and the caller treats it as not-a-success.
    if hasattr(env, "is_success"):
        try:
            success = bool(env.is_success())
        except Exception:
            success = None
    else:
        success = None
    return {{"reward": total_reward, "steps": steps, "success": success, "error": None}}

seed = {seed}
max_steps = {max_steps}
result = run_episode(seed, max_steps)
print(json.dumps(result))
'''


# The probe-run env. The model's `# MODE: main` code sees `env`: the same env
# class with the SAME hidden parameters as the final run (the one that is
# scored), on a different random batch. The hidden parameters are masked: the
# way to learn about them is to run the env and watch what happens, which is
# also all a real plant would allow.
_PROBE_ENV_SECTION = '''\
# --- Environment code ---
{env_code}

# --- Probe-run env ---
# `env` is the same env class with the SAME hidden parameters as the final run
# (the one that is scored), on a different random batch. The hidden parameters
# are not readable; learn about them by running the env and watching what
# happens. Anything else you build (new env instances with hidden values of
# your own choosing, helper functions) is up to you.
def _make_probe_env(_inner):
    _blocked = {{"hidden"}}
    _allowed = {{"__class__", "__doc__", "__repr__", "__str__", "__dir__",
                "__init__", "__init_subclass__", "__subclasshook__"}}

    class ProbeEnv:
        """Same class and hidden parameters as the final run; the hidden parameters themselves are not readable."""
        def __getattribute__(self, name):
            if name in _allowed:
                return object.__getattribute__(self, name)
            if name in _blocked or name.startswith("_"):
                raise AttributeError(f"{{name!r}} is not available in the probe run")
            return getattr(_inner, name)
        def __setattr__(self, name, value):
            raise AttributeError("the probe-run env is read-only; use its methods")
        def __dir__(self):
            return sorted(n for n in dir(_inner) if not n.startswith("_") and n not in _blocked)
        def __repr__(self):
            return "ProbeEnv(" + type(_inner).__name__ + ")"
    return ProbeEnv()

env = _make_probe_env({env_class}(seed={seed}, hidden=HiddenParams(**{hidden!r})))
del _make_probe_env
# The script text holds the hidden values as a literal; keep the model's code
# from reading its own source.
try:
    del __file__
except NameError:
    pass
import builtins as _builtins
_builtins.open = None
del _builtins
'''


_RUNNER_TEMPLATE_MAIN = '''\
import math
import random

{env_section}
# --- Probe-run code (the model's) ---
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


def _hidden_kwargs(scenario: dict | None) -> dict:
    """The scenario's hidden parameters as keyword arguments for HiddenParams."""
    return dict((scenario or {}).get("hidden") or {})


def run_policy_in_subprocess(
    code: str,
    env_code: str,
    scenario: dict,
    timeout: int = POLICY_DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
    env_class: str = DEFAULT_ENV_CLASS,
) -> dict:
    """Execute policy in a subprocess with timeout (the final run).

    The env is built as `env_class(seed=scenario["seed"],
    hidden=HiddenParams(**scenario["hidden"]))`.
    Returns {"reward", "steps", "success", "error"}.
    """
    script = _RUNNER_TEMPLATE_POLICY.format(
        env_code=env_code,
        policy_code=code,
        seed=scenario.get("seed", 0),
        hidden=_hidden_kwargs(scenario),
        max_steps=max_steps,
        env_class=env_class,
    )
    outcome = _run_script_in_subprocess(script, timeout=timeout)

    if outcome.timed_out:
        return {"reward": 0.0, "steps": 0, "success": None, "error": "Timeout"}
    if outcome.returncode != 0:
        return {
            "reward": 0.0, "steps": 0, "success": None,
            "error": f"Process error: {outcome.stderr_tail()}",
        }
    try:
        result = json.loads(outcome.stdout.strip())
        result.setdefault("success", None)
        result.setdefault("steps", 0)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        return {"reward": 0.0, "steps": 0, "success": None, "error": f"Output parse error: {e}"}


def run_main_in_subprocess(
    code: str,
    env_code: str = "",
    scenario: dict | None = None,
    env_class: str = DEFAULT_ENV_CLASS,
    timeout: int = MAIN_DEFAULT_TIMEOUT,
) -> dict:
    """Execute a `main` turn (the probe run) in the sandbox and capture stdout.

    When env_code is provided, the env class is injected and a pre-instantiated
    probe-run `env` is available to the model's code: same hidden parameters as
    the final run, batch seed `scenario["probe_seed"]` (falls back to
    `scenario["seed"]`), hidden parameters masked. Returns
    {"stdout": str, "error": str | None}. Stdout is truncated to
    `MAIN_STDOUT_TRUNCATE` bytes — the captured text is the model's "tool
    result" and an unbounded print loop would blow the chat context.
    """
    scenario = scenario or {}
    if env_code:
        env_section = _PROBE_ENV_SECTION.format(
            env_code=env_code,
            env_class=env_class,
            seed=scenario.get("probe_seed", scenario.get("seed", 0)),
            hidden=_hidden_kwargs(scenario),
        )
    else:
        env_section = ""
    script = _RUNNER_TEMPLATE_MAIN.format(env_section=env_section, main_code=code)
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


def _time_metrics(success: bool, steps: int, oracle_steps) -> tuple[bool, float]:
    """(fast, normalised time) for one scenario.

    fast: success within FAST_FACTOR * oracle_steps. norm_time: oracle_steps /
    own steps, 0 on failure. A scenario without `oracle_steps` has no time
    reference: fast == success and norm_time is 1 on success.
    """
    if not success:
        return False, 0.0
    if not oracle_steps or steps <= 0:
        return True, 1.0
    return steps <= FAST_FACTOR * oracle_steps, oracle_steps / steps


def error_floor_result(task: dict, errors: list[str]) -> dict:
    """The score dict for a sample that never ran (validation failure or no
    valid final policy): every scenario at ERROR_PENALTY_PER_SCENARIO, no
    success, no fast success, normalised time 0."""
    scenarios = task.get("scenarios", [])
    ids = [s["scenario_id"] for s in scenarios]
    return {
        "reward_total": ERROR_PENALTY_PER_SCENARIO * len(scenarios),
        "reward_per_scenario": {sid: ERROR_PENALTY_PER_SCENARIO for sid in ids},
        "success_per_scenario": {sid: False for sid in ids},
        "n_scenarios_success": 0,
        "fast_per_scenario": {sid: False for sid in ids},
        "n_scenarios_fast": 0,
        "norm_time_per_scenario": {sid: 0.0 for sid in ids},
        "mean_norm_time": 0.0,
        "steps_per_scenario": {sid: 0 for sid in ids},
        "errors": list(errors),
        "ran_successfully": False,
    }


def score_sample(
    code: str,
    task: dict,
    timeout: int = POLICY_DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict:
    """Score a single code sample across all scenarios in the task.

    Returns the reward per scenario and three task-outcome metrics per
    scenario: `success` (the env's own `is_success()`), `fast` (success within
    FAST_FACTOR of the scenario's `oracle_steps`) and `norm_time`
    (oracle_steps / own steps, 0 on failure).

    Failed validation and per-scenario runtime errors are scored at
    `ERROR_PENALTY_PER_SCENARIO` so that erroring out is always strictly
    worse than running and doing badly. See the constant's docstring above.
    """
    scenarios = task["scenarios"]

    is_valid, error_msg = validate_code(code)
    if not is_valid:
        return error_floor_result(task, [error_msg])

    reward_per_scenario: dict = {}
    success_per_scenario: dict = {}
    fast_per_scenario: dict = {}
    norm_time_per_scenario: dict = {}
    steps_per_scenario: dict = {}
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
            success_per_scenario[sid] = False  # didn't even run cleanly
            fast_per_scenario[sid] = False
            norm_time_per_scenario[sid] = 0.0
            steps_per_scenario[sid] = int(result.get("steps") or 0)
            errors.append(f"{sid}: {result['error']}")
        else:
            reward_per_scenario[sid] = result["reward"]
            # `success` is the env's task-outcome check; None if the env
            # doesn't define is_success(). Unknown counts as not-a-success.
            success = bool(result.get("success") or False)
            steps = int(result.get("steps") or 0)
            fast, norm_time = _time_metrics(success, steps, scenario.get("oracle_steps"))
            success_per_scenario[sid] = success
            fast_per_scenario[sid] = fast
            norm_time_per_scenario[sid] = norm_time
            steps_per_scenario[sid] = steps
            ran_any = True

    n = max(len(scenarios), 1)
    return {
        "reward_total": sum(reward_per_scenario.values()),
        "reward_per_scenario": reward_per_scenario,
        "success_per_scenario": success_per_scenario,
        "n_scenarios_success": sum(success_per_scenario.values()),
        "fast_per_scenario": fast_per_scenario,
        "n_scenarios_fast": sum(fast_per_scenario.values()),
        "norm_time_per_scenario": norm_time_per_scenario,
        "mean_norm_time": sum(norm_time_per_scenario.values()) / n,
        "steps_per_scenario": steps_per_scenario,
        "errors": errors,
        "ran_successfully": ran_any,
    }


# Alias for clarity in multi-turn code paths. `score_policy_turn` is the same
# operation as `score_sample` — both score a `policy(state)` body across all
# of a task's scenarios. Kept as a separate name so callers reading a
# branched-rollout function can see at a glance which mode is being scored.
score_policy_turn = score_sample


def score_main_turn(
    code: str,
    env_code: str = "",
    scenario: dict | None = None,
    env_class: str = DEFAULT_ENV_CLASS,
    timeout: int = MAIN_DEFAULT_TIMEOUT,
) -> dict:
    """Score a `main` turn (the probe run): execute the code in the sandbox
    and return its stdout.

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
    result = run_main_in_subprocess(
        code, env_code=env_code, scenario=scenario, env_class=env_class, timeout=timeout,
    )
    if result["error"]:
        return {
            "stdout": f"{result['stdout']}\n[main error: {result['error']}]".strip(),
            "error": result["error"],
        }
    return {"stdout": result["stdout"], "error": None}
