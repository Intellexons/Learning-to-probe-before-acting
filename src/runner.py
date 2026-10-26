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
    # introspection: the model's code must not reach the env object or the script source
    "gc", "inspect", "dis", "weakref", "linecache", "tracemalloc", "faulthandler",
    "code", "codeop", "pdb", "trace", "atexit",
}

# Builtins the model's code may not name: they are the ways round the probe-run
# wrapper and the sandbox (reading attributes by string, reaching globals, running
# source, reading files).
_FORBIDDEN_NAMES = {
    "getattr", "setattr", "delattr", "vars", "globals", "locals", "eval", "exec",
    "compile", "__import__", "breakpoint", "open", "input",
}

# Per-scenario penalty for a sample that fails validation or errors at runtime.
# Must be lower than the worst-possible-but-running trajectory in any supported
# env, so GRPO's advantage ranks "broken code" strictly below "code that runs
# and does badly." Otherwise errored samples score 0.0 (no reward accumulated)
# and end up *higher* than running-but-failing samples, and GRPO rewards the
# model for producing errors. The crystallizer's worst running trajectory is a
# failure on the last step: -0.3 * 600 - 80 = -260.
ERROR_PENALTY_PER_SCENARIO = -300.0

# Cap on bytes of stdout returned from a `main` turn. Prevents a runaway
# print-loop from blowing the chat context.
MAIN_STDOUT_TRUNCATE = 2000

# Default subprocess timeouts (seconds). A final run may take the full
# 600-step horizon; a probe run may run many batches of its own. The brief
# tells the model the probe-run budget, so keep it in step with the prompt.
POLICY_DEFAULT_TIMEOUT = 30
MAIN_DEFAULT_TIMEOUT = 30

# How many trailing bytes of stderr to surface in error messages.
STDERR_TAIL_BYTES = 500

# Default max env steps per scenario rollout. Matches the crystallizer's own
# horizon (EnvConfig.max_steps = 600) and the scoring config defaults.
DEFAULT_MAX_STEPS = 600
DEFAULT_ENV_CLASS = "CrystallizerEnv"

# A final run counts as "fast" when it succeeds within this factor of the
# time-optimal batch time stored on the scenario (`oracle_steps`).
FAST_FACTOR = 1.25


# Fenced-block matcher. A completion is one code block, fenced or bare; the turn
# position decides what it is (probe code on every turn but the last, the
# controller on the last), so no directive is needed.
_FENCE_RE = re.compile(r"```(\w+)?\s*\n(.*?)```(?!\w)", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```\w*\s*\n(.*)\Z", re.DOTALL)


def extract_code(text: str) -> str:
    """The code in a model completion.

    The first fenced block if there is one; the text after an unclosed fence
    (a completion cut off by the token budget); otherwise the whole text,
    which is what a model that answers with bare code produces. A leading
    `# MODE: ...` line from the old protocol is just a comment and stays.
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return (m.group(2) or "").strip()
    m = _OPEN_FENCE_RE.search(text)
    if m:
        return (m.group(1) or "").strip()
    return text.strip()


def _check_sandbox_ast(tree: ast.AST) -> str | None:
    """Static checks shared by policy and probe code: no dangerous imports, none
    of the forbidden builtins, no dunder attribute or name access (the routes to
    the env object behind the probe-run wrapper, or to the script's constants)."""
    bad = _check_dangerous_imports(tree)
    if bad is not None:
        return bad
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id in _FORBIDDEN_NAMES or node.id.startswith("__")):
            return f"Forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"Dunder attribute access is not allowed: {node.attr}"
    return None


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

    bad = _check_sandbox_ast(tree)
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

    bad = _check_sandbox_ast(tree)
    if bad is not None:
        return False, bad

    return True, ""


# ---------------------------------------------------------------------------
# Sandboxed subprocess execution
# ---------------------------------------------------------------------------


_RUNNER_TEMPLATE_POLICY = '''\
import json
import math
import random

# --- Environment, built in a private namespace. The policy code below sees its
#     own names, `math`, `random` and nothing of the environment's source. ---
def _build_env_factory(env_src, env_class, hidden):
    ns = {{}}
    exec(compile(env_src, "<env>", "exec"), ns)
    cls = ns[env_class]
    hidden_params = ns["HiddenParams"](**hidden)
    def make(seed):
        return cls(seed=seed, hidden=hidden_params)
    return make

_make_env = _build_env_factory({env_code!r}, {env_class!r}, {hidden!r})
del _build_env_factory
try:
    del __file__
except NameError:
    pass
import builtins as _builtins
_builtins.open = None
del _builtins

# --- Policy code (the model's) ---
{policy_code}

# --- Runner ---
def _run_episode(seed, max_steps):
    env = _make_env(seed)
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

print(json.dumps(_run_episode({seed}, {max_steps})))
'''


# The probe-run env. The model's probe code sees `env`: the same env class with
# the SAME hidden parameters as the final run (the one that is scored), on a
# different random batch, behind a wrapper that offers exactly what the brief
# promises (reset, step, the outcome flags) and nothing else. The env source is
# executed in a private namespace, so the model's code has no class to
# instantiate and no constants to read: the way to learn about the hidden
# parameters is to run the env and watch what happens, as in a real plant.
_PROBE_ENV_SECTION = '''\
# --- Probe-run env, built in a private namespace ---
def _build_probe_env(env_src, env_class, seed, hidden):
    ns = {{}}
    exec(compile(env_src, "<env>", "exec"), ns)
    inner = ns[env_class](seed=seed, hidden=ns["HiddenParams"](**hidden))
    offered = ("reset", "step", "done", "showered", "failure_reason", "is_success", "is_unsafe")
    allowed_dunders = {{"__class__", "__doc__", "__repr__", "__str__", "__dir__",
                       "__init__", "__init_subclass__", "__subclasshook__"}}

    class ProbeEnv:
        # The probe-run env: same hidden parameters as the scored batch, a different batch.
        def __getattribute__(self, name):
            if name in allowed_dunders:
                return object.__getattribute__(self, name)
            if name in offered:
                return getattr(inner, name)
            raise AttributeError(
                f"{{name!r}} is not available in the probe run; env offers " + ", ".join(offered))
        def __setattr__(self, name, value):
            raise AttributeError("the probe-run env is read-only; use its methods")
        def __dir__(self):
            return list(offered)
        def __repr__(self):
            return "ProbeEnv(" + ", ".join(offered) + ")"
    return ProbeEnv()

env = _build_probe_env({env_code!r}, {env_class!r}, {seed}, {hidden!r})
del _build_probe_env
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
