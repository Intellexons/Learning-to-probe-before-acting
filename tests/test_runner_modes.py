"""Tests for the policy/main mode dispatch added in the multi-turn refactor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runner import (  # noqa: E402
    extract_code_with_mode,
    score_main_turn,
    score_policy_turn,
    score_sample,
    validate_main_code,
)


def test_extract_policy_fence():
    text = """Here is my plan.
```policy
def policy(state):
    return (state['gx'] - state['px'], 0.0)
```
"""
    mode, code = extract_code_with_mode(text)
    assert mode == "policy"
    assert "def policy" in code
    assert "```" not in code


def test_extract_main_fence():
    text = """I want to compute something.
```main
import math
print(math.sqrt(2))
```
"""
    mode, code = extract_code_with_mode(text)
    assert mode == "main"
    assert "import math" in code


def test_extract_python_fence_without_mode_is_invalid():
    """Plain ```python``` with no `# MODE:` directive is rejected."""
    text = """```python
def policy(state):
    return 0
```"""
    mode, _ = extract_code_with_mode(text)
    assert mode == "invalid"


def test_extract_python_fence_with_policy_directive():
    """```python``` plus a leading `# MODE: policy` comment dispatches to policy."""
    text = """```python
# MODE: policy
def policy(state):
    return 0
```"""
    mode, code = extract_code_with_mode(text)
    assert mode == "policy"
    assert "def policy" in code
    # The directive comment is preserved in the returned code (it's a no-op
    # Python comment, and stripping it would shift line numbers).
    assert "# MODE: policy" in code


def test_extract_python_fence_with_main_directive():
    text = """```python
# MODE: main
print(env.reset())
```"""
    mode, code = extract_code_with_mode(text)
    assert mode == "main"
    assert "print(env.reset())" in code


def test_extract_directive_tolerates_blank_lines_and_case():
    text = """```python

# mode:Main
print('hi')
```"""
    mode, _ = extract_code_with_mode(text)
    assert mode == "main"


def test_extract_directive_must_be_first_nonblank_line():
    """A `# MODE:` line later in the block does NOT count."""
    text = """```python
import math
# MODE: policy
def policy(state):
    return 0
```"""
    mode, _ = extract_code_with_mode(text)
    assert mode == "invalid"


def test_extract_unfenced_is_invalid():
    text = """def policy(state):
    return 0
"""
    mode, _ = extract_code_with_mode(text)
    assert mode == "invalid"


def test_extract_empty_is_invalid():
    assert extract_code_with_mode("") == ("invalid", "")


def test_extract_takes_first_fence():
    """If multiple fenced blocks are present, only the first is used."""
    text = """```policy
def policy(s): return 1
```
```main
print(2)
```
"""
    mode, _ = extract_code_with_mode(text)
    assert mode == "policy"


def test_score_policy_turn_alias_matches_score_sample():
    assert score_policy_turn is score_sample


def test_validate_main_allows_no_policy_def():
    """main turn doesn't need def policy; validate_main_code permits it."""
    is_valid, err = validate_main_code("import math\nprint(math.pi)\n")
    assert is_valid, err


def test_validate_main_rejects_dangerous_imports():
    is_valid, err = validate_main_code("import os\n")
    assert not is_valid
    assert "os" in err


def test_score_main_runs_simple_code_and_captures_stdout():
    code = "print('hello world')\nx = 1 + 2\nprint(x)\n"
    result = score_main_turn(code, timeout=5)
    assert result["error"] is None
    assert "hello world" in result["stdout"]
    assert "3" in result["stdout"]


def test_score_main_truncates_runaway_stdout():
    """A print loop should not blow the chat context."""
    code = "for _ in range(10000): print('x' * 100)\n"
    result = score_main_turn(code, timeout=5)
    # Truncated to MAIN_STDOUT_TRUNCATE bytes (default 2000)
    assert len(result["stdout"]) <= 2000


def test_score_main_rejects_invalid_syntax():
    result = score_main_turn("def (\n", timeout=5)
    assert result["error"] is not None
    assert "rejected" in result["stdout"]


def test_score_main_does_not_expose_env_class():
    """Without env_code, main() runs without the env class in its
    namespace. Trying to reference one should raise NameError."""
    code = "CrystallizerEnv()\n"
    result = score_main_turn(code, timeout=5)
    # The subprocess will have raised NameError; we get an error back.
    assert result["error"] is not None or "NameError" in result["stdout"]


# ---------------------------------------------------------------------------
# The probe-run env: same hidden parameters as the final run, masked
# ---------------------------------------------------------------------------

_PROBE_ENV_SRC = '''
class HiddenParams:
    def __init__(self, **kw):
        self.kw = kw


class _ProbeTestEnv:
    def __init__(self, seed=0, hidden=None):
        self.seed = seed
        self.hidden = hidden
        self.public_note = "visible"

    def reset(self, seed=None):
        return {"seed": float(self.seed)}, {}

    def step(self, action):
        return {"seed": float(self.seed)}, 0.0, True, False, {}

    def is_success(self):
        return False
'''
_SCENARIO = {"scenario_id": "s0", "seed": 3, "probe_seed": 44, "hidden": {"limit": 0.123}}


def _probe(code: str) -> dict:
    return score_main_turn(code, env_code=_PROBE_ENV_SRC, scenario=_SCENARIO,
                           env_class="_ProbeTestEnv", timeout=10)


def test_probe_env_runs_on_the_probe_batch_not_the_scoring_batch():
    out = _probe("s, _ = env.reset()\nprint('seed', s['seed'])\n")
    assert out["error"] is None, out
    assert "seed 44.0" in out["stdout"]


def test_probe_env_hides_the_hidden_parameters():
    out = _probe("print(env.hidden)\n")
    assert out["error"] is not None
    assert "not available in the probe run" in out["error"]
    assert "0.123" not in out["stdout"]


def test_probe_env_exposes_public_methods_and_attributes():
    out = _probe(
        "print(env.public_note)\n"
        "s, _ = env.reset()\n"
        "print('done', env.step(0.0)[2])\n"
        "print('success', env.is_success())\n"
        "print('hidden listed', 'hidden' in dir(env))\n"
        "print(repr(env))\n"
    )
    assert out["error"] is None, out
    assert "visible" in out["stdout"]
    assert "done True" in out["stdout"]
    assert "success False" in out["stdout"]
    assert "hidden listed False" in out["stdout"]
    assert "ProbeEnv(_ProbeTestEnv)" in out["stdout"]


def test_probe_env_is_read_only_and_blocks_private_names():
    out = _probe(
        "for stmt in ('env.hidden = 1', 'env._inner', 'env.__dict__'):\n"
        "    try:\n"
        "        eval(stmt) if '=' not in stmt else exec(stmt)\n"
        "        print('ALLOWED', stmt)\n"
        "    except AttributeError:\n"
        "        print('blocked', stmt)\n"
    )
    assert out["error"] is None, out
    assert "ALLOWED" not in out["stdout"]
    assert out["stdout"].count("blocked") == 3


def test_probe_run_cannot_read_its_own_script():
    """The script text holds the hidden values as a literal."""
    out = _probe(
        "try:\n"
        "    print(open(__file__).read()[:20])\n"
        "except Exception as e:\n"
        "    print('no source access:', type(e).__name__)\n"
    )
    assert out["error"] is None, out
    assert "no source access" in out["stdout"]
    assert "0.123" not in out["stdout"]


def test_score_main_without_scenario_still_builds_the_env():
    out = score_main_turn("s, _ = env.reset()\nprint('seed', s['seed'])\n",
                          env_code=_PROBE_ENV_SRC, env_class="_ProbeTestEnv", timeout=10)
    assert out["error"] is None, out
    assert "seed 0.0" in out["stdout"]
