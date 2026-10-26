"""The runner's code extraction, sandbox checks and the probe-run env."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runner import (  # noqa: E402
    extract_code,
    score_main_turn,
    score_policy_turn,
    score_sample,
    validate_code,
    validate_main_code,
)


# ---------------------------------------------------------------------------
# extract_code: one block, fenced or bare; the turn position says what it is
# ---------------------------------------------------------------------------


def test_extract_fenced_block_any_tag():
    for tag in ("python", "py", "", "policy", "main"):
        text = f"Here is my plan.\n```{tag}\ndef policy(state):\n    return 0.0\n```\n"
        code = extract_code(text)
        assert code.startswith("def policy") and "```" not in code


def test_extract_bare_code():
    text = "def policy(state):\n    return 0\n"
    assert extract_code(text) == text.strip()


def test_extract_keeps_a_legacy_mode_line_as_a_comment():
    code = extract_code("```python\n# MODE: policy\ndef policy(state):\n    return 0\n```")
    assert code.startswith("# MODE: policy")
    assert "def policy" in code


def test_extract_unclosed_fence_takes_what_follows_the_opener():
    """A completion cut off by the token budget still yields its code."""
    code = extract_code("```python\nx = 1\nprint(x)\n")
    assert code == "x = 1\nprint(x)"


def test_extract_takes_first_fence():
    text = "```python\nfirst = 1\n```\n```python\nsecond = 2\n```\n"
    assert extract_code(text) == "first = 1"


def test_extract_empty_is_empty():
    assert extract_code("") == ""
    assert extract_code("   \n") == ""


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def test_score_policy_turn_alias_matches_score_sample():
    assert score_policy_turn is score_sample


def test_validate_main_allows_no_policy_def():
    is_valid, err = validate_main_code("import math\nprint(math.pi)\n")
    assert is_valid, err


def test_validate_rejects_dangerous_imports():
    for src in ("import os\n", "import gc\n", "import inspect\n", "from weakref import ref\n"):
        is_valid, err = validate_main_code(src)
        assert not is_valid, src
        is_valid, err = validate_code("def policy(state):\n    return 0\n" + src)
        assert not is_valid, src


def test_validate_rejects_forbidden_names_and_dunders():
    cases = {
        "getattr(env, 'hidden')\n": "Forbidden name: getattr",
        "print(globals())\n": "Forbidden name: globals",
        "open('x')\n": "Forbidden name: open",
        "exec('1')\n": "Forbidden name: exec",
        "print(__builtins__)\n": "Forbidden name: __builtins__",
        "type(env).__getattribute__\n": "Dunder attribute access is not allowed: __getattribute__",
        "env.step.__self__\n": "Dunder attribute access is not allowed: __self__",
    }
    for src, msg in cases.items():
        is_valid, err = validate_main_code(src)
        assert not is_valid and err == msg, (src, err)
    is_valid, err = validate_code("def policy(state):\n    return vars(state)\n")
    assert not is_valid and "vars" in err


def test_validate_allows_ordinary_probe_and_controller_code():
    probe = "s, _ = env.reset()\nwhile True:\n    s, r, done, _, _ = env.step(-1.0)\n    if done:\n        break\nprint(dir(env), s['sigma'], env.failure_reason)\n"
    assert validate_main_code(probe) == (True, "")
    ctrl = "mem = {'prev': None}\ndef policy(state):\n    mem['prev'] = state['sigma']\n    return -1.0\n"
    assert validate_code(ctrl) == (True, "")


# ---------------------------------------------------------------------------
# Probe-run execution without an env
# ---------------------------------------------------------------------------


def test_score_main_runs_simple_code_and_captures_stdout():
    result = score_main_turn("print('hello world')\nx = 1 + 2\nprint(x)\n", timeout=5)
    assert result["error"] is None
    assert "hello world" in result["stdout"] and "3" in result["stdout"]


def test_score_main_truncates_runaway_stdout():
    result = score_main_turn("for _ in range(10000): print('x' * 100)\n", timeout=5)
    assert len(result["stdout"]) <= 2000


def test_score_main_rejects_invalid_syntax():
    result = score_main_turn("def (\n", timeout=5)
    assert result["error"] is not None
    assert "rejected" in result["stdout"]


def test_score_main_does_not_expose_env_class():
    result = score_main_turn("CrystallizerEnv()\n", timeout=5)
    assert result["error"] is not None or "NameError" in result["stdout"]


# ---------------------------------------------------------------------------
# The probe-run env: same hidden parameters as the final run, behind a wrapper
# that offers reset, step and the outcome flags, nothing else
# ---------------------------------------------------------------------------

_PROBE_ENV_SRC = '''
class HiddenParams:
    def __init__(self, **kw):
        self.kw = kw


SECRET_CONSTANT = 0.777


class _ProbeTestEnv:
    def __init__(self, seed=0, hidden=None):
        self.seed = seed
        self.hidden = hidden
        self.config = {"k": 1.5e-6}
        self.public_note = "visible"
        self.done = False
        self.showered = False
        self.failure_reason = None

    def reset(self, seed=None):
        return {"seed": float(self.seed)}, {}

    def step(self, action):
        self.done = True
        self.showered = True
        self.failure_reason = "shower"
        return {"seed": float(self.seed)}, 0.0, True, False, {}

    def is_success(self):
        return False

    def is_unsafe(self):
        return True
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


def test_probe_env_offers_exactly_what_the_brief_states():
    out = _probe(
        "s, _ = env.reset()\n"
        "print('done before', env.done)\n"
        "print('done', env.step(0.0)[2], env.done, env.showered, env.failure_reason)\n"
        "print('success', env.is_success(), 'unsafe', env.is_unsafe())\n"
        "print('dir', dir(env))\n"
        "print(repr(env))\n"
    )
    assert out["error"] is None, out
    assert "done before False" in out["stdout"]
    assert "done True True True shower" in out["stdout"]
    assert "success False unsafe True" in out["stdout"]
    # dir() sorts what __dir__ returns
    assert "dir ['done', 'failure_reason', 'is_success', 'is_unsafe', 'reset', 'showered', 'step']" in out["stdout"]
    assert "ProbeEnv(reset, step, done, showered, failure_reason, is_success, is_unsafe)" in out["stdout"]


def test_probe_env_blocks_everything_else_including_public_attributes_and_config():
    out = _probe(
        "for name in ('public_note', 'config', 'seed', 'hidden', '_inner'):\n"
        "    try:\n"
        "        if name == 'public_note': env.public_note\n"
        "        elif name == 'config': env.config\n"
        "        elif name == 'seed': env.seed\n"
        "        elif name == 'hidden': env.hidden\n"
        "        else: env._inner\n"
        "        print('ALLOWED', name)\n"
        "    except AttributeError as e:\n"
        "        print('blocked', name, 'offers' in str(e))\n"
        "try:\n"
        "    env.done = False\n"
        "    print('ALLOWED write')\n"
        "except AttributeError:\n"
        "    print('blocked write')\n"
    )
    assert out["error"] is None, out
    assert "ALLOWED" not in out["stdout"]
    assert out["stdout"].count("blocked") == 6
    assert "blocked config True" in out["stdout"]
    assert "1.5e-06" not in out["stdout"] and "0.777" not in out["stdout"]


def test_probe_run_has_no_env_class_or_module_constants_in_scope():
    out = _probe(
        "for name in ('_ProbeTestEnv', 'HiddenParams', 'SECRET_CONSTANT'):\n"
        "    try:\n"
        "        if name == '_ProbeTestEnv': _ProbeTestEnv\n"
        "        elif name == 'HiddenParams': HiddenParams\n"
        "        else: SECRET_CONSTANT\n"
        "        print('ALLOWED', name)\n"
        "    except NameError:\n"
        "        print('no such name', name)\n"
    )
    assert out["error"] is None, out
    assert "ALLOWED" not in out["stdout"]
    assert out["stdout"].count("no such name") == 3


def test_probe_run_cannot_read_its_own_script():
    """`open` is a forbidden name, so the attempt is rejected before it runs;
    the hidden values never reach stdout."""
    out = _probe("print(open(__file__).read()[:20])\n")
    assert out["error"] is not None and "open" in out["error"]
    assert "0.123" not in out["stdout"]


def test_probe_run_cannot_reach_the_inner_env_through_dunders():
    for src in ("print(type(env).__getattribute__)\n", "print(env.step.__self__.hidden)\n",
                "print(env.__dict__)\n"):
        out = _probe(src)
        assert out["error"] is not None and "Dunder" in out["error"], src
        assert "0.123" not in out["stdout"]


def test_score_main_without_scenario_still_builds_the_env():
    out = score_main_turn("s, _ = env.reset()\nprint('seed', s['seed'])\n",
                          env_code=_PROBE_ENV_SRC, env_class="_ProbeTestEnv", timeout=10)
    assert out["error"] is None, out
    assert "seed 0.0" in out["stdout"]


def test_policy_run_has_no_env_class_in_scope():
    """The final run's code cannot construct or inspect the environment either."""
    task = {"task_id": "t", "env_class": "_ProbeTestEnv", "env_code": _PROBE_ENV_SRC,
            "scenarios": [_SCENARIO]}
    res = score_sample("def policy(state):\n    return _ProbeTestEnv\n", task, timeout=10, max_steps=5)
    assert res["errors"] and "'_ProbeTestEnv' is not defined" in res["errors"][0]
    res = score_sample("def policy(state):\n    return SECRET_CONSTANT\n", task, timeout=10, max_steps=5)
    assert res["errors"] and "'SECRET_CONSTANT' is not defined" in res["errors"][0]
