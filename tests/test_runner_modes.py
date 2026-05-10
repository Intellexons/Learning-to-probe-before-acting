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
    """Per the 2026-04-25 decision, main() runs without the env class
    in its namespace. Trying to reference one should raise NameError."""
    code = "PuckTiltedPlaneEnv()\n"
    result = score_main_turn(code, timeout=5)
    # The subprocess will have raised NameError; we get an error back.
    assert result["error"] is not None or "NameError" in result["stdout"]
