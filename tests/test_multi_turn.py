"""Tests for the branched-rollout pure-math + helper functions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent_loop import TOOL_PROBE, render_final_answer, render_tool_call  # noqa: E402
from src.multi_turn import (  # noqa: E402
    ACTION_FINAL,
    ACTION_INVALID,
    ACTION_PROBE,
    assemble_per_token_advantages,
    assemble_tool_mask,
    compute_branched_advantages,
    error_floor_reward,
    execute_turn,
    rollout_diagnostics,
    run_rollout,
)
from src.runner import ERROR_PENALTY_PER_SCENARIO  # noqa: E402

CONTROLLER = "def policy(state):\n    return (1.0, 0.0)\n"


def _final(code: str = CONTROLLER) -> str:
    return render_final_answer(code)


def _probe(code: str) -> str:
    return render_tool_call(TOOL_PROBE, code)


def test_compute_branched_advantages_uniform_grid_zeros_advantages():
    """When all rewards are equal, every advantage is 0."""
    rewards = [[5.0, 5.0], [5.0, 5.0]]
    v0, v1, a1, a2 = compute_branched_advantages(rewards)
    assert v0 == 5.0
    assert v1 == [5.0, 5.0]
    assert a1 == [[0.0, 0.0], [0.0, 0.0]]
    assert a2 == [[0.0, 0.0], [0.0, 0.0]]


def test_compute_branched_advantages_first_turn_winner():
    """First-turn variant 0 leads to high rewards, variant 1 to low ones.
    A_turn1 should be positive for variant 0 and negative for variant 1.
    A_turn2 should be ~0 within each row (continuations are similar)."""
    rewards = [[10.0, 11.0], [-9.0, -10.0]]
    v0, v1, a1, a2 = compute_branched_advantages(rewards)
    assert v0 == pytest.approx(0.5)
    assert v1[0] == pytest.approx(10.5)
    assert v1[1] == pytest.approx(-9.5)
    # turn-1 advantage discriminates first-turn variants
    assert a1[0][0] > 0
    assert a1[1][0] < 0
    # turn-2 advantage discriminates within a row
    assert a2[0][0] == pytest.approx(-0.5)  # 10 - 10.5
    assert a2[0][1] == pytest.approx(0.5)   # 11 - 10.5


def test_compute_branched_advantages_rejects_ragged_grid():
    with pytest.raises(ValueError):
        compute_branched_advantages([[1.0, 2.0], [3.0]])


def test_compute_branched_advantages_rejects_empty():
    with pytest.raises(ValueError):
        compute_branched_advantages([])


def test_assemble_per_token_advantages_layout():
    adv = assemble_per_token_advantages(3, 2, 4, a_turn1=1.5, a_turn2=-2.0)
    assert adv == [1.5, 1.5, 1.5, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0]


def test_assemble_tool_mask_layout():
    mask = assemble_tool_mask(3, 2, 4)
    assert mask == [1, 1, 1, 0, 0, 1, 1, 1, 1]


def test_rollout_diagnostics_first_turn_dominates():
    """Strong between-variant difference, low within-variant noise →
    high signal_ratio, A_turn1_std > 0."""
    rewards = [[10.0, 10.5, 9.5, 10.0], [-5.0, -5.5, -4.5, -5.0]]
    _, _, a_turn1, a_turn2 = compute_branched_advantages(rewards)
    diag = rollout_diagnostics(rewards, a_turn1, a_turn2)
    assert diag["v1_between_std"] > 5.0  # V_1 differs by ~15
    assert diag["v1_within_std_mean"] < 1.0  # rows are tight
    assert diag["v1_signal_ratio"] > 1.0  # first-turn dominates
    assert diag["a_turn1_std"] > 0  # turn-1 advantages discriminate


def test_rollout_diagnostics_within_variant_noise_dominates():
    """Same V_1 across variants but high within-variant spread →
    A_turn1_std == 0, A_turn2_std > 0."""
    rewards = [[5.0, 0.0, 10.0, -5.0], [5.0, 0.0, 10.0, -5.0]]
    _, _, a_turn1, a_turn2 = compute_branched_advantages(rewards)
    diag = rollout_diagnostics(rewards, a_turn1, a_turn2)
    assert diag["v1_between_std"] == pytest.approx(0.0)
    assert diag["v1_within_std_mean"] > 0
    assert diag["v1_signal_ratio"] == pytest.approx(0.0)
    assert diag["a_turn1_std"] == pytest.approx(0.0)  # NO first-turn signal
    assert diag["a_turn2_std"] > 0


def test_rollout_diagnostics_uniform_grid_is_silent():
    """All rewards equal → every diagnostic is 0; signal_ratio is inf
    (zero-over-zero, defined as inf to flag the degenerate case)."""
    rewards = [[5.0, 5.0], [5.0, 5.0]]
    _, _, a_turn1, a_turn2 = compute_branched_advantages(rewards)
    diag = rollout_diagnostics(rewards, a_turn1, a_turn2)
    assert diag["v1_between_std"] == 0.0
    assert diag["v1_within_std_mean"] == 0.0
    assert diag["a_turn1_std"] == 0.0
    assert diag["a_turn2_std"] == 0.0
    assert diag["v1_signal_ratio"] == float("inf")


def test_error_floor_reward_scales_with_scenarios():
    assert error_floor_reward(1) == ERROR_PENALTY_PER_SCENARIO
    assert error_floor_reward(8) == ERROR_PENALTY_PER_SCENARIO * 8


def _tiny_task_one_scenario():
    """Minimal task — one scenario — for execute_turn integration tests.

    Mirrors the runner's env contract: `Env(seed, hidden=HiddenParams(...))`,
    `reset() -> (state, info)`, `step() -> 5-tuple`."""
    env_code = '''
import math, random
from dataclasses import dataclass

@dataclass(frozen=True)
class HiddenParams:
    limit: float = 1.0

class TinyEnv:
    def __init__(self, seed=0, hidden=None, config=None):
        rng = random.Random(seed)
        self.gx = rng.uniform(-1, 1)
        self.steps = 0
    def reset(self, seed=None):
        return {"px": 0.0, "gx": self.gx}, {}
    def step(self, action):
        self.steps += 1
        done = self.steps >= 5
        return {"px": float(action[0]) * 0.1, "gx": self.gx}, -1.0, done, False, {}
'''
    return {
        "task_id": "tiny",
        "env_class": "TinyEnv",
        "env_code": env_code,
        "scenarios": [{"scenario_id": "s0", "seed": 0}],
    }


def _run(text: str, *, last: bool):
    return execute_turn(text, _tiny_task_one_scenario(), is_last_turn=last, timeout_policy=10, max_steps=10)


def test_a_reply_without_a_tool_call_is_the_final_answer_on_any_reply():
    for last in (True, False):
        tr = _run("Here goes:\n" + _final() + "\nThat is my controller.", last=last)
        assert tr.action == ACTION_FINAL and tr.mode == "policy" and tr.ended is True
        # Reward integrates -1.0 over 5 steps = -5.0 (no goal in tiny env)
        assert tr.reward == pytest.approx(-5.0, abs=0.01)
        assert tr.score["reward_total"] == tr.reward and tr.score["ran_successfully"]
        assert tr.code == CONTROLLER.strip() and tr.messages == ()
        assert "total_reward" in tr.text


def test_a_final_answer_as_bare_code_is_accepted():
    tr = _run(CONTROLLER, last=True)
    assert tr.action == ACTION_FINAL and tr.reward == pytest.approx(-5.0, abs=0.01)


def test_a_final_answer_without_a_controller_scores_the_floor():
    for text in (_final("print('no policy here')"), "I would need more data to answer.", ""):
        tr = _run(text, last=False)
        assert tr.action == ACTION_FINAL and tr.ended is True
        assert tr.reward == error_floor_reward(1)
        assert tr.score["ran_successfully"] is False


def test_a_probe_run_returns_what_it_printed_and_no_reward():
    tr = _run("Let me look.\n" + _probe("import math\nprint('answer:', math.sqrt(16))"), last=False)
    assert tr.action == ACTION_PROBE and tr.mode == "main" and tr.ended is False
    assert tr.reward is None and tr.score is None
    assert tr.messages == ({"role": "tool", "content": "answer: 4.0\n"},)
    assert "answer: 4.0" in tr.text


def test_a_probe_run_sees_the_probe_env():
    tr = _run(_probe("s, _ = env.reset()\nprint(sorted(s))"), last=False)
    assert tr.action == ACTION_PROBE and "['gx', 'px']" in tr.text


def test_a_probe_run_that_does_not_parse_gets_the_rejection_as_its_response():
    """The call is well formed, its code is not: the sandbox's rejection is the
    tool response, and credit comes from the continuations, not from a penalty."""
    tr = _run(_probe("def broken(:"), last=False)
    assert tr.action == ACTION_PROBE and tr.reward is None
    assert tr.messages[0]["role"] == "tool" and "[probe run rejected: Syntax error" in tr.text


def test_a_probe_run_that_crashes_gets_the_error_as_its_response():
    tr = _run(_probe("print('before')\nundefined_name"), last=False)
    assert tr.action == ACTION_PROBE and tr.reward is None
    assert "before" in tr.text and "[probe run error" in tr.text and "NameError" in tr.text


def test_what_a_reply_is_follows_from_the_tool_call_not_from_the_code():
    """A probe run whose code defines `policy` is still a probe run, and a
    reply that calls the tool is a probe run even if a controller sits in a
    code block beside the call: the probe runs and the task goes on."""
    tr = _run(_probe(CONTROLLER + "print('ran as a probe')"), last=False)
    assert tr.action == ACTION_PROBE and tr.ended is False and "ran as a probe" in tr.text
    tr = _run(_probe("print('probe output')") + "\nAnd my controller:\n" + _final(), last=False)
    assert tr.action == ACTION_PROBE and tr.ended is False and tr.reward is None
    assert "probe output" in tr.text


def test_several_probe_runs_in_one_reply_each_get_a_response():
    tr = _run(_probe("print('one')") + "\n" + _probe("print('two')"), last=False)
    assert tr.action == ACTION_PROBE
    assert [m["content"].strip() for m in tr.messages] == ["one", "two"]


def test_a_call_that_cannot_be_acted_on_gets_a_response_that_says_why():
    tr = _run('<tool_call>\n{"name": "probe_run", "arguments": {"code": "print("hi")"}}\n</tool_call>', last=False)
    assert tr.action == ACTION_INVALID and tr.mode == "invalid" and tr.ended is False
    assert tr.reward is None
    assert tr.messages[0]["role"] == "tool"
    assert tr.messages[0]["content"].startswith("[tool call not acted on: the call is not valid JSON")
    tr = _run('<tool_call>\n{"name": "final_run", "arguments": {"code": "x"}}\n</tool_call>', last=False)
    assert tr.action == ACTION_INVALID and "there is no tool named 'final_run'" in tr.text


def test_the_last_reply_that_calls_a_tool_ends_the_task_at_the_floor():
    for text, action in ((_probe("print('too late')"), ACTION_PROBE),
                         ('<tool_call>\n{"name": "nope"}\n</tool_call>', ACTION_INVALID)):
        tr = _run(text, last=True)
        assert tr.action == action and tr.ended is True
        assert tr.reward == error_floor_reward(1)
        assert tr.score["ran_successfully"] is False and "no final run" in tr.score["errors"][0]
        assert "too late" not in tr.text   # nothing is executed


# ---------------------------------------------------------------------------
# run_rollout: the loop evaluation runs
# ---------------------------------------------------------------------------


class _Scripted:
    """Stands in for the model: returns the scripted replies in turn and keeps
    what it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append(([dict(m) for m in messages], tools))
        return self.replies.pop(0)


def test_rollout_probe_then_final_run():
    gen = _Scripted([_probe("print('LIMIT=0.18')"), _final()])
    out = run_rollout(gen, "the brief", _tiny_task_one_scenario(), 2, timeout_policy=10, max_steps=10)
    assert [t.action for t in out["turns"]] == [ACTION_PROBE, ACTION_FINAL]
    assert out["final_run"] is True and out["code"] == CONTROLLER.strip()
    assert out["score"]["reward_total"] == pytest.approx(-5.0, abs=0.01)
    assert [m["role"] for m in out["transcript"]] == ["user", "assistant", "tool", "assistant"]
    assert out["transcript"][2]["content"].strip() == "LIMIT=0.18"
    # the model was offered the tool on both replies, and saw the tool response on the second
    assert [[t["function"]["name"] for t in tools] for _, tools in gen.seen] == [[TOOL_PROBE], [TOOL_PROBE]]
    assert gen.seen[1][0][-1] == {"role": "tool", "content": "LIMIT=0.18\n"}


def test_rollout_final_answer_straight_away_uses_one_reply():
    gen = _Scripted([_final(), "never asked for"])
    out = run_rollout(gen, "the brief", _tiny_task_one_scenario(), 2, timeout_policy=10, max_steps=10)
    assert [t.action for t in out["turns"]] == [ACTION_FINAL]
    assert len(gen.seen) == 1 and len(out["transcript"]) == 2


def test_rollout_without_a_final_run_scores_the_floor():
    gen = _Scripted([_probe("print(1)"), _probe("print(2)")])
    out = run_rollout(gen, "the brief", _tiny_task_one_scenario(), 2, timeout_policy=10, max_steps=10)
    assert out["final_run"] is False and out["code"] == ""
    assert out["score"]["reward_total"] == error_floor_reward(1)


def test_rollout_with_one_reply_offers_no_tool():
    gen = _Scripted([_final()])
    out = run_rollout(gen, "the brief", _tiny_task_one_scenario(), 1, timeout_policy=10, max_steps=10)
    assert out["final_run"] is True
    assert gen.seen[0][1] is None


def test_rollout_one_reply_that_calls_a_tool_scores_the_floor():
    gen = _Scripted([_probe("print(1)")])
    out = run_rollout(gen, "the brief", _tiny_task_one_scenario(), 1, timeout_policy=10, max_steps=10)
    assert out["final_run"] is False and out["score"]["reward_total"] == error_floor_reward(1)
