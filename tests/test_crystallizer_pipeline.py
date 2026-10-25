"""The hand-written probe run and final run through the real execution path.

On the held-out task with the highest hidden limit (where a blind policy
loses the most time): the probe source from the Stage 0 gate goes through
`execute_turn` (sandbox subprocess, probe-run env with the limit masked, probe
batch), its printed bracket is pasted into the final-run template, and the
final policy is scored with `score_sample`. This is the smallest end-to-end
check that the runner carries the crystallizer: the hidden limit reaches the
env, the probe sees the failure flag and nothing else, and the time metrics
come out right. The full Stage 0b measurement is scripts/stage0b_two_turn.py.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "seed_tasks"))

from src.multi_turn import execute_turn  # noqa: E402
from src.runner import extract_code_with_mode, score_main_turn, score_sample  # noqa: E402
from stage0_gate_crystal import FINAL_TEMPLATE, PROBE_SOURCE  # noqa: E402
from crystallizer_env import HiddenParams  # noqa: E402
from crystallizer_utils import cautious_policy, run_episode  # noqa: E402

EVAL = REPO_ROOT / "scripts" / "seed_tasks" / "crystallizer_eval.jsonl"


def _fence(src: str) -> str:
    return "```python\n" + src.strip("\n") + "\n```"


def _hardest_eval_task() -> dict:
    tasks = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    return max(tasks, key=lambda t: t["scenarios"][0]["hidden"]["sigma_max"])


def test_hand_written_probe_and_final_run_through_execute_turn():
    task = _hardest_eval_task()
    sc = task["scenarios"][0]
    smax = sc["hidden"]["sigma_max"]

    # Turn 1: probe run. The tool text carries the bracket on the limit.
    tr = execute_turn(_fence(PROBE_SOURCE), task, is_last_turn=False, max_steps=600)
    assert tr.mode == "main", tr.text
    assert "SIGMA_MAX_LOW=" in tr.text, tr.text
    lo = float(tr.text.split("SIGMA_MAX_LOW=")[1].split()[0])
    assert 0.0 < lo <= smax + 1e-9
    assert smax - lo < 0.01, f"probe bracket too loose: lo={lo} sigma_max={smax}"

    # Turn 2: final run, scored the way the trainer scores it.
    _, code = extract_code_with_mode(_fence(FINAL_TEMPLATE.format(sigma_hat=lo)))
    res = score_sample(code, task, max_steps=600)
    sid = sc["scenario_id"]
    assert res["errors"] == []
    assert res["success_per_scenario"][sid] is True
    assert res["fast_per_scenario"][sid] is True
    assert res["norm_time_per_scenario"][sid] >= 0.9
    assert res["steps_per_scenario"][sid] <= 1.25 * sc["oracle_steps"]

    # And it beats the blind cautious controller on the same batch.
    blind = run_episode(cautious_policy, seed=sc["seed"], hidden=HiddenParams(sigma_max=smax))
    assert res["reward_per_scenario"][sid] > blind["reward"]
    assert res["steps_per_scenario"][sid] < blind["steps"]


def test_probe_run_cannot_read_the_hidden_limit_of_the_real_env():
    task = _hardest_eval_task()
    sc = task["scenarios"][0]
    out = score_main_turn("print(env.hidden)\n", env_code=task["env_code"], scenario=sc,
                          env_class=task["env_class"])
    assert out["error"] is not None
    assert "not available in the probe run" in out["error"]
    assert str(sc["hidden"]["sigma_max"]) not in out["stdout"]


def test_probe_run_sees_a_different_batch_than_the_final_run():
    """Same limit, different batch: the probe env starts at the probe batch's
    initial temperature, not the scoring batch's."""
    from crystallizer_env import CrystallizerEnv

    task = _hardest_eval_task()
    sc = task["scenarios"][0]
    out = score_main_turn("s, _ = env.reset()\nprint('T0', round(s['T'], 6))\n",
                          env_code=task["env_code"], scenario=sc, env_class=task["env_class"])
    assert out["error"] is None, out
    hidden = HiddenParams(sigma_max=sc["hidden"]["sigma_max"])

    def first_batch_after_reset(seed: int) -> float:
        # Construct, then reset() once: what the runner does for both runs.
        env = CrystallizerEnv(seed=seed, hidden=hidden)
        state, _ = env.reset()
        return round(state["T"], 6)

    t0_probe = first_batch_after_reset(sc["probe_seed"])
    t0_scoring = first_batch_after_reset(sc["seed"])
    assert f"T0 {t0_probe}" in out["stdout"]
    assert t0_probe != t0_scoring
