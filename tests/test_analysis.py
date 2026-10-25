"""Rollout-trace analysis tests. Pure, torch-free: ./.venv/bin/pytest tests/test_analysis.py -q"""

import json

import pytest

from src.training import analysis as A


def trace(template="t", reward=0.0, stop="done", commands=("ls",), exits=(0,), cuts=0, tokens=300,
          failures=(), gate=False, seed=1):
    env = "".join(f"\n<|im_start|>user\nout\n[exit {c}]\n(hint)<|im_end|>\n" for c in exits)
    return {"template": template, "seed": seed, "reward": reward, "stop_reason": stop,
            "commands": list(commands), "failures": list(failures), "gate_failed": gate,
            "model_tokens": tokens, "think_cuts": cuts,
            "segments": [{"role": "model", "tokens": 5, "text": "x"},
                         {"role": "env", "tokens": 9, "text": env}]}


@pytest.mark.parametrize("cmd,expect", [
    ("ls -la", True), ("cat removed.txt", True), ("find . -name '*.log'", True),
    ("grep -rl old .", True), ("/bin/ls", True),
    ("find . -name '*.tmp' -delete", False),          # writes
    ("cat a > b", False), ("ls | tee out.txt", False),
    ("find . -exec rm {} \;", False), ("sed -i s/a/b/ f", False),
    ("rm -rf x", False), ("chmod +x f", False), ("echo hi", False),
])
def test_is_inspection(cmd, expect):
    assert A.is_inspection(cmd) is expect


def test_head_skips_env_assignments_and_paths():
    assert A.head("FOO=1 BAR=2 grep x f") == "grep" and A.head("/usr/bin/find .") == "find"


def test_jaccard():
    assert A.jaccard("ls -la", "ls -la") == 1.0 and A.jaccard("ls a", "rm b") == 0.0


def test_exit_codes_are_read_in_order():
    assert A.exit_codes(trace(exits=(0, 2, 127))) == [0, 2, 127]


def test_behaviour_features():
    t = trace(stop="done", commands=("ls -A", "rm *.tmp", "ls -A"), exits=(0, 0, 0))
    b = A.behaviour(t)
    assert b["inspect_first"] is True and b["verify_before_done"] is True
    assert b["premature_done"] is False and b["repetition"] == pytest.approx(1 / 3)  # 3rd repeats 1st
    assert b["error_rate"] == 0 and b["python_use"] is False
    assert b["cmd_diversity"] == pytest.approx(2 / 3)


def test_premature_done_and_not_verified():
    b = A.behaviour(trace(stop="done", commands=("rm *.tmp",), exits=(0,)))
    assert b["premature_done"] is True and b["verify_before_done"] is False and b["inspect_first"] is False


def test_done_features_are_undefined_when_the_episode_did_not_finish():
    b = A.behaviour(trace(stop="max_turns", commands=("ls", "ls")))
    assert b["verify_before_done"] is None and b["premature_done"] is None


def test_a_failing_last_command_before_done_is_premature():
    assert A.behaviour(trace(stop="done", commands=("ls", "bad"), exits=(0, 2)))["premature_done"] is True


def test_python_use_and_errors():
    b = A.behaviour(trace(commands=("python3 x.py",), exits=(127,)))
    assert b["python_use"] is True and b["error_rate"] == 1.0


def test_empty_episode_has_no_defined_command_features():
    b = A.behaviour(trace(stop="parse_failures", commands=(), exits=()))
    assert b["inspect_first"] is None and b["repetition"] is None and b["error_rate"] is None


def test_group_verdicts():
    assert A.group_verdict(0, 8) == "ALL-FAIL" and A.group_verdict(8, 8) == "ALL-PASS"
    assert A.group_verdict(3, 8) == "MIXED"


def test_summary_groups_by_consecutive_blocks():
    # group 1: 2 of 4 pass (mixed); group 2: 0 of 4 (zero-std)
    ts = [trace("a", 1.0, seed=7), trace("a", 1.0, seed=7), trace("a", 0.0, seed=7), trace("a", 0.0, seed=7)] \
       + [trace("b", 0.0, seed=9, stop="max_turns") for _ in range(4)]
    s = A.summarize_rollout(ts, G=4)
    assert s["n"] == 8 and s["passed"] == 2 and s["pass_rate"] == 0.25
    assert [g["verdict"] for g in s["groups"]] == ["MIXED", "ALL-FAIL"]
    assert s["mixed_groups"] == 1 and s["zero_std_groups"] == 1
    assert s["by_template"] == {"a": [2, 4], "b": [0, 4]}
    assert s["stop_reasons"] == {"done": 4, "max_turns": 4}


def test_summary_rates():
    ts = [trace(cuts=2, tokens=100, gate=True, failures=["unexpected changes: ['x']"]),
          trace(cuts=0, tokens=300), trace(cuts=1, tokens=500), trace(cuts=0, tokens=100)]
    s = A.summarize_rollout(ts, G=4)
    assert s["think_cut_rate"] == 0.5 and s["mean_think_cuts_per_episode"] == 0.75
    assert s["mean_model_tokens"] == 250 and s["max_model_tokens"] == 500
    assert s["gate_failed_rate"] == 0.25 and s["collateral_rate"] == 0.25


def test_summary_of_nothing_does_not_crash():
    s = A.summarize_rollout([], G=8)
    assert s["n"] == 0 and s["groups"] == [] and s["pass_rate"] == 0.0


def test_load_jsonl_tolerates_a_half_written_last_line(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2, "b"')      # the run is mid-write
    assert A.load_jsonl(str(p)) == [{"a": 1}]
    assert A.load_jsonl(str(tmp_path / "missing.jsonl")) == []
