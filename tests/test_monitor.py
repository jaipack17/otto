"""Tests for run monitoring, stop/resume plumbing, report and watch.
Run in the training env: ./.venv-train/bin/pytest tests/test_monitor.py -q
No GPU, Docker or model needed."""

import argparse
import json
import os
import signal
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

from src.tasks.splits import HOLDOUT, split
from src.tasks.template import load_templates
from src.training import report, train, watch
from src.training.analysis import load_jsonl
from src.training.monitor import (MetricsLogger, MilestoneSaver, RunMonitor, StepProgress,
                                  StopController, lora_norm)

TEMPLATES = load_templates("data/templates/templates.jsonl")
TRAIN_T, HELD_T = split(TEMPLATES)


def trace(template="t1", seed=1, reward=1.0, stop="done", cmds=("ls", "cat a")):
    return {"template": template, "seed": seed, "reward": reward, "stop_reason": stop,
            "commands": list(cmds), "failures": [], "gate_failed": None, "diff": {},
            "model_tokens": 100, "think_cuts": 0,
            "segments": [{"role": "model", "text": "x"}, {"role": "env", "text": "[exit 0]"}]}


# ---------------------------------------------------------------- StopController
def make_state(step):
    return types.SimpleNamespace(global_step=step)


def make_control():
    return types.SimpleNamespace(should_save=False, should_training_stop=False)


def test_stop_flag_file_triggers_save_and_stop(tmp_path):
    sc = StopController(tmp_path, install_signals=False)
    c = make_control()
    sc.on_step_end(None, make_state(1), c)
    assert not c.should_save and not c.should_training_stop
    (tmp_path / "STOP").touch()
    sc.on_step_end(None, make_state(2), c)
    assert c.should_save and c.should_training_stop


def test_stale_stop_file_is_removed_at_startup(tmp_path):
    (tmp_path / "STOP").touch()
    sc = StopController(tmp_path, install_signals=False)
    c = make_control()
    sc.on_step_end(None, make_state(1), c)
    assert not (tmp_path / "STOP").exists() and not c.should_training_stop


def test_first_signal_requests_second_aborts(tmp_path):
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        sc = StopController(tmp_path)
        os.kill(os.getpid(), signal.SIGTERM)                  # first: graceful
        assert sc.requested
        c = make_control()
        sc.on_step_end(None, make_state(1), c)
        assert c.should_save and c.should_training_stop
        with pytest.raises(KeyboardInterrupt):                # second: immediate
            os.kill(os.getpid(), signal.SIGINT)
    finally:
        for s, h in old.items():
            signal.signal(s, h)


# ---------------------------------------------------------------- MetricsLogger
def test_metrics_logger_writes_rows_with_extras(tmp_path):
    import torch
    model = torch.nn.Module()
    model.lora_B = torch.nn.Parameter(torch.full((2, 2), 3.0))   # norm = sqrt(4*9) = 6
    model.lora_A = torch.nn.Parameter(torch.full((2, 2), 50.0))  # random-init half: ignored
    model.other = torch.nn.Parameter(torch.full((2, 2), 100.0))  # ignored
    assert lora_norm(model) == pytest.approx(6.0)
    ml = MetricsLogger(tmp_path)
    ml.on_log(None, make_state(3), None, logs={"loss": 0.5, "reward": 0.25}, model=model)
    ml.on_log(None, make_state(3), None, logs=None, model=model)   # ignored
    rows = load_jsonl(str(tmp_path / "metrics.jsonl"))
    assert len(rows) == 1
    assert rows[0]["step"] == 3 and rows[0]["loss"] == 0.5 and rows[0]["lora_norm"] == 6.0
    assert "step_secs" in rows[0]


# ---------------------------------------------------------------- MilestoneSaver
def test_milestones_copy_only_every_nth(tmp_path):
    for n in (9, 10):
        d = tmp_path / f"checkpoint-{n}"
        d.mkdir()
        (d / "optimizer.pt").write_text("opt")
    ms = MilestoneSaver(tmp_path, every=10)
    args = types.SimpleNamespace(output_dir=str(tmp_path))
    ms.on_save(args, make_state(9), None)
    ms.on_save(args, make_state(10), None)
    assert not (tmp_path / "milestones" / "checkpoint-9").exists()
    assert (tmp_path / "milestones" / "checkpoint-10" / "optimizer.pt").read_text() == "opt"


# ---------------------------------------------------------------- RunMonitor
def test_wrap_rollout_writes_traces_and_stats_with_trainer_step(tmp_path, capsys):
    mon = RunMonitor(tmp_path, G=2)

    def fake_rollout(prompts, trainer):
        mon.sink += [trace(reward=1.0), trace(reward=0.0)]
        return {"ok": True}

    wrapped = mon.wrap_rollout(fake_rollout)
    trainer = types.SimpleNamespace(state=types.SimpleNamespace(global_step=7))
    assert wrapped(["p"] * 2, trainer) == {"ok": True}        # return value untouched
    assert mon.sink == []                                      # emptied for the next step
    lines = load_jsonl(str(tmp_path / "traces" / "step-0008.jsonl"))   # step = global_step + 1
    assert [t["reward"] for t in lines] == [1.0, 0.0]
    stats = load_jsonl(str(tmp_path / "rollout_stats.jsonl"))
    assert stats[0]["step"] == 8 and stats[0]["mixed_groups"] == 1 and stats[0]["passed"] == 1
    assert "[rollout 8]" in capsys.readouterr().out
    assert not list((tmp_path / "traces").glob("*.tmp"))


def test_wrap_rollout_propagates_errors_and_leaves_no_partial_file(tmp_path):
    mon = RunMonitor(tmp_path, G=2)

    def boom(prompts, trainer):
        raise RuntimeError("docker died")

    trainer = types.SimpleNamespace(state=types.SimpleNamespace(global_step=0))
    with pytest.raises(RuntimeError):
        mon.wrap_rollout(boom)(["p"] * 2, trainer)
    assert list((tmp_path / "traces").iterdir()) == []


# ---------------------------------------------------------------- train.py helpers
def test_heldout_templates_are_excluded_from_prompts():
    ds = train.load_prompts("data/curated/train/ppo.jsonl", TRAIN_T)
    held_instr = {t.instruction for t in HELD_T}
    train_instr = {t.instruction for t in TRAIN_T}
    seen = {row["prompt"][-1]["content"] for row in ds}
    assert seen <= train_instr and not (seen & held_instr)
    assert seen == train_instr                                 # every train template is present
    assert len(HELD_T) == len(HOLDOUT) == 8
    assert all(row["prompt"][0]["role"] == "system" for row in ds)


def test_load_prompts_rejects_when_nothing_matches():
    with pytest.raises(ValueError):
        train.load_prompts("data/curated/train/ppo.jsonl", [types.SimpleNamespace(instruction="zzz")])


def test_latest_checkpoint_picks_highest_complete_one(tmp_path):
    def ck(n, files=("trainer_state.json", "optimizer.pt", "scheduler.pt")):
        d = tmp_path / f"checkpoint-{n}"
        d.mkdir()
        for f in files:
            (d / f).write_text("x")
    ck(2); ck(10); ck(12, files=("trainer_state.json",))       # 12 is incomplete (crash mid-save)
    (tmp_path / "checkpoint-final").mkdir()
    assert train.latest_checkpoint(tmp_path).name == "checkpoint-10"
    assert train.latest_checkpoint(tmp_path / "nowhere") is None


def args(**kw):
    base = dict(name="x", resume=None, model="m", templates="data/templates/templates.jsonl",
                train="t", steps=60, groups=2, G=8, think_budget=256, answer_cap=96,
                max_tokens=2048, lr=2e-5, lora_r=16, save_steps=1, keep=3,
                milestone_every=10, seed=0, only="", batch=8, wandb=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_prepare_run_dir_fresh_then_refuses_to_overwrite(tmp_path):
    a = args()
    run = tmp_path / "r"
    assert train.prepare_run_dir(a, run, train.run_info(a, TRAIN_T)) is None
    info = json.loads((run / "RUN_INFO.json").read_text())
    assert info["args"]["steps"] == 60 and info["system_prompt"] and info["train_templates"]
    (run / "metrics.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit):                            # silent overwrite is the bug to avoid
        train.prepare_run_dir(a, run, train.run_info(a, TRAIN_T))


def test_resume_requires_same_settings_and_records_itself(tmp_path):
    a = args()
    run = tmp_path / "r"
    train.prepare_run_dir(a, run, train.run_info(a, TRAIN_T))
    d = run / "checkpoint-3"
    d.mkdir()
    for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        (d / f).write_text("x")
    for changed in (dict(steps=61), dict(lr=1e-5), dict(G=4)):
        b = args(resume="auto", **changed)
        with pytest.raises(SystemExit):
            train.prepare_run_dir(b, run, train.run_info(b, TRAIN_T))
    b = args(resume="auto")
    assert train.prepare_run_dir(b, run, train.run_info(b, TRAIN_T)) == d
    assert len(json.loads((run / "RUN_INFO.json").read_text())["resumes"]) == 1


def test_resume_auto_with_no_checkpoint_starts_fresh(tmp_path):
    a = args(resume="auto")
    assert train.prepare_run_dir(a, tmp_path / "r", train.run_info(a, TRAIN_T)) is None


def test_install_injects_resume_checkpoint_and_callbacks():
    calls = {}

    class FakeTRL:
        args = types.SimpleNamespace(save_total_limit=None)
        def train(self, *a, **k):
            calls["train"] = k
            return "done"
        def add_callback(self, cb):
            calls.setdefault("cbs", []).append(cb)

    class FakeAT:
        trainer = None
        def setup_trainer(self):
            self.trainer = FakeTRL()

    at = FakeAT()
    train.install(at, lambda p, t: None, "DS", callbacks=["cb1", "cb2"],
                  resume_from=Path("runs/x/checkpoint-5"), save_total_limit=3)
    at.setup_data()
    assert at.train_dataset == "DS" and at.eval_dataset is None
    at.setup_trainer()
    t = at.trainer
    assert t.args.save_total_limit == 3 and calls["cbs"] == ["cb1", "cb2"]
    assert t.reward_func_names == ["episode_reward"] and len(t.reward_weights) == 1
    assert t.train() == "done"
    assert calls["train"] == {"resume_from_checkpoint": "runs/x/checkpoint-5"}


def test_install_without_resume_leaves_train_alone():
    class FakeTRL:
        args = types.SimpleNamespace(save_total_limit=None)
        def train(self, *a, **k):
            return k
        def add_callback(self, cb):
            pass

    class FakeAT:
        def setup_trainer(self):
            self.trainer = FakeTRL()

    at = FakeAT()
    train.install(at, lambda p, t: None, "DS")
    at.setup_trainer()
    assert at.trainer.train() == {} and at.trainer.args.save_total_limit is None


# ---------------------------------------------------------------- report / watch
def synthetic_run(tmp_path, steps=6):
    mon = RunMonitor(tmp_path, G=2)
    for s in range(1, steps + 1):
        batch = [trace("a", s, 1.0 if s > 3 else 0.0), trace("a", s, 0.0),
                 trace("b", s, 1.0), trace("b", s, 0.0)]
        mon.wrap_rollout(lambda p, t, batch=batch: mon.sink.extend(batch))(
            ["p"] * 4, types.SimpleNamespace(state=types.SimpleNamespace(global_step=s - 1)))
        ml = MetricsLogger(tmp_path)
        ml.on_log(None, make_state(s), None, logs={"reward": s / 10, "loss": 0.01 * s,
                  "grad_norm": 0.5, "entropy": 1.0 / s, "frac_reward_zero_std": 0.5,
                  "completions/mean_length": 900.0, "learning_rate": 2e-5})
    return tmp_path


def test_report_on_synthetic_run(tmp_path):
    run = synthetic_run(tmp_path)
    path = report.build(str(run))
    text = path.read_text()
    for needle in ("Training signal", "Rollouts", "Per template", "| a |", "| b |",
                   "Stop reasons", "Sample transcripts", "frac_reward_zero_std"):
        assert needle in text, needle
    assert (run / "plots.png").exists()


def test_summary_row_is_not_treated_as_a_step(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 1, "reward": 0.5}\n{"step": 1, "train_runtime": 9.0, "step_secs": 1.0}\n')
    assert "Steps logged: 1" in report.build(str(tmp_path)).read_text()
    assert watch.render(tmp_path).count("\n") < 6


def test_report_on_empty_and_partial_runs(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 1, "reward": 0.5}\n{"step": 2, "rew')   # killed mid-write
    text = report.build(str(tmp_path)).read_text()
    assert "Steps logged: 1" in text
    empty = tmp_path / "empty"
    empty.mkdir()
    assert report.build(str(empty)).exists()


def test_watch_render(tmp_path):
    run = synthetic_run(tmp_path)
    (run / "STOP").touch()
    (run / "checkpoint-3").mkdir()
    out = watch.render(run)
    assert "STOP requested" in out and "checkpoint-3" in out and "0.6" in out
    assert "no steps logged" in watch.render(tmp_path / "none")


# ---------------------------------------------------------------- wandb mirror
class FakeSink:
    def __init__(self):
        self.rows = []

    def log(self, row, step):
        self.rows.append((step, row))


def test_wandb_mirror_gets_numbers_only_never_transcripts(tmp_path):
    sink = FakeSink()
    mon = RunMonitor(tmp_path, G=2, wandb_sink=sink)
    mon.wrap_rollout(lambda p, t: mon.sink.extend([trace(reward=1.0), trace(reward=0.0)]))(
        ["p"] * 2, types.SimpleNamespace(state=types.SimpleNamespace(global_step=4)))
    MetricsLogger(tmp_path, sink).on_log(None, make_state(5), None, logs={"loss": 0.1})
    MetricsLogger(tmp_path, sink).on_log(None, make_state(5), None, logs={"train_runtime": 3.0})
    steps = [s for s, _ in sink.rows]
    assert steps == [5, 5]                                   # rollout of step 5, metrics of step 5
    flat = {k for _, r in sink.rows for k in r}
    assert "rollout/pass_rate" in flat and "loss" in flat and "stop/done" in flat
    assert not ({"segments", "commands", "diff", "failures"} & flat)
    assert all(isinstance(v, (int, float)) for _, r in sink.rows for v in r.values())
    assert not any("train_runtime" in r for _, r in sink.rows)   # summary row not mirrored


def test_wandb_sink_refuses_when_not_logged_in(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setattr(wandb, "login", lambda **k: False)
    from src.training.monitor import WandbSink
    with pytest.raises(SystemExit):
        WandbSink("p", "n", tmp_path, {})


def test_wandb_log_failure_does_not_raise():
    from src.training.monitor import WandbSink
    s = WandbSink.__new__(WandbSink)
    s.run = types.SimpleNamespace(log=lambda *a, **k: (_ for _ in ()).throw(OSError("net")))
    s.log({"loss": 1.0}, 1)


# ---------------------------------------------------------------- show_trace
def test_show_trace_list_and_render(tmp_path, capsys):
    from src.training import show_trace
    t = trace("stale-logs", 7, reward=0.0, stop="max_turns", cmds=("ls", "rm x"))
    t["failures"] = ["missing removed.txt"]
    t["segments"] = [
        {"role": "model", "tokens": 20, "text": "<think>\nlook first\n</think>\n\n<command>ls -la</command><|im_end|>"},
        {"role": "env", "tokens": 9, "text": "\n<|im_start|>user\nfoo.log\n[exit 0]<|im_end|>\n<|im_start|>assistant\n"},
        {"role": "model", "tokens": 5, "text": "<think>\nthinking forever"},
        {"role": "env", "tokens": 9, "text": "\nI should stop deliberating and act now.\n</think>\n\n"},
        {"role": "model", "tokens": 5, "text": "<command>rm x</command><|im_end|>"},
        {"role": "model", "tokens": 3, "text": "<think>\nok\n</think>\n\n<done></done><|im_end|>"},
    ]
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "step-0002.jsonl").write_text(json.dumps(t) + "\n")
    assert show_trace.main([str(tmp_path)]) == 0
    assert "stale-logs (7)" in capsys.readouterr().out
    assert show_trace.main([str(tmp_path), "--step", "2", "--index", "0"]) == 0
    out = capsys.readouterr().out
    assert "$ ls -la" in out and "foo.log" in out and "<|im_start|>" not in out
    assert "thinking budget hit" in out and "$ rm x" in out and "<done>" in out
    assert "missing removed.txt" in out
    assert show_trace.main([str(tmp_path), "--step", "9"]) == 1          # unknown step
    assert show_trace.main([str(tmp_path), "--index", "5"]) == 1         # out of range
    assert show_trace.main([str(tmp_path / "nowhere")]) == 1


# ---------------------------------------------------------------- within-step progress
def test_monitor_progress_prints_and_writes_one_row_per_event(tmp_path, capsys):
    mon = RunMonitor(tmp_path, G=2)
    mon.step = 4
    mon.progress({"event": "chunk_start", "chunk": 1, "chunks": 2, "episodes": 8})
    mon.progress({"event": "turn", "chunk": 1, "chunks": 2, "turn": 3, "generated": 5,
                  "running": 4, "finished": 4, "tokens": 900, "secs": 12.5})
    mon.progress({"event": "chunk_end", "chunk": 1, "chunks": 2, "secs": 100.0, "passed": 3,
                  "episodes": 8, "stops": {"done": 6, "max_turns": 2}})
    out = capsys.readouterr().out
    assert "[step 4] chunk 1/2: 8 episodes starting" in out
    assert "turn 3: 4 running, 4 finished, 900 tokens in 12.5s" in out
    assert "pass 3/8" in out
    rows = load_jsonl(str(tmp_path / "progress.jsonl"))
    assert [r["event"] for r in rows] == ["chunk_start", "turn", "chunk_end"]
    assert all(r["step"] == 4 and "t" in r for r in rows)


def test_step_progress_counts_micro_batches_and_numbers_steps_correctly(tmp_path, capsys):
    sp = StepProgress(tmp_path)
    args = types.SimpleNamespace(gradient_accumulation_steps=3)
    sp.on_step_begin(args, make_state(6), make_control())        # step 7 is in progress
    for _ in range(2):
        sp.on_substep_end(args, make_state(6), make_control())
    sp.on_step_end(args, make_state(7), make_control())          # trainer bumped global_step first
    out = capsys.readouterr().out
    assert "[step 7] micro-batch 1/3" in out and "[step 7] micro-batch 2/3" in out
    assert "[step 7] all micro-batches done" in out
    rows = load_jsonl(str(tmp_path / "progress.jsonl"))
    assert [(r["event"], r["step"], r["k"]) for r in rows] == \
        [("micro", 7, 1), ("micro", 7, 2), ("step_done", 7, 2)]
    sp.on_step_begin(args, make_state(7), make_control())        # counter resets each step
    sp.on_substep_end(args, make_state(7), make_control())
    assert load_jsonl(str(tmp_path / "progress.jsonl"))[-1]["k"] == 1


def test_watch_shows_the_latest_progress_event(tmp_path):
    mon = RunMonitor(tmp_path, G=2)
    mon.step = 2
    mon.progress({"event": "turn", "chunk": 2, "chunks": 2, "turn": 4, "generated": 3,
                  "running": 3, "finished": 5, "tokens": 700, "secs": 9.0})
    out = watch.render(tmp_path)
    assert "now: step 2 turn chunk 2/2 turn 4: 3 running, 5 finished" in out


def test_only_flag_is_validated_against_the_training_set():
    from src.training import train as T
    for bad in ("count-nested", "no-such-template"):                 # held-out and unknown
        with pytest.raises(SystemExit):
            T.main(["--name", "zz-never-created", "--only", bad, "--steps", "1"])
    assert not Path("runs/zz-never-created").exists() or not any(Path("runs/zz-never-created").iterdir())
