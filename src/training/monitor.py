"""Run monitoring: per-step traces, metrics log, graceful stop, milestone copies.

Everything a run leaves behind lives in runs/<name>/ and is written line by line and
flushed, so a run that dies still leaves valid, readable data:

  traces/step-NNNN.jsonl   every episode of that step, full transcript
  rollout_stats.jsonl      one aggregate row per step (pass rate, mixed groups, behaviours)
  metrics.jsonl            every Trainer log row (loss, reward, grad_norm, entropy, ...)
                           plus step seconds, memory, LoRA weight norm
  checkpoint-N/            adapter + optimizer + scheduler + RNG state (resumable)
  milestones/checkpoint-N/ every Nth checkpoint, kept permanently
"""

from __future__ import annotations

import json
import shutil
import signal
import time
from pathlib import Path

from transformers import TrainerCallback

from ..harness.rollout import release_memory
from .analysis import summarize_rollout


def append_row(path, row: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


class WandbSink:
    """Optional Weights & Biases mirror of the numbers (never transcripts or prompts).
    The run id is the run name, so a resume continues the same W&B run."""

    def __init__(self, project: str, name: str, run_dir, config: dict):
        import wandb
        try:
            ok = bool(wandb.login(prompt=False))
        except Exception:
            ok = False
        if not ok:
            raise SystemExit("wandb is not logged in. Run: .venv-train/bin/wandb login")
        self.run = wandb.init(project=project, name=name, id=name, resume="allow",
                              dir=str(run_dir), config=config)

    def log(self, row: dict, step: int) -> None:
        nums = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        try:
            self.run.log(nums, step=step)
        except Exception as e:                      # a network blip must never stop training
            print(f"[wandb] log failed: {e!r}", flush=True)

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception:
            pass


def rollout_scalars(stats: dict) -> dict:
    keep = ("pass_rate", "mixed_groups", "zero_std_groups", "think_cut_rate", "mean_commands",
            "mean_model_tokens", "gate_failed_rate", "collateral_rate", "rollout_secs")
    out = {f"rollout/{k}": stats[k] for k in keep if stats.get(k) is not None}
    out.update({f"behaviour/{k}": v for k, v in stats.get("behaviour", {}).items() if v is not None})
    out.update({f"stop/{k}": v for k, v in stats.get("stop_reasons", {}).items()})
    return out


class RunMonitor:
    """Owns the trace sink. Build the rollout with make_rollout(trace=monitor.sink) and
    then pass it through wrap_rollout; after each rollout the sink is written out."""

    def __init__(self, run_dir, G: int, wandb_sink=None):
        self.dir, self.G, self.sink, self.wandb = Path(run_dir), G, [], wandb_sink
        self.step = 0
        (self.dir / "traces").mkdir(parents=True, exist_ok=True)

    def progress(self, event: dict) -> None:
        """Within-step visibility: one console line and one progress.jsonl row for each
        chunk start, generation round and chunk end of the rollout."""
        append_row(self.dir / "progress.jsonl", {"t": round(time.time(), 1), "step": self.step, **event})
        kind, c = event["event"], f"chunk {event.get('chunk')}/{event.get('chunks')}"
        if kind == "chunk_start":
            line = f"{c}: {event['episodes']} episodes starting"
        elif kind == "turn":
            line = (f"{c} turn {event['turn']}: {event['running']} running, "
                    f"{event['finished']} finished, {event['tokens']} tokens in {event['secs']}s")
        else:
            line = (f"{c} done in {event['secs']}s: pass {event['passed']}/{event['episodes']}, "
                    f"stops {event['stops']}")
        print(f"[step {self.step}] {line}", flush=True)

    def wrap_rollout(self, rollout_fn):
        def wrapped(prompts, trainer):
            step = trainer.state.global_step + 1          # the step this rollout feeds
            self.step = step
            self.sink.clear()
            t0 = time.time()
            out = rollout_fn(prompts, trainer)
            release_memory()                              # before the update allocates its own
            secs = time.time() - t0
            traces = list(self.sink)
            self.sink.clear()
            tmp = self.dir / "traces" / f"step-{step:04d}.jsonl.tmp"
            with open(tmp, "w") as fh:
                fh.writelines(json.dumps(t) + "\n" for t in traces)
            tmp.replace(self.dir / "traces" / f"step-{step:04d}.jsonl")   # atomic
            stats = {"step": step, "rollout_secs": round(secs, 1),
                     **summarize_rollout(traces, self.G)}
            append_row(self.dir / "rollout_stats.jsonl", stats)
            if self.wandb:
                self.wandb.log(rollout_scalars(stats), step)
            print(f"[rollout {step}] {secs:.0f}s pass={stats['passed']}/{stats['n']} "
                  f"mixed_groups={stats['mixed_groups']}/{len(stats['groups'])} "
                  f"stops={stats['stop_reasons']}", flush=True)
            return out
        return wrapped


class StepProgress(TrainerCallback):
    """Update-phase visibility. A step is one rollout (reported by RunMonitor.progress)
    then `gradient_accumulation_steps` micro-batches of learning; this reports each
    micro-batch as it completes and the whole step at the end."""

    def __init__(self, run_dir):
        self.path, self.t0, self.k = Path(run_dir) / "progress.jsonl", time.time(), 0

    def on_step_begin(self, args, state, control, **kw):
        self.t0, self.k = time.time(), 0

    def on_substep_end(self, args, state, control, **kw):
        self.k += 1
        release_memory()
        self._say("micro", state, args, f"micro-batch {self.k}/{args.gradient_accumulation_steps}")
        return control

    def on_step_end(self, args, state, control, **kw):
        release_memory()
        self._say("step_done", state, args, "all micro-batches done, optimizer stepped")
        return control

    def _say(self, kind, state, args, text):
        secs = round(time.time() - self.t0, 1)
        append_row(self.path, {"t": round(time.time(), 1), "step": state.global_step + (kind == "micro"),
                               "event": kind, "k": self.k, "of": args.gradient_accumulation_steps,
                               "secs_since_step_start": secs})
        n = state.global_step + (kind == "micro")
        print(f"[step {n}] {text} ({secs}s since step start)", flush=True)


class StopController(TrainerCallback):
    """SIGINT / SIGTERM / `touch <run>/STOP` => finish the current step, save a full
    checkpoint, end training normally (so the final adapter and report still happen).
    A second signal raises KeyboardInterrupt to abort at once."""

    def __init__(self, run_dir, install_signals: bool = True):
        self.flag = Path(run_dir) / "STOP"
        self.flag.unlink(missing_ok=True)       # a stale STOP must not kill a fresh run
        self.requested = False
        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\n[stop] requested: finishing the current step, then saving "
              "(signal again to abort immediately)", flush=True)

    def on_step_end(self, args, state, control, **kw):
        if self.requested or self.flag.exists():
            control.should_save = True
            control.should_training_stop = True
        return control


def lora_norm(model) -> float:
    """L2 norm of the LoRA B matrices. B starts at exactly zero (A is random and large, so
    including it would drown the signal), so this is how far the policy has moved: our
    substitute for the KL metric that beta=0 switches off."""
    import torch
    total = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                total += float(p.float().pow(2).sum())
    return total ** 0.5


class MetricsLogger(TrainerCallback):
    def __init__(self, run_dir, wandb_sink=None):
        self.path, self.t_last = Path(run_dir) / "metrics.jsonl", time.time()
        self.wandb = wandb_sink

    def on_log(self, args, state, control, logs=None, model=None, **kw):
        if not logs:
            return
        now = time.time()
        row = {"step": state.global_step, "step_secs": round(now - self.t_last, 1), **logs}
        self.t_last = now
        try:
            import torch
            if torch.backends.mps.is_available():
                row["mps_gb"] = round(torch.mps.current_allocated_memory() / 2**30, 2)
                # what the process really holds; mps_gb alone hid a 15 GB cache
                row["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
        except Exception:
            pass
        if model is not None:
            row["lora_norm"] = round(lora_norm(model), 5)
        append_row(self.path, row)
        if self.wandb and "train_runtime" not in row:
            self.wandb.log(row, state.global_step)


class MilestoneSaver(TrainerCallback):
    """save_total_limit deletes old checkpoints; keep every Nth one for good."""

    def __init__(self, run_dir, every: int = 10):
        self.dest, self.every = Path(run_dir) / "milestones", every

    def on_save(self, args, state, control, **kw):
        if state.global_step % self.every == 0:
            src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if src.exists():
                shutil.copytree(src, self.dest / src.name, dirs_exist_ok=True)
        return control
