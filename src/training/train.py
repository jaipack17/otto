"""Train the terminal agent with multi-turn GRPO.

AlignTune builds the trainer: config, model loading, LoRA, logging, saving. TRL runs
GRPO underneath. We replace exactly two things, through seams AlignTune already has
(`train()` calls setup_model, setup_data, setup_rewards, setup_trainer in turn):

  * where the prompts come from -- our own dataset, after it has been through
    CuratorKit (src/tasks/curate.py), minus the held-out templates;
  * how a rollout happens -- a whole multi-turn episode (src/harness/rollout.py),
    which AlignTune does not expose because it never passes `rollout_func` to TRL.

Around that, src/training/monitor.py adds traces, metrics, a clean stop and milestone
copies. Everything a run produces lives in runs/<name>/.

Run (training env):
  scripts/train.sh smoke --steps 2 --groups 1 --G 2
  scripts/train.sh main  --steps 60 --groups 2 --G 8

Stop early, any time (finishes the current step, saves a full checkpoint, writes the
final adapter and the report):    touch runs/<name>/STOP     or Ctrl-C once
Resume after a stop or a crash (same --steps as the original run):
  scripts/train.sh main --steps 60 --groups 2 --G 8 --resume auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset

from ..harness.episode import SYSTEM_PROMPT
from ..harness.rollout import episode_reward, make_rollout
from ..tasks.splits import split
from ..tasks.template import load_templates
from .monitor import (MetricsLogger, MilestoneSaver, RunMonitor, StepProgress, StopController,
                      WandbSink)

# Settings that must not change between the first run and a resume. Changing --steps
# would also bend the linear learning-rate decay, which is computed from the total.
FIXED_ON_RESUME = ("model", "templates", "train", "steps", "groups", "G", "think_budget",
                   "answer_cap", "max_tokens", "lr", "lora_r", "seed", "only")


def load_prompts(path: str, templates) -> Dataset:
    """Conversations for the templates in `templates` only. The curated file holds just
    the instruction text, so rows are matched to templates by that text; a row whose
    instruction belongs to no template (a held-out one) is dropped, not trained on."""
    allowed = {t.instruction for t in templates}
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r["prompt"] in allowed:
                    rows.append({"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                                            {"role": "user", "content": r["prompt"]}]})
    if not rows:
        raise ValueError(f"no rows of {path} match the given templates")
    return Dataset.from_list(rows)


def latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Highest-numbered checkpoint that holds everything a resume needs."""
    best = None
    for p in run_dir.glob("checkpoint-*"):
        suffix = p.name.split("-", 1)[1]
        if suffix.isdigit() and all((p / f).exists() for f in
                                    ("trainer_state.json", "optimizer.pt", "scheduler.pt")):
            if best is None or int(suffix) > int(best.name.split("-", 1)[1]):
                best = p
    return best


def install(trainer, rollout_fn, train_ds: Dataset, *, callbacks=(), resume_from=None,
            save_total_limit: Optional[int] = None) -> None:
    """Attach our dataset, rollout, callbacks and resume point to an AlignTune TRL GRPO
    trainer."""

    def setup_data() -> None:
        trainer.train_dataset, trainer.eval_dataset = train_ds, None

    original_setup_trainer = trainer.setup_trainer

    def setup_trainer() -> None:
        original_setup_trainer()                           # builds the real TRL GRPOTrainer
        t = trainer.trainer
        t.rollout_func = rollout_fn                        # TRL stores it as a plain attribute
        # AlignTune built one placeholder reward; swap in ours. All four containers must
        # keep the same length or TRL's strict zip over them fails.
        t.reward_funcs = [episode_reward]
        t.reward_func_names = ["episode_reward"]
        t.reward_processing_classes = [None]
        t.reward_weights = torch.ones(1, dtype=torch.float32)
        if save_total_limit is not None:
            t.args.save_total_limit = save_total_limit     # read at save time
        for cb in callbacks:
            t.add_callback(cb)
        if resume_from is not None:
            # AlignTune calls self.trainer.train() with no arguments. Wrap it so the
            # Trainer restores adapter, optimizer, scheduler position, RNG and step count.
            original_train = t.train
            t.train = lambda *a, **k: original_train(*a, resume_from_checkpoint=str(resume_from), **k)

    trainer.setup_data = setup_data
    trainer.setup_trainer = setup_trainer


def run_info(a: argparse.Namespace, train_templates) -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    def version(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return None

    return {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(a),
        "git_sha": sh("git", "rev-parse", "HEAD"),
        "git_dirty": bool(sh("git", "status", "--porcelain")),
        "system_prompt": SYSTEM_PROMPT,
        "templates_sha256": hashlib.sha256(Path(a.templates).read_bytes()).hexdigest(),
        "train_templates": sorted(t.id for t in train_templates),
        "versions": {p: version(p) for p in ("torch", "trl", "transformers", "peft", "aligntune")},
        "python": sys.version.split()[0],
        "resumes": [],
    }


def prepare_run_dir(a: argparse.Namespace, run_dir: Path, info: dict):
    """Create or validate runs/<name>. Returns the checkpoint to resume from, or None."""
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "RUN_INFO.json"
    has_data = (run_dir / "metrics.jsonl").exists() or latest_checkpoint(run_dir) is not None
    if a.resume is None:
        if has_data:
            raise SystemExit(f"{run_dir} already has a run in it. Add `--resume auto` to continue "
                             f"it, or pick a new --name.")
        info_path.write_text(json.dumps(info, indent=2))
        return None
    ckpt = latest_checkpoint(run_dir) if a.resume == "auto" else Path(a.resume)
    if ckpt is None:
        print(f"[resume] no checkpoint in {run_dir}, starting fresh", flush=True)
        if not info_path.exists():
            info_path.write_text(json.dumps(info, indent=2))
        return None
    if not ckpt.exists():
        raise SystemExit(f"checkpoint {ckpt} does not exist")
    if info_path.exists():
        old = json.loads(info_path.read_text())
        changed = {k: (old["args"].get(k), getattr(a, k)) for k in FIXED_ON_RESUME
                   if old["args"].get(k) != getattr(a, k)}
        if changed:
            raise SystemExit(f"cannot resume with different settings (old, new): {changed}")
        old["resumes"].append({"at": info["started"], "from": str(ckpt),
                               "git_sha": info["git_sha"], "git_dirty": info["git_dirty"]})
        info_path.write_text(json.dumps(old, indent=2))
    else:
        info_path.write_text(json.dumps(info, indent=2))
    print(f"[resume] continuing from {ckpt}", flush=True)
    return ckpt


def build(a: argparse.Namespace, run_dir: Path, resume_from: Optional[Path], train_templates):
    from aligntune.core.backend_factory import create_rl_trainer

    sink = (WandbSink(a.wandb, a.name, run_dir, {k: v for k, v in vars(a).items() if k != "wandb"})
            if a.wandb else None)
    monitor = RunMonitor(run_dir, a.G, sink)
    callbacks = [StopController(run_dir), StepProgress(run_dir), MetricsLogger(run_dir, sink),
                 MilestoneSaver(run_dir, a.milestone_every)]
    trainer = create_rl_trainer(
        model_name=a.model,
        dataset_name="local/otto",                          # never loaded: setup_data is replaced
        algorithm="grpo", backend="trl", output_dir=str(run_dir),
        # sequences per optimizer step = batch_size * gradient_accumulation_steps, and TRL
        # needs it to be a whole number of groups. batch_size 1 keeps memory flat.
        batch_size=1, gradient_accumulation_steps=a.groups * a.G, num_generations=a.G,
        learning_rate=a.lr, num_epochs=1, max_steps=a.steps,
        max_completion_length=a.max_tokens, max_seq_length=a.max_tokens + 256,
        temperature=1.0, top_p=1.0,
        loss_type="dapo", beta=0.0,                         # TMax: token-level loss, no KL
        use_peft=True, lora_r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        rewards=[{"type": "length", "weight": 1.0, "params": {}}],     # placeholder, replaced
        bf16=True, use_gradient_checkpointing=True,
        system_prompt=SYSTEM_PROMPT, logging_steps=1, save_steps=a.save_steps, seed=a.seed,
    )
    rollout = make_rollout(train_templates, think_budget=a.think_budget,
                           answer_cap=a.answer_cap, max_tokens=a.max_tokens, trace=monitor.sink,
                           batch=a.batch, progress=monitor.progress)
    install(trainer, monitor.wrap_rollout(rollout), load_prompts(a.train, train_templates),
            callbacks=callbacks, resume_from=resume_from, save_total_limit=a.keep)
    trainer.wandb_sink = sink
    return trainer


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="run directory is runs/<name>")
    ap.add_argument("--resume", default=None, metavar="auto|PATH",
                    help="continue from the newest checkpoint of this run, or from PATH")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--templates", default="data/templates/templates.jsonl")
    # The CURATED file (CuratorKit: schema gate, cleaning, secrets gate, provenance), not the
    # raw dataset. tests/test_curate.py fails if the two ever drift apart.
    ap.add_argument("--train", default="data/curated/train/ppo.jsonl")
    ap.add_argument("--steps", type=int, default=40, help="optimizer steps")
    ap.add_argument("--groups", type=int, default=2, help="prompts (groups) per step")
    ap.add_argument("--G", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--think-budget", type=int, default=256)
    ap.add_argument("--answer-cap", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="whole-episode completion cap; the Mac collapses past ~4k")
    ap.add_argument("--batch", type=int, default=8,
                    help="episodes generating at once (whole groups); 16 swapped the 16 GB Mac")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=1, help="checkpoint every N steps")
    ap.add_argument("--keep", type=int, default=3, help="newest checkpoints to keep")
    ap.add_argument("--milestone-every", type=int, default=10,
                    help="also keep every Nth checkpoint permanently")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", metavar="ID,ID,...",
                    help="train on just these (non-held-out) templates, e.g. an easy curriculum")
    ap.add_argument("--wandb", default=None, metavar="PROJECT",
                    help="also send the metrics (not transcripts) to this W&B project")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    train_templates, held_out = split(load_templates(a.templates))
    if a.only:
        wanted = [x for x in a.only.split(",") if x]
        known = {t.id for t in train_templates}
        bad = [w for w in wanted if w not in known]
        if bad:
            raise SystemExit(f"--only names templates that are unknown or held out: {bad}")
        train_templates = [t for t in train_templates if t.id in set(wanted)]
    run_dir = Path("runs") / a.name
    resume_from = prepare_run_dir(a, run_dir, run_info(a, train_templates))
    (run_dir / "PID").write_text(str(os.getpid()))
    print(f"[run] {run_dir}  train templates {len(train_templates)}, held out {len(held_out)}",
          flush=True)
    trainer = build(a, run_dir, resume_from, train_templates)
    try:
        print(trainer.train())
    finally:
        (run_dir / "PID").unlink(missing_ok=True)
        if getattr(trainer, "wandb_sink", None):
            trainer.wandb_sink.finish()
        try:
            from .report import build as build_report
            print(f"[report] {build_report(str(run_dir))}", flush=True)
        except Exception as e:                              # never mask the real outcome
            print(f"[report] failed: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
