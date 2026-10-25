"""Turn a run directory into report.md and plots. Safe on a live or killed run: it reads
whatever has been flushed so far.

  ./.venv-train/bin/python -m src.training.report runs/<name>

Matplotlib is optional; without it you still get the tables.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Optional

from .analysis import load_jsonl

# (title, metric keys to draw in one panel)
PANELS = [
    ("Reward (mean per step)", ["reward"]),
    ("Fraction of groups with zero reward variance (no gradient)", ["frac_reward_zero_std"]),
    ("Loss", ["loss"]),
    ("Gradient norm (clip at 1.0)", ["grad_norm"]),
    ("Policy entropy", ["entropy"]),
    ("Completion length (tokens)", ["completions/mean_length", "completions/max_length"]),
    ("Clipped-at-max-length ratio", ["completions/clipped_ratio"]),
    ("LoRA weight norm (distance moved, stands in for KL)", ["lora_norm"]),
    ("Learning rate", ["learning_rate"]),
    ("Seconds per step", ["step_secs"]),
]
ROLLOUT_PANELS = [
    ("Pass rate over rollouts", ["pass_rate"]),
    ("Mixed groups per step", ["mixed_groups"]),
    ("Share of episodes with a thinking cut", ["think_cut_rate"]),
    ("Mean commands per episode", ["mean_commands"]),
]


def series(rows: list[dict], key: str):
    pts = [(r["step"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    return [p[0] for p in pts], [p[1] for p in pts]


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def window_mean(rows, key, lo, hi):
    v = [r[key] for r in rows if lo <= r["step"] <= hi and isinstance(r.get(key), (int, float))]
    return statistics.fmean(v) if v else None


def plot(run: Path, metrics: list[dict], stats: list[dict]) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    panels = [(t, k, metrics) for t, k in PANELS] + [(t, k, stats) for t, k in ROLLOUT_PANELS]
    panels = [(t, ks, rows) for t, ks, rows in panels if any(series(rows, k)[0] for k in ks)]
    if not panels:
        return None
    cols = 3
    fig, axes = plt.subplots((len(panels) + cols - 1) // cols, cols,
                             figsize=(15, 3.2 * ((len(panels) + cols - 1) // cols)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, keys, rows) in zip(axes.flat, panels):
        ax.axis("on")
        for k in keys:
            xs, ys = series(rows, k)
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, label=k.split("/")[-1])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
        if len(keys) > 1:
            ax.legend(fontsize=7)
    fig.tight_layout()
    out = run / "plots.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def template_table(stats: list[dict]) -> list[str]:
    """Pass rate per template, first third of the run vs last third."""
    if not stats:
        return []
    last = stats[-1]["step"]
    cut1, cut2 = last / 3, 2 * last / 3
    early: dict[str, list[int]] = {}
    late: dict[str, list[int]] = {}
    for r in stats:
        bucket = early if r["step"] <= cut1 else late if r["step"] > cut2 else None
        if bucket is None:
            continue
        for t, (p, n) in r.get("by_template", {}).items():
            row = bucket.setdefault(t, [0, 0])
            row[0] += p
            row[1] += n
    lines = ["| template | early pass | late pass | change |", "|---|---|---|---|"]
    for t in sorted(set(early) | set(late)):
        e, l = early.get(t), late.get(t)
        er = e[0] / e[1] if e and e[1] else None
        lr = l[0] / l[1] if l and l[1] else None
        d = None if er is None or lr is None else lr - er
        lines.append(f"| {t} | {fmt(er, 2)} ({e[1] if e else 0}) | {fmt(lr, 2)} ({l[1] if l else 0}) "
                     f"| {'-' if d is None else f'{d:+.2f}'} |")
    return lines


def sample_traces(run: Path, k: int = 2) -> list[str]:
    """Best and worst transcripts of the last step that has traces."""
    files = sorted((run / "traces").glob("step-*.jsonl"))
    if not files:
        return []
    traces = load_jsonl(str(files[-1]))
    if not traces:
        return []
    out = [f"## Sample transcripts ({files[-1].name})", ""]
    ordered = sorted(traces, key=lambda t: t["reward"])
    for label, group in (("lowest reward", ordered[:k]), ("highest reward", ordered[-k:])):
        for t in group:
            out.append(f"**{label}** - {t['template']} seed {t['seed']} reward {t['reward']} "
                       f"stop={t['stop_reason']} commands={len(t['commands'])}")
            out += ["", "```"] + [f"$ {c}" for c in t["commands"]] + ["```", ""]
            if t.get("failures"):
                out += [f"failed checks: {t['failures'][:3]}", ""]
    return out


def build(run_dir: str) -> Path:
    run = Path(run_dir)
    # TRL appends one end-of-training summary row (train_runtime, ...); it is not a step
    metrics = [r for r in load_jsonl(str(run / "metrics.jsonl")) if "train_runtime" not in r]
    stats = load_jsonl(str(run / "rollout_stats.jsonl"))
    out = [f"# Run report: {run.name}", ""]
    info = run / "RUN_INFO.json"
    if info.exists():
        out += ["Config and provenance: `RUN_INFO.json`", ""]
    last_m = metrics[-1]["step"] if metrics else 0
    out += [f"Steps logged: {last_m}; rollout steps recorded: {len(stats)}", ""]

    if metrics:
        n = max(1, len(metrics) // 3)
        first, final = metrics[:n], metrics[-n:]
        out += ["## Training signal (mean of first third vs last third of steps)", "",
                "| metric | first third | last third |", "|---|---|---|"]
        for key in ("reward", "frac_reward_zero_std", "loss", "grad_norm", "entropy",
                    "completions/mean_length", "completions/clipped_ratio", "lora_norm", "step_secs"):
            a = window_mean(first, key, 0, 10**9)
            b = window_mean(final, key, 0, 10**9)
            if a is not None or b is not None:
                out.append(f"| {key} | {fmt(a, 4)} | {fmt(b, 4)} |")
        out.append("")
        out += ["Reading guide: reward should trend up; `frac_reward_zero_std` near 1 means "
                "almost no learning signal; entropy falling steadily toward 0 or `grad_norm` "
                "repeatedly at the clip means instability; `lora_norm` is how far the policy has "
                "moved (KL is off).", ""]

    if stats:
        n = max(1, len(stats) // 3)
        out += ["## Rollouts", "", "| window | pass rate | mixed groups | think-cut rate | "
                "commands/episode | gate-failed |", "|---|---|---|---|---|---|"]
        for name, rows in (("first third", stats[:n]), ("last third", stats[-n:])):
            def m(k, rows=rows):
                v = [r[k] for r in rows if r.get(k) is not None]
                return statistics.fmean(v) if v else None
            mixed = sum(r["mixed_groups"] for r in rows) / max(1, sum(len(r["groups"]) for r in rows))
            out.append(f"| {name} | {fmt(m('pass_rate'))} | {fmt(mixed)} | {fmt(m('think_cut_rate'))} "
                       f"| {fmt(m('mean_commands'), 2)} | {fmt(m('gate_failed_rate'))} |")
        out += ["", "### Behaviour (rule-based, observed only, never rewarded)", "",
                "| feature | first third | last third |", "|---|---|---|"]
        for key in sorted(stats[-1].get("behaviour", {})):
            def mb(rows, key=key):
                v = [r["behaviour"][key] for r in rows if r.get("behaviour", {}).get(key) is not None]
                return statistics.fmean(v) if v else None
            out.append(f"| {key} | {fmt(mb(stats[:n]), 3)} | {fmt(mb(stats[-n:]), 3)} |")
        out += ["", "### Per template (early vs late)", ""] + template_table(stats) + [""]
        stops: dict[str, int] = {}
        for r in stats:
            for k, v in r.get("stop_reasons", {}).items():
                stops[k] = stops.get(k, 0) + v
        out += [f"Stop reasons over the whole run: {stops}", ""]

    png = plot(run, metrics, stats)
    if png:
        out += ["## Plots", "", f"![plots]({png.name})", ""]
    out += sample_traces(run)
    report = run / "report.md"
    report.write_text("\n".join(out) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    print(build(ap.parse_args(argv).run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
