"""Multi-turn rollouts for TRL's GRPOTrainer -- the piece AlignTune does not expose.

TRL calls `rollout_func(prompts, trainer)` instead of generating itself and expects
token ids back. Each rollout here is a whole episode: the policy and the sandbox take
turns, and we stitch the model's tokens and the environment's tokens into ONE
sequence, with `env_mask` saying which is which (1 = the model produced it, 0 = it
did not). Training must never learn to predict `find` output, or text the model
did not choose, so every non-model token is masked out of the loss.

Facts this depends on, each verified rather than assumed:
  * TRL repeats every prompt G times CONSECUTIVELY, so a group is a block of G.
  * rollout_func receives only prompts, no dataset columns, so the task is identified
    by its instruction text and a fresh seed is drawn per group.
  * Qwen3: <think> / </think> / <|im_start|> / <|im_end|> are single tokens, and the
    chat template puts "\\n<|im_start|>user\\n{obs}<|im_end|>\\n<|im_start|>assistant\\n"
    between an assistant turn and the next.
  * TRL accepts logprobs=None (its own HF generation path does) and falls back to the
    current model's logprobs, which is standard on-policy GRPO.

Requires torch + transformers: the training environment, not the torch-free .venv.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .episode import EpisodeRun

THINK_CLOSE, IM_END = 151668, 151645     # checked against the tokenizer by check_tokenizer


def check_tokenizer(tok) -> None:
    """Fail loudly if these ids are wrong. They are hard-coded; a different tokenizer
    would otherwise mis-mask every sequence while training appeared to run."""
    assert tok("</think>", add_special_tokens=False).input_ids == [THINK_CLOSE], "THINK_CLOSE id"
    assert tok("<|im_end|>", add_special_tokens=False).input_ids == [IM_END], "IM_END id"
    assert tok.eos_token_id == IM_END, "eos is not <|im_end|>"


# Cutting thinking mid-sentence and forcing only `</think>` leaves the model still
# deliberating, just outside the think block: measured on the real model, 3 of 3
# samples spent their whole answer allowance and never reached a <command>. What
# worked (a llama.cpp server with a thinking budget, 6/6 commands) was a short
# sentence BEFORE the closing tag, so we force that whole sequence.
FORCED_TEXT = "\nI should stop deliberating and act now.\n"


def budget_forcing_ids(tok) -> list[int]:
    """The tokens injected when the thinking budget runs out: a sentence, `</think>`,
    and the blank line the model normally writes after it. None of them is the
    model's choice, so all are masked out of the loss."""
    return (tok(FORCED_TEXT, add_special_tokens=False).input_ids
            + [THINK_CLOSE] + tok("\n\n", add_special_tokens=False).input_ids)


class ThinkBudget(LogitsProcessor):
    """Force `force_ids` once a sequence has used its thinking budget.

    A sequence whose model closed its own thinking is left alone. Records where
    forcing began so those tokens can be masked out of the loss."""

    def __init__(self, prompt_len: int, budget: int, batch: int,
                 force_ids: Optional[list[int]] = None) -> None:
        self.prompt_len, self.budget = prompt_len, budget
        self.force_ids = list(force_ids) if force_ids else [THINK_CLOSE]
        assert THINK_CLOSE in self.force_ids, "forcing must end the thinking"
        self.closed = [False] * batch
        self.start: list[Optional[int]] = [None] * batch

    def __call__(self, input_ids, scores):
        generated = input_ids.shape[1] - self.prompt_len
        last = input_ids[:, -1].tolist()                  # one device sync per step
        for i in range(scores.shape[0]):
            if self.start[i] is None:
                if last[i] == THINK_CLOSE:
                    self.closed[i] = True                 # the model closed it itself
                if not self.closed[i] and generated == self.budget:
                    self.start[i] = generated
            if self.start[i] is not None:
                k = generated - self.start[i]
                if k < len(self.force_ids):
                    scores[i, :] = float("-inf")
                    scores[i, self.force_ids[k]] = 0.0
        return scores


@torch.no_grad()
def generate_batch(model, tok, contexts, device, think_budget, answer_cap, force_ids=None):
    """Left-padded batched sampling at temperature 1.0 with no top-k or top-p, so the
    sampling distribution is the policy's own. Returns [(token_ids, mask)] per row."""
    L = max(map(len, contexts))
    pad = tok.pad_token_id if tok.pad_token_id is not None else IM_END
    ids = torch.full((len(contexts), L), pad, dtype=torch.long)
    attn = torch.zeros((len(contexts), L), dtype=torch.long)
    for i, c in enumerate(contexts):
        ids[i, L - len(c):] = torch.tensor(c)
        attn[i, L - len(c):] = 1
    proc = ThinkBudget(L, think_budget, len(contexts), force_ids)

    was_training = model.training
    model.eval()                                          # no dropout while sampling
    try:
        out = model.generate(
            input_ids=ids.to(device), attention_mask=attn.to(device),
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            max_new_tokens=think_budget + len(proc.force_ids) + answer_cap,
            eos_token_id=IM_END, pad_token_id=pad, use_cache=True,
            logits_processor=LogitsProcessorList([proc]),
        )
    finally:
        if was_training:
            model.train()

    results = []
    for i, row in enumerate(out[:, L:].tolist()):
        if IM_END in row:
            row = row[: row.index(IM_END) + 1]            # keep the model's own <|im_end|>
        else:
            while row and row[-1] == pad:                 # strip padding after truncation
                row.pop()
        mask = [1] * len(row)
        if proc.start[i] is not None:
            for j in range(proc.start[i], min(proc.start[i] + len(proc.force_ids), len(row))):
                mask[j] = 0
        results.append((row, mask))
    return results


def env_chunk(tok, observation: str, turn_was_closed: bool) -> list[int]:
    """The tokens the chat template places between an assistant turn and the next.
    A turn cut off by max_new_tokens was never closed, so close it here (mask 0)."""
    text = f"\n<|im_start|>user\n{observation}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, add_special_tokens=False).input_ids
    return ids if turn_was_closed else [IM_END] + ids


def segments(tok, comp: list, mask: list) -> list[dict]:
    """Split a completion into alternating model / environment runs, as text.

    This is exactly the structure the loss sees: `model` spans are trained on,
    `env` spans (observations, the forced thinking cut) are not. Reading it is the
    fastest way to check the stitching and to see what the policy actually did."""
    out, start = [], 0
    for i in range(1, len(comp) + 1):
        if i == len(comp) or mask[i] != mask[start]:
            out.append({"role": "model" if mask[start] else "env",
                        "tokens": i - start, "text": tok.decode(comp[start:i])})
            start = i
    return out


def release_memory() -> None:
    """Hand cached GPU memory back to the OS. PyTorch's Apple-GPU allocator keeps every
    block it has ever used, and generation/training allocate differently sized tensors
    all the time, so the cache grew to 17 GB while only 2.4 GB was live, which pushed a
    16 GB Mac into swap and slowed everything ~3x. Cheap next to a 30 s generation round."""
    import gc
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def group_seed(step: int, group: int, lo: int = 1_000, hi: int = 899_999) -> int:
    """A fresh fixture per group, reproducible, and never in the eval range (900000+)."""
    return lo + (step * 7919 + group * 104729) % (hi - lo)


@dataclass
class _Seq:
    prompt_ids: list
    ctx: list                                             # everything the model has seen
    comp: list = field(default_factory=list)              # what TRL trains on
    mask: list = field(default_factory=list)              # 1 = model token, 0 = environment


def make_rollout(templates, *, think_budget: int = 256, answer_cap: int = 96,
                 max_tokens: int = 2048, trace: Optional[list] = None, batch: int = 8,
                 progress=None):
    """Build the rollout_func. `max_tokens` caps the whole completion (every turn plus
    every observation): the Mac's training throughput collapses past ~4k tokens.

    Pass a list as `trace` to receive one dict per episode (template, seed, reward,
    why it scored that, the filesystem diff, and the transcript). Training leaves it
    None, so nothing extra is built or kept.

    `batch` caps how many episodes generate at once. A step with 2 groups of 8 is 16
    episodes; generating all 16 together took the Mac to 14 GB resident and into swap
    (0% CPU, no progress), while 8 at a time is the size that has been measured to fit.
    Groups are never split, so the chunk is a whole number of groups, and the training
    update still sees every episode of the step.

    `progress`, if given, is called with a small dict at the start and end of every chunk
    and after every generation round (how many episodes are still running, how many have
    finished). It is for logging only: an exception inside it is swallowed."""
    by_instruction = {t.instruction: t for t in templates}
    assert len(by_instruction) == len(templates), "two templates share an instruction"
    assert max_tokens > think_budget + answer_cap + 32, "max_tokens cannot fit even one turn"

    def emit(event: dict) -> None:
        if progress is not None:
            try:
                progress(event)
            except Exception:                             # logging must never break a rollout
                pass

    def one_batch(prompts, trainer, group_offset, position=(1, 1)):
        tok, device = trainer.processing_class, trainer.accelerator.device
        check_tokenizer(tok)
        forced = budget_forcing_ids(tok)
        model = trainer.accelerator.unwrap_model(trainer.model)
        G, step = trainer.num_generations, trainer.state.global_step
        assert len(prompts) % G == 0, "prompt count is not a multiple of G"

        runs: list[EpisodeRun] = []
        seqs: list[_Seq] = []
        fixtures: list[str] = []
        meta: list[tuple] = []                            # (template id, seed) per episode
        try:
            for g in range(len(prompts) // G):
                block = prompts[g * G:(g + 1) * G]
                assert all(b == block[0] for b in block), "prompts not repeated consecutively"
                template = by_instruction[block[0][-1]["content"]]   # KeyError: not our prompt
                fixture = tempfile.mkdtemp(prefix="rollout-")
                fixtures.append(fixture)
                # ONE instance per group: GRPO's baseline is the group mean, which only
                # means something if all G rollouts faced the same task.
                spec = template.instantiate(group_seed(step, group_offset + g), fixture)
                for messages in block:
                    meta.append((template.id, group_seed(step, group_offset + g)))
                    run = EpisodeRun(spec, fixture, messages=messages)
                    runs.append(run)
                    run.start()
                    text = tok.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
                    ids = tok(text, add_special_tokens=False).input_ids
                    seqs.append(_Seq(ids, list(ids)))

            emit({"event": "chunk_start", "chunk": position[0], "chunks": position[1],
                  "episodes": len(runs)})
            t_chunk = t_round = time.time()
            for turn in range(1, max(r.task.max_turns for r in runs) + 1):
                live = [i for i, r in enumerate(runs) if not r.finished]
                for i in list(live):                      # no room left to generate a turn
                    if len(seqs[i].comp) + think_budget + len(forced) + answer_cap > max_tokens:
                        runs[i].abort("too_long")
                        live.remove(i)
                if not live:
                    break
                gens = generate_batch(model, tok, [seqs[i].ctx for i in live], device,
                                      think_budget, answer_cap, forced)
                for i, (ids, mask) in zip(live, gens):
                    s, run = seqs[i], runs[i]
                    s.comp += ids; s.mask += mask; s.ctx += ids
                    obs = run.step(tok.decode(ids, skip_special_tokens=True))
                    if obs is None:
                        continue                          # finished: no trailing env tokens
                    chunk = env_chunk(tok, obs, bool(ids) and ids[-1] == IM_END)
                    if len(s.comp) + len(chunk) > max_tokens:
                        run.abort("too_long")
                        continue
                    s.comp += chunk; s.mask += [0] * len(chunk); s.ctx += chunk
                now = time.time()
                emit({"event": "turn", "chunk": position[0], "chunks": position[1], "turn": turn,
                      "generated": len(live), "running": sum(1 for r in runs if not r.finished),
                      "finished": sum(1 for r in runs if r.finished),
                      "tokens": sum(len(ids) for ids, _ in gens), "secs": round(now - t_round, 1)})
                t_round = now
                release_memory()

            episodes = [r.finish() for r in runs]
            emit({"event": "chunk_end", "chunk": position[0], "chunks": position[1],
                  "secs": round(time.time() - t_chunk, 1),
                  "passed": sum(1 for e in episodes if e.reward > 0), "episodes": len(episodes),
                  "stops": dict(Counter(e.stop_reason for e in episodes))})
            if trace is not None:
                for (tid, seed), ep, sq in zip(meta, episodes, seqs):
                    segs = segments(tok, sq.comp, sq.mask)
                    trace.append({
                        "template": tid, "seed": seed, "reward": ep.reward,
                        "stop_reason": ep.stop_reason, "commands": ep.commands,
                        "failures": ep.result.failures if ep.result else [],
                        "gate_failed": ep.result.gate_failed if ep.result else None,
                        "diff": ep.diff, "model_tokens": sum(sq.mask), "segments": segs,
                        # how many turns hit the thinking budget and were cut off
                        "think_cuts": sum(1 for g in segs if g["role"] == "env"
                                          and FORCED_TEXT.strip() in g["text"])})
        finally:
            for r in runs:
                r.close()                                 # idempotent
            for f in fixtures:
                shutil.rmtree(f, ignore_errors=True)

        return {
            "prompt_ids": [s.prompt_ids for s in seqs],
            "completion_ids": [s.comp for s in seqs],
            "logprobs": None,
            "env_mask": [s.mask for s in seqs],
            "episode_reward": [e.reward for e in episodes],
            "stop_reason": [e.stop_reason for e in episodes],
            "n_commands": [len(e.commands) for e in episodes],      # diagnostics, also forwarded
        }

    def rollout(prompts, trainer):
        G = trainer.num_generations
        assert len(prompts) % G == 0, "prompt count is not a multiple of G"
        size = max(G, batch // G * G)                     # whole groups only
        merged: dict = {}
        n_chunks = -(-len(prompts) // size)
        for start in range(0, len(prompts), size):
            out = one_batch(prompts[start:start + size], trainer, start // G,
                            position=(start // size + 1, n_chunks))
            for key, value in out.items():
                if value is None:
                    merged[key] = None
                else:
                    merged.setdefault(key, []).extend(value)
        return merged

    return rollout


def episode_reward(prompts, completions, episode_reward, **kwargs):
    """TRL's reward contract. The verifier already scored the episode in the rollout."""
    return [float(r) for r in episode_reward]
