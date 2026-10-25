"""Rollout tests. Run in the training env: ./.venv-train/bin/pytest tests/test_rollout.py -q

Needs torch + transformers, so it is skipped in the torch-free .venv. Nothing here
needs a GPU: generation is replaced by a fake model whose `generate` drives our real
logits processor step by step, so the budget forcing and the mask indices are tested
for real rather than assumed. The end-to-end tests also need Docker and the cached
Qwen3 tokenizer. OTTO_LIVE=1 adds one test against the real model on the Mac's GPU.
"""

import os
import subprocess
import tempfile
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.harness import rollout as R
from src.harness.episode import DONE_HINT, SYSTEM_PROMPT
from src.tasks.template import TaskTemplate, load_templates
from tests.test_harness import needs_docker

try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
except Exception:                                       # not cached, or offline
    TOK = None
needs_tok = pytest.mark.skipif(TOK is None, reason="Qwen3 tokenizer not cached")

VOCAB = 151_936
STUB_TOK = types.SimpleNamespace(pad_token_id=151_643)
TEMPLATES = load_templates("data/templates/templates.jsonl")
# Pinned here, not read from the live set: these tests script `rm *.tmp *.bak`, so they
# must not break when the training templates change.
TPL = TaskTemplate.from_dict({"id": "delete-two-kinds", "instruction": "Delete every .tmp file and every .bak file in /work. Leave all other files alone.", "files": [{"name": "{word}.tmp", "content": "scratch\n", "count": [1, 3], "tag": "junk"}, {"name": "{word}.bak", "content": "old\n", "count": [1, 3], "tag": "junk"}, {"name": "{word}.dat", "content": "live {int}\n", "count": [1, 3], "tag": "keep"}], "expect_deleted": {"tag": "junk"}, "expect_created": [], "expect_modified": [], "checks": [], "reference": ["rm *.tmp *.bak"], "max_turns": 3})


class FakeModel:
    """Stands in for model.generate. Each row follows a script; our REAL logits
    processors are applied to the scores at every step, so forcing happens for real."""

    def __init__(self, scripts):
        self.scripts, self.training, self.calls = scripts, True, []

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, logits_processor, max_new_tokens,
                 eos_token_id, pad_token_id, **kw):
        self.calls.append({"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone(),
                           "training_during": self.training, "kw": kw})
        B, cur, done = input_ids.shape[0], input_ids, [False] * input_ids.shape[0]
        for t in range(max_new_tokens):
            scores = torch.zeros((B, VOCAB))
            for i in range(B):
                scr = self.scripts[i]
                scores[i, scr[t] if t < len(scr) else eos_token_id] = 10.0
            for p in logits_processor:
                scores = p(cur, scores)
            nxt = scores.argmax(-1)
            for i in range(B):
                if done[i]:
                    nxt[i] = pad_token_id
                elif nxt[i].item() == eos_token_id:
                    done[i] = True
            cur = torch.cat([cur, nxt[:, None]], dim=1)
            if all(done):
                break
        return cur


# ====================================================== thinking budget (no GPU)

def test_thinkbudget_forces_close_exactly_at_the_budget():
    proc = R.ThinkBudget(prompt_len=2, budget=2, batch=2)
    # one token generated so far: nothing to force yet
    proc(torch.tensor([[1, 1, 7], [1, 1, R.THINK_CLOSE]]), torch.zeros(2, VOCAB))
    assert proc.start == [None, None]
    # two generated: row 0 never closed -> forced; row 1 closed on its own -> left alone
    out = proc(torch.tensor([[1, 1, 7, 7], [1, 1, R.THINK_CLOSE, 9]]), torch.zeros(2, VOCAB))
    assert out[0].argmax().item() == R.THINK_CLOSE and proc.start[0] == 2
    assert torch.all(out[1] == 0) and proc.start[1] is None


def test_forced_close_is_masked_out_of_the_loss():
    model = FakeModel([[100, 100, 100, 100, 0, 200, 201, R.IM_END],
                       [100, R.THINK_CLOSE, 300, R.IM_END]])
    out = R.generate_batch(model, STUB_TOK, [[1, 2, 3], [5, 6]], "cpu", 4, 8)
    assert out[0] == ([100] * 4 + [R.THINK_CLOSE, 200, 201, R.IM_END], [1, 1, 1, 1, 0, 1, 1, 1])
    # the model closed its own thinking, so nothing is masked
    assert out[1] == ([100, R.THINK_CLOSE, 300, R.IM_END], [1, 1, 1, 1])


def test_truncated_turn_keeps_its_tokens_and_drops_only_padding():
    model = FakeModel([[100] * 20, [100, R.IM_END]])
    out = R.generate_batch(model, STUB_TOK, [[1], [1]], "cpu", 4, 3)       # max_new = 8
    toks, mask = out[0]
    assert len(toks) == 8 and toks[4] == R.THINK_CLOSE and mask[4] == 0 and sum(mask) == 7
    assert out[1] == ([100, R.IM_END], [1, 1])                              # padding stripped


def test_contexts_are_left_padded_and_sampling_is_on_policy():
    model = FakeModel([[R.IM_END], [R.IM_END]])
    R.generate_batch(model, STUB_TOK, [[1, 2, 3], [5]], "cpu", 4, 4)
    call = model.calls[0]
    assert call["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert call["input_ids"][1].tolist() == [151_643, 151_643, 5]
    # temperature 1.0 and no truncation: the sampling distribution IS the policy
    assert call["kw"]["temperature"] == 1.0 and call["kw"]["top_k"] == 0 and call["kw"]["top_p"] == 1.0


def test_model_is_in_eval_mode_while_sampling_and_restored_after():
    model = FakeModel([[R.IM_END]])
    R.generate_batch(model, STUB_TOK, [[1]], "cpu", 4, 4)
    assert model.calls[0]["training_during"] is False and model.training is True


def test_a_multi_token_forced_sequence_is_emitted_in_order():
    proc = R.ThinkBudget(prompt_len=1, budget=2, batch=1, force_ids=[10, 11, R.THINK_CLOSE, 12])
    cur, emitted = [1], []
    for _ in range(8):
        scores = proc(torch.tensor([cur]), torch.zeros(1, VOCAB))
        nxt = 99 if torch.all(scores[0] == 0) else scores[0].argmax().item()
        emitted.append(nxt); cur.append(nxt)
    # two free tokens, then the whole forced sequence, then the model is free again
    assert emitted == [99, 99, 10, 11, R.THINK_CLOSE, 12, 99, 99]


def test_the_whole_forced_span_is_masked_not_just_the_close_tag():
    force = [7, 8, R.THINK_CLOSE, 9]
    model = FakeModel([[100, 100, 100, 0, 0, 0, 0, 200, R.IM_END]])
    toks, mask = R.generate_batch(model, STUB_TOK, [[1, 2]], "cpu", 3, 8, force)[0]
    assert toks == [100, 100, 100, 7, 8, R.THINK_CLOSE, 9, 200, R.IM_END]
    assert mask == [1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_a_model_that_closes_early_is_never_forced_even_with_a_long_sequence():
    model = FakeModel([[100, R.THINK_CLOSE, 300, R.IM_END]])
    toks, mask = R.generate_batch(model, STUB_TOK, [[1]], "cpu", 4, 8, [7, 8, R.THINK_CLOSE, 9])[0]
    assert toks == [100, R.THINK_CLOSE, 300, R.IM_END] and mask == [1, 1, 1, 1]


@needs_tok
def test_budget_forcing_ids_end_the_thinking_and_decode_to_the_sentence():
    ids = R.budget_forcing_ids(TOK)
    assert R.THINK_CLOSE in ids
    assert TOK.decode(ids).startswith(R.FORCED_TEXT) and TOK.decode(ids).endswith("</think>\n\n")


# ============================================================ token stitching

@needs_tok
def test_token_ids_match_the_tokenizer():
    R.check_tokenizer(TOK)


@needs_tok
def test_env_chunk_reproduces_the_chat_template_exactly():
    # The strongest check on the stitching: our hand-built sequence must be the string
    # the chat template itself produces for the same conversation.
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "T"}]
    prompt = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = TOK.apply_chat_template(
        msgs + [{"role": "assistant", "content": "A"}, {"role": "user", "content": "OBS"}],
        tokenize=False, add_generation_prompt=True)
    assert prompt + "A" + "<|im_end|>" + TOK.decode(R.env_chunk(TOK, "OBS", True)) == full


@needs_tok
def test_an_unclosed_turn_is_closed_by_the_environment_chunk():
    assert R.env_chunk(TOK, "x", False)[0] == R.IM_END
    assert R.env_chunk(TOK, "x", True)[0] != R.IM_END


def test_group_seeds_are_reproducible_distinct_and_never_in_the_eval_range():
    seeds = {(s, g): R.group_seed(s, g) for s in range(50) for g in range(16)}
    assert all(1_000 <= v < 899_999 for v in seeds.values())          # eval is 900000+
    assert R.group_seed(3, 5) == R.group_seed(3, 5)
    assert len({R.group_seed(7, g) for g in range(64)}) == 64         # groups differ in a step


# ============================================== the rollout, end to end (Docker)

def scripted(per_turn):
    """A generate_batch that replays text per turn (by row position among live rows)."""
    state = {"turn": 0}

    def gen(model, tok, contexts, device, think_budget, answer_cap, force_ids=None):
        texts = per_turn[min(state["turn"], len(per_turn) - 1)]
        state["turn"] += 1
        out = []
        for pos in range(len(contexts)):
            ids = TOK(texts[min(pos, len(texts) - 1)], add_special_tokens=False).input_ids + [R.IM_END]
            out.append((ids, [1] * len(ids)))
        return out
    return gen


def trainer(G, step=0):
    return types.SimpleNamespace(
        processing_class=TOK, model=object(), num_generations=G,
        accelerator=types.SimpleNamespace(device="cpu", unwrap_model=lambda m: m),
        state=types.SimpleNamespace(global_step=step))


def prompt(tpl=TPL):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": tpl.instruction}]


def runs_of(mask):
    out = []
    for m in mask:
        if out and out[-1][0] == m:
            out[-1][1] += 1
        else:
            out.append([m, 1])
    return out


@needs_docker
@needs_tok
def test_rewards_stitching_and_masks(monkeypatch):
    monkeypatch.setattr(R, "generate_batch", scripted(
        [["<command>rm *.tmp *.bak</command>", "<command>rm -rf *</command>"], ["<done></done>"]]))
    out = R.make_rollout([TPL])([prompt()] * 2, trainer(2))

    # row 0 solves the task, row 1 wipes the directory: per-row isolation + the gate
    assert out["episode_reward"] == [1.0, 0.0]
    assert out["stop_reason"] == ["done", "done"]
    assert out["logprobs"] is None

    for ids, mask, prompt_ids in zip(out["completion_ids"], out["env_mask"], out["prompt_ids"]):
        assert len(ids) == len(mask) and set(mask) <= {0, 1}
        assert ids[-1] == R.IM_END and mask[-1] == 1             # ends on the model's own token
        # exactly: model turn, environment chunk, model turn
        assert [m for m, _ in runs_of(mask)] == [1, 0, 1]
        model1, env, model2 = [TOK.decode(ids[a:b]) for a, b in _spans(mask)]
        assert env.startswith("\n<|im_start|>user\n") and env.endswith("<|im_end|>\n<|im_start|>assistant\n")
        assert "[exit 0]" in env and DONE_HINT.strip() in env
        assert model2 == "<done></done><|im_end|>"
        assert TOK.decode(prompt_ids).endswith("<|im_start|>assistant\n")
    assert TOK.decode(out["completion_ids"][0][: _spans(out["env_mask"][0])[0][1]]) == \
        "<command>rm *.tmp *.bak</command><|im_end|>"


def _spans(mask):
    spans, a = [], 0
    for _, n in runs_of(mask):
        spans.append((a, a + n)); a += n
    return spans


@needs_docker
@needs_tok
def test_one_fixture_instance_per_group_and_fresh_seeds_across_groups(monkeypatch):
    seeds, original = [], TaskTemplate.instantiate

    def spy(self, seed, dest):
        seeds.append(seed)
        return original(self, seed, dest)

    monkeypatch.setattr(TaskTemplate, "instantiate", spy)
    monkeypatch.setattr(R, "generate_batch", scripted([["<done></done>"]]))
    R.make_rollout([TPL])([prompt()] * 4, trainer(2, step=5))          # 2 groups of 2
    assert len(seeds) == 2, "a group must share ONE instance, not one per rollout"
    assert seeds[0] != seeds[1]
    assert seeds == [R.group_seed(5, 0), R.group_seed(5, 1)]


@needs_docker
@needs_tok
def test_the_token_cap_ends_long_episodes_instead_of_overflowing(monkeypatch):
    monkeypatch.setattr(R, "generate_batch", scripted([["<command>true</command>"]]))
    tpl = {t.id: t for t in TEMPLATES}["delete-except-keep"]            # max_turns 4
    out = R.make_rollout([tpl], think_budget=8, answer_cap=8, max_tokens=70)(
        [prompt(tpl)] * 2, trainer(2))
    assert all(len(c) <= 70 for c in out["completion_ids"])
    assert "too_long" in out["stop_reason"]


def _containers():
    return set(subprocess.run(["docker", "ps", "-q", "--filter", "ancestor=otto-sandbox"],
                              capture_output=True, text=True).stdout.split())


@needs_docker
@needs_tok
def test_a_rollout_leaves_no_containers_or_temp_dirs_behind(monkeypatch):
    # Compare before/after: other work may legitimately have containers running, and
    # an absolute "none" would blame this rollout for someone else's orphans.
    monkeypatch.setattr(R, "generate_batch", scripted([["<command>true</command>"], ["<done></done>"]]))
    tmp = tempfile.gettempdir()
    dirs = lambda: {d for d in os.listdir(tmp) if d.startswith(("rollout-", "episode-"))}
    dirs_before, boxes_before = dirs(), _containers()
    R.make_rollout([TPL])([prompt()] * 2, trainer(2))
    assert dirs() <= dirs_before, f"leaked dirs: {dirs() - dirs_before}"
    assert _containers() <= boxes_before, f"leaked containers: {_containers() - boxes_before}"


@needs_docker
@needs_tok
def test_a_failing_rollout_still_cleans_up(monkeypatch):
    # The failure path is where leaks actually happen: generation raising mid-episode.
    def boom(*a, **k):
        raise RuntimeError("generation died")
    monkeypatch.setattr(R, "generate_batch", boom)
    boxes_before = _containers()
    with pytest.raises(RuntimeError, match="generation died"):
        R.make_rollout([TPL])([prompt()] * 2, trainer(2))
    assert _containers() <= boxes_before, "containers survived a crashed rollout"


@needs_docker
@needs_tok
def test_prompts_that_are_not_repeated_consecutively_are_rejected():
    other = {t.id: t for t in TEMPLATES}["delete-dir"]
    with pytest.raises(AssertionError, match="consecutively"):
        R.make_rollout([TPL, other])([prompt(TPL), prompt(other)], trainer(2))


@needs_tok
def test_a_prompt_that_is_not_ours_fails_loudly():
    bad = [{"role": "system", "content": "s"}, {"role": "user", "content": "something else"}]
    with pytest.raises(KeyError):
        R.make_rollout([TPL])([bad] * 2, trainer(2))


def test_make_rollout_validates_its_configuration():
    with pytest.raises(AssertionError, match="cannot fit"):
        R.make_rollout([TPL], think_budget=256, answer_cap=96, max_tokens=300)
    dup = TaskTemplate.from_dict({**TPL.to_dict(), "id": "other-id"})
    with pytest.raises(AssertionError, match="share an instruction"):
        R.make_rollout([TPL, dup])


# ============================================ opt-in: the real model on the GPU

@pytest.mark.skipif(os.environ.get("OTTO_LIVE") != "1" or TOK is None,
                    reason="set OTTO_LIVE=1 to run against the real model")
def test_live_budget_forces_close_on_the_real_model():
    from transformers import AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
                                                 local_files_only=True).to(dev)
    ctx = TOK(TOK.apply_chat_template(prompt(), tokenize=False, add_generation_prompt=True),
              add_special_tokens=False).input_ids
    force = R.budget_forcing_ids(TOK)
    out = R.generate_batch(model, TOK, [ctx, ctx], dev, 16, 24, force)
    for toks, mask in out:
        assert len(toks) <= 16 + len(force) + 24 and len(toks) == len(mask)
        assert R.THINK_CLOSE in toks, "a 16-token budget must have closed the thinking"
        i = toks.index(R.THINK_CLOSE)
        if i >= 16:                                   # forced, not self-closed
            span = range(16, 16 + len(force))
            assert all(mask[j] == 0 for j in span if j < len(mask))


# ================================================================ traces (debug only)

@needs_tok
def test_segments_split_model_and_environment_spans_exactly():
    comp = TOK("<command>ls</command>", add_special_tokens=False).input_ids + [R.IM_END]
    env = R.env_chunk(TOK, "OBS", True)
    tail = TOK("<done></done>", add_special_tokens=False).input_ids + [R.IM_END]
    seq = comp + env + tail
    mask = [1] * len(comp) + [0] * len(env) + [1] * len(tail)
    segs = R.segments(TOK, seq, mask)
    assert [s["role"] for s in segs] == ["model", "env", "model"]
    assert [s["tokens"] for s in segs] == [len(comp), len(env), len(tail)]
    assert "OBS" in segs[1]["text"] and segs[2]["text"] == "<done></done><|im_end|>"


@needs_docker
@needs_tok
def test_trace_sink_records_why_each_episode_scored_what_it_did(monkeypatch):
    monkeypatch.setattr(R, "generate_batch", scripted(
        [["<command>rm *.tmp *.bak</command>", "<command>rm -rf *</command>"], ["<done></done>"]]))
    traces = []
    R.make_rollout([TPL], trace=traces)([prompt()] * 2, trainer(2))
    assert [t["reward"] for t in traces] == [1.0, 0.0]
    assert traces[0]["template"] == TPL.id and traces[0]["commands"] == ["rm *.tmp *.bak"]
    assert traces[0]["failures"] == [] and traces[0]["gate_failed"] is False
    assert traces[1]["gate_failed"] is True and any("unexpected" in f for f in traces[1]["failures"])
    assert "- " in traces[1]["diff"]                                # it deleted things
    assert [s["role"] for s in traces[0]["segments"]] == ["model", "env", "model"]


@needs_docker
@needs_tok
def test_no_trace_is_built_unless_asked_for(monkeypatch):
    monkeypatch.setattr(R, "generate_batch", scripted([["<done></done>"]]))
    out = R.make_rollout([TPL])([prompt()] * 2, trainer(2))          # trace=None: training path
    assert "segments" not in out and "diff" not in out


@needs_docker
@needs_tok
def test_chunked_rollout_keeps_group_seeds_distinct_and_order(monkeypatch):
    # 2 groups of 2 with at most 2 episodes generating at once => two chunks. The second
    # chunk must not restart its group numbering, or both groups would get one fixture.
    seen_batches = []
    base = scripted([["<command>true</command>"], ["<done></done>"]])

    def gen(model, tok, contexts, *a, **k):
        seen_batches.append(len(contexts))
        return base(model, tok, contexts, *a, **k)

    monkeypatch.setattr(R, "generate_batch", gen)
    traces = []
    out = R.make_rollout([TPL], trace=traces, batch=2)([prompt()] * 4, trainer(2, step=5))
    assert max(seen_batches) == 2                                      # never more than a chunk at once
    assert [t["seed"] for t in traces] == [R.group_seed(5, 0)] * 2 + [R.group_seed(5, 1)] * 2
    assert R.group_seed(5, 0) != R.group_seed(5, 1)
    assert len(out["prompt_ids"]) == len(out["completion_ids"]) == len(out["env_mask"]) == 4
    assert len(out["episode_reward"]) == len(out["stop_reason"]) == len(out["n_commands"]) == 4
    assert out["logprobs"] is None


@needs_docker
@needs_tok
def test_batch_smaller_than_a_group_still_runs_whole_groups(monkeypatch):
    seen = []
    base = scripted([["<done></done>"]])

    def gen(model, tok, contexts, *a, **k):
        seen.append(len(contexts))
        return base(model, tok, contexts, *a, **k)

    monkeypatch.setattr(R, "generate_batch", gen)
    R.make_rollout([TPL], batch=1)([prompt()] * 2, trainer(2))       # G=2 > batch=1: a group is never split
    assert max(seen) == 2


@needs_docker
@needs_tok
def test_progress_events_describe_every_chunk_and_never_break_the_rollout(monkeypatch):
    monkeypatch.setattr(R, "generate_batch", scripted(
        [["<command>rm *.tmp *.bak</command>"], ["<done></done>"]]))
    events = []
    R.make_rollout([TPL], batch=2, progress=events.append)([prompt()] * 4, trainer(2))
    kinds = [e["event"] for e in events]
    assert kinds.count("chunk_start") == 2 and kinds.count("chunk_end") == 2
    assert kinds[0] == "chunk_start" and kinds[-1] == "chunk_end"
    assert {e["chunks"] for e in events} == {2} and {e["chunk"] for e in events} == {1, 2}
    for e in events:
        if e["event"] == "turn":
            assert e["running"] + e["finished"] == 2 and e["generated"] <= 2 and e["tokens"] > 0
    ends = [e for e in events if e["event"] == "chunk_end"]
    assert all(e["episodes"] == 2 and sum(e["stops"].values()) == 2 for e in ends)

    def broken(event):
        raise RuntimeError("logging bug")

    monkeypatch.setattr(R, "generate_batch", scripted([["<done></done>"]]))
    out = R.make_rollout([TPL], progress=broken)([prompt()] * 2, trainer(2))
    assert len(out["episode_reward"]) == 2                           # the rollout survived


def test_release_memory_is_safe_everywhere_and_calls_the_allocator(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("empty"))
    R.release_memory()
    assert calls == ["empty"]
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    R.release_memory()                                              # a failure must never propagate
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    R.release_memory()
