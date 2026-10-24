# Terminal agents and RLVR — a primer

How a terminal agent works, how it is trained, and what has to be in place before
training can produce a gradient. Written for this project (a sub-1B policy trained
with RLVR to drive a shell), but the mechanics are general.

Companion docs: [`MODEL_CHOICE.md`](MODEL_CHOICE.md) (why Qwen3-0.6B, and the
pass-rate band) · [`SERVING.md`](SERVING.md) (why SGLang is a server and not a
library dependency).

---

## 1. What a terminal agent actually is

Three separate pieces that are routinely conflated:

```
┌─────────────┐   action   ┌──────────┐   command   ┌────────────┐
│   POLICY    │ ─────────▶ │ HARNESS  │ ──────────▶ │  SANDBOX   │
│  (the LLM)  │            │  (glue)  │             │ (container)│
│  Qwen3-0.6B │ ◀───────── │          │ ◀────────── │            │
└─────────────┘ observation└──────────┘   stdout    └────────────┘
                                │
                          after N turns
                                ▼
                          ┌──────────┐
                          │ VERIFIER │ → reward
                          └──────────┘
```

The **policy** is a text-in/text-out model. It cannot run anything. The **harness**
is what makes it an agent: it builds the prompt, parses the model's output to extract
an action, executes it, formats the result back into the conversation, and decides
when to stop. The **sandbox** is the actual world state.

The harness is the part that gets underestimated. It owns every decision that matters
operationally: the action format, how much stdout to inject before truncating, the
step limit, the per-command timeout, and what happens on a malformed action. TMax
found the Terminus-2 harness "more brittle with small models, as it requires agents
to send raw keystrokes" — same model, different harness, materially different scores.
**For a 0.6B policy, a forgiving harness with a simple action format is worth more
than any training trick.**

### One episode, concretely

```
[system]  You have a bash tool. Work step by step.
[user]    Delete every .log file under /work older than 7 days.

[assistant] <think>I should look before I delete.</think>
            bash: find /work -name '*.log' -mtime +7
[tool]      /work/a.log
            /work/sub/b.log

[assistant] <think>Two matches. Safe to delete those specific paths.</think>
            bash: find /work -name '*.log' -mtime +7 -delete
[tool]      (exit 0, no output)

[assistant] <think>Verify nothing stale remains.</think>
            bash: find /work -name '*.log' -mtime +7
[tool]      (empty)

[assistant] Done.
```

The verifier then runs against the container's **final filesystem state**: are the two
old logs gone, are the recent logs still present, were any non-`.log` files touched?
That boolean is the entire training signal — one number for the whole episode.

Note what the reward does **not** see: whether the model reasoned well, whether step 2
was lucky, whether it would work on a different fixture. That gap is the central
difficulty of the whole enterprise — see §6.

---

## 2. What "training" means here

```
  ┌──────────────────────────────────────────────────────┐
  │  1. ROLLOUT   policy plays G episodes per task       │
  │               (inference engine + harness + sandbox) │
  │  2. VERIFY    each episode → scalar reward           │
  │  3. SCORE     rewards → per-token advantages         │
  │  4. UPDATE    gradient step on the policy            │
  │  5. SYNC      push new weights to inference engine   │
  └──────────────────────────────────────────────────────┘
                        repeat ~500 steps
```

**Step 5 is where most engineering pain lives.** The inference engine holds a *copy*
of the weights. The moment you take an optimizer step, that copy is stale. So every
training step you must either hot-swap weights into the running engine, or (TRL's
colocated mode) put the engine to sleep, train, wake it, reload. This is why TMax ran
**2 nodes for training and 6 for inference** — rollouts dominate, and they are
sequential within an episode.

It also directly causes the subtlest bug in the field (§3, train-inference mismatch).

---

## 3. The algorithm

### GRPO, mechanically

For a task $q$, sample $G$ episodes. Get rewards $r_1 \dots r_G$. Then:

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

Every token in episode $i$ gets advantage $A_i$. Update with a PPO-style clipped
objective:

$$\mathcal{L} = -\frac{1}{\sum_i |o_i|}\sum_i \sum_t \min\!\left(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) A_i\right), \quad \rho_{i,t} = \frac{\pi_\theta(o_{i,t})}{\pi_{\text{old}}(o_{i,t})}$$

**The whole point: the group mean replaces the critic.** PPO needs a learned value
network to estimate "how good is this state" — a second model, as large as the policy,
itself trained and itself unstable. GRPO estimates the baseline by Monte Carlo
instead: play the same task 8 times and the average score *is* the baseline. For a
0.6B policy on a 24 GB card, not having to hold and train a critic is the difference
between feasible and not.

Three consequences drive everything else.

**(a) Uniform groups produce no gradient.** If all $G$ episodes fail,
$r_i - \text{mean}(r) = 0$ for every $i$. Zero advantage, zero gradient, wasted
compute — and $\text{std}=0$ is a division by zero besides. This is the 30–70%
pass-rate band in [`MODEL_CHOICE.md`](MODEL_CHOICE.md), and why TMax sets
`filter zero-std samples = true` with active sampling to refill batches.

> The metric that matters is **the fraction of tasks producing a *mixed* group**, not
> the aggregate pass rate. Your task set must be calibrated to your model, not to the
> problem domain.

**(b) Credit assignment is absent within an episode.** Every token in a 5-turn episode
gets the same advantage. Three smart actions and one dumb one, episode fails — all four
are punished equally. With one reward over ~2,000 tokens the signal is heavily diluted.
This is what MT-GRPO's turn-level credit assignment attacks, and why River's repetition
penalty is applied to *the token indices of the offending turns* rather than the whole
episode.

**(c) Token-level vs sequence-level normalization matters.** Note the
$\frac{1}{\sum_i |o_i|}$ — sum over *all* tokens in the group, divided by the group's
total token count. The naive alternative averages within each sequence first, which
under-weights tokens in long sequences, so a rambling 10-turn failure is penalized less
per-token than a crisp 2-turn failure. DAPO's token-level loss fixes this; TMax uses it.
In AlignTune: `loss_type='dapo'`, `importance_sampling_level='token'`.

### The multi-turn wrinkle: loss masking

An episode is one long token sequence, but **the model only produced some of it.** The
`[tool]` blocks are environment output. Train on those tokens and you are teaching the
model to predict `find` output — useless and actively harmful.

So the harness must emit a **loss mask** marking which spans are assistant-generated.
Getting this wrong is the most common silent bug in multi-turn RL: training proceeds,
loss goes down, nothing improves.

This is exactly the bookkeeping TRL's `rollout_func` exists to supply. AlignTune does
not pass it through — `GRPOTrainer` is constructed with six arguments and `rollout_func`
is not among them (`backends/trl/rl/grpo/grpo.py:345`). Since `setup_trainer()` is an
overridable method and AlignTune pins `trl==1.7.1` (which *has* `rollout_func`), the fix
is a ~30-line subclass rather than a fork:

```python
class TerminalGRPOTrainer(TRLGRPOTrainer):
    def setup_trainer(self):
        super().setup_trainer()
        self.trainer = GRPOTrainer(
            model=self.model, args=self.trainer.args,
            train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer, reward_funcs=self.reward_functions,
            rollout_func=terminal_rollout,       # our episode loop + loss mask
        )
```

### Train-inference mismatch, and what DPPO does

$\rho = \pi_\theta / \pi_{\text{old}}$ assumes $\pi_{\text{old}}$ is *the distribution
that actually generated the data*. But SGLang/vLLM and the HF trainer compute logprobs
with different kernels, different batching, different reduction orders, in bf16. Feed
the same tokens through both and the numbers differ. So $\pi_{\text{old}}$ is measured
slightly wrong, the ratio is wrong, the gradient points slightly wrong — and over long
sequences the per-token discrepancies compound.

Two defenses, both in TMax:

- **FP32 LM head** — fix it at the source; the final projection is where most numeric
  divergence enters. Their Fig. 4 shows this collapsing the max logprob difference for
  Qwen 3.5. **Qwen 3 showed no spikes without it**, which is why we skip it here.
- **DPPO** — detect and discard it: mask tokens where the inference/training divergence
  exceeds a threshold (binary TV approximation, threshold 0.1). If you cannot trust a
  token's $\pi_{\text{old}}$, do not learn from it. Their Fig. 7 shows it limiting
  collapse relative to GRPO.

DPPO is not available in AlignTune. `loss_type='cispo'` is the nearest cousin — same
lineage of handling mismatch by importance-sampling masking.

### Entropy collapse

As training sharpens the policy, entropy falls, exploration stops, and the model locks
into one rigid template — after which no new behaviour can be discovered and reward
plateaus or collapses. The standard fix is **clip-higher**: set
$\epsilon_{\text{high}} > \epsilon_{\text{low}}$ (DAPO uses 0.28/0.20). The upper clip
bound is what blocks *increases* in low-probability tokens, so relaxing it specifically
preserves the ability to push up exploratory ones. In AlignTune: `epsilon_high`.

TMax also found **larger groups stabilize** (G=32 vs G=8, their Fig. 8) — more samples,
lower-variance baseline — and that a small KL penalty reduced collapse but cost reward,
which is why they ship **β = 0**.

---

## 4. What has to be in place

| Component | Role | Ours |
|---|---|---|
| Policy | the model being trained | Qwen3-0.6B |
| Inference engine | fast batched rollouts; weights hot-swapped each step | SGLang (track requirement) |
| Harness | prompt format, action parsing, observation injection, step/token limits, **loss mask** | to build |
| Sandbox | isolated, resettable world state, one per episode | `sandbox/` ✅ |
| Task set | calibrated to land in the mixed-group band | CuratorKit + TMax-15K |
| Verifier | final-state predicates → reward | `src/verifier` |
| Trainer | GRPO loss, optimizer, weight sync | AlignTune |
| Monitor | reads traces for hacking vs. capitulation | `src/monitor` |

Two cost realities worth internalizing. **Rollouts dominate, not gradients** — an
episode is ~5 sequential model calls plus ~5 container executions, and you need $G$ of
them per task. And **sandbox orchestration is the bottleneck**: TMax ran Podman on the
same nodes as inference to save money and hit resource contention, with models even
showing "awareness about their infrastructure setup."

---

## 5. Single-turn vs multi-turn

What decides it is **task shape, not domain**.

| | Shape | Needs loss mask? |
|---|---|---|
| **Single-turn** | prompt → completion → test | no |
| **Self-refine** | write → test → traceback → revise | yes |
| **Agentic** | read → grep → edit → run → read → … | yes |

The deciding question: **does the model need information it does not have in the
prompt?** Fully specified task ("write a function with this signature") → single-turn,
and multi-turn adds only cost. Task requiring discovery of state (which files exist,
what the error says, whether the fixture has 2 or 200 matches) → multi-turn by
necessity.

This project is agentic multi-turn. The behaviour we want to train — inspect before
destroying — exists *only* because looking is a separate turn.

---

## 6. What the reward sees, and what it must not

Two axes that get collapsed into one:

| | **Verifiable** (programmatic) | **Judged** (learned model) |
|---|---|---|
| **Outcome** | unit tests, filesystem state ← *canonical RLVR* | LLM-judge on the final answer |
| **Process** | River's repetition penalty (Jaccard ≥ 0.8) | process reward models, CoT judges |

The **V** in RLVR excludes the right-hand column: the reward must be a *program*, not a
model. It is not a claim about outcome vs. process.

So RLVR does **not** require seeing the reasoning. The trace plays three distinct roles
and only one of them is the reward:

1. **As reward** — not needed. Reward is `f(final state)`; it never reads the trace.
2. **As the thing being optimized** — needed, but unobserved. The gradient flows
   *through* the reasoning tokens because they precede the action. Reasoning is shaped
   without ever being scored; this is how long CoT emerged in R1-style training.
3. **As instrumentation** — needed, and separate from RL. This is `src/monitor`.

### Observe the reasoning. Do not reward it.

River tested putting process into the reward and kept only one of two terms:

- verify-before-done reward → behaviour changed, **no performance gain**. Dropped.
- repetition penalty → behaviour changed, **gain**. Kept.

Verify-before-done is their *strongest single predictor* (AUC 0.632) and rewarding it
did nothing: "verification is correlated with task performance but does not directly
cause task success." Optimizing a correlate of success produces the correlate, not the
success.

The repetition penalty survives as RLVR because it is a Jaccard threshold — a
deterministic degenerate-loop detector, not a grade on reasoning quality.

**Consequences for this project:** reward = outcome verifier, optionally plus the
rule-based repetition penalty. The monitor stays **read-only**. Put the monitor in the
reward and you have left RLVR, created a gameable learned target, and your detector
stops detecting — the policy will optimize against it. Likewise, **no format reward on
`<think>` tags**: that pays for the appearance of reasoning. If the policy needs to
learn the format, do it in SFT where there is no optimization pressure to game it.

### Why keep thinking mode at all

Not because the reward needs it:

1. **Capability** — reasoning tokens are computation before committing to an
   irreversible action; on hard tasks that improves outcomes, which the reward *does* see.
2. **Diagnosis** — you cannot debug a 0.6B agent whose failures are opaque.
3. **Evidence** — PS 6 asks us to improve step-by-step reasoning, so we must
   *demonstrate* reasoning changed. Observation is the deliverable; the reward stays
   outcome-only.

### Measuring reasoning without rewarding it

River's 12 behaviour features, all extracted deterministically by rule-based parsing
with no model in the loop. Univariate AUC-ROC for predicting task success:

| Feature | AUC | Dir |
|---|---|---|
| verify before done — read-only inspection among the ≤3 commands before `<done>` | 0.632 | + |
| repetition — fraction of commands that are near-duplicates (Jaccard ≥ 0.8) | 0.631 | − |
| cmd diversity — distinct command heads ÷ total commands | 0.615 | + |
| python heredoc — fraction invoking `python`/`-c`/heredocs for the task | 0.604 | **−** |
| inspect before act — read-only commands before modifying anything | 0.596 | + |
| n errors · gave up flag · error rate · error recovery · inspect first · tool help · premature done | 0.507–0.563 | mixed |
| **all 12 combined** | **0.735** | |

For contrast, single-turn *skill* features sit at 0.48–0.56 — chance. And from SFT to
RL, behaviour AUC rises 0.64 → 0.74 while skill AUC stays flat: **pretraining and SFT
install the bash knowledge; RL teaches when to look before you leap.**

---

## 7. Why this is hard

- Sparse terminal reward over a long trajectory → diluted signal, no within-episode credit
- Uniform groups → no gradient at all, so difficulty calibration is a first-class problem
- Train-inference logprob mismatch → wrong gradients, compounding with length
- Entropy collapse → exploration dies
- Loss masking on environment tokens → silent failure if wrong
- Variable-length episodes → awkward batching; truncated episodes need masking, not punishment
- Reward hacking → the verifier gets gamed, or the model stubs out the task
- Collapse past ~300 steps, worsening above 10 turns

The mitigation for most of that is one decision: **keep episodes ≤5 turns.** Shorter
context, fewer sequential calls, less mismatch accumulation — and TMax's pilots saw no
instability below 5 turns.

Note that reward hacking has two faces that look identical in a filesystem diff and
completely different in the trace. TMax's §D.6 found three hacking cases in TMax-9B
(stubbed `filter.py`, fake `caffe` binary, mock `povray` wrapper), all scoring 0, and
the CoT showed the model was not trying to fool the verifier — it was *giving up*:
"This is getting too complicated… the simplest remaining approach is to create a
complete mock." The monitor should separate **deception** from **capitulation**.

---

## Sources

Papers are in [`../papers/`](../papers/):

- **TMax** — *A simple recipe for terminal agents*, [arXiv:2606.23321](https://arxiv.org/abs/2606.23321).
  Code [`hamishivi/tmax`](https://github.com/hamishivi/tmax) (Apache-2.0); environments
  [`allenai/TMax-15K`](https://huggingface.co/datasets/allenai/TMax-15K) (14,600 tasks).
- **River** — *Learning Generalizable Behaviors for Terminal Agents*, [arXiv:2608.22631](https://arxiv.org/abs/2608.22631).
- **DAPO** — [arXiv:2503.14476](https://arxiv.org/html/2503.14476v1), for clip-higher and token-level loss.
