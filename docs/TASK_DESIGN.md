# Task design

Why our synthetic tasks look the way they do, grounded in what TMax actually did
and in the independent audit that found most of it unsuitable for RL.

Read alongside [`README.md`](README.md) (how terminal-agent RLVR works) and
[`TMAX.md`](TMAX.md) if present. Sources are listed at the end; where a claim comes
from reading TMax's generation code rather than their paper, it is marked **[src]**.

---

## 0. Why this document exists

A synthetic task set is a claim about what the model should learn. If we cannot say
why a task exists, what axis it varies, how its difficulty was chosen and how we
know its verifier is not gameable, then a reward curve tells us nothing — we would
not be able to distinguish "the policy learned the task" from "the policy learned
our verifier's bugs."

The order here is deliberate: understand TMax's design, read the audit of it, and
only then commit to ours.

---

## 1. How TMax builds tasks

### 1.1 Nine orthogonal axes, sampled independently

Each task is *one draw* from nine axes — no multi-stage refinement. Paper Table 8,
cross-checked against `rl_data/generator/task_template_gen.py::random_user_msg`:

| Axis | Role | Cardinality | Source |
|---|---|---|---|
| Domain | coverage | 9 (`security`, `software_engineering`, `file_operations`, `data_querying`, `data_science`, `debugging`, `scientific_computing`, `data_processing`, `system_administration`) | Pi et al. 2026 |
| Skill type | coverage | 4–7 per domain | Pi et al. 2026 |
| Primitive skills | coverage | 20–40 per domain; **3–5 sampled per task** | Pi et al. 2026 |
| Persona | diversity | **5**–18 per domain | TMax |
| Language | diversity | 8, weighted | TMax |
| Task complexity | **difficulty** | 4: `short`, `moderate`, `complex`, `intricate` (30–60 commands) | TMax |
| Command complexity | **difficulty** | 3: bash-only → bash+code → bash+code+services | TMax |
| Fixture kind | diversity | 7: `text_only`, `image`, `audio`, `video`, `stripped_binary`, `vendored_package`, `multi_service_compose` | TMax |
| Verifier kind | **difficulty** | 5: `exact_text`, `metric_threshold`, `adversarial_corpus`, `fuzz_equivalence`, `multi_protocol` | TMax |

Details only visible in the code **[src]**:

- **Language weights** are explicit: Python 0.35, C 0.15, Bash 0.15, C++ 0.10, Rust
  0.07, Go 0.07, multi-language 0.06, "any" 0.05.
- **A 35% "real software anchor"** (`_ANCHOR_PROBABILITY = 0.35`) injects a named
  real-world system as inspiration, "use as inspiration, not literally".
- **Primitive skills are drawn across the whole domain, not from the chosen skill
  type.** `random_user_msg` builds `all_skills` by flattening *every* skill type in
  the domain before `random.sample`. So `skill_type` labels a task without
  constraining its skills — the two axes are not actually nested, despite reading
  that way in the paper.
- **Corpus kinds with per-axis "bucket upweighting."** `legacy` uses 3 complexity
  values and the legacy verifier/fixture only; `rl_v2` sets `verifier_kind` and
  `fixture_kind` multipliers to infinity so the legacy value is *never* emitted, and
  `task_complexity` M=3.0 → 75% `intricate`. The published 15K is a deliberate blend
  of a 10K legacy corpus and a 5K v2 corpus, engineered so the combined set is
  roughly uniform across complexity buckets.

The generation instruction given to the model (Gemini 3.1 Pro) is worth quoting,
because it is the whole quality bar **[src]**:

> CREATE A NOVEL TASK that: 1. Combines 3-5 of the primitive skills above in a
> creative, unexpected way 2. Is NOT a recreation of common coding challenges
> 3. **Is challenging to solve but easy to verify** 4. Has clear, unambiguous
> specifications 5. Is a realistic end-to-end scenario…

### 1.2 Graded verifiers are the difficulty knob

Paper Table 9. This is the most directly reusable idea in the paper:

| Kind | Pass criterion | Difficulty knob |
|---|---|---|
| `exact_text` | output exactly equals ground truth | — (legacy) |
| `metric_threshold` | a numeric metric meets a threshold (SSIM ≥ 0.95, speedup ≥ 1.3×) | **the threshold value** |
| `adversarial_corpus` | rejects every malicious item, preserves every benign one | corpus size; per-corpus pass rate |
| `fuzz_equivalence` | matches a reference oracle bit-exactly on N random inputs | **N**; input distribution |
| `multi_protocol` | a service the agent brings up answers real protocol requests | number of protocols/conditions |

The point is continuity. Exact-string equality is a cliff: a task is either solved or
not. A threshold is a dial, which is what lets you aim at a pass-rate band instead of
discovering after the fact that everything scores 0 or 1.

### 1.3 What they validate — and what they explicitly do not

From `rl_data/README.md` **[src]**:

> **Scalability via soft filtering.** We *skip* teacher-based correctness validation.
> The pipeline only guarantees each task is **executable** by building a container
> image for it and running its tests; task quality is left to RL's soft filtering
> (zero-pass-rate rollouts contribute no gradient and are dropped at train time).

So the gate is: *does the image build and do the tests run?* Not: *is the task
correct, unambiguous, or its verifier sound?* The paper's §3.1 defends this by noting
the all-zero rate is low in practice (<8 samples filtered per batch).

They do generate a `test_initial_state.py` per task, so the *starting* state is
checked. It is the task–verifier–instruction consistency that is never checked.

---

## 2. The audit: what soft filtering costs

River ran a rubric-based audit of three synthetic environment pools with GPT-5.4,
reading the three artifacts that must agree — instruction, verifier, environment —
and asking one question: *would a competent agent that faithfully follows the
instruction both understand the task and pass the verifier?*

| Pool | CLEAN | dominant defect |
|---|---|---|
| **TMax-15K** | **35.8%** | 40.4% `VERIFIER-TOO-WEAK` |
| TermiGen | 10.1% | 51.4% too-weak, 19.6% mismatch |
| TerminalTraj-5K | 3.3% | 67.2% too-weak |

TMax is the *cleanest* of the three and **fewer than 40% of its environments are
suitable for RL training.** River's curated subset, RIVER-TMax-3.5K, is what remains
after rubric filtering plus removing pass@2 = 0 environments.

### 2.1 The eight verdict categories

This taxonomy is the most valuable artifact in either paper for our purposes, because
every category is a failure we can design against:

| Verdict | Meaning |
|---|---|
| **CLEAN** | Instruction clear and self-sufficient; a correct-per-instruction solution passes; and the verifier genuinely validates correctness — it *recomputes or decodes the true answer from the input*, rather than merely checking output shape. A hard-but-fair task is still CLEAN. |
| **VERIFIER-TOO-WEAK** | Passes without the task being solved: checks only shape/format/types/count, trusts a self-reported value, greps source for keywords, checks a hardcoded subset, or tests only fixed inputs so the answer can be hardcoded. **A fabricated, plausibly-shaped output would pass.** |
| **INSTR-VERIFIER-MISMATCH** | Verifier checks something the instruction does not imply, or contradicts it (different path, conflicting hardcoded value, stricter format). A faithful solution still fails. |
| **INSTR-ENV-MISMATCH** | Instruction or verifier references paths, files, dependencies or services the environment does not provide. |
| **ANSWER-LEAK** | The instruction, or a shipped file, reveals the expected answer. No genuine work required. |
| **INSTR-AMBIGUOUS** | Underspecified: two equally reasonable readings produce different outputs and the verifier accepts only one. (Multiple valid *paths* is fine.) |
| **TASK-TRIVIAL** | No meaningful work regardless of verifier strength: write a constant, echo a value, copy a file. |
| **OTHER** | Buggy or crashing verifier; other genuine defects. |

### 2.2 Two concrete failures

**False positive (reward too high).** A task requires a C++ implementation. The
agent's C++ program keeps timing out, so it writes a Python script instead, which
produces the correct JSON. The trajectory openly admits this. River's analysis:

> The verifier observes an executable C++ file and a correct JSON file, **never checks
> their causal relationship**, and returns full reward. … Existence-only verification
> of the binary lets a forbidden Python bypass receive the same reward as a working
> C++ implementation.

**False negative (reward too low).** A task asks for `p = (1 − exp(−x)) · p_prior`
to six decimal places. For `x = 2.99, p_prior = 0.7` the verifier hardcodes
`0.664815` — which is simply wrong. Correct solutions score 0.

Both are *verifier* defects, not model failures. The first teaches the policy that
shortcuts pay. The second teaches it that correct work doesn't.

### 2.3 Diagnosis

TMax's generation instruction asks for tasks "challenging to solve but **easy to
verify**." With no consistency check, a generator under that instruction optimises
the second clause — and the cheapest way to make something easy to verify is to check
its *shape* rather than recompute its *answer*. That is precisely
`VERIFIER-TOO-WEAK`, and it is 40.4% of the corpus.

So the 40.4% is not sloppiness. It is the predictable consequence of the soft-filtering
decision, and it is the strongest available argument for our phase-0 ordering: build
and test the verifier before generating anything.

**Caveat, stated by River themselves:** the audit "may contain both false positives
and false negatives." Their Table 7 blind re-judge is instructive — of 15 GPT-5.4
`CLEAN` labels an independent judge agreed with only 6, while of 15 `defective`
labels it agreed with 94/105 across categories. **Defect labels are far more reliable
than clean labels.** Read 35.8% CLEAN as an upper bound, and treat "this task is fine"
as the claim needing evidence.

---

## 3. Our design

### 3.1 Principles

1. **The verifier is the artifact; the task is its documentation.** River's audit says
   the verifier is where synthetic task sets fail. We write the verifier first and the
   instruction to match it, not the reverse.
2. **No task enters the set unvalidated.** We reject soft filtering (§1.3). At 0.6B we
   cannot distinguish a broken task from a task the model cannot do, so RL's
   zero-pass-rate filter carries no information for us.
3. **Difficulty is a dial we set, not a property we discover.** Graded verifiers and
   the complexity axes exist to hit the mixed-group band.
4. **Every axis must earn its cardinality.** TMax needed combinatorial breadth for a
   15K corpus. We need a few hundred validated tasks a 0.6B model can sometimes solve.
   Breadth we cannot verify is a liability.

### 3.2 Axes: adopt, reduce, drop

| TMax axis | Ours | Why |
|---|---|---|
| Domain (9) | **reduce to 4–5** | Keep `file_operations`, `system_administration`, `debugging`, `data_processing`, and `security` as a candidate. Drop `data_querying` (51.3% of its tasks name a DB or heavy library), `data_science` (43.5%) and `scientific_computing` (43.3%) — our image has no Python, gcc or make, only coreutils. **Measured, see §5.** |
| Skill type | **drop as an axis** | TMax's own code shows it does not constrain skill sampling **[src]**, so it labels without controlling. We keep primitive skills and discard the layer above. |
| Primitive skills (3–5/task) | **adopt, reduce to 1–2** | 3–5 composed skills is what produces 30–60-command tasks. A 0.6B policy with ≤5 turns can compose two. |
| Persona | **adopt** | Nearly free diversity, and we can seed it from their 14,601 `scenario` values rather than inventing any. |
| Language (8, weighted) | **drop** | Our tasks are bash-only; no compiler in the image. |
| Task complexity (4) | **adopt, shift down** | Our 4 buckets live *below* their `short`: 1-command, 2-command, 3–4-command, 5-command. |
| Command complexity (3) | **collapse to 1** | bash-only. No code, no services. |
| Fixture kind (7) | **reduce to 2** | `text_only` and a small `binary` (a file to inspect with `file`/`xxd`). The rest need OCR/ASR/ffmpeg/compose. |
| Verifier kind (5) | **adopt 3** | See below. |

The net: a far smaller space, every dimension of which our runtime can actually
execute and our verifier can actually check.

### 3.3 Verifier kinds → our declarative checks

Mapping Table 9 onto `src/verifier/task.py`:

| TMax kind | Our implementation | Status |
|---|---|---|
| `exact_text` | `sha256`, `equals` checks | **done** |
| `metric_threshold` | new check kind: `count_threshold` / `fraction_threshold` — e.g. "≥ 90% of lines rewritten". **The threshold is our difficulty dial.** | **to build — highest priority** |
| `adversarial_corpus` | fixture ships files that must be kept *and* files that must be removed. Our Layer 1 path-set gate expresses this natively: the keeps are absent from `expected_paths`, so touching them fails. | **structurally free** |
| `fuzz_equivalence` | needs an oracle binary in-container | **drop** |
| `multi_protocol` | needs services and networking | **drop** |

`adversarial_corpus` is worth dwelling on: TMax implements it with a generated pytest
comparing two corpora, while for us it is just a fixture with distractor files and
nothing extra in the verifier. Our architecture gets their hardest-to-fake verifier
kind for free, because "changed exactly these paths and no others" is the default.

### 3.4 Quality gates — River's eight categories as acceptance tests

Implemented in [`src/tasks/screen.py`](../src/tasks/screen.py); run it with

```bash
./.venv/bin/python -m src.tasks.screen data/templates/templates.jsonl \
    --accepted-out data/templates/accepted.jsonl
```

Gates run cheapest-first so a bad candidate dies before it costs anything, and
the two that need a container (reference, cheats) are skipped with a loud
`ACCEPT?` marker when no daemon is available — a result produced that way does
not qualify a template for training.

| Category | Our gate | Mechanism |
|---|---|---|
| `VERIFIER-TOO-WEAK` | **automated** | `cheats.py`: `known_cheats` + 40 `random_cheats` must all score 0, *and* the reference must score 1. "A fabricated, plausibly-shaped output" is exactly `touch expected outputs (empty)`, which we already test. The Layer 1 path-set gate makes shape-only passing impossible. |
| `INSTR-ENV-MISMATCH` | **automated** | `run_reference()` executes the task's shell reference in the real sandbox. A missing path or tool makes it fail. |
| `OTHER` (buggy verifier) | **automated** | `TaskSpec.validate()` — 8 malformed-task classes rejected at load; plus `run_reference()` must score exactly 1.0. |
| `TASK-TRIVIAL` | **automated** | `validate()` already rejects tasks expecting no changes. Plus a pass-rate floor: if a 1-turn trivial policy solves it, drop it. |
| `ANSWER-LEAK` | **flag for review, not a gate** | The naive rule (no `checks` literal appears verbatim in the instruction) **has false positives**: our own `logs-01` trips it, because "write their names to `removed.txt`" legitimately names the files the checks assert. A real leak reveals something the agent should have had to *discover*; a specification names what it must *do*. Flag computed literals (counts, hashes, extracted values) for review; never auto-reject on path names. **See §5.** |
| `INSTR-VERIFIER-MISMATCH` | **semi** | Every path named in `checks`/`expect_*` must appear in the instruction, and vice versa. Catches path drift; cannot catch semantic contradiction. |
| `INSTR-AMBIGUOUS` | **manual + proxy** | No cheap automated test. Proxy: if a *capable* model (not our 0.6B) fails a task at pass@4 while its reference passes, suspect ambiguity. |
| *(their)* `CLEAN` | **not claimed** | Per Table 7, clean labels are the unreliable ones. We assert "passes all gates above," not "is clean." |

Five of eight automated, two semi, one manual-with-a-proxy — and the automated ones
cover the dominant failure mode. That is the robust base: not "we were careful," but
"here is the gate each known defect class must pass, and here is the test that runs it."

### 3.4b Templates, not tasks

A static fixture is memorisable. Training draws ~4,000 prompts over 500 steps, so
with ~100 tasks each is seen ~40 times — enough for a 0.6B policy to learn a literal
command sequence per instruction and never inspect anything. TMax never faced this:
with 14,601 environments for ~4,000 draws, each was seen less than once.

So the generated artifact is a **`TaskTemplate`** (a fixture *recipe*), and a
`TaskSpec` is what you get by instantiating one with a seed:

```
TaskTemplate --instantiate(seed, dir)--> (fixture on disk, TaskSpec)
```

This changes the shape of the problem. If filenames and counts vary by seed, the
expected diff cannot be a static list — it must be *derived from the fixture that
was just generated*. The fixture builder and the verifier therefore stop being
independent artifacts and become two outputs of one generator, which is the only
way they can be guaranteed to agree.

Expectations are written as **selectors over tags** rather than literal paths. Each
file rule tags what it produces; instantiation resolves tags to the concrete paths
that seed created:

| selector | resolves to |
|---|---|
| `{"tag": T}` | the paths that rule tagged `T` |
| `{"count_of_tag": T}` | how many there were |
| `{"names_of_tag": T}` / `{"basenames_of_tag": T}` | their paths / basenames |
| `{"lines_of_tag": T}` | total non-empty lines across them |
| `{"ints_of_tag": T}` / `{"sum_of_tag": T}` / `{"max_of_tag": T}` | the `{int}` values generated into them |

Selectors nest anywhere, including inside a check's arguments — which is how
`line_count` stays a plain integer comparison and `task.py` needs no knowledge of
templates at all. A check carrying `for_each_tag` instead of `path` is **expanded**
into one check per generated path, which is the only way to assert a per-file
property (a permission, a regex) when the filenames are not known in advance.

Counts are ranges (`"count": [2, 4]`), and that is what forces the agent to look:
it cannot know *how many* files match without inspecting.

### 3.5 Difficulty calibration protocol

Difficulty is set, then measured, then adjusted:

1. Generate a task at a chosen `task_complexity` bucket.
2. `run_reference()` → must score 1.0. (Task is sound.)
3. Run `cheats.py` → all must score 0.0. (Verifier is sound.)
4. Sample **G = 8 rollouts** from the untrained policy through SGLang.
5. Record the **group outcome**: all-pass, all-fail, or **mixed**.
6. Keep tasks producing mixed groups. For all-fail, lower a dial — fewer turns'
   worth of work, or a looser `metric_threshold`. For all-pass, raise it.

The target metric is **fraction of tasks yielding a mixed group**, not aggregate pass
rate. A set with 50% aggregate pass rate made of half trivial and half impossible
tasks produces *no gradient at all* (see [`README.md`](README.md) §3a). This is also
why `metric_threshold` is the priority check kind: it is the only dial that moves
difficulty continuously without rewriting the task.

### 3.6 Measuring that the set is deliberate

Two numbers to report, both borrowed:

**Balance score** (TMax Eq. 1, derivation §B.3) — the normalised effective number of
categories of an axis's empirical distribution:

$$\text{Balance} = \frac{\exp(H)}{N}, \qquad H = -\sum_{i=1}^{N} p_i \log p_i$$

It lies in $[1/N, 1]$; 1.0 is perfectly uniform coverage, $1/N$ is all mass on one
bucket. Preferred over raw entropy because it is comparable across axes with different
bucket counts and reads as a "fraction of uniform" score. TMax reports 0.998 on domain
and 0.732 on skill type. We report it per axis we keep.

**Decontamination** (TMax §B.4) — 13-gram sliding window, stride 1; a task is flagged
if any window matches any n-gram from an evaluation benchmark. Less critical for us
since we generate rather than scrape, but it is the standard protocol and cheap.

### 3.7 Seeding from TMax without using their tasks

Their environments are not usable for us (per-task Apptainer builds, in-container
verifiers, difficulty calibrated for 9B+ — see the measurement in conversation and
`TMAX.md`). Their **taxonomy is**, and it is already on disk as the parquet's columns:
`domain`, `skill_type`, `primitive_skills`, `scenario`. 14,601 real combinations,
60 MB, no `tasks.zip` needed.

We use these as the conditioning vocabulary for CuratorKit generation — filtered to
the domains we keep and the complexity we need. Borrowing a validated vocabulary is
defensible in a way that inventing one is not.

Note what the parquet does **not** carry: `verifier_kind` and `fixture_kind` are in each
task's `task.json` inside `tasks.zip`, not in the table, and the free-text `truth` column
records them inconsistently. Any per-kind distribution we might quote from the parquet
alone is unreliable (§5).

---

## 4. Open and undecided

- **Graded vs binary reward.** `verify(graded=True)` exists and is off. It is a
  difficulty dial of last resort, and it weakens the RLVR claim. Decide after §3.5
  produces real numbers.
- **How many tasks.** Unknown until we see what fraction survive the gates. River went
  14.6K → 3.5K (24%); our gates are stricter, our generation narrower.
- **Fixture mtimes.** Any task using `-mtime` needs fixture timestamps set explicitly
  with `os.utime` at build time, since `Entry` deliberately excludes mtime. No such
  task exists yet; the fixture builder must handle it before one does.
- **`INSTR-AMBIGUOUS` has no cheap test.** The capable-model proxy in §3.4 costs API
  calls we have not budgeted.
- **Restrictive file modes are unverifiable on macOS.** Docker Desktop's virtiofs
  forces the owner-write bit on across the bind mount, so a file `chmod 0000`'d
  inside the container reads as `0600` on the host (`0400`→`0600`, `0444`→`0644`).
  Mode-based tasks must use a mode that already carries owner-write. Expected to
  differ on Linux; `test_mode_round_trip_through_bind_mount` records the current
  behaviour so a change announces itself. Measured, §5.
- **Content aggregation is now expressible** via `sum_of_tag` / `max_of_tag` /
  `lines_of_tag`, but only over values *instantiation generated*. An answer derived
  from content we did not generate (a checksum of a fixed file, a sort order) still
  has no selector.

---

## Sources

- **TMax** — *A simple recipe for terminal agents*, [arXiv:2606.23321](https://arxiv.org/abs/2606.23321)
  ([`papers/tmax-2606.23321.pdf`](../papers/)). §3.1, Tables 8/9/10, §B.3, §B.4, §C.
- **TMax generation code** — [`hamishivi/tmax`](https://github.com/hamishivi/tmax) (Apache-2.0,
  branch `master`): `rl_data/README.md`, `rl_data/generator/task_template_gen.py`
  (`random_user_msg`, axis constants, weights), `rl_data/generator/completion_test_gen.py`
  (verifier-generation prompts). All **[src]** claims.
- **River** — *Learning Generalizable Behaviors for Terminal Agents*,
  [arXiv:2608.22631](https://arxiv.org/abs/2608.22631)
  ([`papers/river-2608.22631.pdf`](../papers/)). §3.2, §C.1 (rubric), Table 7
  (blind re-judge), §C.4 (failure examples), Fig. 11.
- **TMax-15K** — [dataset](https://huggingface.co/datasets/allenai/TMax-15K) (ODC-BY).
  Parquet columns used as taxonomy vocabulary.

---

## 5. Verification log

Every checkable claim above was tested against the parquet (14,601 rows), the TMax
source, both papers, and our own image. Recorded so the doc can be re-audited.

### Confirmed

| Claim | Result |
|---|---|
| 14,601 tasks | 14,601 ✓ |
| 9 domains, exactly the names listed | ✓ (1,421–1,735 tasks each) |
| 3–5 primitive skills per task | min 3, max 5 ✓ |
| 4–7 skill types per domain | observed 4,5,5,5,6,6,7,7,7 ✓ |
| 8 languages, weighted Python 0.35 / C 0.15 / Bash 0.15 / C++ 0.10 / Rust 0.07 / Go 0.07 / multi 0.06 / any 0.05 | realised 35.1 / 14.6 / 15.0 / 10.2 / 7.2 / 7.2 / 5.7 / 4.9 % ✓ — weights confirmed empirically |
| 4 task-complexity buckets, engineered to be uniform | 25.9 / 25.0 / 24.9 / 24.2 % ✓ |
| 3 command-complexity buckets | 34.0 / 33.2 / 32.8 % ✓ (uniform, consistent with `random.choice`) |
| Our image has no Python, gcc or make | confirmed by running the image: `python3`, `python`, `gcc`, `make` all ABSENT; `jq`, `awk`, `sed`, `grep`, `find` present ✓ |

### Corrected

1. **Persona cardinality is 5–18, not 6–18.** Paper Table 8 says "6–18 / domain";
   `rl_data/README.md` says "5–18"; the data shows `[5, 6, 6, 7, 7, 9, 12, 15, 18]`.
   The code and the data agree against the paper. Doc now says 5–18.

2. **The reason given for dropping `security` was wrong.** Measured fraction of tasks
   whose description or verifier names a heavy library or DB:

   | domain | | heavy-lib mentions |
   |---|---|---|
   | `data_querying` | drop | 51.3% |
   | `data_science` | drop | 43.5% |
   | `scientific_computing` | drop | 43.3% |
   | `data_processing` | keep | 21.9% |
   | `debugging` | keep | 11.7% |
   | `file_operations` | keep | 5.4% |
   | **`security`** | **was "drop"** | **3.8% — the lowest of all nine** |
   | `system_administration` | keep | 3.2% |

   `security` is the *least* library-dependent domain in the corpus. The three
   genuinely unusable domains are `data_querying`, `data_science` and
   `scientific_computing`. `data_processing` at 21.9% is the weakest of the keeps and
   needs per-task filtering.

3. **The `ANSWER-LEAK` gate is not a gate.** Tested the naive rule against our own
   validated task: `logs-01`'s check literals are `a.log` and `sub/b.log`, and both
   appear in its instruction, so the rule rejects a task we know is sound. The
   instruction naming the files *is the specification*; the work is performing the
   deletion and writing the receipt. Downgraded to a review flag restricted to
   computed literals.

4. **`verifier_kind` cannot be measured from the parquet.** It is not a column. An
   earlier estimate of ~8% per graded kind came from regex-matching kind names
   anywhere in the free-text `truth` column; a stricter header-anchored regex matches
   only 0.5% of rows. The loose figure is *consistent* with the documented 10K-legacy
   + 5K-v2 blend (4 kinds × ~8.3%) but is not evidence. Reading `task.json` from
   `tasks.zip` is the only reliable source, and we have not done it at scale.

### Not independently verifiable here

- River's rubric percentages (35.8% CLEAN, 40.4% too-weak) and Table 7 are taken from
  their paper. Reproducing them needs an LLM audit we have not run. River themselves
  state the audit has false positives and negatives.
- TMax's pass@1 42% / pass@8 53% difficulty figures are theirs, measured with
  Gemini-3-Flash-Preview on a 250-task subsample.

### Found while building (added after §5 was first written)

5. **Docker Desktop's virtiofs forces the owner-write bit on.** Measured by setting
   each mode inside the container and reading it back on the host:

   | set inside | seen on host | |
   |---|---|---|
   | `0000` | `0600` | does not round-trip |
   | `0400` | `0600` | does not round-trip |
   | `0444` | `0644` | does not round-trip |
   | `0600`, `0640`, `0644`, `0700`, `0755`, `0777`, `0666` | unchanged | ✓ |

   Any mode lacking owner-write gains it crossing the mount, so the verifier cannot
   observe a non-owner-writable mode on macOS. `quarantine-stale` was rewritten from
   `0000` to `0700`. This is a *platform* limit, not a verifier limit, and should
   relax on Linux.

6. **One of our own cheats was a false positive.** `delete expected, create nothing`
   scored **1.00** against `delete-empty` on all five seeds — because for a task
   whose entire specification *is* a deletion, that cheat performs exactly the
   reference solution. It was rejecting a sound task rather than finding a weak
   verifier. Now emitted only when the task also requires something created or
   modified. The lesson generalises: a cheat suite needs its own false-positive
   discipline, or it quietly shrinks the task set.

### Current state of the set

22 templates, all 22 passing every gate (references executed, ~3,400 cheat runs,
0 escapes). Coverage:

| | |
|---|---|
| operations | create 17, delete 5, modify 5 |
| check kinds | `line_count` 18, `contains_each` 12, `contains` 6, `mode` 3, `equals` 2, `regex` 1 |
| `max_turns` | 3 → 8 templates, 4 → 7, 5 → 7 |

Test coverage: `tests/test_templates.py` (schema, instantiation, variety, leak,
budget, bystander, determinism, filename-collision, every selector, `for_each_tag`
expansion, and five gate-rejection cases), plus one container-gated test that
screens the whole set. Full suite: 236 tests.

Difficulty across the three `max_turns` buckets is still a guess. Only calibration
(§3.5) can settle it, and that needs rollouts.
