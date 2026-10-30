# Verified Repo Distillation

**Nebius x NVIDIA Global AI Hackathon — Coding & Agentic Engineering track**

> A corpus builder that verifies every training pair by **running the commit's own
> tests**, and a measurement apparatus that scores models by **executing** their
> patches. The corpus works. The headline experiment came out negative, and we
> report it that way.

---

## 1. Thesis

Every tool that offers to "fine-tune an LLM on your codebase" mines `git log`
into `(context → patch)` pairs and **assumes each commit was correct**. Nobody
checks. A real repository's history is full of commits whose tests never tested
the change, commits that were green before the "fix", and commits whose patch
was never green at all. We built a pipeline that rejects those commits by
mechanically executing the counterfactual — inject the commit's own new tests
into the *unpatched* parent and require them to **fail**, then apply the commit's
full patch and require them to **pass** — and then we used that verified corpus
to answer the question the field keeps asking with base-vs-tuned comparisons that
cannot separate information from weights: *does tuning on a repository's verified
history change the model's ability to edit that repository?*

The honest answer, at this data scale, is **no**. On 26 temporally held-out
commits, `Qwen3-30B-A3B` base solved 3 and the same model LoRA-tuned on 148
verified pairs from that repository also solved **3** — the same three commits.
Tuning solved no additional task and lost no task. What it did do was make the
adapter markedly worse at emitting a usable SEARCH/REPLACE anchor (patch
application fell from 22/26 to 10/26) while roughly doubling output length
(18,018 → 34,872 completion tokens for the same 26 prompts).

Two things are genuinely novel and both survive this result. First, the
**execution-verified corpus gate** is a mechanism, not a dataset: it is
repository-agnostic, it costs 9.2 s/commit, and it has a named, quantified
failure mode (`reject_pre_passed`) that identifies exactly the commits a
fetch-the-diff pipeline would have ingested as ground truth — 14 of them on
this repository, 2.5% of candidates. Second, the **negative distillation
result** is itself a finding with a mechanism attached: at 148 examples and one
epoch, LoRA on verified repository history shifts *style* (it learned something
about which edit to attempt) and destroys *contract discipline* (it stopped
emitting anchors that apply), without adding capability. That bounds what naive
distillation buys and localises the real bottleneck — anchor emission, not
knowledge of the repository — which is the thing a repair loop or a better
output contract actually fixes. We report the number we measured, we report the
apparatus that makes the number trustworthy (an oracle that scores gold answers
100%, a contamination probe that scores memorisation 0%), and we do not dress
the result up.

---

## 2. Architecture

```
        git repository (hermes-webui, 3526 commits, Mar-May 2026)
                              |
  [1] INGEST                 |  vrd/ingest.py
      classify + select      |  ingest() / classify() / mark_gate_candidate()
                              v
  [2] REPLAY                 |  vrd/replay.py
      hermetic worktree      |  Worktree / build_venv / hermetic_env
      at parent commit       |  run_tests -> run_pytest | run_go_tests
                              v
  [3] GATE  <-- THE CONTRIBUTION
      pre-arm : inject new   |  vrd/replay.py
                tests into   |  counterfactual_verify()
                parent       |    pre.green  -> reject_pre_passed
                must FAIL    |    post not green -> reject_post_failed
      post-arm: apply full   |    both -> verified
                patch, same  |
                tests PASS   |
                              v
  [4] PAIRS                  |  vrd/pairs.py
      windowed context       |  build_pairs() / build_context()
      tests as spec          |  Pair.to_prompt() / Pair.to_target()
      fairness filters       |  edits_visible_in_context() / edits_are_expressible()
                              v
  [5] TRAIN                  |  vrd/train.py
      temporal split         |  export_dataset()
      chat JSONL export      |  qlora_job_spec()
                              v
  [6] EVAL                   |  vrd/evaluate.py
      execution scoring      |  score_patch() / run_arm() / summarise()
      Wilson intervals       |  wilson_interval()
                              |
        +-- run.py oracle:  |  replay gold answers through the SAME scorer
        +-- scripts/contamination_probe.py: forced-continuation memorisation check
```

### Stage-by-stage, with the file and function that implements it

| Stage | File | Key functions | What actually happens |
|---|---|---|---|
| 1. Ingest | `vrd/ingest.py` | `ingest()`, `classify()`, `mark_gate_candidate()`, `GitRepo.log()`, `GitRepo.patch()` | Walks `--first-parent` history with `--numstat`, resolves per-file status via `--name-status`, drops merges / root commits / empty / docs-only / lockfile-only / tests-only / generated / oversize (>25 files, >800 changed lines) commits. A commit becomes a **gate candidate** only if its patch *adds* a test file **and** touches source. `GitRepo.patch()` uses `--binary --full-index`; without those flags any commit touching a PNG or font yields an unappliable patch and the gate records a harness failure as if the commit were unusable. |
| 2. Replay | `vrd/replay.py` | `Worktree`, `build_venv()`, `discover_python_deps()`, `hermetic_env()`, `detect_toolchain()`, `run_pytest()`, `run_go_tests()`, `parse_pytest_summary()`, `parse_go_summary()` | Checks out the **parent** commit into a throwaway `git worktree`. Rebuilds the toolchain: Python via `uv venv` + dependency install, venv cached by the content hash of the dependency files so sibling commits share one install; Go via `go test -count=1` with a shared module cache and `GOTOOLCHAIN=auto`. `hermetic_env()` builds a deliberately small environment — `HOME` inside the sandbox, fixed `PATH`, `PYTHONPATH` pinned to the tree — so nothing is inherited from the operator. |
| 3. **Gate** | `vrd/replay.py` | `counterfactual_verify()`, `verify_many()`, `write_verdicts()` | The two-sided counterfactual. Pre-arm: copy the commit's newly added test files out of the commit and into the unpatched parent tree, run them, require **not green**. Post-arm: delete the injected copies (`git apply` refuses to overwrite them), apply the commit's **full** patch with `git apply --whitespace=nowarn`, run the same tests, require **green**. Any exception becomes `reject_replay` — a harness failure is never allowed to look like a verification. |
| 4. Pairs | `vrd/pairs.py` | `build_pairs()`, `build_context()`, `hunk_windows()`, `_locate_from_patch()`, `edits_are_expressible()`, `edits_visible_in_context()`, `source_only_patch()`, `to_utc_iso()`, `Pair.to_prompt()`, `Pair.to_target()`, `dataset_card()` | Builds the prompt and the target. Context is **windowed around the hunks** (48 KB budget, `VRD_MAX_CONTEXT`), not a per-file truncation, so the lines the answer must reproduce are actually present. Test hunks are stripped from the gold patch (`source_only_patch`) — the tests are the instrument, never the target. Binary blobs and non-code files are excluded. Pairs whose anchors are not visible in the prompt are **dropped as unanswerable**, not scored. |
| 5. Train | `vrd/train.py` | `export_dataset()`, `to_chat_example()`, `qlora_job_spec()`, `write_train_script()` | Exports chat-format JSONL and splits on a **real UTC temporal boundary** (`to_utc_iso` in `vrd/pairs.py`; this repo's commits span `-07:00` to `+08:00`, so lexicographic comparison of raw offsets misplaces commits by up to 15 hours). Emits the Nebius fine-tuning job spec and a QLoRA script. |
| 6. Eval | `vrd/evaluate.py` | `ChatClient.complete()`, `score_patch()`, `run_arm()`, `summarise()`, `wilson_interval()`, `BM25`, `repo_docs()`, `extract_diff()` | Builds the prompt (arm A/C) or a BM25-retrieved context (arm B), calls an OpenAI-compatible endpoint, then scores by **execution**: the patch must apply to a fresh parent worktree **and** the commit's own injected tests must pass. No LLM judge, no similarity heuristic. Pass rates carry Wilson 95% intervals. `--attempts N` adds Aider-style repair turns with the failing test output fed back; `first_attempt_success` is stored separately so pass@1 and pass@k are never conflated. |
| Contract | `vrd/edit_format.py` | `parse_edits()`, `apply_edits()`, `diff_to_edits()`, `render_edits()`, `parse_whole()`, `apply_whole()`, `tree_diff()`, `_locate()` | The SEARCH/REPLACE contract, used identically for training targets and for parsing model output. `diff_to_edits()` is the inverse of the parser so the training target is in the exact format the evaluator parses. Matching is exact and must be unique; a fuzzy fallback (threshold 0.82) exists, but an ambiguous anchor is **refused, never guessed**. Hard caps (20 edits, 300 lines/block, 1200 lines total, 40 000 output chars) are enforced in code, not requested politely in the prompt. |
| Self-check | `run.py oracle` (`cmd_oracle`) | `score_patch()` on `pair.to_target()` | Replays each pair's **own verified patch**, rendered in the model's contract, through the identical scoring path. If a gold answer fails, the harness is wrong and no model number can be trusted. |
| Contamination / dashboard | `scripts/contamination_probe.py`, `dashboard/index.html` | — | The probe asks the model to reproduce held-out post-commit code verbatim and counts line-level matches. The dashboard is a single-file page rendering the ingest funnel and gate verdict breakdown, fed by `run.py report` → `dashboard/report.json`. |

### The prompt contract, verbatim

The prompt the model receives (`Pair.to_prompt()` in `vrd/pairs.py`) is:

```
You are editing an existing repository. Apply the smallest change that implements the request below.

## Request
<commit subject>

## Files
- <path>
...

## Current contents
### <path>
```
<parent-state contents, windowed around the hunks>
```

## Tests that must pass
These are the tests the change has to make pass. They are the specification: do not modify them, and do not assume any other behaviour than what they assert.
### <test file>
```
<the commit's own new test, up to 12 KB>
```

## Output format
Reply with search/replace blocks only. ...
```

The target (`Pair.to_target()`) is the same commit's diff rendered as
`### FILE:` / `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks.

---

## 3. Results

### 3.1 The gate (corpus construction)

Repository: `hermes-webui`, 3 526 commits, Mar–May 2026. 1 813 commits kept by
the ingest classifier, 565 of them gate candidates. Mean 9.2 s per commit
(1.50 hours of wall clock for the full 565).

| Gate verdict | Count | Share of 565 | Meaning |
|---|---|---|---|
| `verified` | **525** | **92.9%** | patch proven **necessary** (test failed without it) and **sufficient** (test passes with it) |
| `reject_pre_passed` | 14 | 2.5% | the commit's own test was already green before the fix — the test did not test the commit |
| `reject_post_failed` | 25 | 4.4% | the commit was never green in this environment |
| `reject_replay` | 1 | 0.2% | harness failure (see the retry note below) |

Ingest funnel (all 3 526 commits): 1 127 merge, 316 docs-only, 121
tests-only, 89 bookkeeping-subject, 51 too-large, 5 too-many-files, 2 empty,
1 generated-file, 1 root-commit → **1 813 kept**, **565 gate candidates**
gated.

The gate ran in two passes. Pass 1 gated all 565 candidates and returned 478
`verified`, 14 `reject_pre_passed`, 23 `reject_post_failed` and **50
`reject_replay`** — harness failures, mostly commits whose patch touched a
binary asset, where `git show --patch` had silently dropped the binary payload
and made the patch unappliable. Pass 2 re-ran *only* the 50 replay failures
(`run.py gate --retry`, which never re-litigates a verification decision) and
recovered **47 of 50** as `verified`, leaving 2 `reject_post_failed` and 1
`reject_replay`. The final table above is the merged result: 525 / 14 / 25 / 1.
Mean wall clock 9.2 s per commit, 1.50 hours total.

### 3.2 The corpus

| property | value |
|---|---|
| pairs | **182** (code-only, after gate + fairness filters) |
| temporal split | p80 boundary `2026-05-19T01:38:13Z` → **148 train / 34 held out** |
| held out and in scope (≤2 files, ≤6 KB patch) | **26** |
| median context | **14 216 bytes** (14.2 KB) |
| median gold patch | **4 018 bytes** (4.0 KB) |
| files touched | 102×1 file, 49×2, 19×3, 9×4, 1×5, 2×6 |
| empty targets | **0** |
| test hunks in targets | **0** |
| binary blobs in targets | **0** |
| non-code files (docs/changelogs) | **0** |
| pairs whose edit anchors are visible in the prompt | **100%** |

Build-time drops, each added because a defect was measured rather than
guessed: **24** patches unplayable in a position-free SEARCH/REPLACE contract
(ambiguous anchor), **285** pairs whose editable lines were not visible in the
prompt (unanswerable as posed), plus docs/changelogs removed from "source
files" after the contamination probe caught `CHANGELOG.md` being treated as
code.

### 3.3 Validation — why the numbers above mean anything

| check | result |
|---|---|
| **Oracle** — each pair's own gold answer replayed through the identical scorer | **26/26 = 100%** on the evaluation set (43/43 = 100% on the full in-scope set) |
| **Contamination** — reproduce held-out post-commit code verbatim | **0/31 lines (0%)**, both models answered `UNKNOWN` |
| **Unit tests** — `pytest tests/ -q` | **48 passed** |
| **Temporal split** | real UTC boundary (it was silently a no-op before the fix: 483 train / 0 test) |

The oracle test found two defects that would otherwise have been reported as
*model* failures: an 80-line block cap that rejected a legitimate 92-line gold
edit, and gold hunks whose SEARCH text occurred twice in the file (a git hunk
is position-anchored; SEARCH/REPLACE is not, so such a patch cannot be
replayed at all). Both were fixed before any model number was quoted.

### 3.4 Benchmark — all four arms

Same 26 held-out pairs, same prompts, one attempt per pair, execution-scored
(the patch must apply **and** the commit's own tests must pass).

| model | patch applied | tests passed | pass rate | Wilson 95% CI | pass per applied |
|---|---|---|---|---|---|
| **Nemotron 3 Ultra 550B** (`nvidia/Nemotron-3-Ultra-550b-a55b`) | 21/26 (80.8%) | **17/26** | **65.4%** | [46.2, 80.6] | 81% |
| **Qwen3-30B-A3B base** (`Qwen/Qwen3-30B-A3B-Instruct-2507`) | 22/26 (84.6%) | **3/26** | **11.5%** | [4.0, 29.0] | 14% |
| **Qwen3-30B-A3B + LoRA** (148 verified pairs) | 10/26 (38.5%) | **3/26** | **11.5%** | [4.0, 29.0] | 30% |
| **Nemotron 3 Super 120B** (`nvidia/nemotron-3-super-120b-a12b`) | 9/26 (34.6%) | **3/26** | **11.5%** | [4.0, 29.0] | 33% |

**The same three commits pass in both Qwen arms**: `4153a47d`, `af1ee81f`,
`f81d24ec`. There is no commit solved by the tuned model that the base model
misses, and none the reverse. Tuning on the verified corpus changed **no**
outcome on this held-out set.

The Ultra arm solves 17 and includes all three of the Qwen arms' commits plus
14 more (`160cd03c`, `1f56fad7`, `2419b3a0`, `3874781a`, `38905f33`,
`39f09a5b`, `3dd2ace4`, `5b9484b8`, `67a20477`, `70402f96`, `8e1ac89b`,
`cc36711b`, `d20da832`, `d6c1b696`). It is reported here as a **reference
ceiling on the same instrument**, not as part of the tuning comparison: the
comparison is strictly within the Qwen family, because Nemotron cannot be
fine-tuned on Nebius Token Factory (both Super-120B and Nano-30B return HTTP
422), so a Nemotron-base vs Nemotron-tuned pair is not constructible on this
platform.

What tuning *did* change, measured:

| metric | Qwen base | Qwen + LoRA | change |
|---|---|---|---|
| patches that apply | 22/26 (84.6%) | 10/26 (38.5%) | **−46.1 points** |
| tests passed | 3/26 | 3/26 | 0 |
| pass rate | 11.5% [4.0, 29.0] | 11.5% [4.0, 29.0] | 0 |
| pass per applied patch | 14% | 30% | +16 points |
| completion tokens, same 26 prompts | 18 018 | 34 872 | +93.6% |
| prompt tokens, same 26 prompts | 120 748 | 120 748 | 0 (byte-identical prompts) |

The prompt token counts are identical across the two Qwen arms by construction:
the tuned arm sees byte-for-byte the same context as the base arm, so any
difference is attributable to the weights.

### 3.5 The fine-tune

| field | value |
|---|---|
| base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| method | LoRA r=16, alpha=32, dropout 0.05, 1 epoch |
| context / batch / packing | 32 768 / 4 / on (mandatory for this model) |
| learning rate | 1e-4, warmup ratio 0.0, weight decay 0.0, max grad norm 1.0 |
| training examples | 148 verified pre-split pairs |
| trained tokens | **1 144 590** (11 steps) |
| job | `ftjob-8c6c6ffcf7734e97b1663d500b64a72f` — **succeeded** |
| artifact | `ftckpt_1b6c9928-7c31-4d78-9595-c2f1df11aa12` |
| adapter files | `adapter_model.safetensors` 26.78 MB + `tokenizer.json` 11.42 MB + configs (38.2 MB total) |
| failed sibling job | dataset rejected: one example 16 751 tokens > 16 384 limit (fixed with 32k context + a length filter) |

Serving: the tuned arm was served on a **rented Vast.ai RTX 6000 Ada (48 GB)**
at $0.502/hour with the `vastai/vllm:v0.32.0-cuda-13.0` image, vLLM with
`--enable-lora`, base `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` plus our PEFT
adapter. The MoE LoRA path was confirmed by vLLM's
`MoE model detected. Using fused MoE LoRA implementation`.

### 3.6 Token budget

| item | tokens |
|---|---|
| inference, Qwen base (26 calls) | 120 748 in / 18 018 out |
| inference, Nemotron Super-120B (26) | 129 227 in / 255 997 out |
| inference, Nemotron Ultra-550B (26) | 129 227 in / 152 653 out |
| contamination probe + smoke tests | ~10 000 |
| fine-tuning (trained tokens) | 1 144 590 |
| **total** | **~1.95M tokens** |

Pricing is not itemised in the platform docs (the billing page covers payment
mechanics, not per-model rates), so exact spend should be read from
**Organisation → Usage**. No dedicated endpoint was ever created, so there are
no Nebius GPU-hour charges; the only GPU cost is the rented Vast instance,
billed per hour and stopped after the eval.

---

## 4. What we claim, precisely — and what we do not

### We claim these things

1. **The gate works and is decisive.** On 565 candidates it kept 525 (92.9%) and
   rejected 40 with a named, reproducible reason. It costs 9.2 s/commit. It
   runs on a 3.7 GB RAM, GPU-less machine.
2. **The gate catches a failure mode that unverified pipelines do not.** The
   `reject_pre_passed` bucket (14 commits, 2.5%) is exactly the class of commit
   a fetch-the-diff pipeline would have ingested as ground truth: the commit's
   own test was already green before the "fix".
3. **The measurement apparatus is trustworthy, and we proved it rather than
   asserting it.** The oracle scores gold answers 26/26 = 100% (43/43 on the
   full in-scope set); the contamination probe finds 0/31 verbatim lines from
   held-out code; the temporal split is a real UTC boundary; scoring is
   execution-only with no LLM judge; 48 unit tests pass.
4. **The corpus is clean.** 0 empty targets, 0 test hunks in targets, 0 binary
   blobs, 0 non-code files, 100% of pairs have their edit anchors visible in
   the prompt.
5. **We measured the tuned arm, and it did not help.** Qwen base 3/26 and
   Qwen+LoRA 3/26, same three commits, identical prompts (120 748 prompt
   tokens in both). Tuning solved no additional held-out task.
6. **Tuning degraded the output contract.** Patch application fell from 22/26
   to 10/26 while output length nearly doubled. This is a measured regression,
   not an interpretation.
7. **A larger model on the same instrument scores much higher.** Nemotron 3
   Ultra 550B reaches 17/26 (65.4%, CI [46.2, 80.6]) where Qwen3-30B reaches
   3/26. The instrument discriminates; it is not saturated at the floor.

### We explicitly do NOT claim any of the following

- **We do not claim that tuning improved anything.** It did not. The pass
  count is identical, the Wilson intervals are identical, and the set of
  solved commits is identical. Any framing of this submission as a tuning
  win would be false.
- **We do not claim that tuning cannot help.** We ran one configuration — 148
  examples, one epoch, LoRA r=16 on attention projections — on a 26-task
  holdout. A larger corpus, more epochs, a different LoRA target set, or a
  repair loop are all untested. The finding is bounded, not universal.
- **We do not claim the 3/26 result reflects Qwen3-30B-A3B's ceiling.** It
  reflects one-shot single-turn editing under a strict execution contract.
  Nemotron Ultra 550B scores 17/26 on the identical instrument, same prompts,
  same scorer.
- **We do not claim our absolute numbers are comparable to NVIDIA's published
  SWE-bench Verified figures (59–60%).** Those come from multi-turn agentic
  harnesses with tool calls on public issues. This is one-shot text editing on
  a private repository — narrower and harder.
- **We do not claim the corpus is general.** One repository at corpus scale.
  `coredns` is wired and gated (3/3 verified at ~173 s/commit), and
  `helm` / `opentelemetry-go` are configured, but none has been run at corpus
  scale. The Go replay path is implemented (`run_go_tests`, shared module
  cache, `GOTOOLCHAIN=auto`) and exercised on coredns, not at scale.
- **We do not claim the contamination probe is conclusive.** It shows the
  held-out code was not reproduced verbatim (0/31 lines), with both models
  answering `UNKNOWN`. A forced-continuation probe without the `UNKNOWN`
  escape hatch would measure absence of evidence rather than assert it; that
  is listed as future work.
- **We do not claim the tuned arm was served on Nebius.** It was not. No
  public API path on Nebius Token Factory creates a model artifact — all 36
  paths in the OpenAPI spec were enumerated and none fall under
  artifact/weights/custom; `GET /v0/model_artifacts` returns 403 (beta,
  on request); `POST /v0/models` with the older LoRA-deploy flow returns 410
  Gone ("LoRA adapter deployment is no longer available"); and every
  `custom_weights_id` form we tried (job id, checkpoint id, the
  `fine_tuned_model_checkpoint` string, `job:ckpt`) returns 422 because the
  validator requires a `model-artifact_` prefix. The adapter is real and
  downloadable (38.2 MB); serving it required a rented GPU.
- **We do not claim Nemotron could have been tuned.** It could not, on this
  platform: both Super-120B and Nano-30B return HTTP 422 as fine-tuning
  bases. That constraint is why the tuned model is Qwen and why the tuned arm
  ran on rented hardware. The corpus, the gate, and the scorer are
  model-agnostic; only the fine-tune step is affected.

---

## 5. Why the negative result is useful to the community

A negative result is worth reporting when it is *mechanistic* — when it tells
you which knob to turn next. Ours does, on two axes.

**First, it bounds what naive distillation buys you.** The common recipe —
"mine the repo's history, LoRA on it, ship it" — produced, on a held-out set
with byte-identical prompts, *zero* additional solved tasks and a large
regression in patch-application rate. If you are about to spend a training run
and a GPU-hour on 148 commits from your own repository, the measured return on
that spend is: no new capability, plus a 46-point drop in contract
reliability. That is a decision-grade number, and it is the kind of thing that
is currently invisible because pipelines report training loss and vibes instead
of a held-out execution-scored comparison.

**Second, it localises the bottleneck, and the bottleneck is not knowledge.**
The tuned model does not do *worse* at the edits it attempts — pass-per-applied
went **up**, from 14% to 30%. It attempts a different, arguably better-targeted
set of edits. What it lost is the mechanical ability to emit a SEARCH block
whose anchor text matches the file exactly: 12 of 26 answers no longer applied
at all. This is a *format and anchoring* failure, not a *domain knowledge*
failure. Two consequences follow:

- **The fix is a repair loop, not more data.** `run.py eval --attempts N`
  already exists in this repo and feeds the failing test output back to the
  model for a corrected patch. Aider's benchmark and NVIDIA's published
  SWE-bench numbers both use repair turns; our one-shot protocol is the
  stricter condition, and our negative result is specifically a one-shot
  result. The most likely explanation for why the adapter's anchor discipline
  decayed is that one-shot, no-feedback training on 148 examples rewards
  "look like a patch" rather than "anchor correctly" — there is no signal in
  the loss that distinguishes the two, and the base model was already better
  at the mechanical part than the corpus was large enough to teach.
- **The measurement tells you which is which.** Because the scorer separates
  `applies` from `tests_pass`, you can see *which* of the two a change
  improved. A pipeline that only reported "did the tests pass" would have
  reported "3/26 → 3/26, no change" and lost the diagnostic. A pipeline that
  reported only an LLM judge's verdict would have reported the tuned model's
  more-plausible-looking edits as an improvement.

There is a third, quieter contribution. The gate itself is a reusable filter:
any team building training data from a repository's history can run it and get
a count of how many of their commits do not test what they claim to test. On
this repository that number is 14 out of 565 candidates, but the *class* of
defect is generic, and the gate is 9.2 s/commit on commodity hardware. The
other measured findings are reusable in the same way: an unfair task produces a
fake finding (an 8 KB per-file context cap made every arm score ~0 because the
model was asked to reproduce text it had never been shown); a corpus can be
silently 36% empty-target from a hunk-header regex that did not tolerate a
git section heading; and a temporal split can be silently a no-op (483 train /
0 test) because the date field was read from the wrong record.

The last one is worth stating plainly, because it is the most transferable
lesson in the project: **every number in this submission is a number the
harness was proven to be able to produce.** The oracle exists precisely so
that "the model failed" and "the harness is broken" cannot be confused. We
found three harness defects that would have been reported as model failures
(an 80-line cap on a 92-line gold edit; an ambiguous SEARCH anchor; a context
window that hid the answer). If you build an agentic coding benchmark without
that self-check, you do not have a benchmark — you have an anecdote.

---

## 6. Reproducing this

### Environment and credentials

```bash
git clone <this-repo> verified-repo-distillation && cd verified-repo-distillation
python3 -m venv .venv && .venv/bin/pip install pytest
```

The harness itself is **standard library only**, on purpose: a tool that claims
to reconstruct environments must not depend on a fragile environment itself.
Externally it needs `uv` (Python repos), a Go toolchain (Go repos), and `git`.
Keys are read from an env **file path** at runtime — never written into the
repo, never hardcoded, never logged:

```bash
export OPENROUTER_API_KEY=...     # default provider
export NEBIUS_API_KEY=...         # Nebius Token Factory
export VRD_ENV_FILE=/path/to/env  # optional: point at a different file
```

### The full pipeline

```bash
REPO=~/projects/hermes-webui
DATA=data/hermes-webui-all
SPLIT=2026-05-19T01:38:13Z

python run.py ingest --repo $REPO --data $DATA                    # 1. classify
python run.py gate   --repo $REPO --data $DATA                    # 2+3. gate (~9.2 s/commit)
python run.py gate   --repo $REPO --data $DATA --retry $DATA/verdicts.jsonl  # retry replay failures only
python run.py pairs  --repo $REPO --data $DATA --out $DATA/dataset # 4. build corpus
python run.py oracle --repo $REPO --data $DATA/dataset --time-split $SPLIT  # self-check FIRST
python run.py train  --data $DATA/dataset --out $DATA/export --time-split $SPLIT  # 5. export
python run.py eval   --repo $REPO --data $DATA/dataset --arms A \
  --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --time-split $SPLIT   # 6. evaluate
python run.py report --data $DATA --out dashboard/report.json      # dashboard
open dashboard/index.html
```

### Tuning environment variables

| variable | default | effect |
|---|---|---|
| `VRD_EDIT_FORMAT` | `diff` | `diff` = SEARCH/REPLACE blocks; `whole` = return entire files (small repos only) |
| `VRD_MAX_CONTEXT` | `48000` | context budget in bytes (the windowed source context) |
| `VRD_MAX_FILE` | `16000` | per-file byte threshold below which a file is shown whole |
| `VRD_MAX_TOKENS` | `32000` | max output tokens |
| `VRD_MAX_OUTPUT_CHARS` | `40000` | hard cap on model output length |
| `VRD_TEMPERATURE` / `VRD_TOP_P` | `1.0` / `0.95` | NVIDIA's own agentic-coding recipe for Nemotron 3 Super |
| `VRD_REASONING` | `none` | `none` / `budget` / effort level; off by default because reasoning tokens count against `max_tokens` |
| `VRD_PROVIDER` | `openrouter` | `openrouter` or `nebius` |
| `VRD_BASE_URL` | — | point at a dedicated endpoint's regional data plane |
| `VRD_DRY_RUN` | — | `1` exercises the harness end to end without spending tokens |
| `VRD_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | default model |

### Fine-tuning, and serving the tuned arm

```bash
python scripts/ft_submit.py       # -> ftjob-8c6c6ffcf7734e97b1663d500b64a72f
python scripts/fetch_adapter.py   # -> 38.2 MB PEFT adapter
# Nebius deploy path (needs a model-artifact_ id from the beta Custom Weights Hub):
bash scripts/tuned_endpoint.sh model-artifact_XXXXXXXX

# What we actually did — Nebius rejects a checkpoint id in custom_weights_id (422),
# so the tuned arm was served on rented hardware:
python scripts/vast_create.py     # RTX 6000 Ada 48 GB, $0.502/hour, vastai/vllm:v0.32.0-cuda-13.0
#   on the instance:
#   vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
#     --enable-lora --lora-modules vrd-adapter=<path-to-adapter> --port 3000
VRD_PROVIDER=nebius VRD_BASE_URL="http://<host>:3000/v1" \
python run.py eval --repo $REPO --data $DATA/dataset --arms A \
  --model "<served-model-name>" --time-split $SPLIT
```

`scripts/batch2_benchmark.sh` reproduces the three-model head-to-head;
`scripts/contamination_probe.py` the memorisation check; `scripts/deploy_tuned.py`
and `scripts/delete_endpoint.py` the Nebius deploy/teardown.

```bash
pytest tests/ -q      # 48 passed
```

---

## 7. Judge walkthrough — 60 seconds

**0:00–0:10. Open the dashboard.** `dashboard/index.html`, a single file with
no dependencies. It renders two things: the **ingest funnel** (3 526 commits →
1 813 kept → 565 gate candidates, with the drop reasons as bars: 1 127 merges,
316 docs-only, 121 tests-only, 89 bookkeeping) and the **gate verdict
breakdown** (525 verified / 14 pre-passed / 25 post-failed / 1 replay). The
point to make out loud: *525 verified is not "we kept most of them"; it is "we
proved the change was necessary and sufficient by running the tests".*

**0:10–0:20. The gate, in one sentence.** For each candidate commit: check out
the parent, inject only the commit's new tests, they must **fail**; apply the
commit's full patch, they must **pass**. That is `counterfactual_verify()` in
`vrd/replay.py`, 9.2 s per commit, no GPU.

**0:20–0:30. Why it matters.** The 14 `reject_pre_passed` commits are the ones
whose test was already green before the "fix" — the commit's test did not test
the commit. Every fetch-the-diff pipeline ingests those as ground truth. This
is the contribution, and it is a mechanism you can point at in the source.

**0:30–0:40. Prove the instrument.** Two commands, both in the repo:
`python run.py oracle ...` replays each pair's own gold answer through the
identical scorer — **26/26 = 100%** — and
`python scripts/contamination_probe.py` asks the model to reproduce held-out
post-commit code verbatim — **0/31 lines matched**, both models answered
`UNKNOWN`. Without the oracle, "the model failed" and "our harness is broken"
are indistinguishable, and we found three real harness defects before quoting
a single model number.

**0:40–0:52. The result, stated flatly.** Four arms, 26 held-out commits,
identical prompts, execution-scored, Wilson intervals:

| model | applied | solved | pass rate (Wilson 95%) |
|---|---|---|---|
| Nemotron 3 Ultra 550B | 21/26 | **17/26** | 65.4% [46.2, 80.6] |
| Qwen3-30B-A3B base | 22/26 | **3/26** | 11.5% [4.0, 29.0] |
| Qwen3-30B-A3B + LoRA | 10/26 | **3/26** | 11.5% [4.0, 29.0] |

The line to say: *the tuned model solved exactly the same three commits as the
base model — no more, no fewer — and its patches stopped applying, 22/26 down
to 10/26.* Then the diagnosis, because it is the useful part: pass-per-applied
went **up** (14% → 30%), so the adapter did not get dumber about the code; it
got worse at emitting a uniquely-matching anchor. The bottleneck is the output
contract and the absence of a repair loop, not repository knowledge.

**0:52–1:00. Close, and be the one who says it first.** *"We set out to show
that tuning on verified repo history helps. It did not. What we are taking
away is that the corpus builder works and is reusable, the measurement is
trustworthy enough to detect a null result, and 148 examples is far too few to
teach an anchor discipline the base model already had. The next experiment is
a repair loop and a bigger corpus, not more tuning."*

**Files a judge can open, in priority order:**

1. `vrd/replay.py` — `counterfactual_verify()`, the gate (~60 lines, the core).
2. `run.py oracle` — the self-check that makes every other number mean something.
3. `vrd/evaluate.py` — `score_patch()`, execution-only scoring; `wilson_interval()`.
4. `data/hermes-webui-all/dataset/eval_scores_qwen_base.jsonl` and
   `eval_scores_qwen_tuned.jsonl` — the per-pair records behind the table, so
   the "same three commits" claim can be checked directly.
5. `SUBMISSION.md` §4 — what we do not claim.

---

## Appendix: artifact index

| artifact | path |
|---|---|
| CLI | `run.py` (`ingest`, `gate`, `pairs`, `oracle`, `train`, `eval`, `report`) |
| Stage 1 | `vrd/ingest.py` |
| Stages 2–3 (the gate) | `vrd/replay.py` |
| Stage 4 | `vrd/pairs.py` |
| Stage 5 | `vrd/train.py` |
| Stage 6 | `vrd/evaluate.py` |
| Edit contract | `vrd/edit_format.py` |
| Tests | `tests/test_harness.py`, `tests/test_edit_format.py` (48 passing) |
| Gate verdicts | `data/hermes-webui-all/verdicts.jsonl` (565 records) |
| Corpus | `data/hermes-webui-all/dataset/pairs.jsonl` (182 records) + `dataset_card.json` |
| Training export | `data/hermes-webui-all/export/train.jsonl` (148 records) + `job_spec.json` |
| Per-arm scores | `data/hermes-webui-all/dataset/eval_scores_{qwen_base,qwen_tuned,super,ultra}.jsonl` |
| Adapter | `data/hermes-webui-all/lora/ftckpt_1b6c9928-…/` (38.2 MB) |
| Fine-tune job record | `data/hermes-webui-all/ft/job_state.json` |
| Dashboard | `dashboard/index.html` |
| Project report | `REPORT.md` |
| Validation record | `VALIDATION.md` |

License: Apache 2.0.
