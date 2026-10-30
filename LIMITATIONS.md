# Limitations

Date: 2026-10-30 · Nebius × NVIDIA Hackathon, Coding & Agentic Engineering track
Scope: the 26-commit held-out benchmark in `REPORT.md`, and the claim that
LoRA-tuning on 148 verified pairs does not help.

This is the adversarial reading of our own results, written for a reviewer who
wants to reject the work. Every number is recomputed from the run artifacts in
`data/hermes-webui-all/` (`dataset/eval_scores_*.jsonl`, `dataset/pairs.jsonl`,
`export/{train,eval}.jsonl`) rather than restated from `REPORT.md` prose.

---

## 0. Summary: what we can and cannot claim

| Claim | Status |
|---|---|
| The gate produces test-verified training pairs | **Supported.** 26/26 gold answers score as solved through the identical scoring path. |
| Held-out commits are not memorised | **Weakly supported.** 5 probe cases, 2 on a file type later excluded from the corpus (§2.2). |
| Nemotron 3 Ultra 550B > Qwen3-30B-A3B here | **Supported.** 17/26 vs 3/26, Wilson CIs disjoint. |
| LoRA on 148 verified pairs adds no solved tasks | **Not established.** The test had ~0 power to detect a real effect (§3). |
| LoRA on 148 pairs *hurts* patch application | **Supported as an observation** (84.6% → 38.5%), mechanism unverified (§3.4). |
| This generalises beyond `hermes-webui` / Python | **No evidence** (§2). |
| Our 3/26 for Super contradicts NVIDIA's 59–60% SWE-bench | **False — and we must never say it** (§5). |

---

## 1. Statistical power: the single biggest weakness

### 1.1 The intervals we are working with

n = 26. Wilson score intervals, 95%:

| arm | solves | rate | Wilson 95% CI |
|---|---|---|---|
| Qwen3-30B-A3B base | 3/26 | 11.5% | **[4.0%, 29.0%]** |
| Qwen3-30B-A3B + LoRA | 3/26 | 11.5% | **[4.0%, 29.0%]** |
| Nemotron 3 Super 120B | 3/26 | 11.5% | **[4.0%, 29.0%]** |
| Nemotron 3 Ultra 550B | 17/26 | 65.4% | **[46.2%, 80.6%]** |

The interval on the 11.5% result spans a **7.2-fold range in true rate**
(4.0% to 29.0%). "The small model scores 11.5%" is a statement about three
successes; the point estimate is nearly uninformative on its own.

### 1.2 What we COULD detect

- **A large capability gap.** Ultra's 17/26 vs base's 3/26 is the one clean
  result: Ultra's Wilson lower bound (46.2%) is above the other arms' upper
  bound (29.0%), the intervals do not overlap, and a Fisher exact test on the
  2×2 table is far below any conventional threshold. On this benchmark a 550B
  model materially outperformed a 30B MoE.
- **A large degradation in patch-application reliability.** 22/26 → 10/26 on a
  paired binary outcome over the same 26 commits; McNemar's exact test on the
  discordant pairs gives p ≈ 0.002. The strongest tuning-related claim we can
  make.

### 1.3 What we COULD NOT detect

Everything in between. Against a true rate of 11.5%, a
Wilson-lower-bound-above-29.0% rule needs **13/26 = 50.0%** true rate to
trigger. Simulated power (4,000 trials, α ≈ 0.05 via that rule):

| true solve rate | 15% | 20% | 25% | 30% | 35% | 40% | 50% | 65% |
|---|---|---|---|---|---|---|---|---|
| power to clear the 11.5% interval | 0.0% | 0.1% | 0.5% | **2.8%** | 8.8% | 21.3% | 58.3% | 96.0% |

**If tuning had genuinely lifted Qwen from 11.5% to 30%, this experiment
would have detected it 2.8% of the time.** We would have reported "null" for
a real 2.6× improvement. This is why §3 does not conclude "tuning does not
help."

Unpaired (between-arm) n required for **80% power**, two-proportion z-test,
α = 0.05 two-sided, baseline 11.5%: to detect 20% needs **n = 290 per arm**;
25% → 130; 30% → 80; 40% → 40; 50% → 30. We have 26 — short by **3× to 11×**
even for a modest effect — and the corpus cannot currently supply it: only
34 pairs exist past the split.

### 1.4 The paired-design subtlety, which makes it worse

Base and tuned ran on the **same 26 commits**, so the right test is paired and
the right statistic is the count of *discordant* pairs. We have: base solves /
tuned misses = **0**; tuned solves / base misses = **0**.

With zero discordant pairs the exact McNemar p-value is **1.0** — not "high",
exactly 1.0 — and this is **invariant to n**. No sample size fixes zero
discordance, because the discordance count *is* the measurement. The paired
design is more efficient than the unpaired one only if the arms disagree
somewhere; here they agree perfectly: maximum paired efficiency, and a
statistic that cannot reject.

State this plainly: **the "null" is a statement that the two arms produced
identical outcome vectors on 26 items, not a measurement that was powered to
find a difference.** "Tuning added zero solves" is literally true and
analytically empty.

### 1.5 One attempt, one sample, no replication

Every number is one attempt per commit at temperature 1.0 / top_p 0.95, with
no seed replication. The Wilson intervals assume binomial sampling error only
and **ignore model-output variance entirely**. For the arms at 3/26, a
plausible amount of run-to-run variance moves the count by ±1 or ±2 — the
entire size of the effect we are trying to resolve. We never measured this and
cannot assume it is small. Four arms ran over the same 26 commits (104
model-attempts on a shared prompt set) with no multiplicity correction; it
would not change conclusions, since only the Ultra gap survives, but
task-level variance is shared across arms and the nominal intervals
understate uncertainty about *which* commits are hard.

All three arms scoring 3/26 solved **the same three commits** (`d6c1b696…`,
`d20da832…`, `f81d24ec…`), and Ultra solved those three plus 14 more — a
plausible stable easy core. But with n = 26 and 3 solves, the *identity* of
the solved set is not statistically established; a re-run under different
seeds could plausibly solve a different 3 of 26.

---

## 2. Generalisation: one repository, one language, one domain

### 2.1 The corpus is a single repo

`hermes-webui` — 3,526 commits, Mar–May 2026, **Python** app code plus
JavaScript/HTML/CSS front end. Held-out set: 26 commits from that one repo.
**Repos with any data: exactly one.** `coredns` has 3/3 gated commits
(`data/coredns/`) and has never been run at corpus scale, split, trained or
evaluated. It is Go — a second language — but 3 commits proves nothing.

`n = 1` repository. There is no variance estimate across repositories, so no
statement about "how a model does at repo-level editing" can be generalised.
The 26 commits are **not 26 independent draws from a population of commits**;
they are 26 commits from one codebase, by one author group, in one
three-month window, in one product area.

We therefore cannot say anything about **other languages** (Go, Rust,
TypeScript, C++, Java untested); **other repository shapes** (deep call
graphs, build systems, generated code, monorepo layouts); **other commit
styles** (the gate keeps only commits that *added tests*, so a repo whose
interesting commits are refactors would be gated out entirely); or **other
task distributions** (median context 14.2 KB, median patch 4.7 KB — small,
well-localised edits, nothing about large multi-file refactors).

### 2.2 The contamination probe is thinner than the headline suggests

`REPORT.md` says "0/31 lines reproduced". Read the log
(`data/hermes-webui-all/contamination.log`) and probe
(`scripts/contamination_probe.py`) and the actual scope is:

- **5 cases**, not 26. The script caps `cases` at 5 and keeps only files with
  ≥ 120 lines.
- **31 countable lines** across 5 files per model; the log reads
  `matched 0/10 (out 2 lines, 8c)` — the model answered `UNKNOWN`, and the
  counter only accumulates on non-UNKNOWN responses.
- **2 of the 5 cases target `CHANGELOG.md`** (`c26e2541`, `9458df6f`) — a file
  type the pipeline **subsequently removed entirely** as non-source. The
  probe was partly aimed at files the benchmark does not use.
- It covered **Nemotron Nano-30B and Super-120B only** — *not* Qwen base,
  *not* the tuned arm. The two arms whose scores we most need to defend were
  never probed.
- The probe **allows `UNKNOWN` by design**; `REPORT.md` §9 still lists
  "forced continuation rather than allowing UNKNOWN" as to-do. A model that
  answers `UNKNOWN` scores 0 by construction: absence of evidence, not
  evidence of absence.

**What it establishes:** for 5 held-out files, two Nemotron models did not
emit post-commit code verbatim. Real, but narrow. **What it does not:** that
the 26-commit held-out set is free of contamination for the Qwen arms. A
5-case, 2-model, mostly-CHANGELOG probe is thin ground for "demonstrably
unseen."

### 2.3 The temporal split proves novelty, not transferability

The split at `2026-05-19T01:38:13Z` genuinely holds out later commits, and
`commit_date` was empty on all pairs until it was fixed — meaning the first
version of this experiment silently had **483 train / 0 test**. That the bug
was found and fixed is a point in the project's favour, and also a warning:
the pipeline has produced at least one silent, total failure of the holdout
that a reader of an earlier report would not have seen.

What the split proves: **these commits were not in the training set.** What it
does not: **the finding transfers** — to other repos, languages, domains,
task shapes, or later periods. A temporal holdout is a *novelty* control,
not a *generalisability* control. Treating "the commits are unseen" as
"therefore the finding is general" is the category error this project is most
at risk of making.

### 2.4 The scope filter narrows the held-out set and biases it toward one-file edits

The 34 held-out pairs were cut to 26 in scope by **≤ 2 changed files AND
≤ 6,000 bytes of patch** (threshold read from `run.py:181`, confirmed exactly
against the score files: the rule reproduces the 26/8 split with no
exceptions). Verified: all 26 in-scope pairs satisfy both bounds (max 5,943 B,
median 3,859 B); the 8 excluded pairs are 1–4 files with patches from 3,603 B
to **69,077 B**; and **23 of the 26 in-scope pairs are single-file**.

Two effects, pulling in opposite directions. **The excluded 8 are
systematically larger** — 2 of them fail on file count alone at 3,603 B and
8,107 B — so if capability declines with patch size (near-certain) the
in-scope set is **enriched for easy tasks** and Ultra's 65.4% is measured on
a distribution easier than the repo's real commit distribution. And **the
in-scope set is overwhelmingly single-file**, so the benchmark measures
single-file local edits, not multi-file coordination — one of the harder,
more realistic parts of repo editing; a model that cannot coordinate a 4-file
change is barely tested here.

The scope filter is a deliberate fairness choice (it bounds the edit to
something a human could do without repo-wide context). It is also a
**selection effect on the headline number**, and the two effects are not
separable with this data.

---

## 3. The tuning null is under-powered: arguing both directions

`REPORT.md` currently claims "the answer is negative" and "that is a real
result, not a failure to measure." **That framing is not supported by the
data.** Here is the case in both directions.

### 3.1 The argument that the null is REAL

| metric | base | tuned |
|---|---|---|
| patches apply | 22/26 (84.6%) | 10/26 (38.5%) |
| tests pass | 3/26 | 3/26 |
| pass-per-apply | 14% | 30% |
| total generated bytes (26 prompts) | 53,868 | 142,336 |

The adapter learned *something*: the patches it does emit are **more often
correct** (14% → 30% pass-per-apply, 2.1× conditional accuracy), and it got
**2.6× more verbose** (53.9 KB → 142.3 KB). That pattern — better precision,
much worse recall, much more output — is the signature of a model that has
learned *style and hedging* rather than capability: it writes longer, more
elaborate patches, edits more hunks than it should, and every extra hunk is
another chance to miss an anchor.

Supporting facts: **the oracle passed 26/26**, so the harness is not what
broke; **the tuned arm solved only commits the base arm also solved**, so if
tuning were working some of the 23 unsolved commits should have become
solvable; and **the training data was legitimately verified** — each of the
148 pairs had its own tests fail before the patch and pass after, so there
was no obvious label noise to fail to learn from.

### 3.2 The argument that the experiment CANNOT SEE a real effect

**The test could not have detected a meaningful gain.** Per §1.3, a true lift
from 11.5% to 30% would have been missed 97.2% of the time, and the paired
McNemar statistic is *identically* p = 1.0 when the arms agree on every
commit. **A larger n on the same corpus would not have helped unless the
tuned arm actually diverged on some commit** — and a modest real effect
would plausibly produce zero divergence at n = 26.

**The training run was under-resourced by any standard.**

| parameter | value | assessment |
|---|---|---|
| training examples | 148 | small |
| epochs | **1** | the SFT norm is 2–4; 1 is the minimum, not the default |
| LoRA rank | 16 | low for 30B; r = 32–64 is common |
| LoRA targets | attention projections only | **no MLP, no embeddings, no lm_head** — for an MoE model this excludes most of the parameterised compute |
| trained tokens | 1,144,590 | ~1.1M tokens is at the very bottom of useful SFT scale |

A 1.1M-token, one-epoch, attention-only, r=16 LoRA on a 30B MoE is not a
serious attempt to teach a new capability; it is closer to *format
adaptation*. The observed behaviour (format and verbosity drift, anchor
reliability collapse, no capability gain) is exactly what that predicts. The
absence of a capability gain is **predicted by the training configuration**,
not informative about the corpus.

**The training distribution does not match the evaluation contract.**
Training used 32,768-token context with packing; evaluation prompts are
median 21,220 chars (~5.3K tokens), max 51,343 chars (~12.8K tokens). The
tuned arm's output grew to 142,336 bytes across 26 prompts — average 5,474
bytes per answer against the base model's 2,072. Under packed 32K training the
model learns to fill its context budget. **The adapter may simply have learned
to be verbose because that is what packing rewards**, and verbosity is fatal
in a position-free SEARCH/REPLACE contract where every extra hunk is another
chance to miss an anchor.

**The corpus may not contain the capability being sought.** The 148 pairs are
the *same kind of task* as the 26 held-out pairs: same repo, same files, same
author, same three months. The failure mode is *anchor matching under a
position-free contract* — a precise-text-reproduction skill. If the corpus
does not teach that skill, it will not be learned. And the corpus was filtered
for *answerability and expressibility*, not *informativeness* (§6): 148
easy-to-express local edits may teach format and nothing else.

### 3.3 Where this leaves the claim

**Defensible:** *"At 148 examples, one epoch, LoRA r=16 on attention
projections only, tuned on this repo's own verified commits, the adapter
solved no additional held-out commit (0 discordant pairs), and it degraded
patch-application reliability from 84.6% to 38.5%."*

**Indefensible:** *"Fine-tuning on verified repo history does not help"* /
*"the answer is negative"* / *"that is a real result, not a failure to
measure."* Those are claims about the hypothesis; the experiment tests one
small configuration of it, with ~0 power to see a moderate effect.

The honest conclusion is two-layered, and `REPORT.md` collapses it into one:

1. **We have evidence of harm and no evidence of benefit.** The
   patch-application regression is real (McNemar p ≈ 0.002); the benefit
   signal is absent.
2. **We have no evidence about benefit because we could not have detected
   it.** The benefit half of the result is uninformative.

Note (1) does not license (2): a configuration that damages contract
discipline is not evidence that a better configuration would not help. The
most likely reading of our data is "this LoRA configuration, on 1.1M tokens,
broke formatting without buying capability" — a statement about LoRA
hyperparameters, not about verified-repo distillation.

### 3.4 The "degradation" needs a mechanism check before it is reported as a tuning effect

Two alternatives to "tuning damaged the model" have not been excluded:

- **Serving confound.** The tuned arm was served from a rented Vast.ai
  RTX 6000 Ada with vLLM 0.32.0 and `--enable-lora`; the base arm ran on
  Nebius's hosted endpoint. Different stacks, quantisation, kernels and
  sampling defaults. The base model was
  `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (an FP8 build), while the tuned
  adapter merged from the non-FP8 `Qwen3-30B-A3B-Instruct-2507` and was
  served in a different runtime. **A drop from 84.6% to 38.5% is exactly the
  magnitude of thing a serving difference can cause.** The right control is
  the base model served through the identical Vast.ai + vLLM path. Without
  it, the regression is confounded with the infrastructure change.
- **Prompt/template drift.** If the adapter shifted the model into emitting
  extra `### FILE:` headers or duplicated hunks, the anchor failures would be
  a serving artefact. The notes support this reading: `api/routes.py` accounts
  for 3 of the tuned arm's 16 anchor failures and `api/streaming.py` for 2
  more — clustered in a few files, consistent with whole-file-shaped responses
  for large files.

---

## 4. The scoring gives the model the spec: we measure spec-following, not bug-finding

Verified directly from the 34 eval prompts (`export/eval.jsonl`): every one
contains a section headed `## Tests that must pass`, reading *"These are the
tests the change has to make pass. They are the specification: do not modify
them, and do not assume any other behaviour than what they assert."*, followed
by the full source of the commit's own newly-added test file including the
`assert` statements — **34/34 prompts contain `assert` and `def test_`**. The
prompt structure is `## Request` (the commit subject), `## Files`,
`## Current contents`, `## Tests that must pass`, `## Output format`.

This is **generous, and SWE-bench-style**, in the sense that the behavioural
specification of the required change is handed to the model verbatim. It is a
fair and valuable measurement — but of a specific skill:

> **"Given a codebase, a one-line intent, and the exact test assertions, emit a
> patch that satisfies those assertions."**

Consequences for interpretation:

- **We are not measuring bug-finding.** The model is never asked to identify
  what is wrong; the commit's own test file names the behaviour, the
  functions and the expected values. A model that cannot locate a bug but
  *can* satisfy a written assertion is fully credited.
- **We are not measuring intent inference.** `## Request` is the commit
  subject — `feat: add distinct attention request sound` — often an imprecise
  summary of a real multi-part change. The model can largely ignore it and
  work from the tests.
- **The metric is closer to "specification satisfaction" than to "software
  engineering."** A 65.4% score on "satisfy these assertions" is not a 65.4%
  SWE-bench score and must never be presented adjacent to one.
- **The effect on the tuning hypothesis is asymmetric.** The training pairs
  were built the same way (148/148 train prompts also carry
  `## Tests that must pass`, verified in `export/train.jsonl`), so tuning was
  aligned with the task — good. But the corpus therefore teaches
  *spec-to-patch* translation, the narrow skill the model already had, not
  general repo editing.

**The spec leaks the answer's shape.** The test file is the *post-commit*
test; its assertions constrain the patch tightly (which function, which
attribute, which value). This makes the task easier than real bug-fixing and
systematically favours models good at *assertion following* — a plausible
reason a 550B model does well and a 30B model does not. It licenses "Ultra
550B is good at this specification-following task, which is assertion-shaped";
it does not license "Ultra 550B is good at software engineering."

---

## 5. One-shot only: our Super 3/26 is not comparable to NVIDIA's 59–60%

NVIDIA's published SWE-bench-verified figures for Nemotron 3 Super and Ultra
are **59–60%**, from **multi-turn agentic harnesses**: the model searches the
repo, reads files, runs the tests, sees failures, and iterates. Our evaluation
is **strictly one-shot** — one attempt per commit, no repair turn, no
test-error feedback, no repo-wide search, no re-read after a failed anchor, no
chance to correct a near-miss.

`VALIDATION.md` §8 already says this, correctly: *"NVIDIA's 59–60% SWE-bench
figures come from multi-turn agentic harnesses. This eval is one-shot, which
is a narrower and harder task, so absolute rates are not comparable to
theirs."*

Given that our Super number looks catastrophic next to a 550B number, it is
very tempting to write *"our results contradict NVIDIA's published SWE-bench
numbers."* **That sentence would be wrong, and it would be the single most
damaging claim in the project.** It is wrong because of five confounders, in
order of importance:

1. **Different task sets.** SWE-bench-verified uses real GitHub issues
   (django, sympy, flask, scikit-learn, astropy, sphinx, pytest, matplotlib,
   requests, seaborn, xarray) with a fixed ~2,294-instance split. Ours is 26
   commits from one Python web application. Absolute difficulty is not
   comparable in either direction.
2. **Different harness.** Multi-turn agentic vs one-shot emission. This is not
   a detail — it is most of the benchmark. An agentic harness that sees a
   failing test can recover from a bad anchor that is fatal to us. Repair
   turns in this literature commonly add tens of points of absolute rate.
3. **Different measurement.** SWE-bench scores FAIL_TO_PASS **and**
   PASS_TO_PASS; we score only the commit's own newly-added tests. Ours is
   *narrower* (fewer tests to satisfy) but the context is also narrower.
4. **Different model configurations.** NVIDIA's figures use their own serving
   stack, agent scaffolding, prompt templates and retry policy. We use
   temperature 1.0 / top_p 0.95 / reasoning off, per their own cookbook.
5. **We are not running SWE-bench.** This is a self-built holdout in a similar
   genre, not an SWE-bench instance.

What we may legitimately say about Super's 3/26:

- *"On our one-shot, 26-commit, single-repository, assertion-specified
  benchmark, Nemotron 3 Super 120B solved 3/26 (11.5%, Wilson 95% CI
  [4.0%, 29.0%]), applying 9/26 patches (34.6%)."*
- *"This is not comparable to NVIDIA's published multi-turn agentic SWE-bench
  figures, and this benchmark does not contradict them."*
- *"Super applied fewer patches than our Qwen base (9 vs 22) while solving the
  same 3 commits; 17 of its 23 failures were SEARCH-anchor or parse failures
  rather than semantically wrong patches."*

We may **not** say *"NVIDIA's Super scores 59% on SWE-bench but only 11.5%
here, so our benchmark is harder"* — the one-shot/multi-turn confound makes
that comparison uninterpretable. We may not say the reverse either.

### 5.1 The failure taxonomy contradicts one of our own claims

The project narrative says every inspected failure is an anchor-matching
failure. That is true of *patch-application* failures but **not** of the
failures we actually scored. From the `note` and `test_counts` fields:

| arm | applies | solves | anchor/parse failures | applied-but-tests-failed |
|---|---|---|---|---|
| Qwen base | 22/26 | 3/26 | 4 | **19** |
| Qwen tuned | 10/26 | 3/26 | 16 | 7 |
| Nemotron Super | 9/26 | 3/26 | 17 | 6 |
| Nemotron Ultra | 21/26 | 17/26 | 4 | 4 |

**For Qwen base, 19 of 23 failures are patches that applied cleanly and then
failed the tests**, with real assertion counts — e.g. `passed: 31, failed:
13`, `passed: 8, failed: 2`, `passed: 1, failed: 5`. Those are **semantic
failures**: a well-formed patch that does not implement the required
behaviour.

- The anchor-matching story describes *why patches fail to apply* — a
  **contract-discipline** problem.
- The 19 applied-but-failed patches for the base model are a **capability**
  problem, and they are the majority of the failures.
- A benchmark where 4 of 26 tasks are blocked by output-format mechanics and
  the rest by semantics measures two different things, and the format failures
  are the cheaper ones to fix (they would largely disappear with a repair
  loop or a whole-file edit contract).

We should present the taxonomy honestly rather than attributing everything to
anchors: a reviewer who opens the score files will count 19
applied-but-failed patches for the base model and conclude the "all failures
are anchor failures" claim was never checked.

---

## 6. Corpus filtering: 3,526 → 565 → 525 → 460 → 353 → 182 → 26

| stage | count | what was dropped | source |
|---|---|---|---|
| commits ingested | 3,526 | — | `commits.jsonl` |
| gate candidates | 565 | — | `candidates.jsonl`, `verdicts.jsonl` |
| **gate verified** | **525** | 14 `reject_pre_passed`, 25 `reject_post_failed`, 1 `reject_replay` | `verdicts.jsonl` |
| after expressibility filter | **460** | 23/483 unplayable in position-free SEARCH/REPLACE | commit `63a70f6` |
| after context-completeness filter | **353** | 107/483 with SEARCH text not visible in prompt | commit `71862a5` |
| after code-only filter | **182** | docs/changelogs treated as source | commit `866477d` |
| temporal split at p80 | **148 train / 34 held out** | — | `export/{train,eval}.jsonl` |
| scope filter (≤2 files, ≤6,000 B) | **26 evaluated** | 8 held-out pairs | `dataset/eval_scores_*.jsonl` |

Against 525 verified commits the final held-out set is **26 — 5.0%**. Against
3,526 commits, **0.74%**. Every drop was a *good* decision on its own terms;
the point is that they are **not independent** and **not random**. Each
filter correlates with difficulty, consistently in one direction
(*Good* = why the filter is right on its own terms; *Bias* = what it costs):

| filter | n | Good | Bias |
|---|---|---|---|
| `reject_pre_passed` | 14 | tests were already green — these prove nothing | removes weak-signal, typically **small** changes |
| `reject_post_failed` | 25 | never went green here — harness flakiness | removes environment/timing-dependent, typically **integration-style** changes |
| `not_expressible` | 23 | gold answer itself could not be replayed | removes **repeated/boilerplate** code (route tables with similar branches, config blocks, i18n maps) — where much of the work is |
| `context_incomplete` | 107 | answer text was not visible — unanswerable pairs measure nothing | removes **large-file** and **widely-separated-hunk** commits; this repo has a median file size of 521 KB (max 1.4 MB) |
| code-only | 353→182 | docs are not code | removes routine, small, highly templated changelog commits, making the corpus look harder than the repo's real distribution |
| scope | 34→26 | bounds the edit to something a human could do | removes the largest changes; the 8 dropped pairs run 3.6 KB to **69 KB** against a 5.9 KB max for the 26 kept |

**Quantifying the bias.** We cannot compute a corrected rate, because the
models were never run on the excluded pairs. We can bound direction and rough
magnitude: the 8 out-of-scope held-out pairs are, on every axis we can
measure, **harder** than the 26 in-scope ones, so Ultra's 65.4% is measured
on the **easier tail**; `context_incomplete` removed 107/483 = **22.2%** of
pairs concentrated in large files, so the corpus is biased toward
**small-file edits** — precisely the regime where a 14.2 KB context window
suffices, making the benchmark's difficulty profile an artefact of the context
budget rather than of the repo; `not_expressible` removed 23/483 = **4.8%**,
concentrated in repetitive code. Aggregate survivorship: **182/483 = 37.7%**
of pairs reached the final corpus, **26/483 = 5.4%** reached final
evaluation. Any claim of the form "X% of commits are solvable" is a claim
about **the surviving subset**, not about the repo.

**The concrete danger:** the two filters that bite hardest
(`context_incomplete`, `not_expressible`) are precisely the ones that make
the task *winnable* for a model lacking repo-wide search. They remove the
tasks that would most reward an agentic model. Meanwhile what the benchmark
measures well — "can you satisfy these assertions" — is measured where the
assertions and the anchor text are both fully visible. This may explain why
the 550B model does well and the 120B model does not, without any of it
saying anything about general software-engineering ability.

**The filters also bias what the fine-tune learns.** A corpus filtered for
*answerability* is a corpus filtered for **easy, well-anchored, self-contained
edits**, teaching "make a small, uniquely-anchored, local edit that satisfies
the given assertions" — a narrow style. The tuned model's behaviour (longer
output, more hunks, worse anchoring) is consistent with having been trained on
a distribution that excludes the hard cases entirely. **The corpus's selection
bias does not merely affect the headline rate; it shapes what the fine-tune
teaches and therefore biases the tuning conclusion.**

---

## 7. What a reviewer should demand before believing any of this

**Before the tuning conclusion is accepted:**

1. **Power the experiment or stop claiming a null.** Either report the tuning
   result explicitly as *under-powered and inconclusive*, with the §1.3 power
   table attached, or run enough held-out pairs to detect the effect sizes you
   care about (11.5% → 30% needs **n ≈ 80 per arm**; → 25% needs **n ≈ 130**;
   we have 26). The corpus cannot supply this from `hermes-webui` alone.
2. **Run a positive control on the fine-tune.** The adapter demonstrably
   changed behaviour (2.6× output, anchor reliability halved). Prove the
   training pipeline can teach *anything* at this scale: e.g. measure
   held-out-by-epoch performance on the 148 pairs, or train a synthetic
   format task (given input and gold patch, emit the patch) where a working
   fine-tune should approach 100%. If the adapter cannot learn a trivially
   learnable task from 148 examples in 1 epoch, the null says nothing about
   the corpus and only about the recipe.
3. **Remove the serving confound.** Serve the **base** model through the
   identical Vast.ai + vLLM 0.32.0 + `--enable-lora` path used for the tuned
   arm. If base-under-Vast also drops to ~10/26 applied, the "regression" is
   infrastructure. This is cheap (26 calls) and is the single most important
   missing control.
4. **Sweep the LoRA configuration.** At minimum epochs 1→3, rank 16→64,
   targets attention-only→attention+MLP. A null at one hyperparameter setting
   is not a null about the hypothesis. Report the best configuration found,
   not the first.
5. **Match the training distribution to the eval contract.** Training packs to
   32K; eval prompts are ~5K tokens. Retrain at the eval length distribution
   or report a length-matched ablation. The 2.6× output growth suggests the
   adapter learned to fill a packed context.

**Before the benchmark is treated as a general result:**

6. **A second repository at corpus scale, in a second language.** `coredns`
   is wired and gated (3/3 verified). Run the full pipeline on it — corpus,
   split, train, eval — and report it as a separate replication with its own
   CI. One repo gives n = 1; two repos in two languages give a variance
   estimate and a hint of transfer.
7. **A forced-continuation contamination probe across all arms.** Extend to
   all 26 held-out commits and all 4 arms (Qwen base, Qwen tuned, Super,
   Ultra), with the model *forced* to continue rather than allowed to answer
   `UNKNOWN`. Report per-arm and aggregate. The current 5-case, 2-model,
   2-of-5-on-CHANGELOG probe cannot support the "demonstrably unseen" phrasing
   in `REPORT.md` §10.
8. **Report the full funnel as a results table**, not a pipeline description:
   every stage (565 → 525 → 460 → 353 → 182 → 148/34 → 26) with count,
   reason, and a measured difficulty proxy (patch bytes, file count, file
   size). A reviewer should be able to see exactly which tasks were removed
   and infer the bias.
9. **Publish the excluded 8 held-out pairs as a second, harder tier.** We
   already have them. Scoring 26 easy + 8 hard (n = 34) is nearly free and
   gives a difficulty-conditional result: *does Ultra's advantage survive on
   large patches?* More informative than 17/26.

**Before the failure taxonomy can be trusted:**

10. **Publish the full failure breakdown, including applied-but-tests-failed.**
    19 of 23 base-arm failures are applied-but-failed, not anchor failures.
    The claim "every inspected failure is an anchor-matching failure" is not
    supported by the score files and must be corrected wherever it appears.
11. **A repair loop (2–3 attempts with test-failure feedback).** One-shot is
    the harshest possible scoring and `REPORT.md` §9 already identifies this
    as likely worth more than any tuning improvement. It would also convert
    the anchor failures — 4 to 17 per arm — into recoveries, and make the
    benchmark far more comparable to published agentic numbers.
12. **Report pass-per-apply with its own interval**, not just the raw
    fraction. 14% (3/22) vs 30% (3/10) is suggestive but tiny-sample;
    conditional-on-applying rates are post-hoc and subject to collider bias
    (arms that fail to apply are filtered out of the denominator).

**For reproducibility:**

13. **Publish seeds, exact model revisions and serving configs.** Which
    `Qwen3-30B-A3B-Instruct-2507-FP8` revision? Which vLLM flags on the Vast
    instance? What temperature/top_p per arm? A one-shot benchmark at
    temperature 1.0 with no seed control is not reproducible to the precision
    of the intervals we quote.
14. **Publish the per-commit score files as the primary artifact.** They
    already exist — `dataset/eval_scores_*.jsonl` carries `applies`,
    `success`, `tests_pass`, `note` and `test_counts` per SHA — and they are
    what exposed the applied-but-failed issue in §5.1.
15. **Update `VALIDATION.md`, which is stale.** It records the oracle at
    43/43 (pre-code-only) while the corpus is now 182 pairs and the current
    oracle is 26/26, and its §8 still says *"Arm C has never run"* — false,
    since the tuned arm has been evaluated. A reviewer who reads it will
    catch the contradiction.

---

## 8. Things that are *not* limitations, stated so they are not re-litigated

- **"The harness might be broken."** The oracle scores the gold answer as
  solved through the identical parse→apply→run-tests→score path, 26/26. Two
  real defects were found and fixed this way (an 80-line block cap that
  rejected a legitimate 92-line edit; ambiguous SEARCH text), both *before*
  any number was quoted.
- **"The held-out set might be contaminated."** The temporal split is real
  (`2026-05-19T01:38:13Z`, UTC-normalised, because the repo's commits span
  `-07:00` to `+08:00` and lexicographic comparison misplaced them by up to
  15 hours), and 0/31 probe lines were reproduced. Narrow (§2.2), but not
  absent — and the split bug (483 train / 0 test) was found and fixed rather
  than shipped.
- **"The scoring is subjective."** It is execution-only: patch applies, the
  commit's own tests pass. No LLM judge, no similarity heuristic.
- **"The gold patches might be ambiguous or unplayable."** 24 such commits
  were removed at corpus-build time rather than scored as model failures,
  precisely so every arm would not be biased down by the same amount.

---

## 9. Bottom line

The **pipeline** is the strongest part of this project and is well validated:
a two-sided gate that rejects the false pairs every naive pipeline trains on,
an execution-only scorer, an oracle that proves the harness, a real temporal
split, a genuine (if narrow) contamination probe. The **Ultra-vs-Qwen gap**
(17/26 vs 3/26, disjoint Wilson intervals) is a real result on this benchmark.

The **measurement claims are much weaker than the pipeline deserves**, for
reasons structural rather than fixable by more effort at this scale:

- n = 26 gives a Wilson interval of [4.0%, 29.0%] on the 11.5% result and
  power of 2.8% against a genuine lift to 30%. **We could not have detected
  a real effect of the size that would have mattered.**
- n = 1 repository, one language, one product domain, three months, one
  author group. **No transfer claim is supportable.**
- The tuning result is a **null from a configuration too small to see an
  effect** (148 examples, 1 epoch, r = 16, attention-only, 1.1M tokens, with
  a 32K-packed training distribution that does not match a ~5K-token eval
  prompt), **confounded with a serving-stack change** never controlled for.
  The honest statement is "this configuration did not help and damaged
  contract discipline," not "fine-tuning on verified history does not help."
- The benchmark measures **specification satisfaction with the specification
  supplied** — a legitimate, precisely-claimable skill, but not the skill most
  readers will assume it is.
- One-shot scoring against NVIDIA's multi-turn agentic figures is **not a
  contradiction**, and presenting it as one would be the most damaging claim
  in the project.
- Corpus survivorship from 3,526 commits to 26 evaluated tasks (**0.74%**) is
  concentrated in filters that systematically remove large-file, multi-hunk
  and repetitive-code edits. The surviving set is easier than the repo's real
  commit distribution — and it is the set that shaped what the fine-tune
  learned.

**Recommended posture:** present the pipeline and the Ultra-vs-base gap as the
contribution; present the tuning result explicitly as *inconclusive and
under-powered* with the power analysis attached; present Super's 3/26 as a
one-shot single-repo result with no claim against NVIDIA's published agentic
numbers; and fix the two factual errors a reviewer will find immediately —
the "every failure is an anchor failure" claim (§5.1) and the stale
`VALIDATION.md` §8 "Arm C has never run" (§7, item 15).
