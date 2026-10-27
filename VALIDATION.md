# Pipeline validation

Every number this project reports rests on the harness being correct. This file
records what was actually verified, how, and what each check rules out. Numbers
here are from real runs; commands are reproducible.

Machine: 3.7 GB RAM, Intel Atom iGPU, no GPU. Python 3.11 venv at `.venv`.
Corpus: `hermes-webui` (3526 commits, Mar–May 2026).

---

## 1. The gate is two-sided

A commit is `verified` only if **both** hold:

- **pre-arm**: the commit's own newly added tests, injected into the *unpatched*
  parent, **fail**. This is what makes the pair honest — if the test passes
  before the fix, the patch proves nothing.
- **post-arm**: the commit's patch applied to the parent, the same tests
  **pass**.

Result on 565 gate candidates:

| verdict | count | meaning |
|---|---|---|
| `verified` | 525 | 92.9% — patch proven necessary *and* sufficient |
| `reject_pre_passed` | 14 | test was green before the fix; discarded as a false pair |
| `reject_post_failed` | 25 | never green in this environment |
| `reject_replay` | 1 | harness failure (retried; 47 of 50 retried recovered) |

Mean cost 9.2 s/commit. The `reject_pre_passed` bucket is the point of the
design: those 14 commits are exactly what a fetch-the-diff pipeline would have
trained on.

## 2. Oracle test — the harness scores gold answers as correct

`run.py oracle` feeds each pair's **own verified patch**, rendered in the same
SEARCH/REPLACE contract the model must speak, through the identical scoring path
the model output goes through (parse → apply → run the commit's own tests →
score).

If a gold answer fails, the harness is wrong and every arm's number is
meaningless.

**Result: 43/43 = 100%** on all in-scope held-out pairs.

This test found two real defects that would otherwise have been reported as
model failures:

- **80-line block cap.** A real commit had a 92-line edit block; the cap scored
  it as a failure. Caps raised to 300 lines/block and 1200 total — they exist to
  catch whole-file rewrites, not honest edits.
- **Ambiguous SEARCH text.** A gold hunk's search text occurred twice in
  `api/streaming.py`. Git hunks are position-anchored; SEARCH/REPLACE is not, so
  that patch cannot be replayed at all. Such commits are now dropped at corpus
  build time (23 of 483, 4.8%) instead of being scored as failures — otherwise
  every arm would be biased down by the same amount.

## 3. Corpus integrity

| property | value | why it matters |
|---|---|---|
| pairs | 460 | survivors of gate + expressibility filters |
| ≤2 source files | 235 | scope of the eval; larger commits measure model size |
| median context | 16.9 KB | fits comfortably in any context window |
| median patch | 5.9 KB | ~1500 tokens of answer |
| empty targets | **0** | see below |
| test hunks in targets | 0 | tests are the instrument, never the target |
| binary blobs in targets | 0 | base85 PNG/TTF payloads are not text edits |
| duplicate/unexpressible | 23 dropped | unplayable in the contract |

Three corpus defects were found and fixed **before** any number was quoted:

1. **Empty training targets (176/483, 36%).** `diff_to_edits` required a hunk
   header to *end* with `@@`, but real git headers carry a section label
   (`@@ -12,6 +12,8 @@ import time`). Every hunk with a heading was dropped. A
   third of the corpus would have trained the model to answer with nothing.
2. **Unparseable JSONL.** `json.dumps(ensure_ascii=False)` wrote a raw U+0085
   (NEL) from a source file; `str.splitlines()` treats NEL as a line break, so
   readers saw a truncated record. Writers now escape non-ASCII, readers iterate
   the file handle.
3. **Corpus bloat.** Full post-commit file bodies and base85 binary hunks were
   stored in every record: 378 MB for 483 pairs. Now 19 MB.

## 4. Temporal split is real

The claim "held-out commits" requires an actual time boundary.

- `commit_date` had been empty on **all** pairs (read from verdict records that
  do not carry it), so the split silently did nothing: 483 train / 0 test.
- Now read from the commit as `%cI` and **normalised to UTC**. This repo's
  commits span `-07:00` to `+08:00`; comparing those strings lexicographically
  compares local wall-clock time and misplaces commits by up to 15 hours.
- Split at the 80th percentile: **2026-05-19T01:38:13Z** → 367 train /
  93 held out / 43 held out and in scope.

The tuned arm may only see the 367. Every reported arm number is computed on the
43.

## 5. The scorer never asks a model to judge

Scoring is execution only: the patch must apply, and the commit's own tests must
pass. No LLM judge, no similarity heuristic. Confidence intervals are Wilson
intervals on the observed pass rate.

## 6. Model configuration is documented, not guessed

- Sampling: **temperature 1.0, top_p 0.95, max_tokens 32000** — NVIDIA's own
  Nemotron 3 Super agentic-coding cookbook. The harness was first run at
  temperature 0, which is off this model's documented operating point.
- Reasoning: **off** by default. Measured on a real 25 KB prompt: reasoning on
  consumed 8482 reasoning tokens and returned 643 chars; off returned a clean
  `finish_reason=stop` in 320 tokens.
- Output contract: SEARCH/REPLACE blocks (Aider's `diff` format). Aider's
  published benchmark measured whole-file as more reliable for *weaker* models
  (46%/39% vs 30%/19% for GPT-3.5) and documented that weaker models use
  search/replace pathologically by pasting entire files into both sides — which
  is exactly the 222-block output seen here. `diff` is correct for being 59–60%
  SWE-bench class with ~100 KB source files; `VRD_EDIT_FORMAT=whole` remains
  available for small-file repos.

## 7. Unit tests

`pytest tests/ -q` → **44 passed**.

Covers: commit classification, test-hunk stripping, diff→edit conversion on
verbatim git output, the "edits applied == patch applied" invariant, edit caps,
JSONL separator round-trips, pytest and Go summary parsing, BM25 ordering,
Wilson bounds.

## 8. What is *not* validated

- **Arm C (tuned model) has never run.** There is no GPU and no Nebius key on
  this machine. `train.jsonl` (367 examples), `job_spec.json` and
  `train_qlora.py` are generated and ready; the training call has not happened.
  Any claim about weights-vs-retrieval therefore rests on arms A and B only.
- **One repository.** `hermes-webui` is the corpus. CNCF Go repos (coredns,
  helm, opentelemetry-go) are wired and gated (3/3 verified on coredns at
  ~173 s/commit) but not yet run at corpus scale.
- **Single-turn editing.** NVIDIA's 59–60% SWE-bench figures come from
  multi-turn agentic harnesses. This eval is one-shot, which is a narrower and
  harder task, so absolute rates are not comparable to theirs.

---

Reproduce:

```
run.py ingest  --repo <repo> --data data/<name>
run.py gate    --repo <repo> --data data/<name>
run.py pairs   --repo <repo> --data data/<name> --out data/<name>/dataset
run.py oracle  --repo <repo> --data data/<name>/dataset --time-split <iso>
run.py train   --data data/<name>/dataset --out data/<name>/export --time-split <iso>
run.py eval    --repo <repo> --data data/<name>/dataset --arms A,B --time-split <iso>
```
