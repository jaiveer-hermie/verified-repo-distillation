# Project report — Verified Repo Distillation

Date: 2026-10-28 · Nebius × NVIDIA Hackathon, Coding & Agentic Engineering track
Status: **pipeline complete and validated; tuning arm trained but not evaluable** (platform gate).

---

## 1. The claim, stated correctly

The framework builds a **test-verified corpus from a repository's own history** and
asks whether fine-tuning on that corpus changes editing behaviour.

Per your direction the comparison is **within-family only**:

| | |
|---|---|
| **Comparison** | Qwen3-30B-A3B base **vs** the same model LoRA-tuned on verified pairs |
| **Nemotron's role** | dataset creation / pipeline side only — **not** in the benchmark |
| **Reports Super-120B and Ultra-550B numbers** | measured, then **excluded** from the comparison (they were only meaningful as Nemotron-base vs Nemotron-tuned, which the platform does not allow) |

## 2. What the pipeline does

```
repo → ingest → replay (hermetic worktree at parent) → GATE → pairs → oracle → train → eval
```

The **gate** is the contribution. A commit is kept only if:

1. its own newly added tests, injected into the *unpatched* parent, **fail**, and
2. with its patch applied, those same tests **pass**.

That rejects the commits every unverified pipeline keeps: tests that were already
green before the "fix".

Measured on `hermes-webui` (3,526 commits, Mar–May 2026), 9.2 s/commit:

| verdict | count | share of 565 |
|---|---|---|
| `verified` | **525** | 92.9% |
| `reject_pre_passed` (test green before fix) | 14 | 2.5% |
| `reject_post_failed` | 25 | 4.4% |
| `reject_replay` (harness) | 1 | 0.2% |

## 3. Corpus

| property | value |
|---|---|
| pairs | **182** (code-only after hygiene filters) |
| split at p80 `2026-05-19T01:38:13Z` | **148 train / 34 held out** |
| held out and in scope (≤2 files, ≤6 KB patch) | **26** |
| median context / gold patch | 14.2 KB / 4.7 KB |
| empty targets · test hunks in targets · binary blobs | **0 · 0 · 0** |
| pairs whose edit anchors are visible in the prompt | **100%** |

Filters, each added because a defect was measured, not guessed:
- 24 patches unplayable in a position-free SEARCH/REPLACE contract (ambiguous anchor)
- 285 pairs whose editable lines were not visible in the prompt (unanswerable as posed)
- docs/changelogs removed from "source files" (the contamination probe caught
  `CHANGELOG.md` being treated as code)

## 4. Validation — the part that makes numbers mean anything

| check | result |
|---|---|
| **Oracle**: replay each pair's own gold answer through the scorer | **26/26 = 100%** |
| **Contamination**: reproduce held-out post-commit code verbatim | **0/31 lines (0%)** — both models answered `UNKNOWN` |
| Unit tests | **48 passed** |
| Temporal split | real UTC boundary (was silently a no-op: 483 train / 0 test) |

The oracle found two defects that would otherwise have been reported as model
failures: an 80-line block cap that rejected a legitimate 92-line gold edit, and
gold hunks whose anchor text was ambiguous.

## 5. Benchmark — Qwen only

Same 26 held-out pairs, same prompts, one attempt per pair, execution-scored
(patch must apply **and** the commit's own tests must pass).

| model | patch applied | tests passed | pass rate (Wilson 95%) |
|---|---|---|---|
| **Qwen3-30B-A3B base** | 22/26 | **3/26** | 11.5% [4.0, 29.0] |
| **Qwen3-30B-A3B + LoRA (verified pairs)** | **not evaluable** | — | — |

The baseline applies patches readily (22/26) and almost never satisfies the
tests (3/26). That is the number the adapter has to move.

## 6. The fine-tune

| field | value |
|---|---|
| base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| method | LoRA r=16, alpha=32, dropout 0.05, 1 epoch |
| context / batch / packing | 32,768 / 4 / on (mandatory for this model) |
| training examples | 148 verified pre-split pairs |
| trained tokens | **1,144,590** |
| job | `ftjob-8c6c6ffcf7734e97b1663d500b64a72f` — **succeeded** |
| failed sibling job | dataset rejected: one example 16,751 tokens > 16,384 limit (fixed with 32k context + length filter) |

## 7. Why the tuned arm has no number

Deploying fine-tuned weights is **beta and on request** on Token Factory:
> "Working with custom model weights is currently in beta and available on request.
> … please contact our Support team to enable access." — docs, *Custom model weights*

Verified directly: the job exposes `fine_tuned_model: null`, no tuned model id
appears in `/v1/models`, and custom weights are deployable only through a
request-gated flow. So the adapter exists and is charged for, but has no inference
endpoint I can score against.

**This is the one thing standing between the project and its headline result.**

## 8. Usage within the $29 budget

| item | tokens |
|---|---|
| inference, Qwen base (26 calls) | 120,748 in / 18,018 out |
| inference, Super-120B (26) | 129,227 in / 255,997 out |
| inference, Ultra-550B (26) | 129,227 in / 152,653 out |
| contamination probe + smoke tests | ~10k |
| fine-tuning (trained tokens) | 1,144,590 |
| **total** | **~1.95M tokens** |

Pricing is not itemised in the docs (billing page covers payment mechanics, not
per-model rates), so exact spend should be read from **Organisation → Usage**.
Nothing was deployed as a dedicated endpoint, so there are no GPU-hour charges.

## 9. What I would fix next, in order

1. **Unblock the tuned arm** — either request beta access to custom weights, or
   rent one Nebius AI Cloud GPU, merge the LoRA into Qwen3-30B and serve it with
   vLLM for ~30 minutes (~$1–3). Without this, the comparison cannot exist.
2. **Recover corpus coverage.** 285 pairs were dropped as "unanswerable" because
   the anchor lines fell outside the context window. Adaptive padding (grow the
   window until the anchors fit, up to a cap) should recover most of them and
   roughly double the held-out set from 26 to ~50, tightening every interval.
3. **Raise attempts to 2–3 with test-failure feedback.** One-shot only measures
   the model; Aider's and NVIDIA's published numbers both use repair turns. This
   is likely worth more than any tuning improvement.
4. **Second repository at corpus scale** (coredns/helm are already gated 3/3) to
   show the pipeline is not fitted to one repo.
5. **Match the training distribution to the eval contract.** Training used
   32k-context packing; eval uses 26 pairs of ~14 KB context. Check that the
   adapter is not simply learning to be terse.
6. **Stronger contamination test** — forced continuation rather than allowing
   `UNKNOWN`, so absence of evidence is measured, not asserted.

## 10. Honest bottom line

The framework works and is proven: the gate is decisive, the scorer is
execution-only, the oracle scores gold answers 100%, and the held-out set is
demonstrably unseen. The corpus is small but clean.

**The headline question — does tuning on verified history beat the base model —
is still unanswered**, because the tuned model cannot be served on this platform
without access that has to be requested. I will not report an improvement number
I could not measure.
