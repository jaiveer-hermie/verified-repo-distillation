# Project report — Verified Repo Distillation

Date: 2026-10-28 · Nebius × NVIDIA Hackathon, Coding & Agentic Engineering track
Status: **pipeline complete and validated; both arms measured.** Tuning on the
verified corpus produced no additional solved tasks.

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

| model | patch applied | tests passed | pass rate (Wilson 95%) | pass per applied |
|---|---|---|---|---|
| **Qwen3-30B-A3B base** | 22/26 (84.6%) | **3/26** | 11.5% [4.0, 29.0] | 14% |
| **Qwen3-30B-A3B + LoRA (148 verified pairs)** | 10/26 (38.5%) | **3/26** | 11.5% [4.0, 29.0] | 30% |

**The same three commits pass in both arms.** There is no commit solved by the
tuned model that the base model misses, and none the reverse. Tuning on the
verified corpus changed *no* outcome on this held-out set.

What it did change: the adapter became markedly less reliable at emitting a
usable SEARCH/REPLACE anchor (84.6% → 38.5% of patches apply), while the patches
it does manage to emit are more often correct (14% → 30% pass-per-apply). Output
grew from 18,018 to 34,872 tokens for the same 26 prompts.

Interpretation: with 148 training examples and one epoch, the adapter shifts
style, not capability. It learned something about *which* edit to attempt and
lost contract discipline. The task is not solved by adding corpus-specific
weights at this scale.

Served on a rented Vast.ai RTX 6000 Ada (48 GB) with vLLM 0.32.0,
`--enable-lora`, base `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` + our PEFT adapter
(`r=16`, attention projections). MoE LoRA path confirmed by
`MoE model detected. Using fused MoE LoRA implementation`.

The baseline applies patches readily (22/26) and almost never satisfies the
tests (3/26). The adapter was expected to move that number. It did not: it moved
the *other* number instead, cutting patch application to 10/26 while leaving the
3/26 pass count untouched.

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

Every self-serve route to serving this adapter was tested against the live API,
and each one is closed on this platform:

| attempt | result |
|---|---|
| `POST /v0/dedicated_endpoints` with `custom_weights_id = ftjob-…` (job id) | 422 — must start with `model-artifact_` |
| …with `custom_weights_id = ftckpt_1b6c9928…` (checkpoint id) | 422 — same validator |
| …with `custom_weights_id = "ft:Qwen/…:ckpt-step-11"` (the `fine_tuned_model_checkpoint` string) | 422 — same validator |
| …with `custom_weights_id = "job:ckpt"` | 422 — same validator |
| `GET /v0/model_artifacts` (the artifact registry the validator wants) | **403** — route exists, account has no access (beta, on request) |
| `POST /v0/models` with `{"source": "<job>:<ckpt>"}` (older LoRA-deploy flow) | **410 Gone** — "LoRA adapter deployment is no longer available" |

No public API path creates a model artifact: all 36 paths in the OpenAPI spec were
enumerated, and none fall under artifact/weights/custom. The `checkpoints`
endpoint is inspection-only — it returns the checkpoint id, the `ft:` model
string, `train_loss`, and the adapter `result_files`, but no identifier that the
deployment API will accept.

Additional signal: the returned `fine_tuned_model_checkpoint` string contains
literal `org_placeholder` and `IDPlaceholder` segments, i.e. the platform did not
fill in the organisation/artifact identifiers — consistent with the checkpoint
not being registered for deployment on this account.

The adapter itself is real and downloadable (38.2 MB: `adapter_model.safetensors`
26.78 MB, an 11.42 MB shard, plus configs), so the weights exist and are ours.
Serving them requires either beta access to model artifacts, or self-hosting the
merged model on a GPU (docs: *Merging LoRA Adapters* → vLLM).

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
demonstrably unseen (0/31 lines memorised).

The tuned arm finally ran, and the answer is negative: **fine-tuning on 148
verified commits did not solve a single additional task** (3/26 both arms, same
three commits). It reduced patch-application reliability from 84.6% to 38.5%.

That is a real result, not a failure to measure. The contribution is the corpus
construction and the measurement apparatus; the claim "tune on verified history
and the model gets better at this repo" is **not supported at this data scale**
on a 26-task holdout. To test it properly you need either far more pairs
(≈1k+, which would need the 285 dropped pairs recovered and a second repository)
or a repair loop, since one-shot emission is where the adapter lost the plot.
