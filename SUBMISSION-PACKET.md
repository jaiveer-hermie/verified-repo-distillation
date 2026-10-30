# Submission packet — Nebius × NVIDIA Global AI Hackathon

Deadline: **2026-10-30, 10:00 AM PT / 22:30 IST** · Platform: Devpost
Event: https://nebiusglobalaihackathon.devpost.com

Fill the Devpost form with the copy below. Fields marked ⛔ must be produced by you
(they cannot be generated from this machine).

---

## 0. What the rules actually require

From the event's Requirements + "Final stretch" checklist:

1. Project runs on **Nebius Token Factory or Nebius AI Cloud** and uses **at least one NVIDIA open-source model**.
2. Fits **one** of four tracks.
3. **Project description** — what you created, why, how it works.
4. **Working demo URL** — hosted app or test build (we have one via GitHub Pages).
5. **Demo video ≤ 3 min, public on YouTube**, showing the project working and explaining Nebius + NVIDIA usage. ⛔ you must record + upload.
6. **Public repo** with an **open-source license visible at the top** + README with setup/run instructions, and the NVIDIA/Nemotron usage highlighted. ✅ Apache-2.0 at root, README section added.
7. **Feedback** on Nebius Token Factory / AI Cloud / NVIDIA tools — "make it specific".
8. If the project existed before the submission period: a written explanation of what was **significantly updated** during it.

Judging criteria: Technological Implementation · Design · Potential Impact · Quality of the Idea.

---

## 1. Project name

```
Verified Repo Distillation
```

## 2. Track (category)

```
Coding and Agentic Engineering
```

## 3. Tagline / elevator pitch (≤ 200 chars)

```
We execute a repo's own tests on both sides of every commit, keep only pairs that provably flip a test red→green — then measured that 10% of such "verified" data is false and tuning on it buys nothing.
```

Shorter alternative (≤ 120 chars):

```
A gate that proves a git commit actually fixed a test, plus the measurement of what that proof is worth.
```

## 4. Project description

```
Every tool that offers to "fine-tune an LLM on your codebase" does the same thing: it mines git log into (context → patch) pairs and assumes each commit was correct. Nobody checks. Real history is full of commits whose test never tested the change, commits reverted the next day, and commits whose suite was red for unrelated reasons.

Verified Repo Distillation checks, by executing the tests. For a candidate commit C with parent P we check out P, inject ONLY the tests C added, and require the suite to FAIL. We then apply C's full patch and require it to PASS. We go further than the suite-level rule: we parse per-test outcomes from JUnit XML, require at least one NAMED test to transition fail→pass (SWE-bench's FAIL_TO_PASS), require no previously-passing test to regress (PASS_TO_PASS), and re-run the post arm to reject pairs whose outcomes are not stable — because a flaky test is indistinguishable from a fixed one unless you run it twice.

Then we audited our own output. Replaying 40 of the 525 commits our gate had already accepted, under the strict rule, 36 survived (90%, Wilson 95% CI [76.9, 96.0]). Four were never evidence of anything: two had no named transition at all, one does not reproduce, and one has four tests that flip between identical runs — including the two that had been counted as its evidence. A separate 40-commit drift survey, different seed, also came out at 90%. We publish our own corpus's failure rate; "525 verified" was a 10%-wrong claim.

With the survivors we QLoRA-tuned Qwen3-30B-A3B on 148 verified pairs through Nebius Token Factory (1,144,590 tokens, train loss 0.359) and compared it against its own base on one vLLM server, same prompts, same 26 held-out post-cutoff commits, scored only by execution (patch applies AND the commit's tests pass). Base 4/26, tuned 3/26, with the solved sets differing (2 shared, 2 base-only, 1 tuned-only): tuning is net −1 task. Applied-patch rate collapsed 17/26 → 6/26 (Fisher p = 0.0047) while pass-per-applied doubled 24% → 50%. The adapter did not learn the repository; it lost the output contract.

NVIDIA models: Nemotron 3 Ultra 550B solves 17/26 of the same held-out commits through Nebius Token Factory, Nemotron 3 Super 120B solves 3/26. Nemotron cannot be fine-tuned on Token Factory (HTTP 422 for both Super and Nano), which is why the tuned arm is Qwen — stated wherever the numbers appear.

The deliverable is not a better model. It is: a verifier for training data that anyone building SWE-bench-style corpora can run, an execution-scored benchmark whose instrument is proven first (gold answers score 26/26 through the identical path; contamination probe reproduces 0/31 held-out lines), and an honest negative result with a mechanism — the bottleneck for small-model repo editing is anchor emission, not missing knowledge, so more of the same data cannot fix it.
```

## 5. Built with (Devpost tags)

```
python, pytest, git, qlora, peft, lora, vllm, nebius-token-factory, nvidia-nemotron, qwen3, docker-free-hermetic-replay, bm25, junitxml, github-pages
```

## 6. Links

| field | value |
|---|---|
| Repository | https://github.com/jaiveer-hermie/verified-repo-distillation |
| Demo / test build | https://jaiveer-hermie.github.io/verified-repo-distillation/ |
| Video | ⛔ record + upload to YouTube, paste unlisted-or-public link |
| License | Apache-2.0, `LICENSE` at repo root |

## 7. Feedback on Nebius Token Factory, AI Cloud and NVIDIA tooling

```
Specific, from building against it for three days.

1. Fine-tuning rejects datasets without telling you the constraint up front. A job fails with "packing is not supported ... set packing to True", then "batch_size 2 < minimum 4", then "batch_size x context_length >= 131072", then "Sequence length is too long: 16751 > 16384". Each surfaced one at a time, each costing a job submission. A preflight/dry-run endpoint that validates (rows, max sequence length, batch x context) against a model's constraints before accepting the job would have saved most of a day.
2. The fine-tune API returns 422 "Model doesn't exist or you don't have access to it" for the Nemotron models that the inference API happily serves. Serving and fine-tuning catalogues are different and the list endpoint does not say which is which; /v1/fine_tuning/models exists but is undocumented in the fine-tuning guide we followed.
3. Deploying tuned weights is gated behind 'model-artifact_' IDs that no public endpoint creates: GET /v0/model_artifacts returns 403 on this account, POST /v0/models returns 410 "LoRA adapter deployment is no longer available", and enumerating all 36 documented OpenAPI paths finds no artifact-creation route. The docs say you can deploy "straight from the dashboard"; the programmatic path is closed. We ended up downloading adapter + tokenizer via /v1/files/{id}/content and serving on rented vLLM.
4. That file-content endpoint 307-redirects to a signed object-store URL that fails with AccessDenied if the client forwards the Authorization header, and also if the URL keeps the explicit :443 Nebius puts in it. Strip both and it works; that is a 40-minute debugging session for anyone using a plain HTTP client.
5. What was genuinely good: reasoning capability metadata on /api/v1/models (supported_efforts, mandatory, supports_max_tokens) let us stop starving Nemotron's reasoning budget blindly — that is the right shape for a model catalogue and more providers should expose it. Nemotron 3 Ultra at 550B, served on demand with no quota negotiation, is what made a 300B-class arm of a benchmark possible on a student budget.
6. Credit: ~$2 of the $29 budget covered every Nemotron call plus the fine-tune, which is the whole reason this project could afford to iterate on data quality instead of on infrastructure.
```

## 8. Pre-existing project statement

```
No prior release. The repository was created during the submission period (first commit
2026-10-27). The gate, the corpus, the LoRA fine-tune, the benchmark and the audit in
GATE-AUDIT.md were all written in this window.
```

## 9. Demo video — 3-minute script (you must record this)

The box here renders no browser frames, so the recording has to come from your machine.
Screen-record the dashboard and a terminal. Keep it under 3:00 — hard limit.

| time | shot | say |
|---|---|---|
| 0:00–0:15 | Devpost/dashboard hero | "Every tool that fine-tunes on your codebase assumes your commits were correct. We check, by running the tests." |
| 0:15–0:50 | Terminal: `run.py gate --strict --repeats 2` on a few commits, showing verdicts | "Pre-arm must fail, post-arm must pass, and we go further — a named test must flip, nothing green may go red, and we re-run to catch flaky tests." |
| 0:50–1:25 | Dashboard §1 + §2 (90%, the four false pairs) | "Then we audited our own gate. 36 of 40 of the commits it had already accepted survive the strict rule. These four were never evidence: two had no transition at all, one doesn't reproduce, one has four tests flipping between identical runs." |
| 1:25–1:55 | Dashboard §3 (oracle, contamination) | "Before quoting any model number we prove the instrument: gold answers score 26/26 through the same path, and a contamination probe reproduces 0 of 31 held-out lines." |
| 1:55–2:30 | Dashboard §4 (base vs tuned) | "We tuned Qwen3-30B on 148 verified pairs on Nebius Token Factory and ran both arms on one vLLM server. Base 4 of 26, tuned 3 of 26 — net minus one task, and applied patches fell from 17 to 6. The adapter lost the output contract, not the ability to code." |
| 2:30–2:50 | Dashboard §5 (Nemotron arm) | "Nemotron 3 Ultra solves 17 of 26 of the same commits through Token Factory. Nemotron cannot be fine-tuned there, which is why the tuned arm is Qwen." |
| 2:50–3:00 | Repo page, LICENSE + README | "Apache-2.0, everything reproducible, and the audit that produced these numbers is in the repo." |

Recording tips: 1080p, no audio music, speak over it, show the URL bar once so the repo is visible.

## 10. Pre-submit checklist

- [x] Runs on Nebius Token Factory — every Nemotron call
- [x] Uses an NVIDIA open-source model — Nemotron 3 Ultra 550B, Super 120B
- [x] Track selected — Coding and Agentic Engineering
- [x] Description written (above)
- [x] Public repo, Apache-2.0 at root, README with run instructions
- [x] NVIDIA/Nebius usage section in README (required by rules)
- [x] Working demo URL (GitHub Pages dashboard)
- [x] Feedback written, specific
- [ ] ⛔ Demo video ≤ 3 min, public on YouTube
- [ ] ⛔ Paste video URL into the Devpost form
- [ ] ⛔ Add teammates to the submission form if any
- [ ] ⛔ Hit submit before 22:30 IST

## 11. If a judge asks the hard questions

| question | answer |
|---|---|
| Did tuning help? | No. 4/26 base vs 3/26 tuned on one server, same prompts; solved sets differ (2 shared, 2 base-only, 1 tuned-only). Application rate fell 17→6, Fisher p=0.0047. |
| Then what is the contribution? | The verifier and the measurement. It finds that ~10% of commits a suite-level gate accepts were never evidence of anything, and it bounds what naive distillation buys you. |
| Why not Nemotron for tuning? | Token Factory returns 422 for both Super and Nano as fine-tuning bases. Qwen is the substitute, disclosed wherever it appears. |
| Is 26 tasks enough? | No — CI [4, 29]. It licenses "no improvement" and cannot detect a +10-point gain. Stated in LIMITATIONS.md. |
| How do we know the held-out set is unseen? | Temporal split at 2026-05-19T01:38:13Z (all commits post-cutoff) plus a contamination probe that reproduces 0/31 held-out lines. |
| Is the harness itself correct? | Gold answers score 26/26 through the identical parse→apply→test→score path, and 56 unit tests cover the transition rule and the corpus invariants. |
