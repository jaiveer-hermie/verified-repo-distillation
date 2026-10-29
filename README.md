# Verified Repo Distillation

**Does tuning on a repository beat retrieving from it? A test-verified answer.**

Nebius x NVIDIA Global AI Hackathon — Coding and Agentic Engineering Track

---

Every existing "fine-tune an LLM on your codebase" tool has the same blind spot:
it mines `git log` into `(context → patch)` pairs and assumes each commit was
correct. Nobody checks. A real repository's history is full of commits whose
tests do not test the change, commits that were reverted a day later, and commits
whose patch was never green.

This pipeline checks, **by running the tests** — and then answers a question
nobody has measured for code: after tuning, does the model patch held-out commits
better than the same model with the repository retrieved into its context?

---

## The gate (the actual contribution)

For each candidate commit `C` with parent `P`:

1. check out `P`
2. inject **only `C`'s newly added test files**
3. run them → must **fail** (the behaviour does not exist yet)
4. apply `C`'s **full patch**
5. run the same tests → must **pass**

A commit that survives both is a pair where a behavioural change was
*mechanically observed*. Two failure modes are named and counted:

- `reject_pre_passed` — the test passed before the fix. The commit's test does
  not test the commit. Every unverified pipeline ingests this as ground truth.
- `reject_post_failed` — the commit was not green to begin with.

Measured on `hermes-webui` (3526 commits, Mar–May 2026) — a real run, 9.2 s per
commit:

| verdict | count | share of 565 gate candidates |
|---|---|---|
| `verified` | 525 | 92.9% |
| `reject_pre_passed` (test green before the fix) | 14 | 2.5% |
| `reject_post_failed` (never green here) | 25 | 4.4% |
| `reject_replay` (harness failure) | 1 | 0.2% |

## Pipeline

```
repo ──▶ 1 INGEST ──▶ 2 REPLAY ──▶ 3 GATE ──▶ 4 PAIRS ──▶ 5 QLoRA ──▶ 6 EVAL
         classify      reconstruct    counter-     (context,    Nemotron    3 arms
         noise out     env at P       factual      patch)       on Nebius   execution
                                       test         verified                 scored
```

| Stage | Module | What it does |
|---|---|---|
| 1 ingest | `vrd/ingest.py` | walks `git log`, drops merges/docs/tests-only/lockfile/oversize commits, marks gate candidates (patch adds a test **and** touches source) |
| 2 replay | `vrd/replay.py` | hermetic worktree at the parent, toolchain reconstruction (Python via `uv`, Go via module cache), no inherited `PYTHONPATH` |
| 3 gate | `vrd/replay.py` | counterfactual test injection, both arms |
| 4 pairs | `vrd/pairs.py` | windowed context so the editable lines are actually visible, test hunks stripped from the target, fairness filters |
| 5 train | `vrd/train.py` | chat-format export, temporal split, Nebius QLoRA job spec |
| 6 eval | `vrd/evaluate.py` | arms A/B/C, BM25 retrieval, execution scoring, Wilson intervals |
| — | `run.py oracle` | **self-check: replays each pair's own gold answer through the scorer** |

## The corpus that comes out

182 pairs (code-only) from a 525-verified gate:

- median context 14.2 KB, median gold patch 4.7 KB
- **0** pairs with an empty target
- **0** test hunks in targets (the tests are the instrument, never the answer)
- **0** binary blobs, **0** docs/changelogs
- **100%** of pairs have the text the answer must reproduce visible in the prompt
- temporal split at p80 (2026-05-19T01:38:13Z):
  **148 train / 34 held out** (26 held out and in scope)

Build-time filters, each added because a real defect was measured
(`VALIDATION.md`): 24 patches dropped as unplayable in a position-free
SEARCH/REPLACE contract, 285 dropped as unanswerable because their editable lines
had been truncated out of the prompt.

## The three arms

The comparison everyone runs is base-vs-tuned. It cannot tell you whether the
gain came from the repository's *information* or from the *adapter*. So:

| Arm | Model | Context |
|---|---|---|
| **A** | base | minimal — the files the commit touched, at parent state |
| **B** | base | retrieved — BM25 over the repo at the parent commit |
| **C** | tuned | minimal — byte-identical prompt to A |

If C > B, weights won. If B > C, retrieval won — and that is the finding, shipped
rather than buried. If A ≈ C, the adapter learned nothing and the report says so.
Scoring is execution: the produced patch must apply **and** the commit's own test
must pass. Never an LLM judge. Every rate carries a Wilson interval.

`--attempts N` adds Aider-style repair turns: the failing test output is fed back
and the model is asked for a corrected patch. `first_attempt_success` is recorded
separately, so pass@1 and pass@k are never conflated.

## Self-check before any number is quoted

```bash
python run.py oracle --repo <repo> --data <data>/dataset --time-split <iso>
```

Replays each pair's **own verified patch**, rendered in the contract the model
must speak, through the identical scoring path. On the held-out set:
**43/43 = 100%**. If a gold answer fails, the harness is wrong and every model
number is meaningless. This test found two real defects — an 80-line cap that
rejected a legitimate 92-line edit, and gold hunks whose search text was
ambiguous — both fixed before any arm was reported.

## Findings so far

- The gate is cheap and decisive: 9.2 s per commit, and it removes 2.5% of
  candidates as tests that were already passing before the "fix".
- Corpus defects are silent and expensive. Three were found only by mechanical
  checking: 36% of targets were empty (a hunk-header regex bug), the corpus was
  378 MB of binary blobs instead of 19 MB, and `commit_date` was empty on every
  pair so the "temporal split" was silently doing nothing.
- **An unfair task produces a fake finding.** These commits touch files whose
  median size is 521 KB. With an 8 KB per-file context cap the model was asked to
  reproduce text it had never been shown, and every arm scored ~0. That was a
  harness bug, not a model property; fixed with hunk-centred context windows and a
  fairness filter that drops any pair whose anchors are not visible.
- Single-shot editing by a 120B open model is weak on this corpus: most attempts
  either apply without satisfying the test, or fail to match the anchor text at
  all. This is exactly why the repair loop and the tuned arm exist.

## Usage

```bash
# stages 1-3
python run.py ingest --repo /path/to/repo --data data/myrepo
python run.py gate   --repo /path/to/repo --data data/myrepo

# stage 4 + self-check
python run.py pairs  --repo /path/to/repo --data data/myrepo --out data/myrepo/dataset
python run.py oracle --repo /path/to/repo --data data/myrepo/dataset --time-split 2026-05-19T01:38:13Z

# stages 5-6
python run.py train --data data/myrepo/dataset --out data/myrepo/run --time-split <iso>
python run.py eval  --repo /path/to/repo --data data/myrepo/dataset --arms A,B --attempts 2 --time-split <iso>
```

Requires `uv` for Python repos and a Go toolchain for Go repos. No other Python
dependencies — the harness is standard library only, on purpose: a tool that
claims to reconstruct environments must not need a fragile environment itself.
Model credentials are read from `~/.env` at runtime; nothing is written to the
repo. Sampling follows NVIDIA's own agentic-coding recipe for Nemotron 3 Super
(temperature 1.0, top_p 0.95).

## Repositories used

| Repo | Language | Why |
|---|---|---|
| `hermes-webui` | Python | entire history is Mar–May 2026, i.e. post-cutoff, so held-out commits are genuinely unseen; primary corpus |
| `coredns/coredns` | Go (CNCF) | self-contained `go test`, no server boot; 3/3 verified at ~173 s/commit — generality check |
| `helm/helm` | Go (CNCF) | second Go target, different package layout |
| `open-telemetry/opentelemetry-go` | Go (CNCF) | fast unit suite, stress-tests the replay cache |

## Honest limitations

- **Arm C has never run.** There is no GPU and no Nebius key on the build
  machine; `train.jsonl` (277 examples), `job_spec.json` and `train_qlora.py` are
  generated and ready. The weights-vs-retrieval question is therefore still open,
  and this README does not claim otherwise.
- **One primary repository.** CNCF Go repos are supported and gated, but not yet
  run at corpus scale. For public repos only a temporal holdout is honest — the
  model may already know the code.
- **Not comparable to NVIDIA's published numbers.** Nemotron 3 Super reports
  59–60% on SWE-bench Verified in multi-turn agentic harnesses with tool calls.
  This eval is one-shot text editing on a private repo: a narrower, harder task.

## Tool feedback

*(required by the hackathon rules — what worked, what broke, and what we would
tell the maintainers)*

- **`git show --patch` silently drops binary payloads** unless `--binary
  --full-index` is passed. 50 gate failures were our own harness, not the
  commits; 47 recovered on retry. Any tool replaying patches needs this.
- **Reasoning tokens count against `max_tokens`** (per OpenRouter's docs), so a
  reasoning model returns `finish_reason: "length"` with empty content. Reading
  `usage.completion_tokens_details.reasoning_tokens` and subtracting is the
  difference between "the model failed" and "the budget was eaten by thinking".
- **`json.dumps(ensure_ascii=False)` plus `str.splitlines()` is a trap**: JSON
  permits raw U+0085/U+2028 inside strings, and `splitlines()` treats them as
  line breaks, silently truncating JSONL records. Writers must escape, readers
  must iterate the file handle.
- **Nemotron-specific:** at temperature 1.0 the model is markedly more verbose
  than at 0.0 (28 KB answers where 5 KB was expected) and will paste whole files
  into SEARCH blocks — the pathological diff behaviour Aider documents. Hard caps
  in the parser, not polite requests in the prompt, are what actually bound it.

## License

Apache 2.0 — see [LICENSE](LICENSE).
