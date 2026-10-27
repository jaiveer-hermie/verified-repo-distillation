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

## License

Apache 2.0 — see [LICENSE](LICENSE).

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

Measured on `hermes-webui` (3526 commits, Mar–May 2026): the funnel below is
from a real run, not a projection.

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
| 4 pairs | `vrd/pairs.py` | narrow context so arm A and arm C see identical bytes; test hunks stripped from the training target |
| 5 train | `vrd/train.py` | chat-format export, temporal split, Nebius QLoRA job spec |
| 6 eval | `vrd/evaluate.py` | arms A/B/C, BM25 retrieval, execution scoring, Wilson intervals |

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
Scoring is execution: the produced diff must apply **and** the commit's own test
must pass. Never an LLM judge.

## Usage

```bash
# stage 1
python run.py ingest --repo /path/to/repo --out data/myrepo --all-parents

# stages 2+3 (both arms per commit)
python run.py gate --repo /path/to/repo --data data/myrepo --limit 20

# stage 4→6
python run.py pairs --repo /path/to/repo --data data/myrepo --out data/myrepo/dataset --time-split 2026-05-01
python run.py train   --data data/myrepo/dataset --out data/myrepo/run
python run.py eval    --repo /path/to/repo --data data/myrepo --arms A,B,C
```

Requires `uv` for Python repos and a Go toolchain for Go repos. No other Python
dependencies — the harness is standard library only, on purpose: a tool that
claims to reconstruct environments must not need a fragile environment itself.

## Repositories used

| Repo | Language | Why |
|---|---|---|
| `hermes-webui` | Python | entire history is Mar–May 2026, i.e. post-cutoff, so held-out commits are genuinely unseen |
| `coredns/coredns` | Go (CNCF) | self-contained `go test`, no server boot, tests the harness against a real CNCF codebase |
| `helm/helm` | Go (CNCF) | second Go target, different package layout |
| `open-telemetry/opentelemetry-go` | Go (CNCF) | fast unit suite, stress-tests the replay cache |

## Tool feedback

*(section required by the hackathon rules — filled in as the build proceeds,
recording what worked, what broke, and what we would tell the maintainers)*
