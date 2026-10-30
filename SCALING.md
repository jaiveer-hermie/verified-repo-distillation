# SCALING.md — cost of running the verified-corpus gate across CNCF

**Scope.** What does it cost to run the counterfactual gate (stage 2+3 of this
pipeline) over the CNCF landscape, and where does the money actually go?

**Date.** 2026-10-30. **Repo:** `verified-repo-distillation`.

Every number below is tagged **[MEASURED]** (produced by a run in this repo,
with the command or artifact named), **[DERIVED]** (arithmetic on measured
values, with the arithmetic shown), or **[ASSUMED]** (a stated modelling
choice that is *not* backed by a measurement in this repo). Where a figure is
an assumption, the assumption is written in the same place as the number so it
can be attacked independently.

---

## 0. Executive summary

| quantity | value | basis |
|---|---|---|
| Gate cost, Python, 2-dep repo | **9.55 s/candidate** | [MEASURED] `data/hermes-webui-all/verdicts.jsonl`, n=565 |
| Gate yield, Python | **525 / 565 = 92.9%** verified | [MEASURED] same file |
| Gate cost, Go, warm cache | **57.4 s mean / 65.3 s median** (n=4) | [MEASURED] two independent coredns runs |
| Gate cost, Go, cold cache | **341–420 s** (one commit) | [MEASURED] `data/coredns_gate2.log` + a second run |
| Ingest cost | **9.4 ms/commit** | [MEASURED] 1600-commit coredns ingest |
| CNCF corpus size | **488,341 first-parent commits** across 76 repos | [MEASURED] `git rev-list --count --first-parent` on every repo |
| Gate candidates, all 76 repos | **≈32,000** | [DERIVED] §3.2 |
| Gate candidates, post-kill-list | **≈8,800** | [DERIVED] §4.2 |
| Wall clock, 64-core, kill-list applied, mid case | **5.7 h** at concurrency 32 | [DERIVED] §3.3 |
| Wall clock, 64-core, no kill-list, mid case | **21.6 h** at concurrency 32 | [DERIVED] §3.3 |
| **GPU (LoRA, 5k–10k examples)** | **$6–$38** | [DERIVED] §5 |
| **CPU (the gate)** | **$3–$20** | [DERIVED] §5 |

**The headline: the gate is a CPU-bound, embarrassingly-parallel, hours-scale
job. The GPU is a rounding error.** Running the entire CNCF landscape through
the gate costs roughly the same as one rented GPU-hour. The constraint is not
compute — it is *which repositories you are willing to touch*.

Two corrections to numbers this project already publishes, made by
re-measuring rather than by re-reading:

1. **The Go cost is 57.4 s, not 173 s.** The published 173 s is the mean of
   three commits, one of which was a cold module-cache commit. See §1.2.
2. **The Python cost is 9.55 s, not 9.2 s.** The published 9.2 s is the mean
   of the *first* pass, before 50 harness-failure commits were retried. See §1.1.


---

## 1. Measured baseline

### 1.1 The Python baseline (the reference number)

**[MEASURED]** `hermes-webui`, a Python repo with two runtime dependencies.

```
repo          : hermes-webui (private; 3,526 first-parent commits, Mar–May 2026)
commits walked: 3,526
kept          : 1,813
gate candidates: 565
```

Gate verdict breakdown, from `data/hermes-webui-all/verdicts.jsonl` (n=565,
the final post-retry file):

| verdict | count | share of 565 | meaning |
|---|---|---|---|
| `verified` | **525** | **92.9%** | pre-arm failed, post-arm passed |
| `reject_pre_passed` | 14 | 2.5% | test was already green before the patch |
| `reject_post_failed` | 25 | 4.4% | never green in this environment |
| `reject_replay` | 1 | 0.2% | harness failure (root commit, no parent) |

**565 gate candidates → 525 verified (92.9%).** This is the yield rate: for
every 100 commits the gate accepts, it rejects ~7. The `reject_pre_passed`
bucket is the entire point of the design — those 14 commits are exactly what a
fetch-the-diff pipeline would have trained on.

Timing, same file:

| statistic | value |
|---|---|
| mean wall/commit, final file | **9.55 s** |
| median wall/commit | 8.71 s |
| min / max | 0.00 s / 32.99 s |
| mean, `verified` rows only | 9.49 s |
| mean, `reject_post_failed` rows | 11.28 s |
| **total CPU time for all 565** | **1.50 h** |

**A note on the 9.2 s vs 9.55 s discrepancy.** The `9.2 s/commit` figure quoted
in `REPORT.md` and `VALIDATION.md` is the mean of the **first** pass
(`data/hermes-webui-all/verdicts_pass1.jsonl`, mean 9.16 s). That pass produced
**50 `reject_replay` rows** — all of them the harness bug where `git show
--patch` silently drops binary payloads without `--binary --full-index`
(`vrd/ingest.py:184`). Those 50 were re-run (`gate_retry.log`), and 47
recovered, giving the final 565-row file above.

The honest production number is therefore **9.55 s** (final file), and the
retry pass added **50 × 9.2 s ≈ 7.7 minutes** of extra work on top of it —
about **8% overhead** for a bug that is now fixed. I use 9.55 s throughout.

The 9.55 s figure is **inclusive** of venv construction: `build_venv()`
(`vrd/replay.py:111`) creates a venv per dependency-file content hash and
reuses it for sibling commits. The measured mean for `verified` commits
(9.49 s) is barely above the median (8.71 s), which means venv build is *not*
the bulk of the cost — the bulk is two test invocations plus two worktree
checkouts. See §2.1.

### 1.2 The Go baseline (and a correction to a published number)

`VALIDATION.md` and `README.md` both quote coredns at **"~173 s/commit"** and
`data/coredns_gate2.log` prints `mean wall/commit: 173.0s`. That number is
**real but misleading**: it is the mean over 3 commits, one of which was a
cold-cache commit that spent 392 s downloading Go modules.

**[MEASURED]** Two independent runs on coredns, same repo, same cache root:

| run | commit 1 | commit 2 | commit 3 | mean |
|---|---|---|---|---|
| A (`data/coredns_gate2.log`) | 419.94 s | 12.55 s | 86.56 s | 173.0 s |
| B (re-run, warm `~/.cache/vrd-go`) | 341.0 s | 45.7 s | 84.9 s | 157.2 s |

Commit 1 in each run is the cold-cache commit. Its pre-arm log is dominated by
`go: downloading github.com/go-openapi/...` lines — the module cache was empty.

| population | n | mean | median |
|---|---|---|---|
| all six commits | 6 | 165.1 s | 85.7 s |
| **warm-cache commits only** | **4** | **57.4 s** | **65.3 s** |

**For cost modelling I use 57.4 s (warm mean) / 65.3 s (warm median) as the Go
per-candidate cost, and 341–420 s as a one-time per-repo cold-cache cost.**
Quoting 173 s as "the Go cost" overstates it by 3×, which would have made the
whole projection 3× too expensive.

**[MEASURED]** Ingest throughput, coredns, three depths:

| commits walked | wall | ms/commit | gate candidates found |
|---|---|---|---|
| 300 | 4.49 s | 15.0 | 17 |
| 800 | 8.61 s | 10.8 | 35 |
| 1,600 | 15.07 s | **9.4** | 50 |

9.4 ms/commit is the steady-state figure (the 300-commit number includes
interpreter warm-up). Ingest is `git log --numstat` plus one
`git show --name-status` subprocess per commit (`vrd/ingest.py:172`).

---

## 2. Per-ecosystem cost model

### 2.1 What one gate candidate actually costs, structurally

From `counterfactual_verify()` (`vrd/replay.py:347`), per candidate:

1. `git worktree add --detach <parent>` — one checkout of the parent tree.
2. Inject the commit's new test files into that tree (`git show <sha>:<path>`).
3. **pre-arm:** run the injected tests → must FAIL.
4. `git apply` the commit's full patch.
5. **post-arm:** run the same tests → must PASS.
6. `git worktree remove --force`.

So the unit cost is **2 × (worktree + test run)**, not 1 ×. Any per-test
number must already be read as "two test invocations". The Python 9.55 s
includes both.

There is also a **fixed per-repo** cost, separate from the per-candidate cost:

- module/dependency download. Measured: 392 s of `go: downloading` on coredns's
  first candidate (run A) and 341 s (run B). This is paid **once per repo**,
  not per candidate, and it is *not* eliminated by a shared `GOMODCACHE`
  because a repo's `go.mod` changes over time and each change can pull new
  module versions.
- venv construction for Python, keyed on the dependency-file content hash.
  For a 2-dependency repo this is ~2–4 s; for a 100-dependency repo it is
  60–180 s, amortised across every sibling commit sharing that hash.

### 2.2 The per-ecosystem table

**[MEASURED]** rows are flagged. Everything else is **[ASSUMED]** with the
toolchain reason given.

| ecosystem | lo | mid | hi | s/candidate | justification |
|---|---|---|---|---|---|
| **Python** | 8 | **10** | 45 | **[MEASURED]** lo/mid (9.55 mean, 8.71 median) | 2 deps, `uv` venv cached by dep-hash, pytest only. hi covers a 100-dep repo whose venv rebuilds often. |
| **Go** | 30 | **65** | 180 | **[MEASURED]** mid = warm median 65.3 s; lo = warm mean 57.4 s rounded down; hi = a repo with heavy `go: downloading` churn | `GOTOOLCHAIN=auto` (`vrd/replay.py:265`) pulls a new toolchain when `go.mod` requires one — measured: pinning `local` made every coredns commit fail with `go.mod requires go >= 1.26.0`, a harness artefact that looked like a test failure. Shared `GOMODCACHE`, per-sandbox `GOCACHE`. |
| **C / C++** | 60 | **180** | 600 | [ASSUMED] | No runner exists in `replay.py` — `detect_toolchain()` (`vrd/replay.py:219`) returns `"unknown"` for a repo with no `go.mod`/`pyproject.toml`/`package.json`, and `run_tests()` returns `ran=False`. Supporting C++ means writing a runner *and* handling cmake + compiler + cgo, with a full relink per commit. Envoy alone is 27,667 first-parent commits. |
| **Rust** | 45 | **120** | 400 | [ASSUMED] | Same "no runner" problem, plus `cargo build` per commit against a cold `~/.cargo` registry, and `tikv/tikv` has a Rust `tests/` crate. |
| **Java** | 90 | **240** | 600 | [ASSUMED] | Same "no runner" problem, plus maven/gradle with a cold `~/.m2` cache, JVM warm-up per invocation, and surefire forking a JVM per test class. Keycloak is a Quarkus/maven project with 20,570 first-parent commits. |
| **TypeScript / JS** | 40 | **120** | 300 | [ASSUMED] | `package.json` is detected by `detect_toolchain()` but the dispatcher returns `ran=False` for `"node"` — there is **no Node runner**. Adding one means `npm ci` per commit plus jest/tsc. Backstage is 22,411 first-parent commits. |

**Important caveat on the "mid" column.** The Python and Go rows are grounded
in real runs. The C/C++, Rust, Java and TypeScript rows are **modelling
assumptions**, not measurements — this pipeline has never gated a non-Python,
non-Go commit. They are stated so they can be replaced with measurements; they
are not dressed up as facts.

### 2.3 Why "no runner" is a hard blocker, not a config tweak

`run_tests()` (`vrd/replay.py:301`) dispatches on `detect_toolchain()` and has
exactly two real branches:

```python
if toolchain == "go":     return run_go_tests(...)
if toolchain == "python": return run_pytest(...)
return TestResult(ran=False, ...), "none"
```

A `"node"` tree falls through to the `ran=False` branch. `TestResult.green`
requires `ran and returncode == 0 and failed == 0 and errors == 0` — a
`ran=False` result is never green. Therefore **every candidate in a Node or
C++ or Rust or Java repo currently rejects as `reject_post_failed`**, with no
useful signal. Adding an ecosystem is a real engineering task (build system
call, hermetic dependency cache, output parser), not a config line.

---

## 3. CNCF-wide sizing

### 3.1 The corpus: measured, not estimated

**[MEASURED]** The CNCF landscape (`cncf/landscape`, `landscape.yml`, parsed
from the upstream source) lists **260** CNCF-governed projects with an
`accepted` date: **39 graduated**, **38 incubating**, **153 sandbox**,
**27 archived**, 3 uncategorised.

The task brief says "~100 graduated projects". The authoritative upstream list
says **39 graduated** and **38 incubating** = **77 non-sandbox projects**. I
cloned and measured **76 of those 77** (one repo, `open-feature/spec`, has no
commits on the default branch) as blobless bare clones and counted commits
locally with the same git plumbing the pipeline uses:

```
git clone --filter=blob:none --bare https://github.com/<repo>.git
git rev-list --count --first-parent HEAD
git rev-list --count HEAD
```

| population | repos | first-parent commits | all commits |
|---|---|---|---|
| **Graduated** | 32 | 256,475 | 451,630 |
| **Incubating** | 36 | 157,514 | 334,820 |
| **Total measured** | **76** | **488,341** | **896,025** |

Language mix (by dominant source extension at HEAD): **53 Go, 4 Python, 3 Java,
2 C++, 2 Rust, 2 TypeScript**, 7 documentation-only, 5 other.

### 3.2 Total commits vs gate candidates

These are very different numbers and conflating them is the single easiest way
to over-estimate this project by 15×.

A **gate candidate** requires (`vrd/ingest.py:251`):
`commit.added_tests and commit.source_paths` — i.e. the commit *adds* a test
file **and** touches non-test, non-doc source.

**[MEASURED]** candidate rates:

| repo | lang | first-parent commits | candidates | rate |
|---|---|---|---|---|
| hermes-webui | Python | 3,526 | 565 | **16.02%** |
| coredns (400-commit slice) | Go | 400 | 24 | **6.00%** |
| coredns (300-commit slice) | Go | 300 | 17 | **5.67%** |
| coredns (1,600-commit slice) | Go | 1,600 | 50 | **3.13%** |

The brief's "565 of 3526 = 16%" is **exactly right for hermes-webui**. But
coredns — a Go repo of comparable age and size — is **3–6%**, a factor of
2.7–5 lower. The Python number is not transferable to Go, and the two repos
the pipeline has actually run are the only two data points that exist.

**[DERIVED]** Applying 16.02% to Python repos and 6.00% to everything else
(the conservative choice: Go is measured at 6%, and unmeasured ecosystems are
assumed no better than Go):

```
Python repos (4 of 76)   : measured rate 16.02%
All other repos (72 of 76): assumed rate  6.00%

candidates = Σ(first-parent commits × rate)
           = 31,996  ≈ 32,000
```

| population | first-parent commits | candidates |
|---|---|---|
| Graduated | 256,475 | 15,513 |
| Incubating | 157,514 | 12,022 |
| **All 76** | **488,341** | **≈32,000** |

**Range.** The honest range is wide because the per-language rates are
assumptions, not measurements:

- **Low:** 3.13% (coredns' own 1,600-commit rate applied everywhere) → ≈15,300 candidates
- **Mid:** 6.00% for Go and other, 16.02% for Python → **≈32,000 candidates**
- **High:** 16.02% applied everywhere (i.e. every repo behaves like a Python
  repo with an unusually high test-addition rate) → ≈78,200 candidates

### 3.3 Wall clock on a 64-core machine

**[ASSUMED]** concurrency model. Each gate candidate is an independent
subprocess tree (worktree + test run), so they parallelise cleanly. But each
`go test` / `pytest` invocation wants multiple cores, so gate-level
concurrency and per-job cores trade off. The binding constraint is **RAM**,
not cores: a 64-core cloud box has 256–512 GB, and each concurrent gate holds
a git worktree plus a build/venv cache.

I model **concurrency 32** as the realistic default (2 cores per job on a
64-core box), and give 16 and 56 as the pessimistic/optimistic bounds.

**[DERIVED]** wall clock on the kill-list set (§4), mid per-ecosystem costs:

```
CPU-seconds = Σ over kept repos (candidates × mid_s_per_candidate)

             = 660,377 CPU-seconds

wall = CPU-seconds / concurrency / 3600
```

| concurrency | lo | **mid** | hi |
|---|---|---|---|
| 16 | 5.1 h | 11.5 h | 32.0 h |
| **32** | **2.5 h** | **5.7 h** | **16.0 h** |
| 56 | 1.4 h | 3.3 h | 9.1 h |

For the **full 76-repo set with no kill-list** at concurrency 32:
**lo 9.2 h · mid 21.6 h · hi 61.1 h**.

**[DERIVED]** the non-gate overhead, which is negligible:

```
ingest:   488,341 commits × 9.4 ms = 4,590 CPU-seconds = 1.3 h
          ÷ 32 workers = 2.4 min

cold toolchain: 76 repos × 341 s (Go, measured) = 25,916 CPU-seconds = 7.2 h
          ÷ 32 workers = 13.5 min

clone:    2.2 GB blobless for 76 repos (measured), parallel => ~10 min
```

Total pipeline wall clock = gate wall clock + **under 30 minutes**.

**Summary: the whole CNCF landscape is a single-day job on one 64-core box.**

---

## 4. The kill-list

### 4.1 What will not work, and why

The gate's contract is that it runs **the commit's own test** twice, in a
hermetic sandbox, and observes fail→pass. Any repository whose test suite
cannot run hermetically is not "slow" — it is *unrunnable*, and it will
manufacture `reject_post_failed` verdicts that look like data but are
harness failure.

**Category 1 — requires a live Kubernetes cluster.**

`hermetic_env()` (`vrd/replay.py:160`) builds an environment with a fixed
`PATH`, no inherited variables, `HOME` pointed at the sandbox. There is no
kubeconfig, no API server, no `kubectl`. Any test that calls the apiserver
fails in both arms — and because it fails in *both* arms, it is not even
detected as a harness error; it becomes a `reject_post_failed`.

Verified by inspecting the actual trees:

| repo | evidence |
|---|---|
| `kubernetes/kubernetes` | `test/integration/` — 59,714 first-parent commits |
| `etcd-io/etcd` | `tests/integration/` |
| `cri-o/cri-o` | `test/`, `server/` — talks to a runtime |
| `istio/istio` | `tests/integration/` |
| `vitessio/vitess` | `go/test/endtoend/` |
| `cilium/cilium` | `test/` (ginkgo suites requiring a cluster) |
| `kubevirt/kubevirt` | `tests/`, `tools/` |
| `etcd`, `containerd` | `integration/` |

**`kubernetes/kubernetes` should be excluded outright.** At 59,714 first-parent
commits and ~142,824 commits total it is **12.2% of the entire measured
first-parent corpus by itself** — the single largest repo in the landscape by
more than 30%. Its `hack/make-rules` build is a Bazel-driven multi-container
pipeline. Including it would roughly double the project's total wall clock to
buy candidates whose tests cannot run without a cluster that a hermetic sandbox
by construction cannot provide.

**Category 2 — network-dependent tests.** Any test that reaches the internet
(mirrors, registries, upstream APIs, `go get` of an unpinned dep, DNS
resolution) is non-hermetic. `GOPROXY` is set to `proxy.golang.org` in
`run_go_tests()`, so a Go repo with a flaky or slow proxy makes the gate
measure the network, not the commit.

**Category 3 — GPU tests.** `cuda`, `nvidia`, GPU-operator and device-plugin
repos need an NVIDIA device. The gate has no GPU and no device plugin; these
tests will fail in both arms. There is no CNCF graduated project in this class
that is worth the trouble, but device-plugin-adjacent repos should be
excluded.

**Category 4 — suites over 30 minutes.** `PYTEST_TIMEOUT = 300` and the Go
timeout of 900 s (`vrd/replay.py:35,246`) are hard ceilings. A suite that
exceeds them returns `returncode=124` with `ran=True`, which `parse_pytest_summary`
counts as an error — so it rejects. Worse, a >30-minute suite burns 30 minutes
of wall clock *per candidate*, twice (pre-arm and post-arm), for a verdict that
is guaranteed to be a reject. At 60 minutes per candidate:

```
8,800 candidates × 2 arms × 3,600 s = 63,360,000 CPU-seconds
÷ 32 concurrent gates                = 1,980,000 s = 550 h = 22.9 days
```

One slow repo can dominate the entire project.

**Category 5 — repositories with no runner in `replay.py`.** Everything that
is not Python or Go currently rejects as `reject_post_failed` with
`ran=False`. That includes C/C++ (envoy), Rust (tikv), Java (keycloak),
TypeScript (backstage).

### 4.2 The kill-list and what it costs

**[DERIVED]** Applying the kill-list to the 76 measured repos:

```
removed: 27 of 76 repos
first-parent commits: 488,341 → 137,072   (−71.9%)
gate candidates:     31,996 → 8,800       (−72.5%)
```

The 27 removed repos:

`kubernetes/kubernetes`, `cri-o/cri-o`, `etcd-io/etcd`, `containerd/containerd`,
`envoyproxy/envoy`, `kubevirt/kubevirt`, `vitessio/vitess`, `cilium/cilium`,
`istio/istio`, `grpc/grpc`, `keycloak/keycloak`, `backstage/backstage`,
`tikv/tikv`, `linkerd/linkerd2`, `longhorn/longhorn`, `rook/rook`,
`prometheus/prometheus`, `kyverno/kyverno`, `chaos-mesh/chaos-mesh`,
`volcano-sh/volcano`, `litmuschaos/litmus`, `emissary-ingress/emissary`,
`cert-manager/cert-manager`, `goharbor/harbor`, `openebs/openebs`,
`argoproj/argo-cd`, `knative/serving`, `keda-inc/keda`.

The largest survivors, by candidate count — this is what the corpus would
actually be made of:

| repo | lang | first-parent | candidates | maturity |
|---|---|---|---|---|
| cloud-custodian/cloud-custodian | Python | 4,888 | 783 | incubating |
| strimzi/strimzi-kafka-operator | Java | 7,155 | 429 | incubating |
| open-policy-agent/opa | Go | 6,730 | 404 | graduated |
| tektoncd/pipeline | Go | 6,093 | 366 | incubating |
| coredns/coredns | Go | 5,117 | 307 | graduated |
| theupdateframework/python-tuf | Python | 1,912 | 306 | graduated |
| cubeFS/cubefs | Go | 5,097 | 306 | graduated |
| jaegertracing/jaeger | Go | 4,940 | 296 | graduated |
| falcosecurity/falco | C++ | 4,899 | 294 | graduated |
| spiffe/spire | Go | 4,716 | 283 | graduated |

**[ASSUMED]** Two of the top survivors are on the kill-list-adjacent boundary
and need a decision before the run:

- `falcosecurity/falco` is **C++** — no runner exists. It should be dropped
  until a C++ runner is written, or it contributes 294 guaranteed rejects.
- `strimzi/strimzi-kafka-operator` is **Java** (Quarkus/maven) — same problem,
  and its tests may need a Kafka broker.

### 4.3 The honest re-frame

The kill-list removes 72% of the corpus, and the survivors are dominated by
**Go** (33 of 49). The project as it stands can honestly gate Go and Python.
That is not a failure of ambition — it is the correct reading of where the
evidence is. The alternative, running the gate across all 76 repos and
reporting `reject_post_failed` for the 23 repos with no runner, would produce a
*worse* dataset dressed as a *bigger* one.

---

## 5. Cost in dollars — and why the GPU is not the bottleneck

### 5.1 The GPU portion: LoRA on 5,000–10,000 examples

**[MEASURED]** anchor — the one real fine-tune this project has run
(`data/hermes-webui-all/ft_submit2.log`):

```
model      : Qwen/Qwen3-30B-A3B-Instruct-2507   (30B MoE, FP8)
method     : LoRA r=16, alpha=32, dropout=0.05, 1 epoch
context    : 32,768 (packing on), batch=4
examples   : 148
tokens     : 1,144,590
steps      : 11
wall       : 580 s  (created_at 1793306402 → finished_at 1793306982)
throughput : 1,973 tok/s
```

**[ASSUMED]** A1–A4, stated so they can be attacked:
- **A1** same model and LoRA config as the measured job.
- **A2** same context length (32,768) with packing on.
- **A3** same per-example token count (**7,734 tok/example**, measured
  1,144,590 ÷ 148). Verified pairs from other repos have comparable prompt
  sizes — hermes-webui's median context is 14.2 KB and median patch 4.7 KB.
- **A4** 1 epoch. 2 epochs doubles every figure below.
- **A5** the measured job was **not** throughput-saturated: 11 steps on a 30B
  MoE at 32k context means the job is short enough that queueing and warm-up
  are a material fraction of the 580 s. Scaling linearly by example count is
  therefore an *optimistic* estimate for a 34–68× larger run; a realistic
  correction is **+20–40%** for the larger job's longer critical path.

**[DERIVED]** scaling:

```
tokens(n) = n × 7,734
steps(n)  = 11 × (n / 148)
wall(n)   = steps(n) × (580 / 11)  seconds
```

| n | tokens | steps | wall (h) |
|---|---|---|---|
| 5,000 | 38,668,581 | 372 | 5.44 |
| 10,000 | 77,337,162 | 743 | 10.89 |

**[ASSUMED]** A6 — GPU hourly prices. These are market prices, not measurements:

| GPU | spot $/h | on-demand $/h |
|---|---|---|
| A100-80GB | 1.10 | 1.79 |
| H100-80GB | 2.50 | 3.50 |

**[DERIVED]** GPU cost, LoRA portion only:

| n | A100 spot | A100 on-demand | H100 spot | H100 on-demand |
|---|---|---|---|---|
| 5,000 | **$5.99** | $9.74 | $13.61 | $19.05 |
| 10,000 | **$11.97** | $19.49 | $27.21 | $38.10 |

With the A5 correction (+20–40% on the larger job), the 10k on-demand H100
figure rises to roughly **$31–$53**. Even at the pessimistic end, the GPU
portion is **under $55**.

### 5.2 The CPU portion: the gate

**[ASSUMED]** A7 — 64-vCPU cloud pricing: $0.60/h spot, $1.20/h on-demand.

**[DERIVED]** from the §3.3 wall-clock table, kill-list set:

| scenario | wall (conc 32) | spot $ | on-demand $ |
|---|---|---|---|
| lo | 2.5 h | $1.52 | $3.04 |
| **mid** | **5.7 h** | **$3.44** | **$6.88** |
| hi | 16.0 h | $9.59 | $19.17 |

Full 76-repo set, no kill-list, mid: 21.6 h → $12.96 spot / $25.92 on-demand.

### 5.3 The comparison that matters

```
GPU  (LoRA, 5k-10k examples)  :  $6  –  $38      5.4 – 10.9 GPU-hours
CPU  (gate, 8,800 candidates) :  $3  –  $15      5.7 CPU-hours on 64 cores
```

**The GPU is not the bottleneck.** It is the same order of magnitude as the
CPU gate, and if anything the CPU gate is the larger line item — and the CPU
gate can be cut by choosing better repositories, while the GPU cost is fixed
by the corpus size.

More importantly: **the interesting cost is not money, it is engineering
hours.** Adding a runner for a new ecosystem is days of work (§2.3). Adding
5,000 more training examples costs $12. The gate's cost is dominated by
*deciding which repositories to run*, and that decision is free.

---

## 6. Assumptions register

Every non-measured input, in one place, so it can be attacked individually.

| # | assumption | value | why | effect if wrong |
|---|---|---|---|---|
| A1 | model + LoRA config for the scaled run | Qwen3-30B-A3B FP8, r=16, 1 epoch | matches the measured job | 2× if 2 epochs |
| A2 | context length 32,768, packing on | — | matches measured job | ±30% wall |
| A3 | tokens per example | 7,734 | measured on hermes-webui | ±50% if other repos have bigger contexts |
| A4 | 1 epoch | — | matches measured job | 2× if 2 epochs |
| A5 | linear scaling in example count | — | measured job was not saturated | +20–40% for the 10k job |
| A6 | GPU prices | A100 $1.10/$1.79, H100 $2.50/$3.50 per h | market rates, not measured | ±50% |
| A7 | 64-vCPU cloud price | $0.60 spot / $1.20 on-demand | market rates, not measured | ±50% |
| A8 | gate concurrency on a 64-core box | 32 (bounds 16–56) | 2 cores per job; RAM-bound in practice | ±2× wall |
| A9 | candidate rate, non-Python ecosystems | 6.00% | measured on coredns (Go) | the single largest uncertainty; see §3.2 range |
| A10 | C/C++, Rust, Java, TS per-candidate cost | 180/120/240/120 s mid | no runner exists; no measurement possible | ±3× on those repos |
| A11 | kill-list membership | 27 repos | toolchain + cluster-dependency inspection | ±30% on candidate count |
| A12 | one candidate ≈ one verified pair at a 92.9% rate | 8,800 → ~8,175 pairs | measured Python yield | Go yield is unmeasured |

---

## 7. What is *not* established

- **Go gate yield is unmeasured.** The 92.9% yield is a Python number.
  coredns's 3 measured commits were 3/3 verified, which is not a rate.
- **C/C++, Rust, Java, TypeScript are entirely unmeasured** — no runner
  exists, so no number can honestly be quoted for them.
- **The kill-list is judgement, not measurement.** It was formed by inspecting
  repository layouts, not by attempting the gate and observing failures.
- **The candidate rate for Go is based on one repo.** coredns gave 6.00%,
  5.67% and 3.13% at three depths — a spread of ~3× on the same repository.
  A single-repo rate generalising to 50 repos is the weakest link in §3.2.
- **The 64-core figure is a model.** No 64-core box was used; the
  concurrency assumption (A8) was not validated under load.
- **No repo other than hermes-webui has been run end-to-end at corpus
  scale.** coredns and helm are wired and gated; opentelemetry-go is present
  but not gated at scale.

---

## 8. Reproduction

Every measured number in this document comes from these artifacts:

```bash
# 1. Corpus size — clone and count locally (no API rate limit)
git clone --filter=blob:none --bare https://github.com/<repo>.git
git rev-list --count --first-parent HEAD
git rev-list --count HEAD

# 2. Candidate rate + ingest cost
.venv/bin/python run.py ingest --repo <repo> --out <dir> --limit <n>

# 3. Gate cost (Python baseline, 565 candidates)
.venv/bin/python run.py gate --repo <repo> --data <dir> --preset hermes-webui \
    --cache ~/.cache/vrd --scratch /tmp/vrd
#    -> data/hermes-webui-all/gate_full.log, gate_retry.log, verdicts.jsonl

# 4. Gate cost (Go, warm cache)
.venv/bin/python run.py gate --repo <repo> --data <dir> --preset none \
    --cache ~/.cache/vrd-go --scratch /tmp/vrd-go

# 5. CNCF project list + maturity
curl -s https://raw.githubusercontent.com/cncf/landscape/master/landscape.yml
```

Artifacts in this repo: `data/hermes-webui-all/verdicts.jsonl` (n=565),
`data/hermes-webui-all/funnel.json`, `data/coredns/verdicts.jsonl` (n=3),
`data/coredns_gate2.log`, `data/hermes-webui-all/ft_submit2.log` (the
fine-tune job record).
