# SCALING.md — cost of running the verified-corpus gate across CNCF

**Scope.** What does it cost to run the counterfactual gate (stages 2+3) over
the CNCF landscape, and where does the money actually go?

**Date.** 2026-10-30. **Repo:** `verified-repo-distillation`.

Every number is tagged **[MEASURED]** (a run in this repo, artifact named),
**[DERIVED]** (arithmetic on measured values, shown), or **[ASSUMED]** (a
stated modelling choice with no measurement behind it). Assumptions are written
next to the number they affect so each can be attacked independently.

---

## 0. Executive summary

| quantity | value | basis |
|---|---|---|
| Gate cost, Python, 2-dep repo | **9.55 s/candidate** | [MEASURED] `verdicts.jsonl` n=565 |
| Gate yield, Python | **525 / 565 = 92.9%** verified | [MEASURED] same file |
| Gate cost, Go, warm cache | **51.7 s mean / 54.9 s median** (n=7) | [MEASURED] 3 coredns runs |
| Gate cost, Go, cold cache | **341–420 s** (one commit) | [MEASURED] `coredns_gate2.log` + a rerun |
| Ingest cost | **9.4 ms/commit** | [MEASURED] 1600-commit coredns ingest |
| CNCF corpus | **488,341 first-parent commits**, 76 repos | [MEASURED] `git rev-list --count --first-parent` |
| Gate candidates, all 76 repos | **≈32,000** | [DERIVED] §3.2 |
| Gate candidates, post-kill-list | **≈8,800** | [DERIVED] §4.2 |
| Wall clock, 64-core, kill-list, mid | **5.2 h** @ concurrency 32 | [DERIVED] §3.3 |
| Wall clock, 64-core, no kill-list, mid | **19.8 h** @ concurrency 32 | [DERIVED] §3.3 |
| **GPU (LoRA, 5k–10k examples)** | **$6–$38** | [DERIVED] §5 |
| **CPU (the gate)** | **$2–$19** | [DERIVED] §5 |

**The headline: the gate is a CPU-bound, embarrassingly-parallel, hours-scale
job. The GPU is a rounding error.** The constraint is not compute — it is
*which repositories you are willing to touch*.

Two corrections to numbers this project already publishes, made by
re-measuring rather than by re-reading: **the Go cost is 51.7 s, not 173 s**
(§1.2) and **the Python cost is 9.55 s, not 9.2 s** (§1.1).


---

## 1. Measured baseline

### 1.1 The Python baseline (the reference number)

**[MEASURED]** `hermes-webui`, a Python repo with two runtime dependencies:
3,526 first-parent commits (Mar–May 2026), 1,813 kept after classification,
**565 gate candidates**.

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
| median | 8.71 s |
| min / max | 0.00 s / 32.99 s |
| mean, `verified` / `reject_post_failed` rows | 9.49 s / 11.28 s |
| **total CPU time for all 565** | **1.50 h** |

**A note on 9.2 s vs 9.55 s.** The `9.2 s/commit` figure in `REPORT.md` and
`VALIDATION.md` is the mean of the **first** pass
(`verdicts_pass1.jsonl`, mean 9.16 s), which produced **50 `reject_replay`
rows** — all the harness bug where `git show --patch` silently drops binary
payloads without `--binary --full-index` (`vrd/ingest.py:184`). Those 50 were
re-run (`gate_retry.log`) and 47 recovered, giving the final file above. The
honest production number is **9.55 s**; the retry added ~7.7 min (~8%
overhead) for a bug that is now fixed.

The 9.55 s is **inclusive** of venv construction: `build_venv()`
(`vrd/replay.py:111`) keys a venv on the dependency-file content hash and
reuses it for sibling commits. The `verified` mean (9.49 s) is barely above the
median (8.71 s), so venv build is *not* the bulk — the bulk is two test
invocations plus two worktree checkouts (§2.1).

### 1.2 The Go baseline (and a correction to a published number)

`VALIDATION.md` and `README.md` quote coredns at **"~173 s/commit"**
(`data/coredns_gate2.log` prints `mean wall/commit: 173.0s`). That number is
**real but misleading**: it is the mean over 3 commits, one of which was a
cold-cache commit that spent 392 s downloading Go modules.

**[MEASURED]** Three independent runs on coredns, same repo, same cache root:

| run | cache state | commit 1 | commit 2 | commit 3 | mean |
|---|---|---|---|---|---|
| A (`data/coredns_gate2.log`) | cold | 419.94 s | 12.55 s | 86.56 s | 173.0 s |
| B (re-run) | module cache warming | 341.0 s | 45.7 s | 84.9 s | 157.2 s |
| C (re-run) | **fully warm** | 54.9 s | 14.2 s | 62.9 s | 44.0 s |

Commit 1 in runs A and B is the cold-cache commit: its pre-arm took 392.81 s
against 5.73 s and 45.7 s for warm commits, with an output tail entirely
`go: downloading github.com/go-openapi/...`. The 392 s is the **pre-arm
total**, not download time in isolation (downloads interleave with the test
run); what is unambiguous is that the cold commit is 3–8× the warm ones.

| population | n | mean | median |
|---|---|---|---|
| all nine commits | 9 | 124.7 s | 62.9 s |
| **warm-cache commits only** | **7** | **51.7 s** | **54.9 s** |

**I use 51.7 s (warm mean) / 54.9 s (warm median) as the Go per-candidate
cost, and 341–420 s as a one-time per-repo cold-cache cost.** Quoting 173 s
overstates it by 3.4×, which would have made the whole projection 3.4× too
expensive.


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

From `counterfactual_verify()` (`vrd/replay.py:347`), per candidate: add a
`git worktree` of the parent → inject the commit's new test files
(`git show <sha>:<path>`) → **pre-arm** run them (must FAIL) → `git apply` the
patch → **post-arm** run the same tests (must PASS) → remove the worktree.

So the unit cost is **2 × (worktree + test run)**, not 1 ×. Any per-test
number must already be read as "two test invocations". The Python 9.55 s
includes both.

There is also a **fixed per-repo** cost, separate from the per-candidate cost:

- **module/dependency download.** The cold candidate's pre-arm took 392.81 s
  (run A) and the whole candidate 341.0 s (run B), against 5.7–45.7 s for warm
  ones. Paid roughly **once per repo**, and *not* eliminated by a shared
  `GOMODCACHE` because `go.mod` changes over time and each change can pull new
  module versions.
- **venv construction** for Python, keyed on the dependency-file content hash:
  ~2–4 s for a 2-dependency repo, 60–180 s for a 100-dependency repo,
  amortised across every sibling commit sharing that hash.

### 2.2 The per-ecosystem table

**[MEASURED]** rows are flagged. Everything else is **[ASSUMED]** with the
toolchain reason given.

| ecosystem | lo | mid | hi | s/candidate | justification |
|---|---|---|---|---|---|
| **Python** | 8 | **10** | 45 | **[MEASURED]** lo/mid (9.55 mean, 8.71 median) | 2 deps, `uv` venv cached by dep-hash, pytest only. hi covers a 100-dep repo whose venv rebuilds often. |
| **Go** | 30 | **55** | 180 | **[MEASURED]** mid = warm median 54.9 s (n=7); lo = warm mean 51.7 s rounded down; hi = a repo with heavy `go: downloading` churn | `GOTOOLCHAIN=auto` (`vrd/replay.py:265`) pulls a new toolchain when `go.mod` requires one — measured: pinning `local` made every coredns commit fail with `go.mod requires go >= 1.26.0`, a harness artefact that looked like a test failure. Shared `GOMODCACHE`, per-sandbox `GOCACHE`. |
| **C / C++** | 60 | **180** | 600 | [ASSUMED] | No runner exists in `replay.py` — `detect_toolchain()` (`vrd/replay.py:219`) returns `"unknown"` for a repo with no `go.mod`/`pyproject.toml`/`package.json`, and `run_tests()` returns `ran=False`. Supporting C++ means writing a runner *and* handling cmake + compiler + cgo, with a full relink per commit. **`protoc` adds a codegen step**: envoy has **904 `.proto` files** at HEAD (kubernetes 136, cilium 129, grpc 166), so a C++ runner must either run `protoc` per commit or pin generated sources — the former is another per-commit cost, the latter means the injected test may reference symbols that do not exist yet, which the gate would read as a build failure rather than a test failure. Envoy alone is 27,667 first-parent commits. |
| **Rust** | 45 | **120** | 400 | [ASSUMED] | Same "no runner" problem, plus `cargo build` per commit against a cold `~/.cargo` registry, and `tikv/tikv` has a Rust `tests/` crate. |
| **Java** | 90 | **240** | 600 | [ASSUMED] | Same "no runner" problem, plus maven/gradle with a cold `~/.m2` cache, JVM warm-up per invocation, and surefire forking a JVM per test class. Keycloak is a Quarkus/maven project with 20,570 first-parent commits. |
| **TypeScript / JS** | 40 | **120** | 300 | [ASSUMED] | `package.json` is detected by `detect_toolchain()` but the dispatcher returns `ran=False` for `"node"` — there is **no Node runner**. Adding one means `npm ci` per commit plus jest/tsc. Backstage is 22,411 first-parent commits. |

**Caveat on the "mid" column.** The Python and Go rows are grounded in real
runs. The C/C++, Rust, Java and TypeScript rows are **modelling assumptions** —
this pipeline has never gated a non-Python, non-Go commit. They are stated so
they can be replaced with measurements, not dressed up as facts.

### 2.3 Why "no runner" is a hard blocker, not a config tweak

`run_tests()` (`vrd/replay.py:301`) dispatches on `detect_toolchain()` and has
exactly two real branches — `go` → `run_go_tests()`, `python` → `run_pytest()`,
everything else → `TestResult(ran=False, ...)`.

A `"node"` tree falls through to the `ran=False` branch, and `TestResult.green`
requires `ran and returncode == 0 and failed == 0 and errors == 0` — a
`ran=False` result is never green. So **every candidate in a Node, C++, Rust
or Java repo currently rejects as `reject_post_failed`** with no useful signal.
Adding an ecosystem is a real engineering task (build-system call, hermetic
dependency cache, output parser), not a config line.

---

## 3. CNCF-wide sizing

### 3.1 The corpus: measured, not estimated

**[MEASURED]** The CNCF landscape (`cncf/landscape`, `landscape.yml`, parsed
from the upstream source) lists **260** CNCF-governed projects with an
`accepted` date: **39 graduated**, **38 incubating**, **153 sandbox**,
**27 archived**, 3 uncategorised.

The brief says "~100 graduated projects". The authoritative upstream list says
**39 graduated** and **38 incubating** = **77 non-sandbox projects**. I cloned
and measured **76 of those 77** (the exception, `meshery/meshery`, is a large
TypeScript monorepo whose clone did not finish; by GitHub's commit API it has
~60,700 commits, so excluding it is conservative for a repo with no runner) as
blobless bare clones, counting commits locally with the same git plumbing the
pipeline uses:

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

Language mix (dominant source extension at HEAD): **53 Go, 4 Python, 3 Java,
2 C++, 2 Rust, 2 TypeScript**, 7 doc-only, 5 other.

### 3.2 Total commits vs gate candidates

These are very different numbers; conflating them is the easiest way to
over-estimate this project by 15×. A **gate candidate** requires
(`vrd/ingest.py:251`) `commit.added_tests and commit.source_paths` — the
commit *adds* a test file **and** touches non-test, non-doc source.

**[MEASURED]** candidate rates:

| repo | lang | first-parent commits | candidates | rate |
|---|---|---|---|---|
| hermes-webui | Python | 3,526 | 565 | **16.02%** |
| coredns (400-commit slice) | Go | 400 | 24 | **6.00%** |
| coredns (300-commit slice) | Go | 300 | 17 | **5.67%** |
| coredns (1,600-commit slice) | Go | 1,600 | 50 | **3.13%** |

The brief's "565 of 3526 = 16%" is **exactly right for hermes-webui**. But
coredns — a Go repo of comparable age and size — is **3–6%**, a factor of
2.7–5 lower. The Python rate is not transferable to Go, and these two repos are
the only data points that exist.

**[DERIVED]** 16.02% for the 4 Python repos (measured), 6.00% for the other 72
(Go is measured at 6%; unmeasured ecosystems are assumed no better than Go):

`candidates = Σ(first-parent commits × rate) = 31,996 ≈ 32,000`.

| population | first-parent commits | candidates |
|---|---|---|
| Graduated | 256,475 | 15,513 |
| Incubating | 157,514 | 12,022 |
| **All 76** | **488,341** | **≈32,000** |

**Range** (wide, because the per-language rates are assumptions):

| case | rates applied | candidates |
|---|---|---|
| Low | 3.13% everywhere (coredns' own 1,600-commit rate) | ≈15,300 |
| **Mid** | 16.02% Python, 6.00% other | **≈32,000** |
| High | 16.02% everywhere | ≈78,200 |

### 3.3 Wall clock on a 64-core machine

**[ASSUMED]** concurrency model. Each candidate is an independent subprocess
tree, so they parallelise cleanly — but each `go test` / `pytest` wants
multiple cores, so gate-level concurrency and per-job cores trade off. The
binding constraint is **RAM**, not cores: a 64-core cloud box has 256–512 GB,
and each concurrent gate holds a worktree plus a build/venv cache.

I model **concurrency 32** as the realistic default (2 cores per job), with 16
and 56 as pessimistic/optimistic bounds.

**[DERIVED]** wall clock on the kill-list set (§4), mid per-ecosystem costs:
`CPU-seconds = Σ (candidates × mid_s_per_candidate) = 598,067`;
`wall = CPU-seconds / concurrency / 3600`.

| concurrency | lo | **mid** | hi |
|---|---|---|---|
| 16 | 5.1 h | 10.4 h | 32.0 h |
| **32** | **2.5 h** | **5.2 h** | **16.0 h** |
| 56 | 1.4 h | 3.0 h | 9.1 h |

For the **full 76-repo set with no kill-list** at concurrency 32:
**lo 9.2 h · mid 19.8 h · hi 61.1 h**.

**[DERIVED]** the non-gate overhead, which is negligible:

```
ingest          488,341 commits × 9.4 ms          =  4,590 CPU-s ÷ 32 =  2.4 min
cold toolchain  76 repos × 341 s (Go, measured)   = 25,916 CPU-s ÷ 32 = 13.5 min
clone           2.2 GB blobless, 76 repos (measured)          parallel ≈ 10 min
```

Total pipeline wall clock = gate wall clock + **under 30 minutes**. The whole
CNCF landscape is a single-day job on one 64-core box.

---

## 4. The kill-list

### 4.1 What will not work, and why

The gate's contract is that it runs **the commit's own test** twice, in a
hermetic sandbox, and observes fail→pass. A repository whose suite cannot run
hermetically is not "slow" — it is *unrunnable*, and it manufactures
`reject_post_failed` verdicts that look like data but are harness failure.

**Category 1 — requires a live Kubernetes cluster.**

`hermetic_env()` (`vrd/replay.py:160`) builds an environment with a fixed
`PATH`, no inherited variables, `HOME` in the sandbox. There is no kubeconfig,
no API server, no `kubectl`. A test that calls the apiserver fails in *both*
arms — so it is not even detected as a harness error; it becomes a
`reject_post_failed`. Verified by inspecting the actual trees:

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
commits (~142,824 total) it is **12.2% of the entire measured first-parent
corpus by itself** — the largest repo in the landscape by more than 30%. Its
`hack/make-rules` build is a Bazel-driven multi-container pipeline. Including
it would roughly double the project's wall clock to buy candidates whose tests
cannot run without a cluster a hermetic sandbox by construction cannot
provide.

**Category 2 — network-dependent tests.** Any test reaching the internet
(mirrors, registries, upstream APIs, unpinned `go get`, DNS) is non-hermetic.
`GOPROXY` is `proxy.golang.org` in `run_go_tests()`, so a Go repo with a slow
proxy makes the gate measure the network, not the commit.

**Category 3 — GPU tests.** `cuda`, `nvidia`, GPU-operator and device-plugin
repos need an NVIDIA device; the gate has no GPU, so these fail in both arms.
No CNCF graduated project in this class is worth the trouble, but
device-plugin-adjacent repos should be excluded.

**Category 4 — suites over 30 minutes.** `PYTEST_TIMEOUT = 300` and the Go
timeout of 900 s (`vrd/replay.py:35,246`) are hard ceilings; exceeding them
returns `returncode=124` with `ran=True`, which `parse_pytest_summary` counts
as an error, so it rejects. Worse, a 30-minute suite burns 30 min *per
candidate*, twice (pre-arm and post-arm), for a guaranteed reject. At 60 min:

`8,800 × 2 arms × 3,600 s = 63,360,000 CPU-s ÷ 32 = 550 h = 22.9 days`.

One slow repo can dominate the entire project.

**Category 5 — no runner in `replay.py`.** Everything that is not Python or Go
rejects as `reject_post_failed` with `ran=False`: C/C++ (envoy), Rust (tikv),
Java (keycloak), TypeScript (backstage).

### 4.2 The kill-list and what it costs

**[DERIVED]** Applying the kill-list to the 76 measured repos: **27 of 76
repos removed**; first-parent commits 488,341 → 137,072 (−71.9%); gate
candidates 31,996 → 8,800 (−72.5%).

The 27 removed repos:

`kubernetes/kubernetes`, `cri-o/cri-o`, `etcd-io/etcd`, `containerd/containerd`,
`envoyproxy/envoy`, `kubevirt/kubevirt`, `vitessio/vitess`, `cilium/cilium`,
`istio/istio`, `grpc/grpc`, `keycloak/keycloak`, `backstage/backstage`,
`tikv/tikv`, `linkerd/linkerd2`, `longhorn/longhorn`, `rook/rook`,
`prometheus/prometheus`, `kyverno/kyverno`, `chaos-mesh/chaos-mesh`,
`volcano-sh/volcano`, `litmuschaos/litmus`, `emissary-ingress/emissary`,
`cert-manager/cert-manager`, `goharbor/harbor`, `openebs/openebs`,
`argoproj/argo-cd`, `knative/serving`, `keda-inc/keda`.

The largest survivors by candidate count — what the corpus would actually be
made of:

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

`falcosecurity/falco` (C++) and `strimzi/strimzi-kafka-operator` (Java,
Quarkus/maven, may need a Kafka broker) are the two that need a decision
before the run — see §4.3.

### 4.3 The honest re-frame

The kill-list removes 72% of the corpus, and the survivors are dominated by
**Go** (33 of 49) and Python (3 of 49). The project as it stands can honestly
gate Go and Python. That is not a failure of ambition — it is the correct
reading of where the evidence is. The alternative, running the gate across all
76 repos and reporting `reject_post_failed` for the 19 repos with no runner,
would produce a *worse* dataset dressed as a *bigger* one.

**[DERIVED]** the residual the kill-list does *not* remove — survivors with no
runner: strimzi/strimzi-kafka-operator (Java, 429), falcosecurity/falco
(C++, 294), fluent/fluentd (Ruby, 168), kubeflow/kubeflow (mixed, 151),
wasmCloud/wasmCloud (Rust, 125), artifacthub/hub (TypeScript, 121),
microcks/microcks (Java, 82), plus 6 documentation-only repos (0).

**1,370 of 8,800 surviving candidates (15.6%) have no runner** — they are
guaranteed `reject_post_failed`, not data.

**Recommendation:** add these 7 to the kill-list, or write the runners. The
second is worth more: 15.6% more corpus, and a Java runner is one `mvn -o test`
invocation plus a surefire XML parser.


---

## 5. Cost in dollars — and why the GPU is not the bottleneck

### 5.1 The GPU portion: LoRA on 5,000–10,000 examples

**[MEASURED]** anchor — the one real fine-tune this project has run
(`data/hermes-webui-all/ft_submit2.log`):

```
model **Qwen/Qwen3-30B-A3B-Instruct-2507** (30B MoE, FP8); LoRA r=16,
alpha=32, dropout=0.05, 1 epoch; context 32,768 (packing on), batch=4;
**148 examples, 1,144,590 tokens, 11 steps, 580 s** (created_at 1793306402 →
finished_at 1793306982) = **1,973 tok/s**.
```

**[ASSUMED]** A1–A5, stated so they can be attacked:

- **A1** same model and LoRA config as the measured job. **A2** same context
  length (32,768), packing on.
- **A3** same per-example token count (**7,734 tok/example**, measured
  1,144,590 ÷ 148). Verified pairs from other repos have comparable prompt
  sizes — hermes-webui's median context is 14.2 KB, median patch 4.7 KB.
- **A4** 1 epoch. 2 epochs doubles every figure below.
- **A5** the measured job was **not** throughput-saturated: 11 steps on a 30B
  MoE at 32k context is short enough that queueing and warm-up are a material
  fraction of the 580 s. Linear scaling is therefore *optimistic* for a
  34–68× larger run; a realistic correction is **+20–40%**.

**[DERIVED]** scaling: `tokens(n) = n × 7,734`;
`steps(n) = 11 × (n / 148)`; `wall(n) = steps(n) × (580 / 11)` seconds.

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
| **mid** | **5.2 h** | **$3.11** | **$6.23** |
| hi | 16.0 h | $9.59 | $19.17 |

Full 76-repo set, no kill-list, mid: 19.8 h → $11.87 spot / $23.73 on-demand.

### 5.3 The comparison that matters

```
GPU  (LoRA, 5k-10k examples)  :  $6  –  $38      5.4 – 10.9 GPU-hours
CPU  (gate, 8,800 candidates) :  $2  –  $19      2.5 – 16.0 CPU-hours on 64 cores
```

**The GPU is not the bottleneck.** They are the same order of magnitude, and at
the mid case the CPU gate ($3.11) is *cheaper* than the LoRA run ($11.97 for
10k examples on A100 spot). And the asymmetry matters: the CPU gate's cost can
be cut by choosing better repositories, while the GPU cost is fixed by the
corpus size you decide to train on.

More importantly: **the interesting cost is not money, it is engineering
hours.** Adding a runner for a new ecosystem is days of work (§2.3); adding
5,000 more training examples costs $12. The gate's cost is dominated by
*deciding which repositories to run* — and that decision is free.

---

## 6. Assumptions register

Every non-measured input, in one place, so each can be attacked individually.

| # | assumption | value | why | effect if wrong |
|---|---|---|---|---|
| A1 | model + LoRA config for the scaled run | Qwen3-30B-A3B FP8, r=16, 1 epoch | matches the measured job | 2× if 2 epochs |
| A2 | context length 32,768, packing on | — | matches measured job | ±30% wall |
| A3 | tokens per example | 7,734 | **measured** on hermes-webui (1,144,590 ÷ 148); *assumed* to transfer to other repos | ±50% if other repos have bigger contexts |
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

- **Go gate yield is unmeasured.** The 92.9% yield is a Python number. On
  coredns, 3 *unique* commits (9 gate invocations across runs A–C) were all
  verified — 3/3 is not a rate.
- **C/C++, Rust, Java, TypeScript are entirely unmeasured** — no runner exists,
  so no number can honestly be quoted for them.
- **The kill-list is judgement, not measurement**: formed by inspecting
  repository layouts, not by attempting the gate and observing failures.
- **The Go candidate rate rests on one repo.** coredns gave 6.00%, 5.67% and
  3.13% at three depths — a ~3× spread on the same repository. Generalising a
  single-repo rate to 50 repos is the weakest link in §3.2.
- **The 64-core figure is a model.** No 64-core box was used; the concurrency
  assumption (A8) was not validated under load.
- **No repo other than hermes-webui has been run end-to-end at corpus scale.**
  coredns and helm are wired and gated; opentelemetry-go is present but not
  gated at scale.

---

## 8. Reproduction

Every measured number here comes from these artifacts and commands:

```bash
# 1. Corpus size — clone and count locally (no API rate limit)
git clone --filter=blob:none --bare https://github.com/<repo>.git
git rev-list --count --first-parent HEAD

# 2. Candidate rate + ingest cost
.venv/bin/python run.py ingest --repo <repo> --out <dir> --limit <n>

# 3. Gate cost (Python baseline, 565 candidates)
.venv/bin/python run.py gate --repo <repo> --data <dir> --preset hermes-webui \
    --cache ~/.cache/vrd --scratch /tmp/vrd
#    -> data/hermes-webui-all/{gate_full.log,gate_retry.log,verdicts.jsonl}

# 4. Gate cost (Go, warm cache)
.venv/bin/python run.py gate --repo <repo> --data <dir> --preset none \
    --cache ~/.cache/vrd-go --scratch /tmp/vrd-go

# 5. CNCF project list + maturity
curl -s https://raw.githubusercontent.com/cncf/landscape/master/landscape.yml
```

Artifacts: `data/hermes-webui-all/verdicts.jsonl` (n=565),
`data/hermes-webui-all/funnel.json`, `data/coredns/verdicts.jsonl` (n=3),
`data/coredns_gate2.log`, `data/hermes-webui-all/ft_submit2.log` (the
fine-tune job record).
