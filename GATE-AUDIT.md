# Gate audit: how precise is the two-sided gate?

Date: 2026-10-30
Corpus: `hermes-webui` (3526 first-parent commits → 565 gate candidates → 525 "verified")
Audit sample: 40 of the 525, uniformly sampled with a fixed seed (`--seed 7`)
Result file: `data/hermes-webui-all/audit_strict.json`

---

## 1. The problem with the rule we shipped

The original gate accepted a commit when the test suite was **red at the parent
commit** and **green after the patch applied**:

```
pre-arm  : inject the commit's new tests into the parent tree   -> must FAIL
post-arm : apply the commit's patch, same tests                 -> must PASS
```

That is a suite-level rule, and it cannot distinguish a real fix from three
things that look identical to it:

1. **A flaky test** that happened to fail during the pre-arm run. Nothing about
   the patch caused the red; re-running would have gone green anyway.
2. **An unrelated red.** The suite failed for a reason the commit does not
   touch (import error in a sibling module, missing fixture, stale cache), and
   the patch fixed that instead.
3. **A fix that also breaks something.** The target test goes green, a
   previously-passing test goes red, and the suite still exits non-zero — or,
   worse, the broken test is skipped and the exit code is clean.

Every one of those produces a training pair that teaches "this context maps to
this patch" when the patch was never the cause. This is the same assumption
failure the corpus exists to correct, one level up: we were trusting an exit
code instead of a named transition.

## 2. What industry does instead

SWE-bench (`arXiv:2310.06770`, Jimenez et al., ICLR 2024) does not accept a
task because a suite went from red to green. It records two explicit sets per
instance:

- **FAIL_TO_PASS** — tests that fail before the patch and pass after it.
  These are the tests the patch is credited with fixing.
- **PASS_TO_PASS** — tests that pass before and must still pass after.

A patch is scored correct only when every test in FAIL_TO_PASS passes and every
test in PASS_TO_PASS still passes. The unit of evidence is a *named test*, not a
process exit code, and the second set exists to catch regressions.

The motivation for worrying about flakiness is not hypothetical. Luo et al.
(FSE 2014, "An empirical analysis of flaky tests") studied 201 commits that fix
flaky tests across 51 open-source projects, and found non-deterministic outcomes
common enough to undermine regression testing as a signal.

## 3. The stricter gate

Implemented in `vrd/replay.py`; exposed as `run.py gate --strict --repeats N`.

Per commit, for the committed test files:

1. Run the pre-arm and post-arm with `--junitxml`, and parse **per-test**
   outcomes (`parse_junit`). Missing or malformed XML yields "no data", never a
   silent pass.
2. Compute, via the pure function `classify_transitions(pre, post)`:
   - `fail_to_pass` — named tests that were `failed`/`error` and are now `passed`
   - `pass_to_pass` — tests green in both arms (count only)
   - `regressions` — tests green before, red after
3. Re-run the post-arm `--repeats-1` additional times and diff the outcome maps.
   Any test whose outcome is not stable across identical runs is recorded as
   `flaky`.
4. Verdict:

| condition | verdict |
|---|---|
| pre-arm green | `reject_pre_passed` |
| post-arm not green | `reject_post_failed` |
| unstable tests (strict) | `reject_flaky` |
| no named fail→pass transition (strict) | `reject_no_transition` |
| any pass→pass regression (strict) | `reject_regression` |
| otherwise | `verified` |

The pure classification function is covered by 8 unit tests
(`tests/test_strict_gate.py`), including the case that matters most: a suite
that is green after the patch but where nothing can be shown to have moved.

## 4. Audit of the already-accepted set

Rather than assert the old gate was fine, we replayed a sample of the commits it
had already accepted, under the strict rule, with `--repeats 2`.

```
sampled              : 40
survived strict gate : 36 (90.0%)   Wilson 95% CI [76.9%, 96.0%]
  verified               36
  reject_no_transition    2
  reject_post_failed      1
  reject_flaky            1
mean wall per commit : 24.4s   (vs 9.2s for the suite-level rule)
total audit time     : 976s
```

### The four false pairs

| commit | strict verdict | why |
|---|---|---|
| `b2771ebf` | `reject_no_transition` | Suite went red→green, but no named test can be shown to have flipped. Nothing to credit to the patch. |
| `c616c8e7` | `reject_no_transition` | Same: green after, zero fail→pass transitions. |
| `5fc7aee7` | `reject_post_failed` | Re-verified today, the post-arm does not pass at all. The original verdict does not reproduce. |
| `3fd20599` | `reject_flaky` | **4 tests flip outcome between identical runs**, including the 2 that had been counted as the fail→pass evidence. The "fix" is a coin flip. |

The flaky case is the important one. Under the old rule it was a textbook
verified pair: the tests failed at the parent, passed after the patch. Under
repeated execution, the same tests in the same tree disagree with themselves, so
the patch was never the cause of the transition. That is exactly the class of
noise a corpus like this must not contain, and it is only visible if you run the
thing more than once.

## 5. What this changes

- **Claim strength.** "525 verified" should be read as "525 accepted by the
  suite-level rule, ~90% [77, 96] of which survive a per-test, flakiness-checked
  rule". The honest headline is **90% gate precision on a 40-sample audit**, not
  100%.
- **Recommended default.** Use `--strict --repeats 2` for any corpus that will
  train a model. The cost is 2.7× wall clock (24.4s vs 9.2s per commit on this
  repo); the cost of shipping flaky pairs into a training set is paid later and
  invisibly.
- **Cost of the audit itself.** 40 commits, 16 minutes, no GPU, no paid API.
  Re-running the whole 565 with `--strict --repeats 2` is ~3.8 hours on one
  core, which is the honest price of a corpus you can defend.

## 6. Reproducibility: can the gate reproduce itself?

An audit that cannot be re-run is an anecdote. Two things were missing:

**Determinism controls.** `hermetic_env` pinned PATH and HOME but not the two
things that actually move test outcomes: `PYTHONHASHSEED` (iteration order of
sets and dicts, which leaks into anything that serialises them) and the clock
(`TZ`, `SOURCE_DATE_EPOCH`). Both are now pinned by default, with `NO_COLOR` and
`TERM=dumb` so terminal-aware test code cannot change behaviour. A test that only
becomes unstable when the hash seed varies is a real flake this probe cannot see;
varying the seed across repeats is a deliberate, more expensive mode and is not
the default.

**Verdict drift measurement.** `scripts/reverify.py` replays commits whose verdict
is already stored, compares the fresh result against the stored one, and reports
every disagreement with the per-test evidence behind it. This is the check that
matters for a corpus other people will train on: if `verdicts.jsonl` says a commit
was verified and a re-run disagrees, the corpus is not reproducible and its size
is not a fact.

It has already earned its keep. During the strict audit, commit `5fc7aee7` — recorded
as `verified` — came back `reject_post_failed` on re-run. A verdict that changes
between runs means the recorded status was never a property of the commit alone.

The dedicated drift survey (independent 40-commit sample, different seed from the
strict audit) measured it:

```
agreement: 36/40 = 90.0%   Wilson 95% CI [76.9, 96.0]
transitions: verified -> reject_no_transition : 4
exit code: 1
```

Four commits recorded as `verified` cannot produce a single named test that
transitions fail→pass when replayed. They are not flaky and they do not fail — they
were simply never evidence of anything, and the suite-level rule accepted them
because the suite happened to be red before and green after.

That two independent 40-commit samples, drawn with different seeds and scored under
different rules, both land at **90%** is the useful signal: the gate's real precision
is ~90%, not the 100% implied by "525 verified".

```
python scripts/reverify.py --repo ~/projects/hermes-webui \
    --data data/hermes-webui-all --sample 40 --strict --repeats 2
```

Exit code is non-zero when any sampled commit drifts, so it can gate CI the same
way the oracle does.

## 7. Reproducing

```bash
# strict gate on new candidates
python run.py gate --repo ~/projects/hermes-webui --data data/hermes-webui-all \
                   --strict --repeats 2

# audit a sample of what a previous run already accepted
python scripts/audit_gate.py --repo ~/projects/hermes-webui \
    --data data/hermes-webui-all --sample 40 --repeats 2 --seed 7
```

Both write machine-readable output (`verdicts.jsonl`, `audit_strict.json`) whose
rows name the individual tests, so any verdict can be re-checked later without
re-running the pipeline.

## 8. References

- Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan. *SWE-bench: Can Language
  Models Resolve Real-World GitHub Issues?* arXiv:2310.06770, ICLR 2024. Source
  of the FAIL_TO_PASS / PASS_TO_PASS construction.
- Luo, Hariri, Eloussi, Marinov. *An Empirical Analysis of Flaky Tests.*
  FSE 2014. 201 flaky-test-fixing commits across 51 projects.
