"""Tests for the harness itself.

The pipeline's whole claim is that it mechanically checks other people's commits.
That claim is worthless if its own parsing and classification are unverified, so
these tests pin the pieces that decide pass/fail:

  - commit classification (what the funnel drops, and why)
  - test-hunk stripping (the contamination boundary)
  - diff extraction from model output
  - runner summary parsing for both toolchains
  - the context budget
  - BM25 ordering
  - Wilson interval bounds
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrd.evaluate import BM25, extract_diff, wilson_interval
from vrd.ingest import Commit, FileChange, classify, funnel, mark_gate_candidate
from vrd.pairs import MAX_CONTEXT_BYTES, build_context, source_only_patch
from vrd.replay import detect_toolchain, go_packages, parse_go_summary, parse_pytest_summary


def commit(subject="fix: real change", files=None, parents=("a" * 40,)):
    return Commit(
        sha="b" * 40,
        parents=list(parents),
        author_date="2026-05-01T00:00:00+00:00",
        subject=subject,
        files=files or [FileChange("api/streaming.py", 10, 2, "M")],
    )


# ── ingest classification ──────────────────────────────────────────────────


def test_merge_is_dropped():
    c = classify(commit(parents=("a" * 40, "b" * 40)))
    assert (c.verdict, c.reason) == ("dropped", "merge")


def test_root_commit_is_dropped():
    c = classify(commit(parents=()))
    assert (c.verdict, c.reason) == ("dropped", "root-commit")


def test_docs_only_is_dropped():
    c = classify(commit(files=[FileChange("README.md", 5, 0, "M"),
                               FileChange("docs/design.md", 3, 0, "A")]))
    assert (c.verdict, c.reason) == ("dropped", "docs-only")


def test_tests_only_is_dropped():
    c = classify(commit(files=[FileChange("tests/test_x.py", 20, 0, "A")]))
    assert (c.verdict, c.reason) == ("dropped", "tests-only")


def test_lockfile_only_is_dropped():
    c = classify(commit(files=[FileChange("pnpm-lock.yaml", 900, 400, "M")]))
    assert (c.verdict, c.reason) == ("dropped", "lockfile-only")


def test_oversized_commit_is_dropped():
    big = FileChange("api/big.py", 600, 300, "M")
    c = classify(commit(files=[big]))
    assert (c.verdict, c.reason) == ("dropped", "too-large")


def test_release_subject_is_dropped():
    c = classify(commit(subject="v0.51.99"))
    assert (c.verdict, c.reason) == ("dropped", "bookkeeping-subject")


def test_source_change_is_kept():
    c = classify(commit())
    assert c.verdict == "kept"
    assert c.reason == ""


def test_gate_candidate_needs_added_test_and_source():
    c = classify(commit(files=[FileChange("api/x.py", 10, 1, "M"),
                               FileChange("tests/test_new.py", 40, 0, "A")]))
    mark_gate_candidate(c)
    assert c.candidate is True

    only_source = classify(commit(files=[FileChange("api/x.py", 10, 1, "M")]))
    mark_gate_candidate(only_source)
    assert only_source.candidate is False


def test_funnel_counts_add_up():
    commits = [classify(commit()), classify(commit(parents=("a" * 40, "c" * 40)))]
    report = funnel(commits)
    assert report["kept"] + report["dropped"] == report["commits_total"] == 2


# ── contamination boundary ────────────────────────────────────────────────


PATCH = """diff --git a/api/streaming.py b/api/streaming.py
--- a/api/streaming.py
+++ b/api/streaming.py
@@ -1 +1 @@
-old
+new
diff --git a/tests/test_streaming.py b/tests/test_streaming.py
--- a/tests/test_streaming.py
+++ b/tests/test_streaming.py
@@ -1 +1 @@
-assert False
+assert True
"""


def test_source_only_patch_strips_test_hunks():
    out = source_only_patch(PATCH)
    assert "api/streaming.py" in out
    assert "tests/test_streaming.py" not in out
    assert "assert True" not in out


# ── model output handling ─────────────────────────────────────────────────


def test_extract_diff_from_fence():
    text = "Sure, here you go:\n```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n```\n"
    assert extract_diff(text).startswith("diff --git a/x b/x")


def test_extract_diff_bare():
    text = "diff --git a/x b/x\n--- a/x\n+++ b/x\n"
    assert extract_diff(text).strip().endswith("+++ b/x")


def test_extract_diff_rejects_prose_only():
    assert extract_diff("I could not do that.") == ""
    assert extract_diff("") == ""


# ── runner parsing ────────────────────────────────────────────────────────


def test_parse_pytest_summary():
    out = "....F\n1 failed, 4 passed, 2 skipped in 1.20s"
    res = parse_pytest_summary(out, 1.2, 1)
    assert (res.passed, res.failed, res.skipped) == (4, 1, 2)
    assert not res.green


def test_parse_pytest_all_green():
    res = parse_pytest_summary("5 passed in 0.40s", 0.4, 0)
    assert res.green and res.passed == 5


def test_parse_pytest_collection_error_counts_as_error():
    res = parse_pytest_summary("E   ModuleNotFoundError: No module named 'api'", 0.2, 2)
    assert res.errors >= 1
    assert not res.green


def test_parse_go_build_failure_is_a_failure():
    out = "FAIL\tgithub.com/coredns/coredns/plugin/foo [build failed]\n"
    res = parse_go_summary(out, 1.0, 1)
    assert res.failed >= 1 and not res.green


def test_parse_go_pass():
    out = "--- PASS: TestX (0.00s)\nok  \tgithub.com/coredns/coredns/plugin/x\t0.023s\n"
    res = parse_go_summary(out, 0.1, 0)
    assert res.green and res.passed == 1


def test_toolchain_detection(tmp_path):
    (tmp_path / "go.mod").write_text("module x\n")
    (tmp_path / "requirements.txt").write_text("pyyaml\n")
    assert detect_toolchain(tmp_path) == "go"        # go wins over a stray reqs file
    (tmp_path / "go.mod").unlink()
    assert detect_toolchain(tmp_path) == "python"


def test_go_packages_from_test_paths():
    assert go_packages(["plugin/x/x_test.go"]) == ["./plugin/x"]
    assert go_packages(["root_test.go"]) == ["."]


# ── context budget ────────────────────────────────────────────────────────


def test_context_never_exceeds_budget():
    big = {"api/huge.py": "x = 1\n" * 20_000}
    ctx = build_context(big, [])
    assert len(ctx) <= MAX_CONTEXT_BYTES + 200      # + fence overhead
    assert "truncated" in ctx


def test_context_keeps_multiple_small_files():
    files = {"api/a.py": "a = 1\n", "api/b.py": "b = 2\n"}
    ctx = build_context(files, ["tests/test_a.py"])
    assert "api/a.py" in ctx and "api/b.py" in ctx and "tests/test_a.py" in ctx


# ── retrieval ─────────────────────────────────────────────────────────────


def test_bm25_ranks_the_relevant_file_first():
    docs = {
        "api/streaming.py": "cancel_stream queue thread lock release",
        "static/style.css": ".button { color: red }",
        "docs/readme.md": "installation instructions for linux",
    }
    bm25 = BM25(docs)
    hits = bm25.search("cancel stream lock", k=3)
    assert hits[0][0] == "api/streaming.py"


def test_bm25_returns_nothing_for_unmatched_query():
    bm25 = BM25({"a.py": "alpha beta"})
    assert bm25.search("zzzz") == []


# ── statistics ────────────────────────────────────────────────────────────


def test_wilson_interval_is_bounded():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and hi > 0.0


def test_wilson_interval_zero_n():
    assert wilson_interval(0, 0) == (0.0, 0.0)
