"""Tests for the search/replace edit contract.

The contract exists because the model could not produce a valid unified diff:
first a bare `@@`, then hunk line counts that disagreed with its own bodies. Those
are formatting failures and must not be scored as editing failures. These tests
pin both directions of the conversion and the refusal cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrd.edit_format import (Edit, apply_edits, diff_to_edits, parse_edits,
                             render_edits)

MODEL_OUTPUT = """### FILE: api/clarify.py
<<<<<<< SEARCH
    entry = _ClarifyEntry(data)
=======
    entry = _ClarifyEntry(data, notify=True)
>>>>>>> REPLACE
"""


def test_parse_single_block():
    edits, err = parse_edits(MODEL_OUTPUT)
    assert err == ""
    assert len(edits) == 1
    assert edits[0].path == "api/clarify.py"
    assert "notify=True" in edits[0].replace


def test_parse_multiple_files_and_blocks():
    text = (
        "### FILE: a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
        "### FILE: b.py\n<<<<<<< SEARCH\ny = 1\n=======\ny = 3\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nz = 1\n=======\nz = 4\n>>>>>>> REPLACE\n"
    )
    edits, err = parse_edits(text)
    assert err == "" and len(edits) == 3
    assert [e.path for e in edits] == ["a.py", "b.py", "b.py"]


def test_parse_rejects_prose_only():
    edits, err = parse_edits("I have updated the file for you.")
    assert edits == [] and "no '### FILE:' header" in err


def test_parse_rejects_header_without_block():
    edits, err = parse_edits("### FILE: a.py\nnothing here\n")
    assert edits == [] and "no search/replace block" in err


def test_apply_exact_match(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    ok, note = apply_edits(tmp_path, [Edit("a.py", "x = 1", "x = 99")])
    assert ok, note
    assert f.read_text() == "x = 99\ny = 2\n"


def test_apply_refuses_ambiguous_match(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\nx = 1\n")
    ok, note = apply_edits(tmp_path, [Edit("a.py", "x = 1", "x = 2")])
    assert not ok and "must be unique" in note
    assert f.read_text() == "x = 1\nx = 1\n"      # untouched


def test_apply_refuses_missing_match(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    ok, note = apply_edits(tmp_path, [Edit("a.py", "nope = 0", "nope = 1")])
    assert not ok and "SEARCH not found" in note


def test_apply_refuses_path_escape(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    ok, note = apply_edits(tmp_path, [Edit("../outside.py", "x", "y")])
    assert not ok and "escapes repository" in note


def test_apply_tolerates_trailing_whitespace(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():   \n    return 1\n")
    ok, note = apply_edits(tmp_path, [Edit("a.py", "def f():\n    return 1",
                                           "def f():\n    return 2")])
    assert ok, note
    assert "return 2" in f.read_text()


def test_diff_to_edits_roundtrip():
    patch = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        " keep\n"
        "-old\n"
        "+new\n"
    )
    edits = diff_to_edits(patch)
    assert len(edits) == 1
    assert edits[0].path == "a.py"
    assert edits[0].search == "keep\nold"
    assert edits[0].replace == "keep\nnew"


def test_render_edits_groups_by_file():
    text = render_edits([Edit("a.py", "1", "2"), Edit("a.py", "3", "4")])
    assert text.count("### FILE: a.py") == 1
    assert text.count("<<<<<<< SEARCH") == 2


def test_roundtrip_survives_multiple_hunks():
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,1 +1,1 @@\n-a\n+A\n"
        "@@ -9,1 +9,1 @@\n-b\n+B\n"
    )
    edits = diff_to_edits(patch)
    assert [e.search for e in edits] == ["a", "b"]
    assert [e.replace for e in edits] == ["A", "B"]


# ── diff_to_edits: the real git output ────────────────────────────────────
# Regression: the hunk-header check demanded a line ending in "@@", so every
# hunk carrying a section heading was dropped and 176/483 training targets came
# out empty. These use verbatim git output, headings included.

GIT_PATCH = """diff --git a/api/clarify.py b/api/clarify.py
index f6d16bb..f43d40d 100644
--- a/api/clarify.py
+++ b/api/clarify.py
@@ -12,6 +12,8 @@ import time
 import uuid
 from typing import Optional
 
+from api.session_events import publish_session_list_changed
+
 
 DEFAULT_TIMEOUT_SECONDS = 120
"""


def test_diff_to_edits_keeps_hunks_with_section_headings():
    edits = diff_to_edits(GIT_PATCH)
    assert len(edits) == 1
    assert edits[0].path == "api/clarify.py"
    assert "publish_session_list_changed" in edits[0].replace
    assert "publish_session_list_changed" not in edits[0].search


def test_diff_to_edits_multi_file_multi_hunk():
    patch = GIT_PATCH + """diff --git a/vrd/x.py b/vrd/x.py
--- a/vrd/x.py
+++ b/vrd/x.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 3
 c = 4
"""
    edits = diff_to_edits(patch)
    assert [e.path for e in edits] == ["api/clarify.py", "vrd/x.py"]


def test_edits_reproduce_the_patch_exactly(tmp_path):
    """The invariant that matters: edits applied == patch applied."""
    original = """import time
import uuid
from typing import Optional


DEFAULT_TIMEOUT_SECONDS = 120
a = 1
b = 2
c = 4
"""
    (tmp_path / "m.py").write_text(original)

    patch = """--- a/m.py
+++ b/m.py
@@ -1,3 +1,4 @@
 import time
 import uuid
+import json
 from typing import Optional
@@ -6,3 +7,3 @@ DEFAULT_TIMEOUT_SECONDS = 120
 a = 1
-b = 2
+b = 3
 c = 4
"""
    edits = diff_to_edits(patch)
    assert len(edits) == 2
    ok, note = apply_edits(tmp_path, edits)
    assert ok, note

    expected = original.replace("import uuid\n", "import uuid\nimport json\n").replace("b = 2", "b = 3")
    assert (tmp_path / "m.py").read_text() == expected


# ── fuzzy anchor matching ─────────────────────────────────────────────────
# Aider matches blocks fuzzily because models reproduce anchors imperfectly.
# The tests still decide correctness; this only measures whether the edit
# landed, not whether the model can copy bytes.

def _tree(tmp_path, name, body):
    (tmp_path / name).write_text(body)
    return tmp_path


def test_whitespace_only_difference_still_applies(tmp_path):
    body = "def f():\n    x = 1\n    return x\n"
    tree = _tree(tmp_path, "a.py", body)
    edit = Edit(path="a.py", search="def f():\n  x = 1\n  return x", replace="def f():\n  x = 2\n  return x")
    ok, note = apply_edits(tree, [edit])
    assert ok, note
    assert "x = 2" in (tree / "a.py").read_text()


def test_near_miss_line_applies_at_best_position(tmp_path):
    body = ("import os\n\n\nclass A:\n    def run(self):\n"
            "        value = compute(alpha, beta)\n        return value\n")
    tree = _tree(tmp_path, "b.py", body)
    # one character differs in the anchor (compute -> comput)
    edit = Edit(path="b.py",
                search="    def run(self):\n        value = comput(alpha, beta)\n        return value",
                replace="    def run(self):\n        value = compute(alpha, beta, 1)\n        return value")
    ok, note = apply_edits(tree, [edit])
    assert ok, note
    assert "compute(alpha, beta, 1)" in (tree / "b.py").read_text()


def test_ambiguous_anchor_is_refused_not_guessed(tmp_path):
    body = "x = 1\ny = 2\n" * 3
    tree = _tree(tmp_path, "c.py", body)
    edit = Edit(path="c.py", search="x = 1\ny = 2", replace="x = 9\ny = 2")
    ok, note = apply_edits(tree, [edit])
    assert not ok
    assert "matches" in note or "not found" in note


def test_unrelated_anchor_is_reported_not_applied(tmp_path):
    tree = _tree(tmp_path, "d.py", "def f():\n    return 1\n")
    edit = Edit(path="d.py", search="class Completely:\n    different = True",
                replace="class Completely:\n    different = False")
    ok, note = apply_edits(tree, [edit])
    assert not ok
    assert "not found" in note
