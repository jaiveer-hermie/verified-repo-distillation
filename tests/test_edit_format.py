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
