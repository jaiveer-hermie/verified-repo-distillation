"""Tests for the strict (per-test) gate.

The original gate trusted pytest's exit code: red before, green after. These
tests pin the stricter rule — a named test must flip fail->pass, nothing that
was green may go red, and an unstable test must not be counted as a fix.
"""
from __future__ import annotations

import textwrap

from vrd.replay import classify_transitions, parse_junit

JUNIT_OK = textwrap.dedent(
    """<?xml version="1.0" encoding="utf-8"?>
    <testsuites>
      <testsuite name="pytest" tests="3">
        <testcase classname="tests.test_api" name="test_create" time="0.01"/>
        <testcase classname="tests.test_api" name="test_delete" time="0.02">
          <failure message="boom">AssertionError</failure>
        </testcase>
        <testcase classname="tests.test_api" name="test_skip" time="0.0">
          <skipped message="not today"/>
        </testcase>
      </testsuite>
    </testsuites>
    """
)


def _write(tmp_path, text):
    p = tmp_path / "report.xml"
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_junit_reads_each_outcome(tmp_path):
    outcomes = parse_junit(_write(tmp_path, JUNIT_OK))
    assert outcomes["tests.test_api::test_create"] == "passed"
    assert outcomes["tests.test_api::test_delete"] == "failed"
    assert outcomes["tests.test_api::test_skip"] == "skipped"


def test_parse_junit_missing_file_is_empty(tmp_path):
    # A crash before pytest writes XML must not raise; it must read as "no data"
    assert parse_junit(tmp_path / "nope.xml") == {}


def test_parse_junit_malformed_is_empty(tmp_path):
    assert parse_junit(_write(tmp_path, "<testsuites><oops")) == {}


def test_classify_detects_real_fail_to_pass():
    pre = {"a::t1": "failed", "a::t2": "passed"}
    post = {"a::t1": "passed", "a::t2": "passed"}
    f2p, p2p, reg = classify_transitions(pre, post)
    assert f2p == ["a::t1"]
    assert p2p == 1
    assert reg == []


def test_classify_errors_count_as_red():
    pre = {"a::t1": "error"}
    post = {"a::t1": "passed"}
    f2p, _, _ = classify_transitions(pre, post)
    assert f2p == ["a::t1"]


def test_classify_flags_regression():
    pre = {"a::t1": "failed", "a::t2": "passed"}
    post = {"a::t1": "passed", "a::t2": "failed"}
    f2p, p2p, reg = classify_transitions(pre, post)
    assert f2p == ["a::t1"]
    assert p2p == 0
    assert reg == ["a::t2"]


def test_classify_no_transition_when_nothing_moved():
    # This is the false-positive the strict gate exists to catch: the suite was
    # green after, but no named test can be shown to have been fixed.
    pre = {"a::t1": "passed"}
    post = {"a::t1": "passed"}
    f2p, p2p, reg = classify_transitions(pre, post)
    assert f2p == []
    assert p2p == 1
    assert reg == []


def test_classify_ignores_skips_as_transitions():
    pre = {"a::t1": "skipped"}
    post = {"a::t1": "passed"}
    f2p, _, _ = classify_transitions(pre, post)
    assert f2p == []
