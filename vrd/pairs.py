"""Stage 4 — PAIR BUILDER.

Turns verified commits into instruction pairs. The design constraint that makes
or breaks the experiment: arm A (base model, minimal context) and arm C (tuned
model) must see the *same* context. Only then does a score difference belong to
the weights. Dumping the whole repo here would silently convert arm A into a
retrieval arm and destroy the comparison.

So the context is deliberately narrow:
    - the pre-patch contents of every non-test file the commit touched
    - the names (not the bodies) of tests already covering those files
    - the commit message

Arm B gets the wide context. Arm A and arm C get this.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from vrd.ingest import Commit, GitRepo, is_test_path, read_jsonl
from vrd.edit_format import (HUNK_RE, INSTRUCTIONS as EDIT_INSTRUCTIONS,
                             INSTRUCTIONS_WHOLE, apply_edits, apply_whole,
                             diff_to_edits, parse_edits, parse_whole,
                             render_edits, render_whole)

# Which contract the model is asked to speak.
#   diff  (default) — Aider's "diff": search/replace blocks, harness computes the
#                     patch. Aider uses this for GPT-4-class models, and this repo
#                     has ~100KB source files where whole-file output is impractical.
#                     Nemotron Super is 59-60% SWE-bench Verified, i.e. that class.
#   whole           — return the entire file. Aider measured this as the more
#                     reliable format for *weaker* models (46%/39% vs 30%/19% for
#                     GPT-3.5), but it needs the whole file in context and output,
#                     so it only works on small files.
EDIT_FORMAT = os.getenv("VRD_EDIT_FORMAT", "diff").lower()

MAX_CONTEXT_BYTES = int(os.getenv("VRD_MAX_CONTEXT", "48000"))  # what arm A/C see
MAX_FILE_BYTES = int(os.getenv("VRD_MAX_FILE", "16000"))        # whole-file threshold
TRIVIAL_LINE_DIFF = 3             # <= this many changed lines is noise, not a task

# Binary assets cannot be a source edit; keeping them out of pairs and context
# avoids both decode errors and nonsense diffs.
BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
                   ".ttf", ".otf", ".woff", ".woff2", ".eot", ".zip", ".gz",
                   ".so", ".bin", ".mp3", ".mp4", ".webm", ".svgz")


def to_utc_iso(ts: str) -> str:
    """Normalise a git ISO timestamp to UTC.

    Commits here carry offsets from -07:00 to +08:00. Comparing those strings
    lexicographically compares local wall-clock times, so a "temporal" split
    would put commits in the wrong side of the boundary by up to 15 hours.
    """
    ts = (ts or "").strip()
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_binary_path(path: str) -> bool:
    return path.lower().endswith(BINARY_SUFFIXES)


def edits_are_expressible(repo: GitRepo, parent: str, patch: str) -> bool:
    """Can this patch be replayed as unique SEARCH/REPLACE blocks?

    Checks each parsed edit's SEARCH text against the real file at the parent
    commit: it must occur exactly once. An empty SEARCH (pure insertion) is
    accepted, since there is nothing to disambiguate.
    """
    file_cache: dict[str, str] = {}
    for edit in diff_to_edits(patch):
        if not edit.search.strip():
            continue
        if edit.path not in file_cache:
            try:
                file_cache[edit.path] = repo.show_file(parent, edit.path)
            except RuntimeError:
                return False        # added by the commit: no SEARCH target
        if file_cache[edit.path].count(edit.search) != 1:
            return False
    return True


@dataclass
class Pair:
    sha: str
    repo: str
    commit_date: str
    subject: str
    message: str
    changed_files: list[str]
    context: str                  # what arm A and arm C see
    patch: str                    # gold unified diff, source files only
    test_files: list[str]         # the verification tests (held out from context)
    source_files: list[str] = field(default_factory=list)
    target_files: dict[str, str] = field(default_factory=dict)  # post-commit contents

    def to_prompt(self) -> str:
        instructions = INSTRUCTIONS_WHOLE if EDIT_FORMAT == "whole" else EDIT_INSTRUCTIONS
        return (
            "You are editing an existing repository. Apply the smallest change that "
            "implements the request below.\n\n"
            f"## Request\n{self.subject}\n\n"
            f"## Files\n" + "\n".join(f"- {p}" for p in self.changed_files) + "\n\n"
            f"## Current contents\n{self.context}\n\n"
            + instructions
        )

    def to_target(self) -> str:
        """The gold answer, in the same contract the evaluator parses.

        Training on unified diffs while scoring block edits would make the tuned
        arm fail on format rather than on edits.
        """
        if EDIT_FORMAT == "whole" and self.target_files:
            return render_whole(self.target_files)
        return render_edits(diff_to_edits(self.patch))


def _fence(path: str, body: str) -> str:
    return f"### {path}\n```\n{body}\n```\n"


def hunk_windows(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Per file, the (start, end) line ranges the patch touches at the parent."""
    windows: dict[str, list[tuple[int, int]]] = {}
    path = ""
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            path = ""
            continue
        if line.startswith("+++ "):
            candidate = line[4:].strip()
            path = candidate[2:] if candidate.startswith("b/") else candidate
            continue
        m = HUNK_RE.match(line)
        if m and path:
            span = m.group(1)
            start = int(span.split(",")[0])
            count = int(span.split(",")[1]) if "," in span else 1
            windows.setdefault(path, []).append((start, start + max(count, 1) - 1))
    return windows


def build_context(tree_files: dict[str, str], test_names: list[str],
                  windows: dict[str, list[tuple[int, int]]] | None = None,
                  pad: int = 60) -> str:
    """Assemble the context, guaranteeing the editable region is visible.

    These commits touch files with a median size of 521 KB, so truncating to a
    per-file byte cap showed the model the first 8 KB of a 500 KB file and then
    asked it to reproduce exact text it had never seen. Every arm scored ~0 for
    that reason alone — a harness artifact, not a model property.

    So: small files are included whole. Large files are shown as windows around
    the lines the commit actually changes (the parent content there, which is
    exactly what an edit must match), each labelled with its line range.
    """
    chunks: list[str] = []
    budget = MAX_CONTEXT_BYTES
    for path, body in tree_files.items():
        if budget <= 500:
            break
        if len(body) <= MAX_FILE_BYTES or not windows or path not in windows:
            cap = min(MAX_FILE_BYTES, budget)
            shown = body if len(body) <= cap else (
                body[:cap] + f"\n... [truncated {len(body) - cap} bytes]")
            piece = _fence(path, shown)
            chunks.append(piece)
            budget -= len(piece)
            continue

        lines = body.splitlines()
        total = len(lines)
        ranges: list[tuple[int, int]] = []
        for start, end in sorted(windows[path]):
            lo = max(1, start - pad)
            hi = min(total, end + pad)
            if ranges and lo <= ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], hi))
            else:
                ranges.append((lo, hi))

        for lo, hi in ranges:
            if budget <= 500:
                break
            excerpt = "\n".join(lines[lo - 1:hi])
            header = f"### {path} (lines {lo}-{hi} of {total})\n"
            cap = min(len(excerpt), budget - len(header) - 20)
            if cap <= 0:
                break
            piece = header + "```\n" + excerpt[:cap] + ("\n```" if cap == len(excerpt) else
                                                        "\n... [excerpt truncated]\n```")
            chunks.append(piece)
            budget -= len(piece) + 2
    if test_names:
        chunks.append("### Tests that cover this code\n" +
                      "\n".join(f"- {t}" for t in test_names) + "\n")
    return "".join(chunks)


def edits_visible_in_context(patch: str, context: str) -> bool:
    """Is every edit's SEARCH text present verbatim in the prompt?

    Trailing whitespace is ignored on both sides (the applier is too), but the
    lines themselves must be there: an edit whose anchor was truncated out of
    the prompt is unanswerable, so scoring it measures nothing about the model.
    """
    def norm(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.splitlines())

    ctx = norm(context)
    for edit in diff_to_edits(patch):
        search = norm(edit.search).strip()
        if search and search not in ctx:
            return False
    return True


def source_only_patch(patch: str) -> str:
    """Strip test-file and binary hunks out of the gold patch.

    The model is not being trained to write the tests — the tests are the
    verification instrument. If a hunk that edits tests reaches the training
    target, the corpus learns to satisfy the gate rather than the codebase.
    Binary hunks (base85 blobs for PNGs and fonts) are dropped for the same
    reason they are excluded from context: they are not a text edit, and they
    made the corpus 28MB instead of 13MB.
    """
    out: list[str] = []
    skip = False
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            path = line.split(" b/")[-1].strip()
            skip = (" tests/" in line or "/tests/" in line or is_binary_path(path))
        if not skip:
            out.append(line)
    return "".join(out)


def test_names_touching(repo: GitRepo, tree_sha: str, source_paths: list[str]) -> list[str]:
    """Existing test files whose names reference a module touched by the commit."""
    stems = {Path(p).stem for p in source_paths}
    names: list[str] = []
    listing = repo._git("ls-tree", "-r", "--name-only", tree_sha)
    for path in listing.splitlines():
        if not is_test_path(path):
            continue
        stem = Path(path).stem
        if any(s and s in stem for s in stems):
            names.append(path)
    return sorted(names)[:20]


def build_pairs(repo_path: str | Path, verdicts_path: str | Path, out_path: str | Path,
                limit: int | None = None) -> list[Pair]:
    repo = GitRepo(repo_path)
    pairs: list[Pair] = []
    skipped = 0
    not_expressible = 0
    context_incomplete = 0
    for v in read_jsonl(verdicts_path):
        if v["status"] != "verified":
            continue
        sha = v["sha"]
        parent = repo._git("rev-parse", f"{sha}^").strip()
        subject = v["subject"]
        message = repo._git("show", "-s", "--format=%B", sha).strip()
        # The whole evaluation design rests on a temporal split, so the date has
        # to be read from the commit itself. Verdict records do not carry it, and
        # reading it from there left every pair with an empty date: no split.
        commit_date = to_utc_iso(v.get("commit_date") or repo._git(
            "show", "-s", "--format=%cI", sha).strip())

        # source files the commit touched, at the parent state
        changed = repo._git("show", "--name-only", "--format=", sha).split()
        source_files = [p for p in changed
                        if not is_test_path(p) and not is_binary_path(p)]
        if len(source_files) > 8 or not source_files:
            continue

        patch = source_only_patch(repo.patch(sha))
        if not patch.strip():
            continue
        lines = [l for l in patch.splitlines()
                 if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---"))]
        if len(lines) <= TRIVIAL_LINE_DIFF:
            continue

        # The contract is SEARCH/REPLACE, which is position-free. A git hunk is
        # position-anchored, so a hunk whose search text occurs twice in the file
        # cannot be replayed: the gold answer itself would fail. Those commits
        # are removed from the corpus instead of being scored as model failures
        # (the oracle test found one; leaving it in would bias every arm down by
        # the same amount).
        if not edits_are_expressible(repo, parent, patch):
            not_expressible += 1
            continue

        tree_files: dict[str, str] = {}
        for path in source_files:
            try:
                tree_files[path] = repo.show_file(parent, path)
            except RuntimeError:
                continue        # file added by the commit: nothing to show
        if not tree_files:
            continue

        context = build_context(tree_files, test_names_touching(repo, parent, source_files),
                                windows=hunk_windows(patch))

        # A pair is only fair if the text the answer must reproduce is actually
        # in the prompt. Half of these commits touch ~500KB files, so this is
        # the difference between "edit this code" and "guess text you were never
        # shown". Unfair pairs are dropped, not scored.
        if not edits_visible_in_context(patch, context):
            context_incomplete += 1
            continue

        # Whole-file answers require the whole file in context. A truncated
        # context plus a "return the entire file" contract is unwinnable, so
        # those pairs are excluded rather than silently guaranteed to fail.
        if EDIT_FORMAT == "whole" and sum(len(v) for v in tree_files.values()) > MAX_CONTEXT_BYTES:
            skipped += 1
            continue

        # Whole-file targets are large (a 100KB source file becomes a 100KB
        # record), so only carry them when the whole contract is in use.
        target_files: dict[str, str] = {}
        if EDIT_FORMAT == "whole":
            for path in source_files:
                try:
                    target_files[path] = repo.show_file(sha, path)
                except RuntimeError:
                    continue    # deleted by the commit; not expressible as whole-file

        pairs.append(Pair(
            sha=sha,
            repo=repo.path.name,
            commit_date=commit_date,
            subject=subject,
            message=message,
            changed_files=list(tree_files),
            context=context,
            patch=patch,
            test_files=v["test_files"],
            source_files=source_files,
            target_files=target_files,
        ))
        if limit and len(pairs) >= limit:
            break
    if skipped:
        print(f"pairs: skipped {skipped} whose files exceed the {MAX_CONTEXT_BYTES}B "
              f"context budget (whole-file contract needs the whole file)")
    if not_expressible:
        print(f"pairs: dropped {not_expressible} whose patch cannot be written as "
              f"unique SEARCH/REPLACE blocks (ambiguous search text)")
    if context_incomplete:
        print(f"pairs: dropped {context_incomplete} whose editable lines were not "
              f"visible in the prompt (unanswerable as posed)")
    write_pairs(pairs, out_path)
    return pairs


def write_pairs(pairs: list[Pair], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            # ensure_ascii=True keeps U+0085/U+2028 out of the raw file; see
            # read_jsonl() in vrd.ingest for why that matters.
            fh.write(json.dumps(asdict(p), ensure_ascii=True) + "\n")
    return path


def dataset_card(pairs: list[Pair]) -> dict:
    """The artifact that makes the corpus auditable."""
    from collections import Counter

    return {
        "pairs": len(pairs),
        "repos": dict(Counter(p.repo for p in pairs)),
        "median_context_bytes": sorted(len(p.context) for p in pairs)[len(pairs) // 2] if pairs else 0,
        "median_patch_bytes": sorted(len(p.patch) for p in pairs)[len(pairs) // 2] if pairs else 0,
        "files_touched_histogram": dict(sorted(Counter(len(p.changed_files) for p in pairs).items())),
        "verification": "each pair's added test failed on the unpatched parent and passed after the patch",
    }
