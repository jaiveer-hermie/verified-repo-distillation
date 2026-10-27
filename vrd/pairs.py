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
from pathlib import Path

from vrd.ingest import Commit, GitRepo, is_test_path, read_jsonl
from vrd.edit_format import (INSTRUCTIONS as EDIT_INSTRUCTIONS,
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

MAX_CONTEXT_BYTES = 24_000        # hard cap on what arm A and arm C see
MAX_FILE_BYTES = 8_000            # per-file cap; big files are truncated, not included whole
TRIVIAL_LINE_DIFF = 3             # <= this many changed lines is noise, not a task

# Binary assets cannot be a source edit; keeping them out of pairs and context
# avoids both decode errors and nonsense diffs.
BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
                   ".ttf", ".otf", ".woff", ".woff2", ".eot", ".zip", ".gz",
                   ".so", ".bin", ".mp3", ".mp4", ".webm", ".svgz")


def is_binary_path(path: str) -> bool:
    return path.lower().endswith(BINARY_SUFFIXES)


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


def build_context(tree_files: dict[str, str], test_names: list[str]) -> str:
    """Assemble the minimal context. Both caps are enforced, always.

    An earlier version let the first file through whole "to keep something
    complete", which produced 861KB prompts on a repo with large JS assets.
    A budget with an exception is not a budget.
    """
    chunks: list[str] = []
    budget = MAX_CONTEXT_BYTES
    for path, body in tree_files.items():
        if budget <= 500:
            break
        cap = min(MAX_FILE_BYTES, budget)
        if len(body) > cap:
            dropped = len(body) - cap
            body = body[:cap] + f"\n... [truncated {dropped} bytes]"
        piece = _fence(path, body)
        chunks.append(piece)
        budget -= len(piece)
    if test_names:
        chunks.append("### Tests that cover this code\n" +
                      "\n".join(f"- {t}" for t in test_names) + "\n")
    return "".join(chunks)


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
    for v in read_jsonl(verdicts_path):
        if v["status"] != "verified":
            continue
        sha = v["sha"]
        parent = repo._git("rev-parse", f"{sha}^").strip()
        subject = v["subject"]
        message = repo._git("show", "-s", "--format=%B", sha).strip()

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

        tree_files: dict[str, str] = {}
        for path in source_files:
            try:
                tree_files[path] = repo.show_file(parent, path)
            except RuntimeError:
                continue        # file added by the commit: nothing to show
        if not tree_files:
            continue

        context = build_context(tree_files, test_names_touching(repo, parent, source_files))

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
            commit_date=v.get("commit_date", ""),
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
