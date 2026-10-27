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
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

from vrd.ingest import Commit, GitRepo, is_test_path

MAX_CONTEXT_BYTES = 24_000        # hard cap on what arm A and arm C see
MAX_FILE_BYTES = 8_000            # per-file cap; big files are truncated, not included whole
TRIVIAL_LINE_DIFF = 3             # <= this many changed lines is noise, not a task


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

    def to_prompt(self) -> str:
        return (
            "You are editing an existing repository. Apply the smallest change that "
            "implements the request below.\n\n"
            f"## Request\n{self.subject}\n\n"
            f"## Files\n" + "\n".join(f"- {p}" for p in self.changed_files) + "\n\n"
            f"## Current contents\n{self.context}\n\n"
            "## Output\nReturn a unified diff only. No prose, no fences.\n"
        )


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
    """Strip test-file hunks out of the gold patch.

    The model is not being trained to write the tests — the tests are the
    verification instrument. If a hunk that edits tests reaches the training
    target, the corpus learns to satisfy the gate rather than the codebase.
    """
    keep, current_is_test, out = [], False, []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            current_is_test = " tests/" in line or "/tests/" in line
        if not current_is_test:
            out.append(line)
        elif line.startswith("diff --git "):
            continue
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
    for line in Path(verdicts_path).read_text().splitlines():
        v = json.loads(line)
        if v["status"] != "verified":
            continue
        sha = v["sha"]
        parent = repo._git("rev-parse", f"{sha}^").strip()
        subject = v["subject"]
        message = repo._git("show", "-s", "--format=%B", sha).strip()

        # source files the commit touched, at the parent state
        changed = repo._git("show", "--name-only", "--format=", sha).split()
        source_files = [p for p in changed if not is_test_path(p)]
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
        ))
        if limit and len(pairs) >= limit:
            break
    write_pairs(pairs, out_path)
    return pairs


def write_pairs(pairs: list[Pair], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
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
