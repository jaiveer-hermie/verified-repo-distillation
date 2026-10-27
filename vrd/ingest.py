"""Stage 1 — INGEST.

Walks a repository's history and classifies every commit as either training
material or noise. Nothing downstream sees a commit that fails here, so the
classification rules are deliberately conservative: we would rather drop a
usable commit than admit one that carries no real behavioural change.

Output: one JSON object per commit, plus an aggregate funnel report.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# ── Path classification ────────────────────────────────────────────────────

DOC_PATTERNS = re.compile(
    r"^(docs?/|\.github/|.*\.(md|rst|txt|png|jpg|jpeg|gif|svg|webp|ico|pdf)$"
    r"|LICEN[CS]E|CHANGELOG|CONTRIBUTORS|AUTHORS|NOTICE)",
    re.IGNORECASE,
)

LOCKFILE_PATTERNS = re.compile(
    r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock"
    r"|Pipfile\.lock|uv\.lock|go\.sum|Cargo\.lock|requirements.*\.txt"
    r"|composer\.lock|Gemfile\.lock)$",
    re.IGNORECASE,
)

TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)(test_[^/]*|[^/]*_test)\.(py|go|js|ts|tsx|jsx)$")

# Commit subjects that signal bookkeeping rather than behaviour.
NOISE_SUBJECT = re.compile(
    r"^(revert\b|merge\b|v?\d+\.\d+\.\d+\s*$|bump\b|release\b|chore\(release\)"
    r"|update (changelog|version)|version bump|wip$|fix typo in readme)",
    re.IGNORECASE,
)

# A commit bigger than this is not a single well-specified task.
MAX_FILES_CHANGED = 25
MAX_LINES_CHANGED = 800


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH.search(path))


def is_doc_path(path: str) -> bool:
    return bool(DOC_PATTERNS.match(path))


def is_lockfile_path(path: str) -> bool:
    return bool(LOCKFILE_PATTERNS.search(path))


# ── Data model ─────────────────────────────────────────────────────────────


@dataclass
class FileChange:
    path: str
    added: int
    deleted: int
    status: str  # A, M, D, R100, ...


@dataclass
class Commit:
    sha: str
    parents: list[str]
    author_date: str
    subject: str
    files: list[FileChange] = field(default_factory=list)

    # derived
    verdict: str = "kept"          # kept | dropped
    reason: str = ""               # why dropped, or "" when kept
    candidate: bool = False        # eligible for the counterfactual gate
    mainline: bool = False         # reached by --first-parent walk

    @property
    def insertions(self) -> int:
        return sum(f.added for f in self.files)

    @property
    def deletions(self) -> int:
        return sum(f.deleted for f in self.files)

    def paths(self, status_filter: set[str] | None = None) -> list[str]:
        if status_filter is None:
            return [f.path for f in self.files]
        return [f.path for f in self.files if f.status[:1] in status_filter]


# ── Git plumbing ───────────────────────────────────────────────────────────


class GitRepo:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not (self.path / ".git").exists():
            raise SystemExit(f"not a git repository: {self.path}")

    def _git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True,
            text=True,
            errors="replace",   # commits touch PNG/TTF assets; never explode on bytes
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def head_sha(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def commit_count(self) -> int:
        return int(self._git("rev-list", "--count", "HEAD").strip())

    def log(self, first_parent: bool = True, limit: int | None = None) -> list[Commit]:
        """Return commits newest-first with their file changes resolved."""
        args = ["log", "--numstat", "--format=%x00%H%x1f%P%x1f%aI%x1f%s"]
        if first_parent:
            args.append("--first-parent")
        if limit:
            args.append(f"-n{limit}")
        raw = self._git(*args)
        return list(self._parse_log(raw, first_parent))

    def _parse_log(self, raw: str, mainline: bool) -> Iterable[Commit]:
        for chunk in raw.split("\x00"):
            chunk = chunk.strip("\n")
            if not chunk:
                continue
            header, _, body = chunk.partition("\n")
            parts = header.split("\x1f")
            if len(parts) < 4:
                continue
            sha, parents, date, subject = parts[0], parts[1], parts[2], parts[3]
            files: list[FileChange] = []
            for line in body.splitlines():
                if not line.strip():
                    continue
                cols = line.split("\t")
                if len(cols) < 3:
                    continue
                added_s, deleted_s, path = cols[0], cols[1], cols[2]
                files.append(
                    FileChange(
                        path=path,
                        added=0 if added_s == "-" else int(added_s),
                        deleted=0 if deleted_s == "-" else int(deleted_s),
                        status="M",  # refined below via name-status when needed
                    )
                )
            yield Commit(
                sha=sha,
                parents=parents.split() if parents else [],
                author_date=date,
                subject=subject,
                files=files,
                mainline=mainline,
            )

    def name_status(self, sha: str) -> dict[str, str]:
        """Map path -> status letter for one commit (A/M/D/R)."""
        raw = self._git("show", "--name-status", "--format=", sha)
        out: dict[str, str] = {}
        for line in raw.splitlines():
            cols = line.split("\t")
            if len(cols) >= 2:
                status = cols[0]
                path = cols[-1]
                out[path] = status
        return out

    def patch(self, sha: str) -> str:
        # --binary and --full-index are required: without them any commit that
        # touches a PNG or a font produces an unappliable patch, and the gate
        # records a harness failure as if the commit were unusable.
        return self._git("show", "--format=", "--patch", "--binary", "--full-index", sha)

    def show_file(self, sha: str, path: str) -> str:
        return self._git("show", f"{sha}:{path}")


# ── Classification ─────────────────────────────────────────────────────────


def classify(commit: Commit) -> Commit:
    """Assign verdict/reason/candidate in place and return the commit."""
    if len(commit.parents) > 1:
        commit.verdict, commit.reason = "dropped", "merge"
        return commit
    if not commit.parents:
        commit.verdict, commit.reason = "dropped", "root-commit"
        return commit
    if not commit.files:
        commit.verdict, commit.reason = "dropped", "empty"
        return commit
    if NOISE_SUBJECT.match(commit.subject.strip()):
        commit.verdict, commit.reason = "dropped", "bookkeeping-subject"
        return commit

    paths = [f.path for f in commit.files]
    if all(is_doc_path(p) for p in paths):
        commit.verdict, commit.reason = "dropped", "docs-only"
        return commit
    if all(is_lockfile_path(p) for p in paths):
        commit.verdict, commit.reason = "dropped", "lockfile-only"
        return commit
    if all(is_test_path(p) for p in paths):
        commit.verdict, commit.reason = "dropped", "tests-only"
        return commit
    if len(paths) > MAX_FILES_CHANGED:
        commit.verdict, commit.reason = "dropped", "too-many-files"
        return commit
    if commit.insertions + commit.deletions > MAX_LINES_CHANGED:
        commit.verdict, commit.reason = "dropped", "too-large"
        return commit
    if any(stripped_is_generated(p) for p in paths):
        commit.verdict, commit.reason = "dropped", "generated-file"
        return commit

    commit.verdict = "kept"
    commit.reason = ""
    return commit


GENERATED_HINTS = (".min.js", ".min.css", "package-lock.json", "_pb2.py", ".lock")


def stripped_is_generated(path: str) -> bool:
    return any(h in path for h in GENERATED_HINTS)


def mark_gate_candidate(commit: Commit) -> Commit:
    """A commit is gate-eligible when its patch *adds* a test that did not exist.

    That is what makes counterfactual verification possible: the new test can be
    replayed against the unpatched parent, where it must fail, and against the
    patched tree, where it must pass.
    """
    commit.candidate = bool(commit.added_tests) and bool(commit.source_paths)
    return commit


# ── Funnel report ──────────────────────────────────────────────────────────


def funnel(commits: list[Commit]) -> dict:
    from collections import Counter

    reasons = Counter(c.reason for c in commits if c.verdict == "dropped")
    kept = [c for c in commits if c.verdict == "kept"]
    return {
        "commits_total": len(commits),
        "kept": len(kept),
        "dropped": len(commits) - len(kept),
        "drop_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "gate_candidates": sum(1 for c in kept if c.candidate),
    }


def write_jsonl(commits: list[Commit], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in commits:
            row = asdict(c)
            row["insertions"] = c.insertions
            row["deletions"] = c.deletions
            row["added_tests"] = c.added_tests
            row["source_paths"] = c.source_paths
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# ── Convenience properties used by later stages ────────────────────────────
# Defined here rather than in the dataclass so the dataclass stays a plain
# record of what git reported.


def _added_tests(self: Commit) -> list[str]:
    return [f.path for f in self.files if is_test_path(f.path) and f.status.startswith("A")]


def _source_paths(self: Commit) -> list[str]:
    return [f.path for f in self.files if not is_test_path(f.path) and not is_doc_path(f.path)]


Commit.added_tests = property(_added_tests)  # type: ignore[attr-defined]
Commit.source_paths = property(_source_paths)  # type: ignore[attr-defined]


def ingest(repo_path: str | Path, first_parent: bool = True, limit: int | None = None,
           refine_status: bool = True) -> list[Commit]:
    """Full stage 1: walk history, resolve file statuses, classify, mark candidates."""
    repo = GitRepo(repo_path)
    commits = repo.log(first_parent=first_parent, limit=limit)
    for c in commits:
        if refine_status and c.files:
            status = repo.name_status(c.sha)
            for f in c.files:
                f.status = status.get(f.path, "M")
        classify(c)
        mark_gate_candidate(c)
    return commits
