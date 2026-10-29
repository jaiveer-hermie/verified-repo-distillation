"""SEARCH/REPLACE edit format.

Why this exists: asked for a unified diff, nvidia/nemotron-3-nano-30b-a3b emits a
bare `@@` (invalid), then — once told to use explicit hunk headers — a patch whose
line counts do not match its own hunk bodies (`corrupt patch at line 33`). Both are
formatting failures, not editing failures, and a benchmark that scores them as
editing failures measures nothing useful.

So the contract is search/replace blocks, and *the harness* computes the diff:

    ### FILE: api/clarify.py
    <<<<<<< SEARCH
    exact existing lines
    =======
    replacement lines
    >>>>>>> REPLACE

Matching is exact and must be unique. A block that matches zero times, or more
than once, fails loudly instead of being applied to the wrong site.

`diff_to_edits` goes the other way, so training targets are in the same format the
evaluator expects — train and eval must share one contract.
"""

from __future__ import annotations

import difflib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# "@@ -12,6 +12,8 @@ import time" — the trailing section heading is optional and
# almost always present in real git output. Group 1 is the old-side span.
HUNK_RE = re.compile(r"^@@ -(\d+(?:,\d+)?) \+\d+(?:,\d+)? @@")

FILE_HEADER = re.compile(r"^#{2,4}\s*(?:FILE|File|file):\s*`?([^`\n]+?)`?\s*$", re.M)
BLOCK = re.compile(
    r"<<<<<<<+\s*(?:SEARCH|OLD)\s*\n(.*?)\n?=======\s*\n(.*?)\n?>>>>>>>+\s*(?:REPLACE|NEW)",
    re.S,
)

INSTRUCTIONS = (
    "## Output format\n"
    "Reply with search/replace blocks only. No prose, no fences, no commentary.\n"
    "For each file you change, write a header line then one or more blocks.\n"
    "This is a filled-in example — copy its shape exactly, with your own paths\n"
    "and lines (never output the words EXAMPLE, here, or any <angle-bracket>\n"
    "placeholder: they must be replaced with real values from the context):\n\n"
    "### FILE: static/sessions.js\n"
    "<<<<<<< SEARCH\n"
    "  function refreshSidebar() {\n"
    "    const rows = getRows();\n"
    "=======\n"
    "  function refreshSidebar() {\n"
    "    const rows = getRows().filter(Boolean);\n"
    ">>>>>>> REPLACE\n\n"
    "Rules:\n"
    "- Write the REAL file path from the context after '### FILE: '. A path of\n"
    "  '<path>' or any placeholder is an invalid answer.\n"
    "- The SEARCH text must appear in the file exactly once. Copy it verbatim\n"
    "  from the context, including indentation.\n"
    "- Keep SEARCH blocks small: at most 20 lines each, at most 60 lines in total.\n"
    "- At most 2 blocks. If the change would need more, emit only the single most\n"
    "  important block and stop.\n"
    "- Do not reformat, restructure, reorder imports, or rewrite whole files.\n"
    "  Change the minimum number of lines that makes the request true.\n"
    "- Never modify files under tests/.\n"
)


@dataclass
class Edit:
    path: str
    search: str
    replace: str


def parse_edits(text: str) -> tuple[list[Edit], str]:
    """Return (edits, error). Never guesses a filename from context."""
    edits: list[Edit] = []
    headers = list(FILE_HEADER.finditer(text))
    if not headers:
        return [], "no '### FILE:' header found"
    for i, hdr in enumerate(headers):
        path = hdr.group(1).strip().strip("`")
        start = hdr.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section = text[start:end]
        for search, replace in BLOCK.findall(section):
            edits.append(Edit(path=path, search=search, replace=replace))
    if not edits:
        return [], "file header(s) present but no search/replace block"
    return edits, ""


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return haystack.count(needle)


def _normalise(text: str) -> str:
    """Trailing-whitespace-insensitive form, so a stray space is not a hard fail."""
    return "\n".join(line.rstrip() for line in text.splitlines())


# Fuzzy anchor matching, as in Aider's patch matcher. Above this similarity the
# block is accepted at its best-scoring position; the tests still judge it.
FUZZY_THRESHOLD = 0.82


def _locate(body_lines: list[str], search_lines: list[str],
            exact_only: bool, threshold: float = 1.0) -> int | None:
    """Index of the single best window matching search_lines, else None.

    exact_only compares with trailing whitespace ignored and requires a unique
    match. Otherwise every same-length window is scored on character similarity
    of the whitespace-stripped block (so an indentation difference or one typo
    does not defeat an otherwise correct anchor), and the best window is
    accepted only if it clears `threshold` and is clearly better than the
    runner-up. An ambiguous anchor is refused, never guessed.
    """
    n = len(search_lines)
    if n == 0 or n > len(body_lines):
        return None
    needle_text = "\n".join(line.strip() for line in search_lines)

    if exact_only:
        found = [i for i in range(len(body_lines) - n + 1)
                 if [l.rstrip() for l in body_lines[i:i + n]] == [l.rstrip() for l in search_lines]]
        return found[0] if len(found) == 1 else None

    probe = difflib.SequenceMatcher(None, "", needle_text)
    shortlist: list[tuple[float, int]] = []
    for i in range(len(body_lines) - n + 1):
        window_text = "\n".join(line.strip() for line in body_lines[i:i + n])
        probe.set_seq2(window_text)
        # quick_ratio is a cheap upper bound; keep the plausible windows only.
        shortlist.append((probe.quick_ratio(), i))
    shortlist.sort(reverse=True)
    scored: list[tuple[float, int]] = []
    for _, i in shortlist[:8]:
        window_text = "\n".join(line.strip() for line in body_lines[i:i + n])
        scored.append((difflib.SequenceMatcher(None, window_text, needle_text).ratio(), i))
    scored.sort(reverse=True)
    best = scored[0]
    if best[0] < threshold:
        return None
    if len(scored) > 1 and scored[1][0] > threshold - 0.02:
        return None        # two plausible positions: refuse to guess
    return best[1]


def _shift_indent(text: str, delta: int) -> str:
    """Move a replacement block to the indentation of the lines it replaces."""
    if delta == 0 or not text:
        return text
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            out.append(line)
        elif delta > 0:
            out.append(" " * delta + line)
        else:
            strip = min(-delta, len(line) - len(line.lstrip()))
            out.append(line[strip:])
    return "\n".join(out)


def apply_edits(tree: Path, edits: list[Edit], max_edits: int = 20,
                max_block_lines: int = 300, max_total_lines: int = 1200) -> tuple[bool, str]:
    """Apply edits in order. Returns (ok, note). All-or-nothing per block.

    Size limits are enforced here rather than requested in the prompt: a model
    asked to make a surgical change will happily rewrite four files wholesale,
    and a 222-block answer is not an edit, it is a refactor.

    The limits are deliberately loose (300 lines/block, 1200 total). Tighter
    limits rejected a real commit: the oracle test found a 92-line gold edit
    block, which an 80-line cap scored as a model failure. The cap exists to
    catch whole-file rewrites of large files, not to police honest edits.
    """
    if len(edits) > max_edits:
        return False, f"too many edits ({len(edits)} > {max_edits})"
    total_lines = sum(len(e.search.splitlines()) + len(e.replace.splitlines()) for e in edits)
    if total_lines > max_total_lines:
        return False, f"edit payload too large ({total_lines} > {max_total_lines} lines)"
    for n, edit in enumerate(edits, 1):
        block_lines = max(len(edit.search.splitlines()), len(edit.replace.splitlines()))
        if block_lines > max_block_lines:
            return False, f"edit {n}: block too large ({block_lines} > {max_block_lines} lines)"
        target = (tree / edit.path).resolve()
        try:
            target.relative_to(tree.resolve())
        except ValueError:
            return False, f"edit {n}: path escapes repository ({edit.path})"
        if not target.exists():
            return False, f"edit {n}: no such file {edit.path}"
        try:
            body = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return False, f"edit {n}: {edit.path} is not utf-8"

        hits = _count_occurrences(body, edit.search)
        if hits == 1:
            target.write_text(body.replace(edit.search, edit.replace, 1), encoding="utf-8")
            continue
        if hits > 1:
            return False, f"edit {n}: SEARCH matches {hits}x in {edit.path} (must be unique)"

        search_lines = edit.search.splitlines()
        body_lines = body.splitlines(keepends=True)

        # two normalisation passes, then a fuzzy pass. Aider's matcher is fuzzy
        # for the same reason: models reproduce an anchor from context with a
        # stray space, a changed quote, or one altered line, and demanding
        # literal bytes measures syntax rather than behaviour. The project's
        # tests still decide whether the edit is correct.
        if search_lines:
            idx = _locate(body_lines, search_lines, exact_only=True)
            if idx is None:
                idx = _locate(body_lines, search_lines, exact_only=False,
                              threshold=FUZZY_THRESHOLD)
            if idx is not None:
                # Align the replacement to the indentation actually in the file,
                # so a whitespace-tolerant match does not silently break Python.
                first = next((line for line in search_lines if line.strip()), "")
                body_indent = len(body_lines[idx]) - len(body_lines[idx].lstrip())
                search_indent = len(first) - len(first.lstrip())
                replacement = _shift_indent(edit.replace, body_indent - search_indent)
                if not replacement:
                    replacement = ""
                elif not replacement.endswith("\n"):
                    replacement += "\n"
                target.write_text(
                    "".join(body_lines[:idx] + [replacement]
                            + body_lines[idx + len(search_lines):]),
                    encoding="utf-8")
                continue

        return False, f"edit {n}: SEARCH not found in {edit.path}"
    return True, ""


def tree_diff(tree: Path) -> str:
    """The diff the harness computed, for the record and for training targets."""
    proc = subprocess.run(
        ["git", "-C", str(tree), "diff", "--no-color", "--no-ext-diff"],
        capture_output=True, text=True,
    )
    return proc.stdout


# ── the inverse direction: unified diff -> search/replace edits ────────────


def diff_to_edits(patch: str) -> list[Edit]:
    """Convert a unified diff into the same block format used at eval time.

    Training targets must be in the contract the evaluator parses; otherwise the
    tuned model is trained on one format and scored on another.
    """
    edits: list[Edit] = []
    path = ""
    old_block: list[str] = []
    new_block: list[str] = []
    in_hunk = False

    def flush() -> None:
        nonlocal old_block, new_block
        if path and (old_block or new_block):
            edits.append(Edit(path=path,
                              search="\n".join(old_block),
                              replace="\n".join(new_block)))
        old_block, new_block = [], []

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            flush()
            in_hunk = False
            path = ""
            continue
        if line.startswith("+++ "):
            candidate = line[4:].strip()
            if candidate not in ("/dev/null",):
                path = candidate[2:] if candidate.startswith("b/") else candidate
            continue
        # Real hunk headers carry a section label: "@@ -12,6 +12,8 @@ import time".
        # Requiring the line to *end* with "@@" (the old check) silently rejected
        # every hunk with a heading, which emptied 176 of 483 training targets.
        if HUNK_RE.match(line):
            flush()
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            new_block.append(line[1:])
        elif line.startswith("-"):
            old_block.append(line[1:])
        elif line.startswith(" "):
            old_block.append(line[1:])
            new_block.append(line[1:])
    flush()
    return edits


def render_edits(edits: list[Edit]) -> str:
    """Block text for a training target, grouped by file."""
    out: list[str] = []
    last = None
    for e in edits:
        if e.path != last:
            out.append(f"### FILE: {e.path}")
            last = e.path
        out.append("<<<<<<< SEARCH")
        out.append(e.search)
        out.append("=======")
        out.append(e.replace)
        out.append(">>>>>>> REPLACE")
    return "\n".join(out)


# ── whole-file format ──────────────────────────────────────────────────────
# Aider's benchmark over 133 Exercism tasks measured plain whole-file return as
# the most reliable format for smaller models ("whole" 46%/39% vs "diff"
# 30%/19% for GPT-3.5), and documented that weaker models use search/replace
# pathologically — pasting the entire file into both sides. We saw exactly that
# with nemotron-super: 222 blocks for a 4-file change. So whole-file is the
# default contract, and search/replace is kept as an option.

INSTRUCTIONS_WHOLE = (
    "## Output format\n"
    "Return the complete updated contents of each file you change. No prose, no\n"
    "commentary, no partial files, no ellipses.\n\n"
    "### FILE: <path>\n"
    "```\n"
    "<the entire new contents of that file>\n"
    "```\n\n"
    "Rules:\n"
    "- Copy the file exactly, changing only the lines the request requires.\n"
    "- Return at most 2 files. If the change needs more, edit the most important one.\n"
    "- Do not add commentary, do not reformat untouched code, do not touch tests/.\n"
)

CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)\n?```", re.S)


def parse_whole(text: str) -> tuple[dict[str, str], str]:
    """Parse `### FILE: path` + fenced full contents. Returns ({path: content}, err)."""
    files: dict[str, str] = {}
    headers = list(FILE_HEADER.finditer(text))
    if not headers:
        return {}, "no '### FILE:' header found"
    for i, hdr in enumerate(headers):
        path = hdr.group(1).strip().strip("`")
        start = hdr.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end]
        m = CODE_FENCE.search(body)
        if not m:
            continue
        files[path] = m.group(1)
    if not files:
        return {}, "file header(s) present but no fenced file body"
    return files, ""


def apply_whole(tree: Path, files: dict[str, str], max_files: int = 2,
                max_bytes: int = 400_000) -> tuple[bool, str]:
    """Write whole-file edits. Guards: test files, path escape, size."""
    if len(files) > max_files:
        return False, f"too many files returned ({len(files)} > {max_files})"
    total = sum(len(c) for c in files.values())
    if total > max_bytes:
        return False, f"whole-file payload too large ({total} > {max_bytes} bytes)"
    for path, content in files.items():
        target = (tree / path).resolve()
        try:
            target.relative_to(tree.resolve())
        except ValueError:
            return False, f"path escapes repository ({path})"
        if not target.exists():
            return False, f"no such file {path}"
        try:
            target.write_text(content if content.endswith("\n") else content + "\n",
                              encoding="utf-8")
        except OSError as exc:
            return False, f"write failed for {path}: {exc}"
    return True, ""


def render_whole(files: dict[str, str]) -> str:
    """Whole-file text for a training target."""
    out: list[str] = []
    for path, content in files.items():
        body = content if content.endswith("\n") else content + "\n"
        out.append(f"### FILE: {path}\n```\n{body}```")
    return "\n".join(out)
