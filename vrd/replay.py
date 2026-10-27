"""Stage 2 (REPLAY) and stage 3 (GATE).

Replay answers: "what did this repository actually do at commit C-1?" It checks
out the parent, reconstructs the toolchain, and runs tests — inside a hermetic
environment, because a replay that inherits the operator's PYTHONPATH is not a
replay.

Gate answers the question every other "fine-tune on your repo" tool skips:
"was this commit's training pair ever *correct*?" We answer it by counterfactual
test injection:

    take the test the commit adds, run it against the unpatched parent  -> must FAIL
    apply the commit's full patch, run the same test                    -> must PASS

A commit that survives both is a pair where we have mechanically observed a
behaviour change. A commit that fails the first half only added a test for
something that already worked. A commit that fails the second half was never
green to begin with.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from vrd.ingest import Commit, GitRepo, is_test_path

PYTEST_TIMEOUT = 300          # seconds per test invocation
INSTALL_TIMEOUT = 900
DEFAULT_PYTHON = "3.11"

# Environment presets. A repo whose test suite needs particular variables
# declares them here rather than the harness guessing medium-dependent config.
ENV_PRESETS: dict[str, dict[str, str]] = {
    # hermes-webui's tests/conftest.py boots a server and, left alone, discovers
    # ~/.hermes/hermes-agent, runs server.py with the wrong cwd, and leaks a
    # foreign site-packages into sys.path. Pinning the agent dir to an empty
    # sandbox makes WORKDIR fall back to the repo root and makes agent-dependent
    # tests skip deterministically instead of erroring.
    "hermes-webui": {
        "HERMES_WEBUI_AGENT_DIR": "{sandbox}/no-agent",
        "HERMES_HOME": "{sandbox}",
        "HERMES_BASE_HOME": "{sandbox}",
        "HERMES_WEBUI_STATE_DIR": "{sandbox}/state",
        "HERMES_WEBUI_TEST_STATE_DIR": "{sandbox}/state",
    },
}


@dataclass
class TestResult:
    ran: bool
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    returncode: int | None = None
    output_tail: str = ""

    @property
    def green(self) -> bool:
        return self.ran and self.returncode == 0 and self.failed == 0 and self.errors == 0


@dataclass
class Verdict:
    sha: str
    subject: str
    status: str                     # verified | reject_pre_passed | reject_post_failed | reject_replay
    detail: str = ""
    pre: TestResult | None = None
    post: TestResult | None = None
    test_files: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    wall_s: float = 0.0


# ── Environment reconstruction ─────────────────────────────────────────────


def _hash_files(tree: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = tree / name
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def discover_python_deps(tree: Path) -> tuple[str, list[str]]:
    """Return (installer, files) for a Python project at this tree."""
    for req in ("requirements.txt", "requirements-dev.txt"):
        if (tree / req).exists():
            return "requirements", [req]
    if (tree / "pyproject.toml").exists():
        return "pyproject", ["pyproject.toml"] + (["uv.lock"] if (tree / "uv.lock").exists() else [])
    if (tree / "poetry.lock").exists():
        return "poetry", ["poetry.lock", "pyproject.toml"]
    return "none", []


def build_venv(tree: Path, cache_root: Path, python: str = DEFAULT_PYTHON) -> tuple[Path, str]:
    """Create or reuse a virtualenv for this tree's declared dependencies.

    Cached by the content hash of the dependency files, so sibling commits that
    share a requirements.txt share one install.
    """
    installer, files = discover_python_deps(tree)
    key = f"{installer}-{_hash_files(tree, files)}" if files else f"{installer}-bare"
    venv = cache_root / "venvs" / key
    marker = venv / ".vrd-ready"
    if marker.exists():
        return venv, key

    venv.parent.mkdir(parents=True, exist_ok=True)
    if venv.exists():
        shutil.rmtree(venv)
    _run(["uv", "venv", "-q", str(venv), "--python", python], timeout=INSTALL_TIMEOUT)
    py = venv / "bin/python"

    if installer in ("requirements", "pyproject"):
        args = ["uv", "pip", "install", "-q", "--python", str(py)]
        for f in files:
            if f == "uv.lock":
                continue
            args += ["-r", str(tree / f)] if f != "pyproject.toml" else [str(tree / f)]
        proc = _run(args, timeout=INSTALL_TIMEOUT, check=False)
        if proc.returncode != 0:
            # a partial install still beats abandoning the commit outright
            (venv / "install.log").write_text(proc.stderr[-4000:])
    elif installer == "poetry":
        _run(["uv", "pip", "install", "-q", "--python", str(py), "poetry"], timeout=INSTALL_TIMEOUT, check=False)

    # pytest is part of the harness, not the project
    _run(["uv", "pip", "install", "-q", "--python", str(py), "pytest"], timeout=INSTALL_TIMEOUT, check=False)
    marker.write_text(key)
    return venv, key


# ── Process plumbing ───────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int, cwd: Path | None = None,
         env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed rc={proc.returncode}: {proc.stderr[-800:]}")
    return proc


def hermetic_env(sandbox: Path, preset: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    """A deliberately small environment. Nothing is inherited from the operator."""
    env = {
        "HOME": str(sandbox),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(sandbox / "tmp"),
    }
    (sandbox / "tmp").mkdir(parents=True, exist_ok=True)
    if preset and preset in ENV_PRESETS:
        for k, v in ENV_PRESETS[preset].items():
            env[k] = v.format(sandbox=sandbox)
    if extra:
        env.update(extra)
    return env


def parse_pytest_summary(output: str, duration: float, rc: int) -> TestResult:
    """Pull pass/fail/error/skip counts out of pytest's last summary line."""
    import re

    res = TestResult(ran=True, duration_s=duration, returncode=rc)
    tail = output.strip().splitlines()[-1] if output.strip() else ""
    for count, label in re.findall(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)", tail):
        n = int(count)
        if label == "passed":
            res.passed = n
        elif label == "failed":
            res.failed = n
        elif label.startswith("error"):
            res.errors = n
        elif label == "skipped":
            res.skipped = n
    if res.passed == res.failed == res.errors == 0 and rc != 0:
        # collection error or hard crash
        res.errors = 1
    res.output_tail = "\n".join(output.strip().splitlines()[-12:])
    return res


def run_pytest(tree: Path, venv: Path, test_paths: list[str], sandbox: Path,
               preset: str | None, timeout: int = PYTEST_TIMEOUT) -> TestResult:
    env = hermetic_env(sandbox, preset, {"PYTHONPATH": str(tree)})
    cmd = [str(venv / "bin/python"), "-m", "pytest", *test_paths, "-q", "-p", "no:cacheprovider"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        res = parse_pytest_summary(out if isinstance(out, str) else out.decode(errors="replace"),
                                   time.time() - t0, 124)
        res.output_tail = f"TIMEOUT after {timeout}s\n" + res.output_tail
        return res
    out = proc.stdout + proc.stderr
    return parse_pytest_summary(out, time.time() - t0, proc.returncode)


def detect_toolchain(tree: Path) -> str:
    """Which runner does this tree speak? Order matters: a Go repo with a stray
    requirements.txt for docs tooling must still be tested as Go."""
    if (tree / "go.mod").exists():
        return "go"
    if any((tree / f).exists() for f in
           ("requirements.txt", "pyproject.toml", "setup.py", "poetry.lock")):
        return "python"
    if (tree / "package.json").exists():
        return "node"
    return "unknown"


# ── Go toolchain (CNCF repos) ──────────────────────────────────────────────

GO_PATH = "/usr/local/go/bin:/usr/bin:/bin"


def go_packages(test_paths: list[str]) -> list[str]:
    """Turn test file paths into the package patterns `go test` understands."""
    pkgs = set()
    for p in test_paths:
        parent = Path(p).parent
        pkgs.add("." if str(parent) in (".", "") else "./" + parent.as_posix())
    return sorted(pkgs)


def run_go_tests(tree: Path, test_paths: list[str], sandbox: Path, cache_root: Path,
                 timeout: int = 900) -> TestResult:
    """`go test -count=1 <pkgs>` in a build cache shared across commits.

    The module cache is shared deliberately: it is a pure download cache, so
    sharing it changes speed and nothing else. The build cache is per-sandbox.
    """
    env = hermetic_env(sandbox, None)
    env.update({
        "PATH": GO_PATH,
        "HOME": str(sandbox),
        "GOPATH": str(sandbox / "gopath"),
        "GOCACHE": str(cache_root / "gocache"),
        "GOMODCACHE": str(cache_root / "gomodcache"),
        "GOFLAGS": "-mod=mod",
        "GOPROXY": "https://proxy.golang.org,direct",
        # auto, not local: a repo pinned to a newer Go must pull that toolchain.
        # Pinning `local` turned every coredns commit into "go.mod requires
        # go >= 1.26.0" — a harness artefact reported as a test failure.
        "GOTOOLCHAIN": "auto",
    })
    (cache_root / "gocache").mkdir(parents=True, exist_ok=True)
    (cache_root / "gomodcache").mkdir(parents=True, exist_ok=True)
    cmd = ["go", "test", "-count=1", *go_packages(test_paths)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        res = parse_go_summary(out if isinstance(out, str) else out.decode(errors="replace"),
                               time.time() - t0, 124)
        res.output_tail = f"TIMEOUT after {timeout}s\n" + res.output_tail
        return res
    return parse_go_summary(proc.stdout + proc.stderr, time.time() - t0, proc.returncode)


def parse_go_summary(output: str, duration: float, rc: int) -> TestResult:
    import re as _re

    res = TestResult(ran=True, duration_s=duration, returncode=rc)
    res.passed = len(_re.findall(r"^--- PASS: ", output, _re.M))
    res.failed = len(_re.findall(r"^--- FAIL: ", output, _re.M))
    res.skipped = len(_re.findall(r"^--- SKIP: ", output, _re.M))
    build_failed = "[build failed]" in output or "cannot find package" in output
    if build_failed:
        # A build failure is a real gate signal: the injected test references
        # something that does not exist at the parent. Count it as a failure.
        res.failed = max(res.failed, 1)
    if rc != 0 and res.passed == res.failed == 0:
        res.errors = 1
    res.output_tail = "\n".join(output.strip().splitlines()[-12:])
    return res


def run_tests(tree: Path, test_paths: list[str], sandbox: Path, preset: str | None,
              cache_root: Path) -> tuple[TestResult, str]:
    """Dispatch to the right runner. Returns (result, env_description)."""
    toolchain = detect_toolchain(tree)
    if toolchain == "go":
        return run_go_tests(tree, test_paths, sandbox, cache_root), "go-modcache-shared"
    if toolchain == "python":
        venv, key = build_venv(tree, cache_root)
        return run_pytest(tree, venv, test_paths, sandbox, preset), f"python-{key}"
    return TestResult(ran=False, returncode=None, output_tail=f"unsupported toolchain: {toolchain}"), "none"


# ── Worktrees ──────────────────────────────────────────────────────────────


class Worktree:
    """A throwaway checkout of one commit, removed on exit."""

    def __init__(self, repo: GitRepo, sha: str, root: Path):
        self.repo = repo
        self.sha = sha
        self.root = root

    def __enter__(self) -> Path:
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        _run(["git", "-C", str(self.repo.path), "worktree", "add", "--detach",
              "--force", str(self.root), self.sha], timeout=300)
        return self.root

    def __exit__(self, *exc) -> None:
        _run(["git", "-C", str(self.repo.path), "worktree", "remove", "--force",
              str(self.root)], timeout=300, check=False)
        shutil.rmtree(self.root, ignore_errors=True)


def _git_show_to_file(repo: GitRepo, sha: str, path: str, dest: Path) -> None:
    content = repo._git("show", f"{sha}:{path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")


# ── The gate ───────────────────────────────────────────────────────────────


def counterfactual_verify(repo: GitRepo, commit: Commit, cache_root: Path,
                          scratch_root: Path, preset: str | None = "hermes-webui",
                          keep: bool = False) -> Verdict:
    """Run the two-sided gate on one candidate commit."""
    t_start = time.time()
    tests = [p for p in commit.added_tests if is_test_path(p)]
    verdict = Verdict(
        sha=commit.sha,
        subject=commit.subject,
        status="reject_replay",
        test_files=tests,
        source_files=commit.source_paths,
    )
    if not tests or not commit.parents:
        verdict.detail = "no added test or no parent"
        return verdict

    parent = commit.parents[0]
    sandbox = scratch_root / "sandbox" / commit.sha[:12]
    sandbox.mkdir(parents=True, exist_ok=True)
    tree_root = scratch_root / "trees" / commit.sha[:12]

    try:
        with Worktree(repo, parent, tree_root) as tree:
            # ── pre-arm: inject the commit's new tests, unpatched source ──
            injected: list[str] = []
            for t in tests:
                dest = tree / t
                _git_show_to_file(repo, commit.sha, t, dest)
                injected.append(t)
            pre, env_desc = run_tests(tree, injected, sandbox, preset, cache_root)
            verdict.detail = f"env={env_desc}"
            verdict.pre = pre

            # ── post-arm: full patch applied, same tests ──
            # The injected copies must go first: the commit's own patch adds
            # these same files, and `git apply` refuses to overwrite them.
            for t in injected:
                (tree / t).unlink(missing_ok=True)
            patch = repo.patch(commit.sha)
            apply_proc = subprocess.run(
                ["git", "-C", str(tree), "apply", "--whitespace=nowarn", "-"],
                input=patch, text=True, capture_output=True,
            )
            if apply_proc.returncode != 0:
                verdict.status = "reject_replay"
                verdict.detail = f"patch did not apply: {apply_proc.stderr.strip()[:200]}"
                verdict.wall_s = time.time() - t_start
                return verdict

            post, _ = run_tests(tree, injected, sandbox, preset, cache_root)
            verdict.post = post

        if pre.green:
            verdict.status = "reject_pre_passed"
        elif not post.green:
            verdict.status = "reject_post_failed"
        else:
            verdict.status = "verified"
    except Exception as exc:  # replay failure must never look like a verification
        verdict.status = "reject_replay"
        verdict.detail = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        verdict.wall_s = time.time() - t_start
        if not keep:
            shutil.rmtree(sandbox, ignore_errors=True)

    return verdict


def verify_many(repo_path: str | Path, commits: list[Commit], cache_root: str | Path,
                scratch_root: str | Path, preset: str | None = "hermes-webui",
                limit: int | None = None, progress=None) -> list[Verdict]:
    repo = GitRepo(repo_path)
    cache_root = Path(cache_root)
    scratch_root = Path(scratch_root)
    results: list[Verdict] = []
    todo = [c for c in commits if c.candidate][: limit or None]
    for i, commit in enumerate(todo, 1):
        verdict = counterfactual_verify(repo, commit, cache_root, scratch_root, preset=preset)
        results.append(verdict)
        if progress:
            progress(i, len(todo), verdict)
    return results


def write_verdicts(verdicts: list[Verdict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for v in verdicts:
            row = asdict(v)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
