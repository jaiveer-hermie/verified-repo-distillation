"""Stage 6 — THREE-ARM EVAL.

The comparison everyone else runs is base-vs-tuned. It is the wrong comparison,
because it cannot tell you whether the gain came from the repository's
information or from the adapter. So we run three arms on identical held-out
commits with identical scoring:

    A  base model, minimal context        (what arm C sees)
    B  base model, retrieved context      (the whole repo is searchable)
    C  tuned model, minimal context       (same prompt bytes as A)

If C beats B, weights won. If B beats C, retrieval won and that is the finding —
shipped, not buried. If A and C tie, the adapter learned nothing and we say so.

Scoring is execution, never a judge: the produced patch must apply to the
parent tree AND the commit's own test must pass afterwards. Both are mechanical.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

from vrd.pairs import Pair
from vrd.replay import (TestResult, Worktree, hermetic_env, run_tests)
from vrd.ingest import GitRepo

# ── Model client (Nebius Token Factory is OpenAI-compatible) ───────────────

TOKEN_FACTORY_URL = os.getenv(
    "NEBIUS_TOKEN_FACTORY_URL",
    "https://api.tokenfactory.nebius.com/v1/chat/completions",
)
DEFAULT_MODEL = os.getenv("VRD_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")


@dataclass
class Generation:
    text: str
    model: str
    arm: str
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


class ChatClient:
    """Minimal OpenAI-compatible client. Deliberately dependency-free.

    Set NEBIUS_API_KEY to use Token Factory; set VRD_DRY_RUN=1 to score the
    harness end to end without spending tokens (returns empty patches, which
    the evaluator counts as failures — a real number, not a fake one).
    """

    def __init__(self, api_key: str | None = None, url: str = TOKEN_FACTORY_URL):
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY", "")
        self.url = url
        self.dry_run = os.getenv("VRD_DRY_RUN") == "1" or not self.api_key

    def complete(self, prompt: str, model: str = DEFAULT_MODEL, arm: str = "A",
                 max_tokens: int = 1400, temperature: float = 0.0) -> Generation:
        if self.dry_run:
            return Generation(text="", model="dry-run", arm=arm,
                              error="NEBIUS_API_KEY not set (dry run)")
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            return Generation(text="", model=model, arm=arm,
                              latency_s=time.time() - t0, error=str(exc)[:300])
        choice = (payload.get("choices") or [{}])[0]
        usage = payload.get("usage", {})
        return Generation(
            text=(choice.get("message") or {}).get("content", ""),
            model=model, arm=arm, latency_s=time.time() - t0,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


# ── Patch extraction ───────────────────────────────────────────────────────

FENCE = re.compile(r"```(?:diff|patch)?\n(.*?)```", re.S)


def extract_diff(text: str) -> str:
    """Models wrap diffs in fences, prose, or nothing. Recover the diff only."""
    if not text.strip():
        return ""
    m = FENCE.search(text)
    body = m.group(1) if m else text
    idx = body.find("diff --git")
    if idx == -1:
        idx = body.find("--- ")
        if idx == -1:
            return ""
    body = body[idx:]
    lines = []
    for line in body.splitlines():
        if line.startswith("```"):
            break
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


# ── Arm B: retrieval ───────────────────────────────────────────────────────


class BM25:
    """BM25 over repo files at the held-out commit's parent.

    Deterministic, free, no embeddings to justify. If retrieval wins here it
    wins against the *weakest* reasonable retrieval, which makes the finding
    conservative rather than flattering.
    """

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.paths_by_doc = docs
        self.paths = list(docs)
        self.docs = [self._tok(docs[p]) for p in self.paths]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        self.df: dict[str, int] = {}
        for d in self.docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1
        self.N = len(self.docs)

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text.lower())

    def search(self, query: str, k: int = 6) -> list[tuple[str, float]]:
        q = self._tok(query)
        scored: list[tuple[str, float]] = []
        for path, doc in zip(self.paths, self.docs):
            if not doc:
                continue
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
                score += idf * tf[term] * (self.k1 + 1) / denom
            if score > 0:
                scored.append((path, score))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:k]


def repo_docs(repo: GitRepo, sha: str, max_files: int = 4000, max_bytes: int = 400_000) -> dict[str, str]:
    """Read the repo's source files at a tree, bounded."""
    listing = repo._git("ls-tree", "-r", "--name-only", sha).splitlines()
    docs: dict[str, str] = {}
    total = 0
    for path in listing:
        if not path.endswith((".py", ".go", ".js", ".ts", ".tsx", ".md")):
            continue
        if path.startswith(("node_modules/", "vendor/", "static/")):
            continue
        try:
            body = repo.show_file(sha, path)
        except RuntimeError:
            continue
        if len(body) > max_bytes // 10:
            continue
        docs[path] = body
        total += len(body)
        if total > max_bytes or len(docs) >= max_files:
            break
    return docs


# ── Scoring ────────────────────────────────────────────────────────────────


@dataclass
class Score:
    sha: str
    arm: str
    generated_bytes: int
    applies: bool
    tests_pass: bool
    test_result: dict = field(default_factory=dict)
    note: str = ""

    @property
    def success(self) -> bool:
        return self.applies and self.tests_pass


def score_patch(repo: GitRepo, pair: Pair, diff: str, cache_root: Path,
                scratch_root: Path, preset: str | None = "hermes-webui") -> Score:
    """Apply the model's diff to the parent tree and run the commit's own test."""
    if not diff.strip():
        return Score(sha=pair.sha, arm="", generated_bytes=0, applies=False,
                     tests_pass=False, note="empty diff")
    parent = repo._git("rev-parse", f"{pair.sha}^").strip()
    sandbox = scratch_root / "eval" / f"{pair.sha[:10]}"
    sandbox.mkdir(parents=True, exist_ok=True)
    tree_root = scratch_root / "eval-trees" / pair.sha[:10]
    try:
        with Worktree(repo, parent, tree_root) as tree:
            # inject the commit's test (post-arm only) — never the source patch
            for t in pair.test_files:
                dest = tree / t
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(repo.show_file(pair.sha, t), encoding="utf-8")
            proc = subprocess.run(
                ["git", "-C", str(tree), "apply", "--whitespace=nowarn", "-"],
                input=diff, text=True, capture_output=True,
            )
            if proc.returncode != 0:
                return Score(sha=pair.sha, arm="", generated_bytes=len(diff),
                             applies=False, tests_pass=False,
                             note=proc.stderr.strip()[:200])
            # Same dispatcher the gate uses: Python via uv, Go via module cache.
            # An eval that scores every Go pair as "no python env" would report a
            # 0% pass rate that belongs to the harness, not to the model.
            res, env_desc = run_tests(tree, pair.test_files, sandbox, preset, cache_root)
            if not res.ran:
                return Score(sha=pair.sha, arm="", generated_bytes=len(diff),
                             applies=True, tests_pass=False, note=env_desc)
            return Score(sha=pair.sha, arm="", generated_bytes=len(diff),
                         applies=True, tests_pass=res.green, test_result=asdict(res))
    finally:
        import shutil
        shutil.rmtree(sandbox, ignore_errors=True)


# ── Arm runner ─────────────────────────────────────────────────────────────


def run_arm(repo_path: str | Path, pairs: list[Pair], arm: str, client: ChatClient,
            cache_root: Path, scratch_root: Path, model: str = DEFAULT_MODEL,
            limit: int | None = None, adapter: str | None = None,
            preset: str | None = "hermes-webui") -> list[Score]:
    repo = GitRepo(repo_path)
    scores: list[Score] = []
    docs_cache: dict[str, BM25] = {}
    for pair in pairs[: limit or None]:
        if arm == "B":
            parent = repo._git("rev-parse", f"{pair.sha}^").strip()
            if parent not in docs_cache:
                docs = repo_docs(repo, parent)
                docs_cache[parent] = BM25(docs)
            bm25 = docs_cache[parent]
            query = pair.subject + " " + " ".join(pair.changed_files)
            hits = bm25.search(query, k=6)
            blocks = "\n".join(
                f"### {p}\n```\n{bm25.paths_by_doc[p][:6000]}\n```\n" for p, _ in hits
            )
            prompt = (
                "You are editing an existing repository. Apply the smallest change "
                "that implements the request below.\n\n"
                f"## Request\n{pair.subject}\n\n"
                f"## Retrieved repository context\n{blocks}\n\n"
                "## Output\nReturn a unified diff only. No prose, no fences.\n"
            )
        else:
            prompt = pair.to_prompt()

        gen = client.complete(prompt, model=model, arm=arm)
        diff = extract_diff(gen.text)
        score = score_patch(repo, pair, diff, cache_root, scratch_root, preset)
        score.arm = arm
        score.test_result.setdefault("generation", {})
        score.test_result["generation"] = {
            "latency_s": round(gen.latency_s, 2),
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "error": gen.error,
            "adapter": adapter,
            "prompt_sha8": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:8],
        }
        scores.append(score)
    return scores


# ── Aggregate reporting ────────────────────────────────────────────────────


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(all_scores: dict[str, list[Score]]) -> dict:
    out = {"arms": {}, "n_commits": {arm: len(s) for arm, s in all_scores.items()}}
    for arm, scores in all_scores.items():
        n = len(scores)
        ok = sum(1 for s in scores if s.success)
        applies = sum(1 for s in scores if s.applies)
        lo, hi = wilson_interval(ok, n)
        out["arms"][arm] = {
            "n": n,
            "passed": ok,
            "pass_rate": round(ok / n, 4) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)],
            "applies_rate": round(applies / n, 4) if n else 0.0,
            "mean_generated_bytes": round(sum(s.generated_bytes for s in scores) / n, 1) if n else 0,
        }
    return out


def write_report(summary: dict, path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def markdown_table(summary: dict) -> str:
    lines = [
        "| Arm | Context | n | passed | pass rate | 95% CI | applies |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {"A": "base model, minimal context",
              "B": "base model, retrieved context",
              "C": "tuned model, minimal context"}
    for arm in ("A", "B", "C"):
        a = summary["arms"].get(arm)
        if not a:
            continue
        lo, hi = a["ci95"]
        lines.append(
            f"| {arm} | {labels[arm]} | {a['n']} | {a['passed']} | "
            f"{a['pass_rate']*100:.1f}% | [{lo*100:.1f}, {hi*100:.1f}] | "
            f"{a['applies_rate']*100:.1f}% |"
        )
    return "\n".join(lines)
