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
from vrd.ingest import GitRepo, is_test_path
from vrd.edit_format import (INSTRUCTIONS as EDIT_INSTRUCTIONS, apply_edits,
                             apply_whole, parse_edits, parse_whole, tree_diff)

# ── Model client (OpenRouter now, Nebius Token Factory when credits land) ───

TOKEN_FACTORY_URL = os.getenv(
    "NEBIUS_TOKEN_FACTORY_URL",
    "https://api.tokenfactory.nebius.com/v1/chat/completions",
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The key is read from an env *file path* at runtime. It is never copied into
# this repository, never hardcoded, never logged. Point VRD_ENV_FILE elsewhere
# to use a different file; the first candidate that contains the name wins.
ENV_FILE_CANDIDATES = (
    os.getenv("VRD_ENV_FILE", ""),
    str(Path.home() / ".env"),
    str(Path.home() / ".hermes/.env"),
)

PROVIDER_URLS = {"openrouter": OPENROUTER_URL, "nebius": TOKEN_FACTORY_URL}
PROVIDER_KEY_NAMES = {"openrouter": "OPENROUTER_API_KEY", "nebius": "NEBIUS_API_KEY"}

DEFAULT_MODEL = os.getenv("VRD_MODEL", "nvidia/nemotron-3-super-120b-a12b")

# Models differ in what they expose: some accept effort levels, some accept a
# reasoning token budget, some are mandatory-reasoning and reject `none`.
# GET /api/v1/models publishes this per model, so we read it rather than guess.
MODEL_CAPS_CACHE = Path(os.getenv("VRD_CACHE", str(Path.home() / ".cache/vrd"))) / "model_caps.json"
_EFFORTS = ("none", "minimal", "low", "medium", "high", "max")


def fetch_reasoning_caps(model: str) -> dict:
    """Cached per-model reasoning capabilities. Empty dict if unknown."""
    cache: dict = {}
    try:
        if MODEL_CAPS_CACHE.exists():
            cache = json.loads(MODEL_CAPS_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    if model in cache:
        return cache[model]
    key, _src = read_env_key("OPENROUTER_API_KEY")
    if not key:
        return {}
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read()).get("data", [])
        for entry in data:
            cache[entry["id"]] = entry.get("reasoning") or {}
        MODEL_CAPS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MODEL_CAPS_CACHE.write_text(json.dumps(cache, indent=1))
    except Exception:
        return {}
    return cache.get(model, {})


def read_env_key(name: str, candidates: tuple[str, ...] = ENV_FILE_CANDIDATES) -> tuple[str, str]:
    """Return (key, source_path) without exposing the value anywhere else."""
    if os.getenv(name):
        return os.environ[name], "environment"
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = re.match(rf"\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.+)", line)
                    if m:
                        return m.group(1).strip().strip('"').strip("'"), path
        except OSError:
            continue
    return "", ""


@dataclass
class Generation:
    text: str
    model: str
    arm: str
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    visible_output_tokens: int = 0
    finish_reason: str = ""
    content_chars: int = 0
    reasoning_chars: int = 0
    error: str = ""


class ChatClient:
    """Minimal OpenAI-compatible client. Deliberately dependency-free.

    provider=openrouter (default) reads OPENROUTER_API_KEY from an env file by
    path; provider=nebius reads NEBIUS_API_KEY. VRD_DRY_RUN=1 forces dry mode,
    which scores the harness end to end without spending tokens: empty diffs are
    counted as failures, so the numbers are honest about what was measured.
    """

    def __init__(self, api_key: str | None = None, url: str | None = None,
                 provider: str | None = None, env_file: str | None = None):
        self.provider = (provider or os.getenv("VRD_PROVIDER") or "openrouter").lower()
        candidates = ((env_file,) if env_file else ()) + ENV_FILE_CANDIDATES
        key_name = PROVIDER_KEY_NAMES.get(self.provider, "OPENROUTER_API_KEY")
        self.api_key, self.key_source = (api_key, "argument") if api_key else read_env_key(key_name, candidates)
        self.url = url or PROVIDER_URLS.get(self.provider, OPENROUTER_URL)
        self.dry_run = os.getenv("VRD_DRY_RUN") == "1" or not self.api_key

    def complete(self, prompt: str, model: str = DEFAULT_MODEL, arm: str = "A",
                 max_tokens: int | None = None, temperature: float | None = None,
                 top_p: float | None = None) -> Generation:
        # Sampling follows NVIDIA's own cookbook for Nemotron 3 Super agentic
        # coding: temperature 1.0, top_p 0.95, max output 32000. Running this
        # model at temperature 0 (which is what this harness did first) is off
        # its documented operating point.
        temperature = float(os.getenv("VRD_TEMPERATURE", "1.0")) if temperature is None else temperature
        top_p = float(os.getenv("VRD_TOP_P", "0.95")) if top_p is None else top_p
        max_tokens = max_tokens or int(os.getenv("VRD_MAX_TOKENS", "32000"))
        if self.dry_run:
            return Generation(text="", model="dry-run", arm=arm,
                              error=f"no key for provider {self.provider} (dry run)")
        if getattr(self, "_caps_model", None) != model:
            self.caps = fetch_reasoning_caps(model) if self.provider == "openrouter" else {}
            self._caps_model = model
        body_obj = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        # Measured on a real 25KB prompt, 8192 output budget, DeepInfra:
        #   default (reasoning on) -> finish=length, 643 chars content, 8482 reasoning tokens
        #   effort=low             -> finish=length, 8192 tokens, no usable output
        #   effort=none            -> finish=stop,  1049 chars content, 320 tokens
        # This model reasons until the budget is gone for structured output, so the
        # default is off and the switch stays available for a larger budget.
        self.reasoning_mode = os.getenv("VRD_REASONING", "none").lower()
        caps = self.caps or {}
        mandatory = bool(caps.get("mandatory"))
        supported = caps.get("supported_efforts") or []

        if self.reasoning_mode == "budget" and caps.get("supports_max_tokens"):
            body_obj["reasoning"] = {"max_tokens": int(os.getenv("VRD_REASONING_BUDGET", "2048"))}
        elif self.reasoning_mode in ("none", "off") and not mandatory:
            body_obj["reasoning"] = {"enabled": False}
        elif self.reasoning_mode in _EFFORTS:
            effort = self.reasoning_mode
            if supported and effort not in supported:
                effort = "medium" if "medium" in supported else supported[-1]
            body_obj["reasoning"] = {"effort": effort}
        # "default" omits the field entirely
        body = json.dumps(body_obj).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/verified-repo-distillation"
            headers["X-Title"] = "Verified Repo Distillation"
        req = urllib.request.Request(self.url, data=body, headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            return Generation(text="", model=model, arm=arm,
                              latency_s=time.time() - t0, error=str(exc)[:300])
        if "error" in payload and not payload.get("choices"):
            return Generation(text="", model=model, arm=arm, latency_s=time.time() - t0,
                              error=str(payload["error"])[:300])
        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        usage = payload.get("usage", {})
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        visible_tokens = max(0, completion_tokens - reasoning_tokens)
        # The starvation signal, per the docs: output budget consumed by thinking.
        error = ""
        if not content and visible_tokens <= 1:
            upstream = payload.get("error")
            detail = ""
            if upstream:
                detail = str(upstream.get("message") if isinstance(upstream, dict) else upstream)[:200]
            error = (f"no visible output: finish={choice.get('finish_reason')} "
                     f"reasoning_tokens={reasoning_tokens} visible_tokens={visible_tokens}"
                     + (f" | provider_error={detail}" if detail else ""))
        return Generation(
            text=content,
            model=payload.get("model", model), arm=arm, latency_s=time.time() - t0,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            visible_output_tokens=visible_tokens,
            finish_reason=choice.get("finish_reason") or "",
            content_chars=len(content),
            reasoning_chars=len(msg.get("reasoning") or ""),
            error=error,
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


def score_patch(repo: GitRepo, pair: Pair, model_output: str, cache_root: Path,
                scratch_root: Path, preset: str | None = "hermes-webui") -> Score:
    """Apply the model's edit blocks to the parent tree and run the commit's test.

    Two accepted shapes: search/replace blocks (primary contract) and a unified
    diff (fallback for models that do produce valid ones). A search/replace block
    that matches nothing, or matches twice, fails loudly — it is never guessed
    into place.
    """
    if not model_output.strip():
        return Score(sha=pair.sha, arm="", generated_bytes=0, applies=False,
                     tests_pass=False, note="empty output")

    # Hard cap, not a request. A 66KB answer to a 2KB task is truncation waiting
    # to happen, and a polite instruction in the prompt does not hold.
    char_cap = int(os.getenv("VRD_MAX_OUTPUT_CHARS", "40000"))
    if len(model_output) > char_cap:
        return Score(sha=pair.sha, arm="", generated_bytes=len(model_output),
                     applies=False, tests_pass=False,
                     note=f"output too large ({len(model_output)} > {char_cap} chars)")

    fmt = os.getenv("VRD_EDIT_FORMAT", "diff").lower()
    edits: list = []
    whole_files: dict[str, str] = {}
    parse_error = ""
    if fmt == "whole":
        whole_files, parse_error = parse_whole(model_output)
        for path in whole_files:
            if is_test_path(path):
                return Score(sha=pair.sha, arm="", generated_bytes=len(model_output),
                             applies=False, tests_pass=False,
                             note=f"model tried to edit a test file: {path}")
    else:
        edits, parse_error = parse_edits(model_output)
        if edits:
            for e in edits:
                if is_test_path(e.path):
                    return Score(sha=pair.sha, arm="", generated_bytes=len(model_output),
                                 applies=False, tests_pass=False,
                                 note=f"model tried to edit a test file: {e.path}")

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

            if fmt == "whole":
                if not whole_files:
                    return Score(sha=pair.sha, arm="", generated_bytes=len(model_output),
                                 applies=False, tests_pass=False,
                                 note=f"unparseable output ({parse_error})")
                ok, note = apply_whole(tree, whole_files)
                if not ok:
                    return Score(sha=pair.sha, arm="", generated_bytes=len(model_output),
                                 applies=False, tests_pass=False, note=note[:220])
                diff = tree_diff(tree)
            elif edits:
                ok, note = apply_edits(tree, edits)
                if not ok:
                    return Score(sha=pair.sha, arm="", generated_bytes=len(model_output),
                                 applies=False, tests_pass=False, note=note[:220])
                diff = tree_diff(tree)
                if not diff.strip():
                    return Score(sha=pair.sha, arm="", generated_bytes=len(model_output),
                                 applies=False, tests_pass=False,
                                 note="edits applied but produced no diff")
            else:
                diff = extract_diff(model_output)
                if not diff.strip():
                    return Score(sha=pair.sha, arm="", generated_bytes=0, applies=False,
                                 tests_pass=False, note=f"unparseable output ({parse_error})")
                proc = subprocess.run(
                    ["git", "-C", str(tree), "apply", "--whitespace=nowarn", "-"],
                    input=diff, text=True, capture_output=True,
                )
                if proc.returncode != 0:
                    return Score(sha=pair.sha, arm="", generated_bytes=len(diff),
                                 applies=False, tests_pass=False,
                                 note=proc.stderr.strip()[:200])

            # Same dispatcher the gate uses: Python via uv, Go via module cache.
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
            preset: str | None = "hermes-webui",
            progress=None) -> list[Score]:
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
                + EDIT_INSTRUCTIONS
            )
        else:
            prompt = pair.to_prompt()

        gen = client.complete(prompt, model=model, arm=arm)
        score = score_patch(repo, pair, gen.text, cache_root, scratch_root, preset)
        score.arm = arm
        score.test_result.setdefault("generation", {})
        score.test_result["generation"] = {
            "latency_s": round(gen.latency_s, 2),
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "finish_reason": gen.finish_reason,
            "content_chars": gen.content_chars,
            "reasoning_chars": gen.reasoning_chars,
            "reasoning_tokens": gen.reasoning_tokens,
            "visible_output_tokens": gen.visible_output_tokens,
            "reasoning_mode": getattr(client, "reasoning_mode", "default"),
            "error": gen.error,
            "adapter": adapter,
            "prompt_sha8": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:8],
        }
        scores.append(score)
        if progress is not None:
            # Emit per pair: a silent 40-minute run is indistinguishable from a
            # hung one, and an aggregate written only at the end is lost if the
            # run dies.
            progress(arm, pair, score, gen)
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
