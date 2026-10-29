"""Contamination probe: have the base models memorised hermes-webui?

Method: take held-out (post-split) commits. From each, take the POST-commit file
content, which the model has never been shown in any prompt. Show the model the
40 lines preceding a random offset and ask it to reproduce the next 10 lines
verbatim. Exact multi-line reproduction of arbitrary private code is close to
impossible by chance; if it happens, the model has seen the repository and every
arm's score is inflated.

Cheap by construction: a handful of short calls.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, "/home/jaiveer2507/projects/verified-repo-distillation")
from vrd.evaluate import ChatClient
from vrd.ingest import GitRepo, read_jsonl
from vrd.pairs import Pair

SPLIT = "2026-05-19T01:38:13Z"
DATA = Path("/home/jaiveer2507/projects/verified-repo-distillation/data/hermes-webui-all/dataset")
REPO = "/home/jaiveer2507/projects/hermes-webui"

rows = read_jsonl(DATA / "pairs.jsonl")
held = [Pair(**r) for r in rows if r["commit_date"] >= SPLIT]
held = [p for p in held if len(p.changed_files) <= 2][:8]

repo = GitRepo(REPO)
client = ChatClient()
print(f"provider={client.provider} model_env={client.dry_run=}")

MODELS = ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/nemotron-3-super-120b-a12b"]

rng = random.Random(7)
cases = []
for p in held:
    for f in p.source_files:
        try:
            post = repo.show_file(p.sha, f)          # post-commit, never shown to any model
        except RuntimeError:
            continue
        lines = post.splitlines()
        if len(lines) < 120:
            continue
        off = rng.randint(60, len(lines) - 60)
        cases.append((p.sha, f, lines, off))
        break
    if len(cases) >= 5:
        break

print(f"cases: {len(cases)}")
for model in MODELS:
    exact_hits = 0
    total_lines = 0
    for sha, f, lines, off in cases:
        context = "\n".join(lines[off - 40:off])
        expected = lines[off:off + 10]
        prompt = (
            "Below are consecutive lines from a source file in a private repository.\n"
            "Reproduce the NEXT 10 lines of that file exactly as they appear, one per line.\n"
            "No commentary, no code fences. If you do not know the file, output exactly: UNKNOWN\n\n"
            f"---\n{context}\n---\n"
        )
        gen = client.complete(prompt, model=model, arm="contam")
        out = [l.rstrip() for l in (gen.text or "").splitlines()]
        if out and out[0].strip() == "UNKNOWN":
            print(f"  {model.split('/')[-1]:28s} {sha[:8]} {f[:28]:28s} -> UNKNOWN")
            continue
        hits = sum(1 for e in expected if e.rstrip() in out and len(e.strip()) > 3)
        exact_hits += hits
        total_lines += sum(1 for e in expected if len(e.strip()) > 3)
        print(f"  {model.split('/')[-1]:28s} {sha[:8]} {f[:28]:28s} "
              f"matched {hits}/10 lines (out {len(out)} lines, {gen.content_chars}c)")
    pct = (exact_hits / total_lines * 100) if total_lines else 0.0
    print(f"== {model}: verbatim line match {exact_hits}/{total_lines} = {pct:.0f}%")
