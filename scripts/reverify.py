"""Re-verify a corpus and report verdict drift.

A gate that cannot reproduce its own verdicts is not auditable. This replays
commits whose verdict is already recorded, compares the fresh result against the
stored one, and reports every disagreement with the per-test evidence behind it.

Usage:
    python scripts/reverify.py --repo ~/projects/hermes-webui \
        --data data/hermes-webui-all --sample 40 --strict --repeats 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrd.ingest import Commit, FileChange, read_jsonl  # noqa: E402
from vrd.replay import GitRepo, counterfactual_verify  # noqa: E402


def load_commits(path: Path) -> dict[str, Commit]:
    out: dict[str, Commit] = {}
    for row in read_jsonl(path):
        c = Commit(
            sha=row["sha"],
            parents=row.get("parents", []),
            author_date=row.get("author_date", ""),
            subject=row.get("subject", ""),
            files=[FileChange(**f) for f in row.get("files", [])],
        )
        c.candidate = row.get("candidate", False)
        out[row["sha"]] = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--status", default="verified",
                    help="only re-check stored verdicts with this status ('' for all)")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    stored = {r["sha"]: r for r in read_jsonl(data / "verdicts.jsonl")}
    pool = [s for s, r in stored.items()
            if not args.status or r.get("status") == args.status]
    commits = load_commits(data / "candidates.jsonl")
    pool = [s for s in pool if s in commits]
    print(f"stored verdicts: {len(stored)} | pool('{args.status}'): {len(pool)}")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.sample, len(pool)))
    print(f"re-verifying {len(sample)} commits (strict={args.strict}, "
          f"repeats={args.repeats})\n")

    repo = GitRepo(args.repo)
    cache = Path.home() / ".cache" / "vrd"
    scratch = data / "reverify_scratch"
    rows = []
    t0 = time.time()
    for i, sha in enumerate(sample, 1):
        v = counterfactual_verify(repo, commits[sha], cache, scratch,
                                  preset="hermes-webui", strict=args.strict,
                                  repeats=args.repeats)
        was = stored[sha].get("status")
        drift = "OK" if was == v.status else f"DRIFT {was} -> {v.status}"
        rows.append({
            "sha": sha, "stored": was, "fresh": v.status,
            "agree": was == v.status,
            "fail_to_pass": v.fail_to_pass[:8],
            "regressions": v.regressions[:8],
            "flaky": v.flaky[:8],
            "wall_s": round(v.wall_s, 1),
        })
        print(f"[{i:2d}/{len(sample)}] {sha[:8]} {drift:44s} "
              f"f2p={len(v.fail_to_pass):2d} flaky={len(v.flaky)} {v.wall_s:5.1f}s")

    agree = sum(1 for r in rows if r["agree"])
    n = len(rows) or 1
    report = {
        "sampled": len(rows),
        "agree": agree,
        "agreement_rate": round(agree / n, 4),
        "strict": args.strict,
        "repeats": args.repeats,
        "seed": args.seed,
        "total_wall_s": round(time.time() - t0, 1),
        "transitions": dict(Counter(f"{r['stored']}->{r['fresh']}" for r in rows
                                    if not r["agree"])),
        "rows": rows,
    }
    out = data / "reverify.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*62}")
    print(f"agreement: {agree}/{report['sampled']} = {report['agreement_rate']*100:.1f}%")
    for pair, c in report["transitions"].items():
        print(f"  {pair}: {c}")
    print(f"report -> {out}")
    return 0 if agree == report["sampled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
