"""Audit the gate's own verdicts under stricter rules.

The published gate accepted a commit when the suite was red before the change
and green after it. That rule cannot distinguish a real fix from:

  * a flaky test that happened to fail during the pre-arm run,
  * a suite that went red for a reason unrelated to the commit's target,
  * a fix that also breaks a test which used to pass.

This script replays a sample of commits that the gate already marked
``verified`` under the stricter rule (named fail->pass transition, no
regressions, stable across repeated runs) and reports how many survive.

Usage:
    python scripts/audit_gate.py --repo ~/projects/hermes-webui \
        --data data/hermes-webui-all --sample 40 --repeats 2
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


def load_commits(cand_path: Path) -> dict[str, Commit]:
    out: dict[str, Commit] = {}
    for row in read_jsonl(cand_path):
        files = [FileChange(**f) for f in row.get("files", [])]
        commit = Commit(
            sha=row["sha"],
            parents=row.get("parents", []),
            author_date=row.get("author_date", ""),
            subject=row.get("subject", ""),
            files=files,
        )
        commit.candidate = row.get("candidate", False)
        commit.mainline = row.get("mainline", False)
        # added_tests / source_paths are read-only properties computed from
        # `files`, so they need no assignment here.
        out[row["sha"]] = commit
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Absolute paths matter: `git worktree add` resolves a relative target
    # against the *source* repo, so a relative scratch dir silently creates
    # worktrees inside the repo being audited and every toolchain probe then
    # fails with "unsupported toolchain: unknown".
    data = Path(args.data).resolve()
    verdicts = list(read_jsonl(data / "verdicts.jsonl"))
    verified = [v["sha"] for v in verdicts if v.get("status") == "verified"]
    print(f"gate verdicts: {len(verdicts)} | verified: {len(verified)}")

    commits = load_commits(data / "candidates.jsonl")
    pool = [s for s in verified if s in commits]
    print(f"verified with commit metadata: {len(pool)}")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.sample, len(pool)))
    print(f"auditing {len(sample)} sampled commits, repeats={args.repeats}\n")

    repo = GitRepo(args.repo)
    cache = Path.home() / ".cache" / "vrd"
    scratch = (data / "audit_scratch").resolve()
    results = []
    t0 = time.time()
    for i, sha in enumerate(sample, 1):
        v = counterfactual_verify(
            repo, commits[sha], cache, scratch,
            preset="hermes-webui", strict=True, repeats=args.repeats,
        )
        results.append(v)
        f2p = len(v.fail_to_pass)
        print(f"[{i:2d}/{len(sample)}] {sha[:8]} {v.status:22s} "
              f"f2p={f2p:2d} p2p={v.pass_to_pass:3d} reg={len(v.regressions)} "
              f"flaky={len(v.flaky)} {v.wall_s:5.1f}s")

    counts = Counter(v.status for v in results)
    survived = counts.get("verified", 0)
    n = len(results) or 1
    report = {
        "sampled": len(results),
        "survived_strict": survived,
        "survival_rate": round(survived / n, 4),
        "status_counts": dict(counts),
        "mean_wall_s": round(sum(v.wall_s for v in results) / n, 1),
        "repeats": args.repeats,
        "seed": args.seed,
        "failures": [
            {
                "sha": v.sha,
                "status": v.status,
                "fail_to_pass": v.fail_to_pass[:5],
                "regressions": v.regressions[:5],
                "flaky": v.flaky[:5],
            }
            for v in results if v.status != "verified"
        ],
    }
    out = Path(args.out) if args.out else data / "audit_strict.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*62}")
    print(f"sampled              : {report['sampled']}")
    print(f"survived strict gate : {survived} ({report['survival_rate']*100:.1f}%)")
    for status, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:22s} {c}")
    print(f"mean wall per commit : {report['mean_wall_s']}s")
    print(f"total audit time     : {time.time()-t0:.0f}s")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
