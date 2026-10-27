#!/usr/bin/env python3
"""VRD command line.

    python run.py ingest --repo <path>
    python run.py gate   --repo <path>      (stage 2+3, see vrd/replay.py)

Stage 1 only needs the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vrd.ingest import GitRepo, funnel, ingest, write_jsonl


def cmd_ingest(args: argparse.Namespace) -> int:
    repo = GitRepo(args.repo)
    out_dir = Path(args.out)
    print(f"repo      : {repo.path}")
    print(f"head      : {repo.head_sha()}")
    print(f"commits   : {repo.commit_count()}")
    print("walking history...")

    commits = ingest(repo.path, first_parent=not args.all_parents, limit=args.limit)
    report = funnel(commits)

    commits_path = write_jsonl(commits, out_dir / "commits.jsonl")
    cands = [c for c in commits if c.candidate]
    cand_path = write_jsonl(cands, out_dir / "candidates.jsonl")
    (out_dir / "funnel.json").write_text(json.dumps(report, indent=2) + "\n")

    print()
    print(f"commits walked      : {report['commits_total']}")
    print(f"kept                : {report['kept']}")
    print(f"dropped             : {report['dropped']}")
    print(f"gate candidates     : {report['gate_candidates']}")
    print()
    print("drop reasons:")
    for reason, count in report["drop_reasons"].items():
        print(f"  {reason:<24}{count:>6}")
    print()
    print(f"wrote {commits_path}")
    print(f"wrote {cand_path}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Stage 2+3: replay each candidate at its parent and apply the counterfactual gate."""
    import time as _time

    from vrd.ingest import Commit as _Commit, FileChange
    from vrd.replay import verify_many, write_verdicts

    cand_path = Path(args.data) / "candidates.jsonl"
    if not cand_path.exists():
        print(f"missing {cand_path} — run `ingest` first")
        return 2
    commits = []
    for line in cand_path.read_text().splitlines():
        row = json.loads(line)
        if args.retry:
            # never re-litigate a verification decision; only retry harness failures
            keep = {r["sha"] for r in (
                json.loads(l) for l in Path(args.retry).read_text().splitlines() if l.strip()
            ) if r["status"] == "reject_replay"}
            if row["sha"] not in keep:
                continue
        files = [FileChange(path=f["path"], added=f["added"],
                            deleted=f["deleted"], status=f["status"])
                 for f in row.get("files", [])]
        commits.append(_Commit(
            sha=row["sha"], parents=row["parents"], author_date=row["author_date"],
            subject=row["subject"], files=files, verdict=row["verdict"],
            reason=row["reason"], candidate=row["candidate"],
        ))

    out_path = Path(args.data) / "verdicts.jsonl"
    t0 = _time.time()

    def progress(i: int, total: int, v) -> None:
        elapsed = _time.time() - t0
        eta = elapsed / i * (total - i)
        print(f"[{i}/{total}] {v.status:<18} {v.sha[:10]} {v.wall_s:6.1f}s "
              f"eta {eta/60:5.1f}m  {v.subject[:58]}")

    verdicts = verify_many(
        args.repo, commits, cache_root=args.cache, scratch_root=args.scratch,
        preset=args.preset, limit=args.limit, progress=progress,
    )
    if args.retry and out_path.exists():
        # merge: replace only the retried shas, keep every prior decision intact
        existing = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        new_by_sha = {v.sha: v for v in verdicts}
        merged = [new_by_sha.get(r["sha"], r) for r in existing] + [
            v for v in verdicts if v.sha not in {r["sha"] for r in existing}
        ]
        from dataclasses import asdict as _asdict
        with out_path.open("w", encoding="utf-8") as fh:
            for row in merged:
                fh.write(json.dumps(row if isinstance(row, dict) else _asdict(row)) + "\n")
    else:
        write_verdicts(verdicts, out_path)

    from collections import Counter
    counts = Counter(v.status for v in verdicts)
    print()
    print(f"gate ran       : {len(verdicts)}")
    for status, n in counts.most_common():
        print(f"  {status:<20}{n:>5}")
    if verdicts:
        mean = sum(v.wall_s for v in verdicts) / len(verdicts)
        print(f"mean wall/commit: {mean:.1f}s")
    print(f"wrote {out_path}")
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    """Stage 4: verified commits -> (context, patch) pairs with a dataset card."""
    from vrd.pairs import build_pairs, dataset_card

    verdicts = Path(args.data) / "verdicts.jsonl"
    if not verdicts.exists():
        print(f"missing {verdicts} — run `gate` first")
        return 2
    out_dir = Path(args.out)
    pairs = build_pairs(args.repo, verdicts, out_dir / "pairs.jsonl")
    card = dataset_card(pairs)
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"pairs built      : {len(pairs)}")
    print(f"median context   : {card['median_context_bytes']} bytes")
    print(f"median patch     : {card['median_patch_bytes']} bytes")
    print(f"files/patch hist : {card['files_touched_histogram']}")
    print(f"wrote {out_dir/'pairs.jsonl'}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Stage 5: export chat JSONL, emit the Nebius QLoRA job spec and script."""
    from vrd.pairs import Pair
    from vrd.train import export_dataset, qlora_job_spec, write_train_script

    pairs_path = Path(args.data) / "pairs.jsonl"
    if not pairs_path.exists():
        print(f"missing {pairs_path} — run `pairs` first")
        return 2
    pairs = [Pair(**json.loads(line)) for line in pairs_path.read_text().splitlines() if line.strip()]
    out_dir = Path(args.out)
    card = export_dataset(pairs, out_dir, replay_dir=args.replay, time_split=args.time_split)
    spec = qlora_job_spec(str(out_dir / "train.jsonl"), str(out_dir / "adapter"))
    (out_dir / "job_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    write_train_script(out_dir)
    print(f"train examples   : {card['train_examples']}")
    print(f"replay examples  : {card['replay_examples']} (fraction {card['replay_fraction_actual']})")
    print(f"held-out (time)  : {card['eval_examples']} (split {card['time_split']})")
    print(f"wrote {out_dir/'train.jsonl'}, job_spec.json, train_qlora.py")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Stage 6: run the arms and print the table. Execution scoring, never a judge."""
    from vrd.evaluate import (ChatClient, markdown_table, run_arm, summarise,
                              write_report)
    from vrd.pairs import Pair

    pairs_path = Path(args.data) / "pairs.jsonl"
    if not pairs_path.exists():
        print(f"missing {pairs_path} — run `pairs` first")
        return 2
    pairs = [Pair(**json.loads(line)) for line in pairs_path.read_text().splitlines() if line.strip()]
    if args.time_split:
        pairs = [p for p in pairs if p.commit_date >= args.time_split]

    # Scope filter. A 4-file, 20KB commit is not a fair task for a 30-120B model,
    # and scoring it produces a number about model size rather than about tuning.
    # Restrict to small, few-file commits and say so in the report.
    before = len(pairs)
    pairs = [p for p in pairs
             if len(p.changed_files) <= args.max_files
             and len(p.patch) <= args.max_patch_bytes]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"pairs: {before} total -> {len(pairs)} within scope "
          f"(<= {args.max_files} files, <= {args.max_patch_bytes} patch bytes)")

    client = ChatClient()
    if client.dry_run:
        print("NOTE: no NEBIUS_API_KEY — dry run. Empty diffs are scored as failures;")
        print("      this validates the harness, it does not measure the model.\n")

    cache_root = Path(args.cache)
    scratch_root = Path(args.scratch)
    results = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"arm {arm}: {len(pairs)} commits")
        results[arm] = run_arm(args.repo, pairs, arm, client, cache_root, scratch_root,
                               model=args.model, preset=args.preset)
    summary = summarise(results)
    out = Path(args.data) / "eval_report.json"
    write_report(summary, out, extra={"model": args.model, "n_pairs": len(pairs)})

    # Per-pair records: an aggregate can hide that every failure was the same
    # harness artefact (diff did not apply) rather than a model failure.
    scores_path = Path(args.data) / "eval_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as fh:
        for arm, arm_scores in results.items():
            for s in arm_scores:
                row = {
                    "arm": arm, "sha": s.sha, "applies": s.applies,
                    "tests_pass": s.tests_pass, "success": s.success,
                    "generated_bytes": s.generated_bytes, "note": s.note,
                    "generation": (s.test_result or {}).get("generation", {}),
                    "test_counts": {k: (s.test_result or {}).get(k) for k in
                                    ("passed", "failed", "errors", "skipped")},
                }
                fh.write(json.dumps(row) + "\n")

    print()
    print(markdown_table(summary))
    print()
    notes = [s.note for arm_scores in results.values() for s in arm_scores if s.note]
    if notes:
        print("first failure notes:")
        for n in notes[:3]:
            print(f"  - {n[:160]}")
    print(f"wrote {out}")
    print(f"wrote {scores_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Aggregate the funnel and gate verdicts into one JSON the demo page reads."""
    from collections import Counter

    data = Path(args.data)
    verdicts_path = data / "verdicts.jsonl"
    funnel_path = data / "funnel.json"
    commits_path = data / "commits.jsonl"

    verdicts = []
    if verdicts_path.exists():
        verdicts = [json.loads(l) for l in verdicts_path.read_text().splitlines() if l.strip()]

    status = Counter(v["status"] for v in verdicts)
    pre_fail = [v for v in verdicts if v["status"] == "verified"]
    showcase = [v for v in verdicts if v["status"] == "reject_pre_passed"]

    def counts(v, arm):
        r = v.get(arm) or {}
        return {"passed": r.get("passed", 0), "failed": r.get("failed", 0),
                "errors": r.get("errors", 0), "skipped": r.get("skipped", 0),
                "seconds": round(r.get("duration_s", 0), 2)}

    report = {
        "funnel": json.loads(funnel_path.read_text()) if funnel_path.exists() else {},
        "gate": {
            "ran": len(verdicts),
            "status": dict(status),
            "verified": status.get("verified", 0),
            "mean_wall_s": round(sum(v.get("wall_s", 0) for v in verdicts) / len(verdicts), 1) if verdicts else 0,
        },
        "showcase_rejections": [
            {
                "sha": v["sha"], "subject": v["subject"],
                "pre": counts(v, "pre"), "post": counts(v, "post"),
                "test_files": v.get("test_files", []),
            }
            for v in showcase[:25]
        ],
        "verified_commits": [
            {
                "sha": v["sha"], "subject": v["subject"], "wall_s": round(v.get("wall_s", 0), 1),
                "pre": counts(v, "pre"), "post": counts(v, "post"),
                "test_files": v.get("test_files", []),
            }
            for v in pre_fail
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"gate ran        : {report['gate']['ran']}")
    for k, n in status.most_common():
        print(f"  {k:<20}{n:>5}")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vrd", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="stage 1: walk and classify git history")
    p_ingest.add_argument("--repo", required=True)
    p_ingest.add_argument("--out", default="data")
    p_ingest.add_argument("--limit", type=int, default=None)
    p_ingest.add_argument("--all-parents", action="store_true",
                          help="walk all commits instead of --first-parent")
    p_ingest.set_defaults(func=cmd_ingest)

    p_gate = sub.add_parser("gate", help="stage 2+3: counterfactual verification")
    p_gate.add_argument("--repo", required=True)
    p_gate.add_argument("--data", required=True, help="dir holding candidates.jsonl")
    p_gate.add_argument("--limit", type=int, default=None)
    p_gate.add_argument("--preset", default="hermes-webui",
                        help="env preset from vrd.replay.ENV_PRESETS, or 'none'")
    p_gate.add_argument("--cache", default=str(Path.home() / ".cache/vrd"))
    p_gate.add_argument("--scratch", default=str(Path.home() / ".cache/vrd/work"))
    p_gate.add_argument("--retry", default=None,
                        help="path to an existing verdicts.jsonl; only retry reject_replay entries")
    p_gate.set_defaults(func=cmd_gate)

    p_pairs = sub.add_parser("pairs", help="stage 4: build training pairs")
    p_pairs.add_argument("--repo", required=True)
    p_pairs.add_argument("--data", required=True)
    p_pairs.add_argument("--out", default="data/pairs")
    p_pairs.set_defaults(func=cmd_pairs)

    p_train = sub.add_parser("train", help="stage 5: export dataset + Nebius job spec")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--out", default="data/run")
    p_train.add_argument("--replay", default=None, help="dir with replay.jsonl")
    p_train.add_argument("--time-split", default=None, help="ISO date; later commits are held out")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="stage 6: three-arm execution-scored comparison")
    p_eval.add_argument("--repo", required=True)
    p_eval.add_argument("--data", required=True)
    p_eval.add_argument("--arms", default="A,B,C")
    p_eval.add_argument("--model", default=None)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--max-files", type=int, default=2,
                        help="keep pairs touching at most this many source files")
    p_eval.add_argument("--max-patch-bytes", type=int, default=6000,
                        help="keep pairs whose gold patch is at most this large")
    p_eval.add_argument("--time-split", default=None)
    p_eval.add_argument("--preset", default="hermes-webui")
    p_eval.add_argument("--cache", default=str(Path.home() / ".cache/vrd"))
    p_eval.add_argument("--scratch", default=str(Path.home() / ".cache/vrd/work"))
    p_eval.set_defaults(func=cmd_eval)

    p_report = sub.add_parser("report", help="aggregate funnel + verdicts for the demo page")
    p_report.add_argument("--data", required=True)
    p_report.add_argument("--out", default="dashboard/report.json")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    if getattr(args, "preset", None) == "none":
        args.preset = None
    if getattr(args, "model", None) is None:
        from vrd.evaluate import DEFAULT_MODEL
        args.model = DEFAULT_MODEL
    return args.func(args)



if __name__ == "__main__":
    sys.exit(main())
