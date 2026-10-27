"""Verified Repo Distillation — build execution-verified training corpora from git history.

Stages:
    1. ingest   — walk git history, classify commits, drop noise
    2. replay   — reconstruct the environment at a commit's parent, run its tests
    3. gate     — counterfactual test injection: does the patch actually fix behaviour?
    4. pairs    — build (context, patch) training pairs from verified commits only
    5. train    — QLoRA on Nemotron via Nebius AI Cloud
    6. eval     — three-arm comparison: base / base+retrieval / tuned
"""

__version__ = "0.1.0"
