#!/bin/bash
# Batch 1 on Nebius Token Factory: Super-120B vs Ultra-550B, held-out pairs only.
# 16384 output budget: on Nebius these models spend the first few thousand tokens
# on reasoning, and 4096 produced empty content with finish_reason=length.
set -u
cd /home/jaiveer2507/projects/verified-repo-distillation || exit 1

export VRD_PROVIDER=nebius
export VRD_MAX_TOKENS=16384
D=data/hermes-webui-all/dataset
L=data/hermes-webui-all
SPLIT=2026-05-19T01:38:13Z
REPO="$HOME/projects/hermes-webui"

echo "=== SUPER 120B ==="
.venv/bin/python run.py eval --repo "$REPO" --data "$D" \
  --model nvidia/nemotron-3-super-120b-a12b --arms A --attempts 2 \
  --time-split "$SPLIT" > "$L/eval_nebius_super.log" 2>&1
cp "$D/eval_scores.jsonl" "$D/eval_scores_super.jsonl"
echo "super done: $(grep -c PASS "$L/eval_nebius_super.log")"

echo "=== ULTRA 550B ==="
.venv/bin/python run.py eval --repo "$REPO" --data "$D" \
  --model nvidia/Nemotron-3-Ultra-550b-a55b --arms A --attempts 2 \
  --time-split "$SPLIT" > "$L/eval_nebius_ultra.log" 2>&1
cp "$D/eval_scores.jsonl" "$D/eval_scores_ultra.jsonl"
echo "ultra done: $(grep -c PASS "$L/eval_nebius_ultra.log")"

echo BATCH1_DONE
