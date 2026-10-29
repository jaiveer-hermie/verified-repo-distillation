#!/bin/bash
# Batch 2: the head-to-head on the code-only, oracle-verified held-out set.
# One attempt per pair to keep spend bounded; retries are a separate question.
set -u
cd /home/jaiveer2507/projects/verified-repo-distillation || exit 1

export VRD_PROVIDER=nebius
export VRD_MAX_TOKENS=16384
D=data/hermes-webui-all/dataset
L=data/hermes-webui-all
SPLIT=2026-05-19T01:38:13Z
REPO="$HOME/projects/hermes-webui"

run() {  # $1=label  $2=model
  echo "=== $1 ($2) ==="
  .venv/bin/python run.py eval --repo "$REPO" --data "$D" \
    --model "$2" --arms A --attempts 1 --time-split "$SPLIT" \
    > "$L/eval_${1}.log" 2>&1
  cp "$D/eval_scores.jsonl" "$D/eval_scores_${1}.jsonl" 2>/dev/null
  echo "$1: $(grep -c PASS "$L/eval_${1}.log") passes / $(grep -cE '^  A ' "$L/eval_${1}.log") pairs"
}

run super "nvidia/nemotron-3-super-120b-a12b"
run ultra "nvidia/Nemotron-3-Ultra-550b-a55b"
run qwen_base "Qwen/Qwen3-30B-A3B-Instruct-2507"

echo BATCH2_DONE
