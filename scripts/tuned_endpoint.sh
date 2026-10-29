#!/bin/bash
# Full arm-C cycle: deploy the registered checkpoint, score it on the held-out
# pairs, then delete the endpoint immediately so billing stops.
#
#   bash scripts/tuned_endpoint.sh model-artifact_XXXXXXXX
#
# The artifact id comes from the Custom Weights Hub (console, beta/on-request).
# Nothing here creates compute until the deploy step succeeds.
set -u
cd /home/jaiveer2507/projects/verified-repo-distillation || exit 1

ARTIFACT="${1:-}"
if [ -z "$ARTIFACT" ]; then
  echo "usage: bash scripts/tuned_endpoint.sh model-artifact_XXXXXXXX"
  exit 2
fi
case "$ARTIFACT" in
  model-artifact_*) : ;;
  *) echo "refusing: custom_weights_id must start with 'model-artifact_'"; exit 2 ;;
esac

FT=data/hermes-webui-all/ft
echo "$ARTIFACT" > "$FT/artifact_id.txt"
SPLIT=2026-05-19T01:38:13Z
D=data/hermes-webui-all/dataset

echo "=== deploy ==="
VRD_ARTIFACT="$ARTIFACT" .venv/bin/python scripts/deploy_tuned.py || { echo "deploy failed"; exit 1; }

ROUTING=$(.venv/bin/python -c "import json;print(json.load(open('$FT/endpoint.json'))['routing_key'])")
echo "routing_key=$ROUTING"

echo "=== eval tuned (held-out only) ==="
VRD_PROVIDER=nebius \
VRD_BASE_URL="https://api.tokenfactory.us-central1.nebius.com/v1" \
VRD_MAX_TOKENS=4096 \
.venv/bin/python run.py eval --repo "$HOME/projects/hermes-webui" --data "$D" \
  --model "$ROUTING" --arms A --attempts 1 --time-split "$SPLIT" \
  > data/hermes-webui-all/eval_qwen_tuned.log 2>&1
cp "$D/eval_scores.jsonl" "$D/eval_scores_qwen_tuned.jsonl" 2>/dev/null
echo "tuned: $(grep -c PASS data/hermes-webui-all/eval_qwen_tuned.log) passes"

echo "=== delete endpoint (stop billing) ==="
.venv/bin/python scripts/delete_endpoint.py
echo TUNED_CYCLE_DONE
