#!/bin/bash
# Rebuild the code-only corpus, re-export the pre-split training set, then submit
# the arm-C fine-tune. Sequenced because the upload must use the rebuilt data.
set -u
cd /home/jaiveer2507/projects/verified-repo-distillation || exit 1

bash scripts/kill_stale.sh
SPLIT=2026-05-19T01:38:13Z

echo "=== rebuild pairs (code-only) ==="
.venv/bin/python run.py pairs --repo "$HOME/projects/hermes-webui" \
  --data data/hermes-webui-all --out data/hermes-webui-all/dataset

echo "=== re-export train ==="
.venv/bin/python run.py train --data data/hermes-webui-all/dataset \
  --out data/hermes-webui-all/export --time-split "$SPLIT"

echo "=== submit fine-tune ==="
.venv/bin/python scripts/ft_submit.py

echo REBUILD_SUBMIT_DONE
