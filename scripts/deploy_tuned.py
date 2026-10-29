"""Deploy the arm-C checkpoint to a dedicated endpoint.

Cost discipline: one replica, one H200, us-central1 (the only config the template
allows for this model). Nothing is billed until the endpoint reports ready, so a
rejected request is free. The endpoint is deleted immediately after the eval.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

key = ""
for line in (Path.home() / ".hermes" / ".env").read_text().splitlines():
    line = line.strip()
    if line.startswith("NEBIUS_API_KEY") and "=" in line:
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

CP = "https://api.tokenfactory.nebius.com"
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
FT = Path("/home/jaiveer2507/projects/verified-repo-distillation/data/hermes-webui-all/ft")
CKPT = (os.getenv("VRD_ARTIFACT")
        or ((FT / "artifact_id.txt").read_text().strip()
            if (FT / "artifact_id.txt").exists() else "")
        or ((FT / "checkpoint_id.txt").read_text().strip()
            if (FT / "checkpoint_id.txt").exists() else "")
        or "ftckpt_1b6c9928-7c31-4d78-9595-c2f1df11aa12")


def call(method, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(CP + path, data=body, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"{}")
        except Exception:
            return exc.code, {"raw": raw[:400].decode("utf-8", "replace")}


payload = {
    "name": "vrd-qwen30b-tuned",
    "description": "arm C: Qwen3-30B-A3B LoRA on 148 verified hermes-webui pairs",
    "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "flavor_name": "base",
    "gpu_type": "gpu-h200-sxm",
    "gpu_count": 1,
    "region": "us-central1",
    "scaling": {"min_replicas": 1, "max_replicas": 1},
    "custom_weights_id": CKPT,
}

status, resp = call("POST", "/v0/dedicated_endpoints", payload)
print(f"create -> HTTP {status}")
print(json.dumps(resp)[:800])
if status >= 300:
    print("\nDEPLOY_REJECTED — no compute created, nothing billed.")
    sys.exit(1)

ep = resp.get("endpoint", resp)
ep_id = ep.get("id")
routing_key = ep.get("routing_key")
(FT / "endpoint.json").write_text(json.dumps(ep, indent=2))
print(f"endpoint_id={ep_id}\nrouting_key={routing_key}")

for i in range(40):
    time.sleep(15)
    st, cur = call("GET", f"/v0/dedicated_endpoints/{ep_id}")
    dep = (cur.get("endpoint") or cur).get("deployment") or {}
    print(f"  [{i}] status={dep.get('status')} readiness={dep.get('readiness')} "
          f"ready_replicas={dep.get('ready_replicas')}", flush=True)
    if dep.get("readiness") == "ready":
        print("ENDPOINT_READY")
        break
    if dep.get("status") in ("error", "stopped"):
        print("ENDPOINT_FAILED")
        break
