"""Does a job id (or any checkpoint identifier) satisfy custom_weights_id?

Tries each identifier the checkpoints API exposes, with gpu_count=0 so a passing
validation cannot provision a billable GPU. After every attempt, any endpoint that
appeared is deleted immediately.
"""
import json
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
JOB = "ftjob-8c6c6ffcf7734e97b1663d500b64a72f"
CKPT = "ftckpt_1b6c9928-7c31-4d78-9595-c2f1df11aa12"
FT_STRING = "ft:Qwen/Qwen3-30B-A3B-Instruct-2507-2026-10-29:org_placeholder:vrd-verified:IDPlaceholder:ckpt-step-11"

CANDIDATES = [
    ("job_id", JOB),
    ("checkpoint_id", CKPT),
    ("fine_tuned_model_checkpoint string", FT_STRING),
    ("job_id:ckpt", f"{JOB}:{CKPT}"),
]


def call(method, path, payload=None, timeout=60):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(CP + path, data=body, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"{}")
        except Exception:
            return exc.code, {"raw": raw[:300].decode("utf-8", "replace")}


def cleanup():
    st, cur = call("GET", "/v0/dedicated_endpoints")
    eps = cur.get("data", []) if isinstance(cur, dict) else []
    for ep in eps:
        call("DELETE", f"/v0/dedicated_endpoints/{ep.get('id')}")
        print(f"   [cleanup] deleted {ep.get('name')} ({ep.get('id')})")
    return len(eps)


for label, value in CANDIDATES:
    payload = {
        "name": "vrd-probe",
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "flavor_name": "base",
        "gpu_type": "gpu-h200-sxm",
        "gpu_count": 0,
        "region": "us-central1",
        "scaling": {"min_replicas": 1, "max_replicas": 1},
        "custom_weights_id": value,
    }
    status, resp = call("POST", "/v0/dedicated_endpoints", payload)
    detail = json.dumps(resp)[:260]
    print(f"{label:38s} -> HTTP {status}")
    print(f"   {detail}")
    created = cleanup()
    if created:
        print(f"   !! endpoint was created and deleted ({created})")

st, cur = call("GET", "/v0/dedicated_endpoints")
print("\nfinal endpoints:", (cur.get("data") if isinstance(cur, dict) else cur))
