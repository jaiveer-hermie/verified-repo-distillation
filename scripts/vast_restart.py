"""Restart the Vast instance so the newly registered SSH key is injected.

Ports are re-mapped on start, so the SSH endpoint is re-read afterwards.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

key = ""
for line in (Path.home() / ".hermes" / ".env").read_text().splitlines():
    line = line.strip()
    if line.startswith("VAST_API_KEY") and "=" in line:
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

BASE = "https://console.vast.ai"
H = {"Authorization": f"Bearer {key}", "Accept": "application/json",
     "Content-Type": "application/json"}
IID = 58395735


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, headers=H,
                                 data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:300]


def state():
    _, d = call("GET", "/api/v1/instances/")
    inst = (d.get("instances") or d)
    inst = inst[0] if isinstance(inst, list) else inst
    return inst


print("stop ->", call("PUT", f"/api/v0/instances/{IID}/", {"state": "stopped"}))
for _ in range(40):
    time.sleep(6)
    st = state().get("actual_status")
    print("  ", st)
    if st == "stopped":
        break

print("start ->", call("PUT", f"/api/v0/instances/{IID}/", {"state": "running"}))
for _ in range(60):
    time.sleep(6)
    inst = state()
    st = inst.get("actual_status")
    print("  ", st, inst.get("ssh_host"), inst.get("ssh_port"))
    if st == "running":
        out = {"ssh_host": inst.get("ssh_host"), "ssh_port": inst.get("ssh_port"),
               "public_ipaddr": inst.get("public_ipaddr"),
               "ports": inst.get("ports"), "dph_total": inst.get("dph_total")}
        Path("/home/jaiveer2507/projects/verified-repo-distillation/data/hermes-webui-all/vast_instance.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out)[:400])
        break
