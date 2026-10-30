"""Create a Vast instance that is born with our SSH key, then retire the old one.

The existing instance was created before any key existed on the account and Vast
does not retro-inject keys, so SSH could never work there.
"""
import json
import time
import urllib.error
import urllib.parse
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
OLD = 58395735
KEY_ID = 1731152


def call(method, path, body=None, timeout=90):
    req = urllib.request.Request(BASE + path, method=method, headers=H,
                                 data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:300]


query = {
    "gpu_name": {"eq": "RTX 6000Ada"},
    "num_gpus": {"eq": 1},
    "rentable": {"eq": True},
    "disk_space": {"gte": 120},
    "order": [["dph_total", "asc"]],
    "type": "on-demand",
}
status, offers = call("GET", "/api/v0/bundles/?" + urllib.parse.urlencode({"q": json.dumps(query)}))
print("offers ->", status)
rows = offers.get("offers", []) if isinstance(offers, dict) else []
print(f"{len(rows)} offers")
if not rows:
    raise SystemExit("no offers found")

pick = rows[0]
for o in rows[:6]:
    print(f"  ask={o['id']} machine={o.get('machine_id')} dph=${o.get('dph_total'):.3f} "
          f"disk={o.get('disk_space'):.0f}GB inet_up={o.get('inet_up')} verified={o.get('verified')}")
    print(f"    {o.get('gpu_name')} x{o.get('num_gpus')} ram={o.get('gpu_ram')} "
          f"cpu={o.get('cpu_cores_effective')} dl={o.get('inet_down'):.0f}Mbps")

create = {
    "client_id": "me",
    "image": "vastai/vllm:v0.32.0-cuda-13.0",
    "disk": 176.0,
    "ssh_key_ids": [KEY_ID],
    "runtype": "ssh_direct",
    "label": "vrd-eval",
    "onstart": "",
}
status, resp = call("PUT", f"/api/v0/asks/{pick['id']}/", create)
print("create ->", status, resp)
if status >= 300:
    raise SystemExit("create failed")

new_id = resp.get("new_contract")
print("new instance id:", new_id)
for _ in range(70):
    time.sleep(8)
    _, d = call("GET", "/api/v1/instances/")
    insts = d.get("instances") or d
    inst = next((i for i in insts if i.get("id") == new_id), None)
    if not inst:
        print("  waiting for record…")
        continue
    st = inst.get("actual_status")
    print(f"  {st} ssh={inst.get('ssh_host')}:{inst.get('ssh_port')} "
          f"pub={inst.get('public_ipaddr')} ports22={(inst.get('ports') or {}).get('22/tcp')}")
    if st == "running":
        out = {
            "id": new_id,
            "ssh_host": inst.get("ssh_host"),
            "ssh_port": inst.get("ssh_port"),
            "public_ipaddr": inst.get("public_ipaddr"),
            "ssh_direct_port": ((inst.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort"),
            "ports": inst.get("ports"),
            "dph_total": inst.get("dph_total"),
            "gpu": inst.get("gpu_name"),
            "disk": inst.get("disk_space"),
        }
        Path("/home/jaiveer2507/projects/verified-repo-distillation/data/hermes-webui-all/vast_instance.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=1)[:600])
        break
