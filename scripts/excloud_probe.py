"""Query the Excloud compute API directly: quota, instance catalogue, instances.

Read-only GETs only. Prints status and a trimmed body; never prints the key.
"""
import json
import urllib.error
import urllib.request
from pathlib import Path

key = ""
for line in (Path.home() / ".hermes" / ".env").read_text().splitlines():
    line = line.strip()
    if line.startswith("EXCLOUD_API_KEY") and "=" in line:
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

BASE = "https://compute.excloud.in"
H = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

PATHS = [
    "/me",
    "/org",
    "/quota",
    "/compute/list",
    "/compute/instancetype/list",
    "/instancetype/list",
    "/compute/instancetype",
    "/compute/image/list",
    "/billing/estimate",
    "/billing/get",
]

for path in PATHS:
    try:
        req = urllib.request.Request(BASE + path, headers=H)
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read() or b"{}")
        if isinstance(body, dict):
            data = body.get("data", body)
            if isinstance(data, list):
                print(f"{path:30s} 200  list[{len(data)}]")
                for item in data[:14]:
                    if isinstance(item, dict):
                        keys = [k for k in ("name", "instance_type", "vcpu", "memory",
                                            "gpu", "gpu_type", "price", "hourly", "id",
                                            "status", "quota", "limit", "used") if k in item]
                        print("    ", {k: item[k] for k in keys})
                    else:
                        print("    ", item)
            else:
                print(f"{path:30s} 200  {json.dumps(data)[:300]}")
        else:
            print(f"{path:30s} 200  {json.dumps(body)[:300]}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()[:200].decode("utf-8", "replace")
        print(f"{path:30s} {exc.code}  {raw}")
    except Exception as exc:
        print(f"{path:30s} ERR  {str(exc)[:100]}")
