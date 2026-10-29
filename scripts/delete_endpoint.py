"""Delete the arm-C endpoint so billing stops. Safe to run when none exists."""
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

H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
CP = "https://api.tokenfactory.nebius.com"
FT = Path("/home/jaiveer2507/projects/verified-repo-distillation/data/hermes-webui-all/ft")

ep_path = FT / "endpoint.json"
if not ep_path.exists():
    print("no endpoint.json — nothing to delete")
    raise SystemExit(0)

ep_id = json.loads(ep_path.read_text()).get("id")
req = urllib.request.Request(f"{CP}/v0/dedicated_endpoints/{ep_id}", headers=H, method="DELETE")
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print(f"delete {ep_id} -> HTTP {r.status}")
except urllib.error.HTTPError as exc:
    print(f"delete {ep_id} -> HTTP {exc.code} {exc.read()[:200]}")
    raise SystemExit(1)

# confirm it is gone rather than trusting the response
req = urllib.request.Request(f"{CP}/v0/dedicated_endpoints", headers=H)
with urllib.request.urlopen(req, timeout=60) as r:
    remaining = json.loads(r.read()).get("data", [])
print(f"endpoints remaining: {[e.get('name') for e in remaining]}")
