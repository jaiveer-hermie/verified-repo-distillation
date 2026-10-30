"""Download the LoRA adapter files from Nebius.

The /content endpoint 302-redirects to a signed object URL. Sending our
Authorization header along to that storage host causes a 403, so the redirect
handler strips it.
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

OUT = Path("/home/jaiveer2507/projects/verified-repo-distillation/data/hermes-webui-all/lora")
IDS = [
    "file-d560a198217a56729fbf92957e3fa317",
    "file-d7e6bed86ad15e33bf51b4f4a056ccc7",
    "file-a6125a81573f5701bbd1c5832dada628",
    "file-c532e2ff41995632914301911b47d4ec",
    "file-03dbd2ccbcf052f68fbc63c970ee1a85",
    "file-784a45b8c8965faf8cd5019eadbf4c38",
]


class StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The signed storage URL is invalidated by two things: carrying our
        # Authorization header to the object store, and the explicit :443 that
        # Nebius puts in the redirect target. Either one yields AccessDenied.
        newurl = newurl.replace(":443/", "/")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and "tokenfactory.nebius.com" not in newurl:
            new.headers.pop("Authorization", None)
            new.headers.pop("authorization", None)
        return new


opener = urllib.request.build_opener(StripAuthOnRedirect)
AUTH = {"Authorization": f"Bearer {key}"}

for fid in IDS:
    meta = json.loads(urllib.request.urlopen(
        urllib.request.Request(f"https://api.tokenfactory.nebius.com/v1/files/{fid}",
                               headers=AUTH), timeout=30).read())
    name = (meta.get("filename") or fid).lstrip("/")
    dest = OUT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"https://api.tokenfactory.nebius.com/v1/files/{fid}/content",
                                 headers=AUTH)
    with opener.open(req, timeout=300) as r:
        data = r.read()
    dest.write_bytes(data)
    print(f"{name:55s} {len(data)/1e6:8.2f} MB")

print("\n--- adapter dir ---")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size/1e6:.2f} MB")
