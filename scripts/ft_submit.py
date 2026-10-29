"""Submit the arm-C fine-tune: Qwen3-30B-A3B LoRA on the 277 verified pre-split pairs.

Nemotron cannot be a fine-tuning base on this platform (HTTP 422 for Super-120B
and Nano-30B), so arm C uses a supported base at the same scale. The framework is
model-agnostic: the corpus, gate and oracle are what transfer, and this same
recipe applies to Nemotron wherever it becomes tunable.

Budget: 1 epoch, LoRA r=16, context 16384, packing off.
"""
import json
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

BASE = "https://api.tokenfactory.nebius.com/v1"
H = {"Authorization": f"Bearer {key}"}
ROOT = Path("/home/jaiveer2507/projects/verified-repo-distillation/data/hermes-webui-all")
EXPORT = ROOT / "export" / "train.jsonl"
FT = ROOT / "ft"
FT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in EXPORT.read_text().splitlines() if l.strip()]
# Some examples exceed the training context: one was 16,751 tokens against a
# 16,384 limit, which fails the whole job at dataset validation. Drop the tail
# rather than let one long example kill the run.
MAX_CHARS = 110_000          # ~30k tokens at 3.7 chars/token
kept = [r for r in rows
        if sum(len(m.get("content", "")) for m in r["messages"]) <= MAX_CHARS]
print(f"kept {len(kept)}/{len(rows)} examples under {MAX_CHARS} chars")
rows = kept
payload = FT / "train_ft.jsonl"
with payload.open("w") as fh:
    for r in rows:
        fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=True) + "\n")
print(f"training rows: {len(rows)}  payload: {payload.stat().st_size/1e6:.1f} MB")

boundary = "----vrdtrain"
body = (
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nfine-tune\r\n"
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"train_ft.jsonl\"\r\n"
    f"Content-Type: application/jsonl\r\n\r\n{payload.read_text()}\r\n"
    f"--{boundary}--\r\n"
).encode()
req = urllib.request.Request(BASE + "/files", data=body, headers={
    **H, "Content-Type": f"multipart/form-data; boundary={boundary}"})
with urllib.request.urlopen(req, timeout=600) as r:
    file_obj = json.loads(r.read())
print(f"uploaded: id={file_obj['id']} bytes={file_obj.get('bytes')}")

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
job_req = {
    "model": MODEL,
    "training_file": file_obj["id"],
    "suffix": "vrd-verified",
    "hyperparameters": {
        "lora": True, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "n_epochs": 1, "context_length": 32768, "batch_size": 4,
        "learning_rate": 1e-4,
        # Mandatory for some large MoE models (docs: "packing = false is not
        # allowed for those models"); the API rejects false with 422.
        "packing": True,
    },
}
req = urllib.request.Request(BASE + "/fine_tuning/jobs", data=json.dumps(job_req).encode(),
                             headers={**H, "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        job = json.loads(r.read())
except urllib.error.HTTPError as exc:
    print("CREATE FAILED:", exc.code, exc.read()[:600])
    raise SystemExit(1)

job_id = job["id"]
(FT / "job_id.txt").write_text(job_id)
print("job created:", job_id, "status:", job.get("status"))

for i in range(40):
    time.sleep(30)
    req = urllib.request.Request(f"{BASE}/fine_tuning/jobs/{job_id}", headers=H)
    with urllib.request.urlopen(req, timeout=60) as r:
        cur = json.loads(r.read())
    print(f"  [{i}] {cur.get('status')} tokens={cur.get('trained_tokens')} "
          f"steps={cur.get('trained_steps')}/{cur.get('total_steps')}", flush=True)
    (FT / "job_state.json").write_text(json.dumps(cur, indent=2))
    if cur.get("status") in ("succeeded", "failed", "cancelled"):
        print("TERMINAL:", json.dumps(cur)[:900])
        break
