"""Stage 5 — TRAIN.

Exports the verified corpus to a chat-format JSONL and emits the QLoRA job spec
for Nebius AI Cloud. The training call itself happens on a GPU box; nothing here
pretends to train locally.

Two decisions worth defending:

1. **Replay slice.** 15-25% public-code examples are mixed in. Supervised replay
   is the intervention that most consistently protects old capabilities
   (arXiv:2610.30191); regularisation on the adapter's subspace does not.

2. **Test files are never in the training target.** The gate proves a commit by
   running its test. If the model is trained to *write* that test, the corpus
   teaches it to satisfy the instrument. `pairs.source_only_patch` strips those
   hunks before they can reach the target here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from vrd.pairs import Pair
from vrd.ingest import read_jsonl

SYSTEM = (
    "You are a repository editing agent. You are given the current contents of the "
    "files you may change and a request. Reply with search/replace blocks in the "
    "exact format requested. Never rewrite a file wholesale when a small block will "
    "do, and never modify tests."
)


@dataclass
class TrainingConfig:
    base_model: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"
    quantization: str = "nf4"
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 16
    max_seq_len: int = 4096
    replay_fraction: float = 0.20
    gpu: str = "1x H100 80GB"
    seed: int = 17


def to_chat_example(pair: Pair) -> dict:
    """One training example. Context bytes are identical to what arm A/C see at eval."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": pair.to_prompt()},
            {"role": "assistant", "content": pair.to_target()},
        ],
        "meta": {
            "sha": pair.sha,
            "repo": pair.repo,
            "files": pair.changed_files,
            "verified_by": pair.test_files,
        },
    }


def load_replay_examples(cache_dir: str | Path | None = None) -> list[dict]:
    """Public-code replay examples, if the operator has staged any.

    Staged as JSONL with the same `messages` shape. No synthetic data is ever
    generated here: an empty list means the run reports replay_fraction=0.0
    instead of silently claiming a mix it did not apply.
    """
    if not cache_dir:
        return []
    path = Path(cache_dir) / "replay.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def export_dataset(pairs: list[Pair], out_dir: str | Path,
                   replay_dir: str | Path | None = None,
                   time_split: str | None = None) -> dict:
    """Write train.jsonl and a dataset card. Splits are temporal, never random.

    A random split leaks the future into training. `time_split` is an ISO date:
    pairs dated before it train, pairs dated after it are the held-out eval set.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, eval_pairs = pairs, []
    if time_split:
        train_pairs = [p for p in pairs if p.commit_date < time_split]
        eval_pairs = [p for p in pairs if p.commit_date >= time_split]

    train_rows = [to_chat_example(p) for p in train_pairs]
    replay_rows = load_replay_examples(replay_dir)
    n_replay = int(len(train_rows) * TrainingConfig.replay_fraction
                   / max(1e-9, 1 - TrainingConfig.replay_fraction)) if train_rows else 0
    replay_used = replay_rows[:n_replay]
    mixed = train_rows + replay_used

    (out_dir / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mixed), encoding="utf-8")
    (out_dir / "eval.jsonl").write_text(
        "".join(json.dumps(to_chat_example(p), ensure_ascii=False) + "\n" for p in eval_pairs),
        encoding="utf-8")

    card = {
        "train_examples": len(train_rows),
        "replay_examples": len(replay_used),
        "replay_fraction_actual": round(len(replay_used) / max(len(mixed), 1), 4),
        "eval_examples": len(eval_pairs),
        "time_split": time_split,
        "corpus_source": "commits verified by counterfactual test injection",
    }
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n")
    return card


def qlora_job_spec(dataset_path: str, out_path: str, cfg: TrainingConfig | None = None) -> dict:
    """A Nebius AI Cloud job spec. Written as data so the submission shows the recipe."""
    cfg = cfg or TrainingConfig()
    return {
        "name": "vrd-qlora",
        "image": "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        "gpu": cfg.gpu,
        "timeout_s": 6 * 3600,
        "env": {
            "BASE_MODEL": cfg.base_model,
            "DATASET": dataset_path,
            "OUTPUT_DIR": out_path,
        },
        "command": [
            "bash", "-lc",
            "pip install -q 'torch>=2.4' 'transformers>=4.44' 'peft>=0.12' "
            "'trl>=0.9' 'bitsandbytes>=0.43' 'datasets>=2.20' && "
            "python train_qlora.py "
            f"--base {cfg.base_model} --data {dataset_path} --out {out_path} "
            f"--rank {cfg.rank} --alpha {cfg.alpha} --dropout {cfg.dropout} "
            f"--lr {cfg.learning_rate} --epochs {cfg.epochs} "
            f"--batch {cfg.batch_size} --grad-accum {cfg.grad_accum} "
            f"--max-seq {cfg.max_seq_len} --quant {cfg.quantization} "
            f"--targets {cfg.target_modules} --seed {cfg.seed}",
        ],
        "config": asdict(cfg),
    }


TRAIN_SCRIPT = '''"""QLoRA on NVIDIA Nemotron. Runs on the Nebius GPU box, not locally."""
import argparse, json, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=4096); ap.add_argument("--quant", default="nf4")
    ap.add_argument("--targets", default="all-linear"); ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=a.quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.base, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
        target_modules=a.targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=a.data, split="train")

    def fmt(row):
        text = tok.apply_chat_template(row["messages"], tokenize=False,
                                       add_generation_prompt=False)
        return tok(text, truncation=True, max_length=a.max_seq)

    ds = ds.map(fmt, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
        num_train_epochs=a.epochs, bf16=True, logging_steps=5, save_strategy="epoch",
        optim="paged_adamw_8bit", seed=a.seed, report_to=[])
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(json.dumps({"saved": a.out, "examples": len(ds)}))


if __name__ == "__main__":
    main()
'''


def write_train_script(out_dir: str | Path) -> Path:
    path = Path(out_dir) / "train_qlora.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TRAIN_SCRIPT, encoding="utf-8")
    return path
