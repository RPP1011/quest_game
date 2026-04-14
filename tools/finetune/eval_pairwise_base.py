"""Eval BASE LFM (no LoRA) on the same pairwise test set.

Floor check — tells us whether the finetune helped, hurt, or did nothing.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
TEST = Path("data/calibration/pairs_test.jsonl")
MAX_EVAL = 1200


def load_jsonl(path, limit):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_dim: dict[str, list[dict]] = {}
    for r in rows:
        by_dim.setdefault(r["meta"]["dim"], []).append(r)
    per_dim = max(1, limit // len(by_dim))
    out = []
    for d, rs in sorted(by_dim.items()):
        rng = random.Random(f"eval:{d}")
        rng.shuffle(rs)
        out.extend(rs[:per_dim])
    return out[:limit]


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]

    rows = load_jsonl(TEST, MAX_EVAL)
    print(f"eval rows: {len(rows)}")

    by_dim = defaultdict(list)
    for i, row in enumerate(rows):
        messages = row["messages"][:-1]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        choice = "A" if logits[a_id] >= logits[b_id] else "B"
        target = row["messages"][-1]["content"].strip()
        by_dim[row["meta"]["dim"]].append(choice == target)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}")

    print(f"\n{'dim':<28}{'n':>5}{'acc':>8}")
    print("-" * 45)
    all_hits = []
    for dim in sorted(by_dim):
        hits = by_dim[dim]
        all_hits.extend(hits)
        print(f"{dim:<28}{len(hits):>5}{sum(hits)/len(hits):>8.3f}")
    print(f"{'OVERALL':<28}{len(all_hits):>5}{sum(all_hits)/len(all_hits):>8.3f}")


if __name__ == "__main__":
    main()
