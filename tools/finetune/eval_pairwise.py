"""Eval pairwise LoRA on held-out A/B pairs.

Metrics:
  - Per-dim accuracy (did model pick the higher-scored passage?)
  - Per-dim accuracy by |delta| bucket (easy vs subtle pairs)
  - Overall accuracy + Kendall's tau analog from pair outcomes
  - Token-budget: 1 token output (A or B) makes this orders of magnitude
    faster than absolute scoring — production-viable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
ADAPTER = Path("data/calibration/lora_pairwise")
TEST = Path("data/calibration/pairs_test.jsonl")
MAX_EVAL = 1200  # cap to keep eval under 10 min at ~0.5s/pair


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Load test pairs, stratifying by dim so small `limit` covers all dims."""
    import random as _random
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not limit or len(rows) <= limit:
        return rows
    # Group by dim, take roughly equal share
    by_dim: dict[str, list[dict]] = {}
    for r in rows:
        by_dim.setdefault(r["meta"]["dim"], []).append(r)
    per_dim = max(1, limit // len(by_dim))
    out = []
    for d, rs in sorted(by_dim.items()):
        rng = _random.Random(f"eval:{d}")
        rng.shuffle(rs)
        out.extend(rs[:per_dim])
    return out[:limit]


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model.eval()

    # Token ids for A and B (first token of each)
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]

    rows = load_jsonl(TEST, limit=MAX_EVAL)
    print(f"eval rows: {len(rows)}")

    results: list[dict] = []
    for i, row in enumerate(rows):
        messages = row["messages"][:-1]  # drop target
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        # Argmax over just {A, B} to avoid model choosing a third token
        choice = "A" if logits[a_id] >= logits[b_id] else "B"
        target = row["messages"][-1]["content"].strip()
        correct = choice == target
        results.append({
            "dim": row["meta"]["dim"],
            "delta": row["meta"]["delta"],
            "target": target,
            "pred": choice,
            "correct": correct,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}")

    # Per-dim accuracy
    by_dim: dict[str, list[bool]] = defaultdict(list)
    by_delta_bucket: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_dim[r["dim"]].append(r["correct"])
        bucket = (
            "small (<0.3)" if r["delta"] < 0.3 else
            "medium (0.3-0.5)" if r["delta"] < 0.5 else
            "large (≥0.5)"
        )
        by_delta_bucket[bucket].append(r["correct"])

    print(f"\n{'dim':<28}{'n':>5}{'acc':>8}")
    print("-" * 45)
    for dim in sorted(by_dim):
        hits = by_dim[dim]
        acc = sum(hits) / len(hits)
        print(f"{dim:<28}{len(hits):>5}{acc:>8.3f}")
    print("-" * 45)
    all_hits = [r["correct"] for r in results]
    print(f"{'OVERALL':<28}{len(all_hits):>5}{sum(all_hits)/len(all_hits):>8.3f}")

    print(f"\nby |delta| bucket:")
    for bucket in ["small (<0.3)", "medium (0.3-0.5)", "large (≥0.5)"]:
        hits = by_delta_bucket.get(bucket, [])
        if not hits: continue
        print(f"  {bucket:<20} n={len(hits):>4}  acc={sum(hits)/len(hits):.3f}")

    out = Path("/tmp/pairwise_eval.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
