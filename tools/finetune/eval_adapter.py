"""Run the LoRA adapter on held-out test rows; compare scores to Claude.

Loads LFM2.5-1.2B + LoRA, generates (score, rationale) for each row in
data/calibration/test.jsonl, then reports per-dim MAE and Pearson r vs the
Claude teacher scores embedded in each row's meta.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
ADAPTER = Path("data/calibration/lora_adapter")
TEST = Path("data/calibration/test.jsonl")


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def mae(xs, ys):
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs) if xs else 0.0


def extract_scores(text: str) -> dict[str, float] | None:
    """Parse batched response: {dim: {score, rationale}} or {dim: float}."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start: end + 1])
        out = {}
        for dim, v in obj.items():
            if isinstance(v, (int, float)):
                out[dim] = float(v)
            elif isinstance(v, dict) and "score" in v:
                out[dim] = float(v["score"])
        return out
    except Exception:
        return None


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model.eval()

    rows = [json.loads(line) for line in TEST.read_text().splitlines() if line.strip()]
    print(f"eval rows: {len(rows)}")

    results = []
    for i, row in enumerate(rows):
        messages = row["messages"][:-1]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=1500, do_sample=True,
                temperature=0.3, top_p=0.95, repetition_penalty=1.1,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores = extract_scores(gen) or {}
        target_dims = row["meta"]["dims"]
        results.append({
            "work": row["meta"]["work_id"],
            "passage": row["meta"]["passage_id"],
            "claude": target_dims,
            "lora": scores,
            "raw": gen[:400],
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(rows)}")

    by_dim = defaultdict(list)
    failed = 0
    for r in results:
        if not r["lora"]:
            failed += 1
            continue
        for dim, claude_v in r["claude"].items():
            if dim in r["lora"]:
                by_dim[dim].append((r["lora"][dim], claude_v))

    print(f"\nparse failures: {failed}/{len(results)}")
    print(f"{'dim':<28}{'n':>4}{'MAE':>8}{'r':>8}")
    print("-" * 50)
    all_pairs = []
    for dim in sorted(by_dim):
        pairs = by_dim[dim]
        xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
        print(f"{dim:<28}{len(pairs):>4}{mae(xs, ys):>8.3f}{pearson(xs, ys):>8.3f}")
        all_pairs.extend(pairs)
    xs = [p[0] for p in all_pairs]; ys = [p[1] for p in all_pairs]
    print("-" * 50)
    print(f"{'OVERALL':<28}{len(all_pairs):>4}{mae(xs, ys):>8.3f}{pearson(xs, ys):>8.3f}")

    out_path = Path("/tmp/lora_eval.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    main()
