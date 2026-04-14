"""Local A/B runner that bypasses vllm — generates prose from two LoRA
adapters loaded directly via HF-transformers + PEFT. Used when a vllm
server owns the GPU and a new adapter can't be hot-loaded.

The comparison focuses on **adapter effect on writer output only**: both
adapters see the same craft brief, generate prose greedily (temperature
0) so variance is isolated to the adapter. No planning pipeline, no
scoring feedback loop — just adapter-vs-adapter head-to-head on shared
briefs, scored after the fact with :class:`app.scoring.Scorer`.

Usage::

    uv run python tools/ab_writer_lora_local.py \\
        --adapter-a data/sft/lora_writer_v1 \\
        --adapter-b data/sft/lora_writer_v3 \\
        --label-a v1 --label-b v3 \\
        --briefs-file data/sft/train.jsonl \\
        --n 3 \\
        --out data/sft/ab_v1_vs_v3_local.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU — GPU is vllm's

from app.scoring import Scorer, DIMENSION_NAMES  # noqa: E402


def _load_briefs(path: Path, n: int) -> list[dict]:
    briefs: list[dict] = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        row = json.loads(line)
        briefs.append({
            "meta": row.get("meta", {}),
            "messages": row.get("messages", []),
        })
    return briefs


def _generate_side(
    base_model: str,
    adapter_path: Path,
    briefs: list[dict],
    *,
    max_new_tokens: int = 400,
    device: str = "cpu",
) -> list[dict]:
    """Generate one completion per brief from ``base_model`` + ``adapter_path``.

    Returns list of dicts: ``{"prose": str, "brief_index": int, "meta": {...}}``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, device_map=device,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    results: list[dict] = []
    for i, brief in enumerate(briefs):
        # Take the first two messages (system + user); we generate the
        # assistant turn ourselves.
        msgs = brief["messages"][:2]
        t0 = time.perf_counter()
        templated = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        inputs = tokenizer(
            templated, return_tensors="pt", add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = out_ids[0, input_ids.shape[1]:]
        prose = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        latency = time.perf_counter() - t0
        results.append({
            "brief_index": i,
            "prose": prose,
            "latency_s": round(latency, 2),
            "meta": brief["meta"],
        })
        print(f"  [{i+1}/{len(briefs)}] t={latency:.1f}s "
              f"chars={len(prose)} prose[:80]={prose[:80]!r}")

    # Free memory before next adapter load.
    del model
    del tokenizer
    import gc
    gc.collect()
    return results


def _score_scenes(results: list[dict], *, scorer: Scorer) -> dict:
    scored: list[dict] = []
    for r in results:
        prose = r["prose"] or ""
        if not prose:
            scored.append({**r, "overall_score": None, "dimension_scores": {}})
            continue
        card = scorer.score(prose, craft_plan=None, narrator=None,
                            world=None, player_action=None)
        dims = {n: float(getattr(card, n, 0.0)) for n in DIMENSION_NAMES}
        scored.append({
            **r,
            "overall_score": float(card.overall_score),
            "dimension_scores": dims,
        })

    summary: dict[str, float] = {}
    valid = [s for s in scored if s.get("overall_score") is not None]
    if valid:
        summary["overall_mean"] = mean(s["overall_score"] for s in valid)
        for n in DIMENSION_NAMES:
            summary[n] = mean(s["dimension_scores"][n] for s in valid)
    return {"scenes": scored, "summary": summary}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ab_writer_lora_local")
    ap.add_argument("--base-model", type=str,
                    default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--adapter-a", type=Path, required=True)
    ap.add_argument("--adapter-b", type=Path, required=True)
    ap.add_argument("--label-a", type=str, default="a")
    ap.add_argument("--label-b", type=str, default="b")
    ap.add_argument("--briefs-file", type=Path, required=True,
                    help="Path to a JSONL where each row has a "
                         "messages[] list (at least system + user).")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=Path("data/sft/ab_writer_local.json"))
    ap.add_argument("--device", type=str, default="cpu",
                    choices=("cpu", "cuda"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    briefs = _load_briefs(args.briefs_file, args.n)
    print(f"Loaded {len(briefs)} briefs from {args.briefs_file}")

    print(f"\n=== {args.label_a} ({args.adapter_a}) ===")
    results_a = _generate_side(
        args.base_model, args.adapter_a, briefs,
        max_new_tokens=args.max_new_tokens, device=args.device,
    )

    print(f"\n=== {args.label_b} ({args.adapter_b}) ===")
    results_b = _generate_side(
        args.base_model, args.adapter_b, briefs,
        max_new_tokens=args.max_new_tokens, device=args.device,
    )

    scorer = Scorer()
    scored_a = _score_scenes(results_a, scorer=scorer)
    scored_b = _score_scenes(results_b, scorer=scorer)

    deltas: dict[str, float] = {}
    sa = scored_a.get("summary") or {}
    sb = scored_b.get("summary") or {}
    for k in sb:
        if k in sa:
            deltas[k] = sb[k] - sa[k]

    out = {
        "base_model": args.base_model,
        "n_briefs": len(briefs),
        "device": args.device,
        "targets": [
            {"label": args.label_a, "adapter": str(args.adapter_a)},
            {"label": args.label_b, "adapter": str(args.adapter_b)},
        ],
        args.label_a: scored_a,
        args.label_b: scored_b,
        f"delta_{args.label_b}_minus_{args.label_a}": deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.out}")

    print(f"\nSummary ({args.label_a} | {args.label_b} | Δ):")
    for k in sb:
        a = sa.get(k, 0.0)
        b = sb[k]
        d = deltas.get(k, 0.0)
        print(f"  {k:32s}  {a:7.4f} | {b:7.4f} | {d:+.4f}")


if __name__ == "__main__":
    main()
