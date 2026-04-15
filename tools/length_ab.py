"""One-off length A/B: generate N samples per variant from a shared context
and report length + heuristic-quality stats.

Variants are defined inline (model id + optional prompt tweaks); edit the
``VARIANTS`` list to add more. Hits the live vllm server at
``--server`` (default 127.0.0.1:8082). Requires writer LoRAs to already
be hot-loaded on the server.

Usage::

    uv run python tools/length_ab.py --n 10 \\
        --out data/length_ab/$(date +%Y%m%d-%H%M).json

Outputs: per-sample prose + metrics, and a per-variant summary table to
stdout. Metrics are mechanical (word/sentence/paragraph count, repeated
bigram ratio) plus Scorer heuristic dims (sentence_variance, pacing,
sensory_density, dialogue_ratio).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime.client import ChatMessage, InferenceClient  # noqa: E402
from app.scoring import Scorer  # noqa: E402


SYSTEM_BASE = (
    "You are the writer. Execute the plan faithfully and make the prose "
    "feel alive. SV-quest style: 2nd person past tense. No preface, no "
    "plan summaries, no meta-commentary. Render spoken words as quoted "
    "dialogue with beats. Vary sentence length for rhythm."
)

SYSTEM_LENGTH = SYSTEM_BASE + (
    " Write a full chapter-length update: target 1,800-2,200 words "
    "across 12-20 paragraphs. Let the scene breathe — sensory register, "
    "dialogue in real time with reactions and subtext, closing beat "
    "that earns the next choice. Do not summarize or compress."
)

USER = """## World
- Mara: wiry traveler, salt-stained boots, leather notebook, searching for her missing brother.
- Hale: barrel-chested innkeeper of The Grey Kettle, remembers every face twice.
- The Grey Kettle: low-beamed coastal tavern, woodsmoke and brine.

## Plan
- Mara crosses the threshold of the tavern; register the room through her senses.
- She spots Hale and approaches the counter.
- She asks, carefully, after her missing brother — half-hoping, half-braced.
- Hale answers in a way that gives her less than she wanted, more than she expected.

## Player Action
Mara pushes open the tavern door and looks for the innkeeper.

Write the scene now. Output prose only — no headings, no labels.
"""


VARIANTS: list[dict[str, Any]] = [
    {
        "label": "base_nolen",
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "system": SYSTEM_BASE,
        "user": USER,
        "max_tokens": 3000,
    },
    {
        "label": "base_lenhint",
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "system": SYSTEM_LENGTH,
        "user": USER,
        "max_tokens": 3000,
    },
    {
        "label": "writer_v3_nolen",
        "model": "writer_v3",
        "system": SYSTEM_BASE,
        "user": USER,
        "max_tokens": 3000,
    },
    {
        "label": "writer_v3_lenhint",
        "model": "writer_v3",
        "system": SYSTEM_LENGTH,
        "user": USER,
        "max_tokens": 3000,
    },
]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD = re.compile(r"\b\w+\b")


def _metrics(prose: str) -> dict[str, float]:
    prose = prose.strip()
    words = _WORD.findall(prose)
    sentences = [s for s in _SENT_SPLIT.split(prose) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", prose) if p.strip()]
    lower_words = [w.lower() for w in words]
    bigrams = list(zip(lower_words, lower_words[1:]))
    rep_bigram_ratio = (
        1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    )
    ttr = len(set(lower_words)) / len(lower_words) if lower_words else 0.0
    return {
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "type_token_ratio": ttr,
        "repeated_bigram_ratio": rep_bigram_ratio,
    }


async def _sample(client: InferenceClient, variant: dict[str, Any]) -> str:
    return await client.chat(
        messages=[
            ChatMessage(role="system", content=variant["system"]),
            ChatMessage(role="user", content=variant["user"]),
        ],
        temperature=0.8,
        max_tokens=variant["max_tokens"],
    )


async def _run_variant(
    client_base_url: str, variant: dict[str, Any], n: int,
) -> list[dict[str, Any]]:
    # One client per variant so we can set the model id.
    client = InferenceClient(base_url=client_base_url, model=variant["model"],
                             timeout=180.0, retries=1)
    # Concurrent sampling — vllm batches them efficiently.
    tasks = [_sample(client, variant) for _ in range(n)]
    t0 = time.perf_counter()
    prose_list = await asyncio.gather(*tasks, return_exceptions=True)
    latency = time.perf_counter() - t0

    scorer = Scorer()
    samples: list[dict[str, Any]] = []
    for i, prose in enumerate(prose_list):
        if isinstance(prose, Exception):
            samples.append({"index": i, "error": str(prose)})
            continue
        card = scorer.score(prose, craft_plan=None, narrator=None,
                            world=None, player_action=None)
        samples.append({
            "index": i,
            "prose": prose,
            "metrics": _metrics(prose),
            "scorecard": {
                "overall": float(card.overall_score),
                "sentence_variance": float(card.sentence_variance),
                "dialogue_ratio": float(card.dialogue_ratio),
                "pacing": float(card.pacing),
                "sensory_density": float(card.sensory_density),
            },
        })
    return [{"_latency_s": latency}, *samples]


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    real = [s for s in samples if "metrics" in s]
    if not real:
        return {"n_ok": 0}

    def ms(key: str, group: str = "metrics") -> tuple[float, float]:
        vals = [s[group][key] for s in real]
        return fmean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

    words_mean, words_sd = ms("words")
    sent_mean, sent_sd = ms("sentences")
    para_mean, para_sd = ms("paragraphs")
    ttr_mean, _ = ms("type_token_ratio")
    rep_mean, _ = ms("repeated_bigram_ratio")
    overall_mean, _ = ms("overall", group="scorecard")
    sv_mean, _ = ms("sentence_variance", group="scorecard")
    return {
        "n_ok": len(real),
        "words_mean": words_mean, "words_sd": words_sd,
        "sentences_mean": sent_mean, "sentences_sd": sent_sd,
        "paragraphs_mean": para_mean, "paragraphs_sd": para_sd,
        "type_token_ratio_mean": ttr_mean,
        "repeated_bigram_ratio_mean": rep_mean,
        "scorer_overall_mean": overall_mean,
        "scorer_sentence_variance_mean": sv_mean,
    }


def _print_table(results: dict[str, Any]) -> None:
    hdr = ("variant", "n", "words", "sent", "para", "rep_bi", "score")
    print(f"{hdr[0]:<22} {hdr[1]:>3} {hdr[2]:>11} {hdr[3]:>9} {hdr[4]:>7} {hdr[5]:>7} {hdr[6]:>6}")
    print("-" * 72)
    for label, v in results["variants"].items():
        s = v["summary"]
        if not s.get("n_ok"):
            print(f"{label:<22}  (no successful samples)")
            continue
        words = f"{s['words_mean']:.0f}±{s['words_sd']:.0f}"
        sent = f"{s['sentences_mean']:.1f}±{s['sentences_sd']:.1f}"
        para = f"{s['paragraphs_mean']:.1f}"
        rep = f"{s['repeated_bigram_ratio_mean']:.2f}"
        overall = f"{s['scorer_overall_mean']:.2f}"
        print(f"{label:<22} {s['n_ok']:>3} {words:>11} {sent:>9} {para:>7} {rep:>7} {overall:>6}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="length_ab")
    ap.add_argument("--server", default="http://127.0.0.1:8082")
    ap.add_argument("--n", type=int, default=10,
                    help="Samples per variant.")
    ap.add_argument("--out", type=Path,
                    default=Path("data/length_ab") / f"run-{int(time.time())}.json")
    return ap.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    results: dict[str, Any] = {
        "server": args.server, "n": args.n,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variants": {},
    }
    for variant in VARIANTS:
        label = variant["label"]
        print(f"\n=== {label}  (model={variant['model']}) ===")
        samples = await _run_variant(args.server, variant, args.n)
        summary = _summarize(samples)
        results["variants"][label] = {"variant": variant, "samples": samples,
                                      "summary": summary}
        print(f"  {summary.get('n_ok', 0)}/{args.n} ok  "
              f"mean_words={summary.get('words_mean', 0):.0f}  "
              f"mean_overall={summary.get('scorer_overall_mean', 0):.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}\n")
    _print_table(results)


def main() -> None:
    asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    main()
