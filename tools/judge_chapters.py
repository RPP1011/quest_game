"""Chapter-scale LLM judging.

Per chapter of a fetched corpus, assembles the chapter-scale dim rubrics
under ``prompts/scoring/chapter_dims/`` and calls a model for a single
batched structured-output judgement covering all 8 dims at once.

Writes ``data/calibration/annotations/<work>/chap_NNNN.judge.<label>.json``
with per-dim {score, rationale} and aggregates to a work-level rollup.

Usage::

    uv run python tools/judge_chapters.py \\
        --work pale_lights \\
        --server http://127.0.0.1:8082 \\
        --label gemma4

The judge runs full chapters (no slicing). Length is a signal, not
something to normalize away.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime.client import ChatMessage, InferenceClient  # noqa: E402


RAW_ROOT = Path("data/calibration/raw")
ANN_ROOT = Path("data/calibration/annotations")
RUBRICS_DIR = ROOT / "prompts" / "scoring" / "chapter_dims"


DEFAULT_DIMS = [
    "tension_execution",
    "emotional_trajectory",
    "choice_hook_quality",
    "update_self_containment",
    "voice_distinctiveness",
    "thematic_presence",
    "subtext_presence",
    "interiority_depth",
]


def _load_rubric(dim: str) -> str:
    p = RUBRICS_DIR / f"{dim}.j2"
    if not p.is_file():
        raise FileNotFoundError(f"missing rubric: {p}")
    return p.read_text()


def _build_schema(dims: list[str]) -> dict:
    return {
        "type": "object",
        "required": dims,
        "properties": {
            d: {
                "type": "object",
                "required": ["score", "rationale"],
                "properties": {
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
            }
            for d in dims
        },
    }


SYSTEM_PROMPT = """You are a strict literary-quality judge. You score a FULL CHAPTER of a novel or web serial on several dimensions simultaneously. Use the full 0.0-1.0 range and avoid clustering near 0.5. Each rubric below is self-contained and anchored — follow its anchors rather than importing prior priors. Score the chapter on the page, not the situation it describes. For each dimension, emit a score in [0.0, 1.0] and a one-sentence rationale as defined by the rubric. Return ONLY the JSON object matching the response schema — no preamble."""


def _build_user_prompt(chapter_text: str, dims: list[str]) -> str:
    parts = [
        "Score the following chapter on each of these dimensions.",
        "",
        "DIMENSIONS (self-contained rubrics):",
        "",
    ]
    for d in dims:
        parts.append(f"=== {d} ===")
        parts.append(_load_rubric(d))
        parts.append("")
    parts.append("CHAPTER:")
    parts.append("<<<")
    parts.append(chapter_text)
    parts.append(">>>")
    parts.append("")
    parts.append("Return JSON only.")
    return "\n".join(parts)


async def _judge_one(client: InferenceClient, chapter_text: str,
                     dims: list[str], max_tokens: int) -> tuple[dict, float]:
    t0 = time.perf_counter()
    raw = await client.chat_structured(
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=_build_user_prompt(chapter_text, dims)),
        ],
        json_schema=_build_schema(dims),
        schema_name="chapter_scores",
        temperature=0.2,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0
    return json.loads(raw), latency


async def _main_async(args: argparse.Namespace) -> None:
    dims = DEFAULT_DIMS if args.dims == "all" else args.dims.split(",")
    chapters_dir = RAW_ROOT / args.work
    out_dir = ANN_ROOT / args.work
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(chapters_dir.glob("chap_*.txt"))
    if args.start is not None:
        files = [f for f in files if int(f.stem.split("_")[-1]) >= args.start]
    if args.end is not None:
        files = [f for f in files if int(f.stem.split("_")[-1]) <= args.end]

    client = InferenceClient(base_url=args.server, model=args.model,
                             timeout=600.0, retries=1)

    for f in files:
        n = int(f.stem.split("_")[-1])
        out_file = out_dir / f"chap_{n:04d}.judge.{args.label}.json"
        if out_file.is_file() and not args.force:
            continue
        text = f.read_text()
        word_count = len(text.split())
        try:
            scores, latency = await _judge_one(
                client, text, dims, args.max_tokens,
            )
        except Exception as e:
            print(f"ch {n}: FAILED {e}")
            continue
        out = {
            "work": args.work,
            "chapter": n,
            "word_count": word_count,
            "model": args.model,
            "label": args.label,
            "latency_s": round(latency, 2),
            "dims": dims,
            "scores": scores,
        }
        out_file.write_text(json.dumps(out, indent=2))
        dim_scores = " ".join(
            f"{d[:6]}={scores[d]['score']:.2f}" for d in dims if d in scores
        )
        print(f"ch {n}: {word_count}w {latency:.1f}s  {dim_scores}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="judge_chapters")
    ap.add_argument("--work", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="gemma4")
    ap.add_argument("--dims", default="all",
                    help="comma-separated dim names, or 'all' for the default 8.")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--start", type=int, default=None, help="start chapter (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="end chapter (inclusive)")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    main()
