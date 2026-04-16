"""Annotate a fetched corpus chapter with structured narrative breakdown.

Reads a plain-text chapter, sends it to an OpenAI-compatible server with a
structured-output schema, and writes the result to
``data/calibration/annotations/<work>/chap_NNNN.json``.

Purpose: build per-chapter beat / scene / character / relationship
annotations that roll up into arc summaries, for use as training data
(beats -> chapter) and as few-shot examples for the planner stack.

Usage::

    uv run python tools/annotate_chapter.py \\
        --work pale_lights \\
        --chapter 1 \\
        --server http://127.0.0.1:8082 \\
        --model writer_v3   # or "mythos-..." for gemma4, or base LFM id

The schema is intentionally flat and small so a 1.2B model has a shot at
producing something usable; depth can be added per-arc rollup later.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime.client import ChatMessage, InferenceClient  # noqa: E402


RAW_ROOT = Path("data/calibration/raw")
ANN_ROOT = Path("data/calibration/annotations")


ANNOTATION_SCHEMA = {
    "type": "object",
    "required": ["summary", "pov", "scenes", "characters_present"],
    "properties": {
        "summary": {"type": "string",
                    "description": "One-paragraph summary of the chapter's events."},
        "pov": {"type": "string",
                "description": "POV mode and character (e.g., 'third limited, Tristan')."},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["scene_id", "location", "dramatic_question",
                             "outcome", "beats"],
                "properties": {
                    "scene_id": {"type": "integer"},
                    "location": {"type": "string"},
                    "start_state": {"type": "string",
                                    "description": "What the protagonist wants / is doing at scene start."},
                    "end_state": {"type": "string",
                                  "description": "What has changed by scene end."},
                    "dramatic_question": {"type": "string"},
                    "outcome": {"type": "string",
                                "description": "success|failure|partial|deferred|invalid, with one clause of context."},
                    "beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["n", "description", "beat_type"],
                            "properties": {
                                "n": {"type": "integer"},
                                "description": {"type": "string",
                                                "description": "One sentence: what concretely happens in this beat."},
                                "beat_type": {"type": "string",
                                              "enum": ["action", "dialogue", "reveal",
                                                      "decision", "interiority", "transition"]},
                                "excerpt": {"type": "string",
                                            "description": "Short verbatim quote (<30 words) illustrating the beat."}
                            }
                        }
                    }
                }
            }
        },
        "characters_present": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "role"],
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string",
                             "description": "protagonist|antagonist|ally|foil|bystander|mentor|other"},
                    "first_introduced_here": {"type": "boolean"},
                    "characterization": {"type": "string",
                                         "description": "One sentence capturing how they are shown — not told."},
                    "characterization_excerpt": {"type": "string",
                                                 "description": "Verbatim quote illustrating the above (<30 words)."}
                }
            }
        },
        "relationships_shown": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["a", "b", "nature"],
                "properties": {
                    "a": {"type": "string"}, "b": {"type": "string"},
                    "nature": {"type": "string",
                               "description": "One phrase: e.g. 'bound by pact, affectionate but wary'."},
                    "evidence": {"type": "string"}
                }
            }
        },
        "world_facts_established": {
            "type": "array",
            "items": {"type": "string",
                      "description": "Concrete worldbuilding fact introduced — named places, factions, systems, idioms."}
        },
        "hooks_planted": {
            "type": "array",
            "items": {"type": "string",
                      "description": "Unresolved question / promise / threat set up for later payoff."}
        },
        "hooks_paid_off": {
            "type": "array",
            "items": {"type": "string"}
        },
        "arc_movement": {
            "type": "string",
            "description": "One sentence: how this chapter moves the book-level arc."
        }
    }
}


SYSTEM_PROMPT = """You are a literary analyst. You read one chapter of a novel or web serial and produce a structured breakdown in JSON conforming to the provided schema.

Principles:
- Be specific and verbatim when asked for excerpts. Never paraphrase into an "excerpt" field.
- Scenes are bounded by a change of location, time, or dramatic premise.
- Beats are the minimum unit within a scene: one action, one exchange, one reveal, one decision.
- "characterization" fields describe how a trait is *shown* — action, dialogue rhythm, noticed detail. Do not write personality adjectives alone.
- "world_facts_established" means concrete, named things introduced in this chapter: factions, idioms, institutions, named systems. Not generic descriptions.
- "hooks_planted" are unresolved promises, threats, or questions the reader is meant to carry forward.
- "relationships_shown" MUST be populated whenever two named characters interact on-page. For each pair, describe the nature of the relationship as it appears in this chapter (e.g. "contractor to patron god, affectionate but wary", "allies of convenience, mutually suspicious", "mentor to protege, withholding") and cite a short quote as evidence. Do not skip this field — if any two characters exchange dialogue, share a scene, or act toward each other, they form a pair to record. Prefer pairs that reveal power, trust, or emotional dynamics over purely incidental co-presence.
- If a field has no content for this chapter, return an empty list or short phrase — do not invent.

Return JSON only; no preamble."""


def _read_chapter(work: str, chapter: int) -> str:
    p = RAW_ROOT / work / f"chap_{chapter:04d}.txt"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.read_text()


async def _annotate(server: str, model: str | None, chapter_text: str,
                    max_tokens: int) -> tuple[str, float]:
    client = InferenceClient(base_url=server, model=model, timeout=600.0, retries=1)
    user = (
        "Chapter text below. Produce the full structured annotation.\n\n"
        "---\n"
        f"{chapter_text}\n"
        "---"
    )
    t0 = time.perf_counter()
    raw = await client.chat_structured(
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user),
        ],
        json_schema=ANNOTATION_SCHEMA,
        schema_name="chapter_annotation",
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return raw, time.perf_counter() - t0


async def _main_async(args: argparse.Namespace) -> None:
    text = _read_chapter(args.work, args.chapter)
    print(f"Chapter {args.chapter}: {len(text.split())} words")
    raw, latency = await _annotate(args.server, args.model, text, args.max_tokens)
    try:
        parsed = json.loads(raw)
    except Exception as e:
        print(f"JSON parse failed: {e}\nRaw:\n{raw[:2000]}")
        sys.exit(2)

    out_dir = ANN_ROOT / args.work
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or (args.model or "default").split("/")[-1]
    out = out_dir / f"chap_{args.chapter:04d}.{label}.json"
    payload = {
        "work": args.work,
        "chapter": args.chapter,
        "model": args.model,
        "label": label,
        "latency_s": round(latency, 2),
        "annotation": parsed,
    }
    out.write_text(json.dumps(payload, indent=2))
    n_scenes = len(parsed.get("scenes", []))
    n_beats = sum(len(s.get("beats", [])) for s in parsed.get("scenes", []))
    n_chars = len(parsed.get("characters_present", []))
    print(f"  -> {out}  ({latency:.1f}s, {n_scenes} scenes, {n_beats} beats, "
          f"{n_chars} characters)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="annotate_chapter")
    ap.add_argument("--work", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default=None,
                    help="Model id to request (vllm). Omit for llama-server default.")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--label", default=None,
                    help="Label for output filename (default: derived from --model).")
    return ap.parse_args()


def main() -> None:
    asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    main()
