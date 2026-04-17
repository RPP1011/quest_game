"""Chapter-scale scoring for rollout chapters (Phase 4 of story-rollout).

Wraps the existing 8-dim chapter judge (rubrics under
``prompts/scoring/chapter_dims/``) and the same structured-output
schema used by ``tools/judge_chapters.py`` and ``tools/strategy_sweep.py``,
exposing it as a simple async function the rollout harness can call
after each chapter is committed.

Persists scores in two places:
- ``RolloutChapter.judge_scores`` (one JSON blob on the chapter row,
  fast access for one chapter)
- ``kb_chapter_scores`` (one row per dim, indexed for cross-rollout
  aggregation queries)
"""
from __future__ import annotations

import json
from pathlib import Path

from app.runtime.client import ChatMessage, InferenceClient
from app.world.schema import RolloutChapter
from app.world.state_manager import WorldStateManager


REPO = Path(__file__).resolve().parent.parent.parent
RUBRICS_DIR = REPO / "prompts" / "scoring" / "chapter_dims"

DEFAULT_DIMS: tuple[str, ...] = (
    "tension_execution",
    "emotional_trajectory",
    "choice_hook_quality",
    "update_self_containment",
    "voice_distinctiveness",
    "thematic_presence",
    "subtext_presence",
    "interiority_depth",
)


JUDGE_SYSTEM = (
    "You are a strict literary-quality judge. You score a FULL CHAPTER of "
    "a novel or web serial on several dimensions simultaneously. Use the "
    "full 0.0-1.0 range and avoid clustering near 0.5. Each rubric below "
    "is self-contained and anchored — follow its anchors rather than "
    "importing prior priors. Score the chapter on the page, not the "
    "situation it describes. For each dimension, emit a score in [0.0, "
    "1.0] and a one-sentence rationale as defined by the rubric. Return "
    "ONLY the JSON object matching the response schema — no preamble."
)


def _load_rubric(dim: str) -> str:
    path = RUBRICS_DIR / f"{dim}.j2"
    if not path.is_file():
        raise FileNotFoundError(f"missing rubric: {path}")
    return path.read_text()


def _build_schema(dims: list[str]) -> dict:
    return {
        "type": "object",
        "required": list(dims),
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


async def score_chapter(
    *,
    client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> dict[str, dict]:
    """Run the 8-dim judge on one chapter. Returns ``{dim: {score, rationale}}``.

    Single batched structured call. Designed to run in seconds.
    """
    use_dims = list(dims or DEFAULT_DIMS)
    schema = _build_schema(use_dims)
    raw = await client.chat_structured(
        messages=[
            ChatMessage(role="system", content=JUDGE_SYSTEM),
            ChatMessage(role="user", content=_build_user_prompt(chapter_text, use_dims)),
        ],
        json_schema=schema,
        schema_name="chapter_scores",
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=False,  # judge must emit pure JSON, no chain-of-thought
    )
    # Defensive JSON extraction: if the model leaked any prefix before the
    # JSON object (rare but possible even with structured output), pull
    # the first {...} block out by brace matching.
    raw = raw.strip()
    if raw and not raw.startswith("{"):
        i = raw.find("{")
        if i >= 0:
            raw = raw[i:]
    parsed = json.loads(raw)
    # Ensure each dim is well-formed; drop missing dims rather than crash.
    out: dict[str, dict] = {}
    for d in use_dims:
        v = parsed.get(d)
        if isinstance(v, dict) and "score" in v:
            out[d] = {
                "score": float(v.get("score", 0.0)),
                "rationale": str(v.get("rationale", "")),
            }
    return out


async def score_and_persist_chapter(
    *,
    client: InferenceClient,
    world: WorldStateManager,
    rollout_id: str,
    chapter: RolloutChapter,
    dims: list[str] | None = None,
) -> dict[str, dict]:
    """Score a RolloutChapter's prose, persist to both the chapter row's
    judge_scores blob AND the kb_chapter_scores table.

    Idempotent: if the chapter already has judge_scores, this returns them
    without re-scoring. Pass ``dims=...`` explicitly to force a re-score
    on a different dim set.
    """
    if chapter.judge_scores and not dims:
        return chapter.judge_scores

    scores = await score_chapter(
        client=client, chapter_text=chapter.prose, dims=dims,
    )
    # Persist per-dim rows for aggregation
    world.save_chapter_scores(rollout_id, chapter.chapter_index, scores)
    # Update the chapter row's judge_scores blob
    flat = {dim: payload["score"] for dim, payload in scores.items()}
    chapter.judge_scores = flat
    world.save_rollout_chapter(chapter)
    return scores
