"""Chapter-scale scoring for rollout chapters.

Two scoring modes:

1. **Logprob-weighted absolute** (``score_chapter_logprob``): Prompts for
   integer 1–10 per dim, extracts the token-level probability distribution
   via logprobs, computes E[score] (continuous [0,1]) and confidence
   (1 - normalized_entropy). Zero extra forward passes — just richer
   extraction from the same call.

2. **Pairwise comparison** (``compare_chapters``): A/B prompt, judge
   picks one. P(A wins) from logprobs on the A/B token. No quantization.

The legacy ``score_chapter`` (float 0.0–1.0 structured output) is kept
for backwards compatibility but new code should use ``score_chapter_logprob``.

Collapsed dims (from PCA on first rollout):
- prose_execution: tension + emotion + voice + theme + interiority (PC1)
- subtext: kept as own dim (anti-correlated with PC1 in current rubric)
- hook_quality: choice_hook_quality (PC3, independent)

Dropped: update_self_containment (rewards recap in serial fiction),
one of emotional_trajectory/interiority_depth (r=0.84, redundant).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from app.runtime.client import ChatMessage, ChatWithLogprobs, InferenceClient
from app.world.schema import RolloutChapter
from app.world.state_manager import WorldStateManager


REPO = Path(__file__).resolve().parent.parent.parent
RUBRICS_DIR = REPO / "prompts" / "scoring" / "chapter_dims"

LEGACY_DIMS: tuple[str, ...] = (
    "tension_execution", "emotional_trajectory", "choice_hook_quality",
    "update_self_containment", "voice_distinctiveness", "thematic_presence",
    "subtext_presence", "interiority_depth",
)

COLLAPSED_DIMS: tuple[str, ...] = (
    "prose_execution",
    "subtext",
    "hook_quality",
)

DEFAULT_DIMS = COLLAPSED_DIMS


def _load_rubric(dim: str) -> str:
    path = RUBRICS_DIR / f"{dim}.j2"
    if not path.is_file():
        raise FileNotFoundError(f"missing rubric: {path}")
    return path.read_text()


# ---------------------------------------------------------------------------
# Logprob-weighted absolute scoring
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_LOGPROB = (
    "You are a strict literary-quality judge. You score a FULL CHAPTER of "
    "a novel or web serial on several dimensions. Use the full 1–10 range. "
    "Each rubric is self-contained and anchored — follow its anchors. "
    "Score the chapter on the page, not the situation it describes.\n\n"
    "For each dimension, output ONLY the integer score (1–10) on its own "
    "line, prefixed by the dimension name and a colon. Example:\n"
    "prose_execution: 7\n"
    "subtext: 5\n"
    "hook_quality: 8\n\n"
    "No rationale, no explanation, no JSON. Just name: integer."
)


def _build_logprob_user_prompt(chapter_text: str, dims: list[str]) -> str:
    parts = [
        "Score the following chapter on each dimension (1–10 integer).",
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
    parts.append("Score each dimension now. Output format: dim_name: integer")
    return "\n".join(parts)


def _parse_logprob_scores(
    result: ChatWithLogprobs, dims: list[str],
) -> dict[str, dict]:
    """Extract E[score] and confidence for each dim from the logprob response.

    The model outputs lines like "prose_execution: 7". For each dim, we find
    the token position of the score digit and compute expected_score from
    the logprob distribution at that position.
    """
    out: dict[str, dict] = {}
    tokens = result.token_logprobs

    # Walk token positions and find score digits that follow dim names
    text_so_far = ""
    for i, tok in enumerate(tokens):
        text_so_far += tok.token
        # Check if any dim's score just appeared
        for dim in dims:
            pattern = f"{dim}:"
            if pattern in text_so_far and dim not in out:
                # The score digit should be at or near this position
                # Check if this token IS the digit
                stripped = tok.token.strip()
                if stripped in [str(d) for d in range(1, 11)]:
                    e_score, confidence = result.expected_score(
                        i, min_val=1, max_val=10,
                    )
                    out[dim] = {
                        "score": round(e_score, 4),
                        "sampled": int(stripped),
                        "confidence": round(confidence, 4),
                    }
                    break
                # Or check next few tokens for the digit
                for j in range(i + 1, min(i + 4, len(tokens))):
                    s = tokens[j].token.strip()
                    if s in [str(d) for d in range(1, 11)]:
                        e_score, confidence = result.expected_score(
                            j, min_val=1, max_val=10,
                        )
                        out[dim] = {
                            "score": round(e_score, 4),
                            "sampled": int(s),
                            "confidence": round(confidence, 4),
                        }
                        break
                break

    # Fallback: if we missed any dims, parse from text
    for dim in dims:
        if dim not in out:
            # Try to parse "dim: N" from the content
            for line in result.content.split("\n"):
                if f"{dim}:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            val = int(parts[-1].strip())
                            out[dim] = {
                                "score": (val - 1) / 9,
                                "sampled": val,
                                "confidence": 0.0,  # no logprob data
                            }
                        except ValueError:
                            pass
                    break

    return out


async def score_chapter_logprob(
    *,
    client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
    max_tokens: int = 200,
    top_logprobs: int = 20,
) -> dict[str, dict]:
    """Score a chapter using logprob-weighted E[score].

    Returns ``{dim: {score, sampled, confidence}}``:
    - score: E[score] in [0, 1], computed from logprob distribution
    - sampled: the integer the model actually emitted (1–10)
    - confidence: 1 - normalized_entropy of the distribution
    """
    use_dims = list(dims or DEFAULT_DIMS)
    result = await client.chat_with_logprobs(
        messages=[
            ChatMessage(role="system", content=JUDGE_SYSTEM_LOGPROB),
            ChatMessage(role="user", content=_build_logprob_user_prompt(
                chapter_text, use_dims,
            )),
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        top_logprobs=top_logprobs,
        thinking=False,
    )
    return _parse_logprob_scores(result, use_dims)


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

PAIRWISE_SYSTEM = (
    "You are judging two chapter excerpts. Read both carefully. "
    "Which is stronger on the specified dimension? "
    "Answer with a single letter: A or B. Nothing else."
)


def _build_pairwise_prompt(
    text_a: str, text_b: str, dim: str,
) -> str:
    rubric = _load_rubric(dim)
    return (
        f"DIMENSION: {dim}\n\n"
        f"{rubric}\n\n"
        f"CHAPTER A:\n<<<\n{text_a}\n>>>\n\n"
        f"CHAPTER B:\n<<<\n{text_b}\n>>>\n\n"
        f"Which chapter is stronger on {dim}? Answer A or B only."
    )


async def compare_chapters(
    *,
    client: InferenceClient,
    text_a: str,
    text_b: str,
    dim: str,
    top_logprobs: int = 20,
) -> dict:
    """Pairwise comparison on one dim. Returns {p_a_wins, p_b_wins, confidence}.

    P(A wins) is extracted from logprobs at the A/B token position.
    Confidence = |P(A) - 0.5| * 2 — distance from toss-up, scaled [0,1].
    """
    result = await client.chat_with_logprobs(
        messages=[
            ChatMessage(role="system", content=PAIRWISE_SYSTEM),
            ChatMessage(role="user", content=_build_pairwise_prompt(
                text_a, text_b, dim,
            )),
        ],
        temperature=0.0,
        max_tokens=4,
        top_logprobs=top_logprobs,
        thinking=False,
    )
    # Find the A/B token
    for i, tok in enumerate(result.token_logprobs):
        if tok.token.strip() in ("A", "B"):
            dist = result.score_token_distribution(i, ["A", "B"])
            p_a = dist.get("A", 0.5)
            p_b = dist.get("B", 0.5)
            confidence = abs(p_a - 0.5) * 2
            return {
                "p_a_wins": round(p_a, 4),
                "p_b_wins": round(p_b, 4),
                "sampled": tok.token.strip(),
                "confidence": round(confidence, 4),
            }
    # Fallback: parse from content
    pick = result.content.strip().upper()
    if pick.startswith("A"):
        return {"p_a_wins": 1.0, "p_b_wins": 0.0, "sampled": "A", "confidence": 0.0}
    elif pick.startswith("B"):
        return {"p_a_wins": 0.0, "p_b_wins": 1.0, "sampled": "B", "confidence": 0.0}
    return {"p_a_wins": 0.5, "p_b_wins": 0.5, "sampled": "?", "confidence": 0.0}


async def compare_chapters_all_dims(
    *,
    client: InferenceClient,
    text_a: str,
    text_b: str,
    dims: list[str] | None = None,
) -> dict[str, dict]:
    """Run pairwise comparison on all dims. Returns {dim: {p_a_wins, ...}}."""
    use_dims = list(dims or DEFAULT_DIMS)
    out: dict[str, dict] = {}
    for dim in use_dims:
        out[dim] = await compare_chapters(
            client=client, text_a=text_a, text_b=text_b, dim=dim,
        )
    return out


# ---------------------------------------------------------------------------
# Legacy structured-output scorer (backwards compatibility)
# ---------------------------------------------------------------------------

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
        "Score the following chapter on each of these dimensions.", "",
        "DIMENSIONS (self-contained rubrics):", "",
    ]
    for d in dims:
        parts.append(f"=== {d} ===")
        parts.append(_load_rubric(d))
        parts.append("")
    parts.append("CHAPTER:\n<<<")
    parts.append(chapter_text)
    parts.append(">>>\n\nReturn JSON only.")
    return "\n".join(parts)


async def score_chapter(
    *,
    client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> dict[str, dict]:
    """Legacy scorer. Returns {dim: {score, rationale}}."""
    use_dims = list(dims or LEGACY_DIMS)
    schema = _build_schema(use_dims)
    raw = await client.chat_structured(
        messages=[
            ChatMessage(role="system", content=JUDGE_SYSTEM),
            ChatMessage(role="user", content=_build_user_prompt(chapter_text, use_dims)),
        ],
        json_schema=schema, schema_name="chapter_scores",
        temperature=temperature, max_tokens=max_tokens,
        thinking=False,
    )
    raw = raw.strip()
    if raw and not raw.startswith("{"):
        i = raw.find("{")
        if i >= 0:
            raw = raw[i:]
    parsed = json.loads(raw)
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
    """Score + persist (legacy path). Uses structured output scorer."""
    if chapter.judge_scores and not dims:
        return chapter.judge_scores
    scores = await score_chapter(
        client=client, chapter_text=chapter.prose, dims=dims,
    )
    world.save_chapter_scores(rollout_id, chapter.chapter_index, scores)
    flat = {dim: payload["score"] for dim, payload in scores.items()}
    chapter.judge_scores = flat
    world.save_rollout_chapter(chapter)
    return scores
