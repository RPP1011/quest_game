"""Chapter-scale scoring via dual-rating with logprob E[score].

All scoring — both absolute (single chapter) and comparative (two
chapters) — uses the same mechanism: the model writes a brief analysis
then emits an integer 1–10 rating. E[score] and confidence are
extracted from the logprob distribution at the rating token.

For comparisons, BOTH chapters are rated in one call (dual rating).
The comparison is derived from the two ratings — no forced A/B pick,
which eliminates the catastrophic position bias (76–96 pts) that
direct pairwise suffers from. Residual position bias on dual rating
is 0–3.4 pts; both-directions averaging reduces it further.

Collapsed dims (PCA-validated):
- prose_execution: tension + emotion + voice + theme + interiority
- subtext: kept as own dim (rubric decouples from prose quality)
- hook_quality: chapter ending hook (genuinely independent)

Dropped: update_self_containment (rewards recap in serial fiction),
emotional_trajectory/interiority_depth redundancy (r=0.84).
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
    "metaphor_variety",
)

DEFAULT_DIMS = COLLAPSED_DIMS


def _load_rubric(dim: str, **template_vars) -> str:
    """Load a rubric template. Supports simple Jinja variable injection
    for cross-chapter context (e.g., prior_chapter_families)."""
    path = RUBRICS_DIR / f"{dim}.j2"
    if not path.is_file():
        raise FileNotFoundError(f"missing rubric: {path}")
    text = path.read_text()
    if template_vars:
        from jinja2 import Template
        text = Template(text).render(**template_vars)
    return text


def _find_score_at_marker(
    result: ChatWithLogprobs, marker: str, start_from: int = 0,
) -> dict | None:
    """Find the integer score token after ``marker`` in the logprob stream.

    Returns ``{score, sampled, confidence}`` or None if not found.
    """
    text_so_far = "".join(
        t.token for t in result.token_logprobs[:start_from]
    )
    for i in range(start_from, len(result.token_logprobs)):
        text_so_far += result.token_logprobs[i].token
        if marker not in text_so_far:
            continue
        # Found marker — look for the digit in this or next few tokens
        for j in range(i, min(i + 5, len(result.token_logprobs))):
            s = result.token_logprobs[j].token.strip()
            if s in [str(d) for d in range(1, 11)]:
                e_score, confidence = result.expected_score(
                    j, min_val=1, max_val=10,
                )
                return {
                    "score": round(e_score, 4),
                    "sampled": int(s),
                    "confidence": round(confidence, 4),
                }
        return None
    return None


def _fallback_parse_score(content: str, marker: str) -> dict | None:
    """Parse 'marker: N' from text when logprobs aren't available."""
    for line in content.split("\n"):
        if marker in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    val = int(parts[-1].strip())
                    return {
                        "score": round((val - 1) / 9, 4),
                        "sampled": val,
                        "confidence": 0.0,
                    }
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Single-chapter scoring (absolute)
# ---------------------------------------------------------------------------

SINGLE_SYSTEM = (
    "You are a literary judge. Read the chapter, write a 1-sentence "
    "observation per dimension, then rate it (1-10 integer).\n\n"
    "Format:\n"
    "prose_execution observation: [your observation]\n"
    "prose_execution score: [integer 1-10]\n"
    "subtext observation: [your observation]\n"
    "subtext score: [integer 1-10]\n"
    "hook_quality observation: [your observation]\n"
    "hook_quality score: [integer 1-10]\n\n"
    "Be precise. Use the full 1-10 range."
)


def _build_single_prompt(chapter_text: str, dims: list[str]) -> str:
    parts = ["Score the following chapter.\n"]
    for d in dims:
        parts.append(f"=== {d} ===")
        parts.append(_load_rubric(d))
        parts.append("")
    parts.append("CHAPTER:\n<<<")
    parts.append(chapter_text)
    parts.append(">>>\n\nAnalyze and rate each dimension now.")
    return "\n".join(parts)


async def score_chapter(
    *,
    client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
    max_tokens: int = 400,
    top_logprobs: int = 20,
) -> dict[str, dict]:
    """Score a single chapter. Returns ``{dim: {score, sampled, confidence}}``.

    Uses analysis-then-rate format with logprob E[score] extraction.
    """
    use_dims = list(dims or DEFAULT_DIMS)
    result = await client.chat_with_logprobs(
        messages=[
            ChatMessage(role="system", content=SINGLE_SYSTEM),
            ChatMessage(role="user", content=_build_single_prompt(
                chapter_text, use_dims,
            )),
        ],
        temperature=0.3, max_tokens=max_tokens,
        top_logprobs=top_logprobs, thinking=False,
    )
    out: dict[str, dict] = {}
    for dim in use_dims:
        marker = f"{dim} score:"
        found = _find_score_at_marker(result, marker)
        if found is None:
            found = _fallback_parse_score(result.content, f"{dim} score")
        if found is None:
            found = _fallback_parse_score(result.content, dim)
        if found is not None:
            out[dim] = found
    return out


async def score_chapter_independent(
    *, client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
    max_tokens: int = 400,
    top_logprobs: int = 20,
) -> dict[str, dict]:
    """Score a chapter with one LLM call per dimension (no anchoring).

    Each dim gets its own prompt containing only that dim's rubric.
    All dims run in parallel via asyncio.gather.
    """
    import asyncio
    use_dims = dims or list(COLLAPSED_DIMS)

    async def _score_one(dim: str) -> tuple[str, dict]:
        return dim, await _score_single_dim(
            client=client, chapter_text=chapter_text,
            dim=dim, max_tokens=max_tokens, top_logprobs=top_logprobs,
        )

    results = await asyncio.gather(*[_score_one(d) for d in use_dims])
    return dict(results)


async def _score_single_dim(
    *, client: InferenceClient, chapter_text: str,
    dim: str, max_tokens: int = 400, top_logprobs: int = 20,
) -> dict:
    """Score a single dimension in isolation."""
    import re

    # Truncate long chapters to avoid blowing context window.
    # Keep first 1500 words + last 500 words for representative sample.
    words = chapter_text.split()
    if len(words) > 2500:
        first = " ".join(words[:1500])
        last = " ".join(words[-500:])
        chapter_text = first + "\n\n[...]\n\n" + last

    rubric = _load_rubric(dim)
    prompt = (
        f"{rubric}\n\n"
        f"CHAPTER:\n{chapter_text}\n\n"
        f"Write a 1-sentence observation, then rate this chapter on "
        f"{dim.replace('_', ' ')} (1-10).\n"
        f"Format: {dim} score: N"
    )

    result = await client.chat_with_logprobs(
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=max_tokens,
        temperature=0.3,
        top_logprobs=top_logprobs,
    )

    # Find the score token
    content = result.content or ""
    marker = f"{dim} score:"
    marker_pos = content.lower().find(marker.lower())

    if marker_pos >= 0 and result.token_logprobs:
        char_count = 0
        for i, tlp in enumerate(result.token_logprobs):
            char_count += len(tlp.token)
            if char_count >= marker_pos + len(marker):
                for j in range(i + 1, min(i + 4, len(result.token_logprobs))):
                    tok = result.token_logprobs[j].token.strip()
                    if tok.isdigit() and 1 <= int(tok) <= 10:
                        score_val = result.expected_score(j)
                        return {
                            "score": score_val[0],
                            "sampled": int(tok),
                            "confidence": score_val[1],
                        }
                break

    # Fallback: text parse
    match = re.search(rf"{dim}\s*score:\s*(\d+)", content, re.IGNORECASE)
    if match:
        sampled = int(match.group(1))
        return {"score": (sampled - 1) / 9, "sampled": sampled, "confidence": 0.0}

    return {"score": 0.5, "sampled": 5, "confidence": 0.0}


# ---------------------------------------------------------------------------
# Dual-rating comparison (two chapters rated in one call)
# ---------------------------------------------------------------------------

DUAL_SYSTEM = (
    "You are a literary judge. You will read two chapter excerpts and "
    "rate EACH on the specified dimension (1-10 integer). First write "
    "a brief 1-sentence analysis of each, then give your rating.\n\n"
    "Format:\n"
    "Chapter A analysis: [1 sentence]\n"
    "Chapter A score: [integer 1-10]\n"
    "Chapter B analysis: [1 sentence]\n"
    "Chapter B score: [integer 1-10]\n\n"
    "Be precise. Use the full 1-10 range. The chapters may differ in "
    "quality — rate each on its own merits."
)


def _build_dual_prompt(text_a: str, text_b: str, dim: str) -> str:
    rubric = _load_rubric(dim)
    return (
        f"DIMENSION: {dim}\n\n{rubric}\n\n"
        f"CHAPTER A:\n<<<\n{text_a}\n>>>\n\n"
        f"CHAPTER B:\n<<<\n{text_b}\n>>>\n\n"
        f"Analyze and rate each chapter on {dim} (1-10)."
    )


async def compare_chapters(
    *,
    client: InferenceClient,
    text_a: str,
    text_b: str,
    dim: str,
    max_tokens: int = 400,
    top_logprobs: int = 20,
) -> dict:
    """Dual-rating comparison on one dim.

    Returns ``{a: {score, sampled, confidence}, b: {score, ...},
    delta: float, analysis: str}``.

    ``delta`` = a.score - b.score (positive means A is better).
    """
    result = await client.chat_with_logprobs(
        messages=[
            ChatMessage(role="system", content=DUAL_SYSTEM),
            ChatMessage(role="user", content=_build_dual_prompt(
                text_a, text_b, dim,
            )),
        ],
        temperature=0.0, max_tokens=max_tokens,
        top_logprobs=top_logprobs, thinking=False,
    )
    a_score = _find_score_at_marker(result, "Chapter A score:")
    b_score = _find_score_at_marker(result, "Chapter B score:")
    if a_score is None:
        a_score = _fallback_parse_score(result.content, "Chapter A score") or {
            "score": 0.5, "sampled": 5, "confidence": 0.0,
        }
    if b_score is None:
        b_score = _fallback_parse_score(result.content, "Chapter B score") or {
            "score": 0.5, "sampled": 5, "confidence": 0.0,
        }
    return {
        "a": a_score,
        "b": b_score,
        "delta": round(a_score["score"] - b_score["score"], 4),
        "analysis": result.content,
    }


async def compare_chapters_corrected(
    *,
    client: InferenceClient,
    text_a: str,
    text_b: str,
    dim: str,
    **kwargs,
) -> dict:
    """Bias-corrected dual-rating: runs both directions and averages.

    Returns ``{delta_corrected, a_score, b_score, position_bias,
    ab_result, ba_result}``.

    ``delta_corrected`` > 0 means text_a is better after correction.
    """
    r_ab = await compare_chapters(
        client=client, text_a=text_a, text_b=text_b, dim=dim, **kwargs,
    )
    r_ba = await compare_chapters(
        client=client, text_a=text_b, text_b=text_a, dim=dim, **kwargs,
    )
    # In AB: delta = a_score - b_score = textA - textB
    # In BA: delta = a_score - b_score = textB - textA → flip sign for textA - textB
    delta_ab = r_ab["delta"]
    delta_ba = -r_ba["delta"]
    delta_corrected = (delta_ab + delta_ba) / 2
    position_bias = delta_ab - delta_ba
    # Average scores for A (text_a) across both positions
    a_score = (r_ab["a"]["score"] + r_ba["b"]["score"]) / 2
    b_score = (r_ab["b"]["score"] + r_ba["a"]["score"]) / 2
    return {
        "delta_corrected": round(delta_corrected, 4),
        "a_score": round(a_score, 4),
        "b_score": round(b_score, 4),
        "position_bias": round(position_bias, 4),
        "ab_result": r_ab,
        "ba_result": r_ba,
    }


async def compare_chapters_all_dims(
    *,
    client: InferenceClient,
    text_a: str,
    text_b: str,
    dims: list[str] | None = None,
    corrected: bool = True,
) -> dict[str, dict]:
    """Run dual-rating comparison on all dims.

    If ``corrected=True`` (default), runs both directions per dim for
    bias correction (2× cost). If False, single-direction only.
    """
    use_dims = list(dims or DEFAULT_DIMS)
    out: dict[str, dict] = {}
    for dim in use_dims:
        if corrected:
            out[dim] = await compare_chapters_corrected(
                client=client, text_a=text_a, text_b=text_b, dim=dim,
            )
        else:
            out[dim] = await compare_chapters(
                client=client, text_a=text_a, text_b=text_b, dim=dim,
            )
    return out


# ---------------------------------------------------------------------------
# Cross-chapter coherence (scored on consecutive pairs)
# ---------------------------------------------------------------------------

COHERENCE_SYSTEM = (
    "You are a literary judge evaluating how well two consecutive chapters "
    "connect. Read Chapter A, then Chapter B. Write a 1-sentence observation "
    "about the transition quality, then rate the coherence (1-10 integer).\n\n"
    "Format:\n"
    "Observation: [1 sentence about the A→B transition]\n"
    "Coherence score: [integer 1-10]"
)


async def score_chapter_pair_coherence(
    *,
    client: InferenceClient,
    chapter_a_prose: str,
    chapter_b_prose: str,
    max_tokens: int = 200,
    top_logprobs: int = 20,
) -> dict:
    """Score the coherence between two consecutive chapters.

    Returns ``{score, sampled, confidence, observation}``.
    """
    rubric = _load_rubric("cross_chapter_coherence")
    # Truncate to keep prompt manageable — last 2k of A, first 2k of B
    # captures the transition zone
    a_tail = chapter_a_prose[-2000:] if len(chapter_a_prose) > 2000 else chapter_a_prose
    b_head = chapter_b_prose[:2000] if len(chapter_b_prose) > 2000 else chapter_b_prose

    user_prompt = (
        f"{rubric}\n\n"
        f"CHAPTER A (ending):\n<<<\n{a_tail}\n>>>\n\n"
        f"CHAPTER B (opening):\n<<<\n{b_head}\n>>>\n\n"
        f"Observe the transition, then rate coherence (1-10)."
    )
    result = await client.chat_with_logprobs(
        messages=[
            ChatMessage(role="system", content=COHERENCE_SYSTEM),
            ChatMessage(role="user", content=user_prompt),
        ],
        temperature=0.3, max_tokens=max_tokens,
        top_logprobs=top_logprobs, thinking=False,
    )
    found = _find_score_at_marker(result, "Coherence score:")
    if found is None:
        found = _fallback_parse_score(result.content, "Coherence score") or {
            "score": 0.5, "sampled": 5, "confidence": 0.0,
        }
    # Extract observation
    obs = ""
    for line in result.content.split("\n"):
        if line.strip().startswith("Observation:"):
            obs = line.split(":", 1)[-1].strip()
            break
    found["observation"] = obs
    return found


async def score_rollout_coherence(
    *,
    client: InferenceClient,
    chapters: list,
) -> list[dict]:
    """Score coherence for all consecutive pairs in a rollout.

    Returns a list of ``{chapter_a, chapter_b, score, sampled,
    confidence, observation}`` — one per adjacent pair.
    """
    results: list[dict] = []
    sorted_chs = sorted(chapters, key=lambda c: c.chapter_index)
    for i in range(len(sorted_chs) - 1):
        a = sorted_chs[i]
        b = sorted_chs[i + 1]
        try:
            r = await score_chapter_pair_coherence(
                client=client,
                chapter_a_prose=a.prose,
                chapter_b_prose=b.prose,
            )
            r["chapter_a"] = a.chapter_index
            r["chapter_b"] = b.chapter_index
            results.append(r)
        except Exception as e:
            results.append({
                "chapter_a": a.chapter_index,
                "chapter_b": b.chapter_index,
                "score": 0.0, "sampled": 0, "confidence": 0.0,
                "observation": f"scoring failed: {e}",
            })
    return results


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

async def score_and_persist_chapter(
    *,
    client: InferenceClient,
    world: WorldStateManager,
    rollout_id: str,
    chapter: RolloutChapter,
    dims: list[str] | None = None,
) -> dict[str, dict]:
    """Score a chapter and persist to KB + chapter row.

    Uses the single-chapter scorer (analysis + rating with logprobs).
    Idempotent: skips if chapter already has judge_scores.
    """
    if chapter.judge_scores and not dims:
        return chapter.judge_scores
    scores = await score_chapter_independent(
        client=client, chapter_text=chapter.prose, dims=dims,
    )
    # Persist per-dim rows for aggregation (adapting to the KB schema
    # which expects {dim: {score, rationale}})
    kb_scores = {
        dim: {"score": payload["score"], "rationale": f"sampled={payload['sampled']} conf={payload['confidence']:.3f}"}
        for dim, payload in scores.items()
    }
    world.save_chapter_scores(rollout_id, chapter.chapter_index, kb_scores)
    flat = {dim: payload["score"] for dim, payload in scores.items()}
    chapter.judge_scores = flat
    world.save_rollout_chapter(chapter)
    return scores
