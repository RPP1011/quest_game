"""Typed edit-pass — span-level prose fixes with a fixed taxonomy.

Replaces open-ended revise loops for prose-quality issues. World-rule
violations still go through the full reviser; prose-quality problems
get surgical span replacements with named failure modes.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

EDIT_TYPES = frozenset({
    "cliche",
    "purple_prose",
    "forced_metaphor",
    "unnecessary_exposition",
    "abrupt_transition",
    "bland_dialogue",
    "weak_closure",
    "continuity_break",
    "character_voice_drift",
    "timeline_error",
    "entity_contradiction",
    "repetition",
})

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "stages" / "typed_edit" / "system.j2"
_USER_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "stages" / "typed_edit" / "user.j2"


def apply_edits(prose: str, edits: list[dict]) -> str:
    """Apply span-level edits in reverse position order.

    Skips any edit where original_text doesn't match the actual span
    content (safety against stale offsets).
    """
    sorted_edits = sorted(edits, key=lambda e: e["span_start"], reverse=True)
    result = prose
    for edit in sorted_edits:
        start = edit["span_start"]
        end = edit["span_end"]
        original = edit["original_text"]
        replacement = edit["replacement"]
        actual = result[start : start + len(original)]
        if actual != original:
            continue
        result = result[:start] + replacement + result[end:]
    return result


async def detect_edits(
    client: "InferenceClient", prose: str,
) -> list[dict]:
    """LLM call to detect span-level prose issues."""
    from jinja2 import Template
    from app.runtime.client import ChatMessage

    system = Template(_SYSTEM_PROMPT_PATH.read_text()).render(
        edit_types=sorted(EDIT_TYPES),
    )
    user = Template(_USER_PROMPT_PATH.read_text()).render(prose=prose)

    raw = await client.chat(
        messages=[
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ],
        max_tokens=3000,
        temperature=0.2,
        thinking=False,
    )

    content = raw.strip()
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]

    try:
        parsed = json.loads(content.strip())
        edits = parsed.get("edits", parsed) if isinstance(parsed, dict) else parsed
        valid = []
        for e in edits:
            if (isinstance(e, dict)
                    and "span_start" in e and "span_end" in e
                    and "original_text" in e and "replacement" in e
                    and e.get("edit_type", "") in EDIT_TYPES):
                valid.append(e)
        return valid
    except (json.JSONDecodeError, TypeError):
        return []


def _find_sentence(prose: str, quote: str) -> tuple[int, int, str] | None:
    """Find the sentence in prose that contains the given quote.

    Returns (start, end, sentence_text) or None if not found.
    """
    import re
    idx = prose.lower().find(quote.lower())
    if idx < 0:
        return None
    # Walk backwards to sentence start
    start = idx
    while start > 0 and prose[start - 1] not in ".!?\n":
        start -= 1
    if start > 0 and prose[start] == " ":
        start += 1
    # Walk forward to sentence end
    end = idx + len(quote)
    while end < len(prose) and prose[end] not in ".!?\n":
        end += 1
    if end < len(prose):
        end += 1  # include the punctuation
    return start, end, prose[start:end]


async def detect_metaphor_edits(
    client: "InferenceClient",
    prose: str,
    classification: dict,
    max_per_family: int = 3,
) -> list[dict]:
    """Generate targeted edits for over-budget imagery families.

    For each excess metaphor: finds the containing sentence in Python,
    sends only that sentence to the model for rewriting. The model
    never sees the full chapter during edits.
    """
    import asyncio
    from app.runtime.client import ChatMessage

    families = classification.get("families", {})
    excess: list[dict] = []
    for family_name, data in families.items():
        count = data.get("count", 0)
        if count <= max_per_family:
            continue
        quotes = data.get("quotes", [])
        for quote in quotes[max_per_family:]:
            excess.append({"family": family_name, "quote": quote})

    if not excess:
        return []

    # Find sentences containing each excess quote (Python, no LLM)
    targets: list[dict] = []
    for eq in excess[:12]:
        found = _find_sentence(prose, eq["quote"])
        if found:
            start, end, sentence = found
            targets.append({
                "family": eq["family"],
                "quote": eq["quote"],
                "span_start": start,
                "span_end": end,
                "sentence": sentence,
            })

    if not targets:
        return []

    # Rewrite each sentence individually (parallel, small calls)
    async def _rewrite_one(target: dict) -> dict | None:
        prompt = (
            f'Rewrite this sentence, replacing the "{target["family"]}" '
            f'metaphor with a different imagery family (bodily, architectural, '
            f'textile, spatial, weather, or sensory). Keep the same meaning '
            f'and approximate length. Output ONLY the rewritten sentence.\n\n'
            f'Original: {target["sentence"]}'
        )
        try:
            raw = await client.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=200,
                temperature=0.4,
                thinking=False,
            )
            rewritten = raw.strip().strip('"')
            if len(rewritten) > 10 and rewritten != target["sentence"]:
                return {
                    "span_start": target["span_start"],
                    "span_end": target["span_end"],
                    "original_text": target["sentence"],
                    "replacement": rewritten,
                    "edit_type": "forced_metaphor",
                    "reason": f'{target["family"]} over budget',
                }
        except Exception:
            pass
        return None

    results = await asyncio.gather(*[_rewrite_one(t) for t in targets])
    return [r for r in results if r is not None]


def persist_edits(
    conn, edits: list[dict], *,
    trace_id: str | None = None,
    rollout_id: str | None = None,
    chapter_index: int | None = None,
) -> None:
    """Save applied edits to the typed_edits audit table."""
    for edit in edits:
        conn.execute(
            "INSERT INTO typed_edits "
            "(id, trace_id, rollout_id, chapter_index, edit_type, "
            "original_text, replacement, span_start, span_end, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"te_{uuid.uuid4().hex[:8]}",
                trace_id, rollout_id, chapter_index,
                edit["edit_type"], edit["original_text"],
                edit["replacement"], edit["span_start"], edit["span_end"],
                edit.get("reason", ""),
            ),
        )
    conn.commit()


async def remove_weak_metaphors(
    client: "InferenceClient",
    prose: str,
    classification: dict,
    target_per_1000: float = 8.0,
) -> str:
    """Remove the weakest metaphors to bring density toward target.

    Finds sentences containing the shortest metaphors (in Python),
    sends each sentence individually to the model for literal rewriting.
    """
    import asyncio
    from app.runtime.client import ChatMessage

    total_words = len(prose.split())
    total_fig = classification.get("total_figurative", 0)
    current_per_1000 = total_fig / max(total_words, 1) * 1000

    if current_per_1000 <= target_per_1000:
        return prose

    target_count = int(total_words / 1000 * target_per_1000)
    to_remove = total_fig - target_count
    if to_remove <= 0:
        return prose

    # Collect all quotes sorted shortest first
    all_quotes: list[dict] = []
    for fam, data in classification.get("families", {}).items():
        for q in data.get("quotes", []):
            all_quotes.append({"quote": q, "family": fam, "words": len(q.split())})
    all_quotes.sort(key=lambda x: x["words"])

    # Find sentences for the shortest quotes (Python, no LLM)
    targets: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()
    for item in all_quotes:
        if len(targets) >= min(to_remove, 10):
            break
        found = _find_sentence(prose, item["quote"])
        if found:
            start, end, sentence = found
            if (start, end) not in seen_spans:
                seen_spans.add((start, end))
                targets.append({
                    "quote": item["quote"],
                    "family": item["family"],
                    "span_start": start,
                    "span_end": end,
                    "sentence": sentence,
                })

    if not targets:
        return prose

    # Rewrite each sentence literally (parallel, small calls)
    async def _literalize(target: dict) -> tuple[int, int, str, str] | None:
        prompt = (
            f'Rewrite this sentence literally — remove the figurative language '
            f'and express the same idea in plain, concrete terms. '
            f'Keep the meaning and approximately the same length. '
            f'Output ONLY the rewritten sentence.\n\n'
            f'Original: {target["sentence"]}'
        )
        try:
            raw = await client.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=200,
                temperature=0.3,
                thinking=False,
            )
            rewritten = raw.strip().strip('"')
            if len(rewritten) > 10 and rewritten != target["sentence"]:
                return (target["span_start"], target["span_end"],
                        target["sentence"], rewritten)
        except Exception:
            pass
        return None

    results = await asyncio.gather(*[_literalize(t) for t in targets])
    valid = [r for r in results if r is not None]

    # Apply in reverse order
    result = prose
    for start, end, original, replacement in sorted(valid, key=lambda x: x[0], reverse=True):
        result = result[:start] + replacement + result[end:]
    return result


_CONSOLIDATE_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "stages" / "typed_edit" / "consolidate.j2"

# Imagery families to rotate through when consolidating
_CONSOLIDATE_FAMILIES = [
    "bodily", "architectural", "textile", "spatial",
    "weather", "sensory", "weight_gravity", "water_ocean",
]


async def find_metaphor_clusters(
    classification: dict,
    prose: str,
    max_per_family: int = 3,
    min_cluster_size: int = 3,
) -> list[dict]:
    """Find clusters of short metaphors that should be consolidated.

    A cluster is a group of 3+ figurative quotes from the same family
    within a ~500 character window. Returns clusters with their
    approximate span in the prose.
    """
    clusters: list[dict] = []
    for family, data in classification.get("families", {}).items():
        quotes = data.get("quotes", [])
        if len(quotes) < min_cluster_size:
            continue

        # Find positions of each quote in the prose
        positioned = []
        for q in quotes:
            idx = prose.find(q)
            if idx >= 0:
                positioned.append({"quote": q, "pos": idx, "end": idx + len(q)})
        positioned.sort(key=lambda x: x["pos"])

        # Sliding window: find groups of 3+ within 500 chars
        for i in range(len(positioned)):
            group = [positioned[i]]
            for j in range(i + 1, len(positioned)):
                if positioned[j]["pos"] - positioned[i]["pos"] < 500:
                    group.append(positioned[j])
                else:
                    break
            if len(group) >= min_cluster_size:
                # Extract the passage spanning all quotes in the cluster
                span_start = max(0, group[0]["pos"] - 50)
                span_end = min(len(prose), group[-1]["end"] + 50)
                # Expand to sentence boundaries
                while span_start > 0 and prose[span_start] not in ".!?\n":
                    span_start -= 1
                if span_start > 0:
                    span_start += 2  # skip past the period + space
                while span_end < len(prose) and prose[span_end] not in ".!?\n":
                    span_end += 1
                if span_end < len(prose):
                    span_end += 1  # include the period

                passage = prose[span_start:span_end]
                clusters.append({
                    "family": family,
                    "quotes": [g["quote"] for g in group],
                    "cluster_count": len(group),
                    "span_start": span_start,
                    "span_end": span_end,
                    "passage": passage,
                    "word_count": len(passage.split()),
                })
                break  # one cluster per family for now

    return clusters


async def consolidate_metaphor_clusters(
    client: "InferenceClient",
    prose: str,
    classification: dict,
    min_cluster_size: int = 3,
) -> str:
    """Find clusters of short metaphors and consolidate each into one
    developed metaphor. Returns the modified prose."""
    from jinja2 import Template
    from app.runtime.client import ChatMessage

    clusters = await find_metaphor_clusters(
        classification, prose, min_cluster_size=min_cluster_size,
    )
    if not clusters:
        return prose

    template = Template(_CONSOLIDATE_PROMPT_PATH.read_text())

    # Process clusters in reverse order so span positions stay valid
    clusters.sort(key=lambda c: c["span_start"], reverse=True)
    result = prose
    family_idx = 0

    for cluster in clusters:
        target_family = _CONSOLIDATE_FAMILIES[family_idx % len(_CONSOLIDATE_FAMILIES)]
        family_idx += 1

        prompt = template.render(
            cluster_count=cluster["cluster_count"],
            word_count=cluster["word_count"],
            target_family=target_family,
            passage=cluster["passage"],
        )

        try:
            rewritten = await client.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=1000,
                temperature=0.6,
                thinking=False,
            )
            rewritten = rewritten.strip()
            if len(rewritten) > 20:  # sanity check
                result = (
                    result[:cluster["span_start"]]
                    + rewritten
                    + result[cluster["span_end"]:]
                )
        except Exception:
            continue  # skip this cluster on failure

    return result
