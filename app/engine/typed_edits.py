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


async def detect_metaphor_edits(
    client: "InferenceClient",
    prose: str,
    classification: dict,
    max_per_family: int = 3,
) -> list[dict]:
    """Generate targeted edits for over-budget imagery families.

    Takes the LLM classifier output and asks the model to replace
    specific excess quotes with non-gambling alternatives.
    """
    from app.runtime.client import ChatMessage

    families = classification.get("families", {})
    excess_quotes: list[dict] = []
    for family_name, data in families.items():
        count = data.get("count", 0)
        if count <= max_per_family:
            continue
        quotes = data.get("quotes", [])
        # Keep first max_per_family, replace the rest
        for quote in quotes[max_per_family:]:
            excess_quotes.append({"family": family_name, "quote": quote})

    if not excess_quotes:
        return []

    # Build a targeted prompt
    lines = []
    for eq in excess_quotes[:12]:  # cap at 12 edits
        lines.append(f'- FIND: "{eq["quote"]}" (family: {eq["family"]})')

    prompt = (
        "Replace each of the following metaphors/figurative phrases in the text. "
        "Each replacement must use a DIFFERENT imagery family — bodily sensation, "
        "architecture, textile, spatial, weather, or sensory. Keep the same meaning "
        "and approximate length. Do NOT use the same family as the original.\n\n"
        + "\n".join(lines)
        + "\n\nFor each, find the phrase in the text and output a JSON edit with "
        "span_start, span_end, original_text (the exact text at that span in the "
        "chapter), edit_type (always \"forced_metaphor\"), reason, and replacement.\n\n"
        "Output JSON only:\n"
        '{"edits": [{"span_start": N, "span_end": N, "original_text": "...", '
        '"edit_type": "forced_metaphor", "reason": "...", "replacement": "..."}]}\n\n'
        f"TEXT:\n{prose}"
    )

    raw = await client.chat(
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=4000,
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
                    and "original_text" in e and "replacement" in e
                    and e.get("edit_type") == "forced_metaphor"):
                # Find the actual span in the prose
                orig = e["original_text"]
                idx = prose.find(orig)
                if idx >= 0:
                    e["span_start"] = idx
                    e["span_end"] = idx + len(orig)
                    valid.append(e)
        return valid
    except (json.JSONDecodeError, TypeError):
        return []


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
