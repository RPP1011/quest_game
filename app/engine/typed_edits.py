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
