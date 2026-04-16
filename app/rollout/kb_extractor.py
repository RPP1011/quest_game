"""Knowledge-base extraction for rollout chapters (Phase 4).

Given a rollout chapter's trace + prose, extract:
- Hook payoffs: from the extract stage's foreshadowing_updates rows
- Entity introductions: DORMANT→ACTIVE entity_updates from extract;
  also entity ids whose name appears in the prose for mention tracking
- Thread advances: from the dramatic plan's thread_advances field

Persists per-rollout, per-chapter rows into kb_hook_payoffs and
kb_entity_usage. The mention-chapters list is the union of chapters
in which the entity's display name appears in prose.

Best-effort: silent skip on missing/malformed trace data; rollouts
should never fail because extraction failed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.world.schema import Entity, EntityStatus
from app.world.state_manager import WorldStateManager


def _load_trace(trace_path: Path) -> dict | None:
    if not trace_path.is_file():
        return None
    try:
        return json.loads(trace_path.read_text())
    except Exception:
        return None


def _get_stage_output(trace: dict, stage_name: str) -> Any:
    """Return the parsed_output of the LAST stage with this name (the
    final write of a multi-pass stage like check or revise)."""
    last = None
    for s in trace.get("stages", []) or []:
        if s.get("stage_name") == stage_name:
            last = s
    return (last or {}).get("parsed_output")


def _all_stage_outputs(trace: dict, stage_name: str) -> list[Any]:
    return [
        s.get("parsed_output")
        for s in trace.get("stages", []) or []
        if s.get("stage_name") == stage_name
    ]


def extract_hook_events(trace: dict) -> list[dict]:
    """Return [{'hook_id': str, 'new_status': str}] from the extract stage."""
    out = _get_stage_output(trace, "extract") or {}
    if not isinstance(out, dict):
        return []
    rows: list[dict] = []
    for fs in out.get("foreshadowing_updates", []) or []:
        if not isinstance(fs, dict):
            continue
        hid = fs.get("id")
        ns = fs.get("new_status")
        if hid and ns:
            rows.append({"hook_id": hid, "new_status": ns})
    return rows


def extract_entity_introductions(trace: dict) -> list[str]:
    """Entities promoted to ACTIVE this chapter. Pulled from extract's
    entity_updates with patch.status='active'."""
    out = _get_stage_output(trace, "extract") or {}
    if not isinstance(out, dict):
        return []
    intro: list[str] = []
    for u in out.get("entity_updates", []) or []:
        if not isinstance(u, dict):
            continue
        eid = u.get("id")
        patch = u.get("patch") or {}
        if eid and patch.get("status") == "active":
            intro.append(eid)
    return intro


def extract_thread_advances(trace: dict) -> dict[str, str]:
    """Return {thread_id: target_state} from the dramatic plan's
    thread_advances field (last dramatic stage)."""
    out = _get_stage_output(trace, "dramatic") or {}
    if not isinstance(out, dict):
        return {}
    advances: dict[str, str] = {}
    for ta in out.get("thread_advances", []) or []:
        if not isinstance(ta, dict):
            continue
        tid = ta.get("thread_id") or ta.get("id")
        target = (
            ta.get("target_arc_position")
            or ta.get("new_arc_position")
            or ta.get("target")
            or ""
        )
        if tid:
            advances[tid] = str(target)
    return advances


def find_entity_mentions(prose: str, entities: list[Entity]) -> list[str]:
    """Return ids of entities whose display name (full token, case-insensitive)
    appears anywhere in the prose. Conservative: requires a word-boundary
    match on the *full* name to avoid false positives ("Lan" matching "Lanier").
    """
    import re
    out: list[str] = []
    text = prose or ""
    for e in entities:
        name = (e.name or "").strip()
        if not name:
            continue
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            out.append(e.id)
    return out


def persist_chapter_kb(
    *,
    world: WorldStateManager,
    quest_id: str,
    rollout_id: str,
    chapter_index: int,
    prose: str,
    trace: dict | None,
    all_entities: list[Entity] | None = None,
) -> dict:
    """Run extraction for one chapter and write the KB rows.

    Returns a summary ``{hooks_planted, hooks_paid_off, entities_introduced,
    entities_mentioned, thread_advances}`` for the caller to log/display.
    Resilient to missing trace data.
    """
    summary: dict[str, Any] = {
        "hooks_planted": [],
        "hooks_paid_off": [],
        "entities_introduced": [],
        "entities_mentioned": [],
        "thread_advances": {},
    }
    if trace is None:
        trace = {}

    # Hooks
    for ev in extract_hook_events(trace):
        hid = ev["hook_id"]
        ns = ev["new_status"]
        if ns == "planted":
            summary["hooks_planted"].append(hid)
            world.save_hook_payoff(
                quest_id=quest_id, rollout_id=rollout_id, hook_id=hid,
                planted_at_chapter=chapter_index,
            )
        elif ns == "paid_off":
            summary["hooks_paid_off"].append(hid)
            world.save_hook_payoff(
                quest_id=quest_id, rollout_id=rollout_id, hook_id=hid,
                paid_off_at_chapter=chapter_index,
            )

    # Entity introductions (DORMANT → ACTIVE this chapter)
    introduced = extract_entity_introductions(trace)
    summary["entities_introduced"] = introduced
    for eid in introduced:
        # Merge with existing mentions list for this entity in this rollout
        existing = next(
            (r for r in world.list_entity_usage(quest_id)
             if r["rollout_id"] == rollout_id and r["entity_id"] == eid),
            None,
        )
        mentions = (existing or {}).get("mention_chapters", []) or []
        if chapter_index not in mentions:
            mentions = sorted(set(mentions + [chapter_index]))
        world.save_entity_usage(
            quest_id=quest_id, rollout_id=rollout_id, entity_id=eid,
            introduced_at_chapter=chapter_index,
            mention_chapters=mentions,
        )

    # Entity mentions in prose (for screen-time aggregation)
    if all_entities:
        mentioned = find_entity_mentions(prose, all_entities)
        summary["entities_mentioned"] = mentioned
        for eid in mentioned:
            existing = next(
                (r for r in world.list_entity_usage(quest_id)
                 if r["rollout_id"] == rollout_id and r["entity_id"] == eid),
                None,
            )
            mentions = (existing or {}).get("mention_chapters", []) or []
            if chapter_index not in mentions:
                mentions = sorted(set(mentions + [chapter_index]))
            world.save_entity_usage(
                quest_id=quest_id, rollout_id=rollout_id, entity_id=eid,
                mention_chapters=mentions,
            )

    # Thread advances
    summary["thread_advances"] = extract_thread_advances(trace)

    return summary
