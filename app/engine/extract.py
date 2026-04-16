"""Extract stage: converts LLM-produced world-state diff into a StateDelta."""
from __future__ import annotations
from typing import Any
from app.world.delta import (
    EntityUpdate,
    FSUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
    ValidationIssue,
)
from app.world.schema import Relationship, TimelineEvent

# TODO(G13): the LLM-side assessment of *when* an unconscious motive has
# resolved is still a stub. For now, ``motive_resolutions`` is wired only
# as a persistence path: upstream callers (tests, future LLM emissions)
# can set resolved_at_update explicitly; no heuristic inference here yet.

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entity_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "patch": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "last_referenced_update": {"type": "integer"},
                            "data": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["id", "patch"],
                "additionalProperties": False,
            },
        },
        "new_relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "rel_type": {"type": "string"},
                    "data": {"type": "object"},
                },
                "required": ["source_id", "target_id", "rel_type"],
                "additionalProperties": False,
            },
        },
        "removed_relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "rel_type": {"type": "string"},
                },
                "required": ["source_id", "target_id", "rel_type"],
                "additionalProperties": False,
            },
        },
        "timeline_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "involved_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        },
        "foreshadowing_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "new_status": {
                        "type": "string",
                        "enum": ["planted", "referenced", "paid_off", "abandoned"],
                    },
                    "add_reference": {"type": "integer"},
                },
                "required": ["id", "new_status"],
                "additionalProperties": False,
            },
        },
        "theme_stance_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "new_stance": {
                        "type": "string",
                        "enum": ["exploring", "affirming", "questioning", "subverting"],
                    },
                },
                "required": ["id", "new_stance"],
                "additionalProperties": False,
            },
        },
        "motive_resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "character_id": {"type": "string"},
                    "motive_id": {"type": "string"},
                    "resolved_at_update": {"type": "integer"},
                },
                "required": ["character_id", "motive_id", "resolved_at_update"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "entity_updates",
        "new_relationships",
        "removed_relationships",
        "timeline_events",
        "foreshadowing_updates",
    ],
    "additionalProperties": False,
}


def build_delta(
    extracted: dict[str, Any],
    update_number: int,
    known_ids: set[str] | None = None,
    *,
    world: Any | None = None,
) -> tuple[StateDelta, list[ValidationIssue]]:
    """Convert LLM-produced extracted dict into a StateDelta.

    Returns (delta, validation_issues).  Unknown entity ids are filtered out
    and recorded as validation issues rather than raising.  The caller must
    check the issues list and decide whether to apply the delta.
    """
    issues: list[ValidationIssue] = []
    valid = known_ids  # None means no filtering

    from app.world.schema import EntityStatus
    valid_statuses = {s.value for s in EntityStatus}
    entity_updates: list[EntityUpdate] = []
    for item in extracted.get("entity_updates", []):
        eid = item.get("id", "")
        if valid is not None and eid not in valid:
            issues.append(ValidationIssue(
                severity="error",
                message=f"entity_update references unknown id: {eid}",
                subject=eid,
            ))
            continue
        patch = dict(item.get("patch", {}))
        # Drop unknown enum values (e.g. LLM emitting status="revealed")
        # so a free-form output doesn't crash later Entity validation.
        if "status" in patch and patch["status"] not in valid_statuses:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"dropped invalid entity status {patch['status']!r} for {eid}",
                subject=eid,
            ))
            patch.pop("status", None)
        entity_updates.append(EntityUpdate(id=eid, patch=patch))

    relationship_changes: list[RelChange] = []
    for item in extracted.get("new_relationships", []):
        src = item.get("source_id", "")
        tgt = item.get("target_id", "")
        bad = []
        if valid is not None and src not in valid:
            bad.append(src)
        if valid is not None and tgt not in valid:
            bad.append(tgt)
        if bad:
            for b in bad:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"new_relationship endpoint references unknown id: {b}",
                    subject=b,
                ))
            continue
        relationship_changes.append(RelChange(
            action="add",
            relationship=Relationship(
                source_id=src,
                target_id=tgt,
                rel_type=item.get("rel_type", ""),
                data=item.get("data", {}),
                established_at_update=update_number,
            ),
        ))

    for item in extracted.get("removed_relationships", []):
        src = item.get("source_id", "")
        tgt = item.get("target_id", "")
        relationship_changes.append(RelChange(
            action="remove",
            relationship=Relationship(
                source_id=src,
                target_id=tgt,
                rel_type=item.get("rel_type", ""),
            ),
        ))

    timeline_events: list[TimelineEventOp] = []
    for idx, item in enumerate(extracted.get("timeline_events", [])):
        timeline_events.append(TimelineEventOp(
            event=TimelineEvent(
                update_number=update_number,
                event_index=idx,
                description=item.get("description", ""),
                involved_entities=item.get("involved_entities", []),
            ),
        ))

    _VALID_FS_STATUSES = {"planted", "referenced", "paid_off", "abandoned"}
    # Build a set of known hook ids if a world is wired; used to drop
    # hallucinated hook references (e.g. model emitting a location id as
    # if it were a hook) as warnings instead of letting them blow the
    # whole extract by failing world-side validation.
    known_hook_ids: set[str] | None = None
    if world is not None:
        try:
            rows = world._conn.execute("SELECT id FROM foreshadowing").fetchall()
            known_hook_ids = {r[0] for r in rows}
        except Exception:
            known_hook_ids = None
    foreshadowing_updates: list[FSUpdate] = []
    for item in extracted.get("foreshadowing_updates", []):
        status = item.get("new_status", "referenced")
        hook_id = item.get("id", "")
        if status not in _VALID_FS_STATUSES:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"extract: dropped foreshadowing update with invalid status {status!r}",
                subject=hook_id,
            ))
            continue
        if known_hook_ids is not None and hook_id not in known_hook_ids:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"extract: dropped foreshadowing update for unknown hook {hook_id!r}",
                subject=hook_id,
            ))
            continue
        foreshadowing_updates.append(FSUpdate(
            id=hook_id,
            new_status=status,
            add_reference=item.get("add_reference"),
        ))

    # ---- G13 motive resolutions ----
    # Group by character_id; for each, read current motives list from world
    # (if provided), mark matching entries resolved, and append an
    # EntityUpdate that overwrites the unconscious_motives list on data.
    resolutions_by_char: dict[str, list[dict[str, Any]]] = {}
    for item in extracted.get("motive_resolutions", []) or []:
        cid = item.get("character_id", "")
        if not cid:
            continue
        if valid is not None and cid not in valid:
            issues.append(ValidationIssue(
                severity="error",
                message=f"motive_resolution references unknown character: {cid}",
                subject=cid,
            ))
            continue
        resolutions_by_char.setdefault(cid, []).append(item)

    if resolutions_by_char and world is not None:
        # Local import to avoid circular dependency
        from app.planning.motives import apply_motive_resolutions
        for cid, items in resolutions_by_char.items():
            try:
                entity = world.get_entity(cid)
            except Exception:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=f"motive_resolution: entity {cid} not found",
                    subject=cid,
                ))
                continue
            new_data = apply_motive_resolutions(entity.data, items)
            if new_data is entity.data:
                continue  # nothing changed
            # Shallow-merge semantics on data mean we only need to pass the
            # updated unconscious_motives key.
            entity_updates.append(EntityUpdate(
                id=cid,
                patch={"data": {"unconscious_motives": new_data["unconscious_motives"]}},
            ))

    delta = StateDelta(
        entity_updates=entity_updates,
        relationship_changes=relationship_changes,
        timeline_events=timeline_events,
        foreshadowing_updates=foreshadowing_updates,
    )
    return delta, issues
