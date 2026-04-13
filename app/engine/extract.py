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
) -> tuple[StateDelta, list[ValidationIssue]]:
    """Convert LLM-produced extracted dict into a StateDelta.

    Returns (delta, validation_issues).  Unknown entity ids are filtered out
    and recorded as validation issues rather than raising.  The caller must
    check the issues list and decide whether to apply the delta.
    """
    issues: list[ValidationIssue] = []
    valid = known_ids  # None means no filtering

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
        entity_updates.append(EntityUpdate(id=eid, patch=item.get("patch", {})))

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
    foreshadowing_updates: list[FSUpdate] = []
    for item in extracted.get("foreshadowing_updates", []):
        status = item.get("new_status", "referenced")
        if status not in _VALID_FS_STATUSES:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"extract: dropped foreshadowing update with invalid status {status!r}",
                subject=item.get("id"),
            ))
            continue
        foreshadowing_updates.append(FSUpdate(
            id=item.get("id", ""),
            new_status=status,
            add_reference=item.get("add_reference"),
        ))

    delta = StateDelta(
        entity_updates=entity_updates,
        relationship_changes=relationship_changes,
        timeline_events=timeline_events,
        foreshadowing_updates=foreshadowing_updates,
    )
    return delta, issues
