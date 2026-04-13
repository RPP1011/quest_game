"""Tests for app.engine.extract — build_delta shape and validation."""
from __future__ import annotations
import pytest
from app.engine.extract import build_delta, EXTRACT_SCHEMA
from app.world.delta import StateDelta, RelChange, EntityUpdate, TimelineEventOp, FSUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY: dict = {
    "entity_updates": [],
    "new_relationships": [],
    "removed_relationships": [],
    "timeline_events": [],
    "foreshadowing_updates": [],
}


def _merge(**kw) -> dict:
    return {**_EMPTY, **kw}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_delta_empty_produces_empty_state_delta():
    delta, issues = build_delta(_EMPTY, update_number=5)
    assert isinstance(delta, StateDelta)
    assert delta.entity_updates == []
    assert delta.relationship_changes == []
    assert delta.timeline_events == []
    assert delta.foreshadowing_updates == []
    assert issues == []


def test_build_delta_entity_update_shape():
    extracted = _merge(entity_updates=[{"id": "alice", "patch": {"status": "dormant"}}])
    delta, issues = build_delta(extracted, update_number=3, known_ids={"alice"})
    assert issues == []
    assert len(delta.entity_updates) == 1
    eu: EntityUpdate = delta.entity_updates[0]
    assert eu.id == "alice"
    assert eu.patch == {"status": "dormant"}


def test_unknown_entity_id_filtered_to_validation_error():
    extracted = _merge(entity_updates=[{"id": "ghost", "patch": {"status": "active"}}])
    delta, issues = build_delta(extracted, update_number=3, known_ids={"alice"})
    # ghost is unknown → filtered out, recorded as error
    assert delta.entity_updates == []
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "ghost" in issues[0].message


def test_add_relationship_shape():
    extracted = _merge(new_relationships=[{
        "source_id": "alice", "target_id": "bob", "rel_type": "ally",
    }])
    delta, issues = build_delta(extracted, update_number=4, known_ids={"alice", "bob"})
    assert issues == []
    assert len(delta.relationship_changes) == 1
    rc: RelChange = delta.relationship_changes[0]
    assert rc.action == "add"
    assert rc.relationship.source_id == "alice"
    assert rc.relationship.target_id == "bob"
    assert rc.relationship.rel_type == "ally"
    assert rc.relationship.established_at_update == 4


def test_remove_relationship_shape():
    extracted = _merge(removed_relationships=[{
        "source_id": "alice", "target_id": "bob", "rel_type": "enemy",
    }])
    delta, issues = build_delta(extracted, update_number=4, known_ids={"alice", "bob"})
    assert issues == []
    assert len(delta.relationship_changes) == 1
    rc: RelChange = delta.relationship_changes[0]
    assert rc.action == "remove"
    assert rc.relationship.source_id == "alice"
    assert rc.relationship.target_id == "bob"
    assert rc.relationship.rel_type == "enemy"


def test_timeline_events_ordered_by_position():
    extracted = _merge(timeline_events=[
        {"description": "First event", "involved_entities": ["alice"]},
        {"description": "Second event", "involved_entities": ["bob"]},
        {"description": "Third event"},
    ])
    delta, issues = build_delta(extracted, update_number=7)
    assert issues == []
    assert len(delta.timeline_events) == 3
    for idx, op in enumerate(delta.timeline_events):
        assert op.event.event_index == idx, f"Expected event_index {idx}, got {op.event.event_index}"
        assert op.event.update_number == 7


def test_foreshadowing_update_shape():
    extracted = _merge(foreshadowing_updates=[{
        "id": "hook:1", "new_status": "paid_off", "add_reference": 5,
    }])
    delta, issues = build_delta(extracted, update_number=5)
    assert issues == []
    assert len(delta.foreshadowing_updates) == 1
    fu: FSUpdate = delta.foreshadowing_updates[0]
    assert fu.id == "hook:1"
    assert fu.new_status == "paid_off"
    assert fu.add_reference == 5


def test_new_relationship_unknown_endpoint_filtered():
    extracted = _merge(new_relationships=[{
        "source_id": "alice", "target_id": "stranger", "rel_type": "meets",
    }])
    delta, issues = build_delta(extracted, update_number=3, known_ids={"alice"})
    assert delta.relationship_changes == []
    assert any("stranger" in i.message for i in issues)
    assert all(i.severity == "error" for i in issues)


def test_extract_schema_has_required_fields():
    required = set(EXTRACT_SCHEMA["required"])
    assert required == {
        "entity_updates",
        "new_relationships",
        "removed_relationships",
        "timeline_events",
        "foreshadowing_updates",
    }


def test_no_known_ids_skips_id_filtering():
    """When known_ids is None, no id-based filtering occurs."""
    extracted = _merge(entity_updates=[{"id": "anything", "patch": {"status": "dormant"}}])
    delta, issues = build_delta(extracted, update_number=1, known_ids=None)
    assert issues == []
    assert len(delta.entity_updates) == 1
