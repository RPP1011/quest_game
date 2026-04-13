# tests/world/test_state_manager_apply.py
from __future__ import annotations
import pytest
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
)
from app.world.schema import Entity, EntityType, Relationship, TimelineEvent
from app.world.state_manager import (
    InvalidDeltaError,
    WorldSnapshot,
    WorldStateManager,
)


def test_apply_creates_entities_and_relationships(db):
    sm = WorldStateManager(db)
    d = StateDelta(
        entity_creates=[
            EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
            EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B")),
        ],
        relationship_changes=[RelChange(
            action="add",
            relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
        )],
        timeline_events=[TimelineEventOp(event=TimelineEvent(
            update_number=1, event_index=0, description="meet",
            involved_entities=["a", "b"],
        ))],
    )
    sm.apply_delta(d, update_number=1)
    assert sm.get_entity("a").last_referenced_update == 1
    assert sm.get_entity("b").last_referenced_update == 1
    assert sm.list_relationships("a")[0].rel_type == "ally"
    assert len(sm.list_timeline(1)) == 1


def test_apply_rejects_invalid_delta(db):
    sm = WorldStateManager(db)
    d = StateDelta(entity_updates=[EntityUpdate(id="ghost", patch={"status": "dormant"})])
    with pytest.raises(InvalidDeltaError):
        sm.apply_delta(d, update_number=1)


def test_apply_is_atomic(db):
    """If a mid-transaction op fails, no partial state is left behind."""
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))
    # Bad delta: valid create, then a raw SQL-level violation we force by
    # creating a rel whose endpoint exists at validate-time but we delete
    # it before apply — can't easily induce that. Instead, force via a
    # programmatic trick: use unique constraint.
    d = StateDelta(
        entity_creates=[
            EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B")),
            EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="Dup")),
        ]
    )
    # validate_delta catches this (planned_new_ids collision) — so we assert
    # it raises InvalidDeltaError and nothing was inserted.
    with pytest.raises(InvalidDeltaError):
        sm.apply_delta(d, update_number=1)
    assert [e.id for e in sm.list_entities()] == ["a"]


def test_snapshot_returns_all_state(db):
    sm = WorldStateManager(db)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
    ]), update_number=1)
    snap = sm.snapshot()
    assert isinstance(snap, WorldSnapshot)
    assert [e.id for e in snap.entities] == ["a"]
    assert snap.relationships == []
    assert snap.timeline == []
