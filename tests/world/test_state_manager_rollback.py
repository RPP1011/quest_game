# tests/world/test_state_manager_rollback.py
from __future__ import annotations
from app.world.delta import (
    EntityCreate,
    FSUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
)
from app.world.schema import (
    Entity,
    EntityType,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    Relationship,
    TimelineEvent,
)
from app.world.state_manager import WorldStateManager


def _advance(sm: WorldStateManager, update_number: int, delta: StateDelta):
    sm.apply_delta(delta, update_number=update_number)
    sm.write_narrative(NarrativeRecord(update_number=update_number, raw_text=f"u{update_number}"))


def test_rollback_removes_later_entities(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))]))
    _advance(sm, 2, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B"))]))
    sm.rollback(to_update=1)
    ids = [e.id for e in sm.list_entities()]
    assert ids == ["a"]


def test_rollback_removes_later_narrative_and_timeline(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))]))
    _advance(sm, 2, StateDelta(timeline_events=[TimelineEventOp(event=TimelineEvent(
        update_number=2, event_index=0, description="event2"))]))
    sm.rollback(to_update=1)
    assert [n.update_number for n in sm.list_narrative()] == [1]
    assert sm.list_timeline() == []


def test_rollback_removes_later_relationships(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
        EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B")),
    ]))
    _advance(sm, 2, StateDelta(relationship_changes=[RelChange(
        action="add",
        relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
    )]))
    assert len(sm.list_relationships("a")) == 1
    sm.rollback(to_update=1)
    assert sm.list_relationships("a") == []


def test_rollback_reverts_foreshadowing_payoff(db):
    sm = WorldStateManager(db)
    hook = ForeshadowingHook(id="fs:1", description="d", planted_at_update=1, payoff_target="x")
    sm.add_foreshadowing(hook)
    sm.apply_delta(StateDelta(foreshadowing_updates=[
        FSUpdate(id="fs:1", new_status="paid_off", paid_off_at_update=3, add_reference=2),
    ]), update_number=3)
    sm.write_narrative(NarrativeRecord(update_number=3, raw_text="u3"))
    sm.rollback(to_update=2)
    got = sm.get_foreshadowing("fs:1")
    assert got.paid_off_at_update is None
    # status reverted to planted (since payoff happened after to_update)
    assert got.status == HookStatus.PLANTED
    assert got.references == [2]  # reference at update 2 stays


def test_rollback_caps_last_referenced(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))]))
    _advance(sm, 5, StateDelta(timeline_events=[TimelineEventOp(event=TimelineEvent(
        update_number=5, event_index=0, description="x", involved_entities=["a"]))]))
    sm.rollback(to_update=2)
    assert sm.get_entity("a").last_referenced_update == 1
