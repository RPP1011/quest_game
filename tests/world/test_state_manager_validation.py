# tests/world/test_state_manager_validation.py
from __future__ import annotations
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    FSUpdate,
    PTUpdate,
    RelChange,
    StateDelta,
)
from app.world.schema import (
    ArcPosition,
    Entity,
    EntityStatus,
    EntityType,
    ForeshadowingHook,
    PlotThread,
    Relationship,
)
from app.world.state_manager import WorldStateManager


def _sm_with_alice_bob(db) -> WorldStateManager:
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="a", entity_type=EntityType.CHARACTER, name="Alice"))
    sm.create_entity(Entity(id="b", entity_type=EntityType.CHARACTER, name="Bob"))
    return sm


def test_valid_empty_delta(db):
    sm = WorldStateManager(db)
    assert sm.validate_delta(StateDelta()).ok


def test_entity_update_missing_id_is_error(db):
    sm = WorldStateManager(db)
    d = StateDelta(entity_updates=[EntityUpdate(id="ghost", patch={"status": "dormant"})])
    r = sm.validate_delta(d)
    assert not r.ok
    assert any("ghost" in i.message for i in r.issues)


def test_entity_create_duplicate_id_is_error(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(entity_creates=[EntityCreate(
        entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="Dup"))])
    r = sm.validate_delta(d)
    assert not r.ok


def test_relationship_add_missing_endpoint_is_error(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(relationship_changes=[RelChange(
        action="add",
        relationship=Relationship(source_id="a", target_id="ghost", rel_type="ally"),
    )])
    r = sm.validate_delta(d)
    assert not r.ok


def test_relationship_remove_missing_is_error(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(relationship_changes=[RelChange(
        action="remove",
        relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
    )])
    r = sm.validate_delta(d)
    assert not r.ok


def test_fs_update_missing_hook_is_error(db):
    sm = WorldStateManager(db)
    d = StateDelta(foreshadowing_updates=[
        FSUpdate(id="fs:missing", new_status="paid_off")])
    assert not sm.validate_delta(d).ok


def test_pt_update_missing_thread_is_error(db):
    sm = WorldStateManager(db)
    d = StateDelta(plot_thread_updates=[PTUpdate(id="pt:missing", patch={"priority": 9})])
    assert not sm.validate_delta(d).ok


def test_acting_on_deceased_entity_is_warning(db):
    sm = _sm_with_alice_bob(db)
    sm.update_entity("a", {"status": EntityStatus.DECEASED.value})
    d = StateDelta(entity_updates=[
        EntityUpdate(id="a", patch={"last_referenced_update": 10})])
    r = sm.validate_delta(d)
    # warning only — not an error
    assert r.ok
    assert any(i.severity == "warning" for i in r.issues)


def test_valid_delta_with_create_and_rel(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(
        entity_creates=[EntityCreate(entity=Entity(
            id="c", entity_type=EntityType.LOCATION, name="Tavern"))],
        relationship_changes=[RelChange(
            action="add",
            relationship=Relationship(source_id="a", target_id="c", rel_type="located_at"),
        )],
    )
    r = sm.validate_delta(d)
    assert r.ok
