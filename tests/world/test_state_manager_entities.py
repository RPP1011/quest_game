# tests/world/test_state_manager_entities.py
from __future__ import annotations
import pytest
from app.world.schema import Entity, EntityStatus, EntityType
from app.world.state_manager import EntityNotFoundError, WorldStateManager


def test_create_and_get_entity(db):
    sm = WorldStateManager(db)
    e = Entity(id="char:alice", entity_type=EntityType.CHARACTER, name="Alice",
               data={"age": 27}, created_at_update=1)
    sm.create_entity(e)
    got = sm.get_entity("char:alice")
    assert got == e


def test_get_missing_entity_raises(db):
    sm = WorldStateManager(db)
    with pytest.raises(EntityNotFoundError):
        sm.get_entity("char:ghost")


def test_list_entities_filters_by_type(db):
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="c1", entity_type=EntityType.CHARACTER, name="A"))
    sm.create_entity(Entity(id="c2", entity_type=EntityType.CHARACTER, name="B"))
    sm.create_entity(Entity(id="l1", entity_type=EntityType.LOCATION, name="Town"))
    chars = sm.list_entities(entity_type=EntityType.CHARACTER)
    assert {e.id for e in chars} == {"c1", "c2"}
    all_ = sm.list_entities()
    assert len(all_) == 3


def test_update_entity_patches_fields(db):
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="c1", entity_type=EntityType.CHARACTER, name="A"))
    sm.update_entity("c1", {"status": "dormant", "last_referenced_update": 5})
    got = sm.get_entity("c1")
    assert got.status == EntityStatus.DORMANT
    assert got.last_referenced_update == 5


def test_update_entity_merges_data_field(db):
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="c1", entity_type=EntityType.CHARACTER, name="A",
                             data={"hp": 10, "mood": "ok"}))
    sm.update_entity("c1", {"data": {"hp": 8}})  # partial data patch merges
    got = sm.get_entity("c1")
    assert got.data == {"hp": 8, "mood": "ok"}


def test_update_missing_entity_raises(db):
    sm = WorldStateManager(db)
    with pytest.raises(EntityNotFoundError):
        sm.update_entity("nope", {"status": "dormant"})
