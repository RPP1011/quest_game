from __future__ import annotations
import pytest
from app.world.schema import Entity, EntityType, Relationship
from app.world.state_manager import RelationshipNotFoundError, WorldStateManager


def _seed_two(db) -> WorldStateManager:
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))
    sm.create_entity(Entity(id="b", entity_type=EntityType.CHARACTER, name="B"))
    return sm


def test_add_and_list_relationships(db):
    sm = _seed_two(db)
    sm.add_relationship(Relationship(source_id="a", target_id="b", rel_type="ally"))
    rels = sm.list_relationships(source_id="a")
    assert len(rels) == 1
    assert rels[0].rel_type == "ally"


def test_remove_relationship(db):
    sm = _seed_two(db)
    r = Relationship(source_id="a", target_id="b", rel_type="ally")
    sm.add_relationship(r)
    sm.remove_relationship("a", "b", "ally")
    assert sm.list_relationships(source_id="a") == []


def test_remove_missing_relationship_raises(db):
    sm = _seed_two(db)
    with pytest.raises(RelationshipNotFoundError):
        sm.remove_relationship("a", "b", "ally")


def test_modify_relationship_updates_data(db):
    sm = _seed_two(db)
    sm.add_relationship(Relationship(source_id="a", target_id="b", rel_type="ally", data={"trust": 5}))
    sm.modify_relationship("a", "b", "ally", {"trust": 8})
    rels = sm.list_relationships(source_id="a")
    assert rels[0].data == {"trust": 8}
