"""Tests for RulesEngine and WorldRule constraint evaluation."""
from __future__ import annotations
import pytest
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    RelChange,
    StateDelta,
    ValidationIssue,
)
from app.world.schema import (
    Entity,
    EntityStatus,
    EntityType,
    Relationship,
    WorldRule,
)
from app.world.rules_engine import RulesEngine
from app.world.state_manager import WorldStateManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sm(db) -> WorldStateManager:
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="alice", entity_type=EntityType.CHARACTER, name="Alice"))
    sm.create_entity(Entity(id="bob", entity_type=EntityType.CHARACTER, name="Bob"))
    return sm


def _rule(rule_id: str, constraints: dict) -> WorldRule:
    return WorldRule(
        id=rule_id,
        category="test",
        description="test rule",
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# forbid_relationship — hit
# ---------------------------------------------------------------------------

def test_forbid_relationship_hit(db):
    sm = _make_sm(db)
    rule = _rule("r1", {"type": "forbid_relationship", "rel_type": "enemy"})
    engine = RulesEngine([rule])
    delta = StateDelta(
        relationship_changes=[
            RelChange(
                action="add",
                relationship=Relationship(source_id="alice", target_id="bob", rel_type="enemy"),
            )
        ]
    )
    issues = engine.evaluate(delta, sm)
    assert any(i.severity == "error" and "r1" in i.message and "enemy" in i.message for i in issues)


# ---------------------------------------------------------------------------
# forbid_relationship — miss (different rel_type)
# ---------------------------------------------------------------------------

def test_forbid_relationship_miss(db):
    sm = _make_sm(db)
    rule = _rule("r1", {"type": "forbid_relationship", "rel_type": "enemy"})
    engine = RulesEngine([rule])
    delta = StateDelta(
        relationship_changes=[
            RelChange(
                action="add",
                relationship=Relationship(source_id="alice", target_id="bob", rel_type="ally"),
            )
        ]
    )
    issues = engine.evaluate(delta, sm)
    assert not any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# forbid_relationship — remove action is not flagged
# ---------------------------------------------------------------------------

def test_forbid_relationship_only_flags_add(db):
    sm = _make_sm(db)
    sm.add_relationship(
        Relationship(source_id="alice", target_id="bob", rel_type="enemy")
    )
    rule = _rule("r1", {"type": "forbid_relationship", "rel_type": "enemy"})
    engine = RulesEngine([rule])
    delta = StateDelta(
        relationship_changes=[
            RelChange(
                action="remove",
                relationship=Relationship(source_id="alice", target_id="bob", rel_type="enemy"),
            )
        ]
    )
    issues = engine.evaluate(delta, sm)
    assert not any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# forbid_status_transition — hit
# ---------------------------------------------------------------------------

def test_forbid_status_transition_hit(db):
    sm = _make_sm(db)
    # Alice is currently "active"
    rule = _rule("r2", {"type": "forbid_status_transition", "from": "active", "to": "dormant"})
    engine = RulesEngine([rule])
    delta = StateDelta(
        entity_updates=[EntityUpdate(id="alice", patch={"status": "dormant"})]
    )
    issues = engine.evaluate(delta, sm)
    assert any(i.severity == "error" and "r2" in i.message for i in issues)


# ---------------------------------------------------------------------------
# forbid_status_transition — miss (different from state)
# ---------------------------------------------------------------------------

def test_forbid_status_transition_miss(db):
    sm = _make_sm(db)
    # Alice is "active", rule forbids deceased->dormant (irrelevant)
    rule = _rule("r2", {"type": "forbid_status_transition", "from": "deceased", "to": "dormant"})
    engine = RulesEngine([rule])
    delta = StateDelta(
        entity_updates=[EntityUpdate(id="alice", patch={"status": "dormant"})]
    )
    issues = engine.evaluate(delta, sm)
    assert not any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# forbid_status_transition — no status in patch → no issue
# ---------------------------------------------------------------------------

def test_forbid_status_transition_no_patch_status(db):
    sm = _make_sm(db)
    rule = _rule("r2", {"type": "forbid_status_transition", "from": "active", "to": "dormant"})
    engine = RulesEngine([rule])
    delta = StateDelta(
        entity_updates=[EntityUpdate(id="alice", patch={"name": "Alicia"})]
    )
    issues = engine.evaluate(delta, sm)
    assert not any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# unique_entity_name — hit
# ---------------------------------------------------------------------------

def test_unique_entity_name_hit(db):
    sm = _make_sm(db)
    rule = _rule("r3", {"type": "unique_entity_name"})
    engine = RulesEngine([rule])
    # "Alice" already exists
    delta = StateDelta(
        entity_creates=[
            EntityCreate(
                entity=Entity(id="alice2", entity_type=EntityType.CHARACTER, name="Alice")
            )
        ]
    )
    issues = engine.evaluate(delta, sm)
    assert any(i.severity == "error" and "r3" in i.message for i in issues)


# ---------------------------------------------------------------------------
# unique_entity_name — miss (new unique name)
# ---------------------------------------------------------------------------

def test_unique_entity_name_miss(db):
    sm = _make_sm(db)
    rule = _rule("r3", {"type": "unique_entity_name"})
    engine = RulesEngine([rule])
    delta = StateDelta(
        entity_creates=[
            EntityCreate(
                entity=Entity(id="charlie", entity_type=EntityType.CHARACTER, name="Charlie")
            )
        ]
    )
    issues = engine.evaluate(delta, sm)
    assert not any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# unknown type → warning only
# ---------------------------------------------------------------------------

def test_unknown_constraint_type_is_warning(db):
    sm = _make_sm(db)
    rule = _rule("r99", {"type": "nonexistent_constraint"})
    engine = RulesEngine([rule])
    issues = engine.evaluate(StateDelta(), sm)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "r99" in issues[0].message


# ---------------------------------------------------------------------------
# missing type key → warning
# ---------------------------------------------------------------------------

def test_missing_type_key_is_warning(db):
    sm = _make_sm(db)
    rule = _rule("r98", {"rel_type": "enemy"})  # no "type" key
    engine = RulesEngine([rule])
    issues = engine.evaluate(StateDelta(), sm)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "r98" in issues[0].message


# ---------------------------------------------------------------------------
# Integration: rules wired into WorldStateManager.validate_delta
# ---------------------------------------------------------------------------

def test_validate_delta_applies_rules_via_state_manager(db):
    sm = _make_sm(db)
    sm.add_rule(
        WorldRule(
            id="rule:no-enemy",
            category="faction",
            description="No enemy relationships allowed",
            constraints={"type": "forbid_relationship", "rel_type": "enemy"},
        )
    )
    delta = StateDelta(
        relationship_changes=[
            RelChange(
                action="add",
                relationship=Relationship(source_id="alice", target_id="bob", rel_type="enemy"),
            )
        ]
    )
    result = sm.validate_delta(delta)
    assert not result.ok
    assert any("forbidden relationship" in i.message for i in result.issues)
