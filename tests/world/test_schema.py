from __future__ import annotations
import pytest
from pydantic import ValidationError
from app.world.schema import (
    ArcPosition,
    Entity,
    EntityStatus,
    EntityType,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    PlotThread,
    Relationship,
    ThreadStatus,
    TimelineEvent,
    WorldRule,
)


def test_entity_defaults():
    e = Entity(id="char:alice", entity_type=EntityType.CHARACTER, name="Alice")
    assert e.status == EntityStatus.ACTIVE
    assert e.data == {}
    assert e.last_referenced_update is None
    assert e.created_at_update is None


def test_entity_accepts_arbitrary_data():
    e = Entity(
        id="loc:tavern",
        entity_type=EntityType.LOCATION,
        name="The Broken Anchor",
        data={"climate": "coastal", "population": 412},
    )
    assert e.data["population"] == 412


def test_entity_type_enum_rejects_unknown():
    with pytest.raises(ValidationError):
        Entity(id="x", entity_type="dragonoid", name="X")


def test_relationship_requires_endpoints():
    r = Relationship(source_id="char:alice", target_id="char:bob", rel_type="ally")
    assert r.data == {}


def test_world_rule_with_constraints():
    r = WorldRule(
        id="rule:no-magic-in-zone-3",
        category="magic_system",
        description="Magic does not function within zone 3.",
        constraints={"zone": 3, "effect": "disabled"},
    )
    assert r.constraints["zone"] == 3


def test_timeline_event_ordering_fields():
    ev = TimelineEvent(
        update_number=5,
        event_index=0,
        description="Alice enters the tavern.",
        involved_entities=["char:alice", "loc:tavern"],
    )
    assert ev.causal_links == []


def test_narrative_record_roundtrip():
    n = NarrativeRecord(
        update_number=3,
        raw_text="She walked in.",
        player_action="Enter the tavern.",
    )
    assert n.summary is None
    assert n.chapter_id is None


def test_foreshadowing_hook_default_status():
    h = ForeshadowingHook(
        id="fs:001",
        description="Alice touches a strange coin.",
        planted_at_update=2,
        payoff_target="reveal of the coin's origin",
    )
    assert h.status == HookStatus.PLANTED
    assert h.references == []


def test_plot_thread_priority_bounds():
    pt = PlotThread(
        id="pt:main",
        name="The Missing Heir",
        description="Find the lost heir.",
        involved_entities=["char:alice"],
        arc_position=ArcPosition.RISING,
    )
    assert pt.priority == 5
    with pytest.raises(ValidationError):
        PlotThread(
            id="pt:bad", name="x", description="x",
            involved_entities=[], arc_position=ArcPosition.RISING, priority=11,
        )


def test_enums_are_strings():
    # Important so they round-trip through JSON/SQLite
    assert EntityStatus.ACTIVE.value == "active"
    assert HookStatus.PAID_OFF.value == "paid_off"
    assert ThreadStatus.DORMANT.value == "dormant"
    assert ArcPosition.CLIMAX.value == "climax"
