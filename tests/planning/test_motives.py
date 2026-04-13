"""G13 — tests for app/planning/motives.py helpers."""
from __future__ import annotations

from app.planning.motives import (
    UnconsciousMotive,
    apply_motive_resolutions,
    pick_primary_motive,
    unconscious_motives_for,
)
from app.world.schema import Entity, EntityType


def _hero(motives: list[dict]) -> Entity:
    return Entity(
        id="hero",
        entity_type=EntityType.CHARACTER,
        name="Hero",
        data={"unconscious_motives": motives},
    )


def test_unconscious_motives_for_returns_active_only():
    entity = _hero([
        {
            "id": "um:hero:erase",
            "motive": "needs to disappear into others' purposes",
            "surface_manifestations": ["defers decisions"],
            "detail_tells": ["watches hands"],
            "what_not_to_say": ["erase", "disappear"],
            "active_since_update": 0,
            "resolved_at_update": None,
        },
        {
            "id": "um:hero:old",
            "motive": "resolved guilt",
            "surface_manifestations": [],
            "detail_tells": [],
            "what_not_to_say": [],
            "active_since_update": 0,
            "resolved_at_update": 3,
        },
    ])
    out = unconscious_motives_for(entity)
    assert len(out) == 1
    assert out[0].id == "um:hero:erase"
    assert isinstance(out[0], UnconsciousMotive)


def test_unconscious_motives_for_missing_key_returns_empty():
    e = Entity(id="x", entity_type=EntityType.CHARACTER, name="X", data={})
    assert unconscious_motives_for(e) == []


def test_unconscious_motives_for_skips_malformed_entries():
    entity = _hero([
        "not a dict",
        {"id": "um:ok", "motive": "fine", "active_since_update": 1},
    ])
    out = unconscious_motives_for(entity)
    assert len(out) == 1
    assert out[0].id == "um:ok"


def test_pick_primary_motive_prefers_earliest_active_since():
    a = UnconsciousMotive(id="a", motive="a", active_since_update=5)
    b = UnconsciousMotive(id="b", motive="b", active_since_update=2)
    c = UnconsciousMotive(id="c", motive="c", active_since_update=8)
    assert pick_primary_motive([a, b, c]).id == "b"


def test_pick_primary_motive_empty_returns_none():
    assert pick_primary_motive([]) is None


def test_apply_motive_resolutions_marks_resolved():
    data = {"unconscious_motives": [
        {"id": "a", "motive": "x", "active_since_update": 0, "resolved_at_update": None},
        {"id": "b", "motive": "y", "active_since_update": 0, "resolved_at_update": None},
    ]}
    new_data = apply_motive_resolutions(data, [{"motive_id": "a", "resolved_at_update": 7}])
    assert new_data["unconscious_motives"][0]["resolved_at_update"] == 7
    assert new_data["unconscious_motives"][1]["resolved_at_update"] is None
    # original untouched
    assert data["unconscious_motives"][0]["resolved_at_update"] is None


def test_apply_motive_resolutions_unknown_id_ignored():
    data = {"unconscious_motives": [
        {"id": "a", "motive": "x", "active_since_update": 0, "resolved_at_update": None},
    ]}
    new_data = apply_motive_resolutions(data, [{"motive_id": "ghost", "resolved_at_update": 7}])
    assert new_data is data  # no change
