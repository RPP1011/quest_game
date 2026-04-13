from __future__ import annotations
import json
from pathlib import Path
import pytest
from app.world.seed import SeedLoader, SeedPayload
from app.world.state_manager import WorldStateManager


def _write_seed(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_full_seed(tmp_path: Path):
    p = _write_seed(tmp_path, {
        "entities": [
            {"id": "char:alice", "entity_type": "character", "name": "Alice"},
            {"id": "loc:tavern", "entity_type": "location", "name": "The Tavern"},
        ],
        "relationships": [
            {"source_id": "char:alice", "target_id": "loc:tavern", "rel_type": "located_at"},
        ],
        "rules": [
            {"id": "r:1", "category": "magic", "description": "No magic here."},
        ],
        "foreshadowing": [
            {"id": "fs:1", "description": "A strange coin.",
             "planted_at_update": 0, "payoff_target": "origin reveal"},
        ],
        "plot_threads": [
            {"id": "pt:main", "name": "Quest", "description": "d",
             "arc_position": "rising"},
        ],
    })
    payload = SeedLoader.load(p)
    assert isinstance(payload, SeedPayload)
    assert len(payload.delta.entity_creates) == 2
    assert len(payload.delta.relationship_changes) == 1
    assert len(payload.rules) == 1
    assert len(payload.foreshadowing) == 1
    assert len(payload.plot_threads) == 1


def test_load_minimal_seed(tmp_path: Path):
    p = _write_seed(tmp_path, {"entities": [
        {"id": "a", "entity_type": "character", "name": "A"},
    ]})
    payload = SeedLoader.load(p)
    assert len(payload.delta.entity_creates) == 1
    assert payload.rules == []


def test_load_rejects_bad_json(tmp_path: Path):
    p = tmp_path / "seed.json"
    p.write_text("{ not json")
    with pytest.raises(ValueError):
        SeedLoader.load(p)


def test_seed_applies_cleanly(db, tmp_path: Path):
    p = _write_seed(tmp_path, {
        "entities": [
            {"id": "a", "entity_type": "character", "name": "A"},
            {"id": "b", "entity_type": "location", "name": "B"},
        ],
        "relationships": [
            {"source_id": "a", "target_id": "b", "rel_type": "located_at"},
        ],
        "plot_threads": [
            {"id": "pt:1", "name": "m", "description": "d", "arc_position": "rising"},
        ],
    })
    payload = SeedLoader.load(p)
    sm = WorldStateManager(db)
    for rule in payload.rules:
        sm.add_rule(rule)
    for h in payload.foreshadowing:
        sm.add_foreshadowing(h)
    for pt in payload.plot_threads:
        sm.add_plot_thread(pt)
    sm.apply_delta(payload.delta, update_number=0)
    assert [e.id for e in sm.list_entities()] == ["a", "b"]
    assert sm.list_relationships("a")[0].target_id == "b"
    assert sm.get_plot_thread("pt:1").name == "m"


def test_seed_preserves_unconscious_motives_in_entity_data(tmp_path: Path):
    """G13: per-character unconscious_motives in seed JSON round-trip via
    Entity.data without loss."""
    from app.planning.motives import unconscious_motives_for
    motives = [
        {
            "id": "um:hero:erase",
            "motive": "needs to disappear into others' purposes",
            "surface_manifestations": ["defers decisions"],
            "detail_tells": ["watches hands"],
            "what_not_to_say": ["erase", "disappear"],
            "active_since_update": 0,
            "resolved_at_update": None,
        }
    ]
    p = _write_seed(tmp_path, {
        "entities": [
            {
                "id": "char:hero",
                "entity_type": "character",
                "name": "Hero",
                "data": {"unconscious_motives": motives},
            },
        ],
    })
    payload = SeedLoader.load(p)
    assert len(payload.delta.entity_creates) == 1
    e = payload.delta.entity_creates[0].entity
    assert e.data["unconscious_motives"] == motives
    # And the helper finds the active motive
    active = unconscious_motives_for(e)
    assert len(active) == 1
    assert active[0].id == "um:hero:erase"
