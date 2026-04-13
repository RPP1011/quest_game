"""Tests for persistent motif tracking (Gap G5)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.planning.world_extensions import Motif, MotifOccurrence, Theme
from app.world.seed import SeedLoader
from app.world.state_manager import WorldStateError, WorldStateManager


def test_motif_roundtrip(db):
    sm = WorldStateManager(db)
    m = Motif(
        id="m:mirror",
        name="mirror",
        description="reflective surface revealing truth or distortion",
        theme_ids=["t:identity"],
        semantic_range=["self-knowledge", "illusion", "other-self"],
        target_interval_min=2,
        target_interval_max=5,
    )
    sm.add_motif("q1", m)

    loaded = sm.get_motif("q1", "m:mirror")
    assert loaded.name == "mirror"
    assert loaded.theme_ids == ["t:identity"]
    assert loaded.semantic_range == ["self-knowledge", "illusion", "other-self"]
    assert loaded.target_interval_min == 2
    assert loaded.target_interval_max == 5


def test_motif_scoped_per_quest(db):
    sm = WorldStateManager(db)
    sm.add_motif("q1", Motif(id="m:x", name="x", description="d"))
    sm.add_motif("q2", Motif(id="m:x", name="x2", description="d2"))
    assert sm.get_motif("q1", "m:x").name == "x"
    assert sm.get_motif("q2", "m:x").name == "x2"
    assert len(sm.list_motifs("q1")) == 1


def test_motif_occurrence_persistence_and_query(db):
    sm = WorldStateManager(db)
    sm.add_motif("q1", Motif(id="m:mirror", name="mirror", description="d",
                             semantic_range=["self-knowledge", "illusion"]))
    sm.record_motif_occurrence("q1", MotifOccurrence(
        motif_id="m:mirror", update_number=2, context="scene A",
        semantic_value="self-knowledge", intensity=0.4,
    ))
    sm.record_motif_occurrence("q1", MotifOccurrence(
        motif_id="m:mirror", update_number=5, context="scene B",
        semantic_value="illusion", intensity=0.8,
    ))
    occ = sm.list_motif_occurrences("q1", motif_id="m:mirror")
    assert len(occ) == 2
    assert occ[0].update_number == 2
    assert occ[1].update_number == 5
    last = sm.last_motif_occurrence("q1", "m:mirror")
    assert last is not None
    assert last.update_number == 5
    assert last.semantic_value == "illusion"


def test_last_motif_occurrence_none_when_unused(db):
    sm = WorldStateManager(db)
    sm.add_motif("q1", Motif(id="m:x", name="x", description="d"))
    assert sm.last_motif_occurrence("q1", "m:x") is None


def test_seed_loads_structured_motifs(tmp_path: Path):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({
        "themes": [{"id": "t:identity", "proposition": "who am I"}],
        "motifs": [
            {
                "id": "m:mirror",
                "name": "mirror",
                "description": "reflective surface",
                "theme_ids": ["t:identity"],
                "semantic_range": ["self-knowledge", "illusion"],
                "target_interval_min": 2,
                "target_interval_max": 5,
            },
        ],
    }))
    payload = SeedLoader.load(p)
    assert len(payload.motifs) == 1
    m = payload.motifs[0]
    assert m.name == "mirror"
    assert m.theme_ids == ["t:identity"]
    assert m.semantic_range == ["self-knowledge", "illusion"]


def test_seed_wraps_plain_string_motifs(tmp_path: Path):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({"motifs": ["mirror", "raven"]}))
    payload = SeedLoader.load(p)
    assert len(payload.motifs) == 2
    assert payload.motifs[0].name == "mirror"
    assert payload.motifs[0].id == "motif:0"


def test_theme_motif_linkage_survives_seed_roundtrip(tmp_path: Path, db):
    """Theme.motif_ids references should round-trip through seed + DB, and
    the Motif.theme_ids on the other side should match."""
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({
        "themes": [
            {
                "id": "t:identity",
                "proposition": "who am I becoming",
                "motif_ids": ["m:mirror"],
            },
        ],
        "motifs": [
            {
                "id": "m:mirror",
                "name": "mirror",
                "description": "reflective surface",
                "theme_ids": ["t:identity"],
                "semantic_range": ["self-knowledge"],
            },
        ],
    }))
    payload = SeedLoader.load(p)
    sm = WorldStateManager(db)
    for th in payload.themes:
        sm.add_theme("q1", th)
    for mo in payload.motifs:
        sm.add_motif("q1", mo)

    theme = sm.get_theme("q1", "t:identity")
    motif = sm.get_motif("q1", "m:mirror")
    assert "m:mirror" in theme.motif_ids
    assert "t:identity" in motif.theme_ids


def test_unknown_motif_raises(db):
    sm = WorldStateManager(db)
    with pytest.raises(WorldStateError):
        sm.get_motif("q1", "nope")
