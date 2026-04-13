"""Tests for persistent theme tracking (Gap G4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.planning.world_extensions import Theme
from app.world.seed import SeedLoader
from app.world.state_manager import WorldStateError, WorldStateManager


def test_theme_roundtrip(db):
    sm = WorldStateManager(db)
    th = Theme(
        id="t:loyalty",
        proposition="loyalty demands self-erasure",
        stance="questioning",
        motif_ids=["m:mirror"],
        thesis_character_ids=["char:ally"],
        key_scenes=["scene:betrayal", "Act 2: Confrontation"],
    )
    sm.add_theme("q1", th)

    loaded = sm.get_theme("q1", "t:loyalty")
    assert loaded.proposition == "loyalty demands self-erasure"
    assert loaded.stance == "questioning"
    assert loaded.motif_ids == ["m:mirror"]
    assert loaded.thesis_character_ids == ["char:ally"]
    assert "scene:betrayal" in loaded.key_scenes


def test_theme_scoped_per_quest(db):
    sm = WorldStateManager(db)
    sm.add_theme("q1", Theme(id="t:a", proposition="a"))
    sm.add_theme("q2", Theme(id="t:a", proposition="a in other quest"))
    assert sm.get_theme("q1", "t:a").proposition == "a"
    assert sm.get_theme("q2", "t:a").proposition == "a in other quest"
    assert len(sm.list_themes("q1")) == 1


def test_update_theme_stance_writes(db):
    sm = WorldStateManager(db)
    sm.add_theme("q1", Theme(id="t:x", proposition="p", stance="exploring"))
    sm.update_theme_stance("q1", "t:x", "subverting")
    assert sm.get_theme("q1", "t:x").stance == "subverting"


def test_update_theme_stance_unknown_raises(db):
    sm = WorldStateManager(db)
    with pytest.raises(WorldStateError):
        sm.update_theme_stance("q1", "nope", "affirming")


def test_seed_loads_structured_themes(tmp_path: Path):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({
        "themes": [
            {
                "id": "t:loyalty",
                "proposition": "loyalty demands self-erasure",
                "stance": "questioning",
                "motif_ids": ["m:mirror"],
                "thesis_character_ids": ["char:ally"],
                "key_scenes": ["scene:betrayal"],
            },
        ],
    }))
    payload = SeedLoader.load(p)
    assert len(payload.themes) == 1
    th = payload.themes[0]
    assert th.proposition == "loyalty demands self-erasure"
    assert th.stance == "questioning"
    assert th.motif_ids == ["m:mirror"]


def test_seed_wraps_plain_string_themes(tmp_path: Path):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({
        "themes": ["loyalty", "power corrupts"],
    }))
    payload = SeedLoader.load(p)
    assert len(payload.themes) == 2
    assert payload.themes[0].proposition == "loyalty"
    assert payload.themes[0].stance == "exploring"
    assert payload.themes[0].id == "theme:0"
    assert payload.themes[1].proposition == "power corrupts"


def test_seed_accepts_legacy_name_description(tmp_path: Path):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({
        "themes": [
            {"id": "t:x", "name": "loyalty", "description": "loyalty demands self-erasure"},
        ],
    }))
    payload = SeedLoader.load(p)
    assert payload.themes[0].proposition == "loyalty demands self-erasure"
