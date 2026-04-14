"""Shared fixtures for Day 7 optimization tests.

Provides a bare ``WorldStateManager`` backed by an in-memory sqlite with
the canonical schema, seeded with a handful of scorecards + narratives
that exercise both the weak-dim detector and the curator.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.scoring import Scorecard
from app.world.db import open_db
from app.world.schema import NarrativeRecord
from app.world.state_manager import WorldStateManager


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "quest.db"


@pytest.fixture
def world(db_path: Path):
    conn = open_db(db_path)
    try:
        yield WorldStateManager(conn)
    finally:
        conn.close()


def _make_card(**overrides: float) -> Scorecard:
    """Build a Scorecard with uniform 0.5 dims, overridden per-dim."""
    defaults = {
        "sentence_variance": 0.5,
        "dialogue_ratio": 0.5,
        "pacing": 0.5,
        "sensory_density": 0.5,
        "free_indirect_quality": 0.5,
        "detail_characterization": 0.5,
        "metaphor_domains_score": 0.5,
        "indirection_score": 0.5,
        "pov_adherence": 0.5,
        "named_entity_presence": 0.5,
        "narrator_sensory_match": 0.5,
        "action_fidelity": 0.5,
    }
    defaults.update(overrides)
    overall = sum(defaults.values()) / len(defaults)
    return Scorecard(overall_score=overall, **defaults)


@pytest.fixture
def make_card():
    """Factory exposed to tests that want custom dim overrides."""
    return _make_card


@pytest.fixture
def seeded_world(world):
    """Populate scorecards + narratives with a known weak dim pattern.

    free_indirect_quality is pinned low across multiple scorecards;
    the rest hover at 0.5. Each scorecard is bound to a narrative row
    with a distinctive prose preview so snippet lookup works.
    """
    # free_indirect_quality mean = (0.1+0.15+0.2+0.85+0.9)/5 = 0.44
    # pov_adherence mean         = (0.4+0.5+0.45+0.5+0.5)/5 = 0.47
    # So free_indirect_quality is the weakest dim, AND it has some >= 0.5
    # entries (for top-example mining) plus some <= 0.5 entries (for
    # bottom-example mining).
    specs = [
        ("tr-1", 1, {"free_indirect_quality": 0.1, "pov_adherence": 0.4}),
        ("tr-2", 2, {"free_indirect_quality": 0.15}),
        ("tr-3", 3, {"free_indirect_quality": 0.2, "pov_adherence": 0.45}),
        ("tr-4", 4, {"free_indirect_quality": 0.85, "detail_characterization": 0.9}),
        ("tr-5", 5, {"free_indirect_quality": 0.9, "detail_characterization": 0.95}),
    ]
    for trace_id, u, overrides in specs:
        card = _make_card(**overrides)
        world.write_narrative(NarrativeRecord(
            update_number=u,
            raw_text=f"Prose for trace {trace_id} at update {u}.",
            summary=f"summary-{u}",
            chapter_id=u,
            state_diff={},
            player_action=f"Act {u}",
            pipeline_trace_id=trace_id,
            pov_character_id=None,
        ))
        world.save_scorecard(
            card,
            quest_id="qA",
            update_number=u,
            scene_index=0,
            pipeline_trace_id=trace_id,
        )
    return world
