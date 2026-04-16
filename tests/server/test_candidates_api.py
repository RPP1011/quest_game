from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, PlotThread, QuestArcState, ReaderState,
    ThreadStatus,
)
from app.world.state_manager import WorldStateManager


CANNED_CANDIDATES = json.dumps({
    "candidates": [
        {
            "title": "Heist", "synopsis": "A straightforward heist.",
            "primary_thread_ids": ["pt:main"], "secondary_thread_ids": [],
            "protagonist_character_id": "char:hero",
            "emphasized_theme_ids": [], "climax_description": "The vault opens.",
            "expected_chapter_count": 10,
        },
        {
            "title": "Feud", "synopsis": "Hero and rival escalate.",
            "primary_thread_ids": ["pt:rival"], "secondary_thread_ids": [],
            "protagonist_character_id": "char:rival",
            "emphasized_theme_ids": [], "climax_description": "The duel.",
            "expected_chapter_count": 12,
        },
    ]
})


def _init_quest(quests_dir: Path, qid: str = "q1") -> None:
    paths = quests_dir / qid
    paths.mkdir(parents=True, exist_ok=True)
    (paths / "traces").mkdir(exist_ok=True)
    conn = open_db(paths / "quest.db")
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(id="char:hero", entity_type=EntityType.CHARACTER,
                             name="Hero", data={"description": "The hero"}))
    sm.create_entity(Entity(id="char:rival", entity_type=EntityType.CHARACTER,
                             name="Rival", data={"description": "The rival"}))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="Main", description="The main thread.",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_plot_thread(PlotThread(
        id="pt:rival", name="Rival", description="The rival thread.",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=6,
    ))
    sm.upsert_arc(QuestArcState(
        quest_id=qid, arc_id="main", structure_id="three_act",
        scale="campaign", current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=qid))
    conn.close()
    # Minimal config.json so pick_candidate can merge into it
    (paths / "config.json").write_text("{}")


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw):
        return self._response


def test_list_generate_pick(tmp_path: Path):
    quests_dir = tmp_path / "quests"
    _init_quest(quests_dir)
    app = create_app(quests_dir=quests_dir, server_url="http://unused")

    # Patch the StoryCandidatePlanner's client at generation time
    from app.planning import story_candidate_planner as scp_mod
    real_class = scp_mod.StoryCandidatePlanner

    class PatchedPlanner(real_class):
        def __init__(self, *a, **kw):
            super().__init__(FakeClient(CANNED_CANDIDATES), a[1] if len(a) > 1 else kw.get("renderer"))

    with patch.object(scp_mod, "StoryCandidatePlanner", PatchedPlanner):
        with TestClient(app) as tc:
            # Initially empty
            r = tc.get("/api/quests/q1/candidates")
            assert r.status_code == 200
            assert r.json() == []

            # Generate
            r = tc.post("/api/quests/q1/candidates/generate?n=2")
            assert r.status_code == 200
            cands = r.json()
            assert len(cands) == 2
            assert cands[0]["title"] == "Heist"
            assert cands[1]["title"] == "Feud"

            # List shows them
            r = tc.get("/api/quests/q1/candidates")
            listed = r.json()
            assert len(listed) == 2

            # Pick one
            cid = cands[0]["id"]
            r = tc.post(f"/api/quests/q1/candidates/{cid}/pick")
            assert r.status_code == 200
            picked = r.json()
            assert picked["status"] == "picked"

            # config.json now carries picked_candidate
            cfg = json.loads((quests_dir / "q1" / "config.json").read_text())
            assert cfg["picked_candidate"]["id"] == cid
            assert cfg["picked_candidate"]["title"] == "Heist"

            # Picking the other swaps
            other = cands[1]["id"]
            r = tc.post(f"/api/quests/q1/candidates/{other}/pick")
            assert r.status_code == 200
            cfg2 = json.loads((quests_dir / "q1" / "config.json").read_text())
            assert cfg2["picked_candidate"]["id"] == other


def test_pick_unknown_returns_404(tmp_path: Path):
    quests_dir = tmp_path / "quests"
    _init_quest(quests_dir)
    app = create_app(quests_dir=quests_dir, server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.post("/api/quests/q1/candidates/does_not_exist/pick")
        assert r.status_code == 404
