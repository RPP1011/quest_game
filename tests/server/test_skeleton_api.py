from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, ForeshadowingHook, PlotThread,
    QuestArcState, ReaderState, StoryCandidate, ThreadStatus,
)
from app.world.state_manager import WorldStateManager


CANNED_SKELETON = json.dumps({
    "chapters": [
        {
            "chapter_index": 1, "pov_character_id": "char:hero",
            "location_constraint": None,
            "dramatic_question": "Q1?",
            "required_plot_beats": ["pt:main setup"],
            "target_tension": 0.3, "entities_to_surface": [], "theme_emphasis": [],
        },
        {
            "chapter_index": 2, "pov_character_id": "char:hero",
            "location_constraint": None,
            "dramatic_question": "Q2?",
            "required_plot_beats": ["pt:main climax"],
            "target_tension": 0.9, "entities_to_surface": [], "theme_emphasis": [],
        },
    ],
    "hook_schedule": [
        {"hook_id": "fs:1", "planted_by_chapter": 1, "paid_off_by_chapter": 2},
    ],
    "theme_arc": [],
})


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw):
        return self._response


def _init_quest(quests_dir: Path) -> None:
    paths = quests_dir / "q1"
    paths.mkdir(parents=True, exist_ok=True)
    (paths / "traces").mkdir(exist_ok=True)
    conn = open_db(paths / "quest.db")
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(id="char:hero", entity_type=EntityType.CHARACTER, name="Hero"))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="Main", description="x",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_foreshadowing(ForeshadowingHook(
        id="fs:1", description="hook", planted_at_update=0, payoff_target="target",
    ))
    sm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="T", synopsis="S",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:hero", emphasized_theme_ids=[],
        climax_description="The climax.", expected_chapter_count=2,
    ))
    sm.upsert_arc(QuestArcState(
        quest_id="q1", arc_id="main", structure_id="three_act",
        scale="campaign", current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id="q1"))
    conn.close()
    (paths / "config.json").write_text("{}")


def test_get_skeleton_404_before_generate(tmp_path: Path):
    _init_quest(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.get("/api/quests/q1/candidates/cand_1/skeleton")
        assert r.status_code == 404


def test_generate_then_get(tmp_path: Path):
    _init_quest(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")

    from app.server import create_app as _  # keep imported
    from app.planning import arc_skeleton_planner as ask
    real = ask.ArcSkeletonPlanner

    class Patched(real):
        def __init__(self, client, renderer):
            super().__init__(FakeClient(CANNED_SKELETON), renderer)

    with patch.object(ask, "ArcSkeletonPlanner", Patched):
        with TestClient(app) as tc:
            r = tc.post("/api/quests/q1/candidates/cand_1/skeleton/generate")
            assert r.status_code == 200, r.text
            skel = r.json()
            assert len(skel["chapters"]) == 2
            assert skel["chapters"][0]["pov_character_id"] == "char:hero"
            sid = skel["id"]

            # GET now returns it
            r2 = tc.get("/api/quests/q1/candidates/cand_1/skeleton")
            assert r2.status_code == 200
            assert r2.json()["id"] == sid


def test_generate_unknown_candidate_404(tmp_path: Path):
    _init_quest(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.post("/api/quests/q1/candidates/nope/skeleton/generate")
        assert r.status_code == 404
