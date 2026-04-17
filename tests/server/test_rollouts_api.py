from __future__ import annotations
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.server import create_app
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, PlotThread, QuestArcState, ReaderState,
    RolloutRun, StoryCandidate, ThreadStatus,
)
from app.world.state_manager import WorldStateManager


def _init(quests_dir: Path, qid: str = "q1") -> None:
    paths = quests_dir / qid
    paths.mkdir(parents=True, exist_ok=True)
    (paths / "traces").mkdir(exist_ok=True)
    conn = open_db(paths / "quest.db")
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(id="char:hero", entity_type=EntityType.CHARACTER, name="Hero"))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="Main", description="x",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id=qid, title="T", synopsis="S",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:hero", emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=3,
    ))
    sm.upsert_arc(QuestArcState(
        quest_id=qid, arc_id="main", structure_id="three_act",
        scale="campaign", current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=qid))
    conn.close()
    (paths / "config.json").write_text("{}")


def test_list_profiles_endpoint(tmp_path: Path):
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.get("/api/rollout-profiles")
        assert r.status_code == 200
        ids = {p["id"] for p in r.json()}
        assert {"impulsive", "cautious", "honor_bound"}.issubset(ids)


def test_list_rollouts_empty(tmp_path: Path):
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.get("/api/quests/q1/rollouts")
        assert r.status_code == 200
        assert r.json() == []


def test_get_rollout_404(tmp_path: Path):
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.get("/api/quests/q1/rollouts/no_such")
        assert r.status_code == 404


def test_start_rollout_returns_202(tmp_path: Path):
    """Launch endpoint returns immediately with a rollout id; execution
    actually happens in a background task. We verify the row is created;
    we don't wait for execution (which would require a real LLM)."""
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    # Monkey-patch run_rollout to a no-op so the background task doesn't
    # blow up trying to reach a real LLM server.
    from unittest.mock import patch
    async def noop(**kw):
        return None
    with patch("app.rollout.harness.run_rollout", noop):
        with TestClient(app) as tc:
            r = tc.post(
                "/api/quests/q1/candidates/cand_1/rollouts/start"
                "?profile=impulsive&chapters=2"
            )
            assert r.status_code == 202, r.text
            body = r.json()
            assert body["status"] == "pending"
            rid = body["rollout_id"]
            assert rid.startswith("ro_")

            # Row exists
            r2 = tc.get(f"/api/quests/q1/rollouts/{rid}")
            assert r2.status_code == 200
            detail = r2.json()
            assert detail["id"] == rid
            assert detail["total_chapters_target"] == 2
            assert detail["profile_id"] == "impulsive"


def test_start_rollout_bad_candidate(tmp_path: Path):
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.post("/api/quests/q1/candidates/nope/rollouts/start?chapters=2")
        assert r.status_code == 400
