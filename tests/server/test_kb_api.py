from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from app.server import create_app
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, PlotThread, QuestArcState, ReaderState,
    RolloutChapter, RolloutRun, StoryCandidate, ThreadStatus,
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
        climax_description="", expected_chapter_count=2,
    ))
    sm.upsert_arc(QuestArcState(
        quest_id=qid, arc_id="main", structure_id="three_act",
        scale="campaign", current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=qid))

    # Two rollouts with scores + KB events
    for rid in ("r1", "r2"):
        sm.create_rollout(RolloutRun(
            id=rid, quest_id=qid, candidate_id="cand_1",
            profile_id="impulsive", total_chapters_target=2,
        ))
        for ch in (1, 2):
            sm.save_rollout_chapter(RolloutChapter(
                rollout_id=rid, chapter_index=ch,
                player_action=f"act {ch}", prose=f"prose {ch}",
            ))
            sm.save_chapter_scores(rid, ch, {
                "tension_execution": {"score": 0.5 + 0.1 * ch, "rationale": "."},
                "voice_distinctiveness": {"score": 0.7, "rationale": "."},
            })
        sm.save_hook_payoff(
            quest_id=qid, rollout_id=rid, hook_id="fs:1",
            planted_at_chapter=1,
            paid_off_at_chapter=2 if rid == "r1" else None,
        )
        sm.save_entity_usage(
            quest_id=qid, rollout_id=rid, entity_id="char:hero",
            introduced_at_chapter=1, mention_chapters=[1, 2],
        )
    conn.close()
    (paths / "config.json").write_text("{}")


def test_get_kb_aggregates(tmp_path: Path):
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.get("/api/quests/q1/kb")
        assert r.status_code == 200
        body = r.json()
        assert body["n_rollouts"] == 2

        # Hook payoffs: fs:1 paid off in 1 of 2 rollouts
        hooks = body["hook_payoffs"]
        assert len(hooks) == 1
        assert hooks[0]["hook_id"] == "fs:1"
        assert hooks[0]["paid_off_count"] == 1
        assert hooks[0]["payoff_rate"] == 0.5

        # Entity usage: char:hero in both rollouts
        eu = body["entity_usage"]
        assert len(eu) == 1
        assert eu[0]["entity_id"] == "char:hero"
        assert eu[0]["screen_time"] == 4  # 2 mentions × 2 rollouts

        # Dim means: per (chapter, dim)
        means = {(d["chapter_index"], d["dim"]): d for d in body["dim_means_by_chapter"]}
        # Both rollouts saved tension=0.6 in ch1, 0.7 in ch2
        assert means[(1, "tension_execution")]["mean"] == pytest.approx(0.6)
        assert means[(1, "tension_execution")]["n_rollouts_scored"] == 2
        assert means[(2, "tension_execution")]["mean"] == pytest.approx(0.7)
        assert means[(1, "voice_distinctiveness")]["mean"] == pytest.approx(0.7)


def test_get_rollout_scores(tmp_path: Path):
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.get("/api/quests/q1/rollouts/r1/scores")
        assert r.status_code == 200
        body = r.json()
        assert body["rollout_id"] == "r1"
        assert len(body["chapters"]) == 2
        ch1 = body["chapters"][0]
        assert ch1["chapter_index"] == 1
        assert ch1["dims"]["tension_execution"]["score"] == pytest.approx(0.6)


def test_get_scores_404_unknown_rollout(tmp_path: Path):
    _init(tmp_path / "quests")
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://unused")
    with TestClient(app) as tc:
        r = tc.get("/api/quests/q1/rollouts/nope/scores")
        assert r.status_code == 404


import pytest
