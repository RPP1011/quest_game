from __future__ import annotations
import pytest
from app.world.schema import (
    RolloutChapter, RolloutExtract, RolloutRun, RolloutStatus, StoryCandidate,
)
from app.world.state_manager import WorldStateManager, WorldStateError


@pytest.fixture
def sm(db):
    wsm = WorldStateManager(db)
    wsm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="T", synopsis="S",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=10,
    ))
    return wsm


def _run(rid: str = "r1", **kw) -> RolloutRun:
    return RolloutRun(
        id=rid, quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=3, **kw,
    )


def test_create_and_get_rollout(sm):
    sm.create_rollout(_run())
    got = sm.get_rollout("r1")
    assert got.id == "r1"
    assert got.status == RolloutStatus.PENDING
    assert got.chapters_complete == 0


def test_update_rollout_status_and_progress(sm):
    sm.create_rollout(_run())
    sm.update_rollout("r1", status=RolloutStatus.RUNNING, started_at="2026-04-16T00:00:00Z")
    sm.update_rollout("r1", chapters_complete=2)
    got = sm.get_rollout("r1")
    assert got.status == RolloutStatus.RUNNING
    assert got.chapters_complete == 2
    assert got.started_at == "2026-04-16T00:00:00Z"


def test_update_missing_raises(sm):
    with pytest.raises(WorldStateError):
        sm.update_rollout("nope", status=RolloutStatus.FAILED)


def test_list_rollouts_by_candidate(sm):
    sm.create_rollout(_run("r1"))
    sm.create_rollout(_run("r2"))
    runs = sm.list_rollouts(candidate_id="cand_1")
    assert {r.id for r in runs} == {"r1", "r2"}


def test_save_and_list_chapters(sm):
    sm.create_rollout(_run())
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="act 1", prose="prose 1", trace_id="t1",
        extract=RolloutExtract(hooks_planted=["fs:x"]),
    ))
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r1", chapter_index=2,
        player_action="act 2", prose="prose 2", trace_id="t2",
    ))
    chs = sm.list_rollout_chapters("r1")
    assert len(chs) == 2
    assert chs[0].chapter_index == 1
    assert chs[0].extract.hooks_planted == ["fs:x"]
    assert chs[1].chapter_index == 2


def test_save_chapter_upserts(sm):
    sm.create_rollout(_run())
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r1", chapter_index=1, player_action="a", prose="p1",
    ))
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r1", chapter_index=1, player_action="a2", prose="p2",
    ))
    chs = sm.list_rollout_chapters("r1")
    assert len(chs) == 1
    assert chs[0].prose == "p2"


def test_judge_scores_round_trip(sm):
    sm.create_rollout(_run())
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r1", chapter_index=1, player_action="a", prose="p",
        judge_scores={"tension_execution": 0.8, "voice": 0.6},
    ))
    ch = sm.list_rollout_chapters("r1")[0]
    assert ch.judge_scores == {"tension_execution": 0.8, "voice": 0.6}
