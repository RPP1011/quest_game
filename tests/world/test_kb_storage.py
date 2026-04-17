from __future__ import annotations
import pytest
from app.world.schema import RolloutRun, StoryCandidate
from app.world.state_manager import WorldStateManager


@pytest.fixture
def sm(db):
    wsm = WorldStateManager(db)
    wsm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="T", synopsis="S",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=10,
    ))
    wsm.create_rollout(RolloutRun(
        id="r1", quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=2,
    ))
    return wsm


def test_save_and_list_chapter_scores(sm):
    sm.save_chapter_scores("r1", 1, {
        "tension_execution": {"score": 0.8, "rationale": "good"},
        "voice_distinctiveness": {"score": 0.6, "rationale": "ok"},
    })
    scores = sm.list_chapter_scores("r1", chapter_index=1)
    by_dim = {s["dim"]: s for s in scores}
    assert by_dim["tension_execution"]["score"] == 0.8
    assert by_dim["tension_execution"]["rationale"] == "good"
    assert by_dim["voice_distinctiveness"]["score"] == 0.6


def test_save_chapter_scores_upserts(sm):
    sm.save_chapter_scores("r1", 1, {"tension_execution": {"score": 0.5, "rationale": "first"}})
    sm.save_chapter_scores("r1", 1, {"tension_execution": {"score": 0.9, "rationale": "second"}})
    scores = sm.list_chapter_scores("r1", chapter_index=1)
    assert len(scores) == 1
    assert scores[0]["score"] == 0.9
    assert scores[0]["rationale"] == "second"


def test_list_chapter_scores_all(sm):
    sm.save_chapter_scores("r1", 1, {"a": {"score": 0.1, "rationale": ""}})
    sm.save_chapter_scores("r1", 2, {"a": {"score": 0.2, "rationale": ""}, "b": {"score": 0.3, "rationale": ""}})
    all_scores = sm.list_chapter_scores("r1")
    assert len(all_scores) == 3
    by_ch = {(s["chapter_index"], s["dim"]): s for s in all_scores}
    assert by_ch[(1, "a")]["score"] == 0.1
    assert by_ch[(2, "b")]["score"] == 0.3


def test_save_and_list_hook_payoff(sm):
    sm.save_hook_payoff(
        quest_id="q1", rollout_id="r1", hook_id="fs:1",
        planted_at_chapter=1, paid_off_at_chapter=3,
    )
    rows = sm.list_hook_payoffs("q1")
    assert rows == [{
        "rollout_id": "r1", "hook_id": "fs:1",
        "planted_at_chapter": 1, "paid_off_at_chapter": 3,
    }]


def test_hook_payoff_partial_upsert(sm):
    """Planted-only insert then paid-off update merges via COALESCE."""
    sm.save_hook_payoff(
        quest_id="q1", rollout_id="r1", hook_id="fs:1",
        planted_at_chapter=2,
    )
    sm.save_hook_payoff(
        quest_id="q1", rollout_id="r1", hook_id="fs:1",
        paid_off_at_chapter=5,
    )
    rows = sm.list_hook_payoffs("q1")
    assert rows[0]["planted_at_chapter"] == 2
    assert rows[0]["paid_off_at_chapter"] == 5


def test_save_and_list_entity_usage(sm):
    sm.save_entity_usage(
        quest_id="q1", rollout_id="r1", entity_id="char:hero",
        introduced_at_chapter=1, mention_chapters=[1, 2, 3],
    )
    rows = sm.list_entity_usage("q1")
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "char:hero"
    assert rows[0]["mention_chapters"] == [1, 2, 3]


def test_entity_usage_upserts(sm):
    sm.save_entity_usage(
        quest_id="q1", rollout_id="r1", entity_id="char:hero",
        introduced_at_chapter=1, mention_chapters=[1],
    )
    sm.save_entity_usage(
        quest_id="q1", rollout_id="r1", entity_id="char:hero",
        mention_chapters=[1, 2, 3, 4],
    )
    rows = sm.list_entity_usage("q1")
    assert rows[0]["mention_chapters"] == [1, 2, 3, 4]
    # introduced_at_chapter preserved via COALESCE
    assert rows[0]["introduced_at_chapter"] == 1
