from __future__ import annotations
import pytest
from app.world.schema import StoryCandidate, StoryCandidateStatus
from app.world.state_manager import WorldStateManager, WorldStateError


@pytest.fixture
def sm(db):
    return WorldStateManager(db)


def _cand(cid: str, qid: str = "q1", **kw) -> StoryCandidate:
    defaults = {
        "id": cid, "quest_id": qid, "title": f"Title {cid}",
        "synopsis": f"Synopsis for {cid}",
        "primary_thread_ids": ["pt:main"],
        "secondary_thread_ids": [],
        "protagonist_character_id": "char:hero",
        "emphasized_theme_ids": ["t:honor"],
        "climax_description": "The showdown",
        "expected_chapter_count": 15,
    }
    defaults.update(kw)
    return StoryCandidate(**defaults)


def test_add_and_get_candidate(sm: WorldStateManager):
    c = _cand("cand_1")
    sm.add_story_candidate(c)
    got = sm.get_story_candidate("cand_1")
    assert got.id == "cand_1"
    assert got.title == "Title cand_1"
    assert got.primary_thread_ids == ["pt:main"]
    assert got.emphasized_theme_ids == ["t:honor"]
    assert got.status == StoryCandidateStatus.DRAFT


def test_list_by_quest(sm: WorldStateManager):
    sm.add_story_candidate(_cand("a", qid="q1"))
    sm.add_story_candidate(_cand("b", qid="q1"))
    sm.add_story_candidate(_cand("c", qid="q2"))
    q1 = sm.list_story_candidates("q1")
    q2 = sm.list_story_candidates("q2")
    assert sorted(c.id for c in q1) == ["a", "b"]
    assert [c.id for c in q2] == ["c"]


def test_pick_swaps_picked(sm: WorldStateManager):
    sm.add_story_candidate(_cand("a"))
    sm.add_story_candidate(_cand("b"))
    sm.add_story_candidate(_cand("c"))
    sm.pick_story_candidate("q1", "b")
    assert sm.get_picked_candidate("q1").id == "b"
    # Re-pick swaps: a becomes picked, b reverts to rejected
    sm.pick_story_candidate("q1", "a")
    assert sm.get_picked_candidate("q1").id == "a"
    assert sm.get_story_candidate("b").status == StoryCandidateStatus.REJECTED


def test_pick_rejects_siblings(sm: WorldStateManager):
    sm.add_story_candidate(_cand("a"))
    sm.add_story_candidate(_cand("b"))
    sm.pick_story_candidate("q1", "a")
    # Both a draft and the previously-picked get touched
    assert sm.get_story_candidate("a").status == StoryCandidateStatus.PICKED
    # Sibling that was DRAFT stays DRAFT (only previously-PICKED flips)
    assert sm.get_story_candidate("b").status == StoryCandidateStatus.DRAFT


def test_pick_cross_quest_raises(sm: WorldStateManager):
    sm.add_story_candidate(_cand("a", qid="q1"))
    with pytest.raises(WorldStateError):
        sm.pick_story_candidate("q2", "a")


def test_get_picked_when_none(sm: WorldStateManager):
    sm.add_story_candidate(_cand("a"))
    assert sm.get_picked_candidate("q1") is None


def test_get_missing_raises(sm: WorldStateManager):
    with pytest.raises(WorldStateError):
        sm.get_story_candidate("nope")
