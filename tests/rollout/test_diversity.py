from __future__ import annotations
from pathlib import Path

import pytest

from app.rollout.diversity import measure_rollout_diversity, _jaccard, _ngrams
from app.world.db import open_db
from app.world.schema import (
    Entity, EntityType, RolloutChapter, RolloutRun, StoryCandidate,
)
from app.world.state_manager import WorldStateManager


def test_jaccard_identical():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial():
    assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)


def test_ngrams():
    grams = _ngrams("the quick brown fox jumps", n=3)
    assert ("the", "quick", "brown") in grams
    assert ("brown", "fox", "jumps") in grams


@pytest.fixture
def sm(tmp_path):
    conn = open_db(tmp_path / "w.db")
    wsm = WorldStateManager(conn)
    wsm.create_entity(Entity(
        id="char:hero", entity_type=EntityType.CHARACTER, name="Tristan",
    ))
    wsm.create_entity(Entity(
        id="char:rival", entity_type=EntityType.CHARACTER, name="Cozme",
    ))
    wsm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="T", synopsis="S",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=2,
    ))
    for rid, profile in [("r1", "impulsive"), ("r2", "cautious")]:
        wsm.create_rollout(RolloutRun(
            id=rid, quest_id="q1", candidate_id="cand_1",
            profile_id=profile, total_chapters_target=2,
        ))
    yield wsm
    conn.close()


def test_diversity_different_actions(sm):
    """Different actions → low action Jaccard → high diversity."""
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="charge the enemy head-on with sword drawn",
        prose="Tristan charged. The blade sang.",
    ))
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r2", chapter_index=1,
        player_action="observe the enemy from the shadows carefully",
        prose="Cozme watched from the dark. The silence pressed.",
    ))
    result = measure_rollout_diversity(sm, "r1", "r2", "q1")
    assert result["aggregate"]["action_jaccard_mean"] < 0.3
    assert result["interpretation"]["action_diversity"] in ("moderate", "high")


def test_diversity_identical_actions(sm):
    """Identical actions → high Jaccard → low diversity."""
    action = "charge the enemy head-on"
    prose = "Tristan charged. Cozme fell."
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action=action, prose=prose,
    ))
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id="r2", chapter_index=1,
        player_action=action, prose=prose,
    ))
    result = measure_rollout_diversity(sm, "r1", "r2", "q1")
    assert result["aggregate"]["action_jaccard_mean"] == 1.0
    assert result["aggregate"]["prose_4gram_jaccard_mean"] == 1.0
