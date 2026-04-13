"""Persistence + load-order tests for emotional_beats."""
from __future__ import annotations

import pytest

from app.world.schema import EmotionalBeat
from app.world.state_manager import WorldStateManager


@pytest.fixture
def mgr(db):
    return WorldStateManager(db)


def _beat(quest_id: str, update_number: int, scene_index: int, emotion: str,
          intensity: float = 0.5, source: str = "t") -> EmotionalBeat:
    return EmotionalBeat(
        quest_id=quest_id,
        update_number=update_number,
        scene_index=scene_index,
        primary_emotion=emotion,
        intensity=intensity,
        source=source,
    )


def test_record_and_roundtrip(mgr):
    rid = mgr.record_emotional_beat(_beat("q1", 1, 0, "dread", 0.6, "revelation"))
    assert rid > 0
    beats = mgr.list_recent_emotional_beats("q1")
    assert len(beats) == 1
    b = beats[0]
    assert b.quest_id == "q1"
    assert b.update_number == 1
    assert b.scene_index == 0
    assert b.primary_emotion == "dread"
    assert b.intensity == pytest.approx(0.6)
    assert b.source == "revelation"


def test_list_recent_beats_oldest_to_newest(mgr):
    for u, e in [(1, "dread"), (2, "grief"), (3, "relief"), (4, "awe")]:
        mgr.record_emotional_beat(_beat("q1", u, 0, e))
    beats = mgr.list_recent_emotional_beats("q1", limit=10)
    emotions = [b.primary_emotion for b in beats]
    assert emotions == ["dread", "grief", "relief", "awe"]


def test_list_recent_beats_limit_returns_most_recent(mgr):
    for u, e in [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]:
        mgr.record_emotional_beat(_beat("q1", u, 0, e))
    beats = mgr.list_recent_emotional_beats("q1", limit=3)
    # Last 3 (oldest->newest within the window)
    assert [b.primary_emotion for b in beats] == ["c", "d", "e"]


def test_list_recent_beats_scoped_to_quest(mgr):
    mgr.record_emotional_beat(_beat("q1", 1, 0, "dread"))
    mgr.record_emotional_beat(_beat("q2", 1, 0, "joy"))
    q1 = mgr.list_recent_emotional_beats("q1")
    q2 = mgr.list_recent_emotional_beats("q2")
    assert [b.primary_emotion for b in q1] == ["dread"]
    assert [b.primary_emotion for b in q2] == ["joy"]


def test_list_recent_beats_empty_quest(mgr):
    assert mgr.list_recent_emotional_beats("nope") == []


def test_beats_ordered_by_update_then_scene(mgr):
    mgr.record_emotional_beat(_beat("q1", 1, 1, "b"))
    mgr.record_emotional_beat(_beat("q1", 1, 0, "a"))
    mgr.record_emotional_beat(_beat("q1", 2, 0, "c"))
    beats = mgr.list_recent_emotional_beats("q1")
    assert [(b.update_number, b.scene_index, b.primary_emotion) for b in beats] == [
        (1, 0, "a"), (1, 1, "b"), (2, 0, "c"),
    ]
