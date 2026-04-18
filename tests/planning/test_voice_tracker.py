from __future__ import annotations

from app.planning.voice_tracker import (
    CharacterVoiceTracker, detect_signature_family,
)
from app.world.schema import Entity, EntityType


def _entity(eid: str, tics: list[str] | None = None, worldview: str = "") -> Entity:
    data: dict = {}
    if tics is not None:
        data["voice"] = {"signature_tics": tics}
    if worldview:
        data["worldview"] = worldview
    return Entity(id=eid, entity_type=EntityType.CHARACTER, name=eid, data=data)


def test_detect_gambling_signature():
    e = _entity("char:tristan", tics=["narrates to himself in gambling metaphors"])
    assert detect_signature_family(e) == "gambling"


def test_detect_honor_signature():
    e = _entity("char:angharad", worldview="honor is the weight you carry")
    sig = detect_signature_family(e)
    assert sig == "weight_gravity"


def test_detect_no_signature():
    e = _entity("char:nobody", tics=["coughs a lot"])
    assert detect_signature_family(e) is None


def test_tracker_from_entities():
    entities = [
        _entity("char:tristan", tics=["gambling metaphors"]),
        _entity("char:angharad", worldview="honor and duty"),
        _entity("char:fortuna"),
    ]
    tracker = CharacterVoiceTracker.from_entities(entities)
    assert tracker.signatures["char:tristan"] == "gambling"
    assert tracker.signatures["char:angharad"] == "weight_gravity"
    assert tracker.signatures["char:fortuna"] is None


def test_record_and_guidance():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling", "char:angharad": "weight_gravity"},
        signature_target=3,
    )
    # Record a chapter heavy on gambling
    prose = " ".join(["the odds shifted. the house wins. bad bet. the deck stacked."] * 10)
    tracker.record_chapter(1, "char:tristan", prose)

    guidance = tracker.get_writer_guidance("char:tristan")
    assert "gambling" in guidance.lower()
    assert "signature" in guidance.lower()


def test_guidance_warns_overuse():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling"},
        signature_target=3,
    )
    # 3 chapters with heavy gambling
    for i in range(3):
        prose = " ".join(["the odds the house the bet the dice the deck the pot"] * 5)
        tracker.record_chapter(i + 1, "char:tristan", prose)

    guidance = tracker.get_writer_guidance("char:tristan")
    assert "overused" in guidance.lower() or "target" in guidance.lower()


def test_guidance_suggests_other_character_avoidance():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling", "char:angharad": "weight_gravity"},
        signature_target=3,
    )
    guidance = tracker.get_writer_guidance("char:tristan")
    # Should mention Angharad's register as something to avoid
    assert "angharad" in guidance.lower() or "weight" in guidance.lower()


def test_critic_context_cumulative():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling"},
    )
    tracker.record_chapter(1, "char:tristan", "the odds bad bet the dice")
    tracker.record_chapter(2, "char:tristan", "the odds the bet gamble")
    ctx = tracker.get_critic_context()
    assert ctx.get("gambling", 0) >= 4


def test_chapter_report():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling"},
        signature_target=3,
    )
    prose = "the odds shifted. the house always wins. bad bet. the flood came."
    report = tracker.get_chapter_report(1, "char:tristan", prose)
    assert report["signature_family"] == "gambling"
    assert report["signature_count"] >= 3
    assert report["families_used"] >= 2  # gambling + water


def test_ring_buffer_eviction():
    tracker = CharacterVoiceTracker(
        signatures={"char:x": "gambling"}, max_history=3,
    )
    for i in range(5):
        tracker.record_chapter(i + 1, "char:x", "the odds")
    assert len(tracker.history) == 3
    assert tracker.history[0].chapter_index == 3  # oldest evicted
