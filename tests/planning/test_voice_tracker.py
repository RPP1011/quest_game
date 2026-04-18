from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

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


def _mock_client(families: dict[str, int] | None = None):
    """Create a mock client that returns a fixed LLM classification."""
    fam_data = families or {}
    classification = {
        "families": {k: {"count": v, "quotes": []} for k, v in fam_data.items()},
        "total_figurative": sum(fam_data.values()),
        "dominant_family": max(fam_data, key=fam_data.get) if fam_data else "none",
        "dominant_percentage": 0,
    }

    client = MagicMock()
    client.chat = AsyncMock(return_value='{"families": {}}')

    # Patch classify_metaphors_llm to return our fixed classification
    import app.planning.metaphor_critic as mc
    original = mc.classify_metaphors_llm

    async def mock_classify(c, prose):
        return classification

    mc.classify_metaphors_llm = mock_classify
    client._restore = lambda: setattr(mc, "classify_metaphors_llm", original)
    return client


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


@pytest.mark.asyncio
async def test_record_and_guidance():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling", "char:angharad": "weight_gravity"},
        signature_target=3,
    )
    client = _mock_client({"gambling": 8, "water_ocean": 2})
    try:
        await tracker.record_chapter(client, 1, "char:tristan", "some prose")
        guidance = tracker.get_writer_guidance("char:tristan")
        assert "gambling" in guidance.lower()
    finally:
        client._restore()


@pytest.mark.asyncio
async def test_guidance_warns_overuse():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling"},
        signature_target=3,
    )
    client = _mock_client({"gambling": 10, "fire_light": 2})
    try:
        for i in range(3):
            await tracker.record_chapter(client, i + 1, "char:tristan", "prose")
        guidance = tracker.get_writer_guidance("char:tristan")
        assert "high" in guidance.lower() or "pull back" in guidance.lower()
    finally:
        client._restore()


def test_critic_context_cumulative():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling"},
    )
    # Manually insert snapshots (bypassing LLM)
    from app.planning.voice_tracker import ChapterVoiceSnapshot
    tracker.history.append(ChapterVoiceSnapshot(1, "char:tristan", {"gambling": 3}))
    tracker.history.append(ChapterVoiceSnapshot(2, "char:tristan", {"gambling": 4}))
    ctx = tracker.get_critic_context()
    assert ctx.get("gambling", 0) == 7


@pytest.mark.asyncio
async def test_chapter_report():
    tracker = CharacterVoiceTracker(
        signatures={"char:tristan": "gambling"},
        signature_target=3,
    )
    client = _mock_client({"gambling": 3, "water_ocean": 2})
    try:
        report = await tracker.get_chapter_report(
            client, 1, "char:tristan", "some prose",
        )
        assert report["signature_family"] == "gambling"
        assert report["signature_count"] == 3
        assert report["families_used"] == 2
    finally:
        client._restore()


@pytest.mark.asyncio
async def test_ring_buffer_eviction():
    tracker = CharacterVoiceTracker(
        signatures={"char:x": "gambling"}, max_history=3,
    )
    client = _mock_client({"gambling": 1})
    try:
        for i in range(5):
            await tracker.record_chapter(client, i + 1, "char:x", "prose")
        assert len(tracker.history) == 3
        assert tracker.history[0].chapter_index == 3
    finally:
        client._restore()
