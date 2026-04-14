"""Tests for :class:`app.retrieval.voice_retriever.VoiceRetriever` (Wave 4c).

The light variant attributes quoted dialogue to a character based on the
scene's POV (stored on ``NarrativeRecord.pov_character_id``). Tests cover:

* POV-scoped filtering (char_a vs char_b).
* Recency ordering (newest records surface first).
* Cold-start fallback to ``Entity.data["voice"]["voice_samples"]``.
* Missing ``character_id`` filter → empty result.
* Straight and curly double-quote detection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval import Query, VoiceRetriever
from app.world.db import open_db
from app.world.delta import EntityCreate, StateDelta
from app.world.schema import Entity, EntityType, NarrativeRecord
from app.world.state_manager import WorldStateManager


@pytest.fixture()
def world(tmp_path: Path):
    db_path = tmp_path / "voice_quest.db"
    conn = open_db(db_path)
    sm = WorldStateManager(conn)
    sm.apply_delta(
        StateDelta(entity_creates=[
            EntityCreate(entity=Entity(
                id="char_a",
                entity_type=EntityType.CHARACTER,
                name="Alyss",
            )),
            EntityCreate(entity=Entity(
                id="char_b",
                entity_type=EntityType.CHARACTER,
                name="Benet",
            )),
        ]),
        update_number=1,
    )
    try:
        yield sm
    finally:
        conn.close()


def _write_record(
    world: WorldStateManager,
    *,
    update_number: int,
    pov: str | None,
    raw_text: str,
) -> None:
    world.write_narrative(NarrativeRecord(
        update_number=update_number,
        raw_text=raw_text,
        pov_character_id=pov,
    ))


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------


async def test_character_id_filters_to_pov_scoped_lines(world: WorldStateManager) -> None:
    """Only quotes from records whose POV matches the target character surface."""
    # char_a POV, with two quoted utterances.
    _write_record(
        world,
        update_number=1,
        pov="char_a",
        raw_text=(
            'She paused at the sill. "The frost has come early," she said, '
            'and the wind pressed the glass. "But we are ready."'
        ),
    )
    # char_b POV — should never surface under character_id=char_a.
    _write_record(
        world,
        update_number=2,
        pov="char_b",
        raw_text='He shrugged into his coat. "None of that concerns me."',
    )
    # char_a POV again, a later record.
    _write_record(
        world,
        update_number=3,
        pov="char_a",
        raw_text='"Call it what you will," she said, turning away.',
    )

    vr = VoiceRetriever(world, "q1")
    results = await vr.retrieve(Query(filters={"character_id": "char_a"}), k=5)

    # No char_b material anywhere in the output.
    joined = " ".join(r.text for r in results)
    assert "None of that concerns me" not in joined

    # char_a's lines are all present.
    texts = [r.text for r in results]
    assert "Call it what you will," in texts
    assert "The frost has come early," in texts
    assert "But we are ready." in texts

    # source_id namespace is correct.
    assert all(r.source_id.startswith("voice/q1/char_a/") for r in results)


async def test_most_recent_record_lines_come_first(world: WorldStateManager) -> None:
    """Lines from the newest record rank ahead of older ones."""
    _write_record(
        world,
        update_number=1,
        pov="char_a",
        raw_text='"An old line," she said.',
    )
    _write_record(
        world,
        update_number=5,
        pov="char_a",
        raw_text='"A newer line," she said.',
    )

    vr = VoiceRetriever(world, "q1")
    results = await vr.retrieve(Query(filters={"character_id": "char_a"}), k=5)

    assert [r.text for r in results] == ["A newer line,", "An old line,"]
    assert results[0].metadata["source_update_number"] == 5
    assert results[1].metadata["source_update_number"] == 1


async def test_cold_start_falls_back_to_entity_voice_samples(tmp_path: Path) -> None:
    """No committed records but the entity carries voice_samples → those surface."""
    db_path = tmp_path / "cold.db"
    conn = open_db(db_path)
    try:
        sm = WorldStateManager(conn)
        sm.apply_delta(
            StateDelta(entity_creates=[
                EntityCreate(entity=Entity(
                    id="char_a",
                    entity_type=EntityType.CHARACTER,
                    name="Alyss",
                    data={"voice": {"voice_samples": [
                        "She turned the key in the lock, slowly.",
                        "It was the kind of morning that made her quiet.",
                    ]}},
                )),
            ]),
            update_number=1,
        )

        vr = VoiceRetriever(sm, "q1")
        results = await vr.retrieve(
            Query(filters={"character_id": "char_a"}), k=3,
        )

        assert len(results) == 2
        assert all(r.source_id.startswith("voice/seed/char_a/") for r in results)
        assert results[0].text == "She turned the key in the lock, slowly."
        assert results[0].metadata["seed"] is True
    finally:
        conn.close()


async def test_missing_character_id_filter_returns_empty(
    world: WorldStateManager,
) -> None:
    """Without ``character_id`` the retriever has nothing to attribute to."""
    _write_record(
        world,
        update_number=1,
        pov="char_a",
        raw_text='"Hello world," she said.',
    )

    vr = VoiceRetriever(world, "q1")

    assert await vr.retrieve(Query(), k=5) == []
    assert await vr.retrieve(Query(filters={}), k=5) == []
    # Non-string / empty values also fail closed.
    assert await vr.retrieve(Query(filters={"character_id": ""}), k=5) == []
    assert await vr.retrieve(Query(filters={"character_id": None}), k=5) == []


async def test_regex_catches_both_straight_and_curly_double_quotes(
    world: WorldStateManager,
) -> None:
    """``"..."`` and curly ``“...”`` both parse as quoted lines."""
    _write_record(
        world,
        update_number=1,
        pov="char_a",
        raw_text=(
            'She stepped outside. "Straight-quote line here." '
            '\u201CCurly-quote line here.\u201D'
        ),
    )

    vr = VoiceRetriever(world, "q1")
    results = await vr.retrieve(Query(filters={"character_id": "char_a"}), k=5)

    texts = [r.text for r in results]
    assert "Straight-quote line here." in texts
    assert "Curly-quote line here." in texts


# ---------------------------------------------------------------------------
# Quote-extraction unit coverage
# ---------------------------------------------------------------------------


def test_extract_quoted_lines_filters_by_length() -> None:
    """Lines shorter than 5 chars or longer than 400 are dropped."""
    text = (
        '"a" '                    # too short — 1 char
        '"ok, enough here." '     # valid
        + '"' + ("x" * 500) + '"'  # too long
    )
    out = VoiceRetriever._extract_quoted_lines(text)
    assert out == ["ok, enough here."]


def test_extract_quoted_lines_dedupes_preserving_order() -> None:
    """Duplicate quoted spans are de-duped case-sensitively, first-seen wins."""
    text = '"Hello there." Some narration. "Hello there."  "And another."'
    out = VoiceRetriever._extract_quoted_lines(text)
    assert out == ["Hello there.", "And another."]


def test_extract_quoted_lines_returns_empty_on_blank_input() -> None:
    assert VoiceRetriever._extract_quoted_lines("") == []
    assert VoiceRetriever._extract_quoted_lines("no quotes here at all.") == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_last_n_records_caps_pool(world: WorldStateManager) -> None:
    """Records beyond ``last_n_records`` are invisible to extraction."""
    # Five char_a records; ``last_n_records=2`` sees the newest two only.
    for n in range(1, 6):
        _write_record(
            world,
            update_number=n,
            pov="char_a",
            raw_text=f'"Line number {n}."',
        )

    vr = VoiceRetriever(world, "q1")
    results = await vr.retrieve(
        Query(filters={"character_id": "char_a", "last_n_records": 2}),
        k=10,
    )
    texts = [r.text for r in results]
    assert texts == ["Line number 5.", "Line number 4."]


async def test_k_caps_results_not_records(world: WorldStateManager) -> None:
    """``k`` bounds returned *lines*, even when a single record has many quotes."""
    _write_record(
        world,
        update_number=1,
        pov="char_a",
        raw_text=(
            '"Alpha is first." "Beta is second." '
            '"Gamma is third." "Delta is fourth."'
        ),
    )
    vr = VoiceRetriever(world, "q1")
    results = await vr.retrieve(Query(filters={"character_id": "char_a"}), k=2)
    assert len(results) == 2
    assert [r.text for r in results] == ["Alpha is first.", "Beta is second."]
