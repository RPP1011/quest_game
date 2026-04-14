"""Tests for ``MotifRetriever`` (Wave 4a).

Fixture seeds a :class:`WorldStateManager` with 3 motifs (fresh / due /
overdue) plus sample ``motif_occurrences`` rows so the retriever's
structured-query logic can be exercised without any embedding or
inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.planning.world_extensions import Motif, MotifOccurrence
from app.retrieval import MotifRetriever, Query
from app.world.db import open_db
from app.world.state_manager import WorldStateManager


# -- Fixtures ------------------------------------------------------------


# Motif definitions. The ``current_update`` we anchor against in the
# fixture below is 10, so the due/overdue statuses are:
#
#   * "never" — never occurred; target_interval_min=3 → overdue at u>=3.
#   * "fresh_motif" — last at u=9, intervals 4-8 → gap=1, fresh.
#   * "due_motif" — last at u=4, intervals 3-7 → gap=6, due (not overdue).
#   * "overdue_motif" — last at u=1, intervals 2-4 → gap=9, overdue.
_MOTIFS: list[Motif] = [
    Motif(
        id="never",
        name="The unopened letter",
        description="A letter never yet read; its arrival has meaning.",
        target_interval_min=3,
        target_interval_max=6,
    ),
    Motif(
        id="fresh_motif",
        name="Candle flame",
        description="A candle burning in a dark room.",
        target_interval_min=4,
        target_interval_max=8,
    ),
    Motif(
        id="due_motif",
        name="The cracked mirror",
        description="A mirror with a single spider-web crack across it.",
        target_interval_min=3,
        target_interval_max=7,
    ),
    Motif(
        id="overdue_motif",
        name="Iron key",
        description="A heavy iron key passed from hand to hand.",
        target_interval_min=2,
        target_interval_max=4,
    ),
]


# Occurrences. Each entry is (motif_id, update_number, context, semantic_value).
_OCCURRENCES: list[tuple[str, int, str, str, float]] = [
    ("overdue_motif", 1, "Maren pocketed the iron key as the door locked.",
     "betrayal by concealment", 0.6),
    ("due_motif", 2, "She caught her reflection in the mirror; the crack ran through her face.",
     "identity fracturing under pressure", 0.5),
    ("due_motif", 4, "The mirror split further when the candle flared.",
     "deepening fracture", 0.7),
    ("fresh_motif", 3, "A candle guttered in the hallway draft.",
     "hope embattled", 0.4),
    ("fresh_motif", 9, "The candle steadied between her cupped hands.",
     "hope renewed", 0.6),
]


@pytest.fixture()
def world(tmp_path: Path):
    """World seeded with 4 motifs and several occurrence rows."""
    db_path = tmp_path / "quest.db"
    conn = open_db(db_path)
    sm = WorldStateManager(conn)
    for motif in _MOTIFS:
        sm.add_motif("q1", motif)
    for motif_id, update_number, context, semantic_value, intensity in _OCCURRENCES:
        sm.record_motif_occurrence(
            "q1",
            MotifOccurrence(
                motif_id=motif_id,
                update_number=update_number,
                context=context,
                semantic_value=semantic_value,
                intensity=intensity,
            ),
        )
    try:
        yield sm
    finally:
        conn.close()


@pytest.fixture()
def retriever(world: WorldStateManager) -> MotifRetriever:
    return MotifRetriever(world, "q1")


# -- Tests --------------------------------------------------------------


async def test_overdue_ranks_first_due_second_fresh_excluded(
    retriever: MotifRetriever,
) -> None:
    """Ordering: overdue > due > fresh. Fresh motifs are dropped."""
    q = Query(filters={"current_update": 10})
    results = await retriever.retrieve(q, k=5)

    motif_ids = [r.metadata["motif_id"] for r in results]
    # "never" motif + "overdue_motif" are both overdue; "due_motif" is due;
    # "fresh_motif" is excluded.
    assert "fresh_motif" not in motif_ids
    assert set(motif_ids) == {"never", "overdue_motif", "due_motif"}

    # Overdue statuses both score 1.0 and must come before the due motif.
    statuses = [r.metadata["status"] for r in results]
    assert statuses[0] == "overdue"
    assert statuses[1] == "overdue"
    assert statuses[2] == "due"

    # Scores follow the fixed 1.0 / 0.6 / excluded ladder.
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1.0)
    assert results[2].score == pytest.approx(0.6)

    # Tie-break: the never-seen motif has last_update=-1 (surrogate),
    # which is oldest, so it ranks ahead of overdue_motif (last at u=1).
    assert results[0].metadata["motif_id"] == "never"
    assert results[1].metadata["motif_id"] == "overdue_motif"


async def test_missing_current_update_defaults_to_zero(
    retriever: MotifRetriever,
) -> None:
    """With no ``current_update``, we treat ``now == 0`` (everything early)."""
    results = await retriever.retrieve(Query(), k=10)
    # At u=0, occurred motifs all have gap=-1 or less → fresh.
    motif_ids = {r.metadata["motif_id"] for r in results}
    assert "due_motif" not in motif_ids
    assert "overdue_motif" not in motif_ids
    assert "fresh_motif" not in motif_ids
    # The "never" motif requires current_update >= target_interval_min=3 to
    # be overdue; at u=0 it is still fresh.
    assert "never" not in motif_ids
    assert results == []


async def test_never_occurred_overdue_when_current_update_past_min(
    world: WorldStateManager,
) -> None:
    """A motif with no occurrences is overdue once ``u >= target_interval_min``."""
    retriever = MotifRetriever(world, "q1")
    q = Query(filters={"current_update": 3})
    results = await retriever.retrieve(q, k=10)
    # "never" has target_interval_min=3. At u=3 it should surface as overdue.
    top_ids = [r.metadata["motif_id"] for r in results]
    assert "never" in top_ids
    never = next(r for r in results if r.metadata["motif_id"] == "never")
    assert never.metadata["status"] == "overdue"
    assert never.score == pytest.approx(1.0)
    # A never-seen motif has None for its last_* metadata.
    assert never.metadata["last_update_number"] is None
    assert never.metadata["last_semantic_value"] is None
    assert never.metadata["intervals_since_last"] is None
    # And no recent_contexts.
    assert never.metadata["recent_contexts"] == []


async def test_recent_contexts_carries_last_two_occurrences(
    retriever: MotifRetriever,
) -> None:
    """``recent_contexts`` surfaces the trailing 2 occurrences newest-last."""
    q = Query(filters={"current_update": 10})
    results = await retriever.retrieve(q, k=10)

    due = next(r for r in results if r.metadata["motif_id"] == "due_motif")
    contexts = due.metadata["recent_contexts"]
    assert len(contexts) == 2
    assert [c["update_number"] for c in contexts] == [2, 4]
    # The newest context matches what we seeded.
    assert contexts[-1]["semantic_value"] == "deepening fracture"
    assert contexts[-1]["intensity"] == pytest.approx(0.7)
    assert "mirror split" in contexts[-1]["context"]

    # A motif with only one occurrence returns a single-element list.
    overdue = next(r for r in results if r.metadata["motif_id"] == "overdue_motif")
    assert len(overdue.metadata["recent_contexts"]) == 1
    assert overdue.metadata["recent_contexts"][0]["update_number"] == 1


async def test_result_shape_and_fields(retriever: MotifRetriever) -> None:
    """source_id, text, and metadata match the spec."""
    q = Query(filters={"current_update": 10})
    results = await retriever.retrieve(q, k=10)
    overdue = next(r for r in results if r.metadata["motif_id"] == "overdue_motif")
    assert overdue.source_id == "motif/q1/overdue_motif"
    assert overdue.text == "A heavy iron key passed from hand to hand."
    assert overdue.metadata["name"] == "Iron key"
    assert overdue.metadata["last_update_number"] == 1
    assert overdue.metadata["last_semantic_value"] == "betrayal by concealment"
    assert overdue.metadata["intervals_since_last"] == 9
    assert overdue.metadata["target_interval_min"] == 2
    assert overdue.metadata["target_interval_max"] == 4
    assert overdue.metadata["status"] == "overdue"


async def test_k_limits_results(retriever: MotifRetriever) -> None:
    q = Query(filters={"current_update": 10})
    results = await retriever.retrieve(q, k=1)
    assert len(results) == 1
    assert results[0].metadata["status"] == "overdue"


async def test_no_motifs_returns_empty(tmp_path: Path) -> None:
    """An empty motifs table yields empty results."""
    db_path = tmp_path / "empty.db"
    conn = open_db(db_path)
    try:
        sm = WorldStateManager(conn)
        retriever = MotifRetriever(sm, "q1")
        results = await retriever.retrieve(
            Query(filters={"current_update": 10}), k=3
        )
        assert results == []
    finally:
        conn.close()


async def test_quest_scope_isolation(tmp_path: Path) -> None:
    """Motifs/occurrences from another quest never surface."""
    db_path = tmp_path / "quest.db"
    conn = open_db(db_path)
    try:
        sm = WorldStateManager(conn)
        # Seed q1 with an overdue motif.
        sm.add_motif(
            "q1",
            Motif(
                id="iron_key",
                name="Iron key",
                description="A heavy iron key.",
                target_interval_min=2,
                target_interval_max=4,
            ),
        )
        sm.record_motif_occurrence(
            "q1",
            MotifOccurrence(
                motif_id="iron_key",
                update_number=1,
                semantic_value="betrayal",
            ),
        )
        # Seed other_quest with a fresh motif that would otherwise dominate.
        sm.add_motif(
            "other_quest",
            Motif(
                id="letter",
                name="Sealed letter",
                description="A letter with a black seal.",
                target_interval_min=1,
                target_interval_max=3,
            ),
        )
        retriever = MotifRetriever(sm, "q1")
        results = await retriever.retrieve(
            Query(filters={"current_update": 10}), k=10
        )
        for r in results:
            assert r.source_id.startswith("motif/q1/")
            assert r.metadata["motif_id"] != "letter"
    finally:
        conn.close()
