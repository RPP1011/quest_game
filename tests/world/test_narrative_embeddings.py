"""Tests for the `narrative_embeddings` table and its WorldStateManager
accessors (retrieval layer, Wave 1c).

The embedding column stores raw ``float32`` bytes; round-trip via
``np.frombuffer(..., dtype=np.float32)``. Wave 3b activates the extract
writer hook in ``app/engine/pipeline.py``; this suite covers only the
schema + storage layer.
"""
from __future__ import annotations

import numpy as np

from app.world.state_manager import WorldStateManager


def test_schema_creates_narrative_embeddings_table(db):
    """`open_db` fixture runs SCHEMA_SQL; the table must exist."""
    rows = db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='narrative_embeddings'"
    ).fetchall()
    assert len(rows) == 1

    # And the quest_id index is present too.
    idx = db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND name='idx_ne_quest'"
    ).fetchall()
    assert len(idx) == 1

    # Column set matches spec §2.2.
    cols = {r["name"] for r in db.execute(
        "PRAGMA table_info(narrative_embeddings)"
    ).fetchall()}
    assert cols == {
        "update_number", "scene_index", "quest_id", "embedding", "text_preview",
    }


def test_upsert_and_list_roundtrip_384dim(db):
    """Write a 384-dim embedding, read it back with exact byte fidelity."""
    sm = WorldStateManager(db)
    rng = np.random.default_rng(seed=0xC1)  # deterministic
    vec = rng.standard_normal(384).astype(np.float32)

    sm.upsert_narrative_embedding(
        quest_id="q1",
        update_number=3,
        scene_index=0,
        embedding=vec,
        text_preview="Rain on the flagstones. She stood there, listening.",
    )

    rows = sm.list_narrative_embeddings("q1")
    assert len(rows) == 1
    row = rows[0]
    assert row["update_number"] == 3
    assert row["scene_index"] == 0
    assert row["text_preview"].startswith("Rain on the flagstones")

    loaded = row["embedding"]
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.float32
    assert loaded.shape == (384,)
    np.testing.assert_array_equal(loaded, vec)


def test_list_ordering_newest_first(db):
    """Rows must come back ordered by (update_number DESC, scene_index DESC)."""
    sm = WorldStateManager(db)
    dim = 8
    # Insert out of order to catch accidental insertion-order reliance.
    for update_number, scene_index in [(2, 0), (5, 1), (1, 0), (5, 0), (3, 2)]:
        vec = np.full(dim, float(update_number * 10 + scene_index), dtype=np.float32)
        sm.upsert_narrative_embedding(
            quest_id="q1",
            update_number=update_number,
            scene_index=scene_index,
            embedding=vec,
            text_preview=f"u{update_number}-s{scene_index}",
        )

    rows = sm.list_narrative_embeddings("q1")
    order = [(r["update_number"], r["scene_index"]) for r in rows]
    assert order == [(5, 1), (5, 0), (3, 2), (2, 0), (1, 0)]


def test_list_respects_limit(db):
    sm = WorldStateManager(db)
    dim = 4
    for n in range(5):
        sm.upsert_narrative_embedding(
            quest_id="q1",
            update_number=n,
            scene_index=0,
            embedding=np.zeros(dim, dtype=np.float32),
            text_preview=f"u{n}",
        )

    rows = sm.list_narrative_embeddings("q1", limit=2)
    assert [r["update_number"] for r in rows] == [4, 3]


def test_upsert_replaces_existing_row(db):
    """Primary key collision overwrites embedding + preview."""
    sm = WorldStateManager(db)
    dim = 4
    sm.upsert_narrative_embedding(
        quest_id="q1", update_number=1, scene_index=0,
        embedding=np.zeros(dim, dtype=np.float32),
        text_preview="first",
    )
    sm.upsert_narrative_embedding(
        quest_id="q1", update_number=1, scene_index=0,
        embedding=np.ones(dim, dtype=np.float32),
        text_preview="second",
    )

    rows = sm.list_narrative_embeddings("q1")
    assert len(rows) == 1
    assert rows[0]["text_preview"] == "second"
    np.testing.assert_array_equal(
        rows[0]["embedding"], np.ones(dim, dtype=np.float32),
    )


def test_quest_scope_isolation(db):
    """Listing one quest's embeddings must not leak another quest's rows."""
    sm = WorldStateManager(db)
    dim = 4
    sm.upsert_narrative_embedding(
        quest_id="q1", update_number=1, scene_index=0,
        embedding=np.zeros(dim, dtype=np.float32), text_preview="p1",
    )
    sm.upsert_narrative_embedding(
        quest_id="q2", update_number=1, scene_index=0,
        embedding=np.ones(dim, dtype=np.float32), text_preview="p2",
    )
    r1 = sm.list_narrative_embeddings("q1")
    r2 = sm.list_narrative_embeddings("q2")
    assert len(r1) == 1 and r1[0]["text_preview"] == "p1"
    assert len(r2) == 1 and r2[0]["text_preview"] == "p2"
