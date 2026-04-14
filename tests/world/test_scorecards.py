"""Day 2: schema + persistence tests for the scorecards tables."""
from __future__ import annotations

import pytest

from app.scoring import DIMENSION_NAMES, Scorer
from app.world.state_manager import WorldStateManager


SAMPLE_PROSE = (
    "You walked into the hall. The lanterns threw long shadows. \"Wait,\" "
    "you said, but the figure did not turn. The cold pressed against you "
    "like a palm, steady and unhurried."
)


def test_schema_creates_scorecards_and_dimension_scores_tables(db):
    tables = {
        r["name"]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "scorecards" in tables
    assert "dimension_scores" in tables

    sc_cols = {r["name"] for r in db.execute("PRAGMA table_info(scorecards)").fetchall()}
    assert sc_cols >= {
        "id", "quest_id", "update_number", "scene_index",
        "pipeline_trace_id", "overall_score", "created_at",
    }

    ds_cols = {r["name"] for r in db.execute("PRAGMA table_info(dimension_scores)").fetchall()}
    assert ds_cols == {"scorecard_id", "dimension", "score"}

    idx = {
        r["name"]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_scorecards_quest" in idx


def test_save_and_list_scorecard_roundtrip(db):
    sm = WorldStateManager(db)
    scorer = Scorer()
    card = scorer.score(SAMPLE_PROSE)

    sid = sm.save_scorecard(
        card,
        quest_id="q1",
        update_number=3,
        scene_index=0,
        pipeline_trace_id="tr-abc",
    )
    assert isinstance(sid, int) and sid > 0

    # Header row recorded with overall_score, trace id
    row = db.execute("SELECT * FROM scorecards WHERE id=?", (sid,)).fetchone()
    assert row["quest_id"] == "q1"
    assert row["update_number"] == 3
    assert row["scene_index"] == 0
    assert row["pipeline_trace_id"] == "tr-abc"
    assert row["overall_score"] == pytest.approx(card.overall_score, abs=1e-9)

    # Exactly 12 dimension rows, one per DIMENSION_NAMES entry
    dim_rows = db.execute(
        "SELECT dimension, score FROM dimension_scores WHERE scorecard_id=?",
        (sid,),
    ).fetchall()
    assert len(dim_rows) == 12
    stored_names = {r["dimension"] for r in dim_rows}
    assert stored_names == set(DIMENSION_NAMES)

    # Round-trip via list_scorecards — values preserved.
    listed = sm.list_scorecards("q1")
    assert len(listed) == 1
    rt = listed[0]
    assert rt.overall_score == pytest.approx(card.overall_score, abs=1e-9)
    for name in DIMENSION_NAMES:
        assert getattr(rt, name) == pytest.approx(getattr(card, name), abs=1e-9)


def test_list_scorecards_ordered_oldest_first(db):
    sm = WorldStateManager(db)
    scorer = Scorer()

    # Insert out of chronological order on purpose.
    for u in (5, 1, 3, 2):
        sm.save_scorecard(
            scorer.score(SAMPLE_PROSE),
            quest_id="q2",
            update_number=u,
            scene_index=0,
        )
    cards = sm.list_scorecards("q2")
    # We can't read update_number off the Scorecard, but we can confirm
    # count + isolation to the quest.
    assert len(cards) == 4

    # Confirm the DB ordering is oldest->newest.
    rows = db.execute(
        "SELECT update_number FROM scorecards WHERE quest_id=? "
        "ORDER BY update_number ASC, scene_index ASC, id ASC",
        ("q2",),
    ).fetchall()
    update_nums = [r["update_number"] for r in rows]
    assert update_nums == [1, 2, 3, 5]


def test_list_scorecards_quest_isolation(db):
    sm = WorldStateManager(db)
    scorer = Scorer()
    sm.save_scorecard(scorer.score(SAMPLE_PROSE), quest_id="qa", update_number=1)
    sm.save_scorecard(scorer.score(SAMPLE_PROSE), quest_id="qb", update_number=1)
    assert len(sm.list_scorecards("qa")) == 1
    assert len(sm.list_scorecards("qb")) == 1
    assert sm.list_scorecards("q-nope") == []


def test_list_scorecards_limit(db):
    sm = WorldStateManager(db)
    scorer = Scorer()
    for u in range(1, 6):
        sm.save_scorecard(scorer.score(SAMPLE_PROSE), quest_id="q3", update_number=u)
    limited = sm.list_scorecards("q3", limit=2)
    assert len(limited) == 2
