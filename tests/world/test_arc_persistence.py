from __future__ import annotations
import pytest
from app.world.db import open_db
from app.world.schema import QuestArcState
from app.world.state_manager import WorldStateManager, WorldStateError


@pytest.fixture
def mgr(db):
    return WorldStateManager(db)


def _make_arc(quest_id: str = "q1", arc_id: str = "main") -> QuestArcState:
    return QuestArcState(
        arc_id=arc_id,
        quest_id=quest_id,
        structure_id="three_act",
        scale="chapter",
    )


def test_upsert_and_get_arc_roundtrip(mgr):
    arc = _make_arc()
    mgr.upsert_arc(arc)
    retrieved = mgr.get_arc("q1", "main")
    assert retrieved.arc_id == "main"
    assert retrieved.quest_id == "q1"
    assert retrieved.structure_id == "three_act"
    assert retrieved.scale == "chapter"
    assert retrieved.current_phase_index == 0
    assert retrieved.phase_progress == 0.0
    assert retrieved.tension_observed == []
    assert retrieved.last_directive is None


def test_upsert_overwrites_existing(mgr):
    arc = _make_arc()
    mgr.upsert_arc(arc)

    updated = arc.model_copy(update={
        "current_phase_index": 2,
        "phase_progress": 0.75,
        "last_directive": {"current_phase": "rising"},
    })
    mgr.upsert_arc(updated)

    retrieved = mgr.get_arc("q1", "main")
    assert retrieved.current_phase_index == 2
    assert retrieved.phase_progress == 0.75
    assert retrieved.last_directive == {"current_phase": "rising"}


def test_get_missing_arc_raises(mgr):
    with pytest.raises(WorldStateError):
        mgr.get_arc("no_quest", "no_arc")


def test_list_arcs_for_quest(mgr):
    mgr.upsert_arc(_make_arc(quest_id="q1", arc_id="main"))
    mgr.upsert_arc(_make_arc(quest_id="q1", arc_id="subplot"))
    mgr.upsert_arc(_make_arc(quest_id="q2", arc_id="main"))

    arcs_q1 = mgr.list_arcs("q1")
    assert len(arcs_q1) == 2
    arc_ids = {a.arc_id for a in arcs_q1}
    assert arc_ids == {"main", "subplot"}

    arcs_q2 = mgr.list_arcs("q2")
    assert len(arcs_q2) == 1
    assert arcs_q2[0].arc_id == "main"

    arcs_q3 = mgr.list_arcs("q3")
    assert arcs_q3 == []


def test_record_tension_appends(mgr):
    arc = _make_arc()
    mgr.upsert_arc(arc)

    mgr.record_tension("q1", "main", update_number=1, value=0.4)
    mgr.record_tension("q1", "main", update_number=2, value=0.6)
    mgr.record_tension("q1", "main", update_number=3, value=0.8)

    retrieved = mgr.get_arc("q1", "main")
    assert len(retrieved.tension_observed) == 3
    assert retrieved.tension_observed[0] == (1, 0.4)
    assert retrieved.tension_observed[1] == (2, 0.6)
    assert retrieved.tension_observed[2] == (3, 0.8)
