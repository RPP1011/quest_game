"""Tests for reader_state persistence + accumulation (Gap G6)."""
from __future__ import annotations

import pytest

from app.planning.reader_model import apply_dramatic_plan
from app.planning.schemas import (
    ActionResolution,
    DramaticPlan,
    DramaticScene,
    ThreadAdvance,
)
from app.world.schema import (
    Expectation,
    ExpectationStatus,
    OpenQuestion,
    ReaderState,
)
from app.world.state_manager import WorldStateManager


@pytest.fixture
def mgr(db):
    return WorldStateManager(db)


def _make_plan(
    *,
    questions_opened=(),
    questions_closed=(),
    expectations_set=(),
    expectations_subverted=(),
    tension=0.5,
    scene_tension=0.5,
    thread_advance_type=None,
    reveals=(),
) -> DramaticPlan:
    scene = DramaticScene(
        scene_id=1,
        dramatic_question="Q?",
        outcome="resolved",
        beats=["a"],
        dramatic_function="escalation",
        tension_target=scene_tension,
        reveals=list(reveals),
    )
    advances = []
    if thread_advance_type is not None:
        advances.append(ThreadAdvance(
            thread_id="t1",
            advance_type=thread_advance_type,
            description="x",
        ))
    return DramaticPlan(
        action_resolution=ActionResolution(kind="success", narrative="ok"),
        scenes=[scene],
        update_tension_target=tension,
        ending_hook="hook",
        suggested_choices=[{"title": "c", "description": "", "tags": []}],
        thread_advances=advances,
        questions_opened=list(questions_opened),
        questions_closed=list(questions_closed),
        expectations_set=list(expectations_set),
        expectations_subverted=list(expectations_subverted),
    )


# --- persistence ---


def test_default_reader_state_when_absent(mgr):
    state = mgr.get_reader_state("q1")
    assert state.quest_id == "q1"
    assert state.open_questions == []
    assert state.expectations == []
    assert state.updates_since_major_event == 0


def test_upsert_and_roundtrip(mgr):
    state = ReaderState(
        quest_id="q1",
        known_fact_ids=["f1", "f2"],
        open_questions=[OpenQuestion(id="q_a", text="who?", opened_at_update=1)],
        expectations=[Expectation(
            id="e_a", text="will win", confidence=0.7,
            status=ExpectationStatus.PENDING, set_at_update=1,
        )],
        attachment_levels={"hero": 0.8},
        current_emotional_valence=-0.3,
        updates_since_major_event=2,
        updates_since_revelation=5,
        updates_since_emotional_peak=1,
    )
    mgr.upsert_reader_state(state)
    got = mgr.get_reader_state("q1")
    assert got == state


# --- accumulation logic ---


def test_questions_opened_accumulate():
    s = ReaderState(quest_id="q1")
    s = apply_dramatic_plan(s, _make_plan(questions_opened=["who did it?"]), update_number=1)
    s = apply_dramatic_plan(s, _make_plan(questions_opened=["where is it?"]), update_number=2)
    assert len(s.open_questions) == 2
    assert {q.text for q in s.open_questions} == {"who did it?", "where is it?"}
    assert s.open_questions[0].opened_at_update == 1


def test_questions_closed_remove_by_loose_match():
    s = ReaderState(quest_id="q1")
    s = apply_dramatic_plan(
        s, _make_plan(questions_opened=["Who is the traitor?"]), update_number=1,
    )
    assert len(s.open_questions) == 1
    # Close with a different casing + substring
    s = apply_dramatic_plan(
        s, _make_plan(questions_closed=["who is the traitor"]), update_number=2,
    )
    assert s.open_questions == []


def test_questions_closed_by_explicit_id():
    s = ReaderState(quest_id="q1")
    s = apply_dramatic_plan(
        s, _make_plan(questions_opened=["is it safe?"]), update_number=1,
    )
    qid = s.open_questions[0].id
    s = apply_dramatic_plan(
        s, _make_plan(questions_closed=[f"id:{qid}"]), update_number=2,
    )
    assert s.open_questions == []


def test_expectations_set_and_subverted():
    s = ReaderState(quest_id="q1")
    s = apply_dramatic_plan(
        s, _make_plan(expectations_set=["hero wins"]), update_number=1,
    )
    assert len(s.expectations) == 1
    assert s.expectations[0].status == ExpectationStatus.PENDING
    s = apply_dramatic_plan(
        s, _make_plan(expectations_subverted=["hero wins"]), update_number=2,
    )
    assert s.expectations[0].status == ExpectationStatus.SUBVERTED


def test_counters_increment_each_update():
    s = ReaderState(quest_id="q1")
    # quiet plan (low tension, no reveals, no resolve)
    s = apply_dramatic_plan(s, _make_plan(tension=0.3, scene_tension=0.3), update_number=1)
    assert s.updates_since_major_event == 1
    assert s.updates_since_revelation == 1
    assert s.updates_since_emotional_peak == 1
    s = apply_dramatic_plan(s, _make_plan(tension=0.3, scene_tension=0.3), update_number=2)
    assert s.updates_since_major_event == 2
    assert s.updates_since_revelation == 2


def test_counter_reset_on_major_event_via_thread_resolve():
    s = ReaderState(quest_id="q1", updates_since_major_event=5)
    s = apply_dramatic_plan(
        s,
        _make_plan(tension=0.3, scene_tension=0.3, thread_advance_type="resolves"),
        update_number=1,
    )
    assert s.updates_since_major_event == 0


def test_counter_reset_on_revelation_via_closed_question():
    s = ReaderState(
        quest_id="q1",
        updates_since_revelation=4,
        open_questions=[OpenQuestion(id="q", text="who?", opened_at_update=0)],
    )
    s = apply_dramatic_plan(
        s,
        _make_plan(tension=0.3, scene_tension=0.3, questions_closed=["who"]),
        update_number=1,
    )
    assert s.updates_since_revelation == 0


def test_counter_reset_on_emotional_peak_via_tension():
    s = ReaderState(quest_id="q1", updates_since_emotional_peak=7)
    s = apply_dramatic_plan(
        s,
        _make_plan(tension=0.9, scene_tension=0.9),
        update_number=1,
    )
    assert s.updates_since_emotional_peak == 0
    assert s.updates_since_major_event == 0  # high tension also counts as major
