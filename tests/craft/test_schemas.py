import pytest
from pydantic import ValidationError
from app.craft.schemas import (
    Arc, ArcPhase, Example, Structure, StyleRegister, Tool,
)


def test_arc_phase_roundtrip():
    p = ArcPhase(name="rising", position=1, tension_target=0.5,
                 expected_beats=["chekhovs_gun", "try_fail_cycle"],
                 description="Complications build.")
    assert p.tension_target == 0.5
    assert "chekhovs_gun" in p.expected_beats


def test_structure_validates_phase_ordering():
    s = Structure(
        id="three_act", name="Three-Act",
        description="Setup, confrontation, resolution.",
        scales=["chapter", "campaign"],
        phases=[
            ArcPhase(name="setup", position=0, tension_target=0.2, description="x"),
            ArcPhase(name="confrontation", position=1, tension_target=0.7, description="x"),
            ArcPhase(name="resolution", position=2, tension_target=0.4, description="x"),
        ],
        tension_curve=[(0.0, 0.1), (0.5, 0.7), (0.9, 0.9), (1.0, 0.3)],
    )
    assert len(s.phases) == 3
    # Phases must be sorted by position when indexed
    assert s.phases[0].position == 0
    assert s.phases[-1].position == 2


def test_structure_rejects_duplicate_positions():
    with pytest.raises(ValidationError):
        Structure(
            id="x", name="x", description="x", scales=["scene"],
            phases=[
                ArcPhase(name="a", position=0, tension_target=0.1, description="x"),
                ArcPhase(name="b", position=0, tension_target=0.2, description="x"),
            ],
            tension_curve=[(0.0, 0.0), (1.0, 1.0)],
        )


def test_tool_defaults():
    t = Tool(
        id="chekhovs_gun", name="Chekhov's Gun",
        category="foreshadowing",
        description="Plant an element early so its later use feels earned.",
        preconditions=["Scene has room to introduce an incidental detail."],
        signals=["A later payoff is needed but would feel unearned without prep."],
        anti_patterns=["Paying off something that was never planted."],
        example_ids=["ex_chekhov_coin"],
    )
    assert t.category == "foreshadowing"


def test_example_validates_scale():
    e = Example(
        id="ex_x", tool_ids=["reversal"], source="original",
        scale="scene",
        snippet="She smiled. The smile did not reach her eyes.",
        annotation="Sub-clause undercuts the visible gesture — a micro-reversal.",
    )
    assert e.scale == "scene"
    with pytest.raises(ValidationError):
        Example(id="y", tool_ids=["reversal"], source="original",
                scale="galactic", snippet="x", annotation="x")


def test_style_register_voice_samples_required():
    with pytest.raises(ValidationError):
        StyleRegister(id="x", name="x", description="x",
                      sentence_variance="medium", concrete_abstract_ratio=0.5,
                      interiority_depth="medium", pov_discipline="strict",
                      diction_register="formal", voice_samples=[])


def test_arc_minimal():
    a = Arc(id="main", name="The Ostland Dynasty", scale="campaign",
            structure_id="three_act")
    assert a.current_phase_index == 0
    assert a.phase_progress == 0.0
    assert a.tension_observed == []


def test_arc_records_tension():
    a = Arc(id="x", name="x", scale="chapter", structure_id="three_act",
            tension_observed=[(1, 0.3), (2, 0.4), (3, 0.45)])
    assert a.tension_observed[-1] == (3, 0.45)
