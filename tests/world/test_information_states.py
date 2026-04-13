"""Tests for information_states persistence + asymmetry detection (Gap G7)."""
from __future__ import annotations

import pytest

from app.planning.information_asymmetry import (
    apply_dramatic_plan_reveals,
    compute_asymmetries,
    ripe_asymmetry_count,
)
from app.planning.schemas import (
    ActionResolution,
    DramaticPlan,
    DramaticScene,
)
from app.world.schema import (
    AsymmetryKind,
    Entity,
    EntityType,
    InformationState,
    KnowledgeEntry,
    TimelineEvent,
)
from app.world.state_manager import WorldStateManager


@pytest.fixture
def mgr(db):
    return WorldStateManager(db)


def _plan(scenes):
    return DramaticPlan(
        action_resolution=ActionResolution(kind="success", narrative="ok"),
        scenes=scenes,
        update_tension_target=0.5,
        ending_hook="hook",
        suggested_choices=[{"title": "c", "description": "", "tags": []}],
    )


def _scene(sid, reveals=(), pov="char_a", present=("char_a",)):
    return DramaticScene(
        scene_id=sid,
        pov_character_id=pov,
        characters_present=list(present),
        dramatic_question="Q?",
        outcome="ok",
        beats=["x"],
        dramatic_function="escalation",
        reveals=list(reveals),
    )


# ---- persistence ----


def test_upsert_and_roundtrip(mgr):
    st = InformationState(
        id="f1", quest_id="q1", fact="The king is dead", ground_truth=True,
        known_by={"reader": KnowledgeEntry(learned_at_update=3, learned_how="reveal")},
    )
    mgr.upsert_information_state(st)
    got = mgr.get_information_state("f1")
    assert got.fact == "The king is dead"
    assert "reader" in got.known_by
    assert got.known_by["reader"].learned_at_update == 3


def test_list_information_states_scoped_by_quest(mgr):
    mgr.upsert_information_state(InformationState(id="a", quest_id="q1", fact="x"))
    mgr.upsert_information_state(InformationState(id="b", quest_id="q2", fact="y"))
    assert [s.id for s in mgr.list_information_states("q1")] == ["a"]
    assert [s.id for s in mgr.list_information_states("q2")] == ["b"]


# ---- apply_dramatic_plan_reveals ----


def test_reveal_creates_fact_with_reader_and_pov(mgr):
    plan = _plan([_scene(1, reveals=["The captain is a traitor"])])
    apply_dramatic_plan_reveals(
        world=mgr, quest_id="q1", dramatic=plan, update_number=5,
    )
    states = mgr.list_information_states("q1")
    assert len(states) == 1
    s = states[0]
    assert s.fact == "The captain is a traitor"
    assert "reader" in s.known_by
    assert "char_a" in s.known_by
    assert s.known_by["reader"].learned_at_update == 5


def test_reveal_substring_matches_existing_fact(mgr):
    mgr.upsert_information_state(InformationState(
        id="f0", quest_id="q1",
        fact="The captain is a traitor to the crown",
    ))
    plan = _plan([_scene(1, reveals=["captain is a traitor"])])
    apply_dramatic_plan_reveals(
        world=mgr, quest_id="q1", dramatic=plan, update_number=2,
    )
    states = mgr.list_information_states("q1")
    assert len(states) == 1  # no duplicate row
    assert "reader" in states[0].known_by


def test_reveal_preserves_earliest_learning(mgr):
    plan1 = _plan([_scene(1, reveals=["Secret door exists"])])
    apply_dramatic_plan_reveals(world=mgr, quest_id="q1", dramatic=plan1, update_number=1)
    plan2 = _plan([_scene(2, reveals=["Secret door exists"])])
    apply_dramatic_plan_reveals(world=mgr, quest_id="q1", dramatic=plan2, update_number=4)
    s = mgr.list_information_states("q1")[0]
    # Should keep the first learning event
    assert s.known_by["reader"].learned_at_update == 1


# ---- compute_asymmetries ----


def _add_char(mgr, cid):
    mgr.create_entity(Entity(id=cid, entity_type=EntityType.CHARACTER, name=cid))


def test_dramatic_irony_reader_knows_char_doesnt(mgr):
    _add_char(mgr, "char_a")
    _add_char(mgr, "char_b")
    # Reader learns, char_a learns; char_b does not.
    st = InformationState(
        id="f1", quest_id="q1", fact="bomb is in cellar",
        known_by={
            "reader": KnowledgeEntry(learned_at_update=1),
            "char_a": KnowledgeEntry(learned_at_update=1),
        },
    )
    mgr.upsert_information_state(st)
    mgr.append_timeline_event(TimelineEvent(
        update_number=3, event_index=0, description="now",
    ))
    asyms = compute_asymmetries(mgr, "q1")
    kinds = {a.kind for a in asyms}
    assert AsymmetryKind.DRAMATIC_IRONY in kinds
    # should be a secret too (char_a knows, char_b doesn't)
    assert AsymmetryKind.SECRET in kinds
    di = [a for a in asyms if a.kind == AsymmetryKind.DRAMATIC_IRONY][0]
    assert "char_b" in di.unaware
    assert di.updates_standing == 2


def test_mystery_char_knows_reader_doesnt(mgr):
    _add_char(mgr, "char_a")
    st = InformationState(
        id="f1", quest_id="q1", fact="hidden parentage",
        known_by={
            "char_a": KnowledgeEntry(learned_at_update=2),
        },
    )
    mgr.upsert_information_state(st)
    asyms = compute_asymmetries(mgr, "q1", current_update=5)
    kinds = [a.kind for a in asyms]
    assert AsymmetryKind.MYSTERY in kinds
    assert AsymmetryKind.DRAMATIC_IRONY not in kinds


def test_secret_some_chars_know_others_dont(mgr):
    _add_char(mgr, "char_a")
    _add_char(mgr, "char_b")
    st = InformationState(
        id="f1", quest_id="q1", fact="the password is swordfish",
        known_by={
            "char_a": KnowledgeEntry(learned_at_update=1),
            "reader": KnowledgeEntry(learned_at_update=1),
        },
    )
    mgr.upsert_information_state(st)
    asyms = compute_asymmetries(mgr, "q1", current_update=2)
    secrets = [a for a in asyms if a.kind == AsymmetryKind.SECRET]
    assert len(secrets) == 1
    assert secrets[0].knowers == ["char_a"]
    assert secrets[0].unaware == ["char_b"]


def test_false_belief_when_believes_disagrees_with_ground_truth(mgr):
    _add_char(mgr, "char_a")
    st = InformationState(
        id="f1", quest_id="q1", fact="the heir is dead",
        ground_truth=False,  # the heir is actually alive
        known_by={
            "char_a": KnowledgeEntry(learned_at_update=1, believes=True),
        },
    )
    mgr.upsert_information_state(st)
    asyms = compute_asymmetries(mgr, "q1", current_update=3)
    fb = [a for a in asyms if a.kind == AsymmetryKind.FALSE_BELIEF]
    assert len(fb) == 1
    assert fb[0].believer_id == "char_a"


def test_ripe_asymmetry_count(mgr):
    _add_char(mgr, "char_a")
    st = InformationState(
        id="f1", quest_id="q1", fact="x",
        known_by={"reader": KnowledgeEntry(learned_at_update=1)},
    )
    mgr.upsert_information_state(st)
    asyms = compute_asymmetries(mgr, "q1", current_update=10)  # standing=9
    assert ripe_asymmetry_count(asyms, standing_threshold=3) >= 1


# ---- recommend_tools boost ----


def test_recommend_tools_gets_asymmetry_boost():
    from pathlib import Path
    from app.craft.library import CraftLibrary
    from app.craft.schemas import Arc

    ROOT = Path(__file__).parent.parent.parent / "app" / "craft" / "data"
    lib = CraftLibrary(ROOT)
    arc = Arc(id="a", name="a", scale="chapter", structure_id="three_act",
              current_phase_index=1, phase_progress=0.5,
              tension_observed=[(1, 0.5)])
    baseline = {t.id for t in lib.recommend_tools(
        arc, lib.structure("three_act"), limit=15, ripe_asymmetry_count=0,
    )}
    boosted = {t.id for t in lib.recommend_tools(
        arc, lib.structure("three_act"), limit=15, ripe_asymmetry_count=2,
    )}
    # Boost only adds score, so boosted set should be >= baseline.
    assert boosted >= baseline
