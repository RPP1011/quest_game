"""Day 2 unit tests for :class:`app.scoring.Scorer`.

Verifies:
- All 12 dims present and in [0, 1] for plausible prose.
- Graceful degradation: craft_plan / narrator / world / player_action
  may each be ``None`` without crashing; missing-input dims score 1.0.
- Empty prose doesn't raise.
- ``overall_score`` is the unweighted mean of the 12 dim scores.
"""
from __future__ import annotations

from statistics import fmean

import pytest

from app.scoring import DIMENSION_NAMES, Scorecard, Scorer


SAMPLE_PROSE = (
    "You stepped into the cold hall. The flagstones caught the lantern light "
    "and threw it back up at you, pale and uneven. Somewhere above, a door "
    "closed. \"Wait,\" a voice said. You stopped. The air smelled of iron "
    "and damp wool. Your heart hammered against your ribs. You turned toward "
    "the sound, one hand already reaching for the hilt at your hip."
)


def test_scorer_returns_twelve_dims_in_unit_interval():
    scorer = Scorer()
    card = scorer.score(SAMPLE_PROSE, player_action="Investigate the voice.")
    assert isinstance(card, Scorecard)
    for name in DIMENSION_NAMES:
        val = getattr(card, name)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"
    assert 0.0 <= card.overall_score <= 1.0
    # overall is unweighted mean of dims
    dim_values = [getattr(card, n) for n in DIMENSION_NAMES]
    assert card.overall_score == pytest.approx(fmean(dim_values), abs=1e-9)


def test_scorer_has_exactly_twelve_dimension_names():
    assert len(DIMENSION_NAMES) == 12
    assert len(set(DIMENSION_NAMES)) == 12  # no duplicates


def test_scorer_no_crash_with_all_optional_inputs_missing():
    scorer = Scorer()
    card = scorer.score(SAMPLE_PROSE)  # no craft_plan, no narrator, no world, no action
    assert isinstance(card, Scorecard)
    # craft-plan-dependent critics should score 1.0 (no violations detectable)
    assert card.free_indirect_quality == 1.0
    assert card.detail_characterization == 1.0
    assert card.metaphor_domains_score == 1.0
    assert card.indirection_score == 1.0
    # action_fidelity also score 1.0 when player_action missing
    assert card.action_fidelity == 1.0
    # named_entity_presence score 1.0 when world missing
    assert card.named_entity_presence == 1.0
    # narrator_sensory_match 1.0 when narrator None
    assert card.narrator_sensory_match == 1.0


def test_scorer_handles_empty_prose_without_raising():
    scorer = Scorer()
    card = scorer.score("")
    assert isinstance(card, Scorecard)
    # Heuristic dims are 0.0 on empty text (no words, no sentences).
    assert card.sentence_variance == 0.0
    assert card.dialogue_ratio == 0.0
    assert card.pacing == 0.0
    assert card.sensory_density == 0.0


def test_scorer_pov_drift_penalises_first_person():
    """Prose that collapses to first person should score low on pov_adherence."""
    scorer = Scorer()
    first_person = (
        "I walked into the cold hall. I felt the flagstones under my boots. "
        "I heard a voice. I stopped, and I waited for it to call again."
    )
    card = scorer.score(first_person)
    # Default critic threshold: min_ratio=0.7 for second-person.
    # All-first-person ratio is 0, which is < 0.7 -> one warning -> 0.9.
    assert card.pov_adherence < 1.0


def test_scorer_detects_named_entity_presence_when_world_supplies_names():
    scorer = Scorer()

    class _Entity:
        def __init__(self, name: str) -> None:
            self.name = name

    class _World:
        def list_entities(self):
            return [_Entity("Rook"), _Entity("Elena")]

    prose_with_entity = "Rook crossed the threshold. The hall was silent."
    card = scorer.score(prose_with_entity, world=_World())
    assert card.named_entity_presence == 1.0  # critic emits no issues -> clean


def test_scorer_flags_missing_named_entity():
    scorer = Scorer()

    class _Entity:
        def __init__(self, name: str) -> None:
            self.name = name

    class _World:
        def list_entities(self):
            return [_Entity("Rook"), _Entity("Elena")]

    prose_no_entity = "The hall was silent. The stones were cold underfoot."
    card = scorer.score(prose_no_entity, world=_World())
    # critic emits one warning -> score = 1.0 - 0.10 = 0.9
    assert card.named_entity_presence < 1.0


def test_scorer_action_fidelity_rewards_overlap_and_penalises_miss():
    scorer = Scorer()
    prose = (
        "You examined the carving on the altar stone. Your fingers traced "
        "the deep grooves. It was older than the chapel itself."
    )
    hit_card = scorer.score(prose, player_action="Examine the carving on the altar.")
    miss_card = scorer.score(prose, player_action="Search for a hidden mechanism.")
    assert hit_card.action_fidelity >= miss_card.action_fidelity


def test_scorecard_dimension_items_order_matches_dimension_names():
    scorer = Scorer()
    card = scorer.score(SAMPLE_PROSE)
    names = [n for n, _ in card.dimension_items()]
    assert tuple(names) == DIMENSION_NAMES
