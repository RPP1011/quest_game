"""Day 2: post-commit scoring hook in :class:`app.engine.pipeline.Pipeline`.

The hook must:
- Stay silent (no scorecard rows, no ``scoring`` trace stage) when the
  ``scorer`` kwarg is absent (back-compat).
- Stay silent when a ``scorer`` is wired but
  ``quest_config["scoring"]["enabled"]`` is falsy (default-off).
- Stay silent when no ``quest_id`` is available (scorecards are quest-scoped).
- Fire once per committed chapter when all three gates are open, writing a
  scorecard row (12 dimension rows) and logging a ``scoring`` stage.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.scoring import DIMENSION_NAMES, Scorer
from app.world import (
    ArcPosition, Entity, EntityType, PlotThread, StateDelta, WorldStateManager,
)
from app.world.db import open_db
from app.world.delta import EntityCreate


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


_EMPTY_EXTRACT = (
    '{"entity_updates":[],"new_relationships":[],"removed_relationships":[],'
    '"timeline_events":[],"foreshadowing_updates":[]}'
)


class FakeClient:
    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        if schema_name == "StateDelta":
            return _EMPTY_EXTRACT
        # BeatSheet
        return '{"beats": ["Step into the hall."], "suggested_choices": ["Proceed", "Wait"]}'

    async def chat(self, *, messages, **kw):
        return (
            "You stepped into the cold hall. The lanterns threw long shadows "
            "across the stones. Somewhere above, a door closed. \"Wait,\" a "
            "voice said, quiet and even. You did not move."
        )


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(
        entity_creates=[EntityCreate(entity=Entity(
            id="rook", entity_type=EntityType.CHARACTER, name="Rook",
            data={"description": "A figure in the hall."},
        ))],
    ), update_number=1)
    sm.add_plot_thread(PlotThread(
        id="pt:1", name="Arrival", description="A stranger enters.",
        arc_position=ArcPosition.RISING,
    ))
    yield sm
    conn.close()


def _count_scorecards(world: WorldStateManager, quest_id: str) -> int:
    return len(world.list_scorecards(quest_id))


async def test_no_scorecard_when_scorer_not_wired(world):
    """Back-compat: existing pipelines without a ``scorer`` kwarg must not
    change behavior and must not write scorecards."""
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeClient(),
        quest_id="q-demo",
        quest_config={"scoring": {"enabled": True}},  # flag on, but no scorer
    )
    out = await pipeline.run(player_action="Enter the hall.", update_number=2)
    assert out.trace.outcome == "committed"
    assert _count_scorecards(world, "q-demo") == 0
    assert "scoring" not in {s.stage_name for s in out.trace.stages}


async def test_no_scorecard_when_flag_disabled(world):
    """Scorer wired but flag off ⇒ no scorecard and no ``scoring`` stage."""
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeClient(),
        quest_id="q-demo",
        scorer=Scorer(),
        quest_config={"scoring": {"enabled": False}},
    )
    out = await pipeline.run(player_action="Enter the hall.", update_number=2)
    assert out.trace.outcome == "committed"
    assert _count_scorecards(world, "q-demo") == 0
    assert "scoring" not in {s.stage_name for s in out.trace.stages}


async def test_no_scorecard_when_quest_id_missing(world):
    """Scorecards are quest-scoped; no quest_id ⇒ skip silently."""
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeClient(),
        scorer=Scorer(),
        quest_config={"scoring": {"enabled": True}},
        # no quest_id
    )
    out = await pipeline.run(player_action="Enter the hall.", update_number=2)
    assert out.trace.outcome == "committed"
    # no quest to query by, but assert the trace stage was skipped
    assert "scoring" not in {s.stage_name for s in out.trace.stages}


async def test_scorecard_persisted_when_scorer_and_flag_enabled(world):
    """Scorer + flag on + quest_id ⇒ exactly one scorecard row per commit."""
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeClient(),
        quest_id="q-demo",
        scorer=Scorer(),
        quest_config={"scoring": {"enabled": True}},
    )
    out = await pipeline.run(player_action="Enter the hall.", update_number=2)
    assert out.trace.outcome == "committed"

    cards = world.list_scorecards("q-demo")
    assert len(cards) == 1
    card = cards[0]
    for name in DIMENSION_NAMES:
        val = getattr(card, name)
        assert 0.0 <= val <= 1.0
    assert 0.0 <= card.overall_score <= 1.0

    # Scoring stage recorded in the trace.
    scoring_stages = [s for s in out.trace.stages if s.stage_name == "scoring"]
    assert len(scoring_stages) == 1
    assert "overall_score" in scoring_stages[0].detail
    assert "dimensions" in scoring_stages[0].detail
    assert set(scoring_stages[0].detail["dimensions"].keys()) == set(DIMENSION_NAMES)


async def test_scorecard_trace_id_links_to_pipeline_trace(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeClient(),
        quest_id="q-demo",
        scorer=Scorer(),
        quest_config={"scoring": {"enabled": True}},
    )
    out = await pipeline.run(player_action="Enter.", update_number=2)
    row = world._conn.execute(
        "SELECT pipeline_trace_id FROM scorecards WHERE quest_id=?",
        ("q-demo",),
    ).fetchone()
    assert row["pipeline_trace_id"] == out.trace.trace_id
