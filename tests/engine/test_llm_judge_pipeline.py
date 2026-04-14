"""Day 6 pipeline integration tests.

Verifies that when a ``llm_judge_client`` is wired onto
:class:`app.engine.pipeline.Pipeline`, the post-commit hook schedules an
async task (``asyncio.create_task``) that writes the three extra Day 6
dim rows onto the same scorecard produced by the Day 2 sync path.

Default-off behavior is explicitly asserted: a pipeline constructed
without ``llm_judge_client`` must behave bit-identically to its Day 2
counterpart (no extra dim rows, no leaked tasks).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.scoring import LLM_JUDGE_DIMS, Scorer
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


class FakeWriterClient:
    """Drives the main pipeline (plan + write + extract)."""

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        if schema_name == "StateDelta":
            return _EMPTY_EXTRACT
        # BeatSheet
        return (
            '{"beats": ["Step into the hall."], '
            '"suggested_choices": ["Proceed", "Wait"]}'
        )

    async def chat(self, *, messages, **kw):
        return (
            "You stepped into the cold hall. The lanterns threw long "
            "shadows across the stones. Somewhere above, a door closed. "
            "\"Wait,\" a voice said, quiet and even. You did not move."
        )


class FakeJudgeClient:
    """Separate client dedicated to the Day 6 batched-judge call.

    Captures every structured call so we can assert the schema shape.
    """

    def __init__(self, canned_scores: dict[str, float]) -> None:
        self.canned = canned_scores
        self.call_count = 0
        self.last_schema: dict | None = None

    async def chat_structured(
        self, *, messages, json_schema, schema_name,
        temperature=0.2, max_tokens=2000, thinking=False,
    ) -> str:
        self.call_count += 1
        self.last_schema = json_schema
        return json.dumps({
            d: {"score": v, "rationale": f"stub-{d}"}
            for d, v in self.canned.items()
        })


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


async def test_llm_judge_task_writes_extra_dim_rows(world):
    """With judge client wired, 3 extra dim rows land on the scorecard."""
    canned = {
        "tension_execution": 0.72,
        "emotional_trajectory": 0.55,
        "choice_hook_quality": 0.80,
    }
    judge_client = FakeJudgeClient(canned)
    scorer = Scorer(llm_judge_client=judge_client)
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeWriterClient(),
        quest_id="q-demo",
        scorer=scorer,
        llm_judge_client=judge_client,
        quest_config={"scoring": {"enabled": True}},
    )
    out = await pipeline.run(player_action="Enter the hall.", update_number=2)
    assert out.trace.outcome == "committed"

    # A task was scheduled — await it to let the rows land.
    assert pipeline.last_llm_judge_task is not None
    await pipeline.last_llm_judge_task

    # Exactly one judge call was made (single batched structured call).
    assert judge_client.call_count == 1

    # Fetch the scorecard row and verify the extra dims are present.
    row = world._conn.execute(
        "SELECT id FROM scorecards WHERE quest_id=? ORDER BY id DESC LIMIT 1",
        ("q-demo",),
    ).fetchone()
    assert row is not None
    dims = world.list_dimension_scores(row["id"])
    for name, expected in canned.items():
        assert name in dims
        assert dims[name] == pytest.approx(expected)


async def test_no_extra_rows_without_judge_client(world):
    """Pipeline without judge client must not schedule a task or add rows."""
    scorer = Scorer()
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeWriterClient(),
        quest_id="q-demo",
        scorer=scorer,
        quest_config={"scoring": {"enabled": True}},
    )
    out = await pipeline.run(player_action="Enter the hall.", update_number=2)
    assert out.trace.outcome == "committed"
    assert pipeline.last_llm_judge_task is None

    # Only Day 2 dim rows — no Day 6 names.
    row = world._conn.execute(
        "SELECT id FROM scorecards WHERE quest_id=? ORDER BY id DESC LIMIT 1",
        ("q-demo",),
    ).fetchone()
    dims = world.list_dimension_scores(row["id"])
    # tension_execution and choice_hook_quality are NOT Day 2 dims, so they
    # must be absent. (emotional_trajectory was never on Day 2 either.)
    for day6_only in ("tension_execution", "emotional_trajectory",
                      "choice_hook_quality"):
        assert day6_only not in dims


async def test_judge_task_does_not_block_commit(world):
    """Commit returns before the slow judge task completes."""
    canned = {d: 0.5 for d in LLM_JUDGE_DIMS}

    class SlowJudge:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.released = asyncio.Event()

        async def chat_structured(self, *, messages, json_schema,
                                  schema_name, **kw):
            self.started.set()
            await self.released.wait()
            return json.dumps({
                d: {"score": v, "rationale": "stub"} for d, v in canned.items()
            })

    judge_client = SlowJudge()
    scorer = Scorer(llm_judge_client=judge_client)
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeWriterClient(),
        quest_id="q-demo",
        scorer=scorer,
        llm_judge_client=judge_client,
        quest_config={"scoring": {"enabled": True}},
    )
    # Run commit; the slow judge task holds its own lock.
    out = await pipeline.run(player_action="Enter.", update_number=2)
    assert out.trace.outcome == "committed"

    # Verify the task is still pending (we held `released` the whole time).
    task = pipeline.last_llm_judge_task
    assert task is not None
    # Yield to the event loop so the scheduled task can begin running. The
    # commit path wraps score_with_llm_judges in ``asyncio.create_task`` —
    # the task exists but is not guaranteed to have advanced past its first
    # await point until we yield.
    await asyncio.sleep(0)
    assert not task.done()
    # And the judge got started (hit the ``await released`` inside).
    assert judge_client.started.is_set()

    # Release & await so pytest doesn't complain about a dangling task.
    judge_client.released.set()
    await task


async def test_judge_failure_does_not_rollback_scorecard(world):
    """A crash inside the async judge task must not lose the Day 2 rows."""
    class CrashingJudge:
        async def chat_structured(self, **kw):
            raise RuntimeError("judge upstream down")

    judge_client = CrashingJudge()
    scorer = Scorer(llm_judge_client=judge_client)
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(
        world, cb, FakeWriterClient(),
        quest_id="q-demo",
        scorer=scorer,
        llm_judge_client=judge_client,
        quest_config={"scoring": {"enabled": True}},
    )
    out = await pipeline.run(player_action="Enter.", update_number=2)
    assert out.trace.outcome == "committed"

    assert pipeline.last_llm_judge_task is not None
    # Task completes without raising (we catch-and-log inside the pipeline).
    await pipeline.last_llm_judge_task

    # Day 2 dim rows are still intact.
    cards = world.list_scorecards("q-demo")
    assert len(cards) == 1
    card = cards[0]
    assert 0.0 <= card.overall_score <= 1.0
