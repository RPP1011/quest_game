from pathlib import Path
import pytest
from app.engine import ContextBuilder, PLAN_SPEC, PromptRenderer, TokenBudget, WRITE_SPEC
from app.engine.pipeline import Pipeline, PipelineOutput
from app.world import (
    Entity, EntityType, PlotThread, ArcPosition,
    StateDelta, WorldStateManager,
)
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        self.calls.append(("structured", messages, json_schema))
        return '{"beats": ["Alice greets Bob."], "suggested_choices": ["Ask who they are", "Leave"]}'

    async def chat(self, *, messages, **kw):
        self.calls.append(("chat", messages))
        return "Alice looked up from the bar. \"Can I help you?\" she asked."


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(
        entity_creates=[EntityCreate(entity=Entity(
            id="alice", entity_type=EntityType.CHARACTER, name="Alice",
            data={"description": "The tavern keeper."}))],
    ), update_number=1)
    sm.add_plot_thread(PlotThread(
        id="pt:1", name="Arrival", description="A stranger enters.",
        arc_position=ArcPosition.RISING,
    ))
    yield sm
    conn.close()


async def test_pipeline_runs_plan_and_write(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    fake = FakeClient()
    p = Pipeline(world, cb, fake)
    result = await p.run(player_action="Greet the stranger.", update_number=2)
    assert isinstance(result, PipelineOutput)
    assert "Alice looked up" in result.prose
    assert result.choices == ["Ask who they are", "Leave"]
    assert [s.stage_name for s in result.trace.stages] == ["plan", "write"]
    assert result.trace.outcome == "committed"


async def test_pipeline_persists_narrative(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    fake = FakeClient()
    p = Pipeline(world, cb, fake)
    await p.run(player_action="Greet.", update_number=2)
    records = world.list_narrative()
    assert len(records) == 1
    assert records[0].player_action == "Greet."
    assert "Alice looked up" in records[0].raw_text


async def test_pipeline_surfaces_parse_error(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())

    class BadClient(FakeClient):
        async def chat_structured(self, **kw):
            return "not json at all"

    p = Pipeline(world, cb, BadClient())
    with pytest.raises(Exception):
        await p.run(player_action="x", update_number=2)
