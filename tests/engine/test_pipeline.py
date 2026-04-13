from pathlib import Path
import pytest
from app.engine import ContextBuilder, PLAN_SPEC, PromptRenderer, TokenBudget, WRITE_SPEC
from app.engine.pipeline import Pipeline, PipelineOutput, _normalize_beat_sheet


def test_normalize_accepts_beats_key():
    assert _normalize_beat_sheet({"beats": ["a", "b"], "suggested_choices": ["x"]}) == {
        "beats": ["a", "b"],
        "suggested_choices": [{"title": "x", "description": "", "tags": []}],
    }


def test_normalize_camelcase_and_snakecase_aliases():
    assert _normalize_beat_sheet({"beatSheet": ["a"], "choices": ["x"]}) == {
        "beats": ["a"],
        "suggested_choices": [{"title": "x", "description": "", "tags": []}],
    }
    assert _normalize_beat_sheet({"beat_sheet": ["a"]}) == {
        "beats": ["a"], "suggested_choices": [],
    }


def test_normalize_list_of_dicts_extracts_beat_field():
    result = _normalize_beat_sheet({"beats": [
        {"beat": "Intro", "details": "x"},
        {"text": "Climb the stairs"},
    ]})
    assert result["beats"] == ["Intro", "Climb the stairs"]


def test_normalize_fallback_scans_any_list():
    assert _normalize_beat_sheet({"unknown": ["step 1", "step 2"]})["beats"] == ["step 1", "step 2"]


def test_normalize_nested_beat_sheet_with_key_actions():
    data = {"beat_sheet": {
        "objective": "x",
        "key_actions": ["Approach.", "Retreat."],
        "suggested_choices": ["Approach.", "Retreat."],
    }}
    r = _normalize_beat_sheet(data)
    assert r["beats"] == ["Approach.", "Retreat."]
    assert r["suggested_choices"] == [
        {"title": "Approach.", "description": "", "tags": []},
        {"title": "Retreat.", "description": "", "tags": []},
    ]


def test_normalize_object_choices_preserved():
    """Object choices keep their title, description, and tags."""
    data = {
        "beats": ["Scene starts."],
        "suggested_choices": [
            {"title": "Go left.", "description": "A dark path.", "tags": ["stealth"]},
        ],
    }
    r = _normalize_beat_sheet(data)
    assert r["suggested_choices"] == [
        {"title": "Go left.", "description": "A dark path.", "tags": ["stealth"]},
    ]


def test_normalize_string_choices_coerced():
    """Plain string choices become dicts with empty description and tags."""
    data = {
        "beats": ["Setup."],
        "suggested_choices": ["Run away.", "Stand firm."],
    }
    r = _normalize_beat_sheet(data)
    assert r["suggested_choices"] == [
        {"title": "Run away.", "description": "", "tags": []},
        {"title": "Stand firm.", "description": "", "tags": []},
    ]


def test_normalize_mixed_choice_list():
    """Mixed string/object choice lists are handled correctly."""
    data = {
        "beats": ["Setup."],
        "suggested_choices": [
            "Flee.",
            {"title": "Negotiate.", "description": "Talk it out.", "tags": ["diplomacy"]},
        ],
    }
    r = _normalize_beat_sheet(data)
    assert r["suggested_choices"] == [
        {"title": "Flee.", "description": "", "tags": []},
        {"title": "Negotiate.", "description": "Talk it out.", "tags": ["diplomacy"]},
    ]


from app.world import (
    Entity, EntityType, PlotThread, ArcPosition,
    StateDelta, WorldStateManager,
)
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


_EMPTY_EXTRACT = '{"entity_updates":[],"new_relationships":[],"removed_relationships":[],"timeline_events":[],"foreshadowing_updates":[]}'


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        self.calls.append(("structured", messages, json_schema))
        if schema_name == "StateDelta":
            return _EMPTY_EXTRACT
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
    assert result.choices == [
        {"title": "Ask who they are", "description": "", "tags": []},
        {"title": "Leave", "description": "", "tags": []},
    ]
    assert [s.stage_name for s in result.trace.stages] == ["plan", "write", "check", "extract"]
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


async def test_pipeline_synthesizes_plan_when_beats_missing(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())

    class BeatlessClient(FakeClient):
        async def chat_structured(self, **kw):
            if kw.get("schema_name") == "BeatSheet":
                return '{"beat_sheet": "one long sentence that is not a list"}'
            return await super().chat_structured(**kw)

    p = Pipeline(world, cb, BeatlessClient())
    out = await p.run(player_action="Look around.", update_number=2)
    # Synthetic fallback keeps the chapter alive instead of 500-ing.
    assert out.prose
    plan_stage = next(s for s in out.trace.stages if s.stage_name == "plan")
    assert plan_stage.errors and plan_stage.errors[0].kind == "parse_warning"
    assert plan_stage.parsed_output["beats"] == ["React naturally to: Look around."]


async def test_pipeline_surfaces_parse_error(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())

    class BadClient(FakeClient):
        async def chat_structured(self, **kw):
            return "not json at all"

    p = Pipeline(world, cb, BadClient())
    with pytest.raises(Exception):
        await p.run(player_action="x", update_number=2)
