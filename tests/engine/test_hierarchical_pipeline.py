"""Tests for the hierarchical (4-layer) pipeline flow.

The key invariant: Pipeline with all four hierarchical planners wired in runs
arc → dramatic → emotional → craft → write (per scene) → craft_critics →
check → extract.

Pipeline WITHOUT the hierarchical deps falls back to the flat plan → write →
check → extract flow (existing contract, preserved in other test modules).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline, PipelineOutput
from app.world import Entity, EntityType, StateDelta, WorldStateManager
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"

_EMPTY_EXTRACT = json.dumps({
    "entity_updates": [],
    "new_relationships": [],
    "removed_relationships": [],
    "timeline_events": [],
    "foreshadowing_updates": [],
})

_CHECK_CLEAN = '{"issues": []}'

# ---------------------------------------------------------------------------
# Canned JSON payloads for each hierarchical layer
# ---------------------------------------------------------------------------

_ARC_JSON = json.dumps({
    "current_phase": "rising",
    "phase_assessment": "The quest is building tension.",
    "theme_priorities": [],
    "plot_objectives": [{"description": "Find the artefact", "urgency": "this_phase"}],
    "character_arcs": [],
    "tension_range": [0.3, 0.7],
    "hooks_to_plant": [],
    "hooks_to_pay_off": [],
    "parallels_to_schedule": [],
})

_DRAMATIC_JSON = json.dumps({
    "action_resolution": {"kind": "success", "narrative": "The hero presses forward."},
    "scenes": [
        {
            "scene_id": 1,
            "dramatic_question": "Will the hero cross the bridge?",
            "outcome": "The hero crosses.",
            "beats": ["Hero approaches.", "Hero crosses."],
            "dramatic_function": "escalation",
        }
    ],
    "ending_hook": "A shadow watches from the trees.",
    "suggested_choices": [
        {"title": "Investigate the shadow", "description": "Risky.", "tags": ["danger"]},
        {"title": "Keep moving", "description": "Pragmatic.", "tags": []},
    ],
})

_EMOTIONAL_JSON = json.dumps({
    "scenes": [
        {
            "scene_id": 1,
            "primary_emotion": "anxiety",
            "intensity": 0.6,
            "entry_state": "wary",
            "exit_state": "resolute",
            "transition_type": "escalation",
            "emotional_source": "the unknown threat",
        }
    ],
    "update_emotional_arc": "anxiety → resolve",
    "contrast_strategy": "quiet before the storm",
})

_CRAFT_JSON = json.dumps({
    "scenes": [
        {
            "scene_id": 1,
            "temporal": {"description": "linear present-scene"},
            "register": {
                "sentence_variance": "medium",
                "concrete_abstract_ratio": 0.6,
                "interiority_depth": "medium",
                "sensory_density": "moderate",
                "dialogue_ratio": 0.3,
                "pace": "measured",
            },
        }
    ],
    "briefs": [
        {
            "scene_id": 1,
            "brief": "Write tense, grounded prose as the hero crosses. Ground in physical sensation.",
        }
    ],
})

_PROSE_SCENE_1 = "The bridge groaned underfoot. She crossed in three quick strides, heart in her throat."


# ---------------------------------------------------------------------------
# Scripted client for the hierarchical flow
# ---------------------------------------------------------------------------


class ScriptedClient:
    """Replay scripted responses in order, asserting kind matches."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.log: list[str] = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str:
        r = self._responses.pop(0)
        assert r["kind"] == "structured", (
            f"Expected structured call (schema_name={schema_name!r}), but next response is {r!r}"
        )
        self.log.append(f"structured:{schema_name}")
        return r["content"]

    async def chat(self, *, messages, **kw) -> str:
        r = self._responses.pop(0)
        assert r["kind"] == "chat", (
            f"Expected chat call, but next response is {r!r}"
        )
        self.log.append("chat")
        return r["content"]


# ---------------------------------------------------------------------------
# Fake planners that delegate to the scripted client
# ---------------------------------------------------------------------------


class FakeArcPlanner:
    def __init__(self, client: ScriptedClient) -> None:
        self._client = client

    async def plan(self, *, quest_config, arc_state, world_snapshot, structure):
        from app.planning.schemas import ArcDirective
        raw = await self._client.chat_structured(
            messages=[], json_schema={}, schema_name="ArcDirective"
        )
        from app.world.output_parser import OutputParser
        return OutputParser.parse_json(raw, schema=ArcDirective)


class FakeDramaticPlanner:
    def __init__(self, client: ScriptedClient) -> None:
        self._client = client

    async def plan(self, *, directive, player_action, world, arc, structure,
                   recent_tool_ids=None, quest_id=None):
        from app.planning.schemas import DramaticPlan
        raw = await self._client.chat_structured(
            messages=[], json_schema={}, schema_name="DramaticPlan"
        )
        from app.world.output_parser import OutputParser
        return OutputParser.parse_json(raw, schema=DramaticPlan)


class FakeEmotionalPlanner:
    def __init__(self, client: ScriptedClient) -> None:
        self._client = client

    async def plan(self, *, dramatic, world, recent_prose):
        from app.planning.schemas import EmotionalPlan
        raw = await self._client.chat_structured(
            messages=[], json_schema={}, schema_name="EmotionalPlan"
        )
        from app.world.output_parser import OutputParser
        return OutputParser.parse_json(raw, schema=EmotionalPlan)


class FakeCraftPlanner:
    def __init__(self, client: ScriptedClient) -> None:
        self._client = client

    async def plan(self, *, dramatic, emotional, style_register_id=None):
        from app.planning.schemas import CraftPlan
        raw = await self._client.chat_structured(
            messages=[], json_schema={}, schema_name="CraftPlan"
        )
        from app.world.output_parser import OutputParser
        return OutputParser.parse_json(raw, schema=CraftPlan)


class FakeCraftLibrary:
    """Minimal stub craft library for testing."""

    def tools(self, category=None):
        return []

    def recommend_tools(self, arc, structure, recent_tool_ids=None, limit=5):
        return []

    def examples_for_tool(self, tool_id):
        return []

    def style(self, style_id):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(
            id="hero", entity_type=EntityType.CHARACTER, name="Hero",
        )),
    ]), update_number=1)
    yield sm
    conn.close()


def _cb(world: WorldStateManager) -> ContextBuilder:
    return ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_hierarchical_pipeline_commits_chapter(world):
    """Hierarchical pipeline runs all layers and commits a chapter.

    We do not wire arc_planner (it is optional) so the pipeline synthesises a
    minimal ArcDirective without consuming a scripted response. The four core
    layers (dramatic, emotional, craft, write) plus check and extract are
    scripted in order.
    """
    scripted = ScriptedClient([
        # DramaticPlanner call
        {"kind": "structured", "content": _DRAMATIC_JSON},
        # EmotionalPlanner call
        {"kind": "structured", "content": _EMOTIONAL_JSON},
        # CraftPlanner call
        {"kind": "structured", "content": _CRAFT_JSON},
        # WRITE (one scene)
        {"kind": "chat", "content": _PROSE_SCENE_1},
        # CHECK
        {"kind": "structured", "content": _CHECK_CLEAN},
        # EXTRACT
        {"kind": "structured", "content": _EMPTY_EXTRACT},
    ])

    dramatic_planner = FakeDramaticPlanner(scripted)
    emotional_planner = FakeEmotionalPlanner(scripted)
    craft_planner = FakeCraftPlanner(scripted)
    craft_library = FakeCraftLibrary()

    pipeline = Pipeline(
        world, _cb(world), scripted,
        dramatic_planner=dramatic_planner,
        emotional_planner=emotional_planner,
        craft_planner=craft_planner,
        craft_library=craft_library,
    )

    assert pipeline.is_hierarchical

    result = await pipeline.run(player_action="Cross the bridge.", update_number=2)

    assert isinstance(result, PipelineOutput)
    assert _PROSE_SCENE_1 in result.prose
    assert result.trace.outcome == "committed"

    # Narrative must be persisted
    records = world.list_narrative()
    assert len(records) == 1
    assert records[0].player_action == "Cross the bridge."

    stage_names = [s.stage_name for s in result.trace.stages]

    # All expected stages present in order
    assert "arc" in stage_names
    assert "dramatic" in stage_names
    assert "emotional" in stage_names
    assert "craft" in stage_names
    assert "write" in stage_names
    assert "craft_critics" in stage_names
    assert "check" in stage_names
    assert "extract" in stage_names

    # Ordering: arc before dramatic before emotional before craft before write
    arc_i = stage_names.index("arc")
    dramatic_i = stage_names.index("dramatic")
    emotional_i = stage_names.index("emotional")
    craft_i = stage_names.index("craft")
    write_i = stage_names.index("write")
    check_i = stage_names.index("check")
    extract_i = stage_names.index("extract")

    assert arc_i < dramatic_i < emotional_i < craft_i < write_i < check_i < extract_i

    # Choices come from the dramatic plan
    assert result.choices == [
        {"title": "Investigate the shadow", "description": "Risky.", "tags": ["danger"]},
        {"title": "Keep moving", "description": "Pragmatic.", "tags": []},
    ]


async def test_hierarchical_falls_back_to_flat_when_not_configured(world):
    """Pipeline without hierarchical deps uses the old flat flow."""
    scripted = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["A beat."], "suggested_choices": ["Go left"]}'},
        {"kind": "chat", "content": "Flat prose."},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": _EMPTY_EXTRACT},
    ])

    pipeline = Pipeline(world, _cb(world), scripted)

    assert not pipeline.is_hierarchical

    result = await pipeline.run(player_action="Look around.", update_number=2)

    assert "Flat prose." in result.prose
    assert result.trace.outcome == "committed"

    stage_names = [s.stage_name for s in result.trace.stages]
    assert stage_names == ["plan", "write", "check", "extract"]
    # No hierarchical stages
    assert "dramatic" not in stage_names
    assert "emotional" not in stage_names
    assert "craft" not in stage_names


async def test_hierarchical_is_hierarchical_requires_all_four_planners(world):
    """is_hierarchical is True only when all four layers are provided."""
    cb = _cb(world)
    cl = FakeCraftLibrary()
    scripted = ScriptedClient([])

    # Missing craft_library
    p1 = Pipeline(world, cb, scripted,
                  dramatic_planner=object(),
                  emotional_planner=object(),
                  craft_planner=object())
    assert not p1.is_hierarchical

    # Missing emotional_planner
    p2 = Pipeline(world, cb, scripted,
                  dramatic_planner=object(),
                  craft_planner=object(),
                  craft_library=cl)
    assert not p2.is_hierarchical

    # Missing craft_planner
    p3 = Pipeline(world, cb, scripted,
                  dramatic_planner=object(),
                  emotional_planner=object(),
                  craft_library=cl)
    assert not p3.is_hierarchical

    # All four present
    p4 = Pipeline(world, cb, scripted,
                  dramatic_planner=object(),
                  emotional_planner=object(),
                  craft_planner=object(),
                  craft_library=cl)
    assert p4.is_hierarchical
