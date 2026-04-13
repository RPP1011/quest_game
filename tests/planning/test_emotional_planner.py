"""Tests for app/planning/emotional_planner.py — EmotionalPlanner."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.engine.prompt_renderer import PromptRenderer
from app.planning.emotional_planner import EmotionalPlanner
from app.planning.schemas import (
    ActionResolution,
    DramaticPlan,
    DramaticScene,
    EmotionalPlan,
    EmotionalScenePlan,
    CharacterEmotionalState,
)
from app.world.db import open_db
from app.world.output_parser import ParseError
from app.world.schema import Entity, EntityType
from app.world.state_manager import WorldStateManager

PROMPTS = Path(__file__).parent.parent.parent / "prompts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DRAMATIC_PLAN = DramaticPlan(
    action_resolution=ActionResolution(
        kind="partial",
        narrative="The hero recovers the stolen key but is spotted by a guard.",
    ),
    scenes=[
        DramaticScene(
            scene_id=1,
            dramatic_question="Can the hero retrieve the key without being caught?",
            outcome="Key retrieved; guard raises alarm.",
            beats=["Hero sneaks in", "Finds the key", "Guard turns around"],
            dramatic_function="inciting escalation",
            pov_character_id="hero",
            characters_present=["hero", "guard"],
            reveals=["The key opens a hidden passage."],
            withholds=["The guard recognises Aldric from years ago."],
        ),
        DramaticScene(
            scene_id=2,
            dramatic_question="Will the hero escape the manor grounds?",
            outcome="Hero escapes through the garden, barely.",
            beats=["Alarm sounds", "Hero runs", "Clears the wall"],
            dramatic_function="rising tension",
            pov_character_id="hero",
            characters_present=["hero"],
        ),
    ],
    update_tension_target=0.65,
    ending_hook="The alarm will draw Lord Maren himself.",
    suggested_choices=[
        {"title": "Flee", "description": "Run now.", "tags": ["action"]},
    ],
)

_VALID_EMOTIONAL_PLAN = EmotionalPlan(
    scenes=[
        EmotionalScenePlan(
            scene_id=1,
            primary_emotion="dread",
            secondary_emotion="determination",
            intensity=0.6,
            entry_state="cautious hope",
            exit_state="panicked resolve",
            transition_type="escalation",
            emotional_source="The guard's proximity turns abstract risk into immediate danger.",
            surface_vs_depth="Surface: focused calm. Depth: terror of being recognised.",
            character_emotions={
                "hero": CharacterEmotionalState(
                    internal="terror",
                    displayed="controlled focus",
                    gap="hero is trained to mask fear",
                )
            },
        ),
        EmotionalScenePlan(
            scene_id=2,
            primary_emotion="relief",
            intensity=0.4,
            entry_state="panicked resolve",
            exit_state="shaky relief",
            transition_type="subsidence",
            emotional_source="Escape restores safety but leaves residual adrenaline.",
        ),
    ],
    update_emotional_arc="Dread peaks in scene 1, subsides to fragile relief in scene 2.",
    contrast_strategy="Cut from contained terror to wide-open flight to emphasise release.",
)


class FakeClient:
    """Records calls; returns a canned JSON response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple] = []

    async def chat_structured(
        self, messages, *, json_schema, schema_name="Output", **kw
    ) -> str:
        self.calls.append((messages, json_schema, schema_name))
        return self._response


def _make_world(tmp_path: Path) -> WorldStateManager:
    conn = open_db(tmp_path / "w.db")
    wsm = WorldStateManager(conn)
    wsm.create_entity(Entity(
        id="hero",
        entity_type=EntityType.CHARACTER,
        name="Aldric",
        data={"description": "A young thief with a heart of gold."},
    ))
    wsm.create_entity(Entity(
        id="guard",
        entity_type=EntityType.CHARACTER,
        name="Guard Captain",
        data={"description": "A veteran soldier loyal to Lord Maren."},
    ))
    return wsm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emotional_planner_returns_plan(tmp_path):
    """FakeClient returning valid JSON → plan() returns an EmotionalPlan."""
    raw = _VALID_EMOTIONAL_PLAN.model_dump_json()
    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    world = _make_world(tmp_path)

    planner = EmotionalPlanner(client, renderer)
    result = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        world=world,
        recent_prose=[
            "Aldric slipped through the market crowd.",
            "The guildmaster spread the stolen deed across the table.",
        ],
    )

    assert isinstance(result, EmotionalPlan)
    assert len(result.scenes) == 2
    assert result.scenes[0].primary_emotion == "dread"
    assert result.scenes[0].transition_type == "escalation"
    assert result.scenes[1].primary_emotion == "relief"
    assert result.update_emotional_arc != ""
    assert result.contrast_strategy != ""


@pytest.mark.asyncio
async def test_emotional_planner_scene_count_matches_dramatic(tmp_path):
    """EmotionalPlan must have exactly one EmotionalScenePlan per dramatic scene."""
    raw = _VALID_EMOTIONAL_PLAN.model_dump_json()
    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    world = _make_world(tmp_path)

    planner = EmotionalPlanner(client, renderer)
    result = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        world=world,
        recent_prose=[],
    )

    dramatic_scene_ids = {s.scene_id for s in _DRAMATIC_PLAN.scenes}
    emotional_scene_ids = {s.scene_id for s in result.scenes}

    assert len(result.scenes) == len(_DRAMATIC_PLAN.scenes), (
        f"Expected {len(_DRAMATIC_PLAN.scenes)} emotional scenes, got {len(result.scenes)}"
    )
    assert emotional_scene_ids == dramatic_scene_ids, (
        f"Emotional scene IDs {emotional_scene_ids} do not match dramatic scene IDs {dramatic_scene_ids}"
    )


@pytest.mark.asyncio
async def test_emotional_planner_transition_types_validated(tmp_path):
    """A response with an invalid transition_type literal raises ParseError."""
    import json

    bad_plan = _VALID_EMOTIONAL_PLAN.model_dump()
    bad_plan["scenes"][0]["transition_type"] = "explosion"  # not a valid TransitionType
    raw = json.dumps(bad_plan)

    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    world = _make_world(tmp_path)

    planner = EmotionalPlanner(client, renderer)
    with pytest.raises(ParseError):
        await planner.plan(
            dramatic=_DRAMATIC_PLAN,
            world=world,
            recent_prose=[],
        )
