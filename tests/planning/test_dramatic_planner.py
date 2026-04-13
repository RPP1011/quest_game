"""Tests for app/planning/dramatic_planner.py — DramaticPlanner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.craft.library import CraftLibrary
from app.engine.prompt_renderer import PromptRenderer
from app.planning.dramatic_planner import DramaticPlanner
from app.planning.schemas import (
    ActionResolution,
    ArcDirective,
    DramaticPlan,
    DramaticScene,
    PlotObjective,
    ThemePriority,
)
from app.world.db import open_db
from app.world.output_parser import ParseError
from app.world.schema import ArcPosition, Entity, EntityType, NarrativeRecord, PlotThread
from app.world.state_manager import WorldStateManager

PROMPTS = Path(__file__).parent.parent.parent / "prompts"
CRAFT_DATA = Path(__file__).parent.parent.parent / "app" / "craft" / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PLAN = DramaticPlan(
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
            tools_used=["chekhovs_gun"],
            tension_target=0.6,
        )
    ],
    update_tension_target=0.6,
    ending_hook="The guard's alarm will draw others — the heroes must flee.",
    suggested_choices=[
        {"title": "Flee immediately", "description": "Run before more guards arrive.", "tags": ["action", "risk"]},
        {"title": "Hide and wait", "description": "Conceal yourself and wait for the alarm to die down.", "tags": ["stealth"]},
    ],
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
    # Seed two characters
    wsm.create_entity(Entity(
        id="hero",
        entity_type=EntityType.CHARACTER,
        name="Aldric",
        data={"description": "A young thief with a heart of gold."},
    ))
    wsm.create_entity(Entity(
        id="villain",
        entity_type=EntityType.CHARACTER,
        name="Lord Maren",
        data={"description": "The corrupt lord who stole the village's deed."},
    ))
    # Seed one plot thread
    wsm.add_plot_thread(PlotThread(
        id="pt:deed",
        name="Stolen Deed",
        description="The village deed was stolen by Lord Maren; without it the village will be seized.",
        arc_position=ArcPosition.RISING,
        priority=8,
    ))
    # Seed two narrative records for continuity
    wsm.write_narrative(NarrativeRecord(
        update_number=1,
        raw_text="Aldric slipped through the market crowd, hand already inside the merchant's purse.",
        summary="Aldric pickpockets a merchant.",
        player_action="steal the coin purse",
    ))
    wsm.write_narrative(NarrativeRecord(
        update_number=2,
        raw_text="The guildmaster spread the stolen deed across the table. 'This changes everything,' she said.",
        summary="Guildmaster reveals deed significance.",
        player_action="bring the deed to the guild",
    ))
    return wsm


def _make_directive() -> ArcDirective:
    return ArcDirective(
        current_phase="setup",
        phase_assessment="The hero is establishing contacts and gathering resources.",
        theme_priorities=[ThemePriority(theme_id="loyalty", intensity="emerging")],
        plot_objectives=[PlotObjective(description="Recover the stolen deed", urgency="this_phase", plot_thread_id="pt:deed")],
        tension_range=(0.3, 0.65),
        hooks_to_plant=["A guard recognizes Aldric from a past job."],
    )


def _make_craft_library() -> CraftLibrary:
    return CraftLibrary(CRAFT_DATA)


def _make_arc_and_structure(craft_library: CraftLibrary):
    from app.craft.schemas import Arc
    structure = craft_library.structure("three_act")
    arc = Arc(
        id="main",
        name="Main Arc",
        scale="chapter",
        structure_id="three_act",
        current_phase_index=0,
        phase_progress=0.2,
        required_beats_remaining=["chekhovs_gun"],
    )
    return arc, structure


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dramatic_planner_returns_plan(tmp_path):
    """FakeClient returning valid JSON → plan() returns a DramaticPlan."""
    raw = _VALID_PLAN.model_dump_json()
    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    craft_library = _make_craft_library()
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    planner = DramaticPlanner(client, renderer, craft_library)
    result = await planner.plan(
        directive=directive,
        player_action="attempt to sneak into the manor and recover the deed",
        world=world,
        arc=arc,
        structure=structure,
    )

    assert isinstance(result, DramaticPlan)
    assert result.action_resolution.kind == "partial"
    assert len(result.scenes) == 1
    assert result.scenes[0].dramatic_question != ""
    assert result.scenes[0].outcome != ""
    assert result.ending_hook != ""
    assert len(result.suggested_choices) >= 1


@pytest.mark.asyncio
async def test_dramatic_planner_includes_recommended_tools_in_prompt(tmp_path):
    """Recommended tool names appear in the user prompt sent to the client."""
    raw = _VALID_PLAN.model_dump_json()
    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    craft_library = _make_craft_library()
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    # Find what tools recommend_tools would return so we can assert on a real name
    recommended = craft_library.recommend_tools(arc, structure, limit=5)
    assert recommended, "craft_library must return at least one recommendation for this test"
    expected_tool_name = recommended[0].name

    planner = DramaticPlanner(client, renderer, craft_library)
    await planner.plan(
        directive=directive,
        player_action="sneak into the manor",
        world=world,
        arc=arc,
        structure=structure,
    )

    assert len(client.calls) == 1
    messages, _schema, _name = client.calls[0]
    user_content = messages[1].content

    assert expected_tool_name in user_content, (
        f"Expected tool name {expected_tool_name!r} to appear in user prompt"
    )


@pytest.mark.asyncio
async def test_dramatic_planner_surfaces_parse_error(tmp_path):
    """Garbage response from client raises ParseError."""
    client = FakeClient("this is definitely not json!!!")
    renderer = PromptRenderer(PROMPTS)
    craft_library = _make_craft_library()
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    planner = DramaticPlanner(client, renderer, craft_library)
    with pytest.raises(ParseError):
        await planner.plan(
            directive=directive,
            player_action="try something",
            world=world,
            arc=arc,
            structure=structure,
        )
