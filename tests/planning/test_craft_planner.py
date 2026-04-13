"""Tests for app/planning/craft_planner.py — CraftPlanner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.craft.library import CraftLibrary
from app.engine.prompt_renderer import PromptRenderer
from app.planning.craft_planner import CraftPlanner
from app.planning.schemas import (
    IndirectionInstruction,
    ActionResolution,
    CraftBrief,
    CraftPlan,
    CraftScenePlan,
    DramaticPlan,
    DramaticScene,
    EmotionalPlan,
    EmotionalScenePlan,
    CharacterEmotionalState,
    SceneRegister,
    TemporalStructure,
    ToolSelection,
)

PROMPTS = Path(__file__).parent.parent.parent / "prompts"
CRAFT_DATA = Path(__file__).parent.parent.parent / "app" / "craft" / "data"

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
            tools_used=["chekhovs_gun"],
            tension_target=0.6,
        ),
        DramaticScene(
            scene_id=2,
            dramatic_question="Will the hero escape the manor grounds?",
            outcome="Hero escapes through the garden, barely.",
            beats=["Alarm sounds", "Hero runs", "Clears the wall"],
            dramatic_function="rising tension",
            pov_character_id="hero",
            characters_present=["hero"],
            tension_target=0.75,
        ),
    ],
    update_tension_target=0.65,
    ending_hook="The alarm will draw Lord Maren himself.",
    suggested_choices=[
        {"title": "Flee", "description": "Run now.", "tags": ["action"]},
    ],
    tools_selected=[
        ToolSelection(tool_id="chekhovs_gun", scene_id=1, application="plant the stolen key as a narrative object"),
    ],
)

_EMOTIONAL_PLAN = EmotionalPlan(
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


def _make_craft_plan() -> CraftPlan:
    return CraftPlan(
        scenes=[
            CraftScenePlan(
                scene_id=1,
                temporal=TemporalStructure(description="linear present-scene"),
                register=SceneRegister(
                    sentence_variance="low",
                    concrete_abstract_ratio=0.8,
                    interiority_depth="surface",
                    sensory_density="sparse",
                    dialogue_ratio=0.1,
                    pace="compressed",
                ),
                narrator_focus=["physical details that betray anxiety", "guard's movements"],
                narrator_withholding=["hero's internal panic"],
                opening_instruction="Open mid-action — hero already inside.",
                closing_instruction="End on the guard's eyes meeting Aldric's.",
            ),
            CraftScenePlan(
                scene_id=2,
                temporal=TemporalStructure(description="linear present-scene"),
                register=SceneRegister(
                    sentence_variance="medium",
                    concrete_abstract_ratio=0.7,
                    interiority_depth="surface",
                    sensory_density="moderate",
                    dialogue_ratio=0.0,
                    pace="compressed",
                ),
                narrator_focus=["pace and breath", "distance from the wall"],
            ),
        ],
        briefs=[
            CraftBrief(
                scene_id=1,
                brief=(
                    "This scene should feel like a held breath. The writer enters mid-motion "
                    "— no preamble. Aldric is already inside. Every sentence is short, declarative, "
                    "physical. The narrator stays outside his interiority; we read his terror through "
                    "what he notices — the guard's boot heel, the angle of light, the way his own "
                    "hands have steadied themselves without being told. The key must feel like an object "
                    "with weight; the moment he holds it is the scene's fulcrum. Sentence rhythm "
                    "compresses as the guard turns. Close on eyes meeting — nothing resolved, everything "
                    "at stake."
                ),
            ),
            CraftBrief(
                scene_id=2,
                brief=(
                    "Release. This scene breathes after the coiled tension of scene one — but only "
                    "slightly. The sentences can lengthen into the run, tracking distance and sound "
                    "rather than interiors. The alarm is a physical presence — a sound with texture. "
                    "The wall is the only image that matters; everything else is motion toward it. "
                    "End on the other side: not triumph, just the fact of being clear."
                ),
            ),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_craft_planner_returns_plan_with_briefs_and_scenes():
    """FakeClient returning valid JSON with scenes + briefs → CraftPlan parses correctly."""
    craft_plan = _make_craft_plan()
    raw = craft_plan.model_dump_json()

    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    planner = CraftPlanner(client, renderer, craft_library)
    result = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
    )

    assert isinstance(result, CraftPlan)
    assert len(result.scenes) == 2
    assert len(result.briefs) == 2

    # briefs are CraftBrief instances
    assert all(isinstance(b, CraftBrief) for b in result.briefs)
    # scenes are CraftScenePlan instances
    assert all(isinstance(s, CraftScenePlan) for s in result.scenes)

    # spot-check a brief is non-empty prose
    assert len(result.briefs[0].brief) >= 50


@pytest.mark.asyncio
async def test_craft_planner_scene_ids_match_dramatic():
    """Every dramatic scene_id has a CraftScenePlan and a CraftBrief with matching scene_id."""
    craft_plan = _make_craft_plan()
    raw = craft_plan.model_dump_json()

    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    planner = CraftPlanner(client, renderer, craft_library)
    result = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
    )

    dramatic_ids = {s.scene_id for s in _DRAMATIC_PLAN.scenes}
    craft_scene_ids = {s.scene_id for s in result.scenes}
    brief_ids = {b.scene_id for b in result.briefs}

    assert craft_scene_ids == dramatic_ids, (
        f"CraftScenePlan scene_ids {craft_scene_ids} do not match dramatic ids {dramatic_ids}"
    )
    assert brief_ids == dramatic_ids, (
        f"CraftBrief scene_ids {brief_ids} do not match dramatic ids {dramatic_ids}"
    )


@pytest.mark.asyncio
async def test_craft_planner_injects_style_voice_samples():
    """When style_register_id is set, voice_samples appear in the user prompt."""
    craft_plan = _make_craft_plan()
    raw = craft_plan.model_dump_json()

    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    # Load the style register so we know what samples to expect
    style = craft_library.style("terse_military")
    assert style.voice_samples, "terse_military must have voice_samples for this test"
    # Pick a distinctive substring from the first sample
    expected_fragment = style.voice_samples[0].strip()[:30]

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        style_register_id="terse_military",
    )

    assert len(client.calls) == 1
    messages, _schema, _name = client.calls[0]
    user_content = messages[1].content

    assert expected_fragment in user_content, (
        f"Expected voice sample fragment {expected_fragment!r} to appear in user prompt"
    )


@pytest.mark.asyncio
async def test_craft_planner_includes_tool_examples():
    """Example snippets for tools mentioned in dramatic.tools_selected appear in user prompt."""
    craft_plan = _make_craft_plan()
    raw = craft_plan.model_dump_json()

    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    # chekhovs_gun is in tools_selected and scene tools_used
    examples = craft_library.examples_for_tool("chekhovs_gun")
    assert examples, "chekhovs_gun must have at least one example in the library"
    # A distinctive fragment from the first example snippet
    expected_fragment = examples[0].snippet.strip()[:40]

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
    )

    assert len(client.calls) == 1
    messages, _schema, _name = client.calls[0]
    user_content = messages[1].content

    assert expected_fragment in user_content, (
        f"Expected example fragment {expected_fragment!r} to appear in user prompt"
    )


# ---------------------------------------------------------------------------
# G13 — Indirection grounding
# ---------------------------------------------------------------------------


def _world_with_hero_motives():
    from app.world.db import open_db
    from app.world.schema import Entity, EntityType
    from app.world.state_manager import WorldStateManager
    conn = open_db(":memory:")
    world = WorldStateManager(conn)
    world.create_entity(Entity(
        id="hero",
        entity_type=EntityType.CHARACTER,
        name="Aldric",
        data={
            "unconscious_motives": [
                {
                    "id": "um:hero:erase",
                    "motive": "needs to disappear into others' purposes to avoid self-definition",
                    "surface_manifestations": ["defers decisions", "always 'happy to help'"],
                    "detail_tells": ["watches others' hands, not faces"],
                    "what_not_to_say": ["erase", "disappear", "hide"],
                    "active_since_update": 0,
                    "resolved_at_update": None,
                }
            ]
        },
    ))
    world.create_entity(Entity(
        id="guard", entity_type=EntityType.CHARACTER, name="Guard", data={},
    ))
    return world


@pytest.mark.asyncio
async def test_craft_planner_injects_stored_motives_into_prompt():
    """When world is passed, stored motives appear in the user prompt."""
    craft_plan = _make_craft_plan()
    client = FakeClient(craft_plan.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    world = _world_with_hero_motives()

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        world=world,
    )

    user_content = client.calls[0][0][1].content
    assert "um:hero:erase" in user_content
    assert "disappear into others' purposes" in user_content
    assert "Unconscious Motives" in user_content


@pytest.mark.asyncio
async def test_craft_planner_backfills_indirection_from_stored_motives():
    """If the LLM emits no indirection for a POV char with motives,
    craft planner backfills one from the stored motive."""
    # Use a craft plan that has NO indirection entries at all
    craft_plan = _make_craft_plan()
    assert all(s.indirection == [] for s in craft_plan.scenes)
    client = FakeClient(craft_plan.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    world = _world_with_hero_motives()

    planner = CraftPlanner(client, renderer, craft_library)
    result = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        world=world,
    )

    # Both scenes have POV=hero, both should have a backfilled entry
    for scene in result.scenes:
        hero_instrs = [i for i in scene.indirection if i.character_id == "hero"]
        assert len(hero_instrs) == 1
        instr = hero_instrs[0]
        assert "disappear" in instr.unconscious_motive
        assert "erase" in instr.what_not_to_say
        assert "defers decisions" in instr.surface_manifestations


@pytest.mark.asyncio
async def test_craft_planner_backfills_generic_indirection():
    """An empty/generic LLM indirection entry is replaced by stored-motive content."""
    craft_plan = _make_craft_plan()
    # Inject a generic, essentially-empty indirection for hero into scene 1
    craft_plan.scenes[0].indirection.append(IndirectionInstruction(
        character_id="hero",
        unconscious_motive="",
        surface_manifestations=[],
        detail_tells=[],
        what_not_to_say=[],
        reader_should_infer="something ineffable",
    ))
    client = FakeClient(craft_plan.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    world = _world_with_hero_motives()

    planner = CraftPlanner(client, renderer, craft_library)
    result = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        world=world,
    )

    scene1 = result.scenes[0]
    hero_instrs = [i for i in scene1.indirection if i.character_id == "hero"]
    assert len(hero_instrs) == 1
    instr = hero_instrs[0]
    assert instr.unconscious_motive  # no longer empty
    assert "erase" in instr.what_not_to_say
    # reader_should_infer was preserved (non-generic content)
    assert instr.reader_should_infer == "something ineffable"


@pytest.mark.asyncio
async def test_craft_planner_no_world_no_motive_injection():
    """Without world, user prompt contains no motive block."""
    craft_plan = _make_craft_plan()
    client = FakeClient(craft_plan.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(dramatic=_DRAMATIC_PLAN, emotional=_EMOTIONAL_PLAN)

    user_content = client.calls[0][0][1].content
    assert "Unconscious Motives" not in user_content
