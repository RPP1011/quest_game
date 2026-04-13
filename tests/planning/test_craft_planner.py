"""Tests for app/planning/craft_planner.py — CraftPlanner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.craft.library import CraftLibrary
from app.engine.prompt_renderer import PromptRenderer
from app.planning.craft_planner import CraftPlanner
from app.planning.schemas import (
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


@pytest.mark.asyncio
async def test_craft_planner_injects_narrator():
    """When narrator is passed, its voice samples and worldview appear in the user prompt."""
    from app.craft.schemas import Narrator
    craft_plan = _make_craft_plan()
    raw = craft_plan.model_dump_json()

    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    narrator = Narrator(
        pov_type="third_limited",
        worldview="a quiet observer who watches hands",
        editorial_stance="sympathetic",
        sensory_bias={"visual": 0.5, "tactile": 0.5},
        attention_bias=["hands"],
        voice_samples=["NARRATORSAMPLEXYZ — distinctive token"],
    )

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        narrator=narrator,
    )

    user_content = client.calls[0][0][1].content
    assert "NARRATORSAMPLEXYZ" in user_content
    assert "quiet observer who watches hands" in user_content
    assert "sympathetic" in user_content
