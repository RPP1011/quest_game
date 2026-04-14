"""Wave 4b: CraftPlanner + ForeshadowingRetriever integration.

Stubs the inference client and a retriever satisfying the ``Retriever``
structural protocol, then asserts the rendered craft user prompt
carries the ripe-hook payoff metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.craft.library import CraftLibrary
from app.engine.prompt_renderer import PromptRenderer
from app.planning.craft_planner import CraftPlanner
from app.planning.schemas import (
    ActionResolution,
    CharacterEmotionalState,
    CraftBrief,
    CraftPlan,
    CraftScenePlan,
    DramaticPlan,
    DramaticScene,
    EmotionalPlan,
    EmotionalScenePlan,
    SceneRegister,
    TemporalStructure,
)
from app.retrieval import Query, Result


PROMPTS = Path(__file__).parent.parent.parent / "prompts"
CRAFT_DATA = Path(__file__).parent.parent.parent / "app" / "craft" / "data"


_DRAMATIC_PLAN = DramaticPlan(
    action_resolution=ActionResolution(
        kind="partial",
        narrative="Aldric recovers the key but is spotted.",
    ),
    scenes=[
        DramaticScene(
            scene_id=1,
            dramatic_question="Can Aldric retrieve the key?",
            outcome="Key retrieved; alarm raised.",
            beats=["Sneak", "Find", "Alarm"],
            dramatic_function="escalation",
            pov_character_id="hero",
            characters_present=["hero", "guard"],
            tools_used=["chekhovs_gun"],
            tension_target=0.6,
        ),
    ],
    update_tension_target=0.6,
    ending_hook="Lord Maren comes.",
    suggested_choices=[
        {"title": "Flee", "description": "Run.", "tags": ["action"]},
    ],
)

_EMOTIONAL_PLAN = EmotionalPlan(
    scenes=[
        EmotionalScenePlan(
            scene_id=1,
            primary_emotion="dread",
            intensity=0.6,
            entry_state="cautious hope",
            exit_state="panicked resolve",
            transition_type="escalation",
            emotional_source="proximity to the guard",
            character_emotions={
                "hero": CharacterEmotionalState(
                    internal="terror",
                    displayed="control",
                )
            },
        ),
    ],
    update_emotional_arc="Dread peaks.",
    contrast_strategy="tight → wide.",
)


def _make_craft_plan() -> CraftPlan:
    return CraftPlan(
        scenes=[
            CraftScenePlan(
                scene_id=1,
                temporal=TemporalStructure(description="linear"),
                register=SceneRegister(
                    sentence_variance="low",
                    concrete_abstract_ratio=0.8,
                    interiority_depth="surface",
                    sensory_density="sparse",
                    dialogue_ratio=0.1,
                    pace="compressed",
                ),
                narrator_focus=["physical details"],
            ),
        ],
        briefs=[
            CraftBrief(
                scene_id=1,
                brief=(
                    "This scene is a held breath. Every sentence is short and "
                    "physical. The narrator stays outside his interiority; we "
                    "read his terror through what he notices — the guard's "
                    "boot heel, the angle of light, the way his hands have "
                    "steadied themselves without being told."
                ),
            ),
        ],
    )


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple] = []

    async def chat_structured(
        self, messages, *, json_schema, schema_name="Output", **kw
    ) -> str:
        self.calls.append((messages, json_schema, schema_name))
        return self._response


class FakeForeshadowingRetriever:
    def __init__(self, results: list[Result]) -> None:
        self._results = results
        self.calls: list[tuple[Query, int]] = []

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        self.calls.append((query, k))
        return list(self._results[:k])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retriever_keeps_prompt_clean():
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(dramatic=_DRAMATIC_PLAN, emotional=_EMOTIONAL_PLAN)

    user_prompt = client.calls[0][0][1].content
    assert "Foreshadowing eligible for payoff" not in user_prompt


@pytest.mark.asyncio
async def test_retriever_without_update_number_skips_injection():
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    retriever = FakeForeshadowingRetriever(
        [
            Result(
                source_id="hook/q1/fs:1",
                text="A shadow pendant hangs at the hero's neck.",
                score=1.0,
                metadata={},
            )
        ]
    )
    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        foreshadowing_retriever=retriever,
        # update_number intentionally omitted.
    )

    assert retriever.calls == []
    user_prompt = client.calls[0][0][1].content
    assert "Foreshadowing eligible for payoff" not in user_prompt


@pytest.mark.asyncio
async def test_ripe_hooks_reach_rendered_prompt():
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    results = [
        Result(
            source_id="hook/q1/fs:pendant",
            text="A shadow pendant hangs at the hero's neck.",
            score=1.0,
            metadata={
                "hook_id": "fs:pendant",
                "status": "planted",
                "planted_at_update": 2,
                "target_update_min": 5,
                "target_update_max": 8,
                "payoff_description": "Pendant reveals the villain's face.",
                "ripeness_status": "overdue",
            },
        ),
    ]
    retriever = FakeForeshadowingRetriever(results)

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        foreshadowing_retriever=retriever,
        update_number=10,
    )

    assert len(retriever.calls) == 1
    query, k = retriever.calls[0]
    assert k == 3
    assert query.filters.get("current_update") == 10

    user_prompt = client.calls[0][0][1].content
    assert "Foreshadowing eligible for payoff" in user_prompt
    assert "shadow pendant" in user_prompt
    assert "planted update 2" in user_prompt
    assert "target 5-8" in user_prompt
    assert "status: planted" in user_prompt
    assert "Pendant reveals the villain's face." in user_prompt


@pytest.mark.asyncio
async def test_retriever_failure_does_not_break_planner():
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    class BoomRetriever:
        async def retrieve(self, query: Query, *, k: int = 3):
            raise RuntimeError("boom")

    planner = CraftPlanner(client, renderer, craft_library)
    result = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        foreshadowing_retriever=BoomRetriever(),  # type: ignore[arg-type]
        update_number=5,
    )
    assert isinstance(result, CraftPlan)
    user_prompt = client.calls[0][0][1].content
    assert "Foreshadowing eligible for payoff" not in user_prompt
