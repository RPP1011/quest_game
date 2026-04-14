"""Wave 4a: CraftPlanner motif-retriever integration.

A stubbed ``MotifRetriever`` feeds canned results; the test asserts those
motifs reach the rendered user prompt as the "Due/overdue motifs" section.
Default-off behavior (no retriever kwarg → identical to pre-Wave-4a) is
covered by the existing ``test_craft_planner.py`` suite.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DRAMATIC_PLAN = DramaticPlan(
    action_resolution=ActionResolution(
        kind="partial",
        narrative="The hero recovers the stolen key but is spotted.",
    ),
    scenes=[
        DramaticScene(
            scene_id=1,
            dramatic_question="Can the hero retrieve the key?",
            outcome="Key retrieved; alarm raised.",
            beats=["Hero sneaks in", "Finds the key"],
            dramatic_function="escalation",
            pov_character_id="hero",
            characters_present=["hero"],
            tools_used=["chekhovs_gun"],
            tension_target=0.6,
        ),
    ],
    update_tension_target=0.6,
    ending_hook="The alarm will draw others.",
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
            entry_state="cautious",
            exit_state="panicked",
            transition_type="escalation",
            emotional_source="The guard's proximity.",
            character_emotions={
                "hero": CharacterEmotionalState(
                    internal="terror",
                    displayed="calm",
                ),
            },
        ),
    ],
    update_emotional_arc="Dread peaks in scene 1.",
    contrast_strategy="Contain and release.",
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
                narrator_focus=["guard's movements"],
            ),
        ],
        briefs=[
            CraftBrief(
                scene_id=1,
                brief=(
                    "This scene should feel like a held breath. Short, "
                    "declarative sentences; physical detail. The key is "
                    "the fulcrum. Close on eyes meeting."
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


class FakeMotifRetriever:
    """Satisfies the structural ``Retriever`` protocol; returns canned results."""

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
    """Callers that omit the motif_retriever see no Due/overdue section."""
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
    )

    user_prompt = client.calls[0][0][1].content
    assert "Due/overdue motifs" not in user_prompt


@pytest.mark.asyncio
async def test_retriever_without_update_number_stays_inert():
    """Passing a retriever but no ``update_number`` → retriever is never called."""
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    retriever = FakeMotifRetriever([
        Result(
            source_id="motif/q1/iron_key",
            text="A heavy iron key.",
            score=1.0,
            metadata={
                "motif_id": "iron_key",
                "name": "Iron key",
                "last_update_number": 1,
                "last_semantic_value": "betrayal",
                "intervals_since_last": 9,
                "target_interval_min": 2,
                "target_interval_max": 4,
                "status": "overdue",
                "recent_contexts": [],
            },
        ),
    ])

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        motif_retriever=retriever,
        # update_number intentionally omitted
    )

    assert retriever.calls == []
    user_prompt = client.calls[0][0][1].content
    assert "Due/overdue motifs" not in user_prompt


@pytest.mark.asyncio
async def test_due_motifs_reach_rendered_prompt():
    """With a retriever + update_number, motifs land in the user prompt."""
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    results = [
        Result(
            source_id="motif/q1/iron_key",
            text="A heavy iron key passed from hand to hand.",
            score=1.0,
            metadata={
                "motif_id": "iron_key",
                "name": "Iron key",
                "last_update_number": 1,
                "last_semantic_value": "betrayal by concealment",
                "intervals_since_last": 9,
                "target_interval_min": 2,
                "target_interval_max": 4,
                "status": "overdue",
                "recent_contexts": [
                    {
                        "update_number": 1,
                        "context": "Maren pocketed the key.",
                        "semantic_value": "betrayal by concealment",
                        "intensity": 0.6,
                    },
                ],
            },
        ),
        Result(
            source_id="motif/q1/cracked_mirror",
            text="A mirror with a single crack.",
            score=0.6,
            metadata={
                "motif_id": "cracked_mirror",
                "name": "Cracked mirror",
                "last_update_number": 4,
                "last_semantic_value": "identity fracturing",
                "intervals_since_last": 6,
                "target_interval_min": 3,
                "target_interval_max": 7,
                "status": "due",
                "recent_contexts": [],
            },
        ),
    ]
    retriever = FakeMotifRetriever(results)

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        motif_retriever=retriever,
        update_number=10,
    )

    # Retriever was called exactly once with current_update in the filters.
    assert len(retriever.calls) == 1
    query, k = retriever.calls[0]
    assert k == 3
    assert query.filters.get("current_update") == 10

    user_prompt = client.calls[0][0][1].content
    # Section header is rendered.
    assert "Due/overdue motifs" in user_prompt
    # Both motifs' names + descriptions land.
    assert "Iron key" in user_prompt
    assert "A heavy iron key passed from hand to hand." in user_prompt
    assert "Cracked mirror" in user_prompt
    # Status + last-seen metadata formatted as spec demands.
    assert "last seen update 1" in user_prompt
    assert "last seen update 4" in user_prompt
    assert 'semantic value: "betrayal by concealment"' in user_prompt
    assert "status: overdue" in user_prompt
    assert "status: due" in user_prompt


@pytest.mark.asyncio
async def test_retriever_returning_empty_list_is_silent():
    """Empty retriever result → no Due/overdue section rendered."""
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    retriever = FakeMotifRetriever(results=[])

    planner = CraftPlanner(client, renderer, craft_library)
    await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        motif_retriever=retriever,
        update_number=10,
    )

    assert len(retriever.calls) == 1
    user_prompt = client.calls[0][0][1].content
    assert "Due/overdue motifs" not in user_prompt


@pytest.mark.asyncio
async def test_retriever_failure_does_not_break_planner():
    """A retriever that raises must not crash planning."""
    client = FakeClient(_make_craft_plan().model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)

    class BoomRetriever:
        async def retrieve(self, query: Query, *, k: int = 3):
            raise RuntimeError("boom")

    planner = CraftPlanner(client, renderer, craft_library)
    plan = await planner.plan(
        dramatic=_DRAMATIC_PLAN,
        emotional=_EMOTIONAL_PLAN,
        motif_retriever=BoomRetriever(),  # type: ignore[arg-type]
        update_number=10,
    )
    assert isinstance(plan, CraftPlan)
    user_prompt = client.calls[0][0][1].content
    assert "Due/overdue motifs" not in user_prompt
