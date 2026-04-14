"""Wave 3c: DramaticPlanner scene-exemplar retrieval integration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

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
from app.retrieval import Query, Result, SceneShapeRetriever
from app.world.db import open_db
from app.world.schema import ArcPosition, Entity, EntityType, NarrativeRecord, PlotThread
from app.world.state_manager import WorldStateManager


PROMPTS = Path(__file__).parent.parent.parent / "prompts"
CRAFT_DATA = Path(__file__).parent.parent.parent / "app" / "craft" / "data"


# ---------------------------------------------------------------------------
# Helpers (mirror test_dramatic_planner.py)
# ---------------------------------------------------------------------------


_VALID_PLAN = DramaticPlan(
    action_resolution=ActionResolution(
        kind="partial",
        narrative="The hero recovers the stolen key but is spotted by a guard.",
    ),
    scenes=[
        DramaticScene(
            scene_id=1,
            dramatic_question="Can the hero retrieve the key?",
            outcome="Key retrieved; guard raises alarm.",
            beats=["Hero sneaks in", "Finds the key", "Guard turns around"],
            dramatic_function="escalation",
            pov_character_id="hero",
            characters_present=["hero", "guard"],
            tools_used=["chekhovs_gun"],
            tension_target=0.6,
        ),
    ],
    update_tension_target=0.6,
    ending_hook="The guard's alarm will draw others.",
    suggested_choices=[
        {"title": "Flee immediately", "description": "Run.", "tags": ["risk"]},
        {"title": "Hide and wait", "description": "Conceal.", "tags": ["stealth"]},
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


class FakeSceneRetriever:
    """Captures calls + returns a canned list of ``Result``.

    Satisfies the ``Retriever`` structural protocol, which is all the
    planner requires. Constructed entirely without touching the real
    scenes corpus so the test is hermetic.
    """

    def __init__(self, results: list[Result]) -> None:
        self._results = results
        self.calls: list[tuple[Query, int]] = []

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        self.calls.append((query, k))
        return list(self._results[:k])


def _make_world(tmp_path: Path) -> WorldStateManager:
    conn = open_db(tmp_path / "w.db")
    wsm = WorldStateManager(conn)
    wsm.create_entity(Entity(
        id="hero",
        entity_type=EntityType.CHARACTER,
        name="Aldric",
        data={"description": "A thief."},
    ))
    wsm.create_entity(Entity(
        id="villain",
        entity_type=EntityType.CHARACTER,
        name="Maren",
        data={"description": "The lord."},
    ))
    wsm.add_plot_thread(PlotThread(
        id="pt:deed",
        name="Stolen Deed",
        description="Deed stolen.",
        arc_position=ArcPosition.RISING,
        priority=8,
    ))
    wsm.write_narrative(NarrativeRecord(
        update_number=1,
        raw_text="Aldric slipped through the crowd.",
        summary="pickpocket",
        player_action="steal",
    ))
    return wsm


def _make_directive() -> ArcDirective:
    return ArcDirective(
        current_phase="rising_action",
        phase_assessment="Stakes tightening.",
        theme_priorities=[ThemePriority(theme_id="loyalty", intensity="emerging")],
        plot_objectives=[
            PlotObjective(description="Recover the deed", urgency="this_phase"),
        ],
        tension_range=(0.55, 0.75),
        hooks_to_plant=["A guard recognizes Aldric."],
    )


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
async def test_no_retriever_keeps_prompt_clean(tmp_path):
    """Callers that don't pass a retriever see no Scene-shape Exemplars section."""
    client = FakeClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    planner = DramaticPlanner(client, renderer, craft_library)
    result = await planner.plan(
        directive=directive,
        player_action="sneak into the manor",
        world=world,
        arc=arc,
        structure=structure,
    )

    assert isinstance(result, DramaticPlan)
    assert len(client.calls) == 1
    user_prompt = client.calls[0][0][1].content
    assert "Scene-shape Exemplars" not in user_prompt


@pytest.mark.asyncio
async def test_retriever_injects_exemplars(tmp_path):
    """When a retriever is supplied, exemplar previews land in the prompt."""
    client = FakeClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    results = [
        Result(
            source_id="high_tension_novel/s01",
            text=(
                "The bridge gave beneath her boots and she caught the rope. "
                "Below the river was a dark wide mouth. The rope was fraying. "
                "She had a heartbeat to choose, and she chose wrong."
            ),
            score=0.95,
            metadata={
                "work_id": "high_tension_novel",
                "scene_id": "s01",
                "pov": "third_omniscient",
                "scale": "scene",
                "dramatic_function": "escalation",
                "actual_scores": {
                    "tension_execution": 0.85,
                    "scene_coherence": 0.78,
                },
            },
        ),
    ]
    scene_retriever = FakeSceneRetriever(results)

    planner = DramaticPlanner(client, renderer, craft_library)
    await planner.plan(
        directive=directive,
        player_action="sneak into the manor",
        world=world,
        arc=arc,
        structure=structure,
        scene_retriever=scene_retriever,
    )

    assert len(scene_retriever.calls) == 1
    query, k = scene_retriever.calls[0]
    assert k == 2
    # Score_ranges filter should be derived from the directive's tension envelope.
    score_ranges = query.filters.get("score_ranges")
    assert score_ranges is not None
    assert score_ranges.get("tension_execution") == (0.55, 0.75)
    # scene_coherence floor is set so exemplars are well-executed.
    assert "scene_coherence" in score_ranges

    # Seed text pulls from directive signals.
    assert query.seed_text is not None
    assert "rising_action" in query.seed_text

    user_prompt = client.calls[0][0][1].content
    assert "Scene-shape Exemplars" in user_prompt
    assert "high_tension_novel/s01" in user_prompt
    # Preview contains the opening words of the exemplar text.
    assert "The bridge gave beneath her boots" in user_prompt
    # Dims are rendered as decimals in the preview header.
    assert "0.85" in user_prompt
    assert "escalation" in user_prompt


@pytest.mark.asyncio
async def test_retriever_returning_empty_list_is_silent(tmp_path):
    """Empty retriever result → no exemplars section rendered."""
    client = FakeClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()
    scene_retriever = FakeSceneRetriever(results=[])

    planner = DramaticPlanner(client, renderer, craft_library)
    await planner.plan(
        directive=directive,
        player_action="sneak in",
        world=world,
        arc=arc,
        structure=structure,
        scene_retriever=scene_retriever,
    )

    assert len(scene_retriever.calls) == 1
    user_prompt = client.calls[0][0][1].content
    assert "Scene-shape Exemplars" not in user_prompt


@pytest.mark.asyncio
async def test_retriever_failure_does_not_break_planner(tmp_path):
    """A retriever that raises must not crash the planner."""
    client = FakeClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    class BoomRetriever:
        async def retrieve(self, query: Query, *, k: int = 3):
            raise RuntimeError("boom")

    planner = DramaticPlanner(client, renderer, craft_library)
    # Satisfies the structural retriever protocol but explodes on call.
    result = await planner.plan(
        directive=directive,
        player_action="sneak in",
        world=world,
        arc=arc,
        structure=structure,
        scene_retriever=BoomRetriever(),  # type: ignore[arg-type]
    )
    assert isinstance(result, DramaticPlan)
    user_prompt = client.calls[0][0][1].content
    assert "Scene-shape Exemplars" not in user_prompt
