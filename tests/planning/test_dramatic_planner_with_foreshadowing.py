"""Wave 4b: DramaticPlanner + ForeshadowingRetriever integration.

Mirrors the structure of ``test_dramatic_planner_with_retriever.py``:
stub the inference client, pass a fake retriever satisfying the
``Retriever`` structural protocol, and inspect the rendered user
prompt to assert ripe-hook metadata reaches the template.
"""

from __future__ import annotations

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
from app.retrieval import Query, Result
from app.world.db import open_db
from app.world.schema import ArcPosition, Entity, EntityType, NarrativeRecord, PlotThread
from app.world.state_manager import WorldStateManager


PROMPTS = Path(__file__).parent.parent.parent / "prompts"
CRAFT_DATA = Path(__file__).parent.parent.parent / "app" / "craft" / "data"


_VALID_PLAN = DramaticPlan(
    action_resolution=ActionResolution(
        kind="partial",
        narrative="The hero lands a hit but loses the pendant.",
    ),
    scenes=[
        DramaticScene(
            scene_id=1,
            dramatic_question="Can the hero recover the pendant?",
            outcome="Pendant is lost in the river.",
            beats=["Fight", "Pendant slips", "Current pulls it away"],
            dramatic_function="escalation",
            pov_character_id="hero",
            characters_present=["hero", "villain"],
            tools_used=["chekhovs_gun"],
            tension_target=0.6,
        ),
    ],
    update_tension_target=0.6,
    ending_hook="The pendant will surface again.",
    suggested_choices=[
        {"title": "Dive in", "description": "Chase it.", "tags": ["risk"]},
        {"title": "Let it go", "description": "Move on.", "tags": ["cost"]},
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
    """Captures calls and returns a canned list of :class:`Result`."""

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
    """Callers that don't pass the retriever see no Foreshadowing section."""
    client = FakeClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    planner = DramaticPlanner(client, renderer, craft_library)
    await planner.plan(
        directive=directive,
        player_action="sneak in",
        world=world,
        arc=arc,
        structure=structure,
    )

    user_prompt = client.calls[0][0][1].content
    assert "Foreshadowing eligible for payoff" not in user_prompt


@pytest.mark.asyncio
async def test_retriever_without_update_number_skips_injection(tmp_path):
    """A retriever is only consulted when ``update_number`` is in scope."""
    client = FakeClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

    retriever = FakeForeshadowingRetriever(
        [
            Result(
                source_id="hook/q1/fs:1",
                text="A shadow pendant hangs at the hero's neck.",
                score=1.0,
                metadata={"hook_id": "fs:1"},
            )
        ]
    )
    planner = DramaticPlanner(client, renderer, craft_library)
    await planner.plan(
        directive=directive,
        player_action="sneak in",
        world=world,
        arc=arc,
        structure=structure,
        foreshadowing_retriever=retriever,
        # update_number intentionally omitted.
    )

    assert retriever.calls == []
    user_prompt = client.calls[0][0][1].content
    assert "Foreshadowing eligible for payoff" not in user_prompt


@pytest.mark.asyncio
async def test_ripe_hooks_reach_rendered_prompt(tmp_path):
    client = FakeClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    craft_library = CraftLibrary(CRAFT_DATA)
    arc, structure = _make_arc_and_structure(craft_library)
    world = _make_world(tmp_path)
    directive = _make_directive()

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
                "payoff_description": "The pendant reveals the villain's face.",
                "ripeness_status": "overdue",
            },
        ),
        Result(
            source_id="hook/q1/fs:promise",
            text="A broken promise about the southern gate.",
            score=0.7,
            metadata={
                "hook_id": "fs:promise",
                "status": "referenced",
                "planted_at_update": 4,
                "target_update_min": 8,
                "target_update_max": 12,
                "payoff_description": "Return to the gate; find it barred.",
                "ripeness_status": "ripe",
            },
        ),
    ]
    retriever = FakeForeshadowingRetriever(results)

    planner = DramaticPlanner(client, renderer, craft_library)
    await planner.plan(
        directive=directive,
        player_action="sneak in",
        world=world,
        arc=arc,
        structure=structure,
        foreshadowing_retriever=retriever,
        update_number=10,
    )

    # Retriever was asked with the current_update filter.
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
    assert "The pendant reveals the villain's face." in user_prompt
    # Second hook's ripe window also rendered.
    assert "broken promise" in user_prompt
    assert "target 8-12" in user_prompt
    assert "status: referenced" in user_prompt


@pytest.mark.asyncio
async def test_retriever_failure_does_not_break_planner(tmp_path):
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
    result = await planner.plan(
        directive=directive,
        player_action="sneak in",
        world=world,
        arc=arc,
        structure=structure,
        foreshadowing_retriever=BoomRetriever(),  # type: ignore[arg-type]
        update_number=5,
    )
    assert isinstance(result, DramaticPlan)
    user_prompt = client.calls[0][0][1].content
    assert "Foreshadowing eligible for payoff" not in user_prompt
