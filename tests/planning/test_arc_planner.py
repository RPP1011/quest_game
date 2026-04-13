"""Tests for app/planning/arc_planner.py — ArcPlanner."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.craft.schemas import ArcPhase, Structure
from app.engine.prompt_renderer import PromptRenderer
from app.planning.arc_planner import ArcPlanner
from app.planning.schemas import ArcDirective
from app.world.db import open_db
from app.world.output_parser import ParseError
from app.world.schema import QuestArcState
from app.world.state_manager import WorldStateManager

PROMPTS = Path(__file__).parent.parent.parent / "prompts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_DIRECTIVE = ArcDirective(
    current_phase="Act 1: Setup",
    phase_assessment="The hero has just arrived and alliances are forming.",
    theme_priorities=[],
    plot_objectives=[],
    character_arcs=[],
    tension_range=(0.2, 0.5),
    hooks_to_plant=["Mysterious letter in the innkeeper's drawer"],
    hooks_to_pay_off=[],
    parallels_to_schedule=[],
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


def _make_structure() -> Structure:
    return Structure(
        id="three_act",
        name="Three-Act Structure",
        description="Classic three-act narrative arc.",
        scales=["chapter"],
        phases=[
            ArcPhase(
                name="Act 1: Setup",
                position=0,
                tension_target=0.3,
                description="Establish world and protagonist.",
            ),
            ArcPhase(
                name="Act 2: Confrontation",
                position=1,
                tension_target=0.7,
                description="Rising conflict and complications.",
            ),
            ArcPhase(
                name="Act 3: Resolution",
                position=2,
                tension_target=0.5,
                description="Climax and denouement.",
            ),
        ],
        tension_curve=[(0.0, 0.2), (0.5, 0.8), (1.0, 0.4)],
    )


def _make_arc_state() -> QuestArcState:
    return QuestArcState(
        arc_id="main",
        quest_id="q1",
        structure_id="three_act",
        scale="chapter",
        current_phase_index=0,
        phase_progress=0.25,
        tension_observed=[(1, 0.3), (2, 0.35)],
    )


def _make_world(tmp_path: Path) -> WorldStateManager:
    conn = open_db(tmp_path / "w.db")
    return WorldStateManager(conn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arc_planner_returns_directive(tmp_path):
    """FakeClient returning valid JSON → plan() returns ArcDirective with populated fields."""
    raw = _VALID_DIRECTIVE.model_dump_json()
    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    world = _make_world(tmp_path)
    structure = _make_structure()
    arc_state = _make_arc_state()
    quest_config = {
        "genre": "fantasy",
        "premise": "A young thief discovers a conspiracy threatening the kingdom.",
        "themes": ["loyalty", "power"],
    }

    planner = ArcPlanner(client, renderer)
    directive = await planner.plan(
        quest_config=quest_config,
        arc_state=arc_state,
        world_snapshot=world,
        structure=structure,
    )

    assert isinstance(directive, ArcDirective)
    assert directive.current_phase == "Act 1: Setup"
    assert directive.phase_assessment != ""
    assert directive.hooks_to_plant == ["Mysterious letter in the innkeeper's drawer"]
    assert directive.tension_range == (0.2, 0.5)


@pytest.mark.asyncio
async def test_arc_planner_renders_context(tmp_path):
    """Phase name and plot thread names appear in the prompt sent to the client."""
    from app.world.schema import PlotThread, ArcPosition

    raw = _VALID_DIRECTIVE.model_dump_json()
    client = FakeClient(raw)
    renderer = PromptRenderer(PROMPTS)
    world = _make_world(tmp_path)
    structure = _make_structure()
    arc_state = _make_arc_state()
    quest_config = {
        "genre": "fantasy",
        "premise": "A thief uncovers the truth.",
        "themes": ["deception"],
    }

    # Seed a plot thread
    world.add_plot_thread(
        PlotThread(
            id="pt:guild",
            name="Thieves Guild Schism",
            description="The guild is fracturing over a stolen artifact.",
            arc_position=ArcPosition.RISING,
        )
    )

    planner = ArcPlanner(client, renderer)
    await planner.plan(
        quest_config=quest_config,
        arc_state=arc_state,
        world_snapshot=world,
        structure=structure,
    )

    assert len(client.calls) == 1
    messages, _schema, _name = client.calls[0]
    # User message is the second message
    user_content = messages[1].content

    # Phase name must appear
    assert "Act 1: Setup" in user_content
    # Plot thread name must appear
    assert "Thieves Guild Schism" in user_content
    # Genre must appear
    assert "fantasy" in user_content


@pytest.mark.asyncio
async def test_arc_planner_surfaces_parse_error(tmp_path):
    """FakeClient returning garbage raises ParseError."""
    client = FakeClient("this is not json at all!!!")
    renderer = PromptRenderer(PROMPTS)
    world = _make_world(tmp_path)
    structure = _make_structure()
    arc_state = _make_arc_state()
    quest_config = {"genre": "fantasy", "premise": "x", "themes": []}

    planner = ArcPlanner(client, renderer)
    with pytest.raises(ParseError):
        await planner.plan(
            quest_config=quest_config,
            arc_state=arc_state,
            world_snapshot=world,
            structure=structure,
        )


@pytest.mark.asyncio
async def test_arc_planner_raises_parse_error_on_schema_mismatch(tmp_path):
    """FakeClient returning JSON that fails ArcDirective validation raises ParseError."""
    # Missing required fields 'current_phase' and 'phase_assessment'
    bad_json = json.dumps({"hooks_to_plant": ["x"]})
    client = FakeClient(bad_json)
    renderer = PromptRenderer(PROMPTS)
    world = _make_world(tmp_path)
    structure = _make_structure()
    arc_state = _make_arc_state()
    quest_config = {"genre": "fantasy", "premise": "x", "themes": []}

    planner = ArcPlanner(client, renderer)
    with pytest.raises(ParseError):
        await planner.plan(
            quest_config=quest_config,
            arc_state=arc_state,
            world_snapshot=world,
            structure=structure,
        )
