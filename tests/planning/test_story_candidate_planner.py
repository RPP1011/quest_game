from __future__ import annotations
import json
from pathlib import Path

import pytest

from app.engine.prompt_renderer import PromptRenderer
from app.planning.story_candidate_planner import StoryCandidatePlanner
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, PlotThread, StoryCandidateStatus, ThreadStatus,
)
from app.world.state_manager import WorldStateManager


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple] = []

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw) -> str:
        self.calls.append((messages, json_schema, schema_name))
        return self._response


def _make_world(tmp_path: Path) -> WorldStateManager:
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(
        id="char:hero", entity_type=EntityType.CHARACTER, name="Hero",
        data={"description": "A young thief"},
    ))
    sm.create_entity(Entity(
        id="char:rival", entity_type=EntityType.CHARACTER, name="Rival",
        data={"description": "A noble with a grudge"},
    ))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="Main Quest", description="Steal the thing.",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_plot_thread(PlotThread(
        id="pt:rival", name="Rival Arc", description="Rival stalks hero.",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=6,
    ))
    return sm


CANNED_RESPONSE = json.dumps({
    "candidates": [
        {
            "title": "The Heist",
            "synopsis": "Hero steals the thing before the rival can.",
            "primary_thread_ids": ["pt:main"],
            "secondary_thread_ids": ["pt:rival"],
            "protagonist_character_id": "char:hero",
            "emphasized_theme_ids": [],
            "climax_description": "The vault door opens on a bigger theft.",
            "expected_chapter_count": 12,
        },
        {
            "title": "The Rivalry",
            "synopsis": "Hero and rival turn circumstance into a feud.",
            "primary_thread_ids": ["pt:rival"],
            "secondary_thread_ids": ["pt:main"],
            "protagonist_character_id": "char:rival",
            "emphasized_theme_ids": [],
            "climax_description": "The rival is forced to choose between pride and ruin.",
            "expected_chapter_count": 14,
        },
        {
            "title": "Cold Alliance",
            "synopsis": "Hero and rival must cooperate against a greater threat.",
            "primary_thread_ids": ["pt:main", "pt:rival"],
            "secondary_thread_ids": [],
            "protagonist_character_id": "char:hero",
            "emphasized_theme_ids": [],
            "climax_description": "The alliance breaks at the worst possible moment.",
            "expected_chapter_count": 18,
        },
    ]
})


@pytest.mark.asyncio
async def test_generate_persists_candidates(tmp_path):
    world = _make_world(tmp_path)
    client = FakeClient(CANNED_RESPONSE)
    renderer = PromptRenderer(Path("prompts"))
    planner = StoryCandidatePlanner(client, renderer)

    cands = await planner.generate(world=world, quest_id="q1", n=3)
    assert len(cands) == 3
    titles = [c.title for c in cands]
    assert titles == ["The Heist", "The Rivalry", "Cold Alliance"]

    # Persisted and queryable
    stored = world.list_story_candidates("q1")
    assert len(stored) == 3
    assert all(c.status == StoryCandidateStatus.DRAFT for c in stored)


@pytest.mark.asyncio
async def test_generate_uses_closed_enums_in_schema(tmp_path):
    world = _make_world(tmp_path)
    client = FakeClient(CANNED_RESPONSE)
    renderer = PromptRenderer(Path("prompts"))
    planner = StoryCandidatePlanner(client, renderer)
    await planner.generate(world=world, quest_id="q1", n=3)

    # Schema should constrain thread/character enums to seed values
    _, schema, _ = client.calls[0]
    item = schema["properties"]["candidates"]["items"]
    thread_enum = item["properties"]["primary_thread_ids"]["items"]["enum"]
    char_enum = item["properties"]["protagonist_character_id"]["enum"]
    assert sorted(thread_enum) == ["pt:main", "pt:rival"]
    assert sorted(char_enum) == ["char:hero", "char:rival"]


@pytest.mark.asyncio
async def test_generate_clamps_to_n(tmp_path):
    # Response has 3 candidates but we ask for 2
    world = _make_world(tmp_path)
    client = FakeClient(CANNED_RESPONSE)
    renderer = PromptRenderer(Path("prompts"))
    planner = StoryCandidatePlanner(client, renderer)
    cands = await planner.generate(world=world, quest_id="q1", n=2)
    # Model-returned 3 but caller asked for 2 — planner takes the first N
    # (schema's minItems=maxItems=2 would normally prevent this; this tests
    # defensive slicing when the client mock returns extra)
    assert len(cands) == 2
