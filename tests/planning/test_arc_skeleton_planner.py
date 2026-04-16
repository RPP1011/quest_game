from __future__ import annotations
import json
from pathlib import Path

import pytest

from app.engine.prompt_renderer import PromptRenderer
from app.planning.arc_skeleton_planner import (
    ArcSkeletonPlanner, validate_skeleton_coverage,
)
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, ArcSkeleton, Entity, EntityType, ForeshadowingHook,
    HookPlacement, PlotThread, SkeletonChapter, StoryCandidate, ThreadStatus,
)
from app.world.state_manager import WorldStateManager


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list = []

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw):
        self.calls.append((messages, json_schema, schema_name))
        return self._response


def _make_world(tmp_path: Path) -> WorldStateManager:
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(
        id="char:hero", entity_type=EntityType.CHARACTER, name="Hero",
        data={"role": "protagonist"},
    ))
    sm.create_entity(Entity(
        id="char:rival", entity_type=EntityType.CHARACTER, name="Rival",
        data={"role": "antagonist"},
    ))
    sm.create_entity(Entity(
        id="loc:inn", entity_type=EntityType.LOCATION, name="The Inn",
    ))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="Main", description="Main thread.",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_foreshadowing(ForeshadowingHook(
        id="fs:1", description="A planted hook.", planted_at_update=0,
        payoff_target="Revealed in act 3.",
    ))
    # Seed the candidate so FK is satisfied
    sm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="The Heist",
        synopsis="Hero pulls the heist.", primary_thread_ids=["pt:main"],
        secondary_thread_ids=[], protagonist_character_id="char:hero",
        emphasized_theme_ids=[], climax_description="The vault opens.",
        expected_chapter_count=5,
    ))
    return sm


CANNED = json.dumps({
    "chapters": [
        {
            "chapter_index": 1, "pov_character_id": "char:hero",
            "location_constraint": "loc:inn",
            "dramatic_question": "Can the hero case the vault?",
            "required_plot_beats": [
                "Hero sets up pt:main",
                "Hero encounters char:rival",
            ],
            "target_tension": 0.3,
            "entities_to_surface": [], "theme_emphasis": [],
        },
        {
            "chapter_index": 2, "pov_character_id": "char:rival",
            "location_constraint": None,
            "dramatic_question": "What does the rival know?",
            "required_plot_beats": ["Rival discovers hero's plan for pt:main"],
            "target_tension": 0.5,
            "entities_to_surface": [], "theme_emphasis": [],
        },
        {
            "chapter_index": 3, "pov_character_id": "char:hero",
            "location_constraint": None,
            "dramatic_question": "Does the heist succeed?",
            "required_plot_beats": ["Vault opens — pay off fs:1"],
            "target_tension": 0.9,
            "entities_to_surface": [], "theme_emphasis": [],
        },
        {
            "chapter_index": 4, "pov_character_id": "char:hero",
            "location_constraint": None,
            "dramatic_question": "What is the cost?",
            "required_plot_beats": ["Denouement of pt:main"],
            "target_tension": 0.4,
            "entities_to_surface": [], "theme_emphasis": [],
        },
        {
            "chapter_index": 5, "pov_character_id": "char:hero",
            "location_constraint": None,
            "dramatic_question": "What remains?",
            "required_plot_beats": ["Close on pt:main"],
            "target_tension": 0.2,
            "entities_to_surface": [], "theme_emphasis": [],
        },
    ],
    "hook_schedule": [
        {"hook_id": "fs:1", "planted_by_chapter": 1, "paid_off_by_chapter": 3},
    ],
    "theme_arc": [],
})


@pytest.mark.asyncio
async def test_generate_persists_skeleton(tmp_path):
    world = _make_world(tmp_path)
    cand = world.get_story_candidate("cand_1")
    planner = ArcSkeletonPlanner(FakeClient(CANNED), PromptRenderer(Path("prompts")))

    skel = await planner.generate(world=world, candidate=cand)
    assert skel.candidate_id == "cand_1"
    assert len(skel.chapters) == 5
    assert skel.chapters[0].pov_character_id == "char:hero"
    assert skel.chapters[1].pov_character_id == "char:rival"
    # Monotonic renumbering
    assert [c.chapter_index for c in skel.chapters] == [1, 2, 3, 4, 5]
    # Persisted
    looked_up = world.get_skeleton_for_candidate("cand_1")
    assert looked_up is not None
    assert looked_up.id == skel.id


@pytest.mark.asyncio
async def test_closed_enums_in_schema(tmp_path):
    world = _make_world(tmp_path)
    cand = world.get_story_candidate("cand_1")
    client = FakeClient(CANNED)
    planner = ArcSkeletonPlanner(client, PromptRenderer(Path("prompts")))
    await planner.generate(world=world, candidate=cand)

    _, schema, _ = client.calls[0]
    item = schema["properties"]["chapters"]["items"]
    pov_options = item["properties"]["pov_character_id"]["anyOf"]
    # anyOf: [{enum: [...]}, {type: null}]
    enum_variant = next(v for v in pov_options if "enum" in v)
    assert sorted(enum_variant["enum"]) == ["char:hero", "char:rival"]


def test_validate_coverage_detects_missing_hook():
    skel = ArcSkeleton(
        id="sk", candidate_id="c", quest_id="q",
        chapters=[
            SkeletonChapter(
                chapter_index=1, pov_character_id="char:hero",
                dramatic_question="?", required_plot_beats=["pt:main beat"],
                target_tension=0.5,
            )
        ],
        hook_schedule=[],  # fs:1 NOT scheduled
    )
    cand = StoryCandidate(
        id="c", quest_id="q", title="t", synopsis="s",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:hero", emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=1,
    )
    issues = validate_skeleton_coverage(skel, cand, all_hook_ids=["fs:1"])
    assert any("fs:1" in i for i in issues)


def test_validate_coverage_detects_missing_thread():
    skel = ArcSkeleton(
        id="sk", candidate_id="c", quest_id="q",
        chapters=[
            SkeletonChapter(
                chapter_index=1, pov_character_id="char:hero",
                dramatic_question="?",
                required_plot_beats=["something unrelated"],  # no pt:main mention
                target_tension=0.5,
            )
        ],
    )
    cand = StoryCandidate(
        id="c", quest_id="q", title="t", synopsis="s",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:hero", emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=1,
    )
    issues = validate_skeleton_coverage(skel, cand, all_hook_ids=[])
    assert any("pt:main" in i for i in issues)


def test_validate_coverage_passes_when_good():
    skel = ArcSkeleton(
        id="sk", candidate_id="c", quest_id="q",
        chapters=[
            SkeletonChapter(
                chapter_index=1, pov_character_id="char:hero",
                dramatic_question="?", required_plot_beats=["pay off pt:main"],
                target_tension=0.5,
            )
        ],
        hook_schedule=[HookPlacement(hook_id="fs:1", paid_off_by_chapter=1)],
    )
    cand = StoryCandidate(
        id="c", quest_id="q", title="t", synopsis="s",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:hero", emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=1,
    )
    issues = validate_skeleton_coverage(skel, cand, all_hook_ids=["fs:1"])
    assert issues == []
