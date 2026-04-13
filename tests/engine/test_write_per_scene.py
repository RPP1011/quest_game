"""Tests for _run_write with CraftPlan (per-scene, brief-first) and backwards-compat shim."""
from __future__ import annotations
import uuid
from pathlib import Path
import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.engine.trace import PipelineTrace
from app.planning.schemas import (
    CraftBrief,
    CraftPlan,
    CraftScenePlan,
)
from app.world import StateDelta, WorldStateManager
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SceneProseClient:
    """Returns distinct prose per chat() call; ignores structured calls."""

    def __init__(self, prose_per_call: list[str]) -> None:
        self._prose = list(prose_per_call)
        self.prompts: list[str] = []  # user prompts captured

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str:
        raise AssertionError("chat_structured should not be called in these tests")

    async def chat(self, *, messages, **kw) -> str:
        # capture the user prompt content
        user_msg = next((m.content for m in messages if m.role == "user"), "")
        self.prompts.append(user_msg)
        return self._prose.pop(0)


def _make_world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(), update_number=1)
    return sm, conn


def _make_pipeline(world, client):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    return Pipeline(world, cb, client)


def _make_trace() -> PipelineTrace:
    return PipelineTrace(trace_id=uuid.uuid4().hex, trigger="test")


def _two_scene_plan(brief1="Scene one vision.", brief2="Scene two vision.") -> CraftPlan:
    scene1 = CraftScenePlan(scene_id=1)
    scene2 = CraftScenePlan(scene_id=2)
    brief_obj1 = CraftBrief(scene_id=1, brief=brief1)
    brief_obj2 = CraftBrief(scene_id=2, brief=brief2)
    return CraftPlan(scenes=[scene1, scene2], briefs=[brief_obj1, brief_obj2])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_write_generates_one_prose_block_per_scene(tmp_path):
    """Two-scene CraftPlan → two chat() calls, final prose is concatenation."""
    world, conn = _make_world(tmp_path)
    try:
        prose_a = "You stepped into the dim corridor."
        prose_b = "The door slid shut behind you."
        client = SceneProseClient([prose_a, prose_b])
        pipeline = _make_pipeline(world, client)
        trace = _make_trace()

        craft_plan = _two_scene_plan()
        result = await pipeline._run_write(trace, craft_plan)

        # Two chat() calls were made
        assert len(client.prompts) == 2

        # Final prose is both blocks joined
        assert prose_a in result
        assert prose_b in result
        assert result == f"{prose_a}\n\n{prose_b}"

        # Two StageResult entries, both stage_name="write"
        write_stages = [s for s in trace.stages if s.stage_name == "write"]
        assert len(write_stages) == 2

        # Each stage records its scene_id in detail
        assert write_stages[0].detail["scene_id"] == 1
        assert write_stages[1].detail["scene_id"] == 2
    finally:
        conn.close()


async def test_write_includes_brief_in_prompt(tmp_path):
    """The CraftBrief.brief text must appear in the user prompt for each scene."""
    world, conn = _make_world(tmp_path)
    try:
        brief1 = "The air tasted of copper and regret."
        brief2 = "Rain hammered the cobblestones like accusation."
        client = SceneProseClient(["prose for scene 1", "prose for scene 2"])
        pipeline = _make_pipeline(world, client)
        trace = _make_trace()

        craft_plan = _two_scene_plan(brief1=brief1, brief2=brief2)
        await pipeline._run_write(trace, craft_plan)

        # Brief text must appear in the corresponding user prompt
        assert brief1 in client.prompts[0], "brief1 should be in first scene user prompt"
        assert brief2 in client.prompts[1], "brief2 should be in second scene user prompt"
    finally:
        conn.close()


async def test_write_falls_back_for_old_plan_dict(tmp_path):
    """Passing {'beats': [...]} uses old single-call path and returns prose once."""
    world, conn = _make_world(tmp_path)
    try:
        old_prose = "The stranger watched you from across the room."
        client = SceneProseClient([old_prose])
        pipeline = _make_pipeline(world, client)
        trace = _make_trace()

        old_plan = {"beats": ["The stranger notices you.", "You lock eyes."], "suggested_choices": []}
        result = await pipeline._run_write(trace, old_plan)

        # Exactly one chat() call
        assert len(client.prompts) == 1

        # Result is the prose (possibly trimmed by parse_prose)
        assert old_prose in result

        # Exactly one StageResult
        write_stages = [s for s in trace.stages if s.stage_name == "write"]
        assert len(write_stages) == 1
    finally:
        conn.close()


async def test_write_recent_prose_tail_passed_to_second_scene(tmp_path):
    """After scene 1, recent_prose_tail from scene 1 prose is threaded into scene 2 prompt."""
    world, conn = _make_world(tmp_path)
    try:
        # Make scene 1 prose long enough to produce a non-empty tail
        long_prose_1 = "A" * 400
        prose_2 = "Scene two content."
        client = SceneProseClient([long_prose_1, prose_2])
        pipeline = _make_pipeline(world, client)
        trace = _make_trace()

        craft_plan = _two_scene_plan()
        await pipeline._run_write(trace, craft_plan)

        # The second prompt should contain part of scene 1's prose (last ~300 chars)
        tail_fragment = long_prose_1[-100:]  # definitely in the last 300
        assert tail_fragment in client.prompts[1], (
            "scene 2 prompt should include trailing content from scene 1 prose"
        )
    finally:
        conn.close()


async def test_write_scene_without_brief_does_not_crash(tmp_path):
    """A CraftPlan where no brief exists for a scene should still run (brief is omitted)."""
    world, conn = _make_world(tmp_path)
    try:
        prose = "It worked anyway."
        client = SceneProseClient([prose])
        pipeline = _make_pipeline(world, client)
        trace = _make_trace()

        # No briefs list
        craft_plan = CraftPlan(scenes=[CraftScenePlan(scene_id=1)], briefs=[])
        result = await pipeline._run_write(trace, craft_plan)

        assert prose in result
        assert len(client.prompts) == 1
    finally:
        conn.close()
