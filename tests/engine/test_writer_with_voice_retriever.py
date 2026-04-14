"""Wave 4c: writer integration with :class:`VoiceRetriever`.

Verifies that when a voice retriever is wired in and
``quest_config.retrieval.enabled`` is ``True``, ``_run_write`` pulls
per-POV-character past utterances and injects them into the write prompt
under a ``CHARACTER VOICE`` section. Default-off behavior (retriever
``None`` or flag ``False``) is a strict regression: no voice-continuity
block renders.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.engine.trace import PipelineTrace
from app.planning.schemas import (
    ActionResolution,
    CraftBrief,
    CraftPlan,
    CraftScenePlan,
    DramaticPlan,
    DramaticScene,
)
from app.retrieval.interface import Query, Result
from app.world import Entity, EntityType, StateDelta, WorldStateManager
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CaptureClient:
    """Capture each write prompt and return canned prose per call."""

    def __init__(self, prose_per_call: list[str]) -> None:
        self._prose = list(prose_per_call)
        self.prompts: list[str] = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str:
        raise AssertionError("chat_structured should not be called")

    async def chat(self, *, messages, **kw) -> str:
        user_msg = next((m.content for m in messages if m.role == "user"), "")
        self.prompts.append(user_msg)
        return self._prose.pop(0)


class FakeVoiceRetriever:
    """Returns a fixed list of ``Result`` objects and records the query."""

    def __init__(self, results: list[Result]) -> None:
        self._results = list(results)
        self.queries: list[Query] = []

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        self.queries.append(query)
        return list(self._results[:k])


def _make_world(tmp_path: Path) -> tuple[WorldStateManager, Any]:
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(
            id="alice", entity_type=EntityType.CHARACTER, name="Alice",
        )),
    ]), update_number=1)
    return sm, conn


def _make_context_builder(world: WorldStateManager) -> ContextBuilder:
    return ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())


def _make_trace() -> PipelineTrace:
    return PipelineTrace(trace_id=uuid.uuid4().hex, trigger="t")


def _one_scene_plan() -> CraftPlan:
    scene = CraftScenePlan(scene_id=1)
    brief = CraftBrief(
        scene_id=1,
        brief="Scene one vision: Alice waits at the bridge.",
    )
    return CraftPlan(scenes=[scene], briefs=[brief])


def _dramatic_with_pov(pov: str = "alice") -> DramaticPlan:
    return DramaticPlan(
        action_resolution=ActionResolution(kind="partial", narrative="wait"),
        scenes=[DramaticScene(
            scene_id=1,
            pov_character_id=pov,
            characters_present=[pov],
            dramatic_question="Will she wait?",
            outcome="She waits.",
            beats=["wait"],
            dramatic_function="setup",
        )],
        update_tension_target=0.5,
        ending_hook="Arrival.",
        suggested_choices=[],
    )


def _voice_hit(
    character_id: str,
    text: str,
    *,
    idx: int = 0,
    update_number: int = 3,
    score: float = 0.99,
) -> Result:
    return Result(
        source_id=f"voice/q1/{character_id}/{idx}",
        text=text,
        score=score,
        metadata={
            "character_id": character_id,
            "source_update_number": update_number,
            "position": idx,
        },
    )


# ---------------------------------------------------------------------------
# Writer integration tests
# ---------------------------------------------------------------------------


async def test_write_injects_voice_continuity_when_retrieval_enabled(
    tmp_path: Path,
) -> None:
    """With a voice retriever + flag on, past utterances appear in the prompt."""
    world, conn = _make_world(tmp_path)
    try:
        quote = "I am not the one who left the key."
        retriever = FakeVoiceRetriever([_voice_hit("alice", quote, update_number=7)])
        client = CaptureClient(["Prose output."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            voice_retriever=retriever,
            quest_id="q1",
            quest_config={"retrieval": {"enabled": True}},
        )
        pipeline._last_dramatic = _dramatic_with_pov("alice")

        await pipeline._run_write(_make_trace(), _one_scene_plan())

        assert len(client.prompts) == 1
        prompt = client.prompts[0]
        assert "CHARACTER VOICE" in prompt
        assert quote in prompt
        # The update marker from the metadata is rendered into the line.
        assert "(update 7)" in prompt

        # Retriever was queried with the right character_id filter.
        assert len(retriever.queries) == 1
        q = retriever.queries[0]
        assert q.filters.get("character_id") == "alice"
        assert q.filters.get("last_n_records") == 30
    finally:
        conn.close()


async def test_write_omits_voice_continuity_when_retriever_none(
    tmp_path: Path,
) -> None:
    """No voice retriever → no CHARACTER VOICE block. Strict regression."""
    world, conn = _make_world(tmp_path)
    try:
        client = CaptureClient(["Prose output."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(world, cb, client)  # no voice_retriever, no flag

        await pipeline._run_write(_make_trace(), _one_scene_plan())

        assert "CHARACTER VOICE" not in client.prompts[0]
    finally:
        conn.close()


async def test_write_omits_voice_continuity_when_flag_off(tmp_path: Path) -> None:
    """Voice retriever wired but flag off → no retrieval call, no block."""
    world, conn = _make_world(tmp_path)
    try:
        retriever = FakeVoiceRetriever([_voice_hit("alice", "A past line.")])
        client = CaptureClient(["Prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            voice_retriever=retriever,
            quest_id="q1",
            quest_config={},
        )
        pipeline._last_dramatic = _dramatic_with_pov("alice")

        await pipeline._run_write(_make_trace(), _one_scene_plan())

        assert "CHARACTER VOICE" not in client.prompts[0]
        assert len(retriever.queries) == 0
    finally:
        conn.close()


async def test_write_omits_voice_continuity_when_no_pov(tmp_path: Path) -> None:
    """Scene has no resolvable POV → retriever is not called, no block renders."""
    world, conn = _make_world(tmp_path)
    try:
        retriever = FakeVoiceRetriever([_voice_hit("alice", "A past line.")])
        client = CaptureClient(["Prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            voice_retriever=retriever,
            quest_id="q1",
            quest_config={"retrieval": {"enabled": True}},
        )
        # No _last_dramatic stashed → _scene_pov_character_id returns None.

        await pipeline._run_write(_make_trace(), _one_scene_plan())

        assert "CHARACTER VOICE" not in client.prompts[0]
        assert len(retriever.queries) == 0
    finally:
        conn.close()


async def test_voice_retriever_failure_does_not_break_write(tmp_path: Path) -> None:
    """A retriever that raises must not take down the write stage."""
    world, conn = _make_world(tmp_path)
    try:
        class BrokenRetriever:
            async def retrieve(self, query, *, k=3):
                raise RuntimeError("boom")

        client = CaptureClient(["Prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            voice_retriever=BrokenRetriever(),
            quest_id="q1",
            quest_config={"retrieval": {"enabled": True}},
        )
        pipeline._last_dramatic = _dramatic_with_pov("alice")

        result = await pipeline._run_write(_make_trace(), _one_scene_plan())
        assert result == "Prose."
        assert "CHARACTER VOICE" not in client.prompts[0]
    finally:
        conn.close()
