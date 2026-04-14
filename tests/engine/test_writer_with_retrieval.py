"""Wave 2b: writer integration with ``PassageRetriever``.

Verifies that when a retriever is wired in and ``quest_config.retrieval.enabled``
is ``True``, ``_run_write`` pulls voice anchors and injects them into the
write prompt under a ``VOICE ANCHORS`` section. Default-off behavior
(retriever is ``None`` or flag is ``False``) is a strict regression: no
anchor block renders, and the behavior is bit-for-bit identical to the
pre-Wave-2b baseline.
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
    CraftBrief,
    CraftPlan,
    CraftScenePlan,
    EmotionalPlan,
    EmotionalScenePlan,
    VoicePermeability,
)
from app.retrieval.interface import Query, Result
from app.world import StateDelta, WorldStateManager
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


class FakePassageRetriever:
    """Returns a fixed list of ``Result`` objects and records the query."""

    def __init__(self, results: list[Result]) -> None:
        self._results = list(results)
        self.queries: list[Query] = []

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        self.queries.append(query)
        return list(self._results[:k])


def _make_world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(), update_number=1)
    return sm, conn


def _make_context_builder(world: WorldStateManager) -> ContextBuilder:
    return ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())


def _make_trace() -> PipelineTrace:
    return PipelineTrace(trace_id=uuid.uuid4().hex, trigger="t")


def _one_scene_plan(with_permeability: bool = False) -> CraftPlan:
    if with_permeability:
        scene = CraftScenePlan(
            scene_id=1,
            voice_permeability=VoicePermeability(baseline=0.4, current_target=0.5),
        )
    else:
        scene = CraftScenePlan(scene_id=1)
    brief = CraftBrief(
        scene_id=1,
        brief="Scene one vision: tense, grounded, sensory — cross the bridge.",
    )
    return CraftPlan(scenes=[scene], briefs=[brief])


def _fake_anchor(
    source_id: str,
    text: str,
    *,
    pov: str = "second",
    score: float = 0.92,
) -> Result:
    return Result(
        source_id=source_id,
        text=text,
        score=score,
        metadata={"pov": pov, "work_id": source_id.split("/")[0]},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_write_injects_voice_anchors_when_retrieval_enabled(tmp_path):
    """With retriever + enabled=True, anchor text appears in the write prompt."""
    world, conn = _make_world(tmp_path)
    try:
        anchor_text = (
            "You stood at the edge of the pier, the water below a slate tongue "
            "licking at the pilings. The wind had opinions you could not ignore."
        )
        retriever = FakePassageRetriever(
            [_fake_anchor("shelley_pulp/p0", anchor_text, pov="second", score=0.88)]
        )
        client = CaptureClient(["The bridge held. Barely."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            passage_retriever=retriever,
            quest_config={"retrieval": {"enabled": True}},
        )

        craft_plan = _one_scene_plan()
        await pipeline._run_write(_make_trace(), craft_plan)

        assert len(client.prompts) == 1
        prompt = client.prompts[0]
        # Anchor section header present
        assert "VOICE ANCHORS" in prompt
        # Anchor body (first ~words) embedded in prompt
        # We don't require the full 300-word slice to be present verbatim;
        # the anchor is short so the whole text should render.
        assert "slate tongue" in prompt
        assert "pov: second" in prompt
        # Retriever was queried exactly once (one scene)
        assert len(retriever.queries) == 1
    finally:
        conn.close()


async def test_write_omits_voice_anchors_when_retriever_none(tmp_path):
    """No retriever → no VOICE ANCHORS block. Strict regression check."""
    world, conn = _make_world(tmp_path)
    try:
        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(world, cb, client)  # no passage_retriever, no flag

        craft_plan = _one_scene_plan()
        await pipeline._run_write(_make_trace(), craft_plan)

        assert "VOICE ANCHORS" not in client.prompts[0]
    finally:
        conn.close()


async def test_write_omits_voice_anchors_when_flag_off(tmp_path):
    """Retriever wired but flag off → no retrieval call, no block.

    Success-criteria check: callers that opt in to a retriever but don't flip
    ``retrieval.enabled`` see bit-for-bit pre-Wave-2b behavior.
    """
    world, conn = _make_world(tmp_path)
    try:
        retriever = FakePassageRetriever(
            [_fake_anchor("work/p0", "anchor body text.", pov="second")]
        )
        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            passage_retriever=retriever,
            # flag defaults to False via missing "retrieval" key
            quest_config={},
        )

        craft_plan = _one_scene_plan()
        await pipeline._run_write(_make_trace(), craft_plan)

        assert "VOICE ANCHORS" not in client.prompts[0]
        # Retriever MUST NOT be called when flag is off.
        assert len(retriever.queries) == 0
    finally:
        conn.close()


async def test_voice_anchor_query_carries_permeability_and_pov(tmp_path):
    """Query filters include voice_distinctiveness + free_indirect_quality
    (when permeability is set) and a POV filter."""
    world, conn = _make_world(tmp_path)
    try:
        retriever = FakePassageRetriever([
            _fake_anchor("work/p0", "anchor body.", pov="second", score=0.80),
        ])
        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            passage_retriever=retriever,
            quest_config={"retrieval": {"enabled": True}},
        )

        craft_plan = _one_scene_plan(with_permeability=True)
        # Stash an emotional plan so the seed_text picks it up.
        pipeline._last_emotional = EmotionalPlan(
            scenes=[EmotionalScenePlan(
                scene_id=1,
                primary_emotion="dread",
                intensity=0.7,
                entry_state="alert",
                exit_state="shaken",
                transition_type="escalation",
                emotional_source="the drop below",
            )],
            update_emotional_arc="dread rising",
            contrast_strategy="silence before impact",
        )

        await pipeline._run_write(_make_trace(), craft_plan)

        assert len(retriever.queries) == 1
        q = retriever.queries[0]
        assert q.filters.get("pov") is not None
        ranges = q.filters.get("score_ranges") or {}
        assert "voice_distinctiveness" in ranges
        assert ranges["voice_distinctiveness"] == (0.7, 1.0)
        # Permeability target = 0.5 → free_indirect_quality = (0.3, 0.7).
        fiq = ranges.get("free_indirect_quality")
        assert fiq is not None
        lo, hi = fiq
        assert lo == pytest.approx(0.3)
        assert hi == pytest.approx(0.7)
        # Seed text includes the brief and the target emotion marker.
        assert q.seed_text is not None
        assert "dread" in q.seed_text
        assert len(q.seed_text) <= 600
    finally:
        conn.close()


async def test_retrieval_failure_does_not_break_write(tmp_path):
    """A retriever that raises must not take down the write stage."""
    world, conn = _make_world(tmp_path)
    try:
        class BrokenRetriever:
            async def retrieve(self, query, *, k=3):
                raise RuntimeError("boom")

        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            passage_retriever=BrokenRetriever(),
            quest_config={"retrieval": {"enabled": True}},
        )

        craft_plan = _one_scene_plan()
        result = await pipeline._run_write(_make_trace(), craft_plan)
        assert result == "prose."
        assert "VOICE ANCHORS" not in client.prompts[0]
    finally:
        conn.close()
