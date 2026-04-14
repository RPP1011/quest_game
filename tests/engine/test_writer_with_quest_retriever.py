"""Wave 3b: writer integration with ``QuestRetriever``.

Verifies that when a quest retriever is wired in and
``quest_config.retrieval.enabled`` is ``True``, ``_run_write`` pulls
in-quest callbacks and injects them into the write prompt under an
``IN-QUEST CALLBACKS`` section. Default-off behavior (retriever is
``None`` or flag is ``False``) is a strict regression: no callback block
renders.

Also covers the Wave 1c → Wave 3b embedder activation: when a real
embedder is wired into ``Pipeline(retrieval_embedder=...)``, a synthetic
commit writes an embedding row to the ``narrative_embeddings`` table.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
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


class FakeQuestRetriever:
    """Returns a fixed list of ``Result`` objects and records the query."""

    def __init__(self, results: list[Result]) -> None:
        self._results = list(results)
        self.queries: list[Query] = []

    async def retrieve(self, query: Query, *, k: int = 2) -> list[Result]:
        self.queries.append(query)
        return list(self._results[:k])


class FakeEmbedder:
    """Deterministic embedder returning a fixed 384-dim vector per text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_one(self, text: str) -> np.ndarray:
        self.calls.append(text)
        # Simple hash-based deterministic vector for test assertions.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(384).astype(np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        return v / n


def _make_world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(
            id="alice", entity_type=EntityType.CHARACTER, name="Alice",
        )),
        EntityCreate(entity=Entity(
            id="bob", entity_type=EntityType.CHARACTER, name="Bob",
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
        brief="Scene one vision: Alice waits by the bridge, fearing Bob's return.",
    )
    return CraftPlan(scenes=[scene], briefs=[brief])


def _one_dramatic_plan() -> DramaticPlan:
    return DramaticPlan(
        action_resolution=ActionResolution(kind="partial", narrative="wait"),
        scenes=[DramaticScene(
            scene_id=1,
            pov_character_id="alice",
            characters_present=["alice", "bob"],
            dramatic_question="Will Alice confront Bob?",
            outcome="She waits.",
            beats=["Alice at the bridge."],
            dramatic_function="setup",
        )],
        update_tension_target=0.5,
        ending_hook="Bob arrives.",
        suggested_choices=[],
    )


def _fake_callback(
    quest_id: str,
    update_number: int,
    text: str,
    *,
    scene_index: int = 0,
    score: float = 0.9,
) -> Result:
    return Result(
        source_id=f"{quest_id}/{update_number}/{scene_index}",
        text=text,
        score=score,
        metadata={
            "quest_id": quest_id,
            "update_number": update_number,
            "scene_index": scene_index,
        },
    )


# ---------------------------------------------------------------------------
# Writer integration tests
# ---------------------------------------------------------------------------


async def test_write_injects_quest_callbacks_when_retrieval_enabled(tmp_path):
    """With a quest retriever + enabled=True, callback text appears in the prompt."""
    world, conn = _make_world(tmp_path)
    try:
        callback_text = (
            "Alice had stood in this same spot three nights earlier, watching "
            "the water slide under the pilings like a secret she was not ready to tell."
        )
        retriever = FakeQuestRetriever([
            _fake_callback("q1", 3, callback_text, scene_index=0, score=0.87),
        ])
        client = CaptureClient(["The bridge held."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            quest_retriever=retriever,
            quest_id="q1",
            quest_config={"retrieval": {"enabled": True}},
        )
        # Stash dramatic plan so entity mentions get collected.
        pipeline._last_dramatic = _one_dramatic_plan()

        craft_plan = _one_scene_plan()
        await pipeline._run_write(_make_trace(), craft_plan)

        assert len(client.prompts) == 1
        prompt = client.prompts[0]
        # Callback section header present
        assert "IN-QUEST CALLBACKS" in prompt
        # Callback body (distinctive phrase) embedded verbatim
        assert "secret she was not ready to tell" in prompt
        # Update number rendered
        assert "update 3" in prompt
        # Retriever was queried exactly once (one scene)
        assert len(retriever.queries) == 1
        q = retriever.queries[0]
        assert q.seed_text is not None
        assert "Alice" in q.seed_text
        # last_n_records cap present
        assert q.filters.get("last_n_records") == 12
        # Entity mentions collected: should include alice and/or bob ids.
        mentions = q.filters.get("entity_mentions") or set()
        assert "alice" in mentions
    finally:
        conn.close()


async def test_write_omits_callbacks_when_retriever_none(tmp_path):
    """No quest retriever → no IN-QUEST CALLBACKS block. Strict regression."""
    world, conn = _make_world(tmp_path)
    try:
        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(world, cb, client)  # no quest_retriever, no flag

        craft_plan = _one_scene_plan()
        await pipeline._run_write(_make_trace(), craft_plan)

        assert "IN-QUEST CALLBACKS" not in client.prompts[0]
    finally:
        conn.close()


async def test_write_omits_callbacks_when_flag_off(tmp_path):
    """Quest retriever wired but flag off → no retrieval call, no block."""
    world, conn = _make_world(tmp_path)
    try:
        retriever = FakeQuestRetriever([
            _fake_callback("q1", 2, "earlier body text.", score=0.8),
        ])
        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            quest_retriever=retriever,
            quest_id="q1",
            # flag defaults to False via missing "retrieval" key
            quest_config={},
        )

        craft_plan = _one_scene_plan()
        await pipeline._run_write(_make_trace(), craft_plan)

        assert "IN-QUEST CALLBACKS" not in client.prompts[0]
        assert len(retriever.queries) == 0
    finally:
        conn.close()


async def test_write_omits_callbacks_when_retriever_returns_empty(tmp_path):
    """Retriever returns [] (no hits) → no block, no crash."""
    world, conn = _make_world(tmp_path)
    try:
        retriever = FakeQuestRetriever([])
        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            quest_retriever=retriever,
            quest_id="q1",
            quest_config={"retrieval": {"enabled": True}},
        )

        craft_plan = _one_scene_plan()
        await pipeline._run_write(_make_trace(), craft_plan)

        assert "IN-QUEST CALLBACKS" not in client.prompts[0]
        # Still called once (scene level)
        assert len(retriever.queries) == 1
    finally:
        conn.close()


async def test_quest_retriever_failure_does_not_break_write(tmp_path):
    """A retriever that raises must not take down the write stage."""
    world, conn = _make_world(tmp_path)
    try:
        class BrokenRetriever:
            async def retrieve(self, query, *, k=2):
                raise RuntimeError("boom")

        client = CaptureClient(["prose."])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            quest_retriever=BrokenRetriever(),
            quest_id="q1",
            quest_config={"retrieval": {"enabled": True}},
        )

        craft_plan = _one_scene_plan()
        result = await pipeline._run_write(_make_trace(), craft_plan)
        assert result == "prose."
        assert "IN-QUEST CALLBACKS" not in client.prompts[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Quest retriever (real) end-to-end with persisted rows
# ---------------------------------------------------------------------------


async def test_quest_retriever_reads_persisted_rows(tmp_path):
    """Fixture with committed narrative_embeddings: retriever returns them."""
    from app.retrieval.quest_retriever import QuestRetriever

    world, conn = _make_world(tmp_path)
    try:
        embedder = FakeEmbedder()
        # Seed two persisted rows with known embeddings.
        txt_a = "Alice crossed the bridge in the moonlight; Bob was nowhere."
        txt_b = "A cold kitchen, unrelated to any character in question."
        world.upsert_narrative_embedding(
            quest_id="q1",
            update_number=1,
            scene_index=0,
            embedding=embedder.embed_one(txt_a),
            text_preview=txt_a,
        )
        world.upsert_narrative_embedding(
            quest_id="q1",
            update_number=2,
            scene_index=0,
            embedding=embedder.embed_one(txt_b),
            text_preview=txt_b,
        )

        qr = QuestRetriever(world, "q1", embedder=embedder)
        # Use a seed identical to txt_a → cosine ~= 1.0.
        results = await qr.retrieve(
            Query(seed_text=txt_a, filters={"last_n_records": 12}),
            k=2,
        )
        assert len(results) == 2
        # Top result is the one matching our seed text.
        assert "Alice" in results[0].text
        assert results[0].metadata["update_number"] == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Embedder activation (Wave 3b scope)
# ---------------------------------------------------------------------------


async def test_embedder_writes_row_on_synthetic_commit(tmp_path):
    """Passing a real embedder + quest_id writes a narrative_embeddings row."""
    world, conn = _make_world(tmp_path)
    try:
        embedder = FakeEmbedder()
        client = CaptureClient([])  # not used for this helper
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            retrieval_embedder=embedder,
            quest_id="q1",
        )

        prose = "A synthetic scene. Alice holds her breath."
        pipeline._persist_narrative_embedding(prose, update_number=7)

        rows = world.list_narrative_embeddings("q1")
        assert len(rows) == 1
        assert rows[0]["update_number"] == 7
        assert rows[0]["scene_index"] == 0
        assert rows[0]["text_preview"].startswith("A synthetic scene.")
        # Embedding vector is 384-dim float32 from our FakeEmbedder.
        assert rows[0]["embedding"].dtype == np.float32
        assert rows[0]["embedding"].shape == (384,)
        # Embedder was invoked exactly once for this commit.
        assert embedder.calls == [prose]
    finally:
        conn.close()


async def test_embedder_noop_without_quest_id(tmp_path):
    """No quest_id → no row written even if embedder is provided."""
    world, conn = _make_world(tmp_path)
    try:
        embedder = FakeEmbedder()
        client = CaptureClient([])
        cb = _make_context_builder(world)
        pipeline = Pipeline(
            world, cb, client,
            retrieval_embedder=embedder,
            # No quest_id → helper short-circuits.
        )
        pipeline._persist_narrative_embedding("some prose", update_number=1)
        # No quest id to key on, nothing persisted, embedder not invoked.
        assert embedder.calls == []
    finally:
        conn.close()


async def test_embedder_noop_without_embedder(tmp_path):
    """No embedder → no row written (default-off regression)."""
    world, conn = _make_world(tmp_path)
    try:
        client = CaptureClient([])
        cb = _make_context_builder(world)
        pipeline = Pipeline(world, cb, client, quest_id="q1")
        pipeline._persist_narrative_embedding("some prose", update_number=1)
        rows = world.list_narrative_embeddings("q1")
        assert rows == []
    finally:
        conn.close()
