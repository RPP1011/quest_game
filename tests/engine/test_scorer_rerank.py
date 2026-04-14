"""Day 3: Scorer-driven rerank.

When a ``Scorer`` is wired on the pipeline, ``_generate_scene_candidates``
must:

1. Use the scorer (Day 2's 12-dim scorecard) to score each candidate.
2. Pick the winner by weighted sum over those 12 dims.
3. Log every candidate's dim-level scores + the winning index under a
   single ``write_rerank`` trace stage.
4. Respect the ``rerank_weights`` ctor kwarg (and quest_config override)
   so callers can bias the sum toward specific dims.
5. Fall back to the legacy ``_score_candidate`` path when no ``Scorer``
   is wired (strict back-compat).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import (
    DEFAULT_SCORER_RERANK_WEIGHTS,
    Pipeline,
)
from app.engine.trace import PipelineTrace
from app.planning.schemas import CraftBrief, CraftPlan, CraftScenePlan
from app.scoring import DIMENSION_NAMES, Scorecard, Scorer
from app.world import StateDelta, WorldStateManager
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class VariantClient:
    """Returns pre-staged prose, records temperature + seed per call."""

    def __init__(self, prose_per_call: list[str]) -> None:
        self._prose = list(prose_per_call)
        self.calls: list[dict[str, Any]] = []

    async def chat_structured(self, **_: Any) -> str:
        raise AssertionError("chat_structured should not be called")

    async def chat(self, *, messages, **kw) -> str:
        self.calls.append({"temperature": kw.get("temperature"),
                           "seed": kw.get("seed")})
        return self._prose.pop(0)


class StubScorer:
    """Returns a hardcoded scorecard based on prose prefix.

    Lets us pin exact dim scores for deterministic rerank assertions
    without depending on the behavior of heuristics/critics.
    """

    def __init__(self, prose_to_dims: dict[str, dict[str, float]]) -> None:
        self._map = prose_to_dims
        self.calls: list[str] = []

    def score(self, prose: str, **_: Any) -> Scorecard:
        self.calls.append(prose)
        # Match by prefix for robustness.
        for prefix, dims in self._map.items():
            if prose.startswith(prefix):
                filled = {name: 0.5 for name in DIMENSION_NAMES}
                filled.update(dims)
                overall = sum(filled.values()) / len(filled)
                return Scorecard(overall_score=overall, **filled)
        # Default: neutral scorecard
        filled = {name: 0.5 for name in DIMENSION_NAMES}
        return Scorecard(overall_score=0.5, **filled)


def _make_world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(), update_number=1)
    return sm, conn


def _make_pipeline(world, client, **kw):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    return Pipeline(world, cb, client, **kw)


def _trace() -> PipelineTrace:
    return PipelineTrace(trace_id=uuid.uuid4().hex, trigger="test")


def _one_scene_plan() -> CraftPlan:
    return CraftPlan(
        scenes=[CraftScenePlan(scene_id=1)],
        briefs=[CraftBrief(scene_id=1, brief="Write something.")],
    )


# ---------------------------------------------------------------------------


async def test_scorer_rerank_picks_candidate_with_highest_weighted_sum(tmp_path):
    """Three candidates with pinned scorecards. Equal-weighted winner is
    the one with the highest mean dim score."""
    world, conn = _make_world(tmp_path)
    try:
        prose_a = "alpha: you move cautiously forward."
        prose_b = "beta: you move boldly forward."
        prose_c = "gamma: you stay perfectly still."
        client = VariantClient([prose_a, prose_b, prose_c])

        # Pin dim scores so beta has the best overall mean.
        stub = StubScorer({
            "alpha": {"pov_adherence": 0.6, "action_fidelity": 0.6},
            "beta": {"pov_adherence": 0.9, "action_fidelity": 0.95,
                     "sentence_variance": 0.9, "sensory_density": 0.9},
            "gamma": {"pov_adherence": 0.4, "action_fidelity": 0.3},
        })
        pipeline = _make_pipeline(world, client, n_candidates=3, scorer=stub)
        trace = _trace()

        result = await pipeline._run_write(
            trace, _one_scene_plan(), player_action="advance"
        )

        assert result == prose_b

        # Each candidate got scored exactly once.
        assert len(stub.calls) == 3

        # Rerank stage present and annotated with the scorer source.
        reranks = [s for s in trace.stages if s.stage_name == "write_rerank"]
        assert len(reranks) == 1
        rr = reranks[0].detail
        assert rr["winner_index"] == 1
        assert rr["rerank_source"] == "scorer"
        # Rerank logs every candidate's dim-level scores.
        assert len(rr["candidates"]) == 3
        for cand in rr["candidates"]:
            assert "dimension_scores" in cand
            assert set(cand["dimension_scores"].keys()) == set(DIMENSION_NAMES)

        # Per-write stages carry scorer provenance too.
        writes = [s for s in trace.stages if s.stage_name == "write"]
        for w in writes:
            assert w.detail["rerank_source"] == "scorer"
            assert "overall_score" in w.detail
            assert set(w.detail["dimension_scores"].keys()) == set(DIMENSION_NAMES)
    finally:
        conn.close()


async def test_scorer_rerank_weights_bias_winner(tmp_path):
    """Overriding ``rerank_weights`` with a heavy weight on a single dim
    must flip the winner to the candidate strongest on that dim, even
    when its unweighted mean is lower."""
    world, conn = _make_world(tmp_path)
    try:
        # Alpha is strong on pov_adherence, weak elsewhere.
        # Beta is uniformly medium.
        prose_a = "alpha: you stand firm."
        prose_b = "beta: you move forward."
        client = VariantClient([prose_a, prose_b])

        stub = StubScorer({
            "alpha": {
                "pov_adherence": 1.0,
                # All others 0.3 (overridden below via StubScorer default)
                **{n: 0.3 for n in DIMENSION_NAMES if n != "pov_adherence"},
            },
            "beta": {n: 0.6 for n in DIMENSION_NAMES},
        })

        # First: default weights → beta wins (higher mean).
        p1 = _make_pipeline(world, VariantClient([prose_a, prose_b]),
                            n_candidates=2, scorer=stub)
        t1 = _trace()
        r1 = await p1._run_write(t1, _one_scene_plan(), player_action="x")
        assert r1 == prose_b

        # Second: weight pov_adherence 20x → alpha wins.
        biased = {name: 1.0 for name in DEFAULT_SCORER_RERANK_WEIGHTS}
        biased["pov_adherence"] = 20.0
        p2 = _make_pipeline(world, VariantClient([prose_a, prose_b]),
                            n_candidates=2, scorer=stub,
                            rerank_weights=biased)
        t2 = _trace()
        r2 = await p2._run_write(t2, _one_scene_plan(), player_action="x")
        assert r2 == prose_a

        # Third: same bias via quest_config path (exercised channel).
        p3 = _make_pipeline(
            world, VariantClient([prose_a, prose_b]),
            n_candidates=2, scorer=stub,
            quest_config={"rerank_weights": {"pov_adherence": 20.0}},
        )
        t3 = _trace()
        r3 = await p3._run_write(t3, _one_scene_plan(), player_action="x")
        assert r3 == prose_a
    finally:
        conn.close()


async def test_fallback_to_legacy_when_no_scorer(tmp_path):
    """Back-compat: if no scorer is wired, use the legacy critic-sum
    ``_score_candidate`` and emit ``rerank_source = legacy``."""
    world, conn = _make_world(tmp_path)
    try:
        # POV-clean candidate vs POV-drift candidate. Legacy critic
        # ``validate_pov_adherence`` should penalize the first-person
        # drift and pick the second.
        prose = [
            "I walked alone. I felt hollow. I kept my eyes down.",
            "you walk steady. you watch the room. you wait.",
        ]
        client = VariantClient(prose)
        pipeline = _make_pipeline(world, client, n_candidates=2)  # no scorer
        trace = _trace()

        result = await pipeline._run_write(
            trace, _one_scene_plan(), player_action="wait"
        )
        assert result == prose[1]

        reranks = [s for s in trace.stages if s.stage_name == "write_rerank"]
        assert reranks[0].detail["rerank_source"] == "legacy"

        # Per-write stages too
        writes = [s for s in trace.stages if s.stage_name == "write"]
        for w in writes:
            assert w.detail["rerank_source"] == "legacy"
    finally:
        conn.close()


async def test_scorer_not_called_when_n_is_one(tmp_path):
    """With n_candidates=1 the fast path skips rerank entirely — the
    scorer should not be called from the write path even when wired."""
    world, conn = _make_world(tmp_path)
    try:
        client = VariantClient(["you stand still."])
        stub = StubScorer({})
        pipeline = _make_pipeline(world, client, n_candidates=1, scorer=stub)
        trace = _trace()
        await pipeline._run_write(trace, _one_scene_plan(), player_action="x")

        # No rerank happened → scorer not invoked from write.
        assert stub.calls == []
        writes = [s for s in trace.stages if s.stage_name == "write"]
        reranks = [s for s in trace.stages if s.stage_name == "write_rerank"]
        assert len(writes) == 1
        assert len(reranks) == 0
    finally:
        conn.close()


async def test_default_scorer_rerank_weights_all_ones():
    """Each of the 12 dims defaults to weight 1.0."""
    assert set(DEFAULT_SCORER_RERANK_WEIGHTS.keys()) == set(DIMENSION_NAMES)
    for k, v in DEFAULT_SCORER_RERANK_WEIGHTS.items():
        assert v == 1.0, f"{k} default weight = {v}, expected 1.0"


async def test_real_scorer_end_to_end_picks_cleaner_candidate(tmp_path):
    """Integration: use the real ``Scorer`` (no stub) and verify that a
    candidate without action-fidelity / sensory-density violations beats
    a weaker one."""
    world, conn = _make_world(tmp_path)
    try:
        prose = [
            # Candidate 0: vague, ignores the player action entirely
            "The weather changed.",
            # Candidate 1: echoes the action, sensory detail, clean POV
            "You slip into the corridor and listen for the steps.",
        ]
        client = VariantClient(prose)
        pipeline = _make_pipeline(
            world, client, n_candidates=2, scorer=Scorer()
        )
        trace = _trace()
        result = await pipeline._run_write(
            trace, _one_scene_plan(),
            player_action="slip into the corridor and listen for steps",
        )
        assert result == prose[1]

        reranks = [s for s in trace.stages if s.stage_name == "write_rerank"]
        assert reranks[0].detail["rerank_source"] == "scorer"
        assert reranks[0].detail["winner_index"] == 1

        # Each candidate logged its overall_score
        writes = [s for s in trace.stages if s.stage_name == "write"]
        for w in writes:
            assert 0.0 <= w.detail["overall_score"] <= 1.0
    finally:
        conn.close()
