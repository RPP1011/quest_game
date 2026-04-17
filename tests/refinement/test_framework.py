"""Refinement framework integration tests.

Mocks pipeline + scorer to verify the accept/reject logic and the
persistence path (rollout_chapter update on accept, refinement_attempt
always saved).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.refinement import framework as fw
from app.refinement.framework import (
    ACCEPT_MEAN_DELTA, REJECT_DIM_REGRESSION, RefinementTarget,
    _evaluate_deltas, run_refinement_pass,
)
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, PlotThread, QuestArcState, ReaderState,
    RolloutChapter, RolloutRun, StoryCandidate, ThreadStatus,
)
from app.world.state_manager import WorldStateManager


def test_evaluate_deltas_mean_and_min():
    baseline = {"a": 0.5, "b": 0.6, "c": 0.7}
    refined = {"a": 0.7, "b": 0.5, "c": 0.8}  # +0.2, -0.1, +0.1; mean=+0.067
    mean, mn = _evaluate_deltas(baseline, refined)
    assert mean == pytest.approx(0.0667, abs=1e-3)
    assert mn == pytest.approx(-0.1)


def test_evaluate_deltas_empty_intersection():
    mean, mn = _evaluate_deltas({"a": 0.5}, {"b": 0.5})
    assert mean == 0.0 and mn == 0.0


# ---- Integration test ------------------------------------------------------

def _init_quest_with_rollout(quests_dir: Path, qid: str = "q1") -> str:
    paths = quests_dir / qid
    paths.mkdir(parents=True, exist_ok=True)
    (paths / "traces").mkdir(exist_ok=True)
    db_path = paths / "quest.db"
    conn = open_db(db_path)
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(id="char:hero", entity_type=EntityType.CHARACTER, name="Hero"))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="m", description="x",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id=qid, title="T", synopsis="S",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:hero", emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=1,
    ))
    sm.upsert_arc(QuestArcState(
        quest_id=qid, arc_id="main", structure_id="three_act",
        scale="campaign", current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=qid))
    rid = "r1"
    sm.create_rollout(RolloutRun(
        id=rid, quest_id=qid, candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=1,
    ))
    sm.save_rollout_chapter(RolloutChapter(
        rollout_id=rid, chapter_index=1,
        player_action="orig action",
        prose="ORIGINAL PROSE that was weak.",
    ))
    sm.save_chapter_scores(rid, 1, {
        "a": {"score": 0.4, "rationale": "weak a"},
        "b": {"score": 0.5, "rationale": "ok b"},
    })
    conn.close()
    # Rollout dir setup
    rollout_dir = paths / "rollouts" / rid
    (rollout_dir / "traces").mkdir(parents=True, exist_ok=True)
    rollout_db = rollout_dir / "quest.db"
    rconn = open_db(rollout_db)
    # Minimal world content
    rsm = WorldStateManager(rconn)
    rsm.create_entity(Entity(id="char:hero", entity_type=EntityType.CHARACTER, name="Hero"))
    rconn.close()
    (rollout_dir / "config.json").write_text("{}")
    (paths / "config.json").write_text("{}")
    return rid


class FakeTrace:
    def __init__(self, tid: str = "trefined"):
        self.trace_id = tid
        self.outcome = "committed"
        self.total_latency_ms = 0
        self.trigger = ""
        self.timestamp = None

    def model_dump_json(self):
        return json.dumps({"trace_id": self.trace_id, "stages": []})


class FakePipeline:
    def __init__(self, prose: str):
        self._prose = prose

    async def run(self, *, player_action, update_number):
        return SimpleNamespace(
            prose=self._prose, choices=[], trace=FakeTrace(),
        )


@pytest.mark.asyncio
async def test_accept_when_better(tmp_path: Path):
    """Refined scores beat baseline by mean +0.05+, no regression: accepted."""
    quests = tmp_path / "quests"
    rid = _init_quest_with_rollout(quests)
    main_conn = open_db(quests / "q1" / "quest.db")
    main_sm = WorldStateManager(main_conn)
    target = RefinementTarget(
        rollout_id=rid, chapter_index=1, quest_id="q1",
        strategy="weak_chapter", reason="x", guidance="rewrite",
        baseline_scores={"a": 0.4, "b": 0.5},
    )
    refined_prose = "REFINED PROSE WITH RICH VOICE AND TENSION."

    async def fake_score(*, client, chapter_text, dims=None):
        # Refined: a=0.6 (+0.2), b=0.55 (+0.05) → mean +0.125, min +0.05
        return {
            "a": {"score": 0.6, "rationale": "improved"},
            "b": {"score": 0.55, "rationale": "ok"},
        }

    with patch.object(fw, "_build_pipeline_for_rollout",
                      lambda *a, **kw: (FakePipeline(refined_prose),
                                         WorldStateManager(open_db(quests / "q1" / "rollouts" / rid / "quest.db")),
                                         None)):
        with patch("app.refinement.framework.score_chapter", fake_score):
            results = await run_refinement_pass(
                targets=[target], quests_dir=quests,
                main_world=main_sm, client=SimpleNamespace(),
            )
    assert len(results) == 1
    r = results[0]
    assert r.accepted, r.rejection_reason
    assert r.delta_mean == pytest.approx(0.125)
    assert r.delta_min == pytest.approx(0.05)
    # Canonical chapter was updated
    chs = main_sm.list_rollout_chapters(rid)
    assert chs[0].prose == refined_prose
    assert chs[0].judge_scores == {"a": 0.6, "b": 0.55}
    # Attempt persisted
    attempts = main_sm.list_refinement_attempts(quest_id="q1")
    assert len(attempts) == 1
    assert attempts[0].accepted is True
    main_conn.close()


@pytest.mark.asyncio
async def test_reject_when_mean_below_threshold(tmp_path: Path):
    quests = tmp_path / "quests"
    rid = _init_quest_with_rollout(quests)
    main_conn = open_db(quests / "q1" / "quest.db")
    main_sm = WorldStateManager(main_conn)
    target = RefinementTarget(
        rollout_id=rid, chapter_index=1, quest_id="q1",
        strategy="weak_chapter", reason="x", guidance="g",
        baseline_scores={"a": 0.4, "b": 0.5},
    )

    async def fake_score(*, client, chapter_text, dims=None):
        # Tiny improvement (+0.02 mean), below ACCEPT threshold
        return {
            "a": {"score": 0.42, "rationale": ""},
            "b": {"score": 0.52, "rationale": ""},
        }

    with patch.object(fw, "_build_pipeline_for_rollout",
                      lambda *a, **kw: (FakePipeline("any"),
                                         WorldStateManager(open_db(quests / "q1" / "rollouts" / rid / "quest.db")),
                                         None)):
        with patch("app.refinement.framework.score_chapter", fake_score):
            results = await run_refinement_pass(
                targets=[target], quests_dir=quests,
                main_world=main_sm, client=SimpleNamespace(),
            )
    assert results[0].accepted is False
    assert "mean delta" in results[0].rejection_reason
    # Canonical chapter NOT updated
    chs = main_sm.list_rollout_chapters(rid)
    assert chs[0].prose == "ORIGINAL PROSE that was weak."
    main_conn.close()


@pytest.mark.asyncio
async def test_reject_when_dim_regresses(tmp_path: Path):
    quests = tmp_path / "quests"
    rid = _init_quest_with_rollout(quests)
    main_conn = open_db(quests / "q1" / "quest.db")
    main_sm = WorldStateManager(main_conn)
    target = RefinementTarget(
        rollout_id=rid, chapter_index=1, quest_id="q1",
        strategy="weak_chapter", reason="x", guidance="g",
        baseline_scores={"a": 0.4, "b": 0.7},
    )

    async def fake_score(*, client, chapter_text, dims=None):
        # Big improvement on a (+0.4), but b crashes (-0.5) → mean −0.05
        # mean would actually be (-0.5 + 0.4)/2 = -0.05 → also fails mean
        # Make mean OK but min regression too big:
        # Adjust baseline: a 0.4 → 0.9 (+0.5), b 0.7 → 0.5 (-0.2)
        # mean = +0.15, min = -0.2 → fails dim regression
        return {
            "a": {"score": 0.9, "rationale": ""},
            "b": {"score": 0.5, "rationale": ""},
        }

    with patch.object(fw, "_build_pipeline_for_rollout",
                      lambda *a, **kw: (FakePipeline("any"),
                                         WorldStateManager(open_db(quests / "q1" / "rollouts" / rid / "quest.db")),
                                         None)):
        with patch("app.refinement.framework.score_chapter", fake_score):
            results = await run_refinement_pass(
                targets=[target], quests_dir=quests,
                main_world=main_sm, client=SimpleNamespace(),
            )
    assert results[0].accepted is False
    assert "regression" in results[0].rejection_reason
    main_conn.close()
