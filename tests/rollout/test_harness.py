"""Harness tests.

We don't exercise the full LLM pipeline here; we unit-test the harness
logic: bootstrap, resume, incremental save, error handling. Full
end-to-end is covered by the small-scale verify task against a real
server.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.rollout import harness as harness_mod
from app.rollout.harness import (
    create_rollout_row, run_rollout, _bootstrap_rollout_world,
    _opening_action_from_candidate,
)
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, PlotThread, QuestArcState, ReaderState,
    RolloutStatus, StoryCandidate, ThreadStatus,
)
from app.world.state_manager import WorldStateManager


def _init_main_quest(quests_dir: Path, qid: str = "q1") -> Path:
    paths = quests_dir / qid
    paths.mkdir(parents=True, exist_ok=True)
    (paths / "traces").mkdir(exist_ok=True)
    db_path = paths / "quest.db"
    conn = open_db(db_path)
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(id="char:hero", entity_type=EntityType.CHARACTER, name="Hero"))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="Main", description="x",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id=qid, title="T",
        synopsis="Hero does the thing. Stakes rise. The thing is done.",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:hero", emphasized_theme_ids=[],
        climax_description="The thing is done.", expected_chapter_count=3,
    ))
    sm.upsert_arc(QuestArcState(
        quest_id=qid, arc_id="main", structure_id="three_act",
        scale="campaign", current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=qid))
    conn.close()
    (paths / "config.json").write_text(json.dumps({
        "genre": "test", "premise": "test",
        "picked_candidate": {"id": "cand_1", "title": "T", "synopsis": "s",
                             "primary_thread_ids": ["pt:main"]},
    }))
    return paths


def test_opening_action_from_skeleton_beats():
    cand = StoryCandidate(
        id="c", quest_id="q", title="T", synopsis="Long synopsis.",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=3,
    )
    action = _opening_action_from_candidate(cand, ["Hero enters the vault", "Hero finds the key"])
    assert "vault" in action
    assert "key" in action


def test_opening_action_falls_back_to_synopsis():
    cand = StoryCandidate(
        id="c", quest_id="q", title="T",
        synopsis="This is the first. And the rest follows.",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=3,
    )
    action = _opening_action_from_candidate(cand, None)
    assert action.startswith("This is the first")


def test_bootstrap_copies_and_wipes_narrative(tmp_path: Path):
    quests = tmp_path / "quests"
    paths = _init_main_quest(quests)
    # Add a narrative row to the main DB so we can confirm it's wiped
    conn = open_db(paths / "quest.db")
    conn.execute(
        "INSERT INTO narrative(update_number, raw_text, summary) VALUES (1, 'stale', 's')"
    )
    conn.commit()
    conn.close()

    rollout_dir = quests / "q1" / "rollouts" / "r1"
    _bootstrap_rollout_world(paths / "quest.db", rollout_dir, paths / "config.json")

    # Rollout DB should have entities but no narrative
    rc = open_db(rollout_dir / "quest.db")
    try:
        ent_count = rc.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        nar_count = rc.execute("SELECT COUNT(*) FROM narrative").fetchone()[0]
        assert ent_count >= 1
        assert nar_count == 0
    finally:
        rc.close()

    # Config copied
    assert (rollout_dir / "config.json").is_file()


def test_create_rollout_row(tmp_path: Path):
    quests = tmp_path / "quests"
    _init_main_quest(quests)
    rid = create_rollout_row(
        quests_dir=quests, quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=3,
    )
    assert rid.startswith("ro_")

    # Re-open main DB and verify
    conn = open_db(quests / "q1" / "quest.db")
    try:
        sm = WorldStateManager(conn)
        run = sm.get_rollout(rid)
        assert run.status == RolloutStatus.PENDING
        assert run.total_chapters_target == 3
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_run_rollout_happy_path_resume(tmp_path: Path):
    """Full rollout with mocked pipeline; verify incremental save + resume."""
    quests = tmp_path / "quests"
    paths = _init_main_quest(quests)
    rid = create_rollout_row(
        quests_dir=quests, quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=2,
    )

    # Counter so we can detect re-runs
    call_count = {"n": 0}

    class FakeTrace:
        def __init__(self, n):
            self.trace_id = f"trace_{n}"
            self.stages = []
            self.outcome = "committed"
            self.total_latency_ms = 0
            self.trigger = ""
            self.timestamp = None

    class FakeOutput:
        def __init__(self, prose, choices, trace):
            self.prose = prose
            self.choices = choices
            self.trace = trace

    class FakePipeline:
        def __init__(self, *a, **kw):
            pass

        async def run(self, *, player_action, update_number):
            call_count["n"] += 1
            choices = [
                {"title": "Press on", "description": "impulsive: yes"},
                {"title": "Wait", "description": "cautious: yes"},
            ]
            return FakeOutput(
                prose=f"chapter {update_number} prose (action: {player_action[:20]})",
                choices=choices, trace=FakeTrace(update_number),
            )

    async def fake_select_action(*, client, profile, choices, recent_prose_tail="", skeleton_chapter=None):
        return (0, "forced")  # always pick first

    with patch.object(harness_mod, "_build_pipeline", lambda *a, **kw: FakePipeline()):
        with patch.object(harness_mod, "select_action", fake_select_action):
            await run_rollout(
                quests_dir=quests, quest_id="q1", rollout_id=rid,
                client=SimpleNamespace(), score=False,
            )

    # Verify: 2 chapters committed
    conn = open_db(quests / "q1" / "quest.db")
    try:
        sm = WorldStateManager(conn)
        run = sm.get_rollout(rid)
        assert run.status == RolloutStatus.COMPLETE
        assert run.chapters_complete == 2
        chs = sm.list_rollout_chapters(rid)
        assert len(chs) == 2
        assert chs[0].chapter_index == 1
        assert chs[1].chapter_index == 2
        assert "chapter 1 prose" in chs[0].prose
    finally:
        conn.close()

    # Resume test: re-run with 4-chapter target; first 2 should be skipped
    conn = open_db(quests / "q1" / "quest.db")
    try:
        sm = WorldStateManager(conn)
        sm.update_rollout(rid, status=RolloutStatus.PENDING)
        # Bump target to 4
        conn.execute("UPDATE rollout_runs SET total_chapters_target=4 WHERE id=?", (rid,))
        conn.commit()
    finally:
        conn.close()

    before = call_count["n"]
    with patch.object(harness_mod, "_build_pipeline", lambda *a, **kw: FakePipeline()):
        with patch.object(harness_mod, "select_action", fake_select_action):
            await run_rollout(
                quests_dir=quests, quest_id="q1", rollout_id=rid,
                client=SimpleNamespace(), score=False,
            )
    # Only 2 new pipeline.run calls (chapters 3 and 4)
    assert call_count["n"] - before == 2

    conn = open_db(quests / "q1" / "quest.db")
    try:
        sm = WorldStateManager(conn)
        chs = sm.list_rollout_chapters(rid)
        assert len(chs) == 4
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_run_rollout_failure_marks_status(tmp_path: Path):
    quests = tmp_path / "quests"
    _init_main_quest(quests)
    rid = create_rollout_row(
        quests_dir=quests, quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=2,
    )

    class BrokenPipeline:
        def __init__(self, *a, **kw): pass
        async def run(self, **kw):
            raise RuntimeError("server exploded")

    with patch.object(harness_mod, "_build_pipeline", lambda *a, **kw: BrokenPipeline()):
        with pytest.raises(RuntimeError):
            await run_rollout(
                quests_dir=quests, quest_id="q1", rollout_id=rid,
                client=SimpleNamespace(), score=False,
            )

    conn = open_db(quests / "q1" / "quest.db")
    try:
        sm = WorldStateManager(conn)
        run = sm.get_rollout(rid)
        assert run.status == RolloutStatus.FAILED
        assert "server exploded" in (run.error_message or "")
    finally:
        conn.close()
