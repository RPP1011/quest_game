"""Phase 4 harness wiring — verify scorer + KB extractor run per chapter."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.rollout import harness as harness_mod
from app.rollout.harness import create_rollout_row, run_rollout
from app.runtime.client import ChatWithLogprobs, TokenLogprob
from app.world.db import open_db
from app.world.schema import (
    ArcPosition, Entity, EntityType, ForeshadowingHook, PlotThread,
    QuestArcState, ReaderState, RolloutStatus, StoryCandidate, ThreadStatus,
)
from app.world.state_manager import WorldStateManager


# Canned scorer response: "prose_execution score: 7\nsubtext score: 7\nhook_quality score: 7"
_SCORE_CONTENT = (
    "prose_execution observation: ok\nprose_execution score: 7\n"
    "subtext observation: ok\nsubtext score: 7\n"
    "hook_quality observation: ok\nhook_quality score: 7"
)


class FakeClient:
    """Used as the harness client. Scorer now uses chat_with_logprobs."""

    def __init__(self) -> None:
        self.calls: list = []

    async def chat_with_logprobs(self, messages, *, temperature=0.3,
                                  max_tokens=400, top_logprobs=20,
                                  thinking=False, **kw):
        self.calls.append("logprob_call")
        tokens = []
        for char in _SCORE_CONTENT:
            lp = math.log(0.9) if char == "7" else -0.1
            top = {char: lp}
            if char == "7":
                top = {str(d): math.log(0.01) for d in range(1, 11)}
                top["7"] = math.log(0.9)
            tokens.append(TokenLogprob(char, lp, top))
        return ChatWithLogprobs(content=_SCORE_CONTENT, token_logprobs=tokens)

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw):
        self.calls.append(schema_name)
        return "{}"


class FakeTrace:
    def __init__(self, n: int):
        self.trace_id = f"trace_{n}"
        self.outcome = "committed"
        self.total_latency_ms = 0
        self.trigger = ""
        self.timestamp = None
        # Embed an extract stage with hook + entity events so KB extractor
        # has something real to find
        self._payload = {
            "trace_id": self.trace_id,
            "trigger": "",
            "timestamp": "2026-04-16T00:00:00Z",
            "outcome": "committed",
            "total_latency_ms": 0,
            "stages": [
                {"stage_name": "dramatic", "parsed_output": {
                    "thread_advances": [
                        {"thread_id": "pt:main", "target_arc_position": "rising"},
                    ],
                }},
                {"stage_name": "extract", "parsed_output": {
                    "foreshadowing_updates": [
                        {"id": "fs:1", "new_status": "planted"},
                    ],
                    "entity_updates": [
                        {"id": "char:cozme", "patch": {"status": "active"}},
                    ],
                }},
            ],
        }

    def model_dump_json(self) -> str:
        return json.dumps(self._payload)


class FakeOutput:
    def __init__(self, prose: str, choices: list, trace: FakeTrace):
        self.prose = prose
        self.choices = choices
        self.trace = trace


class FakePipeline:
    def __init__(self, *a, **kw): pass
    async def run(self, *, player_action, update_number):
        return FakeOutput(
            prose=f"Cozme stood in chapter {update_number}.",
            choices=[{"title": "act", "description": ""}],
            trace=FakeTrace(update_number),
        )


def _init_main_quest(quests_dir: Path, qid: str = "q1") -> None:
    paths = quests_dir / qid
    paths.mkdir(parents=True, exist_ok=True)
    (paths / "traces").mkdir(exist_ok=True)
    conn = open_db(paths / "quest.db")
    sm = WorldStateManager(conn)
    sm.create_entity(Entity(
        id="char:cozme", entity_type=EntityType.CHARACTER, name="Cozme",
    ))
    sm.add_plot_thread(PlotThread(
        id="pt:main", name="Main", description="x",
        arc_position=ArcPosition.RISING, status=ThreadStatus.ACTIVE, priority=9,
    ))
    sm.add_foreshadowing(ForeshadowingHook(
        id="fs:1", description="hook", planted_at_update=0, payoff_target="t",
    ))
    sm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id=qid, title="T", synopsis="S",
        primary_thread_ids=["pt:main"], secondary_thread_ids=[],
        protagonist_character_id="char:cozme", emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=2,
    ))
    sm.upsert_arc(QuestArcState(
        quest_id=qid, arc_id="main", structure_id="three_act",
        scale="campaign", current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=qid))
    conn.close()
    (paths / "config.json").write_text("{}")


@pytest.mark.asyncio
async def test_harness_runs_scorer_and_kb_extractor(tmp_path: Path):
    quests = tmp_path / "quests"
    _init_main_quest(quests)
    rid = create_rollout_row(
        quests_dir=quests, quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=2,
    )
    client = FakeClient()

    async def fake_select(*, client, profile, choices, recent_prose_tail=""):
        return (0, "test")

    with patch.object(harness_mod, "_build_pipeline", lambda *a, **kw: FakePipeline()):
        with patch.object(harness_mod, "select_action", fake_select):
            await run_rollout(
                quests_dir=quests, quest_id="q1", rollout_id=rid,
                client=client, score=True,
            )

    conn = open_db(quests / "q1" / "quest.db")
    try:
        sm = WorldStateManager(conn)
        run = sm.get_rollout(rid)
        assert run.status == RolloutStatus.COMPLETE

        # Each chapter should have a judge_scores blob
        chs = sm.list_rollout_chapters(rid)
        assert len(chs) == 2
        for ch in chs:
            assert ch.judge_scores is not None
            assert "prose_execution" in ch.judge_scores

        # kb_chapter_scores: 3 dims × 2 chapters = 6 rows
        all_scores = sm.list_chapter_scores(rid)
        assert len(all_scores) == 6

        # KB extractor: hook fs:1 planted in both chapters; entity char:cozme
        # introduced in both chapters and mentioned (via word match)
        hooks = sm.list_hook_payoffs("q1")
        # One row per (rollout, hook) — the second insert upserts on the
        # primary key, so the latest planted_at_chapter wins
        assert len(hooks) == 1
        assert hooks[0]["hook_id"] == "fs:1"
        assert hooks[0]["planted_at_chapter"] == 2  # last write wins

        eu = sm.list_entity_usage("q1")
        assert len(eu) == 1
        assert eu[0]["entity_id"] == "char:cozme"
        assert sorted(eu[0]["mention_chapters"]) == [1, 2]

        # Scorer was called twice (once per chapter)
        assert client.calls.count("logprob_call") == 2
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_harness_skips_scorer_when_disabled(tmp_path: Path):
    quests = tmp_path / "quests"
    _init_main_quest(quests)
    rid = create_rollout_row(
        quests_dir=quests, quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=1,
    )
    client = FakeClient()

    async def fake_select(*, client, profile, choices, recent_prose_tail=""):
        return (0, "test")

    with patch.object(harness_mod, "_build_pipeline", lambda *a, **kw: FakePipeline()):
        with patch.object(harness_mod, "select_action", fake_select):
            await run_rollout(
                quests_dir=quests, quest_id="q1", rollout_id=rid,
                client=client, score=False,
            )

    # No scorer calls; KB extraction still happens
    assert client.calls == []

    conn = open_db(quests / "q1" / "quest.db")
    try:
        sm = WorldStateManager(conn)
        chs = sm.list_rollout_chapters(rid)
        # judge_scores stays None when score=False
        assert chs[0].judge_scores is None
        # No score rows in kb_chapter_scores
        assert sm.list_chapter_scores(rid) == []
        # But KB extractor still ran — hooks + entities should exist
        assert len(sm.list_hook_payoffs("q1")) == 1
        assert len(sm.list_entity_usage("q1")) == 1
    finally:
        conn.close()
