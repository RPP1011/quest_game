from __future__ import annotations
import json

import pytest

from app.rollout.scorer import (
    DEFAULT_DIMS, score_chapter, score_and_persist_chapter,
)
from app.world.db import open_db
from app.world.schema import RolloutChapter, RolloutRun, StoryCandidate
from app.world.state_manager import WorldStateManager


CANNED = json.dumps({
    "tension_execution": {"score": 0.8, "rationale": "rising stakes well-managed"},
    "emotional_trajectory": {"score": 0.7, "rationale": "decent arc"},
    "choice_hook_quality": {"score": 0.6, "rationale": "ok"},
    "update_self_containment": {"score": 0.7, "rationale": "stands on its own"},
    "voice_distinctiveness": {"score": 0.85, "rationale": "strong narrator voice"},
    "thematic_presence": {"score": 0.7, "rationale": "themes land"},
    "subtext_presence": {"score": 0.6, "rationale": "moderate"},
    "interiority_depth": {"score": 0.7, "rationale": "good interior life"},
})


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list = []

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw):
        self.calls.append((messages, json_schema, schema_name, kw))
        return self._response


@pytest.fixture
def sm(tmp_path):
    conn = open_db(tmp_path / "w.db")
    wsm = WorldStateManager(conn)
    wsm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="T", synopsis="S",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=2,
    ))
    wsm.create_rollout(RolloutRun(
        id="r1", quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=1,
    ))
    yield wsm
    conn.close()


@pytest.mark.asyncio
async def test_score_chapter_returns_all_dims():
    client = FakeClient(CANNED)
    scores = await score_chapter(
        client=client, chapter_text="some prose",
    )
    assert set(scores.keys()) == set(DEFAULT_DIMS)
    assert scores["tension_execution"]["score"] == 0.8
    assert scores["voice_distinctiveness"]["score"] == 0.85


@pytest.mark.asyncio
async def test_score_chapter_schema_has_all_dims():
    client = FakeClient(CANNED)
    await score_chapter(client=client, chapter_text="prose")
    _, schema, _, _ = client.calls[0]
    for d in DEFAULT_DIMS:
        assert d in schema["properties"]


@pytest.mark.asyncio
async def test_score_chapter_drops_missing_dims():
    """If the model omits a dim, scorer should not crash."""
    client = FakeClient(json.dumps({
        "tension_execution": {"score": 0.5, "rationale": "ok"},
        # other dims missing
    }))
    scores = await score_chapter(client=client, chapter_text="prose")
    assert "tension_execution" in scores
    assert "voice_distinctiveness" not in scores


@pytest.mark.asyncio
async def test_score_and_persist_writes_to_both_places(sm):
    client = FakeClient(CANNED)
    chapter = RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="x", prose="prose",
    )
    sm.save_rollout_chapter(chapter)
    scores = await score_and_persist_chapter(
        client=client, world=sm, rollout_id="r1", chapter=chapter,
    )
    assert "tension_execution" in scores

    # kb_chapter_scores rows
    kb_rows = sm.list_chapter_scores("r1", chapter_index=1)
    by_dim = {r["dim"]: r for r in kb_rows}
    assert by_dim["tension_execution"]["score"] == 0.8
    assert by_dim["voice_distinctiveness"]["rationale"] == "strong narrator voice"
    assert len(kb_rows) == len(DEFAULT_DIMS)

    # Chapter row's judge_scores blob
    chs = sm.list_rollout_chapters("r1")
    assert chs[0].judge_scores is not None
    assert chs[0].judge_scores["tension_execution"] == 0.8


@pytest.mark.asyncio
async def test_score_and_persist_idempotent(sm):
    """If chapter already has judge_scores, no LLM call is made."""
    client = FakeClient(CANNED)
    chapter = RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="x", prose="prose",
        judge_scores={"tension_execution": 0.99},  # pre-existing
    )
    sm.save_rollout_chapter(chapter)
    scores = await score_and_persist_chapter(
        client=client, world=sm, rollout_id="r1", chapter=chapter,
    )
    assert scores["tension_execution"] == 0.99
    assert client.calls == []  # no LLM call
