from __future__ import annotations
import json
import math

import pytest

from app.rollout.scorer import (
    COLLAPSED_DIMS, LEGACY_DIMS, score_chapter, score_and_persist_chapter,
    score_chapter_logprob, compare_chapters, _parse_logprob_scores,
)
from app.runtime.client import ChatWithLogprobs, TokenLogprob
from app.world.db import open_db
from app.world.schema import RolloutChapter, RolloutRun, StoryCandidate
from app.world.state_manager import WorldStateManager


# Legacy structured-output canned response (8 dims)
CANNED_LEGACY = json.dumps({
    "tension_execution": {"score": 0.8, "rationale": "ok"},
    "emotional_trajectory": {"score": 0.7, "rationale": "ok"},
    "choice_hook_quality": {"score": 0.6, "rationale": "ok"},
    "update_self_containment": {"score": 0.7, "rationale": "ok"},
    "voice_distinctiveness": {"score": 0.85, "rationale": "ok"},
    "thematic_presence": {"score": 0.7, "rationale": "ok"},
    "subtext_presence": {"score": 0.6, "rationale": "ok"},
    "interiority_depth": {"score": 0.7, "rationale": "ok"},
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


# ---------------------------------------------------------------------------
# Legacy scorer tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_legacy_score_chapter_returns_all_dims():
    client = FakeClient(CANNED_LEGACY)
    scores = await score_chapter(
        client=client, chapter_text="some prose", dims=list(LEGACY_DIMS),
    )
    assert set(scores.keys()) == set(LEGACY_DIMS)
    assert scores["tension_execution"]["score"] == 0.8


@pytest.mark.asyncio
async def test_legacy_score_chapter_drops_missing_dims():
    client = FakeClient(json.dumps({
        "tension_execution": {"score": 0.5, "rationale": "ok"},
    }))
    scores = await score_chapter(client=client, chapter_text="x", dims=list(LEGACY_DIMS))
    assert "tension_execution" in scores
    assert "voice_distinctiveness" not in scores


@pytest.mark.asyncio
async def test_score_and_persist_writes_to_both_places(sm):
    client = FakeClient(CANNED_LEGACY)
    chapter = RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="x", prose="prose",
    )
    sm.save_rollout_chapter(chapter)
    scores = await score_and_persist_chapter(
        client=client, world=sm, rollout_id="r1", chapter=chapter,
        dims=list(LEGACY_DIMS),
    )
    assert "tension_execution" in scores
    kb_rows = sm.list_chapter_scores("r1", chapter_index=1)
    assert len(kb_rows) == len(LEGACY_DIMS)


@pytest.mark.asyncio
async def test_score_and_persist_idempotent(sm):
    client = FakeClient(CANNED_LEGACY)
    chapter = RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="x", prose="prose",
        judge_scores={"tension_execution": 0.99},
    )
    sm.save_rollout_chapter(chapter)
    scores = await score_and_persist_chapter(
        client=client, world=sm, rollout_id="r1", chapter=chapter,
    )
    assert scores["tension_execution"] == 0.99
    assert client.calls == []


# ---------------------------------------------------------------------------
# Logprob scorer tests
# ---------------------------------------------------------------------------

def test_parse_logprob_scores_extracts_dims():
    """Simulate a model response with logprobs for integer scores."""
    tokens = [
        TokenLogprob("prose", -0.1, {}),
        TokenLogprob("_", -0.1, {}),
        TokenLogprob("execution", -0.1, {}),
        TokenLogprob(":", -0.1, {}),
        TokenLogprob(" ", -0.1, {}),
        TokenLogprob("7", math.log(0.85), {
            "6": math.log(0.05), "7": math.log(0.85), "8": math.log(0.10),
        }),
        TokenLogprob("\n", -0.1, {}),
        TokenLogprob("sub", -0.1, {}),
        TokenLogprob("text", -0.1, {}),
        TokenLogprob(":", -0.1, {}),
        TokenLogprob(" ", -0.1, {}),
        TokenLogprob("5", math.log(0.70), {
            "4": math.log(0.15), "5": math.log(0.70), "6": math.log(0.15),
        }),
    ]
    result = ChatWithLogprobs(
        content="prose_execution: 7\nsubtext: 5",
        token_logprobs=tokens,
    )
    scores = _parse_logprob_scores(result, ["prose_execution", "subtext"])
    assert "prose_execution" in scores
    assert scores["prose_execution"]["sampled"] == 7
    assert 0.6 < scores["prose_execution"]["score"] < 0.75
    assert scores["prose_execution"]["confidence"] > 0.5
    assert "subtext" in scores
    assert scores["subtext"]["sampled"] == 5


def test_parse_logprob_scores_fallback_text():
    """If no logprob data, fall back to parsing content text."""
    result = ChatWithLogprobs(
        content="prose_execution: 8\nhook_quality: 6",
        token_logprobs=[],
    )
    scores = _parse_logprob_scores(result, ["prose_execution", "hook_quality"])
    assert scores["prose_execution"]["sampled"] == 8
    assert scores["prose_execution"]["confidence"] == 0.0  # no logprob data
    assert scores["hook_quality"]["sampled"] == 6


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------

def test_collapsed_dims():
    """The default dims are the 3 collapsed dims."""
    assert len(COLLAPSED_DIMS) == 3
    assert "prose_execution" in COLLAPSED_DIMS
    assert "subtext" in COLLAPSED_DIMS
    assert "hook_quality" in COLLAPSED_DIMS
    assert "update_self_containment" not in COLLAPSED_DIMS
