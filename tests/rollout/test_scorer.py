from __future__ import annotations
import json
import math

import pytest

from app.rollout.scorer import (
    COLLAPSED_DIMS, LEGACY_DIMS, score_chapter, score_and_persist_chapter,
    compare_chapters, _find_score_at_marker, _fallback_parse_score,
)
from app.runtime.client import ChatWithLogprobs, TokenLogprob
from app.world.db import open_db
from app.world.schema import RolloutChapter, RolloutRun, StoryCandidate
from app.world.state_manager import WorldStateManager


# Canned logprob response simulating "prose_execution score: 8\nsubtext score: 7\nhook_quality score: 7"
CANNED_LOGPROB_CONTENT = "prose_execution observation: good\nprose_execution score: 8\nsubtext observation: ok\nsubtext score: 7\nhook_quality observation: fine\nhook_quality score: 7"


class FakeLogprobClient:
    """Returns canned ChatWithLogprobs for testing."""
    def __init__(self) -> None:
        self.calls: list = []

    async def chat_with_logprobs(self, messages, *, temperature=0.3,
                                  max_tokens=400, top_logprobs=20,
                                  thinking=False, **kw):
        self.calls.append("logprob_call")
        # Build tokens with logprobs for the score digits
        tokens = []
        for char in CANNED_LOGPROB_CONTENT:
            lp = math.log(0.9) if char in "78" else -0.1
            top = {char: lp}
            if char in "78":
                top = {str(d): math.log(0.01) for d in range(1, 11)}
                top[char] = math.log(0.9)
            tokens.append(TokenLogprob(char, lp, top))
        return ChatWithLogprobs(
            content=CANNED_LOGPROB_CONTENT,
            token_logprobs=tokens,
        )


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
async def test_score_chapter_returns_collapsed_dims():
    client = FakeLogprobClient()
    scores = await score_chapter(client=client, chapter_text="some prose")
    assert "prose_execution" in scores
    assert "subtext" in scores
    assert "hook_quality" in scores
    assert scores["prose_execution"]["sampled"] == 8


@pytest.mark.asyncio
async def test_score_and_persist_writes_to_kb(sm):
    client = FakeLogprobClient()
    chapter = RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="x", prose="prose",
    )
    sm.save_rollout_chapter(chapter)
    scores = await score_and_persist_chapter(
        client=client, world=sm, rollout_id="r1", chapter=chapter,
    )
    assert "prose_execution" in scores
    kb_rows = sm.list_chapter_scores("r1", chapter_index=1)
    assert len(kb_rows) == len(COLLAPSED_DIMS)


@pytest.mark.asyncio
async def test_score_and_persist_idempotent(sm):
    client = FakeLogprobClient()
    chapter = RolloutChapter(
        rollout_id="r1", chapter_index=1,
        player_action="x", prose="prose",
        judge_scores={"prose_execution": 0.99},
    )
    sm.save_rollout_chapter(chapter)
    scores = await score_and_persist_chapter(
        client=client, world=sm, rollout_id="r1", chapter=chapter,
    )
    assert scores["prose_execution"] == 0.99
    assert client.calls == []


# ---------------------------------------------------------------------------
# Logprob scorer tests
# ---------------------------------------------------------------------------

def test_find_score_at_marker():
    """Find the score digit after a marker in the logprob stream."""
    tokens = [
        TokenLogprob("Chapter", -0.1, {}),
        TokenLogprob(" A", -0.1, {}),
        TokenLogprob(" score", -0.1, {}),
        TokenLogprob(":", -0.1, {}),
        TokenLogprob(" ", -0.1, {}),
        TokenLogprob("7", math.log(0.85), {
            "6": math.log(0.05), "7": math.log(0.85), "8": math.log(0.10),
        }),
    ]
    result = ChatWithLogprobs(
        content="Chapter A score: 7",
        token_logprobs=tokens,
    )
    found = _find_score_at_marker(result, "Chapter A score:")
    assert found is not None
    assert found["sampled"] == 7
    assert 0.6 < found["score"] < 0.75
    assert found["confidence"] > 0.5


def test_fallback_parse_score():
    """Parse score from plain text when logprobs aren't available."""
    found = _fallback_parse_score(
        "Chapter A score: 8\nChapter B score: 6",
        "Chapter A score",
    )
    assert found is not None
    assert found["sampled"] == 8
    assert found["confidence"] == 0.0


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
