from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.rollout.scorer import score_chapter_independent, COLLAPSED_DIMS


@pytest.mark.asyncio
async def test_score_chapter_independent_calls_per_dim():
    """Verify that independent scoring makes one call per dim."""
    mock_client = MagicMock()
    call_count = 0

    async def mock_chat_with_logprobs(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_logprob = MagicMock()
        mock_logprob.token = "7"
        mock_logprob.logprob = -0.5
        mock_logprob.top_logprobs = {str(i): -2.0 for i in range(1, 11)}
        mock_logprob.top_logprobs["7"] = -0.5
        result = MagicMock()
        result.content = "Analysis here.\nprose_execution score: 7"
        result.token_logprobs = [mock_logprob]
        return result

    mock_client.chat_with_logprobs = AsyncMock(side_effect=mock_chat_with_logprobs)

    scores = await score_chapter_independent(
        client=mock_client,
        chapter_text="Some prose here about Tristan walking.",
    )

    assert call_count == len(COLLAPSED_DIMS)
    assert set(scores.keys()) == set(COLLAPSED_DIMS)
    for dim, data in scores.items():
        assert "score" in data
