"""Day 6 tests for :meth:`app.scoring.Scorer.score_with_llm_judges`.

These tests use a stub client (no real LLM call) to verify that:

- Building a ``Scorer`` without ``llm_judge_client`` preserves Day 2
  semantics: ``has_llm_judge`` is False and calling
  :meth:`score_with_llm_judges` raises a clear ``RuntimeError``.
- When the client is wired, the three Day 6 dims
  (``tension_execution``, ``emotional_trajectory``, ``choice_hook_quality``)
  land on an :class:`ExtendedScorecard` with the Day 2 base intact.
- Rationales flow through unchanged.
- The judge receives a prompt that names exactly the three Day 6 dims
  (no extra, no missing) — i.e. the scorer doesn't accidentally request
  the full COMMON_LLM_DIMS set.
"""
from __future__ import annotations

import json

import pytest

from app.scoring import ExtendedScorecard, LLM_JUDGE_DIMS, Scorer


SAMPLE = (
    "You stepped into the cold hall. The flagstones caught the lantern "
    "light and threw it back up at you, pale and uneven. Somewhere above, "
    "a door closed. \"Wait,\" a voice said. You stopped. The air smelled of "
    "iron and damp wool. Your heart hammered against your ribs. You turned "
    "toward the sound, one hand already reaching for the hilt at your hip."
)


class StubJudgeClient:
    """Canned-JSON client. Captures the last prompt for assertion."""

    def __init__(self, dim_scores: dict[str, float]) -> None:
        self.dim_scores = dim_scores
        self.last_prompt: str | None = None
        self.last_schema: dict | None = None

    async def chat_structured(
        self, *, messages, json_schema, schema_name,
        temperature, max_tokens, thinking,
    ) -> str:
        self.last_prompt = messages[0].content
        self.last_schema = json_schema
        return json.dumps({
            d: {"score": v, "rationale": f"stub-{d}"}
            for d, v in self.dim_scores.items()
        })


def test_scorer_without_client_preserves_day2_semantics():
    scorer = Scorer()
    assert scorer.has_llm_judge is False
    # Day 2 sync path still works.
    card = scorer.score(SAMPLE)
    assert 0.0 <= card.overall_score <= 1.0


async def test_score_with_llm_judges_requires_client():
    scorer = Scorer()
    with pytest.raises(RuntimeError, match="without llm_judge_client"):
        await scorer.score_with_llm_judges(SAMPLE)


async def test_score_with_llm_judges_adds_three_dims():
    canned = {
        "tension_execution": 0.62,
        "emotional_trajectory": 0.48,
        "choice_hook_quality": 0.71,
    }
    client = StubJudgeClient(canned)
    scorer = Scorer(llm_judge_client=client)
    assert scorer.has_llm_judge is True

    ext = await scorer.score_with_llm_judges(SAMPLE)
    assert isinstance(ext, ExtendedScorecard)
    # Day 2 base intact.
    assert 0.0 <= ext.base.overall_score <= 1.0
    # Day 6 dims landed with the stub values.
    assert set(ext.llm_judge_scores.keys()) == set(LLM_JUDGE_DIMS)
    for d, expected in canned.items():
        assert ext.llm_judge_scores[d] == pytest.approx(expected)
    # Rationales flow through.
    for d in LLM_JUDGE_DIMS:
        assert ext.llm_judge_rationales[d] == f"stub-{d}"


async def test_score_with_llm_judges_prompts_only_for_day6_dims():
    """Scorer must restrict the batched call to the 3 Day 6 dims — not
    request the full COMMON_LLM_DIMS (which would waste tokens)."""
    canned = {d: 0.5 for d in LLM_JUDGE_DIMS}
    client = StubJudgeClient(canned)
    scorer = Scorer(llm_judge_client=client)
    await scorer.score_with_llm_judges(SAMPLE)

    assert client.last_prompt is not None
    # All 3 named dims must be in the rendered prompt.
    for d in LLM_JUDGE_DIMS:
        assert d in client.last_prompt
    # Schema requires exactly the 3 dims — no clarity, no thematic_presence.
    assert client.last_schema is not None
    assert set(client.last_schema["properties"].keys()) == set(LLM_JUDGE_DIMS)
    assert "clarity" not in client.last_schema["properties"]
    assert "thematic_presence" not in client.last_schema["properties"]


async def test_score_with_llm_judges_all_dimension_items_merges():
    """``all_dimension_items()`` surfaces both families for persistence."""
    canned = {d: 0.4 for d in LLM_JUDGE_DIMS}
    client = StubJudgeClient(canned)
    scorer = Scorer(llm_judge_client=client)
    ext = await scorer.score_with_llm_judges(SAMPLE)

    all_items = dict(ext.all_dimension_items())
    # Every Day 2 dim AND every Day 6 dim is present.
    from app.scoring import DIMENSION_NAMES
    for d in DIMENSION_NAMES:
        assert d in all_items
    for d in LLM_JUDGE_DIMS:
        assert d in all_items
        assert all_items[d] == pytest.approx(0.4)


async def test_score_with_llm_judges_clips_out_of_range():
    """Judge that returns 1.5 or -0.2 gets clamped to [0, 1]."""
    canned = {
        "tension_execution": 1.5,
        "emotional_trajectory": -0.3,
        "choice_hook_quality": 0.5,
    }
    client = StubJudgeClient(canned)
    scorer = Scorer(llm_judge_client=client)
    ext = await scorer.score_with_llm_judges(SAMPLE)
    assert ext.llm_judge_scores["tension_execution"] == 1.0
    assert ext.llm_judge_scores["emotional_trajectory"] == 0.0
    assert ext.llm_judge_scores["choice_hook_quality"] == 0.5
