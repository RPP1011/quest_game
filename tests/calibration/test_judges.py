import json
from pathlib import Path

import pytest

from app.calibration.judges import (
    BatchJudge,
    COMMON_LLM_DIMS,
    QUEST_LLM_DIMS,
    dims_for,
    parse_response,
)


PROMPTS = Path("prompts")


class _StubClient:
    """Captures the last prompt and returns a canned JSON string."""

    def __init__(self, canned: dict) -> None:
        self.canned = canned
        self.last_prompt: str | None = None

    async def chat_structured(self, *, messages, json_schema, schema_name,
                              temperature, max_tokens, thinking) -> str:
        self.last_prompt = messages[0].content
        return json.dumps(self.canned)


def test_dims_for_novel_vs_quest():
    assert set(dims_for(False)) == set(COMMON_LLM_DIMS)
    assert set(dims_for(True)) == set(COMMON_LLM_DIMS) | set(QUEST_LLM_DIMS)


def test_render_prompt_includes_dimensions_and_passage():
    judge = BatchJudge(PROMPTS)
    out = judge.render_prompt(
        passage="Here is the passage body.",
        work_id="demo",
        pov="first",
        is_quest=False,
    )
    for d in COMMON_LLM_DIMS:
        assert d in out
    assert "Here is the passage body." in out
    # Quest-only dim absent for non-quest.
    assert "choice_hook_quality" not in out


def test_render_prompt_quest_adds_quest_dims():
    judge = BatchJudge(PROMPTS)
    out = judge.render_prompt(
        passage="Body.",
        work_id="demo",
        pov="second",
        is_quest=True,
    )
    for d in QUEST_LLM_DIMS:
        assert d in out


def test_parse_response_happy_path():
    names = ["clarity", "tension_execution"]
    raw = '```\n' + json.dumps({
        "clarity": {"score": 0.8, "rationale": "clear"},
        "tension_execution": {"score": 0.5, "rationale": "mid"},
    }) + "\n```"
    parsed = parse_response(raw, names)
    assert parsed["clarity"].score == 0.8
    assert parsed["tension_execution"].rationale == "mid"


def test_parse_response_clips_out_of_range():
    parsed = parse_response(
        '{"clarity": {"score": 1.8, "rationale": "x"}}',
        ["clarity"],
    )
    assert parsed["clarity"].score == 1.0


def test_parse_response_missing_dim_raises():
    with pytest.raises(ValueError):
        parse_response('{"clarity": {"score": 0.5, "rationale": "x"}}',
                       ["clarity", "tension_execution"])


async def test_batch_judge_end_to_end_with_stub():
    judge = BatchJudge(PROMPTS)
    canned = {d: {"score": 0.5, "rationale": "stub"} for d in COMMON_LLM_DIMS}
    client = _StubClient(canned)
    scored = await judge.score(
        client=client,
        passage="Sample passage.",
        work_id="demo",
        pov="first",
        is_quest=False,
    )
    assert set(scored) == set(COMMON_LLM_DIMS)
    assert client.last_prompt is not None
    assert "Sample passage." in client.last_prompt
