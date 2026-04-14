"""LLM-judged dimensions — batched single-call scoring.

One structured call per passage returns scores for every LLM-judged dimension
at once. The chat client is any object with an async ``chat_structured`` (or
``chat``) method — we feed a stub in tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from jinja2 import Environment, FileSystemLoader, StrictUndefined


# Dimensions present for every passage (novel + quest).
COMMON_LLM_DIMS: tuple[str, ...] = (
    "free_indirect_quality",
    "interiority_depth",
    "detail_characterization",
    "sensory_density",
    "voice_distinctiveness",
    "tension_execution",
    "thematic_presence",
    "subtext_presence",
    "clarity",
)

# Quest-only additions.
QUEST_LLM_DIMS: tuple[str, ...] = (
    "choice_hook_quality",
    "update_self_containment",
    "choice_meaningfulness",
    "world_state_legibility",
)


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


@dataclass(frozen=True)
class JudgeScore:
    score: float
    rationale: str


def dims_for(is_quest: bool) -> list[str]:
    return list(COMMON_LLM_DIMS) + (list(QUEST_LLM_DIMS) if is_quest else [])


def _response_schema(dim_names: list[str]) -> dict[str, Any]:
    props = {
        d: {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"},
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        }
        for d in dim_names
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(dim_names),
        "additionalProperties": False,
    }


class BatchJudge:
    """Render the batch prompt, call the model, parse JSON -> scores."""

    def __init__(self, prompts_dir: str | Path) -> None:
        self._prompts_dir = Path(prompts_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self._prompts_dir)),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _load_rubric(self, dim: str) -> str:
        tmpl = self._env.get_template(f"scoring/dims/{dim}.j2")
        return tmpl.render().strip()

    def render_prompt(
        self,
        *,
        passage: str,
        work_id: str,
        pov: str,
        is_quest: bool,
        dim_names: list[str] | None = None,
    ) -> str:
        dim_names = dim_names or dims_for(is_quest)
        dims = [{"name": d, "rubric": self._load_rubric(d)} for d in dim_names]
        tmpl = self._env.get_template("scoring/batch.j2")
        return tmpl.render(
            passage=passage,
            work_id=work_id,
            pov=pov,
            is_quest=is_quest,
            dims=dims,
        )

    async def score(
        self,
        *,
        client: ChatLike,
        passage: str,
        work_id: str,
        pov: str,
        is_quest: bool,
        dim_names: list[str] | None = None,
    ) -> dict[str, JudgeScore]:
        dim_names = dim_names or dims_for(is_quest)
        prompt = self.render_prompt(
            passage=passage,
            work_id=work_id,
            pov=pov,
            is_quest=is_quest,
            dim_names=dim_names,
        )
        raw = await _call_model(client, prompt, _response_schema(dim_names))
        return parse_response(raw, dim_names)


async def _call_model(client: ChatLike, prompt: str, schema: dict[str, Any]) -> str:
    """Call ``chat_structured`` when available, else fall back to ``chat``."""
    # Lazy import to avoid hard dependency in tests.
    try:
        from app.runtime.client import ChatMessage  # type: ignore
        msg = ChatMessage(role="user", content=prompt)
    except Exception:  # pragma: no cover - only if runtime not importable
        msg = {"role": "user", "content": prompt}

    chat_structured = getattr(client, "chat_structured", None)
    if chat_structured is not None:
        return await chat_structured(
            messages=[msg],
            json_schema=schema,
            schema_name="PassageScores",
            temperature=0.2,
            max_tokens=2000,
            thinking=False,
        )
    return await client.chat(
        messages=[msg],
        temperature=0.2,
        max_tokens=2000,
        thinking=False,
    )


def parse_response(raw: str, dim_names: list[str]) -> dict[str, JudgeScore]:
    """Tolerant parse: find the first {...} block and parse it."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in judge response: {raw[:200]!r}")
    data = json.loads(raw[start : end + 1])
    out: dict[str, JudgeScore] = {}
    for d in dim_names:
        if d not in data:
            raise ValueError(f"judge response missing dimension {d!r}")
        entry = data[d]
        score = float(entry["score"])
        score = max(0.0, min(1.0, score))
        out[d] = JudgeScore(score=score, rationale=str(entry.get("rationale", "")))
    return out
