from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol
from app.runtime.client import ChatMessage
from app.world.output_parser import OutputParser, ParseError
from app.world.schema import NarrativeRecord
from app.world.state_manager import WorldStateManager
from .context_builder import ContextBuilder
from .context_spec import PLAN_SPEC, WRITE_SPEC
from .inference_params import TokenUsage
from .stages import StageError, StageResult
from .trace import PipelineTrace


_BEAT_KEYS = ("beats", "beat_sheet", "beatsheet", "plan", "outline", "steps", "scenes")
_CHOICE_KEYS = ("suggested_choices", "suggestedchoices", "choices", "options", "actions")


def _normalize_beat_sheet(data: dict) -> dict:
    """Accept common key aliases produced by weaker/non-strict models.

    Handles case variants (beatSheet vs beat_sheet), list-of-dicts (extracts a
    string field), and falls back to scanning any list-of-strings value.
    """
    lowered = {k.lower().replace("_", ""): v for k, v in data.items()}
    beats = _coerce_string_list(_pick(lowered, _BEAT_KEYS))
    if not beats:
        # Last resort: first list-of-something field in the dict.
        for v in data.values():
            beats = _coerce_string_list(v)
            if beats:
                break
    choices = _coerce_string_list(_pick(lowered, _CHOICE_KEYS))
    return {"beats": beats, "suggested_choices": choices}


def _pick(lowered: dict, keys: tuple[str, ...]):
    for k in keys:
        norm = k.lower().replace("_", "")
        if norm in lowered:
            return lowered[norm]
    return None


def _coerce_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            # Prefer conventional fields, else fall back to concatenating values.
            for key in ("beat", "text", "description", "summary", "name", "content"):
                if isinstance(item.get(key), str):
                    out.append(item[key])
                    break
            else:
                parts = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
                if parts:
                    out.append(" — ".join(parts))
    return out


BEAT_SHEET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "beats": {"type": "array", "items": {"type": "string"}},
        "suggested_choices": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["beats", "suggested_choices"],
    "additionalProperties": False,
}


class InferenceClientLike(Protocol):
    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str: ...
    async def chat(self, *, messages, **kw) -> str: ...


@dataclass
class PipelineOutput:
    prose: str
    choices: list[str]
    beats: list[str]
    trace: PipelineTrace


class Pipeline:
    def __init__(
        self,
        world: WorldStateManager,
        context_builder: ContextBuilder,
        client: InferenceClientLike,
    ) -> None:
        self._world = world
        self._cb = context_builder
        self._client = client

    async def run(self, *, player_action: str, update_number: int) -> PipelineOutput:
        trace = PipelineTrace(trace_id=uuid.uuid4().hex, trigger=player_action)

        # ---- PLAN ----
        plan_ctx = self._cb.build(
            spec=PLAN_SPEC,
            stage_name="plan",
            templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
            extras={"player_action": player_action},
        )
        t0 = time.perf_counter()
        plan_raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=plan_ctx.system_prompt),
                ChatMessage(role="user", content=plan_ctx.user_prompt),
            ],
            json_schema=BEAT_SHEET_SCHEMA,
            schema_name="BeatSheet",
            temperature=0.4,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        errors: list[StageError] = []
        try:
            plan_parsed = OutputParser.parse_json(plan_raw)
            if not isinstance(plan_parsed, dict):
                raise ParseError(f"beat sheet malformed: {plan_parsed!r}")
            plan_parsed = _normalize_beat_sheet(plan_parsed)
            if not plan_parsed["beats"]:
                raise ParseError(f"beat sheet has no beats: {plan_raw!r}")
        except ParseError as e:
            errors.append(StageError(kind="parse_error", message=str(e)))
            trace.add_stage(StageResult(
                stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=plan_raw,
                errors=errors, latency_ms=latency,
            ))
            trace.outcome = "failed"
            raise
        trace.add_stage(StageResult(
            stage_name="plan",
            input_prompt=plan_ctx.user_prompt,
            raw_output=plan_raw,
            parsed_output=plan_parsed,
            token_usage=TokenUsage(prompt=plan_ctx.token_estimate),
            latency_ms=latency,
        ))

        # ---- WRITE ----
        plan_text = "\n".join(f"- {b}" for b in plan_parsed["beats"])
        write_ctx = self._cb.build(
            spec=WRITE_SPEC,
            stage_name="write",
            templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
            extras={
                "plan": plan_text,
                "style": "",
                "anti_patterns": [],
            },
        )
        t0 = time.perf_counter()
        prose_raw = await self._client.chat(
            messages=[
                ChatMessage(role="system", content=write_ctx.system_prompt),
                ChatMessage(role="user", content=write_ctx.user_prompt),
            ],
            temperature=0.8,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        prose = OutputParser.parse_prose(prose_raw)
        trace.add_stage(StageResult(
            stage_name="write",
            input_prompt=write_ctx.user_prompt,
            raw_output=prose_raw,
            parsed_output=prose,
            token_usage=TokenUsage(prompt=write_ctx.token_estimate),
            latency_ms=latency,
        ))

        # ---- COMMIT ----
        self._world.write_narrative(NarrativeRecord(
            update_number=update_number,
            raw_text=prose,
            player_action=player_action,
            pipeline_trace_id=trace.trace_id,
        ))
        trace.outcome = "committed"

        return PipelineOutput(
            prose=prose,
            choices=plan_parsed.get("suggested_choices", []),
            beats=plan_parsed["beats"],
            trace=trace,
        )
