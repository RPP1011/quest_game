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
from .check import CHECK_SCHEMA, CheckIssue, CheckOutput
from .context_builder import ContextBuilder
from .context_spec import CHECK_SPEC, EXTRACT_SPEC, PLAN_SPEC, REVISE_SPEC, WRITE_SPEC
from .extract import EXTRACT_SCHEMA, build_delta
from .inference_params import TokenUsage
from .stages import StageError, StageResult
from .trace import PipelineTrace


_BEAT_KEYS = ("beats", "beat_sheet", "beatsheet", "plan", "outline", "steps", "scenes")
_CHOICE_KEYS = ("suggested_choices", "suggestedchoices", "choices", "options", "actions")


def _normalize_beat_sheet(data: dict) -> dict:
    """Accept common key aliases produced by weaker/non-strict models.

    Handles case variants (beatSheet vs beat_sheet), list-of-dicts (extracts a
    string field), nested wrappers (``beat_sheet: {key_actions: [...]}``), and
    falls back to scanning any list-of-strings value.
    """
    lowered = {k.lower().replace("_", ""): v for k, v in data.items()}
    beats = _extract_list(_pick(lowered, _BEAT_KEYS), _BEAT_KEYS)
    if not beats:
        for v in data.values():
            beats = _extract_list(v, _BEAT_KEYS)
            if beats:
                break
    choices = _extract_list(_pick(lowered, _CHOICE_KEYS), _CHOICE_KEYS)
    if not choices:
        # Some models nest suggested_choices inside beat_sheet dict; scan one
        # level into any dict value.
        for v in data.values():
            if isinstance(v, dict):
                inner = {k.lower().replace("_", ""): vv for k, vv in v.items()}
                choices = _coerce_string_list(_pick(inner, _CHOICE_KEYS))
                if choices:
                    break
    return {"beats": beats, "suggested_choices": choices}


def _extract_list(value, nested_keys: tuple[str, ...]) -> list[str]:
    """Coerce value to a list of strings. If value is a dict, look inside it
    for a nested list under any of ``nested_keys``."""
    direct = _coerce_string_list(value)
    if direct:
        return direct
    if isinstance(value, dict):
        inner = {k.lower().replace("_", ""): vv for k, vv in value.items()}
        return _coerce_string_list(_pick(inner, nested_keys + ("keyactions",)))
    return []


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
        replan_attempts = 0
        recheck_done = False
        critical_feedback: list[CheckIssue] = []

        plan_parsed = await self._run_plan(trace, player_action, critical_feedback)
        prose = await self._run_write(trace, plan_parsed)
        check_out = await self._run_check(trace, plan_parsed, prose)

        # REPLAN branch: critical issues, replan once.
        if check_out.has_critical and replan_attempts < 1:
            replan_attempts += 1
            critical_feedback = list(check_out.issues)
            plan_parsed = await self._run_plan(trace, player_action, critical_feedback)
            prose = await self._run_write(trace, plan_parsed)
            check_out = await self._run_check(trace, plan_parsed, prose)

        # REVISE branch: fixable issues, revise + recheck once.
        if check_out.has_fixable and not check_out.has_critical and not recheck_done:
            prose = await self._run_revise(trace, plan_parsed, prose, check_out.issues)
            recheck_done = True
            check_out = await self._run_check(trace, plan_parsed, prose)

        # Determine outcome.
        if check_out.has_critical:
            outcome = "flagged_qm"
        else:
            outcome = "committed"

        self._world.write_narrative(NarrativeRecord(
            update_number=update_number,
            raw_text=prose,
            player_action=player_action,
            pipeline_trace_id=trace.trace_id,
        ))
        trace.outcome = outcome

        # EXTRACT stage: only when not critical.
        if not check_out.has_critical:
            await self._run_extract(trace, plan_parsed, prose, update_number)

        return PipelineOutput(
            prose=prose,
            choices=plan_parsed.get("suggested_choices", []),
            beats=plan_parsed["beats"],
            trace=trace,
        )

    async def _run_plan(
        self, trace: PipelineTrace, player_action: str,
        critical_feedback: list[CheckIssue],
    ) -> dict:
        extras: dict[str, Any] = {"player_action": player_action}
        if critical_feedback:
            extras["critical_feedback"] = "\n".join(
                f"- [{i.severity}/{i.category}] {i.message}" for i in critical_feedback
            )
        plan_ctx = self._cb.build(
            spec=PLAN_SPEC,
            stage_name="plan",
            templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
            extras=extras,
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=plan_ctx.system_prompt),
                ChatMessage(role="user", content=plan_ctx.user_prompt),
            ],
            json_schema=BEAT_SHEET_SCHEMA,
            schema_name="BeatSheet",
            temperature=0.4,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        parsed = OutputParser.parse_json(raw)
        if not isinstance(parsed, dict):
            trace.add_stage(StageResult(
                stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
                errors=[StageError(kind="parse_error", message="not a dict")],
                latency_ms=latency,
            ))
            trace.outcome = "failed"
            raise ParseError(f"plan not a dict: {parsed!r}")
        normalized = _normalize_beat_sheet(parsed)
        if not normalized["beats"]:
            trace.add_stage(StageResult(
                stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
                errors=[StageError(kind="parse_error", message="no beats")],
                latency_ms=latency,
            ))
            trace.outcome = "failed"
            raise ParseError(f"plan has no beats: {raw!r}")
        trace.add_stage(StageResult(
            stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
            parsed_output=normalized,
            token_usage=TokenUsage(prompt=plan_ctx.token_estimate),
            latency_ms=latency,
        ))
        return normalized

    async def _run_write(self, trace: PipelineTrace, plan: dict) -> str:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        write_ctx = self._cb.build(
            spec=WRITE_SPEC,
            stage_name="write",
            templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
            extras={"plan": plan_text, "style": "", "anti_patterns": []},
        )
        t0 = time.perf_counter()
        raw = await self._client.chat(
            messages=[
                ChatMessage(role="system", content=write_ctx.system_prompt),
                ChatMessage(role="user", content=write_ctx.user_prompt),
            ],
            temperature=0.8,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        prose = OutputParser.parse_prose(raw)
        trace.add_stage(StageResult(
            stage_name="write", input_prompt=write_ctx.user_prompt, raw_output=raw,
            parsed_output=prose,
            token_usage=TokenUsage(prompt=write_ctx.token_estimate),
            latency_ms=latency,
        ))
        return prose

    async def _run_check(self, trace: PipelineTrace, plan: dict, prose: str) -> CheckOutput:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        ctx = self._cb.build(
            spec=CHECK_SPEC,
            stage_name="check",
            templates={"system": "stages/check/system.j2", "user": "stages/check/user.j2"},
            extras={"plan": plan_text, "prose": prose},
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            json_schema=CHECK_SCHEMA,
            schema_name="CheckOutput",
            temperature=0.2,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        try:
            parsed = OutputParser.parse_json(raw, schema=CheckOutput)
        except ParseError:
            # Defensive: if we can't parse, treat as clean so we don't block forever.
            parsed = CheckOutput(issues=[])
        trace.add_stage(StageResult(
            stage_name="check", input_prompt=ctx.user_prompt, raw_output=raw,
            parsed_output=parsed.model_dump(),
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))
        return parsed

    async def _run_extract(
        self, trace: PipelineTrace, plan: dict, prose: str, update_number: int,
    ) -> None:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        entities = self._world.list_entities()
        from app.world.schema import EntityStatus
        active_entities = [e for e in entities if e.status == EntityStatus.ACTIVE]
        ctx = self._cb.build(
            spec=EXTRACT_SPEC,
            stage_name="extract",
            templates={
                "system": "stages/extract/system.j2",
                "user": "stages/extract/user.j2",
            },
            extras={"plan": plan_text, "prose": prose, "entities": active_entities},
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            json_schema=EXTRACT_SCHEMA,
            schema_name="StateDelta",
            temperature=0.0,
        )
        latency = int((time.perf_counter() - t0) * 1000)

        from app.world.output_parser import OutputParser, ParseError as _ParseError
        try:
            extracted = OutputParser.parse_json(raw)
        except _ParseError as exc:
            trace.add_stage(StageResult(
                stage_name="extract",
                input_prompt=ctx.user_prompt,
                raw_output=raw,
                errors=[StageError(kind="parse_error", message=str(exc))],
                latency_ms=latency,
            ))
            return

        if not isinstance(extracted, dict):
            trace.add_stage(StageResult(
                stage_name="extract",
                input_prompt=ctx.user_prompt,
                raw_output=raw,
                errors=[StageError(kind="parse_error", message="extract not a dict")],
                latency_ms=latency,
            ))
            return

        known_ids = {e.id for e in active_entities}
        delta, build_issues = build_delta(extracted, update_number, known_ids=known_ids)

        # Also validate via WorldStateManager
        validation = self._world.validate_delta(delta)

        all_errors = (
            [StageError(kind="build_error", message=i.message) for i in build_issues]
            + [
                StageError(kind="validation_error", message=i.message)
                for i in validation.issues
                if i.severity == "error"
            ]
        )

        if all_errors:
            trace.add_stage(StageResult(
                stage_name="extract",
                input_prompt=ctx.user_prompt,
                raw_output=raw,
                parsed_output=extracted,
                errors=all_errors,
                latency_ms=latency,
            ))
            return

        # Apply atomically.
        self._world.apply_delta(delta, update_number)
        trace.add_stage(StageResult(
            stage_name="extract",
            input_prompt=ctx.user_prompt,
            raw_output=raw,
            parsed_output=extracted,
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))

    async def _run_revise(
        self, trace: PipelineTrace, plan: dict, prose: str,
        issues: list[CheckIssue],
    ) -> str:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        ctx = self._cb.build(
            spec=REVISE_SPEC,
            stage_name="revise",
            templates={"system": "stages/revise/system.j2", "user": "stages/revise/user.j2"},
            extras={
                "plan": plan_text,
                "prose": prose,
                "issues": [i.model_dump() for i in issues],
                "style": "",
                "anti_patterns": [],
            },
        )
        t0 = time.perf_counter()
        raw = await self._client.chat(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            temperature=0.6,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        revised = OutputParser.parse_prose(raw)
        trace.add_stage(StageResult(
            stage_name="revise", input_prompt=ctx.user_prompt, raw_output=raw,
            parsed_output=revised,
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))
        return revised
