from __future__ import annotations
import time
from typing import Protocol
from app.runtime.client import ChatMessage
from app.world.output_parser import OutputParser
from .check import CHECK_SCHEMA, CheckOutput
from .pipeline import BEAT_SHEET_SCHEMA, _normalize_beat_sheet
from .stages import StageResult
from .trace_store import TraceStore


class _ClientLike(Protocol):
    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str: ...
    async def chat(self, *, messages, **kw) -> str: ...


_STRUCTURED_STAGES = {
    "plan": (BEAT_SHEET_SCHEMA, "BeatSheet"),
    "check": (CHECK_SCHEMA, "CheckOutput"),
}
_FREE_TEXT_STAGES = {"write", "revise"}


class DiagnosticsManager:
    def __init__(self, *, client: _ClientLike, store: TraceStore) -> None:
        self._client = client
        self._store = store

    async def replay(
        self, trace_id: str, stage_name: str, prompt_override: str | None = None,
    ) -> StageResult:
        trace = self._store.load(trace_id)
        target = next((s for s in trace.stages if s.stage_name == stage_name), None)
        if target is None:
            raise ValueError(f"stage {stage_name!r} not in trace {trace_id}")

        prompt = prompt_override if prompt_override is not None else target.input_prompt
        messages = [ChatMessage(role="user", content=prompt)]

        t0 = time.perf_counter()
        if stage_name in _STRUCTURED_STAGES:
            schema, name = _STRUCTURED_STAGES[stage_name]
            raw = await self._client.chat_structured(
                messages=messages, json_schema=schema, schema_name=name, temperature=0.3,
            )
            if stage_name == "plan":
                parsed = _normalize_beat_sheet(OutputParser.parse_json(raw) or {})
            else:  # check
                try:
                    parsed = OutputParser.parse_json(raw, schema=CheckOutput).model_dump()
                except Exception:
                    parsed = {"issues": []}
        elif stage_name in _FREE_TEXT_STAGES:
            raw = await self._client.chat(messages=messages, temperature=0.7)
            parsed = OutputParser.parse_prose(raw)
        else:
            raise ValueError(f"unknown stage: {stage_name!r}")
        latency = int((time.perf_counter() - t0) * 1000)

        return StageResult(
            stage_name=stage_name,
            input_prompt=prompt,
            raw_output=raw,
            parsed_output=parsed,
            latency_ms=latency,
        )
