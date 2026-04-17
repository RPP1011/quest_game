from __future__ import annotations
from datetime import datetime, timezone
from typing import Callable
from pydantic import BaseModel, Field, PrivateAttr
from .inference_params import TokenUsage
from .stages import StageResult


class PipelineTrace(BaseModel):
    trace_id: str
    trigger: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stages: list[StageResult] = Field(default_factory=list)
    outcome: str = "running"
    total_latency_ms: int = 0

    # Optional callback invoked after every add_stage() — used by the server
    # to incrementally persist the trace so a polling UI can see live
    # progress instead of waiting ~5 minutes for the final write.
    _on_update: Callable[["PipelineTrace"], None] | None = PrivateAttr(default=None)

    def set_on_update(self, fn: Callable[["PipelineTrace"], None] | None) -> None:
        self._on_update = fn

    def _maybe_persist(self) -> None:
        if self._on_update is not None:
            try:
                self._on_update(self)
            except Exception:
                # Live persistence is best-effort; never let a save failure
                # bubble up and break the pipeline.
                pass

    def add_stage(self, result: StageResult) -> None:
        self.stages.append(result)
        self.total_latency_ms += result.latency_ms
        self._maybe_persist()

    def set_outcome(self, outcome: str) -> None:
        """Set the final outcome and trigger an incremental persist so the
        on-disk trace reflects the terminal state (otherwise the save only
        fires on the last add_stage and shows outcome='running')."""
        self.outcome = outcome
        self._maybe_persist()

    @property
    def total_tokens(self) -> TokenUsage:
        total = TokenUsage()
        for s in self.stages:
            total = TokenUsage(
                prompt=total.prompt + s.token_usage.prompt,
                completion=total.completion + s.token_usage.completion,
                thinking=total.thinking + s.token_usage.thinking,
            )
        return total
