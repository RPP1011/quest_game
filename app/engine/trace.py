from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from .inference_params import TokenUsage
from .stages import StageResult


class PipelineTrace(BaseModel):
    trace_id: str
    trigger: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stages: list[StageResult] = Field(default_factory=list)
    outcome: str = "running"
    total_latency_ms: int = 0

    def add_stage(self, result: StageResult) -> None:
        self.stages.append(result)
        self.total_latency_ms += result.latency_ms

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
