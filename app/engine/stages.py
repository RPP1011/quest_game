from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from .inference_params import InferenceParams, TokenUsage


class StageError(BaseModel):
    kind: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class StageConfig(BaseModel):
    name: str
    system_prompt_template: str
    user_prompt_template: str
    output_schema: dict | None = None
    inference_params: InferenceParams = Field(default_factory=InferenceParams)
    max_retries: int = 2


class StageResult(BaseModel):
    stage_name: str
    input_prompt: str
    raw_output: str
    parsed_output: Any = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    retries: int = 0
    errors: list[StageError] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)
