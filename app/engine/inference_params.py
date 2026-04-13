from __future__ import annotations
from pydantic import BaseModel, Field


class InferenceParams(BaseModel):
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    repetition_penalty: float = 1.0
    thinking: bool = True


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    thinking: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion + self.thinking
