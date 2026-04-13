from __future__ import annotations
import json
from typing import AsyncIterator, Literal
import httpx
from pydantic import BaseModel
from .errors import InferenceError


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InferenceClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **extra: object,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, stream=False, extra=extra)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                r = await client.post(f"{self._base_url}/v1/chat/completions", json=payload)
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise InferenceError(str(e)) from e
            data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise InferenceError(f"malformed response: {data!r}") from e

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **extra: object,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, temperature, max_tokens, stream=True, extra=extra)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream("POST", f"{self._base_url}/v1/chat/completions", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            return
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
            except httpx.HTTPError as e:
                raise InferenceError(str(e)) from e

    def _build_payload(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        *,
        stream: bool,
        extra: dict[str, object],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)
        return payload
