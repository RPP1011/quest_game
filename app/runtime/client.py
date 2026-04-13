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
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retries = retries
        self._retry_backoff = retry_backoff

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        thinking: bool = True,
        **extra: object,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, stream=False,
                                      thinking=thinking, extra=extra)
        data = await self._post_with_retry(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise InferenceError(f"malformed response: {data!r}") from e

    async def chat_structured(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict,
        schema_name: str = "Output",
        temperature: float = 0.3,
        max_tokens: int | None = None,
        thinking: bool = True,
        **extra: object,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, stream=False,
                                      thinking=thinking, extra=extra)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
        }
        data = await self._post_with_retry(payload)
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
        thinking: bool = True,
        **extra: object,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, temperature, max_tokens, stream=True,
                                      thinking=thinking, extra=extra)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream("POST", f"{self._base_url}/v1/chat/completions", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line.removeprefix("data:").strip()
                        if data_str == "[DONE]":
                            return
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
            except httpx.HTTPError as e:
                raise InferenceError(str(e)) from e

    async def _post_with_retry(self, payload: dict) -> dict:
        import asyncio
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    r = await client.post(f"{self._base_url}/v1/chat/completions", json=payload)
                    r.raise_for_status()
                    return r.json()
            except httpx.HTTPError as e:
                last = e
                if attempt < self._retries:
                    await asyncio.sleep(self._retry_backoff * (2 ** attempt))
        raise InferenceError(str(last)) from last

    def _build_payload(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        *,
        stream: bool,
        thinking: bool,
        extra: dict[str, object],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)
        return payload
