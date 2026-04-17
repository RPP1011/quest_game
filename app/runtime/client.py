from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal
import httpx
from pydantic import BaseModel
from .errors import InferenceError


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class TokenLogprob:
    """Logprob info for one generated token."""
    token: str
    logprob: float
    top_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class ChatWithLogprobs:
    """Return type for chat calls that request logprobs."""
    content: str
    token_logprobs: list[TokenLogprob] = field(default_factory=list)

    def score_token_distribution(
        self, position: int, candidates: list[str],
    ) -> dict[str, float]:
        """Extract a normalized probability distribution over ``candidates``
        at a specific token position.

        Looks at the top_logprobs dict at that position, filters to
        the candidate tokens, and softmax-normalizes. Tokens not in
        top_logprobs get probability 0.

        Returns ``{candidate: probability}`` summing to 1.0.
        """
        if position < 0 or position >= len(self.token_logprobs):
            return {c: 1.0 / len(candidates) for c in candidates}
        top = self.token_logprobs[position].top_logprobs
        raw = {}
        for c in candidates:
            if c in top:
                raw[c] = math.exp(top[c])
            else:
                raw[c] = 0.0
        total = sum(raw.values())
        if total == 0:
            return {c: 1.0 / len(candidates) for c in candidates}
        return {c: v / total for c, v in raw.items()}

    def expected_score(
        self, position: int, min_val: int = 1, max_val: int = 10,
    ) -> tuple[float, float]:
        """Compute E[score] and confidence at a token position.

        Assumes the token is an integer in [min_val, max_val]. Extracts
        the distribution over digit tokens, computes the expected value,
        and normalizes to [0, 1].

        Returns (e_score, confidence) where:
        - e_score is in [0, 1]
        - confidence is 1 - normalized_entropy, in [0, 1]
          (1.0 = all mass on one digit; 0.0 = uniform)
        """
        candidates = [str(i) for i in range(min_val, max_val + 1)]
        dist = self.score_token_distribution(position, candidates)
        # E[score]
        e_val = sum(int(c) * p for c, p in dist.items())
        e_score = (e_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        # Confidence = 1 - normalized entropy
        n = len(candidates)
        entropy = -sum(p * math.log(p + 1e-12) for p in dist.values())
        max_entropy = math.log(n)
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        return (e_score, confidence)


class InferenceClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        retries: int = 0,
        retry_backoff: float = 0.5,
        model: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retries = retries
        self._retry_backoff = retry_backoff
        self._model = model

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

    async def chat_with_logprobs(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        top_logprobs: int = 20,
        thinking: bool = False,
        **extra: object,
    ) -> ChatWithLogprobs:
        """Like ``chat`` but returns per-token logprobs.

        Uses llama-server's ``logprobs`` + ``top_logprobs`` params on
        the chat completions endpoint. Returns a ``ChatWithLogprobs``
        with the full text plus per-token logprob detail.

        Default ``thinking=False`` because logprob scoring calls want
        clean output tokens with no chain-of-thought prefix.
        """
        payload = self._build_payload(messages, temperature, max_tokens,
                                      stream=False, thinking=thinking, extra=extra)
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
        data = await self._post_with_retry(payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise InferenceError(f"malformed response: {data!r}") from e
        # Parse logprobs from the response
        token_logprobs: list[TokenLogprob] = []
        lp_data = choice.get("logprobs", {}) or {}
        for entry in lp_data.get("content", []) or []:
            token = entry.get("token", "")
            logprob = entry.get("logprob", 0.0)
            top = {}
            for item in entry.get("top_logprobs", []) or []:
                top[item.get("token", "")] = item.get("logprob", 0.0)
            token_logprobs.append(TokenLogprob(
                token=token, logprob=logprob, top_logprobs=top,
            ))
        return ChatWithLogprobs(content=content, token_logprobs=token_logprobs)

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
        if self._model is not None:
            payload["model"] = self._model
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)
        return payload
