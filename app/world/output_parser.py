from __future__ import annotations
import json
import re
from typing import Any, Type, TypeVar
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class ParseError(Exception):
    pass


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"^(?:sure|here(?:'s| is|s)|okay|ok|certainly|absolutely)[^\n]{0,80}[:!.]?\s*\n\s*\n",
    re.IGNORECASE,
)


class OutputParser:
    @staticmethod
    def parse_json(text: str, schema: Type[T] | None = None) -> Any | T:
        cleaned = _THINK_RE.sub("", text).strip()
        m = _FENCE_RE.match(cleaned)
        if m:
            cleaned = m.group(1).strip()

        candidate = cleaned
        parsed: Any = None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = _extract_balanced(candidate)
            if parsed is None:
                raise ParseError(f"no JSON found in text: {text[:200]!r}")

        if schema is None:
            return parsed
        try:
            return schema.model_validate(parsed)
        except ValidationError as e:
            raise ParseError(f"schema validation failed: {e}") from e

    @staticmethod
    def parse_prose(text: str) -> str:
        cleaned = _THINK_RE.sub("", text).strip()
        cleaned = _PREAMBLE_RE.sub("", cleaned, count=1)
        return cleaned.strip()


def _extract_balanced(text: str) -> Any | None:
    """Find the largest balanced {...} or [...] substring and try to parse it."""
    best: Any = None
    best_len = 0
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    snippet = text[start : i + 1]
                    try:
                        parsed = json.loads(snippet)
                    except json.JSONDecodeError:
                        continue
                    if len(snippet) > best_len:
                        best = parsed
                        best_len = len(snippet)
    return best
