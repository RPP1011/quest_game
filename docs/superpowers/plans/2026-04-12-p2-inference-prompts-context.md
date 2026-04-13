# P2 — Inference, Prompts, and Context Assembly

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the inference client with structured-output + thinking support, add a Jinja2-based prompt template system, and build a `ContextBuilder` that assembles per-stage prompts from world state with a token budget.

**Architecture:** New package `app/engine/` holds stage/pipeline abstractions. `InferenceClient` gains a `chat_structured` method (OpenAI `response_format: {"type": "json_schema"}`) and a `thinking` toggle. `PromptRenderer` loads Jinja2 templates from `prompts/`. `ContextBuilder` takes a `ContextSpec`, pulls relevant state from `WorldStateManager`, renders system+user prompts via `PromptRenderer`, budgets tokens (rough char/4 estimate for v1), and returns an `AssembledContext` with a manifest recording what was included and what was dropped under budget pressure.

**Tech Stack:** adds `jinja2`. Keeps Python 3.11, pydantic v2, `httpx`, `pytest`, `pytest-httpx`.

---

## File Structure

**Created:**
- `app/engine/__init__.py` — public surface
- `app/engine/inference_params.py` — `InferenceParams`, `TokenUsage`
- `app/engine/stages.py` — `StageConfig`, `StageResult`, `StageError`
- `app/engine/context_spec.py` — `ContextSpec` pydantic model + filters
- `app/engine/token_budget.py` — `TokenBudget` + rough `estimate_tokens(text)`
- `app/engine/prompt_renderer.py` — `PromptRenderer` (Jinja2 env, template loading)
- `app/engine/context_builder.py` — `ContextBuilder`, `AssembledContext`
- `prompts/stages/plan/system.j2`
- `prompts/stages/plan/user.j2`
- `prompts/stages/write/system.j2`
- `prompts/stages/write/user.j2`
- `prompts/components/entity.j2`
- `prompts/components/relationship.j2`
- `tests/engine/__init__.py`
- `tests/engine/conftest.py`
- `tests/engine/test_inference_structured.py` — extends runtime/test_client coverage
- `tests/engine/test_prompt_renderer.py`
- `tests/engine/test_context_spec.py`
- `tests/engine/test_token_budget.py`
- `tests/engine/test_context_builder.py`

**Modified:**
- `app/runtime/client.py` — add `chat_structured`, `thinking` toggle, retry
- `pyproject.toml` — add `jinja2>=3.1`

---

## Task 1: Extend InferenceClient — thinking toggle + retry

**Files:**
- Modify: `app/runtime/client.py`
- Modify: `tests/runtime/test_client.py` (add tests)

Design: `chat()` and `stream_chat()` gain a `thinking: bool = True` parameter that sets `{"chat_template_kwargs": {"enable_thinking": thinking}}` in the request body (llama.cpp / Qwen / Gemma convention for toggling reasoning). Also add a simple retry wrapper: up to `retries` attempts with 0.5s exponential backoff on `InferenceError`.

- [ ] **Step 1: Add tests to `tests/runtime/test_client.py`**

Append these:

```python
async def test_chat_passes_thinking_toggle(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions",
        method="POST",
        json={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
    )
    client = InferenceClient(base_url=BASE)
    await client.chat(messages=[ChatMessage(role="user", content="hi")], thinking=False)
    import json
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


async def test_chat_default_thinking_is_true(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions",
        method="POST",
        json={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
    )
    client = InferenceClient(base_url=BASE)
    await client.chat(messages=[ChatMessage(role="user", content="hi")])
    import json
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


async def test_chat_retries_on_error_then_succeeds(httpx_mock: HTTPXMock):
    httpx_mock.add_response(url=f"{BASE}/v1/chat/completions", method="POST", status_code=500)
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions",
        method="POST",
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )
    client = InferenceClient(base_url=BASE, retries=2, retry_backoff=0.0)
    assert await client.chat(messages=[ChatMessage(role="user", content="hi")]) == "ok"


async def test_chat_gives_up_after_retries(httpx_mock: HTTPXMock):
    for _ in range(3):
        httpx_mock.add_response(url=f"{BASE}/v1/chat/completions", method="POST", status_code=500)
    client = InferenceClient(base_url=BASE, retries=2, retry_backoff=0.0)
    with pytest.raises(InferenceError):
        await client.chat(messages=[ChatMessage(role="user", content="hi")])
```

- [ ] **Step 2: Run — expect failures**

Run: `uv run pytest tests/runtime/test_client.py -v`

- [ ] **Step 3: Modify `app/runtime/client.py`**

Replace the `InferenceClient` class with:

```python
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
```

- [ ] **Step 4: Run full client test file, all pass (existing 4 + new 4 = 8)**

Run: `uv run pytest tests/runtime/test_client.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/runtime/client.py tests/runtime/test_client.py
git commit -m "feat(runtime): thinking toggle, structured output, retry"
```

---

## Task 2: Structured output end-to-end test

**Files:**
- Create: `tests/engine/__init__.py` (empty)
- Create: `tests/engine/conftest.py`
- Create: `tests/engine/test_inference_structured.py`

- [ ] **Step 1: Write conftest + test**

`tests/engine/__init__.py`: empty.

`tests/engine/conftest.py`:
```python
# Shared fixtures for engine tests.
```

`tests/engine/test_inference_structured.py`:
```python
import json
import pytest
from pytest_httpx import HTTPXMock
from app.runtime.client import ChatMessage, InferenceClient


BASE = "http://127.0.0.1:8090"


async def test_chat_structured_includes_schema(httpx_mock: HTTPXMock):
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions", method="POST",
        json={"choices": [{"message": {"role": "assistant", "content": '{"x": 7}'}}]},
    )
    client = InferenceClient(base_url=BASE)
    result = await client.chat_structured(
        messages=[ChatMessage(role="user", content="give me x")],
        json_schema=schema,
        schema_name="Thing",
    )
    assert json.loads(result) == {"x": 7}
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "Thing"
    assert body["response_format"]["json_schema"]["schema"] == schema
```

- [ ] **Step 2: Run — pass (reuses client from Task 1)**

Run: `uv run pytest tests/engine/ -v`

- [ ] **Step 3: Commit**

```bash
git add tests/engine/__init__.py tests/engine/conftest.py tests/engine/test_inference_structured.py
git commit -m "test(engine): structured output contract via chat_structured"
```

---

## Task 3: InferenceParams + TokenUsage + StageConfig/Result

**Files:**
- Create: `app/engine/__init__.py` (empty for now)
- Create: `app/engine/inference_params.py`
- Create: `app/engine/stages.py`
- Create: `tests/engine/test_stages.py`

Design: all pydantic v2 models. Kept separate from runtime client so engine-level concepts don't pollute the HTTP layer.

- [ ] **Step 1: Empty `app/engine/__init__.py`**

- [ ] **Step 2: Write failing tests at `tests/engine/test_stages.py`**

```python
from app.engine.inference_params import InferenceParams, TokenUsage
from app.engine.stages import StageConfig, StageError, StageResult


def test_inference_params_defaults():
    p = InferenceParams()
    assert p.temperature == 0.7
    assert p.thinking is True
    assert p.max_tokens is None
    assert p.top_p == 1.0


def test_token_usage_total():
    u = TokenUsage(prompt=100, completion=50, thinking=20)
    assert u.total == 170


def test_stage_config_roundtrip():
    c = StageConfig(
        name="plan",
        system_prompt_template="stages/plan/system.j2",
        user_prompt_template="stages/plan/user.j2",
        output_schema={"type": "object"},
        inference_params=InferenceParams(temperature=0.4, thinking=True),
    )
    assert c.name == "plan"
    assert c.max_retries == 2


def test_stage_result_records_everything():
    r = StageResult(
        stage_name="plan",
        input_prompt="hello",
        raw_output="world",
        parsed_output={"beats": []},
        token_usage=TokenUsage(prompt=10, completion=5),
        latency_ms=120,
        retries=0,
        errors=[],
    )
    assert r.parsed_output == {"beats": []}
    assert r.token_usage.total == 15


def test_stage_error_carries_context():
    e = StageError(kind="parse_error", message="bad json", detail={"snippet": "{"})
    assert e.kind == "parse_error"
```

- [ ] **Step 3: Run — ImportError**

- [ ] **Step 4: Write `app/engine/inference_params.py`**

```python
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
```

- [ ] **Step 5: Write `app/engine/stages.py`**

```python
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
```

- [ ] **Step 6: Run all pass**

Run: `uv run pytest tests/engine/test_stages.py -v`

- [ ] **Step 7: Commit**

```bash
git add app/engine/__init__.py app/engine/inference_params.py app/engine/stages.py tests/engine/test_stages.py
git commit -m "feat(engine): stage config and result models"
```

---

## Task 4: TokenBudget + estimate_tokens

**Files:**
- Create: `app/engine/token_budget.py`
- Create: `tests/engine/test_token_budget.py`

Design: no real tokenizer for v1 — use `len(text) // 4` as a rough estimate (industry rule of thumb for English). `TokenBudget` is a dict-like that tracks remaining budget per section.

- [ ] **Step 1: Write failing tests**

```python
# tests/engine/test_token_budget.py
from app.engine.token_budget import TokenBudget, estimate_tokens


def test_estimate_tokens_rough():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 40) == 10


def test_budget_defaults_sum_reasonable():
    b = TokenBudget()
    assert b.total == 200_000
    # sum of sections + headroom + margin shouldn't exceed total
    assert b.world_state + b.narrative_history + b.system_prompt + b.style_config \
           + b.prior_stage_outputs + b.generation_headroom + b.safety_margin <= b.total


def test_budget_remaining():
    b = TokenBudget()
    used = {"world_state": 5_000, "narrative_history": 10_000}
    rem = b.remaining(used)
    assert rem == b.total - 15_000 - b.safety_margin


def test_budget_fits_positive():
    b = TokenBudget(total=1000, safety_margin=100)
    assert b.fits({"x": 500})
    assert not b.fits({"x": 950})
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Write `app/engine/token_budget.py`**

```python
from __future__ import annotations
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class TokenBudget:
    total: int = 200_000
    system_prompt: int = 15_000
    world_state: int = 30_000
    narrative_history: int = 40_000
    style_config: int = 10_000
    prior_stage_outputs: int = 15_000
    generation_headroom: int = 20_000
    safety_margin: int = 10_000

    def remaining(self, used: dict[str, int]) -> int:
        return self.total - sum(used.values()) - self.safety_margin

    def fits(self, used: dict[str, int]) -> bool:
        return self.remaining(used) >= 0
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add app/engine/token_budget.py tests/engine/test_token_budget.py
git commit -m "feat(engine): token budget + rough estimator"
```

---

## Task 5: ContextSpec

**Files:**
- Create: `app/engine/context_spec.py`
- Create: `tests/engine/test_context_spec.py`

Design: a v1-simplified `ContextSpec` that's still expressive enough for PLAN vs WRITE vs CHECK stages. Filters are simple predicates expressed as fields.

- [ ] **Step 1: Tests**

```python
# tests/engine/test_context_spec.py
from app.engine.context_spec import ContextSpec, EntityScope, NarrativeMode


def test_defaults_are_conservative():
    spec = ContextSpec()
    assert spec.entity_scope == EntityScope.RELEVANT
    assert spec.include_relationships is True
    assert spec.include_rules is True
    assert spec.narrative_mode == NarrativeMode.SUMMARY
    assert spec.narrative_window == 3
    assert spec.include_style is False
    assert spec.include_anti_patterns is False


def test_spec_serializable():
    spec = ContextSpec(
        entity_scope=EntityScope.ACTIVE,
        narrative_mode=NarrativeMode.FULL,
        narrative_window=5,
        include_style=True,
        include_character_voices=True,
    )
    dumped = spec.model_dump()
    assert dumped["entity_scope"] == "active"
    assert dumped["narrative_mode"] == "full"


def test_plan_preset_values():
    from app.engine.context_spec import PLAN_SPEC
    assert PLAN_SPEC.include_style is False
    assert PLAN_SPEC.include_rules is True


def test_write_preset_values():
    from app.engine.context_spec import WRITE_SPEC
    assert WRITE_SPEC.include_style is True
    assert WRITE_SPEC.include_character_voices is True
    assert WRITE_SPEC.narrative_mode.value == "full"
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement**

```python
# app/engine/context_spec.py
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class EntityScope(str, Enum):
    ALL = "all"           # every non-destroyed entity
    ACTIVE = "active"     # only status=active
    RELEVANT = "relevant" # active + referenced within the recent window


class NarrativeMode(str, Enum):
    FULL = "full"
    SUMMARY = "summary"
    NONE = "none"


class ContextSpec(BaseModel):
    entity_scope: EntityScope = EntityScope.RELEVANT
    include_relationships: bool = True
    include_rules: bool = True
    timeline_window: int = 10  # last N timeline events

    narrative_mode: NarrativeMode = NarrativeMode.SUMMARY
    narrative_window: int = 3  # last N narrative records (full mode)

    prior_stages: list[str] = Field(default_factory=list)

    include_style: bool = False
    include_character_voices: bool = False
    include_anti_patterns: bool = False
    include_foreshadowing: bool = True
    include_plot_threads: bool = True


PLAN_SPEC = ContextSpec(
    entity_scope=EntityScope.RELEVANT,
    narrative_mode=NarrativeMode.SUMMARY,
    include_style=False,
    include_rules=True,
    include_foreshadowing=True,
    include_plot_threads=True,
)


WRITE_SPEC = ContextSpec(
    entity_scope=EntityScope.ACTIVE,
    narrative_mode=NarrativeMode.FULL,
    narrative_window=3,
    include_style=True,
    include_character_voices=True,
    include_anti_patterns=True,
    include_rules=False,
    include_foreshadowing=False,
    include_plot_threads=False,
    prior_stages=["plan"],
)
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add app/engine/context_spec.py tests/engine/test_context_spec.py
git commit -m "feat(engine): ContextSpec + PLAN/WRITE presets"
```

---

## Task 6: PromptRenderer + starter templates

**Files:**
- Create: `app/engine/prompt_renderer.py`
- Create: `tests/engine/test_prompt_renderer.py`
- Create: `prompts/stages/plan/system.j2`
- Create: `prompts/stages/plan/user.j2`
- Create: `prompts/stages/write/system.j2`
- Create: `prompts/stages/write/user.j2`
- Create: `prompts/components/entity.j2`
- Create: `prompts/components/relationship.j2`
- Modify: `pyproject.toml` (add jinja2 dep)

Design: `PromptRenderer(prompts_dir)` wraps a `jinja2.Environment` with `FileSystemLoader(prompts_dir)`, autoescape off (we're generating prompts, not HTML), `undefined=StrictUndefined` so template bugs surface loudly.

- [ ] **Step 1: Add jinja2 to `pyproject.toml` dependencies**

Edit the `dependencies` list in `pyproject.toml` to include `"jinja2>=3.1",`. Then run `uv sync`.

- [ ] **Step 2: Create starter templates**

`prompts/stages/plan/system.j2`:
```
You are the narrative planner for a text-based quest game. Produce a concise beat sheet for the next chapter.

Respond with JSON matching the provided schema.
```

`prompts/stages/plan/user.j2`:
```
## World State

{% for entity in entities %}
- {{ entity.name }} ({{ entity.entity_type }}){% if entity.data.description %}: {{ entity.data.description }}{% endif %}
{% endfor %}

{% if plot_threads %}
## Active Plot Threads
{% for pt in plot_threads %}
- {{ pt.name }}: {{ pt.description }} ({{ pt.arc_position }})
{% endfor %}
{% endif %}

{% if recent_summaries %}
## Recent Events
{% for s in recent_summaries %}
- Update {{ s.update_number }}: {{ s.summary or s.raw_text[:200] }}
{% endfor %}
{% endif %}

## Player Action
{{ player_action }}

Produce the beat sheet now.
```

`prompts/stages/write/system.j2`:
```
You are the narrator for a text-based quest. Write prose for the next chapter, following the plan exactly.

{% if style %}## Style
{{ style }}{% endif %}

{% if anti_patterns %}## Avoid
{% for p in anti_patterns %}- {{ p }}
{% endfor %}{% endif %}
```

`prompts/stages/write/user.j2`:
```
## Plan
{{ plan }}

{% if recent_prose %}## Recent Prose (for continuity)
{% for p in recent_prose %}
{{ p }}

---
{% endfor %}{% endif %}

Write the next chapter now. Do not explain or preface — output the prose only.
```

`prompts/components/entity.j2`:
```
{{ entity.name }} ({{ entity.entity_type }}){% if entity.data.description %}: {{ entity.data.description }}{% endif %}
```

`prompts/components/relationship.j2`:
```
{{ r.source_id }} -{{ r.rel_type }}-> {{ r.target_id }}
```

- [ ] **Step 3: Write failing tests**

```python
# tests/engine/test_prompt_renderer.py
from pathlib import Path
import pytest
from app.engine.prompt_renderer import PromptRenderer


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


def test_renders_existing_template():
    r = PromptRenderer(PROMPTS)
    out = r.render("stages/plan/system.j2", {})
    assert "narrative planner" in out.lower()


def test_renders_with_context():
    r = PromptRenderer(PROMPTS)
    out = r.render("stages/plan/user.j2", {
        "entities": [],
        "plot_threads": [],
        "recent_summaries": [],
        "player_action": "Enter the tavern.",
    })
    assert "Enter the tavern." in out


def test_missing_variable_raises():
    r = PromptRenderer(PROMPTS)
    with pytest.raises(Exception):  # jinja2.UndefinedError
        r.render("stages/plan/user.j2", {})  # missing player_action


def test_missing_template_raises():
    r = PromptRenderer(PROMPTS)
    with pytest.raises(Exception):
        r.render("stages/nonexistent/system.j2", {})
```

- [ ] **Step 4: Run — ImportError/File-not-found**

- [ ] **Step 5: Implement `app/engine/prompt_renderer.py`**

```python
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptRenderer:
    def __init__(self, prompts_dir: str | Path) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, context: dict) -> str:
        tmpl = self._env.get_template(template_name)
        return tmpl.render(**context)
```

- [ ] **Step 6: Run — pass**

Run: `uv run pytest tests/engine/test_prompt_renderer.py -v`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock app/engine/prompt_renderer.py prompts tests/engine/test_prompt_renderer.py
git commit -m "feat(engine): Jinja2 PromptRenderer + stage template skeletons"
```

---

## Task 7: ContextBuilder

**Files:**
- Create: `app/engine/context_builder.py`
- Create: `tests/engine/test_context_builder.py`

Design: `ContextBuilder(world, renderer, budget)` with a single method `build(spec, stage_name, extras, templates)`. `extras` contains stage-specific kwargs that flow straight into the user-prompt template (e.g. `player_action`, `plan`). `templates` is a dict `{"system": path, "user": path}`. Returns `AssembledContext(system_prompt, user_prompt, token_estimate, manifest)` where `manifest` is a dict recording what was pulled from world state.

Budget compression (v1): if raw estimate exceeds allowed budget, apply in order: trim `recent_summaries` from oldest, drop `relationships` if still over, reduce entities to name-only.

- [ ] **Step 1: Write tests**

```python
# tests/engine/test_context_builder.py
from pathlib import Path
import pytest
from app.engine.context_builder import AssembledContext, ContextBuilder
from app.engine.context_spec import PLAN_SPEC, WRITE_SPEC
from app.engine.prompt_renderer import PromptRenderer
from app.engine.token_budget import TokenBudget
from app.world import (
    ArcPosition,
    Entity,
    EntityType,
    NarrativeRecord,
    PlotThread,
    Relationship,
    StateDelta,
    WorldStateManager,
)
from app.world.delta import EntityCreate, RelChange
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(
        entity_creates=[
            EntityCreate(entity=Entity(id="alice", entity_type=EntityType.CHARACTER,
                                        name="Alice", data={"description": "A tavern keeper."})),
            EntityCreate(entity=Entity(id="tavern", entity_type=EntityType.LOCATION,
                                        name="The Broken Anchor")),
        ],
        relationship_changes=[RelChange(
            action="add",
            relationship=Relationship(source_id="alice", target_id="tavern", rel_type="owns"),
        )],
    ), update_number=1)
    sm.add_plot_thread(PlotThread(
        id="pt:1", name="Mystery", description="A strange ship docked.",
        involved_entities=["alice"], arc_position=ArcPosition.RISING,
    ))
    sm.write_narrative(NarrativeRecord(
        update_number=1, raw_text="Alice wiped down the bar.",
        summary="Alice tidies the tavern.",
    ))
    yield sm
    conn.close()


def test_plan_context_includes_entities_and_action(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    ctx = cb.build(
        spec=PLAN_SPEC,
        stage_name="plan",
        templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
        extras={"player_action": "Greet the stranger."},
    )
    assert isinstance(ctx, AssembledContext)
    assert "Alice" in ctx.user_prompt
    assert "Greet the stranger." in ctx.user_prompt
    assert "Mystery" in ctx.user_prompt
    assert ctx.token_estimate > 0
    assert "entities" in ctx.manifest
    assert ctx.manifest["entities"]["included_count"] == 2


def test_write_context_includes_style_from_extras(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    ctx = cb.build(
        spec=WRITE_SPEC,
        stage_name="write",
        templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
        extras={
            "plan": "Scene: tavern. Beats: 1) Alice greets customer.",
            "style": "Literary, spare prose.",
            "anti_patterns": ["purple prose", "overused adverbs"],
        },
    )
    assert "Literary" in ctx.system_prompt
    assert "purple prose" in ctx.system_prompt
    assert "Scene: tavern" in ctx.user_prompt
    assert "Alice wiped" in ctx.user_prompt  # full narrative mode includes raw_text


def test_manifest_records_drops_under_pressure(world):
    tight = TokenBudget(total=400, system_prompt=100, world_state=50,
                        narrative_history=50, style_config=10,
                        prior_stage_outputs=10, generation_headroom=50,
                        safety_margin=10)
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), tight)
    ctx = cb.build(
        spec=PLAN_SPEC,
        stage_name="plan",
        templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
        extras={"player_action": "Look around."},
    )
    # Compression should have been applied
    assert ctx.manifest["compression_applied"] is True
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `app/engine/context_builder.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .context_spec import ContextSpec, EntityScope, NarrativeMode
from .prompt_renderer import PromptRenderer
from .token_budget import TokenBudget, estimate_tokens
from app.world.schema import EntityStatus
from app.world.state_manager import WorldStateManager


@dataclass
class AssembledContext:
    system_prompt: str
    user_prompt: str
    token_estimate: int
    manifest: dict[str, Any] = field(default_factory=dict)


class ContextBuilder:
    def __init__(
        self,
        world: WorldStateManager,
        renderer: PromptRenderer,
        budget: TokenBudget,
    ) -> None:
        self._world = world
        self._renderer = renderer
        self._budget = budget

    def build(
        self,
        *,
        spec: ContextSpec,
        stage_name: str,
        templates: dict[str, str],
        extras: dict[str, Any] | None = None,
    ) -> AssembledContext:
        extras = dict(extras or {})
        manifest: dict[str, Any] = {"stage": stage_name, "compression_applied": False}

        entities = self._select_entities(spec)
        relationships = self._world.list_relationships() if spec.include_relationships else []
        rules = self._world.list_rules() if spec.include_rules else []
        plot_threads = self._world.list_plot_threads() if spec.include_plot_threads else []
        recent_summaries = self._recent_narrative(spec)

        manifest["entities"] = {"included_count": len(entities)}
        manifest["relationships"] = {"included_count": len(relationships)}
        manifest["plot_threads"] = {"included_count": len(plot_threads)}
        manifest["recent_summaries"] = {"included_count": len(recent_summaries)}

        context = {
            "entities": entities,
            "relationships": relationships,
            "rules": rules,
            "plot_threads": plot_threads,
            "recent_summaries": recent_summaries,
            **extras,
        }

        system_prompt = self._renderer.render(templates["system"], context)
        user_prompt = self._renderer.render(templates["user"], context)
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)

        if total > self._budget.total - self._budget.safety_margin - self._budget.generation_headroom:
            manifest["compression_applied"] = True
            # 1. trim summaries
            while recent_summaries and total > self._budget.total // 2:
                recent_summaries = recent_summaries[1:]
                context["recent_summaries"] = recent_summaries
                user_prompt = self._renderer.render(templates["user"], context)
                total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
            # 2. drop relationships
            if total > self._budget.total // 2 and relationships:
                context["relationships"] = []
                user_prompt = self._renderer.render(templates["user"], context)
                total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
                manifest["relationships"]["dropped"] = True
            # 3. entity name-only
            if total > self._budget.total // 2:
                stripped = [
                    e.model_copy(update={"data": {}}) for e in entities
                ]
                context["entities"] = stripped
                user_prompt = self._renderer.render(templates["user"], context)
                total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
                manifest["entities"]["stripped"] = True

        return AssembledContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            token_estimate=total,
            manifest=manifest,
        )

    def _select_entities(self, spec: ContextSpec):
        all_entities = self._world.list_entities()
        if spec.entity_scope == EntityScope.ALL:
            return [e for e in all_entities if e.status != EntityStatus.DESTROYED]
        if spec.entity_scope == EntityScope.ACTIVE:
            return [e for e in all_entities if e.status == EntityStatus.ACTIVE]
        # RELEVANT: active + recently referenced
        threshold_updates = 10
        from app.world.schema import EntityStatus as ES
        def _relevant(e):
            if e.status != ES.ACTIVE:
                return False
            if e.last_referenced_update is None:
                return True
            return True  # v1: include all active; tighten later
        return [e for e in all_entities if _relevant(e)]

    def _recent_narrative(self, spec: ContextSpec):
        if spec.narrative_mode == NarrativeMode.NONE:
            return []
        records = self._world.list_narrative(limit=max(spec.narrative_window, 1) * 4)
        # Take last N
        return records[-spec.narrative_window:]
```

- [ ] **Step 4: Run tests — all pass**

Run: `uv run pytest tests/engine/test_context_builder.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/engine/context_builder.py tests/engine/test_context_builder.py
git commit -m "feat(engine): ContextBuilder with token-budget compression + manifest"
```

---

## Task 8: Public surface

**Files:**
- Modify: `app/engine/__init__.py`

- [ ] **Step 1: Write**

```python
from .context_builder import AssembledContext, ContextBuilder
from .context_spec import ContextSpec, EntityScope, NarrativeMode, PLAN_SPEC, WRITE_SPEC
from .inference_params import InferenceParams, TokenUsage
from .prompt_renderer import PromptRenderer
from .stages import StageConfig, StageError, StageResult
from .token_budget import TokenBudget, estimate_tokens

__all__ = [
    "AssembledContext",
    "ContextBuilder",
    "ContextSpec",
    "EntityScope",
    "InferenceParams",
    "NarrativeMode",
    "PLAN_SPEC",
    "PromptRenderer",
    "StageConfig",
    "StageError",
    "StageResult",
    "TokenBudget",
    "TokenUsage",
    "WRITE_SPEC",
    "estimate_tokens",
]
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "from app.engine import ContextBuilder, PLAN_SPEC, PromptRenderer; print('ok')"`

- [ ] **Step 3: Run full suite**

Run: `uv run pytest -v`
Expected: all prior tests green + new engine tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/engine/__init__.py
git commit -m "feat(engine): expose P2 public api"
```

---

## Done criteria

- `InferenceClient.chat_structured` round-trips a JSON schema request; retries on transient failure.
- `PromptRenderer` loads Jinja2 templates from `prompts/` with StrictUndefined.
- `ContextBuilder` produces an `AssembledContext` with manifest for both PLAN_SPEC and WRITE_SPEC.
- Token-budget compression kicks in when budget is tight and is recorded in the manifest.
- `uv run pytest` passes (14 runtime + 87 world + P2 engine tests).

Next plan (P3) will build the `Pipeline` orchestrator (linear PLAN → WRITE → commit), a minimal CLI, and a simple `QuestStore` for loading/saving quest sessions.
