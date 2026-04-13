# P3 — Pipeline Orchestrator + CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a linear PLAN → WRITE → COMMIT pipeline and a minimal CLI that plays a quest against a running `llama-server`.

**Architecture:** `app/engine/pipeline.py` holds the `Pipeline` class. Each stage: render context → call LLM (structured for PLAN, streaming prose for WRITE) → parse → record `StageResult`. After WRITE the pipeline commits the chapter via `WorldStateManager.write_narrative`. `PipelineTrace` collects stage results for later diagnostics. `app/cli/play.py` wires everything together: opens a DB, starts a pre-launched llama-server (or connects to one), runs one chapter per input line, prints prose.

**Tech Stack:** additions: `typer` for CLI ergonomics.

---

## File Structure

**Created:**
- `app/engine/trace.py` — `PipelineTrace`
- `app/engine/pipeline.py` — `Pipeline`, `PipelineOutput`, `LinearFlow`
- `app/cli/__init__.py`
- `app/cli/play.py` — typer CLI entrypoint
- `tests/engine/test_pipeline.py`
- `tests/cli/__init__.py`
- `tests/cli/test_cli_smoke.py`

**Modified:**
- `app/engine/__init__.py` (export Pipeline)
- `pyproject.toml` (add `typer`)

---

## Task 1: PipelineTrace

**Files:**
- Create: `app/engine/trace.py`
- Create: `tests/engine/test_trace.py`

- [ ] **Step 1: Test**

```python
# tests/engine/test_trace.py
from app.engine.stages import StageResult, TokenUsage
from app.engine.trace import PipelineTrace


def test_trace_accumulates_stages():
    t = PipelineTrace(trace_id="abc", trigger="greet")
    t.add_stage(StageResult(stage_name="plan", input_prompt="p", raw_output="r",
                             parsed_output={"beats": []},
                             token_usage=TokenUsage(prompt=10, completion=5)))
    t.add_stage(StageResult(stage_name="write", input_prompt="p", raw_output="prose",
                             parsed_output="prose",
                             token_usage=TokenUsage(prompt=20, completion=40)))
    assert t.total_tokens.total == 75
    assert [s.stage_name for s in t.stages] == ["plan", "write"]


def test_trace_serializable():
    t = PipelineTrace(trace_id="abc", trigger="x")
    dumped = t.model_dump()
    assert dumped["trace_id"] == "abc"
    assert dumped["stages"] == []
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement**

```python
# app/engine/trace.py
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
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add app/engine/trace.py tests/engine/test_trace.py
git commit -m "feat(engine): PipelineTrace for accumulated stage results"
```

---

## Task 2: Pipeline (linear PLAN → WRITE → COMMIT)

**Files:**
- Create: `app/engine/pipeline.py`
- Create: `tests/engine/test_pipeline.py`

Design: `Pipeline(world, context_builder, client)` runs one chapter per `.run(player_action)` call. For PLAN stage, calls `client.chat_structured(schema=BEAT_SHEET_SCHEMA)` and parses into a dict (shape: `{beats: [str], suggested_choices: [str]}`). For WRITE stage, calls `client.chat` (non-streaming for simplicity in v1 — streaming can be added later) and returns prose. Then writes narrative + appends the player-action as state (no rich state deltas in v1 — that's for a later milestone).

PLAN schema is intentionally minimal — a dict with `beats` (list of strings) and `suggested_choices` (list of strings). WRITE returns raw prose. That's the MVP; CHECK/REVISE and rich deltas come in P4.

The test uses a fake `InferenceClient`.

- [ ] **Step 1: Tests**

```python
# tests/engine/test_pipeline.py
from pathlib import Path
import pytest
from app.engine import ContextBuilder, PLAN_SPEC, PromptRenderer, TokenBudget, WRITE_SPEC
from app.engine.pipeline import Pipeline, PipelineOutput
from app.world import (
    Entity, EntityType, PlotThread, ArcPosition,
    StateDelta, WorldStateManager,
)
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        self.calls.append(("structured", messages, json_schema))
        return '{"beats": ["Alice greets Bob."], "suggested_choices": ["Ask who they are", "Leave"]}'

    async def chat(self, *, messages, **kw):
        self.calls.append(("chat", messages))
        return "Alice looked up from the bar. \"Can I help you?\" she asked."


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(
        entity_creates=[EntityCreate(entity=Entity(
            id="alice", entity_type=EntityType.CHARACTER, name="Alice",
            data={"description": "The tavern keeper."}))],
    ), update_number=1)
    sm.add_plot_thread(PlotThread(
        id="pt:1", name="Arrival", description="A stranger enters.",
        arc_position=ArcPosition.RISING,
    ))
    yield sm
    conn.close()


async def test_pipeline_runs_plan_and_write(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    fake = FakeClient()
    p = Pipeline(world, cb, fake)
    result = await p.run(player_action="Greet the stranger.", update_number=2)
    assert isinstance(result, PipelineOutput)
    assert "Alice looked up" in result.prose
    assert result.choices == ["Ask who they are", "Leave"]
    assert [s.stage_name for s in result.trace.stages] == ["plan", "write"]
    assert result.trace.outcome == "committed"


async def test_pipeline_persists_narrative(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    fake = FakeClient()
    p = Pipeline(world, cb, fake)
    await p.run(player_action="Greet.", update_number=2)
    records = world.list_narrative()
    assert len(records) == 1
    assert records[0].player_action == "Greet."
    assert "Alice looked up" in records[0].raw_text


async def test_pipeline_surfaces_parse_error(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())

    class BadClient(FakeClient):
        async def chat_structured(self, **kw):
            return "not json at all"

    p = Pipeline(world, cb, BadClient())
    with pytest.raises(Exception):
        await p.run(player_action="x", update_number=2)
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `app/engine/pipeline.py`**

```python
from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol
from app.runtime.client import ChatMessage
from app.world.output_parser import OutputParser, ParseError
from app.world.schema import NarrativeRecord
from app.world.state_manager import WorldStateManager
from .context_builder import ContextBuilder
from .context_spec import PLAN_SPEC, WRITE_SPEC
from .inference_params import TokenUsage
from .stages import StageError, StageResult
from .trace import PipelineTrace


BEAT_SHEET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "beats": {"type": "array", "items": {"type": "string"}},
        "suggested_choices": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["beats", "suggested_choices"],
    "additionalProperties": False,
}


class InferenceClientLike(Protocol):
    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str: ...
    async def chat(self, *, messages, **kw) -> str: ...


@dataclass
class PipelineOutput:
    prose: str
    choices: list[str]
    beats: list[str]
    trace: PipelineTrace


class Pipeline:
    def __init__(
        self,
        world: WorldStateManager,
        context_builder: ContextBuilder,
        client: InferenceClientLike,
    ) -> None:
        self._world = world
        self._cb = context_builder
        self._client = client

    async def run(self, *, player_action: str, update_number: int) -> PipelineOutput:
        trace = PipelineTrace(trace_id=uuid.uuid4().hex, trigger=player_action)

        # ---- PLAN ----
        plan_ctx = self._cb.build(
            spec=PLAN_SPEC,
            stage_name="plan",
            templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
            extras={"player_action": player_action},
        )
        t0 = time.perf_counter()
        plan_raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=plan_ctx.system_prompt),
                ChatMessage(role="user", content=plan_ctx.user_prompt),
            ],
            json_schema=BEAT_SHEET_SCHEMA,
            schema_name="BeatSheet",
            temperature=0.4,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        errors: list[StageError] = []
        try:
            plan_parsed = OutputParser.parse_json(plan_raw)
            if not isinstance(plan_parsed, dict) or "beats" not in plan_parsed:
                raise ParseError(f"beat sheet malformed: {plan_parsed!r}")
        except ParseError as e:
            errors.append(StageError(kind="parse_error", message=str(e)))
            trace.add_stage(StageResult(
                stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=plan_raw,
                errors=errors, latency_ms=latency,
            ))
            trace.outcome = "failed"
            raise
        trace.add_stage(StageResult(
            stage_name="plan",
            input_prompt=plan_ctx.user_prompt,
            raw_output=plan_raw,
            parsed_output=plan_parsed,
            token_usage=TokenUsage(prompt=plan_ctx.token_estimate),
            latency_ms=latency,
        ))

        # ---- WRITE ----
        plan_text = "\n".join(f"- {b}" for b in plan_parsed["beats"])
        write_ctx = self._cb.build(
            spec=WRITE_SPEC,
            stage_name="write",
            templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
            extras={
                "plan": plan_text,
                "style": "",
                "anti_patterns": [],
            },
        )
        t0 = time.perf_counter()
        prose_raw = await self._client.chat(
            messages=[
                ChatMessage(role="system", content=write_ctx.system_prompt),
                ChatMessage(role="user", content=write_ctx.user_prompt),
            ],
            temperature=0.8,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        prose = OutputParser.parse_prose(prose_raw)
        trace.add_stage(StageResult(
            stage_name="write",
            input_prompt=write_ctx.user_prompt,
            raw_output=prose_raw,
            parsed_output=prose,
            token_usage=TokenUsage(prompt=write_ctx.token_estimate),
            latency_ms=latency,
        ))

        # ---- COMMIT ----
        self._world.write_narrative(NarrativeRecord(
            update_number=update_number,
            raw_text=prose,
            player_action=player_action,
            pipeline_trace_id=trace.trace_id,
        ))
        trace.outcome = "committed"

        return PipelineOutput(
            prose=prose,
            choices=plan_parsed.get("suggested_choices", []),
            beats=plan_parsed["beats"],
            trace=trace,
        )
```

- [ ] **Step 4: Run — 3 tests pass**

Run: `uv run pytest tests/engine/test_pipeline.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/engine/pipeline.py tests/engine/test_pipeline.py
git commit -m "feat(engine): linear Pipeline (PLAN->WRITE->COMMIT)"
```

---

## Task 3: Expose Pipeline + PipelineTrace

**Files:**
- Modify: `app/engine/__init__.py`

- [ ] **Step 1: Add to exports (append to existing `__init__.py`)**

Update `app/engine/__init__.py` by adding:

```python
from .pipeline import BEAT_SHEET_SCHEMA, Pipeline, PipelineOutput
from .trace import PipelineTrace
```

And append `"BEAT_SHEET_SCHEMA"`, `"Pipeline"`, `"PipelineOutput"`, `"PipelineTrace"` to `__all__`.

- [ ] **Step 2: Verify**

Run: `uv run python -c "from app.engine import Pipeline, PipelineTrace; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add app/engine/__init__.py
git commit -m "feat(engine): expose Pipeline + PipelineTrace"
```

---

## Task 4: CLI — `quest play`

**Files:**
- Create: `app/cli/__init__.py` (empty)
- Create: `app/cli/play.py`
- Create: `tests/cli/__init__.py` (empty)
- Create: `tests/cli/test_cli_smoke.py`
- Modify: `pyproject.toml` (add `typer>=0.12`; add `[project.scripts]` entry)

Design: a single `quest` command with subcommands. For v1: `quest play --db PATH --seed SEEDFILE --server URL`. Reads lines from stdin as player actions, prints the prose returned by `Pipeline.run`. If `--seed` is given and the DB is new/empty, loads the seed first. Model running is out of scope — user brings their own `llama-server`.

- [ ] **Step 1: Add dependency and script entry in `pyproject.toml`**

Add `"typer>=0.12",` to `dependencies`. Add below `[project]`:

```toml
[project.scripts]
quest = "app.cli.play:app"
```

Run `uv sync`.

- [ ] **Step 2: Write failing test**

```python
# tests/cli/test_cli_smoke.py
import json
from pathlib import Path
from typer.testing import CliRunner
from app.cli.play import app


def test_cli_help():
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "play" in r.stdout.lower()


def test_cli_init_from_seed(tmp_path: Path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({
        "entities": [
            {"id": "alice", "entity_type": "character", "name": "Alice"},
        ],
    }))
    db = tmp_path / "q.db"
    runner = CliRunner()
    r = runner.invoke(app, ["init", "--db", str(db), "--seed", str(seed)])
    assert r.exit_code == 0, r.stdout
    assert db.exists()
    # Verify the seed loaded
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    conn = open_db(db)
    sm = WorldStateManager(conn)
    assert [e.id for e in sm.list_entities()] == ["alice"]
    conn.close()
```

- [ ] **Step 3: Run — ImportError**

- [ ] **Step 4: Implement `app/cli/play.py`**

```python
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import typer
from app.engine import (
    ContextBuilder,
    Pipeline,
    PromptRenderer,
    TokenBudget,
)
from app.runtime.client import InferenceClient
from app.world import SeedLoader, WorldStateManager
from app.world.db import open_db


app = typer.Typer(help="Quest game CLI.")

PROMPTS = Path(__file__).parent.parent.parent / "prompts"


def _open_world(db_path: Path) -> WorldStateManager:
    conn = open_db(db_path)
    return WorldStateManager(conn)


@app.command()
def init(
    db: Path = typer.Option(..., help="Path to the quest SQLite DB (will be created)."),
    seed: Path = typer.Option(..., help="Seed JSON file."),
) -> None:
    """Initialize a new quest database from a seed JSON file."""
    sm = _open_world(db)
    payload = SeedLoader.load(seed)
    for rule in payload.rules:
        sm.add_rule(rule)
    for hook in payload.foreshadowing:
        sm.add_foreshadowing(hook)
    for pt in payload.plot_threads:
        sm.add_plot_thread(pt)
    sm.apply_delta(payload.delta, update_number=0)
    typer.echo(f"Seeded {db} with {len(payload.delta.entity_creates)} entities.")


@app.command()
def play(
    db: Path = typer.Option(..., help="Path to the quest DB."),
    server: str = typer.Option("http://127.0.0.1:8090", help="llama-server base URL."),
) -> None:
    """Play the quest — reads player actions from stdin, prints prose."""
    sm = _open_world(db)
    client = InferenceClient(base_url=server, retries=1)
    cb = ContextBuilder(sm, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(sm, cb, client)

    typer.echo("Quest started. Type an action and press enter. Ctrl-D to quit.")
    # Determine next update number
    records = sm.list_narrative(limit=10_000)
    update_number = (max((r.update_number for r in records), default=0)) + 1

    for line in sys.stdin:
        action = line.strip()
        if not action:
            continue
        typer.echo("\n--- generating chapter ---\n", err=True)
        try:
            out = asyncio.run(pipeline.run(player_action=action, update_number=update_number))
        except Exception as e:
            typer.echo(f"[error] {e}", err=True)
            continue
        typer.echo(out.prose)
        typer.echo("\nChoices:")
        for i, c in enumerate(out.choices, 1):
            typer.echo(f"  {i}. {c}")
        typer.echo()
        update_number += 1
```

- [ ] **Step 5: Run — tests pass**

Run: `uv run pytest tests/cli/ -v`

- [ ] **Step 6: Commit**

```bash
git add app/cli pyproject.toml uv.lock tests/cli
git commit -m "feat(cli): typer-based 'quest' CLI with init + play"
```

---

## Done criteria

- `uv run quest --help` shows `init` and `play` commands.
- `uv run quest init --db /tmp/q.db --seed /tmp/seed.json` creates a seeded DB.
- With a running `llama-server` pointing at Gemma, `uv run quest play --db /tmp/q.db` reads actions from stdin and prints prose.
- Full pytest suite green.
