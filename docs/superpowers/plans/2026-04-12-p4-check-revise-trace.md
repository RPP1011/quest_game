# P4 — CHECK, REVISE, and Trace Persistence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Round out the engine with a CHECK stage that flags prose issues, a REVISE stage that corrects them, branching pipeline flow (no issues → commit; fixable → revise; critical → replan), and file-backed `PipelineTrace` persistence + replay.

**Architecture:** Two new stages added to `Pipeline`, controlled by a severity-aware flow. A `CheckOutput` schema enumerates issues with severity. `TraceStore` writes each trace to `data/traces/<trace_id>.json` on commit (or failure) so they can be replayed. `DiagnosticsManager.replay(trace_id, stage)` re-runs a single stage against the stored context. REPLAN is capped at 1.

**Tech Stack:** No new deps. Same Python 3.11 + pydantic + httpx + Jinja2.

---

## File Structure

**Created:**
- `app/engine/check.py` — `CheckIssue`, `CheckOutput`, `CHECK_SCHEMA`
- `app/engine/trace_store.py` — `TraceStore` (JSON persistence)
- `app/engine/diagnostics.py` — `DiagnosticsManager`
- `prompts/stages/check/system.j2`, `prompts/stages/check/user.j2`
- `prompts/stages/revise/system.j2`, `prompts/stages/revise/user.j2`
- `tests/engine/test_check.py`
- `tests/engine/test_pipeline_branching.py`
- `tests/engine/test_trace_store.py`
- `tests/engine/test_diagnostics.py`

**Modified:**
- `app/engine/pipeline.py` — add CHECK/REVISE stages, branching flow, replan-once cap
- `app/engine/context_spec.py` — add `CHECK_SPEC`, `REVISE_SPEC`
- `app/engine/__init__.py` — export new symbols
- `app/cli/play.py` — write trace on each run, optional `--traces DIR` flag

---

## Task 1: CheckOutput schema + CHECK_SPEC

**Files:**
- Create: `app/engine/check.py`
- Modify: `app/engine/context_spec.py`
- Create: `tests/engine/test_check.py`

Design: `CheckOutput` is a pydantic model with a list of `CheckIssue(severity, category, message, suggested_fix)`. Categories: `continuity`, `world_rule`, `plan_adherence`, `prose_quality`. Severity: `info | warning | error | critical`. `critical` → REPLAN, `error|warning` → REVISE, `info` → commit as-is.

- [ ] **Step 1: Tests**

```python
# tests/engine/test_check.py
from app.engine.check import CHECK_SCHEMA, CheckIssue, CheckOutput
from app.engine.context_spec import CHECK_SPEC, REVISE_SPEC


def test_check_output_summary_classifies_severity():
    out = CheckOutput(issues=[
        CheckIssue(severity="info", category="prose_quality", message="meh"),
        CheckIssue(severity="warning", category="continuity", message="x"),
        CheckIssue(severity="critical", category="world_rule", message="magic banned"),
    ])
    assert out.has_critical is True
    assert out.has_fixable is True   # warning counts as fixable
    assert out.all_trivial is False


def test_check_output_empty_is_clean():
    out = CheckOutput(issues=[])
    assert out.has_critical is False
    assert out.has_fixable is False
    assert out.all_trivial is True


def test_check_schema_is_json_schema_shape():
    assert CHECK_SCHEMA["type"] == "object"
    assert "issues" in CHECK_SCHEMA["properties"]


def test_check_spec_conservative_defaults():
    # CHECK needs to see rules, plot threads, recent prose — not style
    assert CHECK_SPEC.include_rules is True
    assert CHECK_SPEC.include_style is False


def test_revise_spec_like_write_plus_check():
    assert REVISE_SPEC.include_style is True
    assert "check" in REVISE_SPEC.prior_stages
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `app/engine/check.py`**

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


Severity = Literal["info", "warning", "error", "critical"]
Category = Literal["continuity", "world_rule", "plan_adherence", "prose_quality"]


class CheckIssue(BaseModel):
    severity: Severity
    category: Category
    message: str
    suggested_fix: str | None = None


class CheckOutput(BaseModel):
    issues: list[CheckIssue] = Field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    @property
    def has_fixable(self) -> bool:
        return any(i.severity in ("warning", "error") for i in self.issues)

    @property
    def all_trivial(self) -> bool:
        return all(i.severity == "info" for i in self.issues)


CHECK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"]},
                    "category": {"type": "string",
                                 "enum": ["continuity", "world_rule", "plan_adherence", "prose_quality"]},
                    "message": {"type": "string"},
                    "suggested_fix": {"type": ["string", "null"]},
                },
                "required": ["severity", "category", "message"],
            },
        },
    },
    "required": ["issues"],
}
```

- [ ] **Step 4: Append to `app/engine/context_spec.py`**

At the bottom (after `WRITE_SPEC`):

```python
CHECK_SPEC = ContextSpec(
    entity_scope=EntityScope.ACTIVE,
    narrative_mode=NarrativeMode.FULL,
    narrative_window=2,
    include_style=False,
    include_rules=True,
    include_plot_threads=True,
    include_foreshadowing=True,
    prior_stages=["plan", "write"],
)


REVISE_SPEC = ContextSpec(
    entity_scope=EntityScope.ACTIVE,
    narrative_mode=NarrativeMode.FULL,
    narrative_window=2,
    include_style=True,
    include_character_voices=True,
    include_anti_patterns=True,
    include_rules=False,
    prior_stages=["plan", "write", "check"],
)
```

- [ ] **Step 5: Run — all pass**

Run: `uv run pytest tests/engine/test_check.py -v`

- [ ] **Step 6: Commit**

```bash
git add app/engine/check.py app/engine/context_spec.py tests/engine/test_check.py
git commit -m "feat(engine): CheckOutput schema + CHECK/REVISE context specs"
```

---

## Task 2: CHECK + REVISE prompt templates

**Files:**
- Create: `prompts/stages/check/system.j2`
- Create: `prompts/stages/check/user.j2`
- Create: `prompts/stages/revise/system.j2`
- Create: `prompts/stages/revise/user.j2`

- [ ] **Step 1: Write `prompts/stages/check/system.j2`**

```
You are the consistency checker for a text-based quest game. Review the proposed next-chapter prose against the world state, the plan the narrator was given, and the established world rules.

Return a JSON object with an `issues` array. Each issue has:
- severity: "info" | "warning" | "error" | "critical"
- category: "continuity" | "world_rule" | "plan_adherence" | "prose_quality"
- message: describe the problem
- suggested_fix: a concrete edit suggestion, or null

Use "critical" only when the plan itself must be thrown out (world rule violation, plot-breaking contradiction, major plan adherence failure). Use "error"/"warning" for prose-level problems that a revision can fix. Use "info" for trivial observations. Return an empty `issues` array if the prose is clean.
```

- [ ] **Step 2: Write `prompts/stages/check/user.j2`**

```
## Plan the narrator was given
{{ plan }}

## Proposed Prose
{{ prose }}

{% if rules %}
## Active World Rules
{% for r in rules %}
- {{ r.description }}
{% endfor %}
{% endif %}

{% if recent_summaries %}
## Recent Events
{% for s in recent_summaries %}
- Update {{ s.update_number }}: {{ s.summary or s.raw_text[:200] }}
{% endfor %}
{% endif %}

Evaluate the prose and return the issues JSON now.
```

- [ ] **Step 3: Write `prompts/stages/revise/system.j2`**

```
You are the revision pass for a text-based quest game. Take the previous prose and the list of issues raised by the checker, and produce a corrected version.

- Preserve the overall structure and voice.
- Apply each fixable issue's suggested fix.
- Do not add new plot points beyond what the plan specifies.
- Output the revised prose only. No explanation.
```

- [ ] **Step 4: Write `prompts/stages/revise/user.j2`**

```
## Plan
{{ plan }}

## Original Prose
{{ prose }}

## Issues to Fix
{% for i in issues %}
- [{{ i.severity }}/{{ i.category }}] {{ i.message }}{% if i.suggested_fix %} — fix: {{ i.suggested_fix }}{% endif %}
{% endfor %}

Produce the revised prose now.
```

- [ ] **Step 5: Commit**

```bash
git add prompts/stages/check prompts/stages/revise
git commit -m "feat(prompts): CHECK and REVISE stage templates"
```

---

## Task 3: Pipeline branching (CHECK → commit | revise | replan)

**Files:**
- Modify: `app/engine/pipeline.py`
- Create: `tests/engine/test_pipeline_branching.py`

Design additions to `Pipeline`:
- After WRITE, render CHECK_SPEC context with `plan` and `prose` extras, call `chat_structured(CHECK_SCHEMA)`, parse into `CheckOutput`.
- If `has_critical` AND `replan_attempts < 1` — re-run PLAN with a `critical_feedback` extra containing the issues, then re-run WRITE. Then re-CHECK.
- Else if `has_fixable` — run REVISE stage, replace prose, run CHECK again (re-check). After the re-check, commit regardless of remaining issues but record them in the trace as `flagged_qm`.
- Else commit.

REPLAN cap: at most 1 full replan per pipeline invocation. If CHECK still critical after replan, commit with `outcome="flagged_qm"`. RECHECK: one REVISE+RECHECK loop max.

We'll keep the normalizer applied to check output too (for weaker models).

- [ ] **Step 1: Tests**

```python
# tests/engine/test_pipeline_branching.py
from pathlib import Path
import pytest
from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.world import Entity, EntityType, StateDelta, WorldStateManager
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class ScriptedClient:
    """Replay a scripted sequence of responses, one per call in order."""
    def __init__(self, responses: list[dict]) -> None:
        # Each response: {"kind": "structured"|"chat", "content": str}
        self._responses = list(responses)
        self.log: list[str] = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str:
        r = self._responses.pop(0)
        assert r["kind"] == "structured", f"unexpected structured call; next was {r}"
        self.log.append(f"structured:{schema_name}")
        return r["content"]

    async def chat(self, *, messages, **kw) -> str:
        r = self._responses.pop(0)
        assert r["kind"] == "chat", f"unexpected chat call; next was {r}"
        self.log.append("chat")
        return r["content"]


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
    ]), update_number=1)
    yield sm
    conn.close()


def _cb(world):
    return ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())


async def test_clean_check_commits_directly(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["beat1"], "suggested_choices": ["x"]}'},
        {"kind": "chat", "content": "Prose v1."},
        {"kind": "structured", "content": '{"issues": []}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.prose == "Prose v1."
    assert out.trace.outcome == "committed"
    assert [s.stage_name for s in out.trace.stages] == ["plan", "write", "check"]


async def test_warning_triggers_revise_and_recheck(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["beat1"], "suggested_choices": []}'},
        {"kind": "chat", "content": "Prose v1 with purple prose."},
        {"kind": "structured", "content": '{"issues": [{"severity": "warning", "category": "prose_quality", "message": "purple prose", "suggested_fix": "simplify"}]}'},
        {"kind": "chat", "content": "Prose v2 simpler."},
        {"kind": "structured", "content": '{"issues": []}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.prose == "Prose v2 simpler."
    assert out.trace.outcome == "committed"
    stage_names = [s.stage_name for s in out.trace.stages]
    assert stage_names == ["plan", "write", "check", "revise", "check"]


async def test_critical_triggers_replan_once(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["bad beat"], "suggested_choices": []}'},
        {"kind": "chat", "content": "Prose v1."},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "magic banned"}]}'},
        # REPLAN
        {"kind": "structured", "content": '{"beats": ["better beat"], "suggested_choices": []}'},
        {"kind": "chat", "content": "Prose v2."},
        {"kind": "structured", "content": '{"issues": []}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.prose == "Prose v2."
    stage_names = [s.stage_name for s in out.trace.stages]
    assert stage_names == ["plan", "write", "check", "plan", "write", "check"]
    assert out.trace.outcome == "committed"


async def test_critical_after_replan_flags_qm(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["b1"], "suggested_choices": []}'},
        {"kind": "chat", "content": "v1"},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "bad"}]}'},
        # REPLAN
        {"kind": "structured", "content": '{"beats": ["b2"], "suggested_choices": []}'},
        {"kind": "chat", "content": "v2"},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "still bad"}]}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.trace.outcome == "flagged_qm"
    # Narrative is still written (v2), flagged
    assert world.list_narrative()[0].raw_text == "v2"
```

- [ ] **Step 2: Run — failures**

Run: `uv run pytest tests/engine/test_pipeline_branching.py -v`

- [ ] **Step 3: Modify `app/engine/pipeline.py`**

Replace the `Pipeline.run` method (and helper methods) with a refactored version. Add imports at top:

```python
from .check import CHECK_SCHEMA, CheckIssue, CheckOutput
```

Replace the body of `Pipeline.run(...)` with the following (keep the existing `__init__`, `_BEAT_KEYS`, `_normalize_beat_sheet`, etc.):

```python
    async def run(self, *, player_action: str, update_number: int) -> PipelineOutput:
        trace = PipelineTrace(trace_id=uuid.uuid4().hex, trigger=player_action)
        replan_attempts = 0
        recheck_done = False
        critical_feedback: list[CheckIssue] = []

        plan_parsed = await self._run_plan(trace, player_action, critical_feedback)
        prose = await self._run_write(trace, plan_parsed)
        check_out = await self._run_check(trace, plan_parsed, prose)

        # REPLAN branch: critical issues, replan once.
        if check_out.has_critical and replan_attempts < 1:
            replan_attempts += 1
            critical_feedback = list(check_out.issues)
            plan_parsed = await self._run_plan(trace, player_action, critical_feedback)
            prose = await self._run_write(trace, plan_parsed)
            check_out = await self._run_check(trace, plan_parsed, prose)

        # REVISE branch: fixable issues, revise + recheck once.
        if check_out.has_fixable and not check_out.has_critical and not recheck_done:
            prose = await self._run_revise(trace, plan_parsed, prose, check_out.issues)
            recheck_done = True
            check_out = await self._run_check(trace, plan_parsed, prose)

        # Determine outcome.
        if check_out.has_critical:
            outcome = "flagged_qm"
        else:
            outcome = "committed"

        self._world.write_narrative(NarrativeRecord(
            update_number=update_number,
            raw_text=prose,
            player_action=player_action,
            pipeline_trace_id=trace.trace_id,
        ))
        trace.outcome = outcome

        return PipelineOutput(
            prose=prose,
            choices=plan_parsed.get("suggested_choices", []),
            beats=plan_parsed["beats"],
            trace=trace,
        )

    async def _run_plan(
        self, trace: PipelineTrace, player_action: str,
        critical_feedback: list[CheckIssue],
    ) -> dict:
        extras = {"player_action": player_action}
        if critical_feedback:
            extras["critical_feedback"] = "\n".join(
                f"- [{i.severity}/{i.category}] {i.message}" for i in critical_feedback
            )
        plan_ctx = self._cb.build(
            spec=PLAN_SPEC,
            stage_name="plan",
            templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
            extras=extras,
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=plan_ctx.system_prompt),
                ChatMessage(role="user", content=plan_ctx.user_prompt),
            ],
            json_schema=BEAT_SHEET_SCHEMA,
            schema_name="BeatSheet",
            temperature=0.4,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        parsed = OutputParser.parse_json(raw)
        if not isinstance(parsed, dict):
            trace.add_stage(StageResult(
                stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
                errors=[StageError(kind="parse_error", message="not a dict")],
                latency_ms=latency,
            ))
            trace.outcome = "failed"
            raise ParseError(f"plan not a dict: {parsed!r}")
        normalized = _normalize_beat_sheet(parsed)
        if not normalized["beats"]:
            trace.add_stage(StageResult(
                stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
                errors=[StageError(kind="parse_error", message="no beats")],
                latency_ms=latency,
            ))
            trace.outcome = "failed"
            raise ParseError(f"plan has no beats: {raw!r}")
        trace.add_stage(StageResult(
            stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
            parsed_output=normalized,
            token_usage=TokenUsage(prompt=plan_ctx.token_estimate),
            latency_ms=latency,
        ))
        return normalized

    async def _run_write(self, trace: PipelineTrace, plan: dict) -> str:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        write_ctx = self._cb.build(
            spec=WRITE_SPEC,
            stage_name="write",
            templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
            extras={"plan": plan_text, "style": "", "anti_patterns": []},
        )
        t0 = time.perf_counter()
        raw = await self._client.chat(
            messages=[
                ChatMessage(role="system", content=write_ctx.system_prompt),
                ChatMessage(role="user", content=write_ctx.user_prompt),
            ],
            temperature=0.8,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        prose = OutputParser.parse_prose(raw)
        trace.add_stage(StageResult(
            stage_name="write", input_prompt=write_ctx.user_prompt, raw_output=raw,
            parsed_output=prose,
            token_usage=TokenUsage(prompt=write_ctx.token_estimate),
            latency_ms=latency,
        ))
        return prose

    async def _run_check(self, trace: PipelineTrace, plan: dict, prose: str) -> CheckOutput:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        ctx = self._cb.build(
            spec=CHECK_SPEC,
            stage_name="check",
            templates={"system": "stages/check/system.j2", "user": "stages/check/user.j2"},
            extras={"plan": plan_text, "prose": prose},
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            json_schema=CHECK_SCHEMA,
            schema_name="CheckOutput",
            temperature=0.2,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        try:
            parsed = OutputParser.parse_json(raw, schema=CheckOutput)
        except ParseError:
            # Defensive: if we can't parse, treat as clean so we don't block forever.
            parsed = CheckOutput(issues=[])
        trace.add_stage(StageResult(
            stage_name="check", input_prompt=ctx.user_prompt, raw_output=raw,
            parsed_output=parsed.model_dump(),
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))
        return parsed

    async def _run_revise(
        self, trace: PipelineTrace, plan: dict, prose: str,
        issues: list[CheckIssue],
    ) -> str:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        ctx = self._cb.build(
            spec=REVISE_SPEC,
            stage_name="revise",
            templates={"system": "stages/revise/system.j2", "user": "stages/revise/user.j2"},
            extras={
                "plan": plan_text,
                "prose": prose,
                "issues": [i.model_dump() for i in issues],
                "style": "",
                "anti_patterns": [],
            },
        )
        t0 = time.perf_counter()
        raw = await self._client.chat(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            temperature=0.6,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        revised = OutputParser.parse_prose(raw)
        trace.add_stage(StageResult(
            stage_name="revise", input_prompt=ctx.user_prompt, raw_output=raw,
            parsed_output=revised,
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))
        return revised
```

Add the missing imports near the top if not already present:

```python
from .context_spec import CHECK_SPEC, PLAN_SPEC, REVISE_SPEC, WRITE_SPEC
```

Also update the existing top-level imports — the old `ParseError` import from `app.world.output_parser` must still be there.

- [ ] **Step 4: Run branching tests — all 4 PASS**

Run: `uv run pytest tests/engine/test_pipeline_branching.py -v`

- [ ] **Step 5: Run full suite, nothing regressed**

Run: `uv run pytest -v`

- [ ] **Step 6: Commit**

```bash
git add app/engine/pipeline.py tests/engine/test_pipeline_branching.py
git commit -m "feat(engine): branching pipeline flow (check/revise/replan)"
```

---

## Task 4: TraceStore

**Files:**
- Create: `app/engine/trace_store.py`
- Create: `tests/engine/test_trace_store.py`

Design: JSON files, one per trace, at `<root>/<trace_id>.json`. `save(trace)` writes atomically (write to tmp, rename). `load(trace_id)` reads back into a `PipelineTrace`. `list_ids()` returns all known trace ids.

- [ ] **Step 1: Tests**

```python
# tests/engine/test_trace_store.py
from pathlib import Path
from app.engine.stages import StageResult
from app.engine.trace import PipelineTrace
from app.engine.trace_store import TraceStore


def test_save_and_load_roundtrip(tmp_path: Path):
    store = TraceStore(tmp_path)
    t = PipelineTrace(trace_id="abc", trigger="greet")
    t.add_stage(StageResult(stage_name="plan", input_prompt="p", raw_output="r", parsed_output={"beats": ["b"]}))
    t.outcome = "committed"
    store.save(t)
    loaded = store.load("abc")
    assert loaded.trace_id == "abc"
    assert loaded.outcome == "committed"
    assert loaded.stages[0].parsed_output == {"beats": ["b"]}


def test_list_ids(tmp_path: Path):
    store = TraceStore(tmp_path)
    for tid in ("a", "b", "c"):
        store.save(PipelineTrace(trace_id=tid, trigger="x"))
    assert sorted(store.list_ids()) == ["a", "b", "c"]


def test_load_missing_raises(tmp_path: Path):
    import pytest
    store = TraceStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("nope")
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `app/engine/trace_store.py`**

```python
from __future__ import annotations
import json
import os
from pathlib import Path
from .trace import PipelineTrace


class TraceStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, trace: PipelineTrace) -> Path:
        path = self._root / f"{trace.trace_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(trace.model_dump_json(indent=2))
        os.replace(tmp, path)
        return path

    def load(self, trace_id: str) -> PipelineTrace:
        path = self._root / f"{trace_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return PipelineTrace.model_validate_json(path.read_text())

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.json"))
```

- [ ] **Step 4: Run — pass**

- [ ] **Step 5: Commit**

```bash
git add app/engine/trace_store.py tests/engine/test_trace_store.py
git commit -m "feat(engine): JSON-backed TraceStore"
```

---

## Task 5: DiagnosticsManager — replay

**Files:**
- Create: `app/engine/diagnostics.py`
- Create: `tests/engine/test_diagnostics.py`

Design: `DiagnosticsManager(client, renderer, store)` with `replay(trace_id, stage_name, prompt_override=None)`. It loads the trace, finds the stage's `input_prompt`, substitutes with `prompt_override` if given, calls the client (`chat` or `chat_structured` based on stage), returns a new `StageResult`. Doesn't mutate anything on disk.

For v1 we keep it simple: PLAN/CHECK go through `chat_structured` with their respective schemas; WRITE/REVISE go through `chat`. System prompts are not preserved across replays in v1 (we replay with a single user message); that's a known limitation worth calling out.

- [ ] **Step 1: Tests**

```python
# tests/engine/test_diagnostics.py
from pathlib import Path
import pytest
from app.engine.diagnostics import DiagnosticsManager
from app.engine.stages import StageResult
from app.engine.trace import PipelineTrace
from app.engine.trace_store import TraceStore


class RecClient:
    def __init__(self, structured="{}", chat="text"):
        self.structured_response = structured
        self.chat_response = chat
        self.last = None

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        self.last = ("structured", messages, schema_name)
        return self.structured_response

    async def chat(self, *, messages, **kw):
        self.last = ("chat", messages)
        return self.chat_response


def _seed_trace(store: TraceStore) -> str:
    t = PipelineTrace(trace_id="t1", trigger="greet")
    t.add_stage(StageResult(stage_name="plan", input_prompt="orig plan prompt",
                             raw_output="{}", parsed_output={"beats": []}))
    t.add_stage(StageResult(stage_name="write", input_prompt="orig write prompt",
                             raw_output="orig prose", parsed_output="orig prose"))
    store.save(t)
    return "t1"


async def test_replay_plan_stage(tmp_path: Path):
    store = TraceStore(tmp_path)
    tid = _seed_trace(store)
    client = RecClient(structured='{"beats":["new"],"suggested_choices":[]}')
    dm = DiagnosticsManager(client=client, store=store)
    r = await dm.replay(tid, "plan")
    assert r.stage_name == "plan"
    assert client.last[0] == "structured"
    # original prompt was reused
    assert any("orig plan prompt" in m.content for m in client.last[1])
    assert r.raw_output == '{"beats":["new"],"suggested_choices":[]}'


async def test_replay_write_uses_override(tmp_path: Path):
    store = TraceStore(tmp_path)
    tid = _seed_trace(store)
    client = RecClient(chat="new prose")
    dm = DiagnosticsManager(client=client, store=store)
    r = await dm.replay(tid, "write", prompt_override="CHANGED PROMPT")
    assert client.last[0] == "chat"
    assert any(m.content == "CHANGED PROMPT" for m in client.last[1])
    assert r.parsed_output == "new prose"


async def test_replay_missing_stage_raises(tmp_path: Path):
    store = TraceStore(tmp_path)
    tid = _seed_trace(store)
    dm = DiagnosticsManager(client=RecClient(), store=store)
    with pytest.raises(ValueError):
        await dm.replay(tid, "nonexistent")
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `app/engine/diagnostics.py`**

```python
from __future__ import annotations
import time
from typing import Protocol
from app.runtime.client import ChatMessage
from app.world.output_parser import OutputParser
from .check import CHECK_SCHEMA, CheckOutput
from .pipeline import BEAT_SHEET_SCHEMA, _normalize_beat_sheet
from .stages import StageResult
from .trace_store import TraceStore


class _ClientLike(Protocol):
    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str: ...
    async def chat(self, *, messages, **kw) -> str: ...


_STRUCTURED_STAGES = {
    "plan": (BEAT_SHEET_SCHEMA, "BeatSheet"),
    "check": (CHECK_SCHEMA, "CheckOutput"),
}
_FREE_TEXT_STAGES = {"write", "revise"}


class DiagnosticsManager:
    def __init__(self, *, client: _ClientLike, store: TraceStore) -> None:
        self._client = client
        self._store = store

    async def replay(
        self, trace_id: str, stage_name: str, prompt_override: str | None = None,
    ) -> StageResult:
        trace = self._store.load(trace_id)
        target = next((s for s in trace.stages if s.stage_name == stage_name), None)
        if target is None:
            raise ValueError(f"stage {stage_name!r} not in trace {trace_id}")

        prompt = prompt_override if prompt_override is not None else target.input_prompt
        messages = [ChatMessage(role="user", content=prompt)]

        t0 = time.perf_counter()
        if stage_name in _STRUCTURED_STAGES:
            schema, name = _STRUCTURED_STAGES[stage_name]
            raw = await self._client.chat_structured(
                messages=messages, json_schema=schema, schema_name=name, temperature=0.3,
            )
            if stage_name == "plan":
                parsed = _normalize_beat_sheet(OutputParser.parse_json(raw) or {})
            else:  # check
                try:
                    parsed = OutputParser.parse_json(raw, schema=CheckOutput).model_dump()
                except Exception:
                    parsed = {"issues": []}
        elif stage_name in _FREE_TEXT_STAGES:
            raw = await self._client.chat(messages=messages, temperature=0.7)
            parsed = OutputParser.parse_prose(raw)
        else:
            raise ValueError(f"unknown stage: {stage_name!r}")
        latency = int((time.perf_counter() - t0) * 1000)

        return StageResult(
            stage_name=stage_name,
            input_prompt=prompt,
            raw_output=raw,
            parsed_output=parsed,
            latency_ms=latency,
        )
```

- [ ] **Step 4: Run — pass**

Run: `uv run pytest tests/engine/test_diagnostics.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/engine/diagnostics.py tests/engine/test_diagnostics.py
git commit -m "feat(engine): DiagnosticsManager with single-stage replay"
```

---

## Task 6: CLI wires in TraceStore + public surface

**Files:**
- Modify: `app/cli/play.py`
- Modify: `app/engine/__init__.py`

- [ ] **Step 1: Add `--traces` option to CLI and save trace after each chapter**

Edit `app/cli/play.py` — add import:

```python
from app.engine import TraceStore
```

Then modify the `play` command to accept `--traces` and save after each run. Replace the body of `play` with:

```python
@app.command()
def play(
    db: Path = typer.Option(..., help="Path to the quest DB."),
    server: str = typer.Option("http://127.0.0.1:8090", help="llama-server base URL."),
    traces: Path = typer.Option(Path("data/traces"), help="Directory for pipeline trace JSON files."),
) -> None:
    """Play the quest — reads player actions from stdin, prints prose."""
    sm = _open_world(db)
    client = InferenceClient(base_url=server, retries=1)
    cb = ContextBuilder(sm, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(sm, cb, client)
    trace_store = TraceStore(traces)

    typer.echo("Quest started. Type an action and press enter. Ctrl-D to quit.")
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
        trace_store.save(out.trace)
        typer.echo(out.prose)
        typer.echo(f"\n[trace: {out.trace.trace_id}  outcome: {out.trace.outcome}]", err=True)
        typer.echo("\nChoices:")
        for i, c in enumerate(out.choices, 1):
            typer.echo(f"  {i}. {c}")
        typer.echo()
        update_number += 1
```

- [ ] **Step 2: Update `app/engine/__init__.py` to export new symbols**

Add to imports:

```python
from .check import CHECK_SCHEMA, CheckIssue, CheckOutput
from .context_spec import CHECK_SPEC, REVISE_SPEC
from .diagnostics import DiagnosticsManager
from .trace_store import TraceStore
```

Add to `__all__`: `"CHECK_SCHEMA"`, `"CHECK_SPEC"`, `"CheckIssue"`, `"CheckOutput"`, `"DiagnosticsManager"`, `"REVISE_SPEC"`, `"TraceStore"`.

- [ ] **Step 3: Verify imports resolve**

Run: `uv run python -c "from app.engine import DiagnosticsManager, TraceStore, CheckOutput; print('ok')"`

- [ ] **Step 4: Run full suite**

Run: `uv run pytest -v`

- [ ] **Step 5: Commit**

```bash
git add app/cli/play.py app/engine/__init__.py
git commit -m "feat(cli,engine): persist traces, expose P4 public api"
```

---

## Done criteria

- Pipeline runs CHECK after WRITE; commits clean, revises fixable, replans once on critical, flags QM on repeat critical.
- Each chapter writes a trace JSON to the configured directory.
- `DiagnosticsManager.replay(trace_id, stage)` re-runs one stage against the stored prompt.
- Full pytest suite green.
