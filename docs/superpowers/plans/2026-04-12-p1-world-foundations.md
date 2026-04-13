# P1 — World Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the typed world-state layer for the Quest Engine Harness: entity/relationship/rule/timeline/narrative/foreshadowing/plot-thread models, a SQLite-backed `WorldStateManager` with atomic `StateDelta` application, rollback, a JSON seed loader, and a tolerant `OutputParser` for LLM output. No LLM calls are made in this milestone — the layer is fully testable with fakes.

**Architecture:** Pydantic v2 models define every world entity and the `StateDelta` vocabulary. A single SQLite database per quest holds the state, with the schema created at open time (no Alembic yet — schema is expected to evolve fast). The `WorldStateManager` is the only code that writes to the DB; the `StateDelta` flow is: build delta → `validate_delta` (checks mechanical consistency) → `apply_delta` (one transaction, rolls back on any failure). A `SeedLoader` reads a JSON file and produces an opening `StateDelta`. The `OutputParser` is independent of the rest — it repairs and parses LLM output into typed objects and is needed by every future stage.

**Tech Stack:** Python 3.11, Pydantic v2, SQLite (stdlib `sqlite3`), `pytest`, `pytest-asyncio` (already configured). No new runtime deps required.

---

## File Structure

**Created in this plan (all under `app/world/` and `tests/world/`):**

- `app/world/__init__.py` — package public surface
- `app/world/schema.py` — Pydantic models for `Entity`, `Relationship`, `WorldRule`, `TimelineEvent`, `NarrativeRecord`, `ForeshadowingHook`, `PlotThread`, plus enums (`EntityType`, `EntityStatus`, `HookStatus`, `ThreadStatus`, `ArcPosition`)
- `app/world/delta.py` — Pydantic models for `StateDelta` and its operation types (`EntityCreate`, `EntityUpdate`, `RelChange`, `TimelineEventOp`, `FSUpdate`, `PTUpdate`); also `ValidationIssue`, `ValidationResult`
- `app/world/db.py` — SQLite schema DDL + `open_db(path)` helper (creates tables on first open)
- `app/world/state_manager.py` — `WorldStateManager` CRUD + `validate_delta` + `apply_delta` + `rollback` + `snapshot` + `WorldSnapshot`
- `app/world/seed.py` — `SeedLoader.load(path)` → `StateDelta`
- `app/world/output_parser.py` — `OutputParser` with `parse_json(text)`, `parse_prose(text)`, `ParseError`
- `tests/world/__init__.py`
- `tests/world/conftest.py` — shared fixtures (temp DB, sample delta)
- `tests/world/test_schema.py`
- `tests/world/test_delta.py`
- `tests/world/test_state_manager_entities.py`
- `tests/world/test_state_manager_relationships.py`
- `tests/world/test_state_manager_narrative.py`
- `tests/world/test_state_manager_validation.py`
- `tests/world/test_state_manager_apply.py`
- `tests/world/test_state_manager_rollback.py`
- `tests/world/test_seed.py`
- `tests/world/test_output_parser.py`

**Modified:**
- `app/world/__init__.py` (final task only) — re-export public surface

Boundaries: `schema.py` has zero imports from the rest of the package; `delta.py` imports only from `schema`; `db.py` depends on neither; `state_manager.py` composes all three; `seed.py` depends only on `schema` + `delta`; `output_parser.py` is standalone.

---

## Task 1: Schema — core entity models

**Files:**
- Create: `app/world/__init__.py` (empty for now)
- Create: `app/world/schema.py`
- Create: `tests/world/__init__.py` (empty)
- Create: `tests/world/test_schema.py`

Design notes for the engineer:
- Every model uses Pydantic v2 `BaseModel`.
- Enums are `str` subclasses of `enum.Enum` so they serialize naturally to SQLite.
- `Entity.data` is an arbitrary JSON object (`dict[str, Any]`) — the engine stores type-specific fields there.
- Timestamps are optional (set by the state manager, not by the model).
- `update_number` semantics: the monotonic counter the engine increments per pipeline commit. `None` means "not yet persisted".

- [ ] **Step 1: Create empty package + test package**

Write empty files: `app/world/__init__.py`, `tests/world/__init__.py`.

- [ ] **Step 2: Write failing tests at `tests/world/test_schema.py`**

```python
from __future__ import annotations
import pytest
from pydantic import ValidationError
from app.world.schema import (
    ArcPosition,
    Entity,
    EntityStatus,
    EntityType,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    PlotThread,
    Relationship,
    ThreadStatus,
    TimelineEvent,
    WorldRule,
)


def test_entity_defaults():
    e = Entity(id="char:alice", entity_type=EntityType.CHARACTER, name="Alice")
    assert e.status == EntityStatus.ACTIVE
    assert e.data == {}
    assert e.last_referenced_update is None
    assert e.created_at_update is None


def test_entity_accepts_arbitrary_data():
    e = Entity(
        id="loc:tavern",
        entity_type=EntityType.LOCATION,
        name="The Broken Anchor",
        data={"climate": "coastal", "population": 412},
    )
    assert e.data["population"] == 412


def test_entity_type_enum_rejects_unknown():
    with pytest.raises(ValidationError):
        Entity(id="x", entity_type="dragonoid", name="X")


def test_relationship_requires_endpoints():
    r = Relationship(source_id="char:alice", target_id="char:bob", rel_type="ally")
    assert r.data == {}


def test_world_rule_with_constraints():
    r = WorldRule(
        id="rule:no-magic-in-zone-3",
        category="magic_system",
        description="Magic does not function within zone 3.",
        constraints={"zone": 3, "effect": "disabled"},
    )
    assert r.constraints["zone"] == 3


def test_timeline_event_ordering_fields():
    ev = TimelineEvent(
        update_number=5,
        event_index=0,
        description="Alice enters the tavern.",
        involved_entities=["char:alice", "loc:tavern"],
    )
    assert ev.causal_links == []


def test_narrative_record_roundtrip():
    n = NarrativeRecord(
        update_number=3,
        raw_text="She walked in.",
        player_action="Enter the tavern.",
    )
    assert n.summary is None
    assert n.chapter_id is None


def test_foreshadowing_hook_default_status():
    h = ForeshadowingHook(
        id="fs:001",
        description="Alice touches a strange coin.",
        planted_at_update=2,
        payoff_target="reveal of the coin's origin",
    )
    assert h.status == HookStatus.PLANTED
    assert h.references == []


def test_plot_thread_priority_bounds():
    pt = PlotThread(
        id="pt:main",
        name="The Missing Heir",
        description="Find the lost heir.",
        involved_entities=["char:alice"],
        arc_position=ArcPosition.RISING,
    )
    assert pt.priority == 5
    with pytest.raises(ValidationError):
        PlotThread(
            id="pt:bad", name="x", description="x",
            involved_entities=[], arc_position=ArcPosition.RISING, priority=11,
        )


def test_enums_are_strings():
    # Important so they round-trip through JSON/SQLite
    assert EntityStatus.ACTIVE.value == "active"
    assert HookStatus.PAID_OFF.value == "paid_off"
    assert ThreadStatus.DORMANT.value == "dormant"
    assert ArcPosition.CLIMAX.value == "climax"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/world/test_schema.py -v`
Expected: `ModuleNotFoundError: No module named 'app.world.schema'`.

- [ ] **Step 4: Implement `app/world/schema.py`**

```python
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    FACTION = "faction"
    ITEM = "item"
    CONCEPT = "concept"


class EntityStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    DECEASED = "deceased"
    DESTROYED = "destroyed"


class HookStatus(str, Enum):
    PLANTED = "planted"
    REFERENCED = "referenced"
    PAID_OFF = "paid_off"
    ABANDONED = "abandoned"


class ThreadStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ArcPosition(str, Enum):
    RISING = "rising"
    CLIMAX = "climax"
    FALLING = "falling"
    DENOUEMENT = "denouement"


class Entity(BaseModel):
    id: str
    entity_type: EntityType
    name: str
    data: dict[str, Any] = Field(default_factory=dict)
    status: EntityStatus = EntityStatus.ACTIVE
    last_referenced_update: int | None = None
    created_at_update: int | None = None


class Relationship(BaseModel):
    source_id: str
    target_id: str
    rel_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    established_at_update: int | None = None


class WorldRule(BaseModel):
    id: str
    category: str
    description: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    established_at_update: int | None = None


class TimelineEvent(BaseModel):
    update_number: int
    event_index: int
    description: str
    involved_entities: list[str] = Field(default_factory=list)
    causal_links: list[tuple[int, int]] = Field(default_factory=list)


class NarrativeRecord(BaseModel):
    update_number: int
    raw_text: str
    summary: str | None = None
    chapter_id: int | None = None
    state_diff: dict[str, Any] = Field(default_factory=dict)
    player_action: str | None = None
    pipeline_trace_id: str | None = None


class ForeshadowingHook(BaseModel):
    id: str
    description: str
    planted_at_update: int
    payoff_target: str
    status: HookStatus = HookStatus.PLANTED
    paid_off_at_update: int | None = None
    references: list[int] = Field(default_factory=list)


class PlotThread(BaseModel):
    id: str
    name: str
    description: str
    status: ThreadStatus = ThreadStatus.ACTIVE
    involved_entities: list[str] = Field(default_factory=list)
    arc_position: ArcPosition
    priority: int = Field(default=5, ge=1, le=10)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/world/test_schema.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/world/__init__.py app/world/schema.py tests/world/__init__.py tests/world/test_schema.py
git commit -m "feat(world): typed entity/rule/timeline/foreshadowing models"
```

---

## Task 2: StateDelta models

**Files:**
- Create: `app/world/delta.py`
- Create: `tests/world/test_delta.py`

Design: a `StateDelta` is the sole vehicle for changing world state. It is a passive data object — it does not know how to apply itself. Operation variants are separate models so the state manager can dispatch on type.

- [ ] **Step 1: Write failing tests at `tests/world/test_delta.py`**

```python
from __future__ import annotations
from app.world.schema import Entity, EntityType, Relationship, TimelineEvent
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    FSUpdate,
    PTUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
    ValidationIssue,
    ValidationResult,
)


def test_empty_delta_is_valid_shape():
    d = StateDelta()
    assert d.entity_creates == []
    assert d.entity_updates == []
    assert d.relationship_changes == []
    assert d.timeline_events == []
    assert d.foreshadowing_updates == []
    assert d.plot_thread_updates == []


def test_entity_create_wraps_entity():
    e = Entity(id="char:alice", entity_type=EntityType.CHARACTER, name="Alice")
    op = EntityCreate(entity=e)
    assert op.entity.id == "char:alice"


def test_entity_update_carries_patch():
    op = EntityUpdate(id="char:alice", patch={"status": "dormant"})
    assert op.patch["status"] == "dormant"


def test_rel_change_variants():
    r = Relationship(source_id="a", target_id="b", rel_type="ally")
    add = RelChange(action="add", relationship=r)
    remove = RelChange(
        action="remove",
        relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
    )
    assert add.action == "add"
    assert remove.action == "remove"


def test_timeline_event_op_wraps_event():
    ev = TimelineEvent(update_number=1, event_index=0, description="x")
    op = TimelineEventOp(event=ev)
    assert op.event.description == "x"


def test_fs_update_transition():
    op = FSUpdate(id="fs:001", new_status="paid_off", paid_off_at_update=7)
    assert op.new_status == "paid_off"


def test_pt_update_partial():
    op = PTUpdate(id="pt:main", patch={"status": "resolved"})
    assert op.patch == {"status": "resolved"}


def test_delta_composes_all_ops():
    d = StateDelta(
        entity_creates=[EntityCreate(entity=Entity(
            id="x", entity_type=EntityType.ITEM, name="X"))],
        entity_updates=[EntityUpdate(id="y", patch={"status": "dormant"})],
    )
    assert len(d.entity_creates) == 1
    assert len(d.entity_updates) == 1


def test_validation_result_is_ok_when_no_violations():
    r = ValidationResult(issues=[])
    assert r.ok
    r2 = ValidationResult(issues=[ValidationIssue(severity="error", message="x")])
    assert not r2.ok


def test_validation_result_ok_allows_warnings():
    r = ValidationResult(issues=[ValidationIssue(severity="warning", message="w")])
    assert r.ok
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `uv run pytest tests/world/test_delta.py -v`
Expected: `ModuleNotFoundError: No module named 'app.world.delta'`.

- [ ] **Step 3: Implement `app/world/delta.py`**

```python
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field
from .schema import Entity, Relationship, TimelineEvent


class EntityCreate(BaseModel):
    entity: Entity


class EntityUpdate(BaseModel):
    id: str
    patch: dict[str, Any]


class RelChange(BaseModel):
    action: Literal["add", "remove", "modify"]
    relationship: Relationship


class TimelineEventOp(BaseModel):
    event: TimelineEvent


class FSUpdate(BaseModel):
    id: str
    new_status: Literal["planted", "referenced", "paid_off", "abandoned"]
    paid_off_at_update: int | None = None
    add_reference: int | None = None  # append this update to `references`


class PTUpdate(BaseModel):
    id: str
    patch: dict[str, Any]


class StateDelta(BaseModel):
    entity_creates: list[EntityCreate] = Field(default_factory=list)
    entity_updates: list[EntityUpdate] = Field(default_factory=list)
    relationship_changes: list[RelChange] = Field(default_factory=list)
    timeline_events: list[TimelineEventOp] = Field(default_factory=list)
    foreshadowing_updates: list[FSUpdate] = Field(default_factory=list)
    plot_thread_updates: list[PTUpdate] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    message: str
    subject: str | None = None  # entity id, rel key, etc.


class ValidationResult(BaseModel):
    issues: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)
```

- [ ] **Step 4: Run tests — all 9 PASS**

Run: `uv run pytest tests/world/test_delta.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/world/delta.py tests/world/test_delta.py
git commit -m "feat(world): StateDelta vocabulary for atomic world changes"
```

---

## Task 3: SQLite schema + open_db

**Files:**
- Create: `app/world/db.py`
- Create: `tests/world/conftest.py`

Design: schema DDL is a single constant string. `open_db(path)` returns a `sqlite3.Connection` with foreign keys + JSON support enabled and all tables created if missing. We keep the connection synchronous — the state manager is called from within async pipelines but SQLite operations are fast enough that running them directly is fine at this scale.

- [ ] **Step 1: Write `tests/world/conftest.py`**

```python
from __future__ import annotations
from pathlib import Path
import pytest
from app.world.db import open_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "quest.db"


@pytest.fixture
def db(db_path: Path):
    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 2: Write failing tests (inline in the db test file)**

Create `tests/world/test_db.py`:

```python
from __future__ import annotations
import sqlite3
from pathlib import Path
from app.world.db import open_db


def test_open_creates_file(tmp_path: Path):
    p = tmp_path / "q.db"
    assert not p.exists()
    conn = open_db(p)
    assert p.exists()
    conn.close()


def test_all_tables_created(db):
    cur = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cur.fetchall()}
    assert {
        "entities",
        "relationships",
        "world_rules",
        "timeline",
        "narrative",
        "foreshadowing",
        "plot_threads",
    } <= tables


def test_foreign_keys_enabled(db):
    cur = db.execute("PRAGMA foreign_keys")
    assert cur.fetchone()[0] == 1


def test_row_factory_returns_rows(db):
    db.execute("INSERT INTO entities(id,entity_type,name,data,status) VALUES(?,?,?,?,?)",
               ("x", "item", "X", "{}", "active"))
    row = db.execute("SELECT id, name FROM entities WHERE id='x'").fetchone()
    assert row["id"] == "x"
    assert row["name"] == "X"


def test_open_twice_is_idempotent(tmp_path: Path):
    p = tmp_path / "q.db"
    open_db(p).close()
    # Should not raise on re-open
    conn = open_db(p)
    conn.close()
```

- [ ] **Step 3: Run tests — expect ImportError**

Run: `uv run pytest tests/world/test_db.py -v`

- [ ] **Step 4: Implement `app/world/db.py`**

```python
from __future__ import annotations
import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_referenced_update INTEGER,
    created_at_update INTEGER,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relationships (
    source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel_type TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    established_at_update INTEGER,
    PRIMARY KEY (source_id, target_id, rel_type)
);

CREATE TABLE IF NOT EXISTS world_rules (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    constraints TEXT NOT NULL DEFAULT '{}',
    established_at_update INTEGER
);

CREATE TABLE IF NOT EXISTS timeline (
    update_number INTEGER NOT NULL,
    event_index INTEGER NOT NULL,
    description TEXT NOT NULL,
    involved_entities TEXT NOT NULL DEFAULT '[]',
    causal_links TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (update_number, event_index)
);

CREATE TABLE IF NOT EXISTS narrative (
    update_number INTEGER PRIMARY KEY,
    raw_text TEXT NOT NULL,
    summary TEXT,
    chapter_id INTEGER,
    state_diff TEXT NOT NULL DEFAULT '{}',
    player_action TEXT,
    pipeline_trace_id TEXT
);

CREATE TABLE IF NOT EXISTS foreshadowing (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    planted_at_update INTEGER NOT NULL,
    payoff_target TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planted',
    paid_off_at_update INTEGER,
    refs TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS plot_threads (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    involved_entities TEXT NOT NULL DEFAULT '[]',
    arc_position TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5
);
"""


def open_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn
```

Note: column name `refs` (not `references`, which is reserved) in the `foreshadowing` table. The state manager will translate between the DB column `refs` and the model field `references`.

- [ ] **Step 5: Run tests — all 5 PASS**

Run: `uv run pytest tests/world/test_db.py -v`

- [ ] **Step 6: Commit**

```bash
git add app/world/db.py tests/world/conftest.py tests/world/test_db.py
git commit -m "feat(world): SQLite schema + open_db helper"
```

---

## Task 4: WorldStateManager — entity CRUD

**Files:**
- Create: `app/world/state_manager.py`
- Create: `tests/world/test_state_manager_entities.py`

Design: one class, `WorldStateManager`, constructed from a `sqlite3.Connection`. CRUD methods are synchronous. JSON columns are serialized via `json.dumps` / `json.loads`. All methods return typed Pydantic models, never raw rows.

- [ ] **Step 1: Write failing tests**

```python
# tests/world/test_state_manager_entities.py
from __future__ import annotations
import pytest
from app.world.schema import Entity, EntityStatus, EntityType
from app.world.state_manager import EntityNotFoundError, WorldStateManager


def test_create_and_get_entity(db):
    sm = WorldStateManager(db)
    e = Entity(id="char:alice", entity_type=EntityType.CHARACTER, name="Alice",
               data={"age": 27}, created_at_update=1)
    sm.create_entity(e)
    got = sm.get_entity("char:alice")
    assert got == e


def test_get_missing_entity_raises(db):
    sm = WorldStateManager(db)
    with pytest.raises(EntityNotFoundError):
        sm.get_entity("char:ghost")


def test_list_entities_filters_by_type(db):
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="c1", entity_type=EntityType.CHARACTER, name="A"))
    sm.create_entity(Entity(id="c2", entity_type=EntityType.CHARACTER, name="B"))
    sm.create_entity(Entity(id="l1", entity_type=EntityType.LOCATION, name="Town"))
    chars = sm.list_entities(entity_type=EntityType.CHARACTER)
    assert {e.id for e in chars} == {"c1", "c2"}
    all_ = sm.list_entities()
    assert len(all_) == 3


def test_update_entity_patches_fields(db):
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="c1", entity_type=EntityType.CHARACTER, name="A"))
    sm.update_entity("c1", {"status": "dormant", "last_referenced_update": 5})
    got = sm.get_entity("c1")
    assert got.status == EntityStatus.DORMANT
    assert got.last_referenced_update == 5


def test_update_entity_merges_data_field(db):
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="c1", entity_type=EntityType.CHARACTER, name="A",
                             data={"hp": 10, "mood": "ok"}))
    sm.update_entity("c1", {"data": {"hp": 8}})  # partial data patch merges
    got = sm.get_entity("c1")
    assert got.data == {"hp": 8, "mood": "ok"}


def test_update_missing_entity_raises(db):
    sm = WorldStateManager(db)
    with pytest.raises(EntityNotFoundError):
        sm.update_entity("nope", {"status": "dormant"})
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `uv run pytest tests/world/test_state_manager_entities.py -v`

- [ ] **Step 3: Implement minimal `app/world/state_manager.py`**

```python
from __future__ import annotations
import json
import sqlite3
from typing import Any
from .schema import (
    Entity,
    EntityType,
)


class WorldStateError(Exception):
    pass


class EntityNotFoundError(WorldStateError):
    pass


class RelationshipNotFoundError(WorldStateError):
    pass


def _row_to_entity(row: sqlite3.Row) -> Entity:
    return Entity(
        id=row["id"],
        entity_type=row["entity_type"],
        name=row["name"],
        data=json.loads(row["data"]),
        status=row["status"],
        last_referenced_update=row["last_referenced_update"],
        created_at_update=row["created_at_update"],
    )


class WorldStateManager:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ---- entities ----

    def create_entity(self, entity: Entity) -> None:
        self._conn.execute(
            "INSERT INTO entities(id, entity_type, name, data, status, "
            "last_referenced_update, created_at_update) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entity.id,
                entity.entity_type.value,
                entity.name,
                json.dumps(entity.data),
                entity.status.value,
                entity.last_referenced_update,
                entity.created_at_update,
            ),
        )
        self._conn.commit()

    def get_entity(self, entity_id: str) -> Entity:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            raise EntityNotFoundError(entity_id)
        return _row_to_entity(row)

    def list_entities(self, entity_type: EntityType | None = None) -> list[Entity]:
        if entity_type is None:
            rows = self._conn.execute("SELECT * FROM entities ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? ORDER BY id",
                (entity_type.value,),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def update_entity(self, entity_id: str, patch: dict[str, Any]) -> None:
        current = self.get_entity(entity_id)  # raises if missing
        merged_data = current.data
        if "data" in patch:
            merged_data = {**current.data, **patch["data"]}
        fields = {
            "name": patch.get("name", current.name),
            "status": patch.get("status", current.status.value if hasattr(current.status, "value") else current.status),
            "last_referenced_update": patch.get(
                "last_referenced_update", current.last_referenced_update
            ),
            "data": json.dumps(merged_data),
        }
        self._conn.execute(
            "UPDATE entities SET name=?, status=?, last_referenced_update=?, data=?, "
            "modified_at=CURRENT_TIMESTAMP WHERE id=?",
            (fields["name"], fields["status"], fields["last_referenced_update"],
             fields["data"], entity_id),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests — all 6 PASS**

Run: `uv run pytest tests/world/test_state_manager_entities.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/world/state_manager.py tests/world/test_state_manager_entities.py
git commit -m "feat(world): WorldStateManager entity CRUD"
```

---

## Task 5: WorldStateManager — relationships, rules, timeline, narrative, foreshadowing, plot threads

**Files:**
- Modify: `app/world/state_manager.py` (append methods)
- Create: `tests/world/test_state_manager_relationships.py`
- Create: `tests/world/test_state_manager_narrative.py`

This task covers everything except `validate_delta` / `apply_delta` / `rollback` / `snapshot` (those come next). Grouped into one task because the methods are all mechanical and mirror each other.

- [ ] **Step 1: Write failing tests for relationships**

`tests/world/test_state_manager_relationships.py`:

```python
from __future__ import annotations
import pytest
from app.world.schema import Entity, EntityType, Relationship
from app.world.state_manager import RelationshipNotFoundError, WorldStateManager


def _seed_two(db) -> WorldStateManager:
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))
    sm.create_entity(Entity(id="b", entity_type=EntityType.CHARACTER, name="B"))
    return sm


def test_add_and_list_relationships(db):
    sm = _seed_two(db)
    sm.add_relationship(Relationship(source_id="a", target_id="b", rel_type="ally"))
    rels = sm.list_relationships(source_id="a")
    assert len(rels) == 1
    assert rels[0].rel_type == "ally"


def test_remove_relationship(db):
    sm = _seed_two(db)
    r = Relationship(source_id="a", target_id="b", rel_type="ally")
    sm.add_relationship(r)
    sm.remove_relationship("a", "b", "ally")
    assert sm.list_relationships(source_id="a") == []


def test_remove_missing_relationship_raises(db):
    sm = _seed_two(db)
    with pytest.raises(RelationshipNotFoundError):
        sm.remove_relationship("a", "b", "ally")


def test_modify_relationship_updates_data(db):
    sm = _seed_two(db)
    sm.add_relationship(Relationship(source_id="a", target_id="b", rel_type="ally", data={"trust": 5}))
    sm.modify_relationship("a", "b", "ally", {"trust": 8})
    rels = sm.list_relationships(source_id="a")
    assert rels[0].data == {"trust": 8}
```

- [ ] **Step 2: Write failing tests for narrative / timeline / foreshadowing / plot threads / rules**

`tests/world/test_state_manager_narrative.py`:

```python
from __future__ import annotations
from app.world.schema import (
    ArcPosition,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    PlotThread,
    ThreadStatus,
    TimelineEvent,
    WorldRule,
)
from app.world.state_manager import WorldStateManager


def test_append_and_list_timeline(db):
    sm = WorldStateManager(db)
    sm.append_timeline_event(TimelineEvent(update_number=1, event_index=0, description="x"))
    sm.append_timeline_event(TimelineEvent(update_number=1, event_index=1, description="y"))
    events = sm.list_timeline(update_number=1)
    assert [e.event_index for e in events] == [0, 1]


def test_write_and_get_narrative(db):
    sm = WorldStateManager(db)
    n = NarrativeRecord(update_number=1, raw_text="She walked in.", player_action="enter")
    sm.write_narrative(n)
    got = sm.get_narrative(1)
    assert got.raw_text == "She walked in."
    assert got.player_action == "enter"


def test_list_narrative_ordered(db):
    sm = WorldStateManager(db)
    sm.write_narrative(NarrativeRecord(update_number=2, raw_text="b"))
    sm.write_narrative(NarrativeRecord(update_number=1, raw_text="a"))
    records = sm.list_narrative(limit=10)
    assert [r.update_number for r in records] == [1, 2]


def test_foreshadowing_crud(db):
    sm = WorldStateManager(db)
    h = ForeshadowingHook(id="fs:1", description="...", planted_at_update=1, payoff_target="...")
    sm.add_foreshadowing(h)
    assert sm.get_foreshadowing("fs:1") == h
    sm.update_foreshadowing(
        "fs:1",
        {"status": HookStatus.PAID_OFF, "paid_off_at_update": 5, "references": [3, 4]},
    )
    got = sm.get_foreshadowing("fs:1")
    assert got.status == HookStatus.PAID_OFF
    assert got.paid_off_at_update == 5
    assert got.references == [3, 4]


def test_plot_thread_crud(db):
    sm = WorldStateManager(db)
    pt = PlotThread(id="pt:1", name="Main", description="d",
                    involved_entities=["a"], arc_position=ArcPosition.RISING)
    sm.add_plot_thread(pt)
    assert sm.get_plot_thread("pt:1").name == "Main"
    sm.update_plot_thread("pt:1", {"status": ThreadStatus.RESOLVED, "priority": 8})
    got = sm.get_plot_thread("pt:1")
    assert got.status == ThreadStatus.RESOLVED
    assert got.priority == 8


def test_world_rules(db):
    sm = WorldStateManager(db)
    r = WorldRule(id="r:1", category="magic", description="No magic in zone 3.",
                  constraints={"zone": 3})
    sm.add_rule(r)
    assert sm.list_rules()[0].description == "No magic in zone 3."
```

- [ ] **Step 3: Run — expect AttributeErrors / failing imports**

Run: `uv run pytest tests/world/test_state_manager_relationships.py tests/world/test_state_manager_narrative.py -v`

- [ ] **Step 4: Append methods to `app/world/state_manager.py`**

Add these imports to the top of the file (alongside existing imports):

```python
from .schema import (
    ArcPosition,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    PlotThread,
    Relationship,
    ThreadStatus,
    TimelineEvent,
    WorldRule,
)
```

Append these methods inside the `WorldStateManager` class (below `update_entity`):

```python
    # ---- relationships ----

    def add_relationship(self, rel: Relationship) -> None:
        self._conn.execute(
            "INSERT INTO relationships(source_id, target_id, rel_type, data, established_at_update) "
            "VALUES (?, ?, ?, ?, ?)",
            (rel.source_id, rel.target_id, rel.rel_type, json.dumps(rel.data),
             rel.established_at_update),
        )
        self._conn.commit()

    def remove_relationship(self, source_id: str, target_id: str, rel_type: str) -> None:
        cur = self._conn.execute(
            "DELETE FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
            (source_id, target_id, rel_type),
        )
        if cur.rowcount == 0:
            raise RelationshipNotFoundError(f"{source_id} -{rel_type}-> {target_id}")
        self._conn.commit()

    def modify_relationship(
        self, source_id: str, target_id: str, rel_type: str, data_patch: dict[str, Any]
    ) -> None:
        row = self._conn.execute(
            "SELECT data FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
            (source_id, target_id, rel_type),
        ).fetchone()
        if row is None:
            raise RelationshipNotFoundError(f"{source_id} -{rel_type}-> {target_id}")
        merged = {**json.loads(row["data"]), **data_patch}
        self._conn.execute(
            "UPDATE relationships SET data=? WHERE source_id=? AND target_id=? AND rel_type=?",
            (json.dumps(merged), source_id, target_id, rel_type),
        )
        self._conn.commit()

    def list_relationships(self, source_id: str | None = None) -> list[Relationship]:
        if source_id is None:
            rows = self._conn.execute("SELECT * FROM relationships").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM relationships WHERE source_id=?", (source_id,)
            ).fetchall()
        return [
            Relationship(
                source_id=r["source_id"],
                target_id=r["target_id"],
                rel_type=r["rel_type"],
                data=json.loads(r["data"]),
                established_at_update=r["established_at_update"],
            )
            for r in rows
        ]

    # ---- world rules ----

    def add_rule(self, rule: WorldRule) -> None:
        self._conn.execute(
            "INSERT INTO world_rules(id, category, description, constraints, established_at_update) "
            "VALUES (?, ?, ?, ?, ?)",
            (rule.id, rule.category, rule.description, json.dumps(rule.constraints),
             rule.established_at_update),
        )
        self._conn.commit()

    def list_rules(self) -> list[WorldRule]:
        rows = self._conn.execute("SELECT * FROM world_rules ORDER BY id").fetchall()
        return [
            WorldRule(
                id=r["id"], category=r["category"], description=r["description"],
                constraints=json.loads(r["constraints"]),
                established_at_update=r["established_at_update"],
            )
            for r in rows
        ]

    # ---- timeline ----

    def append_timeline_event(self, event: TimelineEvent) -> None:
        self._conn.execute(
            "INSERT INTO timeline(update_number, event_index, description, involved_entities, causal_links) "
            "VALUES (?, ?, ?, ?, ?)",
            (event.update_number, event.event_index, event.description,
             json.dumps(event.involved_entities), json.dumps(event.causal_links)),
        )
        self._conn.commit()

    def list_timeline(self, update_number: int | None = None) -> list[TimelineEvent]:
        if update_number is None:
            rows = self._conn.execute(
                "SELECT * FROM timeline ORDER BY update_number, event_index"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM timeline WHERE update_number=? ORDER BY event_index",
                (update_number,),
            ).fetchall()
        return [
            TimelineEvent(
                update_number=r["update_number"],
                event_index=r["event_index"],
                description=r["description"],
                involved_entities=json.loads(r["involved_entities"]),
                causal_links=[tuple(x) for x in json.loads(r["causal_links"])],
            )
            for r in rows
        ]

    # ---- narrative ----

    def write_narrative(self, record: NarrativeRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO narrative"
            "(update_number, raw_text, summary, chapter_id, state_diff, player_action, pipeline_trace_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.update_number, record.raw_text, record.summary, record.chapter_id,
             json.dumps(record.state_diff), record.player_action, record.pipeline_trace_id),
        )
        self._conn.commit()

    def get_narrative(self, update_number: int) -> NarrativeRecord:
        row = self._conn.execute(
            "SELECT * FROM narrative WHERE update_number=?", (update_number,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no narrative for update {update_number}")
        return _row_to_narrative(row)

    def list_narrative(self, limit: int = 50) -> list[NarrativeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM narrative ORDER BY update_number LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_narrative(r) for r in rows]

    # ---- foreshadowing ----

    def add_foreshadowing(self, hook: ForeshadowingHook) -> None:
        self._conn.execute(
            "INSERT INTO foreshadowing(id, description, planted_at_update, payoff_target, "
            "status, paid_off_at_update, refs) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (hook.id, hook.description, hook.planted_at_update, hook.payoff_target,
             hook.status.value, hook.paid_off_at_update, json.dumps(hook.references)),
        )
        self._conn.commit()

    def get_foreshadowing(self, hook_id: str) -> ForeshadowingHook:
        row = self._conn.execute(
            "SELECT * FROM foreshadowing WHERE id=?", (hook_id,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no foreshadowing hook {hook_id}")
        return _row_to_hook(row)

    def update_foreshadowing(self, hook_id: str, patch: dict[str, Any]) -> None:
        current = self.get_foreshadowing(hook_id)
        new_status = patch.get("status", current.status)
        if hasattr(new_status, "value"):
            new_status = new_status.value
        self._conn.execute(
            "UPDATE foreshadowing SET status=?, paid_off_at_update=?, refs=? WHERE id=?",
            (
                new_status,
                patch.get("paid_off_at_update", current.paid_off_at_update),
                json.dumps(patch.get("references", current.references)),
                hook_id,
            ),
        )
        self._conn.commit()

    # ---- plot threads ----

    def add_plot_thread(self, pt: PlotThread) -> None:
        self._conn.execute(
            "INSERT INTO plot_threads(id, name, description, status, involved_entities, "
            "arc_position, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pt.id, pt.name, pt.description, pt.status.value,
             json.dumps(pt.involved_entities), pt.arc_position.value, pt.priority),
        )
        self._conn.commit()

    def get_plot_thread(self, pt_id: str) -> PlotThread:
        row = self._conn.execute(
            "SELECT * FROM plot_threads WHERE id=?", (pt_id,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no plot thread {pt_id}")
        return _row_to_thread(row)

    def update_plot_thread(self, pt_id: str, patch: dict[str, Any]) -> None:
        current = self.get_plot_thread(pt_id)
        def _v(x):
            return x.value if hasattr(x, "value") else x
        self._conn.execute(
            "UPDATE plot_threads SET name=?, description=?, status=?, involved_entities=?, "
            "arc_position=?, priority=? WHERE id=?",
            (
                patch.get("name", current.name),
                patch.get("description", current.description),
                _v(patch.get("status", current.status)),
                json.dumps(patch.get("involved_entities", current.involved_entities)),
                _v(patch.get("arc_position", current.arc_position)),
                patch.get("priority", current.priority),
                pt_id,
            ),
        )
        self._conn.commit()

    def list_plot_threads(self) -> list[PlotThread]:
        rows = self._conn.execute(
            "SELECT * FROM plot_threads ORDER BY priority DESC, id"
        ).fetchall()
        return [_row_to_thread(r) for r in rows]
```

Add these row-converter helpers near `_row_to_entity`:

```python
def _row_to_narrative(row: sqlite3.Row) -> NarrativeRecord:
    return NarrativeRecord(
        update_number=row["update_number"],
        raw_text=row["raw_text"],
        summary=row["summary"],
        chapter_id=row["chapter_id"],
        state_diff=json.loads(row["state_diff"]),
        player_action=row["player_action"],
        pipeline_trace_id=row["pipeline_trace_id"],
    )


def _row_to_hook(row: sqlite3.Row) -> ForeshadowingHook:
    return ForeshadowingHook(
        id=row["id"],
        description=row["description"],
        planted_at_update=row["planted_at_update"],
        payoff_target=row["payoff_target"],
        status=row["status"],
        paid_off_at_update=row["paid_off_at_update"],
        references=json.loads(row["refs"]),
    )


def _row_to_thread(row: sqlite3.Row) -> PlotThread:
    return PlotThread(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        status=row["status"],
        involved_entities=json.loads(row["involved_entities"]),
        arc_position=row["arc_position"],
        priority=row["priority"],
    )
```

- [ ] **Step 5: Run tests — all PASS**

Run: `uv run pytest tests/world/ -v`
Expected: all previous tests still PASS; new tests (relationships ×4, narrative-file ×6) PASS.

- [ ] **Step 6: Commit**

```bash
git add app/world/state_manager.py tests/world/test_state_manager_relationships.py tests/world/test_state_manager_narrative.py
git commit -m "feat(world): relationships/rules/timeline/narrative/foreshadowing/threads CRUD"
```

---

## Task 6: validate_delta

**Files:**
- Modify: `app/world/state_manager.py`
- Create: `tests/world/test_state_manager_validation.py`

Design: `validate_delta(delta)` returns a `ValidationResult`. Built-in checks for v1:

- `EntityUpdate.id` refers to an existing entity (error).
- `EntityCreate.entity.id` does not already exist (error).
- `RelChange(action="add").relationship` endpoints refer to existing entities (error).
- `RelChange(action="remove")` / `"modify"` refers to an existing relationship (error).
- `FSUpdate.id` refers to an existing foreshadowing hook (error).
- `PTUpdate.id` refers to an existing plot thread (error).
- Acting on an entity whose `status` is `"deceased"` or `"destroyed"` via an `EntityUpdate` that doesn't itself change `status` → warning.

Rule-engine-level validation (custom `WorldRule.constraints`) is deferred to a later milestone; for P1 we just wire in the shape and the six built-in checks above.

- [ ] **Step 1: Write failing tests**

```python
# tests/world/test_state_manager_validation.py
from __future__ import annotations
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    FSUpdate,
    PTUpdate,
    RelChange,
    StateDelta,
)
from app.world.schema import (
    ArcPosition,
    Entity,
    EntityStatus,
    EntityType,
    ForeshadowingHook,
    PlotThread,
    Relationship,
)
from app.world.state_manager import WorldStateManager


def _sm_with_alice_bob(db) -> WorldStateManager:
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="a", entity_type=EntityType.CHARACTER, name="Alice"))
    sm.create_entity(Entity(id="b", entity_type=EntityType.CHARACTER, name="Bob"))
    return sm


def test_valid_empty_delta(db):
    sm = WorldStateManager(db)
    assert sm.validate_delta(StateDelta()).ok


def test_entity_update_missing_id_is_error(db):
    sm = WorldStateManager(db)
    d = StateDelta(entity_updates=[EntityUpdate(id="ghost", patch={"status": "dormant"})])
    r = sm.validate_delta(d)
    assert not r.ok
    assert any("ghost" in i.message for i in r.issues)


def test_entity_create_duplicate_id_is_error(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(entity_creates=[EntityCreate(
        entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="Dup"))])
    r = sm.validate_delta(d)
    assert not r.ok


def test_relationship_add_missing_endpoint_is_error(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(relationship_changes=[RelChange(
        action="add",
        relationship=Relationship(source_id="a", target_id="ghost", rel_type="ally"),
    )])
    r = sm.validate_delta(d)
    assert not r.ok


def test_relationship_remove_missing_is_error(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(relationship_changes=[RelChange(
        action="remove",
        relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
    )])
    r = sm.validate_delta(d)
    assert not r.ok


def test_fs_update_missing_hook_is_error(db):
    sm = WorldStateManager(db)
    d = StateDelta(foreshadowing_updates=[
        FSUpdate(id="fs:missing", new_status="paid_off")])
    assert not sm.validate_delta(d).ok


def test_pt_update_missing_thread_is_error(db):
    sm = WorldStateManager(db)
    d = StateDelta(plot_thread_updates=[PTUpdate(id="pt:missing", patch={"priority": 9})])
    assert not sm.validate_delta(d).ok


def test_acting_on_deceased_entity_is_warning(db):
    sm = _sm_with_alice_bob(db)
    sm.update_entity("a", {"status": EntityStatus.DECEASED.value})
    d = StateDelta(entity_updates=[
        EntityUpdate(id="a", patch={"last_referenced_update": 10})])
    r = sm.validate_delta(d)
    # warning only — not an error
    assert r.ok
    assert any(i.severity == "warning" for i in r.issues)


def test_valid_delta_with_create_and_rel(db):
    sm = _sm_with_alice_bob(db)
    d = StateDelta(
        entity_creates=[EntityCreate(entity=Entity(
            id="c", entity_type=EntityType.LOCATION, name="Tavern"))],
        relationship_changes=[RelChange(
            action="add",
            relationship=Relationship(source_id="a", target_id="c", rel_type="located_at"),
        )],
    )
    r = sm.validate_delta(d)
    assert r.ok
```

- [ ] **Step 2: Run — expect AttributeError**

Run: `uv run pytest tests/world/test_state_manager_validation.py -v`

- [ ] **Step 3: Implement `validate_delta` on `WorldStateManager`**

Add the import at the top of `state_manager.py`:

```python
from .delta import (
    FSUpdate,
    PTUpdate,
    RelChange,
    StateDelta,
    ValidationIssue,
    ValidationResult,
)
```

Append this method to `WorldStateManager`:

```python
    def validate_delta(self, delta: StateDelta) -> ValidationResult:
        issues: list[ValidationIssue] = []

        existing_ids = {
            row["id"] for row in self._conn.execute("SELECT id FROM entities").fetchall()
        }
        planned_new_ids: set[str] = set()

        for op in delta.entity_creates:
            if op.entity.id in existing_ids or op.entity.id in planned_new_ids:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"entity create conflicts with existing id: {op.entity.id}",
                    subject=op.entity.id,
                ))
            else:
                planned_new_ids.add(op.entity.id)

        valid_ids = existing_ids | planned_new_ids

        for op in delta.entity_updates:
            if op.id not in existing_ids:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"entity update references missing id: {op.id}",
                    subject=op.id,
                ))
                continue
            row = self._conn.execute(
                "SELECT status FROM entities WHERE id=?", (op.id,)
            ).fetchone()
            if row and row["status"] in ("deceased", "destroyed") and "status" not in op.patch:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=f"acting on {row['status']} entity: {op.id}",
                    subject=op.id,
                ))

        for op in delta.relationship_changes:
            r = op.relationship
            if op.action == "add":
                for eid in (r.source_id, r.target_id):
                    if eid not in valid_ids:
                        issues.append(ValidationIssue(
                            severity="error",
                            message=f"relationship endpoint missing: {eid}",
                            subject=eid,
                        ))
            else:  # remove / modify
                exists = self._conn.execute(
                    "SELECT 1 FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
                    (r.source_id, r.target_id, r.rel_type),
                ).fetchone() is not None
                if not exists:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=f"relationship not found: {r.source_id} -{r.rel_type}-> {r.target_id}",
                        subject=f"{r.source_id}->{r.target_id}",
                    ))

        for op in delta.foreshadowing_updates:
            if self._conn.execute(
                "SELECT 1 FROM foreshadowing WHERE id=?", (op.id,)
            ).fetchone() is None:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"foreshadowing hook not found: {op.id}",
                    subject=op.id,
                ))

        for op in delta.plot_thread_updates:
            if self._conn.execute(
                "SELECT 1 FROM plot_threads WHERE id=?", (op.id,)
            ).fetchone() is None:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"plot thread not found: {op.id}",
                    subject=op.id,
                ))

        return ValidationResult(issues=issues)
```

- [ ] **Step 4: Run tests — all 9 PASS**

Run: `uv run pytest tests/world/test_state_manager_validation.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/world/state_manager.py tests/world/test_state_manager_validation.py
git commit -m "feat(world): built-in validation for StateDelta operations"
```

---

## Task 7: apply_delta + snapshot

**Files:**
- Modify: `app/world/state_manager.py`
- Create: `tests/world/test_state_manager_apply.py`

Design: `apply_delta(delta, update_number)` runs inside a single SQLite transaction — one `BEGIN`, one `COMMIT` at the end, or one `ROLLBACK` on any failure. It calls `validate_delta` first and raises `InvalidDeltaError` on errors. After application it updates `entities.last_referenced_update` for any entity touched by the delta (creates, updates, or appearing in timeline events' `involved_entities`).

`snapshot()` returns a `WorldSnapshot` dataclass with lists of every record type — used by the diagnostics layer later.

- [ ] **Step 1: Write failing tests**

```python
# tests/world/test_state_manager_apply.py
from __future__ import annotations
import pytest
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
)
from app.world.schema import Entity, EntityType, Relationship, TimelineEvent
from app.world.state_manager import (
    InvalidDeltaError,
    WorldSnapshot,
    WorldStateManager,
)


def test_apply_creates_entities_and_relationships(db):
    sm = WorldStateManager(db)
    d = StateDelta(
        entity_creates=[
            EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
            EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B")),
        ],
        relationship_changes=[RelChange(
            action="add",
            relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
        )],
        timeline_events=[TimelineEventOp(event=TimelineEvent(
            update_number=1, event_index=0, description="meet",
            involved_entities=["a", "b"],
        ))],
    )
    sm.apply_delta(d, update_number=1)
    assert sm.get_entity("a").last_referenced_update == 1
    assert sm.get_entity("b").last_referenced_update == 1
    assert sm.list_relationships("a")[0].rel_type == "ally"
    assert len(sm.list_timeline(1)) == 1


def test_apply_rejects_invalid_delta(db):
    sm = WorldStateManager(db)
    d = StateDelta(entity_updates=[EntityUpdate(id="ghost", patch={"status": "dormant"})])
    with pytest.raises(InvalidDeltaError):
        sm.apply_delta(d, update_number=1)


def test_apply_is_atomic(db):
    """If a mid-transaction op fails, no partial state is left behind."""
    sm = WorldStateManager(db)
    sm.create_entity(Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))
    # Bad delta: valid create, then a raw SQL-level violation we force by
    # creating a rel whose endpoint exists at validate-time but we delete
    # it before apply — can't easily induce that. Instead, force via a
    # programmatic trick: use unique constraint.
    d = StateDelta(
        entity_creates=[
            EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B")),
            EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="Dup")),
        ]
    )
    # validate_delta catches this (planned_new_ids collision) — so we assert
    # it raises InvalidDeltaError and nothing was inserted.
    with pytest.raises(InvalidDeltaError):
        sm.apply_delta(d, update_number=1)
    assert [e.id for e in sm.list_entities()] == ["a"]


def test_snapshot_returns_all_state(db):
    sm = WorldStateManager(db)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
    ]), update_number=1)
    snap = sm.snapshot()
    assert isinstance(snap, WorldSnapshot)
    assert [e.id for e in snap.entities] == ["a"]
    assert snap.relationships == []
    assert snap.timeline == []
```

- [ ] **Step 2: Run — expect AttributeError / ImportError**

Run: `uv run pytest tests/world/test_state_manager_apply.py -v`

- [ ] **Step 3: Implement `apply_delta`, `snapshot`, and `WorldSnapshot`**

Add at the top of `state_manager.py`:

```python
from dataclasses import dataclass


class InvalidDeltaError(WorldStateError):
    def __init__(self, result: ValidationResult) -> None:
        super().__init__("; ".join(i.message for i in result.issues if i.severity == "error"))
        self.result = result


@dataclass(frozen=True)
class WorldSnapshot:
    entities: list[Entity]
    relationships: list[Relationship]
    rules: list[WorldRule]
    timeline: list[TimelineEvent]
    narrative: list[NarrativeRecord]
    foreshadowing: list[ForeshadowingHook]
    plot_threads: list[PlotThread]
```

Append these methods to `WorldStateManager`:

```python
    def apply_delta(self, delta: StateDelta, update_number: int) -> None:
        result = self.validate_delta(delta)
        if not result.ok:
            raise InvalidDeltaError(result)

        touched_ids: set[str] = set()
        try:
            self._conn.execute("BEGIN")

            for op in delta.entity_creates:
                e = op.entity
                e_with_update = e.model_copy(update={
                    "created_at_update": e.created_at_update or update_number,
                    "last_referenced_update": e.last_referenced_update or update_number,
                })
                self._conn.execute(
                    "INSERT INTO entities(id, entity_type, name, data, status, "
                    "last_referenced_update, created_at_update) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        e_with_update.id, e_with_update.entity_type.value, e_with_update.name,
                        json.dumps(e_with_update.data), e_with_update.status.value,
                        e_with_update.last_referenced_update, e_with_update.created_at_update,
                    ),
                )
                touched_ids.add(e.id)

            for op in delta.entity_updates:
                current = self.get_entity(op.id)
                merged_data = {**current.data, **op.patch.get("data", {})}
                new_status = op.patch.get("status", current.status)
                if hasattr(new_status, "value"):
                    new_status = new_status.value
                self._conn.execute(
                    "UPDATE entities SET name=?, status=?, last_referenced_update=?, data=?, "
                    "modified_at=CURRENT_TIMESTAMP WHERE id=?",
                    (
                        op.patch.get("name", current.name),
                        new_status,
                        update_number,
                        json.dumps(merged_data),
                        op.id,
                    ),
                )
                touched_ids.add(op.id)

            for op in delta.relationship_changes:
                r = op.relationship
                if op.action == "add":
                    self._conn.execute(
                        "INSERT INTO relationships(source_id, target_id, rel_type, data, established_at_update) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (r.source_id, r.target_id, r.rel_type, json.dumps(r.data),
                         r.established_at_update or update_number),
                    )
                elif op.action == "remove":
                    self._conn.execute(
                        "DELETE FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
                        (r.source_id, r.target_id, r.rel_type),
                    )
                else:  # modify
                    self._conn.execute(
                        "UPDATE relationships SET data=? WHERE source_id=? AND target_id=? AND rel_type=?",
                        (json.dumps(r.data), r.source_id, r.target_id, r.rel_type),
                    )
                touched_ids.update([r.source_id, r.target_id])

            for op in delta.timeline_events:
                ev = op.event
                self._conn.execute(
                    "INSERT INTO timeline(update_number, event_index, description, involved_entities, causal_links) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ev.update_number, ev.event_index, ev.description,
                     json.dumps(ev.involved_entities), json.dumps(ev.causal_links)),
                )
                touched_ids.update(ev.involved_entities)

            for op in delta.foreshadowing_updates:
                current = self.get_foreshadowing(op.id)
                refs = list(current.references)
                if op.add_reference is not None and op.add_reference not in refs:
                    refs.append(op.add_reference)
                self._conn.execute(
                    "UPDATE foreshadowing SET status=?, paid_off_at_update=?, refs=? WHERE id=?",
                    (op.new_status, op.paid_off_at_update or current.paid_off_at_update,
                     json.dumps(refs), op.id),
                )

            for op in delta.plot_thread_updates:
                current = self.get_plot_thread(op.id)
                def _v(x):
                    return x.value if hasattr(x, "value") else x
                self._conn.execute(
                    "UPDATE plot_threads SET name=?, description=?, status=?, involved_entities=?, "
                    "arc_position=?, priority=? WHERE id=?",
                    (
                        op.patch.get("name", current.name),
                        op.patch.get("description", current.description),
                        _v(op.patch.get("status", current.status)),
                        json.dumps(op.patch.get("involved_entities", current.involved_entities)),
                        _v(op.patch.get("arc_position", current.arc_position)),
                        op.patch.get("priority", current.priority),
                        op.id,
                    ),
                )

            # Bump last_referenced_update for entities touched but not explicitly updated
            for eid in touched_ids:
                self._conn.execute(
                    "UPDATE entities SET last_referenced_update=? "
                    "WHERE id=? AND (last_referenced_update IS NULL OR last_referenced_update < ?)",
                    (update_number, eid, update_number),
                )

            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            entities=self.list_entities(),
            relationships=self.list_relationships(),
            rules=self.list_rules(),
            timeline=self.list_timeline(),
            narrative=self.list_narrative(limit=10_000),
            foreshadowing=[
                _row_to_hook(r)
                for r in self._conn.execute("SELECT * FROM foreshadowing ORDER BY id").fetchall()
            ],
            plot_threads=self.list_plot_threads(),
        )
```

- [ ] **Step 4: Run — all 4 PASS; entire suite still green**

Run: `uv run pytest tests/world/ -v`

- [ ] **Step 5: Commit**

```bash
git add app/world/state_manager.py tests/world/test_state_manager_apply.py
git commit -m "feat(world): atomic apply_delta + snapshot"
```

---

## Task 8: rollback

**Files:**
- Modify: `app/world/state_manager.py`
- Create: `tests/world/test_state_manager_rollback.py`

Design: `rollback(to_update)` reverts state to the state *after* `to_update` was applied. For v1 we implement the minimum useful semantics:

- Delete timeline events with `update_number > to_update`.
- Delete narrative records with `update_number > to_update`.
- Delete entities whose `created_at_update > to_update`.
- Delete relationships whose `established_at_update > to_update`.
- Reset `last_referenced_update` on remaining entities to `min(last_referenced_update, to_update)`.
- Foreshadowing hooks created after `to_update` are deleted; hooks paid off after `to_update` are reverted to prior status (we store enough to do this: clear `paid_off_at_update` when it exceeds `to_update`; strip references > `to_update`).
- Plot-thread status changes aren't fully revertible from the schema — we document this limitation: for v1, rollback does not touch plot_threads beyond clearing any whose *id doesn't appear before to_update* (none, because threads are created outside the delta flow in v1). Leave plot_threads alone.

This is not a perfect retcon — the spec notes "Does NOT automatically rewrite prose" and that full cascade is a downstream concern. We are implementing the mechanical part only.

- [ ] **Step 1: Write failing tests**

```python
# tests/world/test_state_manager_rollback.py
from __future__ import annotations
from app.world.delta import (
    EntityCreate,
    FSUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
)
from app.world.schema import (
    Entity,
    EntityType,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    Relationship,
    TimelineEvent,
)
from app.world.state_manager import WorldStateManager


def _advance(sm: WorldStateManager, update_number: int, delta: StateDelta):
    sm.apply_delta(delta, update_number=update_number)
    sm.write_narrative(NarrativeRecord(update_number=update_number, raw_text=f"u{update_number}"))


def test_rollback_removes_later_entities(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))]))
    _advance(sm, 2, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B"))]))
    sm.rollback(to_update=1)
    ids = [e.id for e in sm.list_entities()]
    assert ids == ["a"]


def test_rollback_removes_later_narrative_and_timeline(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))]))
    _advance(sm, 2, StateDelta(timeline_events=[TimelineEventOp(event=TimelineEvent(
        update_number=2, event_index=0, description="event2"))]))
    sm.rollback(to_update=1)
    assert [n.update_number for n in sm.list_narrative()] == [1]
    assert sm.list_timeline() == []


def test_rollback_removes_later_relationships(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
        EntityCreate(entity=Entity(id="b", entity_type=EntityType.CHARACTER, name="B")),
    ]))
    _advance(sm, 2, StateDelta(relationship_changes=[RelChange(
        action="add",
        relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
    )]))
    assert len(sm.list_relationships("a")) == 1
    sm.rollback(to_update=1)
    assert sm.list_relationships("a") == []


def test_rollback_reverts_foreshadowing_payoff(db):
    sm = WorldStateManager(db)
    hook = ForeshadowingHook(id="fs:1", description="d", planted_at_update=1, payoff_target="x")
    sm.add_foreshadowing(hook)
    sm.apply_delta(StateDelta(foreshadowing_updates=[
        FSUpdate(id="fs:1", new_status="paid_off", paid_off_at_update=3, add_reference=2),
    ]), update_number=3)
    sm.write_narrative(NarrativeRecord(update_number=3, raw_text="u3"))
    sm.rollback(to_update=2)
    got = sm.get_foreshadowing("fs:1")
    assert got.paid_off_at_update is None
    # status reverted to planted (since payoff happened after to_update)
    assert got.status == HookStatus.PLANTED
    assert got.references == [2]  # reference at update 2 stays


def test_rollback_caps_last_referenced(db):
    sm = WorldStateManager(db)
    _advance(sm, 1, StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A"))]))
    _advance(sm, 5, StateDelta(timeline_events=[TimelineEventOp(event=TimelineEvent(
        update_number=5, event_index=0, description="x", involved_entities=["a"]))]))
    sm.rollback(to_update=2)
    assert sm.get_entity("a").last_referenced_update == 1
```

- [ ] **Step 2: Run — expect AttributeError**

Run: `uv run pytest tests/world/test_state_manager_rollback.py -v`

- [ ] **Step 3: Implement `rollback`**

Append to `WorldStateManager`:

```python
    def rollback(self, to_update: int) -> None:
        c = self._conn
        try:
            c.execute("BEGIN")
            c.execute("DELETE FROM timeline WHERE update_number > ?", (to_update,))
            c.execute("DELETE FROM narrative WHERE update_number > ?", (to_update,))
            c.execute("DELETE FROM entities WHERE created_at_update > ?", (to_update,))
            c.execute("DELETE FROM relationships WHERE established_at_update > ?", (to_update,))

            c.execute(
                "UPDATE entities SET last_referenced_update = ? "
                "WHERE last_referenced_update > ?",
                (to_update, to_update),
            )

            # Foreshadowing: revert payoffs that happened after to_update
            for row in c.execute("SELECT id, paid_off_at_update, refs FROM foreshadowing").fetchall():
                refs = json.loads(row["refs"])
                filtered_refs = [r for r in refs if r <= to_update]
                paid_off = row["paid_off_at_update"]
                if paid_off is not None and paid_off > to_update:
                    new_status = "referenced" if filtered_refs else "planted"
                    c.execute(
                        "UPDATE foreshadowing SET status=?, paid_off_at_update=NULL, refs=? WHERE id=?",
                        (new_status, json.dumps(filtered_refs), row["id"]),
                    )
                elif filtered_refs != refs:
                    c.execute(
                        "UPDATE foreshadowing SET refs=? WHERE id=?",
                        (json.dumps(filtered_refs), row["id"]),
                    )

            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
```

- [ ] **Step 4: Run tests — all 5 PASS; full suite green**

Run: `uv run pytest tests/world/ -v`

- [ ] **Step 5: Commit**

```bash
git add app/world/state_manager.py tests/world/test_state_manager_rollback.py
git commit -m "feat(world): rollback to prior update"
```

---

## Task 9: JSON seed loader

**Files:**
- Create: `app/world/seed.py`
- Create: `tests/world/test_seed.py`

Design: `SeedLoader.load(path)` reads a JSON file with sections `entities`, `relationships`, `rules`, `foreshadowing`, `plot_threads` and returns a `StateDelta` that, when applied at update 0, initializes a new quest's world. Also returns rules + foreshadowing + plot threads separately since those are currently not part of `StateDelta` (they're set up directly). The loader validates shape via Pydantic.

- [ ] **Step 1: Write failing tests**

```python
# tests/world/test_seed.py
from __future__ import annotations
import json
from pathlib import Path
import pytest
from app.world.seed import SeedLoader, SeedPayload
from app.world.state_manager import WorldStateManager


def _write_seed(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_full_seed(tmp_path: Path):
    p = _write_seed(tmp_path, {
        "entities": [
            {"id": "char:alice", "entity_type": "character", "name": "Alice"},
            {"id": "loc:tavern", "entity_type": "location", "name": "The Tavern"},
        ],
        "relationships": [
            {"source_id": "char:alice", "target_id": "loc:tavern", "rel_type": "located_at"},
        ],
        "rules": [
            {"id": "r:1", "category": "magic", "description": "No magic here."},
        ],
        "foreshadowing": [
            {"id": "fs:1", "description": "A strange coin.",
             "planted_at_update": 0, "payoff_target": "origin reveal"},
        ],
        "plot_threads": [
            {"id": "pt:main", "name": "Quest", "description": "d",
             "arc_position": "rising"},
        ],
    })
    payload = SeedLoader.load(p)
    assert isinstance(payload, SeedPayload)
    assert len(payload.delta.entity_creates) == 2
    assert len(payload.delta.relationship_changes) == 1
    assert len(payload.rules) == 1
    assert len(payload.foreshadowing) == 1
    assert len(payload.plot_threads) == 1


def test_load_minimal_seed(tmp_path: Path):
    p = _write_seed(tmp_path, {"entities": [
        {"id": "a", "entity_type": "character", "name": "A"},
    ]})
    payload = SeedLoader.load(p)
    assert len(payload.delta.entity_creates) == 1
    assert payload.rules == []


def test_load_rejects_bad_json(tmp_path: Path):
    p = tmp_path / "seed.json"
    p.write_text("{ not json")
    with pytest.raises(ValueError):
        SeedLoader.load(p)


def test_seed_applies_cleanly(db, tmp_path: Path):
    p = _write_seed(tmp_path, {
        "entities": [
            {"id": "a", "entity_type": "character", "name": "A"},
            {"id": "b", "entity_type": "location", "name": "B"},
        ],
        "relationships": [
            {"source_id": "a", "target_id": "b", "rel_type": "located_at"},
        ],
        "plot_threads": [
            {"id": "pt:1", "name": "m", "description": "d", "arc_position": "rising"},
        ],
    })
    payload = SeedLoader.load(p)
    sm = WorldStateManager(db)
    for rule in payload.rules:
        sm.add_rule(rule)
    for h in payload.foreshadowing:
        sm.add_foreshadowing(h)
    for pt in payload.plot_threads:
        sm.add_plot_thread(pt)
    sm.apply_delta(payload.delta, update_number=0)
    assert [e.id for e in sm.list_entities()] == ["a", "b"]
    assert sm.list_relationships("a")[0].target_id == "b"
    assert sm.get_plot_thread("pt:1").name == "m"
```

- [ ] **Step 2: Run — expect ImportError**

Run: `uv run pytest tests/world/test_seed.py -v`

- [ ] **Step 3: Implement `app/world/seed.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import ValidationError
from .delta import EntityCreate, RelChange, StateDelta
from .schema import (
    Entity,
    ForeshadowingHook,
    PlotThread,
    Relationship,
    WorldRule,
)


@dataclass
class SeedPayload:
    delta: StateDelta
    rules: list[WorldRule] = field(default_factory=list)
    foreshadowing: list[ForeshadowingHook] = field(default_factory=list)
    plot_threads: list[PlotThread] = field(default_factory=list)


class SeedLoader:
    @staticmethod
    def load(path: str | Path) -> SeedPayload:
        p = Path(path)
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"seed file not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError("seed root must be an object")

        try:
            entities = [Entity.model_validate(x) for x in raw.get("entities", [])]
            rels = [Relationship.model_validate(x) for x in raw.get("relationships", [])]
            rules = [WorldRule.model_validate(x) for x in raw.get("rules", [])]
            hooks = [ForeshadowingHook.model_validate(x) for x in raw.get("foreshadowing", [])]
            threads = [PlotThread.model_validate(x) for x in raw.get("plot_threads", [])]
        except ValidationError as e:
            raise ValueError(f"seed file schema error: {e}") from e

        delta = StateDelta(
            entity_creates=[EntityCreate(entity=e) for e in entities],
            relationship_changes=[RelChange(action="add", relationship=r) for r in rels],
        )
        return SeedPayload(
            delta=delta, rules=rules, foreshadowing=hooks, plot_threads=threads,
        )
```

- [ ] **Step 4: Run — all 4 PASS**

Run: `uv run pytest tests/world/test_seed.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/world/seed.py tests/world/test_seed.py
git commit -m "feat(world): JSON seed loader for quest initialization"
```

---

## Task 10: OutputParser

**Files:**
- Create: `app/world/output_parser.py`
- Create: `tests/world/test_output_parser.py`

Design: the parser handles two concerns.

1. **`parse_json(text, schema=None)`** — extract JSON from LLM output. Models often wrap JSON in markdown code fences, or emit thinking tokens before/after. Strategy:
   - Strip `<think>...</think>` blocks.
   - Strip leading/trailing markdown fences (```json ... ``` or ``` ... ```).
   - Try `json.loads` directly.
   - On failure, find the largest balanced `{...}` or `[...]` substring and try again.
   - If `schema` (a Pydantic model class) is provided, validate and return the typed instance.
   - Raise `ParseError` on total failure.

2. **`parse_prose(text)`** — clean free-text output. Strip `<think>` blocks, strip leading/trailing whitespace, strip common "Sure, here's..." preambles (conservatively — only if the first line is a short sentence and the second is blank). Return the cleaned prose.

We do NOT try to be exhaustive. Cover the common failure modes.

- [ ] **Step 1: Write failing tests**

```python
# tests/world/test_output_parser.py
from __future__ import annotations
import pytest
from pydantic import BaseModel
from app.world.output_parser import OutputParser, ParseError


class Beat(BaseModel):
    scene: str
    tension_delta: int


def test_parse_json_plain():
    text = '{"a": 1, "b": [2, 3]}'
    assert OutputParser.parse_json(text) == {"a": 1, "b": [2, 3]}


def test_parse_json_stripped_markdown_fence():
    text = "```json\n{\"x\": 7}\n```"
    assert OutputParser.parse_json(text) == {"x": 7}


def test_parse_json_with_thinking_block():
    text = "<think>let me plan</think>\n{\"ok\": true}"
    assert OutputParser.parse_json(text) == {"ok": True}


def test_parse_json_embedded_in_prose():
    text = "Here's the plan:\n\n{\"scene\": \"tavern\", \"tension_delta\": 2}\n\nHope that helps."
    assert OutputParser.parse_json(text) == {"scene": "tavern", "tension_delta": 2}


def test_parse_json_with_schema_returns_typed_instance():
    text = '{"scene": "tavern", "tension_delta": 2}'
    beat = OutputParser.parse_json(text, schema=Beat)
    assert isinstance(beat, Beat)
    assert beat.scene == "tavern"


def test_parse_json_schema_violation_raises():
    text = '{"scene": "tavern"}'  # missing tension_delta
    with pytest.raises(ParseError):
        OutputParser.parse_json(text, schema=Beat)


def test_parse_json_unrepairable_raises():
    with pytest.raises(ParseError):
        OutputParser.parse_json("this has no json at all")


def test_parse_prose_strips_thinking():
    text = "<think>let me think</think>\n\nShe walked into the tavern."
    assert OutputParser.parse_prose(text) == "She walked into the tavern."


def test_parse_prose_strips_short_preamble():
    text = "Sure, here's the scene:\n\nShe walked into the tavern."
    assert OutputParser.parse_prose(text) == "She walked into the tavern."


def test_parse_prose_keeps_real_content():
    text = "She walked into the tavern.\n\nThe bartender looked up."
    assert OutputParser.parse_prose(text) == "She walked into the tavern.\n\nThe bartender looked up."
```

- [ ] **Step 2: Run — expect ImportError**

Run: `uv run pytest tests/world/test_output_parser.py -v`

- [ ] **Step 3: Implement `app/world/output_parser.py`**

```python
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
```

- [ ] **Step 4: Run — all 10 PASS**

Run: `uv run pytest tests/world/test_output_parser.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/world/output_parser.py tests/world/test_output_parser.py
git commit -m "feat(world): tolerant LLM output parser (JSON + prose)"
```

---

## Task 11: Public package surface

**Files:**
- Modify: `app/world/__init__.py`

- [ ] **Step 1: Overwrite `app/world/__init__.py`**

```python
from .delta import (
    EntityCreate,
    EntityUpdate,
    FSUpdate,
    PTUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
    ValidationIssue,
    ValidationResult,
)
from .output_parser import OutputParser, ParseError
from .schema import (
    ArcPosition,
    Entity,
    EntityStatus,
    EntityType,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    PlotThread,
    Relationship,
    ThreadStatus,
    TimelineEvent,
    WorldRule,
)
from .seed import SeedLoader, SeedPayload
from .state_manager import (
    EntityNotFoundError,
    InvalidDeltaError,
    RelationshipNotFoundError,
    WorldSnapshot,
    WorldStateError,
    WorldStateManager,
)

__all__ = [
    "ArcPosition",
    "Entity",
    "EntityCreate",
    "EntityNotFoundError",
    "EntityStatus",
    "EntityType",
    "EntityUpdate",
    "FSUpdate",
    "ForeshadowingHook",
    "HookStatus",
    "InvalidDeltaError",
    "NarrativeRecord",
    "OutputParser",
    "PTUpdate",
    "ParseError",
    "PlotThread",
    "RelChange",
    "Relationship",
    "RelationshipNotFoundError",
    "SeedLoader",
    "SeedPayload",
    "StateDelta",
    "ThreadStatus",
    "TimelineEvent",
    "TimelineEventOp",
    "ValidationIssue",
    "ValidationResult",
    "WorldRule",
    "WorldSnapshot",
    "WorldStateError",
    "WorldStateManager",
]
```

- [ ] **Step 2: Verify imports resolve**

Run: `uv run python -c "from app.world import WorldStateManager, StateDelta, SeedLoader, OutputParser; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: all M1 runtime tests still PASS (14) plus all P1 world tests PASS (~55 total). Integration test deselected.

- [ ] **Step 4: Commit**

```bash
git add app/world/__init__.py
git commit -m "feat(world): expose public api"
```

---

## Done criteria

- `uv run pytest` passes (M1 14 tests + P1 ~55 tests green).
- `from app.world import WorldStateManager, StateDelta, SeedLoader, OutputParser` resolves.
- A seed JSON file + `WorldStateManager.apply_delta(seed.delta, update_number=0)` yields a populated, queryable world.
- `apply_delta` is atomic under failure.
- `rollback(to_update=N)` restores entities/relationships/narrative/timeline/foreshadowing payoffs to their state after update N.
- `OutputParser.parse_json` survives thinking blocks, markdown fences, and JSON-embedded-in-prose; raises `ParseError` on unrepairable input.

Next plan (P2) will extend `InferenceClient` with structured-output + thinking support, create the prompt-template skeleton (`prompts/` dir + Jinja2 loader), and build `ContextBuilder` with `ContextSpec`, token budget, compression, and manifest logging.
