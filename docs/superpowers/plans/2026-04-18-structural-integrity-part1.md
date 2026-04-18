# Structural Integrity Part 1: Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the four independent foundation systems: foreshadow pool with prose verification, typed edit-pass, cross-judge scoring with M-Prometheus-14B, and non-LLM structural metrics.

**Architecture:** Four new modules (`app/planning/foreshadow_pool.py`, `app/engine/typed_edits.py`, `app/scoring/cross_judge.py`, `app/scoring/structural_metrics.py`) each with their own SQLite tables, prompt templates, and test suites. Each integrates into the existing pipeline at specific points but does not depend on the others.

**Tech Stack:** Python 3.11, SQLite, Jinja2, spaCy (syntactic CR), mauve-text (MAUVE), numpy (KL/MTLD), httpx (Prometheus client). All LLM calls via existing `InferenceClient`.

**Spec:** `docs/superpowers/specs/2026-04-18-structural-integrity-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `app/planning/foreshadow_pool.py` | Triple CRUD, predicate evaluation, prose verification, deadline escalation |
| `app/engine/typed_edits.py` | Edit detection (LLM), edit application (deterministic), taxonomy constants |
| `app/scoring/cross_judge.py` | Dual-model scoring, JudgePair dataclass, self-preference detection |
| `app/scoring/structural_metrics.py` | Syntactic CR, MTLD, MAUVE — all deterministic |
| `prompts/critics/foreshadow_verify.j2` | Verification prompt for prose-level hook detection |
| `prompts/stages/typed_edit/system.j2` | Typed edit detector system prompt |
| `prompts/stages/typed_edit/user.j2` | Typed edit detector user prompt |
| `tests/planning/test_foreshadow_pool.py` | Foreshadow pool unit tests |
| `tests/engine/test_typed_edits.py` | Typed edit detection + application tests |
| `tests/scoring/test_cross_judge.py` | Cross-judge scoring tests |
| `tests/scoring/test_structural_metrics.py` | Structural metrics tests |

### Modified Files
| File | Change |
|------|--------|
| `app/world/db.py` | Add 3 new CREATE TABLE statements to SCHEMA_SQL |
| `app/world/state_manager.py` | Add CRUD methods for `foreshadow_triples`, `typed_edits`, `cross_judge_scores` |
| `app/engine/pipeline.py` | Wire typed edits after check stage; wire foreshadow verification after beats |
| `app/rollout/harness.py` | Wire foreshadow trigger checks before beats, verification after beats, cross-judge after scoring |
| `app/rollout/scorer.py` | Refactor `score_chapter` to per-dim independent calls |
| `pyproject.toml` | Add `spacy`, `mauve-text` dependencies |

---

## Task 1: Foreshadow Pool — Schema & CRUD

**Files:**
- Modify: `app/world/db.py` (SCHEMA_SQL, around line 337)
- Modify: `app/world/state_manager.py` (add methods after line 462)
- Create: `tests/planning/test_foreshadow_pool.py`

- [ ] **Step 1: Write the failing test for triple CRUD**

```python
# tests/planning/test_foreshadow_pool.py
from __future__ import annotations
import pytest
from app.world.db import open_db
from app.world.state_manager import WorldStateManager


@pytest.fixture
def sm(tmp_path):
    conn = open_db(tmp_path / "test.db")
    wsm = WorldStateManager(conn)
    yield wsm
    conn.close()


def test_create_and_get_foreshadow_triple(sm):
    sm.create_foreshadow_triple(
        id="ft_abc12345",
        hook_id="fs:pistol",
        foreshadow_text="Tristan notices the pistol's unusual weight",
        trigger_pred={"type": "chapter_gte", "value": 5},
        payoff_text="The pistol fires unexpectedly, revealing its cursed nature",
        planted_chapter=2,
        deadline_chapter=8,
    )
    triple = sm.get_foreshadow_triple("ft_abc12345")
    assert triple["hook_id"] == "fs:pistol"
    assert triple["status"] == "planted"
    assert triple["trigger_pred"] == {"type": "chapter_gte", "value": 5}
    assert triple["deadline_chapter"] == 8
    assert triple["verified_planted"] is None


def test_update_foreshadow_triple_status(sm):
    sm.create_foreshadow_triple(
        id="ft_abc12345",
        hook_id="fs:pistol",
        foreshadow_text="pistol weight",
        trigger_pred={"type": "chapter_gte", "value": 5},
        payoff_text="pistol fires",
        planted_chapter=2,
        deadline_chapter=8,
    )
    sm.update_foreshadow_triple("ft_abc12345", status="triggered")
    assert sm.get_foreshadow_triple("ft_abc12345")["status"] == "triggered"

    sm.update_foreshadow_triple("ft_abc12345", verified_planted=0.85)
    assert sm.get_foreshadow_triple("ft_abc12345")["verified_planted"] == pytest.approx(0.85)


def test_list_foreshadow_triples_by_status(sm):
    for i, status in enumerate(["planted", "planted", "triggered", "paid_off"]):
        sm.create_foreshadow_triple(
            id=f"ft_{i:08d}",
            hook_id=f"fs:hook{i}",
            foreshadow_text=f"text {i}",
            trigger_pred={"type": "chapter_gte", "value": i + 1},
            payoff_text=f"payoff {i}",
            planted_chapter=1,
        )
        if status != "planted":
            sm.update_foreshadow_triple(f"ft_{i:08d}", status=status)

    planted = sm.list_foreshadow_triples(status="planted")
    assert len(planted) == 2
    triggered = sm.list_foreshadow_triples(status="triggered")
    assert len(triggered) == 1


def test_list_overdue_foreshadow_triples(sm):
    sm.create_foreshadow_triple(
        id="ft_overdue1",
        hook_id="fs:overdue",
        foreshadow_text="overdue hook",
        trigger_pred={"type": "chapter_gte", "value": 3},
        payoff_text="should have fired",
        planted_chapter=1,
        deadline_chapter=5,
    )
    overdue = sm.list_overdue_foreshadow_triples(current_chapter=6)
    assert len(overdue) == 1
    assert overdue[0]["id"] == "ft_overdue1"

    not_overdue = sm.list_overdue_foreshadow_triples(current_chapter=4)
    assert len(not_overdue) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/planning/test_foreshadow_pool.py -v`
Expected: FAIL — `create_foreshadow_triple` not defined

- [ ] **Step 3: Add foreshadow_triples table to schema**

In `app/world/db.py`, add to `SCHEMA_SQL` (before the closing `"""` around line 337):

```sql
CREATE TABLE IF NOT EXISTS foreshadow_triples (
    id TEXT PRIMARY KEY,
    hook_id TEXT NOT NULL,
    foreshadow_text TEXT NOT NULL,
    trigger_pred TEXT NOT NULL,  -- JSON
    payoff_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planted',
    planted_chapter INTEGER NOT NULL,
    deadline_chapter INTEGER,
    verified_planted REAL,
    verified_payoff REAL
);
```

- [ ] **Step 4: Add CRUD methods to WorldStateManager**

In `app/world/state_manager.py`, add after the existing `update_foreshadowing` method (around line 475):

```python
def create_foreshadow_triple(
    self, *, id: str, hook_id: str, foreshadow_text: str,
    trigger_pred: dict, payoff_text: str, planted_chapter: int,
    deadline_chapter: int | None = None,
) -> None:
    import json as _json
    self._conn.execute(
        "INSERT INTO foreshadow_triples "
        "(id, hook_id, foreshadow_text, trigger_pred, payoff_text, "
        "planted_chapter, deadline_chapter) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (id, hook_id, foreshadow_text, _json.dumps(trigger_pred),
         payoff_text, planted_chapter, deadline_chapter),
    )
    self._conn.commit()

def get_foreshadow_triple(self, triple_id: str) -> dict | None:
    import json as _json
    row = self._conn.execute(
        "SELECT * FROM foreshadow_triples WHERE id = ?", (triple_id,),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["trigger_pred"] = _json.loads(d["trigger_pred"])
    return d

def update_foreshadow_triple(self, triple_id: str, **fields) -> None:
    import json as _json
    allowed = {"status", "verified_planted", "verified_payoff"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    self._conn.execute(
        f"UPDATE foreshadow_triples SET {set_clause} WHERE id = ?",
        (*updates.values(), triple_id),
    )
    self._conn.commit()

def list_foreshadow_triples(self, status: str | None = None) -> list[dict]:
    import json as _json
    if status:
        rows = self._conn.execute(
            "SELECT * FROM foreshadow_triples WHERE status = ? ORDER BY planted_chapter",
            (status,),
        ).fetchall()
    else:
        rows = self._conn.execute(
            "SELECT * FROM foreshadow_triples ORDER BY planted_chapter",
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["trigger_pred"] = _json.loads(d["trigger_pred"])
        result.append(d)
    return result

def list_overdue_foreshadow_triples(self, current_chapter: int) -> list[dict]:
    import json as _json
    rows = self._conn.execute(
        "SELECT * FROM foreshadow_triples "
        "WHERE status IN ('planted', 'triggered') "
        "AND deadline_chapter IS NOT NULL "
        "AND deadline_chapter < ? "
        "ORDER BY deadline_chapter",
        (current_chapter,),
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["trigger_pred"] = _json.loads(d["trigger_pred"])
        result.append(d)
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/planning/test_foreshadow_pool.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/world/db.py app/world/state_manager.py tests/planning/test_foreshadow_pool.py
git commit -m "feat(foreshadow): triple pool schema + CRUD"
```

---

## Task 2: Foreshadow Pool — Predicate Evaluation

**Files:**
- Create: `app/planning/foreshadow_pool.py`
- Modify: `tests/planning/test_foreshadow_pool.py`

- [ ] **Step 1: Write failing tests for predicate evaluation**

Append to `tests/planning/test_foreshadow_pool.py`:

```python
from app.planning.foreshadow_pool import evaluate_predicate


def test_chapter_gte_predicate():
    pred = {"type": "chapter_gte", "value": 5}
    state = {"current_chapter": 4, "active_entities": [], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False

    state["current_chapter"] = 5
    assert evaluate_predicate(pred, state) is True


def test_entity_active_predicate():
    pred = {"type": "entity_active", "entity_id": "char:cozme"}
    state = {"current_chapter": 1, "active_entities": ["char:tristan"], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False

    state["active_entities"].append("char:cozme")
    assert evaluate_predicate(pred, state) is True


def test_entity_present_predicate():
    pred = {"type": "entity_present", "entity_id": "char:cozme"}
    state = {"current_chapter": 1, "active_entities": [], "present_entities": ["char:tristan"], "events": []}
    assert evaluate_predicate(pred, state) is False

    state["present_entities"].append("char:cozme")
    assert evaluate_predicate(pred, state) is True


def test_event_occurred_predicate():
    pred = {"type": "event_occurred", "event": "tristan_confronts_cozme"}
    state = {"current_chapter": 1, "active_entities": [], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False

    state["events"].append("tristan_confronts_cozme")
    assert evaluate_predicate(pred, state) is True


def test_compound_and_predicate():
    pred = {
        "type": "and",
        "children": [
            {"type": "chapter_gte", "value": 3},
            {"type": "entity_active", "entity_id": "char:cozme"},
        ],
    }
    state = {"current_chapter": 3, "active_entities": [], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False

    state["active_entities"].append("char:cozme")
    assert evaluate_predicate(pred, state) is True


def test_compound_or_predicate():
    pred = {
        "type": "or",
        "children": [
            {"type": "chapter_gte", "value": 10},
            {"type": "entity_active", "entity_id": "char:cozme"},
        ],
    }
    state = {"current_chapter": 3, "active_entities": ["char:cozme"], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/planning/test_foreshadow_pool.py::test_chapter_gte_predicate -v`
Expected: FAIL — cannot import `evaluate_predicate`

- [ ] **Step 3: Implement predicate evaluator**

Create `app/planning/foreshadow_pool.py`:

```python
"""Foreshadow triple pool — CFPG-style promise tracking with prose verification.

Manages (foreshadow, trigger, payoff) triples as first-class objects.
Trigger predicates are evaluated deterministically against world state.
Prose verification uses an LLM call to confirm hooks are legible in text.
"""
from __future__ import annotations


def evaluate_predicate(pred: dict, state: dict) -> bool:
    """Evaluate a trigger predicate against current world state.

    State dict must contain:
        current_chapter: int
        active_entities: list[str]  — entity ids with ACTIVE status
        present_entities: list[str] — entity ids in current scene
        events: list[str]           — KB events logged so far
    """
    ptype = pred["type"]

    if ptype == "chapter_gte":
        return state["current_chapter"] >= pred["value"]

    if ptype == "entity_active":
        return pred["entity_id"] in state["active_entities"]

    if ptype == "entity_present":
        return pred["entity_id"] in state["present_entities"]

    if ptype == "event_occurred":
        return pred["event"] in state["events"]

    if ptype == "and":
        return all(evaluate_predicate(c, state) for c in pred["children"])

    if ptype == "or":
        return any(evaluate_predicate(c, state) for c in pred["children"])

    raise ValueError(f"Unknown predicate type: {ptype}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/planning/test_foreshadow_pool.py -v`
Expected: All 10 tests PASS (4 CRUD + 6 predicate)

- [ ] **Step 5: Commit**

```bash
git add app/planning/foreshadow_pool.py tests/planning/test_foreshadow_pool.py
git commit -m "feat(foreshadow): predicate evaluator for trigger conditions"
```

---

## Task 3: Foreshadow Pool — Prose Verification

**Files:**
- Modify: `app/planning/foreshadow_pool.py`
- Create: `prompts/critics/foreshadow_verify.j2`
- Modify: `tests/planning/test_foreshadow_pool.py`

- [ ] **Step 1: Create the verification prompt template**

Create `prompts/critics/foreshadow_verify.j2`:

```
Does the following prose contain a legible reference to this narrative element? A "legible reference" means a reader would recognize the element — not just a vague thematic echo, but a concrete mention, image, or event that clearly connects to the element described.

Element: "{{ element_text }}"

Prose (last 3 beats):
{{ prose }}

Answer YES or NO.
```

- [ ] **Step 2: Write failing test for verification function**

Append to `tests/planning/test_foreshadow_pool.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_verify_prose_contains_element():
    """Test the verification function structure (mocked LLM)."""
    from app.planning.foreshadow_pool import verify_prose_reference
    from unittest.mock import AsyncMock, MagicMock
    from app.runtime.client import ChatMessage

    mock_client = MagicMock()
    mock_client.chat_with_logprobs = AsyncMock()

    # Simulate high-confidence YES
    mock_logprob = MagicMock()
    mock_logprob.token = "YES"
    mock_logprob.logprob = -0.1  # ~0.90 probability
    mock_logprob.top_logprobs = {"YES": -0.1, "NO": -2.3}
    mock_result = MagicMock()
    mock_result.content = "YES"
    mock_result.token_logprobs = [mock_logprob]
    mock_client.chat_with_logprobs.return_value = mock_result

    confidence = await verify_prose_reference(
        client=mock_client,
        element_text="Tristan notices the pistol's unusual weight",
        prose="He hefted the pistol. It was heavier than it should have been.",
    )
    assert confidence > 0.8
    mock_client.chat_with_logprobs.assert_called_once()


@pytest.mark.asyncio
async def test_verify_prose_low_confidence():
    """Test low confidence when prose doesn't reference element."""
    from app.planning.foreshadow_pool import verify_prose_reference
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.chat_with_logprobs = AsyncMock()

    mock_logprob = MagicMock()
    mock_logprob.token = "NO"
    mock_logprob.logprob = -0.2
    mock_logprob.top_logprobs = {"YES": -3.0, "NO": -0.2}
    mock_result = MagicMock()
    mock_result.content = "NO"
    mock_result.token_logprobs = [mock_logprob]
    mock_client.chat_with_logprobs.return_value = mock_result

    confidence = await verify_prose_reference(
        client=mock_client,
        element_text="Tristan notices the pistol's unusual weight",
        prose="The sun was setting over the harbor.",
    )
    assert confidence < 0.3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/planning/test_foreshadow_pool.py::test_verify_prose_contains_element -v`
Expected: FAIL — cannot import `verify_prose_reference`

- [ ] **Step 4: Implement verification function**

Add to `app/planning/foreshadow_pool.py`:

```python
import math
from pathlib import Path

_VERIFY_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "critics" / "foreshadow_verify.j2"


async def verify_prose_reference(
    *, client: "InferenceClient", element_text: str, prose: str,
) -> float:
    """Check if prose contains a legible reference to a narrative element.

    Returns confidence score [0, 1] from logprob on the YES token.
    """
    from jinja2 import Template
    from app.runtime.client import ChatMessage

    template = Template(_VERIFY_PROMPT_PATH.read_text())
    prompt = template.render(element_text=element_text, prose=prose[-3000:])

    result = await client.chat_with_logprobs(
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=1,
        temperature=0.0,
        top_logprobs=5,
    )

    # Extract logprob for YES token
    if result.token_logprobs:
        top = result.token_logprobs[0].top_logprobs
        yes_logprob = top.get("YES", top.get("Yes", top.get("yes", -10.0)))
        no_logprob = top.get("NO", top.get("No", top.get("no", -10.0)))
        # Softmax over YES/NO
        yes_prob = math.exp(yes_logprob)
        no_prob = math.exp(no_logprob)
        total = yes_prob + no_prob
        return yes_prob / total if total > 0 else 0.0

    # Fallback: text match
    content = result.content.strip().upper()
    return 0.9 if content.startswith("YES") else 0.1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/planning/test_foreshadow_pool.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/planning/foreshadow_pool.py prompts/critics/foreshadow_verify.j2 tests/planning/test_foreshadow_pool.py
git commit -m "feat(foreshadow): prose-level verification via logprob confidence"
```

---

## Task 4: Foreshadow Pool — Pipeline Integration

**Files:**
- Modify: `app/planning/foreshadow_pool.py`
- Modify: `app/rollout/harness.py` (lines 227-330)

- [ ] **Step 1: Add scan_and_fire function to foreshadow_pool**

Add to `app/planning/foreshadow_pool.py`:

```python
async def scan_and_fire(
    *, sm: "WorldStateManager", client: "InferenceClient",
    current_chapter: int, active_entities: list[str],
    present_entities: list[str], events: list[str],
    prose_so_far: str,
) -> dict:
    """Before each beat: check triggers, fire payoffs, verify plants, escalate overdue.

    Returns dict with:
        triggered: list of triples whose triggers just fired
        overdue: list of triples past deadline
        unverified_plants: list of triples needing plant re-injection
    """
    state = {
        "current_chapter": current_chapter,
        "active_entities": active_entities,
        "present_entities": present_entities,
        "events": events,
    }

    result = {"triggered": [], "overdue": [], "unverified_plants": []}

    # Check planted triples for trigger firing
    planted = sm.list_foreshadow_triples(status="planted")
    for triple in planted:
        if evaluate_predicate(triple["trigger_pred"], state):
            sm.update_foreshadow_triple(triple["id"], status="triggered")
            result["triggered"].append(triple)

    # Check for unverified plants (verified_planted < 0.6 or None)
    for triple in planted:
        vp = triple.get("verified_planted")
        if vp is not None and vp < 0.6:
            result["unverified_plants"].append(triple)

    # Check overdue
    result["overdue"] = sm.list_overdue_foreshadow_triples(current_chapter)

    return result


async def verify_and_update(
    *, sm: "WorldStateManager", client: "InferenceClient",
    triple_id: str, field: str, element_text: str, prose: str,
) -> float:
    """After a beat: verify plant or payoff in prose, update confidence."""
    confidence = await verify_prose_reference(
        client=client, element_text=element_text, prose=prose,
    )
    sm.update_foreshadow_triple(triple_id, **{field: confidence})
    if field == "verified_payoff" and confidence >= 0.6:
        sm.update_foreshadow_triple(triple_id, status="paid_off")
    return confidence
```

- [ ] **Step 2: Wire into rollout harness**

In `app/rollout/harness.py`, add imports at the top (after existing imports around line 35):

```python
from app.planning.foreshadow_pool import scan_and_fire, verify_and_update
```

In the per-chapter loop (after line 274 where `pipeline.run()` is called), add foreshadow scanning and verification. The exact integration is after the `out = await pipeline.run(...)` call and before `main_sm.save_rollout_chapter(chapter)`:

```python
                # Foreshadow pool: scan triggers and verify plants
                try:
                    active_ids = [e.id for e in rollout_sm.list_entities()
                                  if e.status and e.status.value == "active"]
                    scene_entities = []  # populated from dramatic plan if available
                    kb_events = []  # populated from KB if available
                    pool_result = await scan_and_fire(
                        sm=main_sm, client=client,
                        current_chapter=ch_idx,
                        active_entities=active_ids,
                        present_entities=scene_entities,
                        events=kb_events,
                        prose_so_far=prose or "",
                    )
                    # Verify plant for any newly planted triples
                    newly_planted = main_sm.list_foreshadow_triples(status="planted")
                    for triple in newly_planted:
                        if triple["planted_chapter"] == ch_idx and triple["verified_planted"] is None:
                            await verify_and_update(
                                sm=main_sm, client=client,
                                triple_id=triple["id"],
                                field="verified_planted",
                                element_text=triple["foreshadow_text"],
                                prose=prose or "",
                            )
                    # Verify payoff for triggered triples
                    for triple in pool_result["triggered"]:
                        await verify_and_update(
                            sm=main_sm, client=client,
                            triple_id=triple["id"],
                            field="verified_payoff",
                            element_text=triple["payoff_text"],
                            prose=prose or "",
                        )
                except Exception:
                    pass  # Foreshadow pool is best-effort
```

- [ ] **Step 3: Run existing rollout tests to confirm no regression**

Run: `uv run pytest tests/rollout/ -v`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add app/planning/foreshadow_pool.py app/rollout/harness.py
git commit -m "feat(foreshadow): wire pool into rollout harness — scan, fire, verify"
```

---

## Task 5: Typed Edits — Schema & Taxonomy

**Files:**
- Modify: `app/world/db.py` (SCHEMA_SQL)
- Create: `app/engine/typed_edits.py`
- Create: `tests/engine/test_typed_edits.py`

- [ ] **Step 1: Write failing tests for edit application**

Create `tests/engine/test_typed_edits.py`:

```python
from __future__ import annotations
import pytest
from app.engine.typed_edits import apply_edits, EDIT_TYPES


def test_edit_types_taxonomy():
    assert "cliche" in EDIT_TYPES
    assert "forced_metaphor" in EDIT_TYPES
    assert "continuity_break" in EDIT_TYPES
    assert len(EDIT_TYPES) >= 10


def test_apply_single_edit():
    prose = "The odds were shifting beneath him like a gambler's last coin on the table."
    edits = [
        {
            "span_start": 0,
            "span_end": 71,
            "original_text": "The odds were shifting beneath him like a gambler's last coin on the",
            "edit_type": "forced_metaphor",
            "reason": "gambling family over budget",
            "replacement": "The ground was tilting beneath him like a floor giving way under the",
        }
    ]
    result = apply_edits(prose, edits)
    assert "ground was tilting" in result
    assert "odds were shifting" not in result


def test_apply_multiple_edits_reverse_order():
    prose = "AAA BBB CCC DDD EEE"
    edits = [
        {"span_start": 0, "span_end": 3, "original_text": "AAA", "edit_type": "cliche",
         "reason": "", "replacement": "XXX"},
        {"span_start": 8, "span_end": 11, "original_text": "CCC", "edit_type": "cliche",
         "reason": "", "replacement": "YYY"},
        {"span_start": 16, "span_end": 19, "original_text": "EEE", "edit_type": "cliche",
         "reason": "", "replacement": "ZZZ"},
    ]
    result = apply_edits(prose, edits)
    assert result == "XXX BBB YYY DDD ZZZ"


def test_apply_edits_empty_list():
    prose = "Unchanged prose."
    assert apply_edits(prose, []) == prose


def test_apply_edits_validates_original_text():
    prose = "The quick brown fox."
    edits = [
        {"span_start": 0, "span_end": 3, "original_text": "WRONG",
         "edit_type": "cliche", "reason": "", "replacement": "RIGHT"},
    ]
    # Should skip edits where original_text doesn't match the span
    result = apply_edits(prose, edits)
    assert result == prose  # unchanged — edit skipped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_typed_edits.py -v`
Expected: FAIL — cannot import `apply_edits`

- [ ] **Step 3: Add typed_edits table to schema**

In `app/world/db.py`, add to `SCHEMA_SQL`:

```sql
CREATE TABLE IF NOT EXISTS typed_edits (
    id TEXT PRIMARY KEY,
    trace_id TEXT,
    rollout_id TEXT,
    chapter_index INTEGER,
    edit_type TEXT NOT NULL,
    original_text TEXT NOT NULL,
    replacement TEXT NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    reason TEXT
);
```

- [ ] **Step 4: Implement typed_edits module**

Create `app/engine/typed_edits.py`:

```python
"""Typed edit-pass — span-level prose fixes with a fixed taxonomy.

Replaces open-ended revise loops for prose-quality issues. World-rule
violations still go through the full reviser; prose-quality problems
get surgical span replacements with named failure modes.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

EDIT_TYPES = frozenset({
    "cliche",
    "purple_prose",
    "forced_metaphor",
    "unnecessary_exposition",
    "abrupt_transition",
    "bland_dialogue",
    "weak_closure",
    "continuity_break",
    "character_voice_drift",
    "timeline_error",
    "entity_contradiction",
    "repetition",
})

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "stages" / "typed_edit" / "system.j2"
_USER_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "stages" / "typed_edit" / "user.j2"


def apply_edits(prose: str, edits: list[dict]) -> str:
    """Apply span-level edits in reverse position order.

    Skips any edit where original_text doesn't match the actual span
    content (safety against stale offsets).
    """
    # Sort by span_start descending so offsets don't shift
    sorted_edits = sorted(edits, key=lambda e: e["span_start"], reverse=True)
    result = prose
    for edit in sorted_edits:
        start = edit["span_start"]
        end = edit["span_end"]
        original = edit["original_text"]
        replacement = edit["replacement"]
        # Validate the span matches
        actual = result[start:end]
        if actual != original:
            continue  # skip — offsets are stale
        result = result[:start] + replacement + result[end:]
    return result


async def detect_edits(
    client: "InferenceClient", prose: str,
) -> list[dict]:
    """LLM call to detect span-level prose issues.

    Returns list of edit dicts with span_start, span_end, original_text,
    edit_type, reason, replacement.
    """
    from jinja2 import Template
    from app.runtime.client import ChatMessage

    system = Template(_SYSTEM_PROMPT_PATH.read_text()).render(
        edit_types=sorted(EDIT_TYPES),
    )
    user = Template(_USER_PROMPT_PATH.read_text()).render(prose=prose)

    raw = await client.chat(
        messages=[
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ],
        max_tokens=3000,
        temperature=0.2,
        thinking=False,
    )

    content = raw.strip()
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]

    try:
        parsed = json.loads(content.strip())
        edits = parsed.get("edits", parsed) if isinstance(parsed, dict) else parsed
        # Validate each edit has required fields and valid type
        valid = []
        for e in edits:
            if (isinstance(e, dict)
                    and "span_start" in e and "span_end" in e
                    and "original_text" in e and "replacement" in e
                    and e.get("edit_type", "") in EDIT_TYPES):
                valid.append(e)
        return valid
    except (json.JSONDecodeError, TypeError):
        return []


def persist_edits(
    conn, edits: list[dict], *,
    trace_id: str | None = None,
    rollout_id: str | None = None,
    chapter_index: int | None = None,
) -> None:
    """Save applied edits to the typed_edits audit table."""
    for edit in edits:
        conn.execute(
            "INSERT INTO typed_edits "
            "(id, trace_id, rollout_id, chapter_index, edit_type, "
            "original_text, replacement, span_start, span_end, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"te_{uuid.uuid4().hex[:8]}",
                trace_id, rollout_id, chapter_index,
                edit["edit_type"], edit["original_text"],
                edit["replacement"], edit["span_start"], edit["span_end"],
                edit.get("reason", ""),
            ),
        )
    conn.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_typed_edits.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/world/db.py app/engine/typed_edits.py tests/engine/test_typed_edits.py
git commit -m "feat(edits): typed edit taxonomy, apply function, detect skeleton"
```

---

## Task 6: Typed Edits — Prompts & Pipeline Integration

**Files:**
- Create: `prompts/stages/typed_edit/system.j2`
- Create: `prompts/stages/typed_edit/user.j2`
- Modify: `app/engine/pipeline.py` (lines 596-618)

- [ ] **Step 1: Create typed edit prompt templates**

Create `prompts/stages/typed_edit/system.j2`:

```
You are a prose editor. Identify specific spans in the text that have quality issues and provide surgical replacements.

For each issue, output:
- The exact start and end character offsets of the problematic span
- The original text at that span (copy it exactly)
- The edit type from this fixed list: {{ edit_types | join(", ") }}
- A brief reason
- A replacement string (same approximate length, different wording)

Rules:
- Only flag genuine issues. Do not flag competent prose.
- Each replacement must be the SAME edit type — don't fix a cliche by introducing purple prose.
- For forced_metaphor: the replacement must use a DIFFERENT imagery family (bodily, architectural, textile, spatial, weather, sensory).
- Keep replacements the same approximate length as the original.
- Maximum 8 edits per pass. Focus on the worst offenders.

Output JSON only:
```json
{"edits": [{"span_start": N, "span_end": N, "original_text": "...", "edit_type": "...", "reason": "...", "replacement": "..."}]}
```
```

Create `prompts/stages/typed_edit/user.j2`:

```
Identify prose quality issues in this text. Output span-level edits as JSON.

TEXT:
{{ prose }}
```

- [ ] **Step 2: Wire typed edits into the pipeline check→revise loop**

In `app/engine/pipeline.py`, modify the check→revise loop (around lines 596-618). After the existing loop exits, add a typed edit pass for prose-quality fixes:

Find this block (around line 614-618):

```python
        if check_out.has_critical:
            outcome = "flagged_qm"
        else:
            outcome = "committed"
```

Insert before it:

```python
        # Typed edit pass — surgical prose-quality fixes after the revise loop
        try:
            from app.engine.typed_edits import detect_edits, apply_edits, persist_edits
            edits = await detect_edits(self._client, prose)
            if edits:
                prose = apply_edits(prose, edits)
                persist_edits(
                    self._world._conn, edits,
                    trace_id=trace.trace_id,
                )
        except Exception:
            pass  # typed edits are best-effort
```

- [ ] **Step 3: Run existing pipeline tests to confirm no regression**

Run: `uv run pytest tests/engine/ -v -x`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add prompts/stages/typed_edit/system.j2 prompts/stages/typed_edit/user.j2 app/engine/pipeline.py
git commit -m "feat(edits): typed edit prompts + pipeline integration"
```

---

## Task 7: Per-Dim Independent Scoring

**Files:**
- Modify: `app/rollout/scorer.py` (lines 143-200)
- Modify: `tests/scoring/test_scorer.py` (or create new)

- [ ] **Step 1: Write failing test for independent per-dim scoring**

Create `tests/scoring/test_independent_scoring.py`:

```python
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.rollout.scorer import score_chapter_independent, COLLAPSED_DIMS


@pytest.mark.asyncio
async def test_score_chapter_independent_calls_per_dim():
    """Verify that independent scoring makes one call per dim."""
    mock_client = MagicMock()
    call_count = 0

    async def mock_chat_with_logprobs(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_logprob = MagicMock()
        mock_logprob.token = "7"
        mock_logprob.logprob = -0.5
        mock_logprob.top_logprobs = {str(i): -2.0 for i in range(1, 11)}
        mock_logprob.top_logprobs["7"] = -0.5
        result = MagicMock()
        result.content = "Analysis here.\nprose_execution score: 7"
        result.token_logprobs = [mock_logprob]
        return result

    mock_client.chat_with_logprobs = AsyncMock(side_effect=mock_chat_with_logprobs)

    scores = await score_chapter_independent(
        client=mock_client,
        chapter_text="Some prose here about Tristan walking.",
    )

    assert call_count == len(COLLAPSED_DIMS)
    assert set(scores.keys()) == set(COLLAPSED_DIMS)
    for dim, data in scores.items():
        assert "score" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_independent_scoring.py -v`
Expected: FAIL — cannot import `score_chapter_independent`

- [ ] **Step 3: Implement score_chapter_independent**

Add to `app/rollout/scorer.py` (after the existing `score_chapter` function, around line 200):

```python
async def score_chapter_independent(
    *, client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
    max_tokens: int = 400,
    top_logprobs: int = 20,
) -> dict[str, dict]:
    """Score a chapter with one LLM call per dimension (no anchoring).

    Each dim gets its own prompt containing only that dim's rubric.
    All dims run in parallel via asyncio.gather.
    """
    import asyncio
    use_dims = dims or list(COLLAPSED_DIMS)

    async def _score_one(dim: str) -> tuple[str, dict]:
        return dim, await _score_single_dim(
            client=client, chapter_text=chapter_text,
            dim=dim, max_tokens=max_tokens, top_logprobs=top_logprobs,
        )

    results = await asyncio.gather(*[_score_one(d) for d in use_dims])
    return dict(results)


async def _score_single_dim(
    *, client: InferenceClient, chapter_text: str,
    dim: str, max_tokens: int = 400, top_logprobs: int = 20,
) -> dict:
    """Score a single dimension in isolation."""
    from app.runtime.client import ChatMessage

    rubric = _load_rubric(dim)
    prompt = (
        f"{rubric}\n\n"
        f"CHAPTER:\n{chapter_text}\n\n"
        f"Write a 1-sentence observation, then rate this chapter on "
        f"{dim.replace('_', ' ')} (1-10).\n"
        f"Format: {dim} score: N"
    )

    result = await client.chat_with_logprobs(
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=max_tokens,
        temperature=0.3,
        top_logprobs=top_logprobs,
    )

    # Find the score token
    content = result.content or ""
    marker = f"{dim} score:"
    marker_pos = content.lower().find(marker.lower())

    if marker_pos >= 0 and result.token_logprobs:
        # Find the token logprob at/after the marker
        char_count = 0
        for i, tlp in enumerate(result.token_logprobs):
            char_count += len(tlp.token)
            if char_count >= marker_pos + len(marker):
                # Next non-whitespace token should be the score
                for j in range(i + 1, min(i + 4, len(result.token_logprobs))):
                    tok = result.token_logprobs[j].token.strip()
                    if tok.isdigit() and 1 <= int(tok) <= 10:
                        score_val = result.expected_score(j)
                        return {
                            "score": score_val[0],
                            "sampled": int(tok),
                            "confidence": score_val[1],
                        }
                break

    # Fallback: text parse
    import re
    match = re.search(rf"{dim}\s*score:\s*(\d+)", content, re.IGNORECASE)
    if match:
        sampled = int(match.group(1))
        return {"score": (sampled - 1) / 9, "sampled": sampled, "confidence": 0.0}

    return {"score": 0.5, "sampled": 5, "confidence": 0.0}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scoring/test_independent_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Update score_and_persist_chapter to use independent scoring**

In `app/rollout/scorer.py`, modify `score_and_persist_chapter` (around line 417) to call `score_chapter_independent` instead of `score_chapter`:

Find: `scores = await score_chapter(`
Replace with: `scores = await score_chapter_independent(`

- [ ] **Step 6: Audit swap-and-average usage**

Verify all pairwise comparison call sites use `compare_chapters_corrected` (not `compare_chapters`). Check:
- `app/refinement/framework.py` — the `_evaluate_deltas` or comparison call
- `app/refinement/selectors.py` — `SiblingOutscoredSelector`

If any call site uses `compare_chapters` directly, change it to `compare_chapters_corrected`.

- [ ] **Step 7: Commit**

```bash
git add app/rollout/scorer.py tests/scoring/test_independent_scoring.py
git commit -m "feat(scoring): per-dim independent scoring — no cross-dim anchoring"
```

---

## Task 8: Cross-Judge — M-Prometheus-14B

**Files:**
- Create: `app/scoring/cross_judge.py`
- Modify: `app/world/db.py` (SCHEMA_SQL)
- Modify: `app/rollout/harness.py`
- Create: `tests/scoring/test_cross_judge.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scoring/test_cross_judge.py`:

```python
from __future__ import annotations
import pytest
from app.scoring.cross_judge import JudgePair, compute_agreement


def test_judge_pair_perfect_agreement():
    pair = JudgePair(
        gemma_scores={"prose_execution": 0.8, "subtext": 0.7},
        prometheus_scores={"prose_execution": 0.8, "subtext": 0.7},
    )
    assert pair.agreement == pytest.approx(0.0)
    assert pair.self_preference_flag is False


def test_judge_pair_mild_disagreement():
    pair = JudgePair(
        gemma_scores={"prose_execution": 0.8, "subtext": 0.7},
        prometheus_scores={"prose_execution": 0.7, "subtext": 0.6},
    )
    assert pair.agreement == pytest.approx(0.1)
    assert pair.self_preference_flag is False


def test_judge_pair_self_preference_flag():
    pair = JudgePair(
        gemma_scores={"prose_execution": 0.9, "subtext": 0.9},
        prometheus_scores={"prose_execution": 0.6, "subtext": 0.5},
    )
    # disagreement > 0.15 (1.5 pts on 10-pt scale) on at least one dim
    assert pair.self_preference_flag is True


def test_compute_agreement():
    a = {"prose_execution": 0.8, "subtext": 0.7, "hook_quality": 0.6}
    b = {"prose_execution": 0.7, "subtext": 0.7, "hook_quality": 0.5}
    assert compute_agreement(a, b) == pytest.approx(1 / 15)  # mean of |0.1, 0, 0.1|
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/scoring/test_cross_judge.py -v`
Expected: FAIL — cannot import

- [ ] **Step 3: Add cross_judge_scores table**

In `app/world/db.py`, add to `SCHEMA_SQL`:

```sql
CREATE TABLE IF NOT EXISTS cross_judge_scores (
    rollout_id TEXT NOT NULL,
    chapter_index INTEGER NOT NULL,
    judge_model TEXT NOT NULL,
    dim TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL,
    PRIMARY KEY (rollout_id, chapter_index, judge_model, dim)
);
```

- [ ] **Step 4: Implement cross_judge module**

Create `app/scoring/cross_judge.py`:

```python
"""Cross-judge scoring — dual-model evaluation for self-preference detection.

Runs M-Prometheus-14B (on CPU, port 8083) alongside Gemma 4 (GPU, port 8082)
to score each chapter. Disagreement > 1.5 pts on any dim flags self-preference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.runtime.client import InferenceClient
from app.rollout.scorer import score_chapter_independent, COLLAPSED_DIMS


def compute_agreement(a: dict[str, float], b: dict[str, float]) -> float:
    """Mean absolute difference across shared dims."""
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    return sum(abs(a[d] - b[d]) for d in shared) / len(shared)


@dataclass
class JudgePair:
    gemma_scores: dict[str, float]
    prometheus_scores: dict[str, float]

    @property
    def agreement(self) -> float:
        return compute_agreement(self.gemma_scores, self.prometheus_scores)

    @property
    def self_preference_flag(self) -> bool:
        for dim in set(self.gemma_scores) & set(self.prometheus_scores):
            if abs(self.gemma_scores[dim] - self.prometheus_scores[dim]) > 0.15:
                return True
        return False


async def score_with_cross_judge(
    *, gemma_client: InferenceClient,
    prometheus_client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
) -> JudgePair:
    """Score a chapter with both judges in parallel."""
    import asyncio
    use_dims = dims or list(COLLAPSED_DIMS)

    gemma_task = score_chapter_independent(
        client=gemma_client, chapter_text=chapter_text, dims=use_dims,
    )
    prometheus_task = score_chapter_independent(
        client=prometheus_client, chapter_text=chapter_text, dims=use_dims,
    )

    gemma_raw, prometheus_raw = await asyncio.gather(gemma_task, prometheus_task)

    return JudgePair(
        gemma_scores={d: v["score"] for d, v in gemma_raw.items()},
        prometheus_scores={d: v["score"] for d, v in prometheus_raw.items()},
    )


def persist_judge_pair(
    conn, rollout_id: str, chapter_index: int, pair: JudgePair,
) -> None:
    """Save both judges' scores to cross_judge_scores table."""
    for model, scores in [("gemma", pair.gemma_scores), ("prometheus", pair.prometheus_scores)]:
        for dim, score in scores.items():
            conn.execute(
                "INSERT OR REPLACE INTO cross_judge_scores "
                "(rollout_id, chapter_index, judge_model, dim, score) "
                "VALUES (?, ?, ?, ?, ?)",
                (rollout_id, chapter_index, model, dim, score),
            )
    conn.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/scoring/test_cross_judge.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Wire into rollout harness**

In `app/rollout/harness.py`, after the existing `score_and_persist_chapter` call (around line 324), add:

```python
                # Cross-judge scoring (best-effort, needs Prometheus server)
                try:
                    from app.scoring.cross_judge import (
                        score_with_cross_judge, persist_judge_pair,
                    )
                    prometheus_client = InferenceClient(
                        base_url="http://127.0.0.1:8083",
                        timeout=120.0, retries=1,
                    )
                    pair = await score_with_cross_judge(
                        gemma_client=client,
                        prometheus_client=prometheus_client,
                        chapter_text=prose or "",
                    )
                    persist_judge_pair(
                        main_conn, rollout_id, ch_idx, pair,
                    )
                except Exception:
                    pass  # Cross-judge is best-effort
```

- [ ] **Step 7: Commit**

```bash
git add app/world/db.py app/scoring/cross_judge.py app/rollout/harness.py tests/scoring/test_cross_judge.py
git commit -m "feat(scoring): cross-judge with M-Prometheus-14B + self-preference detection"
```

---

## Task 9: Structural Metrics — Syntactic CR + MTLD

**Files:**
- Create: `app/scoring/structural_metrics.py`
- Create: `tests/scoring/test_structural_metrics.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add spacy dependency**

In `pyproject.toml`, add `"spacy>=3.7"` to the dependencies list. Then run:

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

- [ ] **Step 2: Write failing tests**

Create `tests/scoring/test_structural_metrics.py`:

```python
from __future__ import annotations
import pytest
from app.scoring.structural_metrics import (
    syntactic_compression_ratio, mtld, mtld_forward,
)


def test_syntactic_cr_repetitive():
    """Highly repetitive syntax should compress well (low ratio)."""
    prose = ". ".join(["She walked", "She talked", "She smiled", "She frowned"] * 10)
    cr = syntactic_compression_ratio(prose)
    assert cr < 0.40


def test_syntactic_cr_varied():
    """Varied syntax should not compress as well."""
    prose = (
        "The rain fell steadily. Under the awning, two men argued about "
        "the price of salt. 'It's robbery,' said the first, a short man "
        "with calloused hands. His companion, taller by a head and broader "
        "by a life of hauling nets, simply shrugged. What could you do? "
        "The caravans set the rates. The fishermen paid them."
    )
    cr = syntactic_compression_ratio(prose)
    assert cr > 0.35


def test_mtld_low_diversity():
    """Repeating the same words → low MTLD."""
    prose = " ".join(["the dog sat on the mat"] * 20)
    score = mtld(prose)
    assert score < 30


def test_mtld_high_diversity():
    """Rich vocabulary → high MTLD."""
    prose = (
        "Tristan navigated the labyrinthine corridors beneath the citadel. "
        "Every junction presented a bifurcation: left toward the armory's "
        "flickering braziers, right toward the subterranean cisterns where "
        "moisture beaded on ancient stonework. He chose instinctively, "
        "following the draft that carried the metallic tang of freshly "
        "forged iron, a scent as familiar as his own heartbeat."
    )
    score = mtld(prose)
    assert score > 50


def test_mtld_forward_basic():
    words = ["the", "dog", "sat", "on", "the", "mat", "the", "cat"]
    score = mtld_forward(words, ttr_threshold=0.72)
    assert score > 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/scoring/test_structural_metrics.py -v`
Expected: FAIL — cannot import

- [ ] **Step 4: Implement structural metrics**

Create `app/scoring/structural_metrics.py`:

```python
"""Non-LLM structural prose metrics — guardrails, not targets.

Three deterministic measurements:
- Syntactic compression ratio (POS-tag gzip)
- MTLD (length-robust lexical diversity)
- MAUVE (distributional divergence from reference corpus)

These are alarms. Never optimize against them directly.
"""
from __future__ import annotations

import gzip
from collections import Counter

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


def syntactic_compression_ratio(prose: str) -> float:
    """Gzip compression ratio of the POS-tag sequence.

    Low ratio = repetitive syntax ("She X'd. She Y'd.").
    Threshold: < 0.35 is syntactically templated.
    """
    nlp = _get_nlp()
    doc = nlp(prose)
    pos_seq = " ".join(tok.pos_ for tok in doc)
    raw = pos_seq.encode("utf-8")
    if not raw:
        return 1.0
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw)


def mtld_forward(words: list[str], ttr_threshold: float = 0.72) -> float:
    """One-direction MTLD pass."""
    factors = 0.0
    factor_length = 0
    types: set[str] = set()

    for word in words:
        types.add(word.lower())
        factor_length += 1
        ttr = len(types) / factor_length
        if ttr <= ttr_threshold:
            factors += 1
            types = set()
            factor_length = 0

    # Partial factor
    if factor_length > 0:
        ttr = len(types) / factor_length
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - ttr_threshold)

    return len(words) / factors if factors > 0 else float(len(words))


def mtld(prose: str, ttr_threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Length-robust type-token ratio. Returns the average of forward
    and backward passes. Higher = more diverse vocabulary.
    Threshold: < 50 suggests vocabulary collapse.
    """
    words = prose.split()
    if len(words) < 10:
        return 0.0
    forward = mtld_forward(words, ttr_threshold)
    backward = mtld_forward(words[::-1], ttr_threshold)
    return (forward + backward) / 2
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/scoring/test_structural_metrics.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml app/scoring/structural_metrics.py tests/scoring/test_structural_metrics.py
git commit -m "feat(metrics): syntactic compression ratio + MTLD lexical diversity"
```

---

## Task 10: Structural Metrics — MAUVE + Harness Integration

**Files:**
- Modify: `app/scoring/structural_metrics.py`
- Modify: `app/rollout/harness.py`
- Modify: `app/world/db.py` (additive migration for new columns)

- [ ] **Step 1: Add mauve-text dependency**

In `pyproject.toml`, add `"mauve-text>=0.4"` to dependencies. Run `uv sync`.

- [ ] **Step 2: Implement MAUVE wrapper**

Add to `app/scoring/structural_metrics.py`:

```python
def compute_mauve(
    generated_texts: list[str],
    reference_texts: list[str],
) -> float:
    """MAUVE score between generated and reference text distributions.

    Returns [0, 1]. < 0.7 suggests significant style drift.
    Needs ~20+ passages per side for stability.
    """
    try:
        import mauve as mauve_lib
        result = mauve_lib.compute_mauve(
            p_text=reference_texts,
            q_text=generated_texts,
            device_id=-1,  # CPU
            max_text_length=512,
            verbose=False,
        )
        return float(result.mauve)
    except Exception:
        return -1.0  # signal that computation failed
```

- [ ] **Step 3: Add columns via additive migration**

In `app/world/db.py`, add to `_apply_additive_migrations()`:

```python
_add_column(conn, "rollout_chapters", "syntactic_cr", "REAL")
_add_column(conn, "rollout_chapters", "mtld", "REAL")
_add_column(conn, "rollout_runs", "rollout_mauve", "REAL")
```

- [ ] **Step 4: Wire per-chapter metrics into rollout harness**

In `app/rollout/harness.py`, after `main_sm.save_rollout_chapter(chapter)` (around line 297), add:

```python
                # Structural metrics (best-effort, no LLM cost)
                try:
                    from app.scoring.structural_metrics import (
                        syntactic_compression_ratio, mtld,
                    )
                    scr = syntactic_compression_ratio(prose or "")
                    mtld_score = mtld(prose or "")
                    main_conn.execute(
                        "UPDATE rollout_chapters SET syntactic_cr = ?, mtld = ? "
                        "WHERE rollout_id = ? AND chapter_index = ?",
                        (scr, mtld_score, rollout_id, ch_idx),
                    )
                    main_conn.commit()
                except Exception:
                    pass
```

- [ ] **Step 5: Run existing tests to confirm no regression**

Run: `uv run pytest tests/rollout/ tests/scoring/ -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml app/world/db.py app/scoring/structural_metrics.py app/rollout/harness.py
git commit -m "feat(metrics): MAUVE + per-chapter structural metrics in harness"
```

---

## Task 11: Voice Drift Monitor (KL Divergence)

**Files:**
- Create: `app/scoring/voice_drift.py`
- Create: `tests/scoring/test_voice_drift.py`

- [ ] **Step 1: Write failing tests**

Create `tests/scoring/test_voice_drift.py`:

```python
from __future__ import annotations
import pytest
import numpy as np
from app.scoring.voice_drift import (
    function_word_distribution, kl_divergence, FUNCTION_WORDS,
)


def test_function_word_distribution_shape():
    text = "the dog sat on the mat and the cat"
    dist = function_word_distribution(text)
    assert dist.shape == (len(FUNCTION_WORDS),)
    assert dist.sum() == pytest.approx(1.0, abs=0.01)


def test_kl_identical_distributions():
    p = np.array([0.3, 0.3, 0.2, 0.2])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_kl_different_distributions():
    p = np.array([0.5, 0.3, 0.1, 0.1])
    q = np.array([0.1, 0.1, 0.4, 0.4])
    kl = kl_divergence(p, q)
    assert kl > 0.5  # substantially different


def test_same_text_zero_drift():
    text = "He walked to the door and she followed him through it."
    p = function_word_distribution(text)
    q = function_word_distribution(text)
    assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-6)


def test_different_style_detectable_drift():
    text_a = (
        "He was not sure if he could do it, but he had to try. "
        "She had told him that it would be difficult, and he "
        "believed her. They were in this together, after all."
    )
    text_b = (
        "Magnificent crystalline structures erupted skyward, "
        "iridescent perfection embodying transcendent beauty. "
        "Luminous ethereal phenomena cascaded downward perpetually."
    )
    p = function_word_distribution(text_a)
    q = function_word_distribution(text_b)
    kl = kl_divergence(p, q)
    assert kl > 0.1  # detectable drift
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/scoring/test_voice_drift.py -v`
Expected: FAIL — cannot import

- [ ] **Step 3: Implement voice drift module**

Create `app/scoring/voice_drift.py`:

```python
"""Rolling function-word KL drift monitor.

Function words (articles, prepositions, auxiliaries, pronouns) are the
most reliable authorship signal because they're topic-independent.
KL divergence from a baseline distribution detects voice drift.

This is diagnostic only — annotates, does not block or revise.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

FUNCTION_WORDS = [
    "the", "a", "an", "of", "in", "to", "and", "but", "or", "not",
    "he", "she", "it", "they", "his", "her", "its", "was", "were",
    "had", "has", "have", "is", "are", "been", "be", "would", "could",
    "should", "will", "can", "do", "did", "that", "this", "which",
    "who", "what", "when", "where", "how", "if", "than", "then",
    "so", "as", "at", "by", "for", "from", "on", "with", "into",
    "no", "nor", "yet", "just", "only", "very", "too", "also",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "any", "own", "same", "about", "after", "before",
    "between", "through", "during", "without", "again", "once",
]


def function_word_distribution(text: str) -> np.ndarray:
    """Compute normalized frequency distribution over function words."""
    tokens = text.lower().split()
    counts = Counter(tokens)
    raw = np.array([counts.get(w, 0) for w in FUNCTION_WORDS], dtype=float)
    total = raw.sum()
    if total == 0:
        return np.ones(len(FUNCTION_WORDS)) / len(FUNCTION_WORDS)
    return raw / total


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL(P || Q) with epsilon smoothing."""
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    # Renormalize after clipping
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def assess_drift(kl: float) -> str:
    """Interpret KL divergence as drift severity."""
    if kl < 0.05:
        return "stable"
    if kl < 0.15:
        return "mild_drift"
    return "voice_drift"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scoring/test_voice_drift.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Add drift columns and wire into harness**

In `app/world/db.py`, add to `_apply_additive_migrations()`:

```python
_add_column(conn, "rollout_chapters", "fw_kl_baseline", "REAL")
_add_column(conn, "rollout_chapters", "fw_kl_window", "REAL")
```

In `app/rollout/harness.py`, after the structural metrics block, add:

```python
                # Voice drift monitor (best-effort)
                try:
                    from app.scoring.voice_drift import (
                        function_word_distribution, kl_divergence,
                    )
                    current_dist = function_word_distribution(prose or "")
                    # Baseline: ch1 of this rollout
                    if ch_idx == 1:
                        # Store as the baseline (nothing to compare yet)
                        pass
                    elif completed:
                        ch1_prose = completed[0].prose or ""
                        baseline_dist = function_word_distribution(ch1_prose)
                        kl_base = kl_divergence(current_dist, baseline_dist)
                        # Sliding window: last 3 chapters
                        window_proses = [c.prose or "" for c in completed[-3:]]
                        window_text = " ".join(window_proses)
                        window_dist = function_word_distribution(window_text)
                        kl_window = kl_divergence(current_dist, window_dist)
                        main_conn.execute(
                            "UPDATE rollout_chapters SET fw_kl_baseline = ?, fw_kl_window = ? "
                            "WHERE rollout_id = ? AND chapter_index = ?",
                            (kl_base, kl_window, rollout_id, ch_idx),
                        )
                        main_conn.commit()
                except Exception:
                    pass
```

- [ ] **Step 6: Commit**

```bash
git add app/scoring/voice_drift.py app/world/db.py app/rollout/harness.py tests/scoring/test_voice_drift.py
git commit -m "feat(metrics): function-word KL drift monitor"
```

---

## Task 12: Final Integration Test

**Files:**
- No new files

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -x -q
```

Expected: All tests PASS with no regressions.

- [ ] **Step 2: Verify new table creation**

```bash
uv run python -c "
from app.world.db import open_db
conn = open_db('/tmp/test_schema.db')
tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
for t in ['foreshadow_triples', 'typed_edits', 'cross_judge_scores']:
    assert t in tables, f'Missing table: {t}'
print('All new tables present')
conn.close()
import os; os.unlink('/tmp/test_schema.db')
"
```

- [ ] **Step 3: Commit CLAUDE.md update**

Update CLAUDE.md to reflect the new modules and commit:

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with new structural integrity modules"
```

- [ ] **Step 4: Push and create PR**

```bash
git push origin worktree-serene-nibbling-marble
```
