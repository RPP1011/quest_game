# P9.1 — Craft Library Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stand up `app/craft/` — the narrative-theory knowledge base the story-generation harness will consult. Pure data + code. No model wiring in this plan.

**Architecture:** Pydantic schemas for `Structure`, `Tool`, `Example`, `StyleRegister`, and `Arc` (+ `ArcPhase`). A YAML-backed `CraftLibrary` loader that indexes everything by id and offers query methods. Seed corpus: 3 structures, 6 tools, 10 examples, 2 style registers. Arc-state utilities (`tension_target`, `tension_gap`, `advance_phase`). `recommend_tools` picks candidates given current arc state.

**Tech Stack:** adds `pyyaml>=6`. Python 3.11, pydantic v2.

---

## File Structure

**Created:**
- `app/craft/__init__.py` — public surface
- `app/craft/schemas.py` — pydantic models
- `app/craft/library.py` — `CraftLibrary` loader + queries
- `app/craft/arc.py` — arc-state utilities
- `app/craft/data/structures/three_act.yaml`
- `app/craft/data/structures/five_act_freytag.yaml`
- `app/craft/data/structures/kishotenketsu.yaml`
- `app/craft/data/tools/chekhovs_gun.yaml`
- `app/craft/data/tools/try_fail_cycle.yaml`
- `app/craft/data/tools/reversal.yaml`
- `app/craft/data/tools/scene_sequel.yaml`
- `app/craft/data/tools/midpoint_shift.yaml`
- `app/craft/data/tools/false_victory.yaml`
- `app/craft/data/examples/beats.yaml`
- `app/craft/data/styles/terse_military.yaml`
- `app/craft/data/styles/lyrical_fantasy.yaml`
- `tests/craft/__init__.py`
- `tests/craft/test_schemas.py`
- `tests/craft/test_library.py`
- `tests/craft/test_arc.py`
- `tests/craft/test_recommend.py`

**Modified:**
- `pyproject.toml` — add `pyyaml>=6`

---

## Task 1: Schemas

**Files:**
- Create: `app/craft/__init__.py` (empty for now)
- Create: `app/craft/schemas.py`
- Create: `tests/craft/__init__.py` (empty)
- Create: `tests/craft/test_schemas.py`

- [ ] **Step 1: Tests**

```python
# tests/craft/test_schemas.py
import pytest
from pydantic import ValidationError
from app.craft.schemas import (
    Arc, ArcPhase, Example, Structure, StyleRegister, Tool,
)


def test_arc_phase_roundtrip():
    p = ArcPhase(name="rising", position=1, tension_target=0.5,
                 expected_beats=["chekhovs_gun", "try_fail_cycle"],
                 description="Complications build.")
    assert p.tension_target == 0.5
    assert "chekhovs_gun" in p.expected_beats


def test_structure_validates_phase_ordering():
    s = Structure(
        id="three_act", name="Three-Act",
        description="Setup, confrontation, resolution.",
        scales=["chapter", "campaign"],
        phases=[
            ArcPhase(name="setup", position=0, tension_target=0.2, description="x"),
            ArcPhase(name="confrontation", position=1, tension_target=0.7, description="x"),
            ArcPhase(name="resolution", position=2, tension_target=0.4, description="x"),
        ],
        tension_curve=[(0.0, 0.1), (0.5, 0.7), (0.9, 0.9), (1.0, 0.3)],
    )
    assert len(s.phases) == 3
    # Phases must be sorted by position when indexed
    assert s.phases[0].position == 0
    assert s.phases[-1].position == 2


def test_structure_rejects_duplicate_positions():
    with pytest.raises(ValidationError):
        Structure(
            id="x", name="x", description="x", scales=["scene"],
            phases=[
                ArcPhase(name="a", position=0, tension_target=0.1, description="x"),
                ArcPhase(name="b", position=0, tension_target=0.2, description="x"),
            ],
            tension_curve=[(0.0, 0.0), (1.0, 1.0)],
        )


def test_tool_defaults():
    t = Tool(
        id="chekhovs_gun", name="Chekhov's Gun",
        category="foreshadowing",
        description="Plant an element early so its later use feels earned.",
        preconditions=["Scene has room to introduce an incidental detail."],
        signals=["A later payoff is needed but would feel unearned without prep."],
        anti_patterns=["Paying off something that was never planted."],
        example_ids=["ex_chekhov_coin"],
    )
    assert t.category == "foreshadowing"


def test_example_validates_scale():
    e = Example(
        id="ex_x", tool_ids=["reversal"], source="original",
        scale="scene",
        snippet="She smiled. The smile did not reach her eyes.",
        annotation="Sub-clause undercuts the visible gesture — a micro-reversal.",
    )
    assert e.scale == "scene"
    with pytest.raises(ValidationError):
        Example(id="y", tool_ids=["reversal"], source="original",
                scale="galactic", snippet="x", annotation="x")


def test_style_register_voice_samples_required():
    with pytest.raises(ValidationError):
        StyleRegister(id="x", name="x", description="x",
                      sentence_variance="medium", concrete_abstract_ratio=0.5,
                      interiority_depth="medium", pov_discipline="strict",
                      diction_register="formal", voice_samples=[])


def test_arc_minimal():
    a = Arc(id="main", name="The Ostland Dynasty", scale="campaign",
            structure_id="three_act")
    assert a.current_phase_index == 0
    assert a.phase_progress == 0.0
    assert a.tension_observed == []


def test_arc_records_tension():
    a = Arc(id="x", name="x", scale="chapter", structure_id="three_act",
            tension_observed=[(1, 0.3), (2, 0.4), (3, 0.45)])
    assert a.tension_observed[-1] == (3, 0.45)
```

- [ ] **Step 2: Run — ImportError**

Run: `uv run pytest tests/craft/test_schemas.py -v`

- [ ] **Step 3: Write `app/craft/schemas.py`**

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


Scale = Literal["scene", "chapter", "campaign", "saga"]
ToolCategory = Literal[
    "foreshadowing", "pacing", "reversal", "dialogue",
    "character", "structural", "tension", "revelation",
]


class ArcPhase(BaseModel):
    name: str
    position: int
    tension_target: float = Field(ge=0.0, le=1.0)
    expected_beats: list[str] = Field(default_factory=list)
    description: str


class Structure(BaseModel):
    id: str
    name: str
    description: str
    scales: list[Scale]
    phases: list[ArcPhase]
    tension_curve: list[tuple[float, float]]  # (position_0_1, tension_0_1)

    @model_validator(mode="after")
    def _validate(self) -> "Structure":
        positions = [p.position for p in self.phases]
        if len(positions) != len(set(positions)):
            raise ValueError("structure phases must have unique positions")
        self.phases = sorted(self.phases, key=lambda p: p.position)
        if self.tension_curve:
            xs = [x for x, _ in self.tension_curve]
            if xs != sorted(xs):
                raise ValueError("tension_curve must be sorted by position")
        return self


class Tool(BaseModel):
    id: str
    name: str
    category: ToolCategory
    description: str
    preconditions: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    example_ids: list[str] = Field(default_factory=list)


class Example(BaseModel):
    id: str
    tool_ids: list[str] = Field(min_length=1)
    source: str
    scale: Scale
    snippet: str
    annotation: str


class StyleRegister(BaseModel):
    id: str
    name: str
    description: str
    sentence_variance: Literal["low", "medium", "high"]
    concrete_abstract_ratio: float = Field(ge=0.0, le=1.0)
    interiority_depth: Literal["surface", "medium", "deep"]
    pov_discipline: Literal["strict", "moderate", "loose"]
    diction_register: str
    voice_samples: list[str] = Field(min_length=1)


class Arc(BaseModel):
    id: str
    name: str
    scale: Scale
    structure_id: str
    current_phase_index: int = 0
    phase_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    theme: str | None = None
    parent_arc_id: str | None = None
    child_arc_ids: list[str] = Field(default_factory=list)
    plot_thread_ids: list[str] = Field(default_factory=list)
    pivot_update_numbers: list[int] = Field(default_factory=list)
    tension_observed: list[tuple[int, float]] = Field(default_factory=list)
    required_beats_remaining: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run — all PASS**

Run: `uv run pytest tests/craft/test_schemas.py -v`

- [ ] **Step 5: Commit**

```bash
git add app/craft tests/craft
git commit -m "feat(craft): pydantic schemas for structures, tools, examples, arcs"
```

---

## Task 2: YAML loader + tests against fixtures

**Files:**
- Create: `app/craft/library.py`
- Create: `tests/craft/test_library.py`
- Modify: `pyproject.toml` — add `pyyaml>=6`

- [ ] **Step 1: Add `pyyaml>=6` to dependencies**

Edit `pyproject.toml` to include `"pyyaml>=6",` in the `dependencies` list. Run `uv sync`.

- [ ] **Step 2: Tests (using a tmp fixture dir, not the real data dir yet)**

```python
# tests/craft/test_library.py
from pathlib import Path
import textwrap
import pytest
from app.craft.library import CraftLibrary
from app.craft.schemas import Structure, Tool


@pytest.fixture
def tiny_library_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "structures").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)
    (root / "examples").mkdir(parents=True)
    (root / "styles").mkdir(parents=True)

    (root / "structures" / "tiny.yaml").write_text(textwrap.dedent("""
        id: tiny
        name: Tiny
        description: A minimal test structure.
        scales: [scene]
        phases:
          - name: open
            position: 0
            tension_target: 0.2
            expected_beats: [scene_sequel]
            description: Opening.
          - name: close
            position: 1
            tension_target: 0.8
            description: Close.
        tension_curve:
          - [0.0, 0.1]
          - [1.0, 0.8]
    """))

    (root / "tools" / "scene_sequel.yaml").write_text(textwrap.dedent("""
        id: scene_sequel
        name: Scene and Sequel
        category: pacing
        description: Scene then reaction.
        preconditions: [A just-resolved action]
        signals: [Reader needs to metabolize what just happened]
        anti_patterns: [Skipping the sequel entirely]
        example_ids: [ex_sq1]
    """))

    (root / "examples" / "beats.yaml").write_text(textwrap.dedent("""
        examples:
          - id: ex_sq1
            tool_ids: [scene_sequel]
            source: original
            scale: scene
            snippet: |
              The door shut behind him. For a long moment she simply stood,
              her hand still raised as if to call him back. Then she lowered it,
              and began, slowly, to clear the cups.
            annotation: Scene ends on action; sequel is the pause + the small task.
    """))

    (root / "styles" / "terse.yaml").write_text(textwrap.dedent("""
        id: terse
        name: Terse
        description: Dry, declarative, low interiority.
        sentence_variance: low
        concrete_abstract_ratio: 0.8
        interiority_depth: surface
        pov_discipline: strict
        diction_register: plain
        voice_samples:
          - |
            He walked. The fire was out. He counted the bullets.
    """))
    return root


def test_load_indexes_everything(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    assert isinstance(lib.structure("tiny"), Structure)
    assert isinstance(lib.tool("scene_sequel"), Tool)
    assert lib.example("ex_sq1").scale == "scene"
    assert lib.style("terse").sentence_variance == "low"


def test_getters_raise_for_missing(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    with pytest.raises(KeyError):
        lib.structure("nope")
    with pytest.raises(KeyError):
        lib.tool("nope")


def test_filter_by_scale(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    scene_structs = lib.structures(scale="scene")
    assert [s.id for s in scene_structs] == ["tiny"]
    assert lib.structures(scale="campaign") == []


def test_filter_tools_by_category(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    pacing = lib.tools(category="pacing")
    assert [t.id for t in pacing] == ["scene_sequel"]
    assert lib.tools(category="reversal") == []


def test_examples_for_tool(tiny_library_root: Path):
    lib = CraftLibrary(tiny_library_root)
    examples = lib.examples_for_tool("scene_sequel")
    assert [e.id for e in examples] == ["ex_sq1"]
    assert lib.examples_for_tool("unknown") == []


def test_load_rejects_duplicate_ids(tiny_library_root: Path):
    (tiny_library_root / "tools" / "dup.yaml").write_text(
        "id: scene_sequel\nname: Dup\ncategory: pacing\ndescription: x\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        CraftLibrary(tiny_library_root)
```

- [ ] **Step 3: Run — ImportError**

- [ ] **Step 4: Implement `app/craft/library.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .schemas import Example, Scale, Structure, StyleRegister, Tool, ToolCategory


class CraftLibrary:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._structures: dict[str, Structure] = {}
        self._tools: dict[str, Tool] = {}
        self._examples: dict[str, Example] = {}
        self._styles: dict[str, StyleRegister] = {}
        self._load()

    def _load(self) -> None:
        self._structures = self._load_dir("structures", Structure)
        self._tools = self._load_dir("tools", Tool)
        self._styles = self._load_dir("styles", StyleRegister)
        self._examples = self._load_examples()

    def _load_dir(self, subdir: str, model) -> dict:
        out: dict[str, Any] = {}
        d = self._root / subdir
        if not d.is_dir():
            return out
        for path in sorted(d.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text())
            if raw is None:
                continue
            obj = model.model_validate(raw)
            if obj.id in out:
                raise ValueError(
                    f"duplicate id {obj.id!r} in {subdir}: {path} and earlier file"
                )
            out[obj.id] = obj
        return out

    def _load_examples(self) -> dict[str, Example]:
        out: dict[str, Example] = {}
        d = self._root / "examples"
        if not d.is_dir():
            return out
        for path in sorted(d.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text())
            if not raw:
                continue
            items = raw.get("examples") if isinstance(raw, dict) else raw
            for item in items or []:
                obj = Example.model_validate(item)
                if obj.id in out:
                    raise ValueError(f"duplicate example id {obj.id!r}")
                out[obj.id] = obj
        return out

    # ---- getters ----

    def structure(self, id: str) -> Structure:
        if id not in self._structures:
            raise KeyError(id)
        return self._structures[id]

    def tool(self, id: str) -> Tool:
        if id not in self._tools:
            raise KeyError(id)
        return self._tools[id]

    def example(self, id: str) -> Example:
        if id not in self._examples:
            raise KeyError(id)
        return self._examples[id]

    def style(self, id: str) -> StyleRegister:
        if id not in self._styles:
            raise KeyError(id)
        return self._styles[id]

    # ---- queries ----

    def structures(self, scale: Scale | None = None) -> list[Structure]:
        values = list(self._structures.values())
        if scale is None:
            return values
        return [s for s in values if scale in s.scales]

    def tools(self, category: ToolCategory | None = None) -> list[Tool]:
        values = list(self._tools.values())
        if category is None:
            return values
        return [t for t in values if t.category == category]

    def examples_for_tool(self, tool_id: str) -> list[Example]:
        return [e for e in self._examples.values() if tool_id in e.tool_ids]

    def all_structures(self) -> list[Structure]:
        return list(self._structures.values())

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def all_examples(self) -> list[Example]:
        return list(self._examples.values())

    def all_styles(self) -> list[StyleRegister]:
        return list(self._styles.values())
```

- [ ] **Step 5: Run — PASS**

- [ ] **Step 6: Commit**

```bash
git add app/craft/library.py tests/craft/test_library.py pyproject.toml uv.lock
git commit -m "feat(craft): YAML-backed CraftLibrary with indexed queries"
```

---

## Task 3: Seed three structures

**Files:**
- Create: `app/craft/data/structures/three_act.yaml`
- Create: `app/craft/data/structures/five_act_freytag.yaml`
- Create: `app/craft/data/structures/kishotenketsu.yaml`

- [ ] **Step 1: Write `three_act.yaml`**

```yaml
id: three_act
name: Three-Act Structure
description: >
  The dominant Western dramatic form. Setup establishes character, world, and
  desire; confrontation escalates obstacles to a crisis at or past the midpoint;
  resolution collapses the tension via climax and denouement. Strong default
  for most quests.
scales: [scene, chapter, campaign, saga]
phases:
  - name: setup
    position: 0
    tension_target: 0.2
    expected_beats: [chekhovs_gun, scene_sequel]
    description: >
      Establish normal. Plant the elements that will later pay off. Introduce
      the protagonist's want.
  - name: rising
    position: 1
    tension_target: 0.55
    expected_beats: [try_fail_cycle, chekhovs_gun]
    description: >
      Obstacles escalate. The protagonist tries, fails, adapts. Complications
      accumulate and force commitment.
  - name: midpoint
    position: 2
    tension_target: 0.7
    expected_beats: [midpoint_shift, reversal]
    description: >
      A shift in knowledge, status, or goal that raises stakes and changes
      how the protagonist pursues the want.
  - name: crisis
    position: 3
    tension_target: 0.85
    expected_beats: [false_victory, reversal, try_fail_cycle]
    description: >
      The worst complications. A false victory or dark-night-of-the-soul beat.
      The protagonist must choose what they truly value.
  - name: resolution
    position: 4
    tension_target: 0.3
    expected_beats: [scene_sequel]
    description: >
      Climax resolves the tension. Denouement metabolizes consequences.
tension_curve:
  - [0.00, 0.10]
  - [0.20, 0.25]
  - [0.45, 0.55]
  - [0.55, 0.70]
  - [0.80, 0.90]
  - [0.92, 0.95]
  - [1.00, 0.25]
```

- [ ] **Step 2: Write `five_act_freytag.yaml`**

```yaml
id: five_act_freytag
name: Freytag's Pyramid (Five-Act)
description: >
  Classical dramatic structure from Gustav Freytag. Exposition, rising action,
  climax at the midpoint, falling action, denouement. Ends in tragedy or
  rebirth depending on the chosen climax inversion. Useful for quests with a
  high-stakes central reversal.
scales: [chapter, campaign, saga]
phases:
  - name: exposition
    position: 0
    tension_target: 0.2
    expected_beats: [chekhovs_gun]
    description: Establish world, characters, and inciting stakes.
  - name: rising_action
    position: 1
    tension_target: 0.5
    expected_beats: [try_fail_cycle, chekhovs_gun]
    description: Complications and obstacles escalate toward the climax.
  - name: climax
    position: 2
    tension_target: 0.95
    expected_beats: [reversal, midpoint_shift]
    description: >
      The peak. A reversal of fortune, for better or worse, that determines
      the trajectory of the rest of the story.
  - name: falling_action
    position: 3
    tension_target: 0.55
    expected_beats: [try_fail_cycle, false_victory]
    description: Consequences of the climax unfold; the outcome narrows.
  - name: denouement
    position: 4
    tension_target: 0.2
    expected_beats: [scene_sequel]
    description: Final state is set. Unresolved threads either settle or remain as scars.
tension_curve:
  - [0.00, 0.10]
  - [0.20, 0.35]
  - [0.45, 0.80]
  - [0.50, 0.95]
  - [0.70, 0.55]
  - [1.00, 0.20]
```

- [ ] **Step 3: Write `kishotenketsu.yaml`**

```yaml
id: kishotenketsu
name: Kishōtenketsu
description: >
  East Asian four-part structure that does not rely on overt conflict.
  Introduction, development, twist (a new element that reframes the previous
  two), and reconciliation (the new frame integrates with the old). Useful
  for slice-of-life, contemplative, or mystery-lite quests where a conflict-
  driven pyramid would feel forced.
scales: [scene, chapter]
phases:
  - name: ki
    position: 0
    tension_target: 0.2
    expected_beats: [scene_sequel, chekhovs_gun]
    description: Introduction. Establish the situation and its texture.
  - name: sho
    position: 1
    tension_target: 0.35
    expected_beats: [scene_sequel]
    description: Development. Deepen the situation without introducing conflict.
  - name: ten
    position: 2
    tension_target: 0.75
    expected_beats: [reversal, midpoint_shift]
    description: >
      The twist. An element arrives that is not a conflict but a new lens —
      something that changes what the prior beats meant.
  - name: ketsu
    position: 3
    tension_target: 0.4
    expected_beats: [scene_sequel]
    description: Reconciliation. The new element and prior situation integrate into a single frame.
tension_curve:
  - [0.00, 0.15]
  - [0.25, 0.25]
  - [0.55, 0.35]
  - [0.65, 0.75]
  - [0.85, 0.5]
  - [1.00, 0.35]
```

- [ ] **Step 4: Load + smoke-test**

Add to `tests/craft/test_library.py`:

```python
def test_real_structures_load():
    from app.craft.library import CraftLibrary
    lib = CraftLibrary(Path(__file__).parent.parent.parent / "app" / "craft" / "data")
    ids = {s.id for s in lib.structures()}
    assert {"three_act", "five_act_freytag", "kishotenketsu"} <= ids
    three = lib.structure("three_act")
    assert len(three.phases) == 5
```

- [ ] **Step 5: Run — PASS**

- [ ] **Step 6: Commit**

```bash
git add app/craft/data/structures tests/craft/test_library.py
git commit -m "feat(craft): seed structures (three-act, five-act, kishotenketsu)"
```

---

## Task 4: Seed six tools

**Files:**
- Create: `app/craft/data/tools/chekhovs_gun.yaml`
- Create: `app/craft/data/tools/try_fail_cycle.yaml`
- Create: `app/craft/data/tools/reversal.yaml`
- Create: `app/craft/data/tools/scene_sequel.yaml`
- Create: `app/craft/data/tools/midpoint_shift.yaml`
- Create: `app/craft/data/tools/false_victory.yaml`

- [ ] **Step 1: `chekhovs_gun.yaml`**

```yaml
id: chekhovs_gun
name: Chekhov's Gun
category: foreshadowing
description: >
  Any element introduced early with enough salience that the reader expects
  it to return, so that when it does return the payoff feels earned rather
  than contrived. The plant must look incidental but not invisible.
preconditions:
  - A later beat requires an element that would feel arbitrary if introduced fresh
  - The current scene has room to dwell on a detail without stalling the forward motion
signals:
  - Planner knows a payoff is coming in the next 2-6 updates
  - Current scene naturally brings the element into view
anti_patterns:
  - Paying off something that was never planted (deus ex machina)
  - Planting something with too much emphasis (broadcasts the payoff)
  - Planting and never paying off (dead weight)
example_ids:
  - ex_chekhov_coin
  - ex_chekhov_letter
```

- [ ] **Step 2: `try_fail_cycle.yaml`**

```yaml
id: try_fail_cycle
name: Try / Fail / Adapt
category: pacing
description: >
  The protagonist attempts something, fails in a way that reveals new
  information or raises stakes, then adapts. A single cycle is a beat; two
  or three strung together form rising action. Each failure should teach —
  either the protagonist, the reader, or both.
preconditions:
  - The protagonist has a clear immediate goal
  - A straightforward success would skip a phase of the arc
signals:
  - Tension is below target and needs escalation via failure
  - The protagonist's approach should mature over the next few beats
anti_patterns:
  - Failure that teaches nothing (pure friction)
  - Success on the first try when stakes require earning
  - Repeating the same failure shape three times in a row (monotonous)
example_ids:
  - ex_try_fail_siege
```

- [ ] **Step 3: `reversal.yaml`**

```yaml
id: reversal
name: Reversal
category: reversal
description: >
  A moment where the situation inverts: an apparent friend is an enemy, a
  failure was actually the goal, a triumph is revealed as hollow. Reversals
  should recontextualize what came before rather than invalidate it.
preconditions:
  - Prior beats supplied enough information that the reversal is recoverable on re-read
  - The reader has a stable reading of the situation that can be overturned
signals:
  - Tension is flat and needs shock
  - A planted irony is ready to collapse
  - The protagonist's model of the world is due to be broken
anti_patterns:
  - Reversal that invalidates the prior scenes rather than reframing them
  - Reversal with no planted clues (feels arbitrary)
  - Too many reversals in succession (diminishing returns)
example_ids:
  - ex_reversal_ally
```

- [ ] **Step 4: `scene_sequel.yaml`**

```yaml
id: scene_sequel
name: Scene and Sequel
category: pacing
description: >
  A scene (action with a goal) is followed by a sequel (reaction, dilemma,
  decision). The sequel lets the reader metabolize and the protagonist
  recommit. Omitting sequels produces plot without story; omitting scenes
  produces rumination without forward motion.
preconditions:
  - A scene has just resolved with outcome (success, failure, or mixed)
signals:
  - Reader needs a beat to absorb what just happened
  - Protagonist's emotional state is about to shift
  - Next scene's goal needs to be recommitted in light of the outcome
anti_patterns:
  - Skipping the sequel entirely (reader bounces)
  - A sequel that doesn't reach a decision (stalls)
  - Sequel becomes exposition (kills momentum)
example_ids:
  - ex_sequel_door
```

- [ ] **Step 5: `midpoint_shift.yaml`**

```yaml
id: midpoint_shift
name: Midpoint Shift
category: structural
description: >
  Around the center of the arc, something changes in the protagonist's
  understanding or situation that reframes the second half. A revelation,
  a lost ally, an upgrade of the threat. Distinguishes stories with two
  halves from stories that merely continue.
preconditions:
  - The first half of the arc has established a clear situation
  - A payoff for a planted irony is available
signals:
  - Arc progress is 0.45-0.55
  - Tension has been climbing but needs a new axis
anti_patterns:
  - Shift that doesn't actually change the pursuit (cosmetic)
  - Shift too early (no first half to reframe)
  - Shift that requires external info not planted earlier
example_ids:
  - ex_midpoint_mirror
```

- [ ] **Step 6: `false_victory.yaml`**

```yaml
id: false_victory
name: False Victory
category: tension
description: >
  An apparent win that the reader can tell is about to unravel. Often placed
  at ~70-85% progress, immediately before the lowest low. Lets the reader
  feel the contrast when the unraveling hits.
preconditions:
  - The protagonist has pursued a visible goal
  - There is a planted condition the victory will violate
signals:
  - Arc progress 0.65-0.80
  - A planted Chekhov's gun is primed to fire against the win
anti_patterns:
  - Victory that actually sticks (deflates expectations)
  - Unraveling with no prior planted condition (feels cheap)
  - Triumph without any sense of cost (reader doesn't care)
example_ids:
  - ex_false_victory_crown
```

- [ ] **Step 7: Load smoke-test**

Add to `tests/craft/test_library.py`:

```python
def test_real_tools_load():
    from app.craft.library import CraftLibrary
    lib = CraftLibrary(Path(__file__).parent.parent.parent / "app" / "craft" / "data")
    ids = {t.id for t in lib.tools()}
    expected = {"chekhovs_gun", "try_fail_cycle", "reversal",
                "scene_sequel", "midpoint_shift", "false_victory"}
    assert expected <= ids
    # Categories distributed
    cats = {t.category for t in lib.tools()}
    assert {"foreshadowing", "pacing", "reversal", "structural", "tension"} <= cats
```

- [ ] **Step 8: Run — PASS**

- [ ] **Step 9: Commit**

```bash
git add app/craft/data/tools tests/craft/test_library.py
git commit -m "feat(craft): seed six literary tools"
```

---

## Task 5: Seed ten examples

**Files:**
- Create: `app/craft/data/examples/beats.yaml`

Each example is an original short snippet (no copyright concerns) annotated with what tool it demonstrates.

- [ ] **Step 1: Write `beats.yaml`**

```yaml
examples:

  - id: ex_chekhov_coin
    tool_ids: [chekhovs_gun]
    source: original
    scale: scene
    snippet: |
      The stranger paid in a coin the innkeeper had never seen before. He
      turned it in the lamplight, frowned, dropped it into the jar with the
      others, and poured another cup. It was only later, when the tax collector
      came demanding the king's silver and the jar was opened before him, that
      the coin's face caught the light a second time and was recognized.
    annotation: >
      The coin is planted in an incidental action (paid in, frowned at,
      dropped), with just enough salience that the reader expects it to
      return. The payoff arrives several beats later, triggered by a
      second framing — the tax collector's presence — rather than by the
      narrator announcing the coin's importance.

  - id: ex_chekhov_letter
    tool_ids: [chekhovs_gun]
    source: original
    scale: chapter
    snippet: |
      She put the letter, unopened, in the drawer with the winter gloves,
      and did not think of it again until March.
    annotation: >
      Plant and payoff in one sentence. The reader is given two things —
      the letter and March — and the deferred opening sets up both a
      later scene and the passage of time between.

  - id: ex_try_fail_siege
    tool_ids: [try_fail_cycle]
    source: original
    scale: chapter
    snippet: |
      The first assault broke on the gatehouse; they left a hundred dead
      in the ditch. The second assault scaled the eastern wall before the
      oil came down; they left two hundred. For the third, he would not
      order another assault. He would wait, and dig, and let hunger do
      the work that steel had not.
    annotation: >
      Three tries compressed into a paragraph. Each failure teaches the
      commander something and raises the cost; the third beat commits to
      a qualitatively different approach (siege discipline rather than
      another charge). Rising action without monotony.

  - id: ex_reversal_ally
    tool_ids: [reversal]
    source: original
    scale: scene
    snippet: |
      "You've been a friend to me, Tomas," he said.
      Tomas smiled. "A friend, yes," he said. "To you, and to others."
      The silence, when it came, was very small, and very complete.
    annotation: >
      Reversal by qualification. The response does not contradict the
      premise — Tomas is a friend — but the addition "and to others"
      reframes the entire relationship. The silence is the scene's last
      beat, doing the work that exposition would have ruined.

  - id: ex_sequel_door
    tool_ids: [scene_sequel]
    source: original
    scale: scene
    snippet: |
      The door shut behind him. For a long moment she simply stood, her hand
      still raised as if to call him back. Then she lowered it, and began,
      slowly, to clear the cups.
    annotation: >
      A scene ends on an action (he leaves). The sequel is the pause,
      the raised hand, the small domestic task. No dialogue, no
      interiority spelled out, but the reader feels the decision she
      has just made to let him go.

  - id: ex_midpoint_mirror
    tool_ids: [midpoint_shift]
    source: original
    scale: chapter
    snippet: |
      It was while she was washing the blood from her hands that she saw
      her own face in the water of the basin, and understood, with a cold
      clarity that no one had given her, that she was going to finish what
      he had started.
    annotation: >
      Midpoint shift via self-recognition. The protagonist's goal inverts
      from vengeance to continuation of the victim's work. The shift is
      internal, delivered through a mirror-image (her face in the water),
      and reframes everything after it.

  - id: ex_false_victory_crown
    tool_ids: [false_victory]
    source: original
    scale: chapter
    snippet: |
      They crowned him on the steps, because there was no cathedral left.
      The horns sounded, the banners came down, and for half an hour he
      believed he had won. Then the runner from the east arrived, and the
      smile he wore for the crowd did not reach the throne room.
    annotation: >
      Win is presented with full ceremony. The "for half an hour" marks
      the reader's signal that the victory is bounded. The arrival of
      the runner is the pivot; what the reader did not yet know will
      now invalidate the preceding ceremony.

  - id: ex_sequel_cold_room
    tool_ids: [scene_sequel]
    source: original
    scale: scene
    snippet: |
      When the others had gone she sat alone in the cold room, and found
      that she could not remember what she had meant to say to him,
      or whether she had meant to say anything at all.
    annotation: >
      Pure sequel. No action, no decision yet — just the emptying-out
      of the aftermath. Often follows a charged confrontation scene;
      earns the next scene's decision by letting this one breathe.

  - id: ex_reversal_fake_map
    tool_ids: [reversal, chekhovs_gun]
    source: original
    scale: chapter
    snippet: |
      The map he had carried all summer, that he had consulted at every
      crossroads, was a forgery; and the man who had given it to him,
      the old priest at the inn, was laughing at an empty road.
    annotation: >
      Reversal compounded with a planted Chekhov's gun (the map and the
      priest were both present from the opening chapters). The reader
      recontextualizes every prior navigation beat.

  - id: ex_try_fail_diplomat
    tool_ids: [try_fail_cycle]
    source: original
    scale: scene
    snippet: |
      She tried courtesy first; the Duke yawned. She tried appeal to his
      vanity; the Duke asked for wine. She tried, finally, to threaten,
      and the Duke, who had been bored, became interested.
    annotation: >
      Three tactical tries within a single scene. Each fails in a
      specific, characterizing way, and the final success is
      ambiguous — the Duke becomes "interested" rather than compliant,
      which sets up a new problem.
```

- [ ] **Step 2: Load smoke-test**

Add to `tests/craft/test_library.py`:

```python
def test_real_examples_load_and_link_tools():
    from app.craft.library import CraftLibrary
    lib = CraftLibrary(Path(__file__).parent.parent.parent / "app" / "craft" / "data")
    assert len(lib.all_examples()) >= 10
    chek = lib.examples_for_tool("chekhovs_gun")
    assert len(chek) >= 2
    # Every example references only real tool ids
    tool_ids = {t.id for t in lib.tools()}
    for ex in lib.all_examples():
        for tid in ex.tool_ids:
            assert tid in tool_ids, f"example {ex.id} references unknown tool {tid}"
```

- [ ] **Step 3: Run — PASS**

- [ ] **Step 4: Commit**

```bash
git add app/craft/data/examples tests/craft/test_library.py
git commit -m "feat(craft): seed ten labeled beat examples"
```

---

## Task 6: Seed two style registers

**Files:**
- Create: `app/craft/data/styles/terse_military.yaml`
- Create: `app/craft/data/styles/lyrical_fantasy.yaml`

- [ ] **Step 1: `terse_military.yaml`**

```yaml
id: terse_military
name: Terse Military
description: >
  Declarative, short-to-medium sentences; concrete physical detail; low
  interiority; POV stays tight. Think professional-soldier voice. Avoids
  abstract emotion words; shows affect through action and observation.
sentence_variance: low
concrete_abstract_ratio: 0.85
interiority_depth: surface
pov_discipline: strict
diction_register: plain
voice_samples:
  - |
    He counted the rounds. Twelve. He counted them again. Still twelve.
    Outside the wind moved through the broken window.
  - |
    The sergeant did not answer. He put the radio down, very carefully,
    and looked at the map. Nothing on the map had changed.
  - |
    She took the jacket off, folded it, set it on the cot, and went to
    find the captain. She did not hurry. She did not slow.
```

- [ ] **Step 2: `lyrical_fantasy.yaml`**

```yaml
id: lyrical_fantasy
name: Lyrical Fantasy
description: >
  Variable sentence length, some long breathed periods punctuated by short
  declarative beats; concrete sensory detail foregrounded; moderate
  interiority rendered through metaphor; diction registers slightly
  archaic but not florid.
sentence_variance: high
concrete_abstract_ratio: 0.6
interiority_depth: medium
pov_discipline: moderate
diction_register: slightly_archaic
voice_samples:
  - |
    There are roads in that country, old roads, older than the stones of
    the king's hall, and she had walked them all. None of them went home.
  - |
    The wind moved in the grass, and in the moving was a whisper of some
    older language; the shepherd, hearing it, crossed himself in the new
    way and continued up the hill.
  - |
    She carried the lamp before her, and the shadow the lamp cast was
    smaller than her own, which she took, for no reason she could name,
    as a kindness.
```

- [ ] **Step 3: Load smoke-test**

Add to `tests/craft/test_library.py`:

```python
def test_real_styles_load():
    from app.craft.library import CraftLibrary
    lib = CraftLibrary(Path(__file__).parent.parent.parent / "app" / "craft" / "data")
    ids = {s.id for s in lib.all_styles()}
    assert {"terse_military", "lyrical_fantasy"} <= ids
    terse = lib.style("terse_military")
    assert terse.sentence_variance == "low"
    assert terse.interiority_depth == "surface"
    assert len(terse.voice_samples) >= 2
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add app/craft/data/styles tests/craft/test_library.py
git commit -m "feat(craft): seed two style registers (terse-military, lyrical-fantasy)"
```

---

## Task 7: Arc utilities

**Files:**
- Create: `app/craft/arc.py`
- Create: `tests/craft/test_arc.py`

Design: pure functions on `Arc` + `Structure`. `tension_target(arc, structure)` linearly interpolates the structure's tension_curve at the arc's current global position. `tension_gap(arc, structure)` returns target minus the average of the last N observed readings. `advance_phase(arc)` returns a new Arc with `current_phase_index += 1` and `phase_progress = 0.0`. `global_progress(arc, structure)` returns a 0..1 value over the entire arc.

- [ ] **Step 1: Tests**

```python
# tests/craft/test_arc.py
import pytest
from app.craft.arc import (
    advance_phase, global_progress, tension_gap, tension_target,
)
from app.craft.schemas import Arc, ArcPhase, Structure


@pytest.fixture
def s():
    return Structure(
        id="s", name="s", description="x", scales=["scene"],
        phases=[
            ArcPhase(name="a", position=0, tension_target=0.2, description="x"),
            ArcPhase(name="b", position=1, tension_target=0.7, description="x"),
            ArcPhase(name="c", position=2, tension_target=0.3, description="x"),
        ],
        tension_curve=[(0.0, 0.1), (0.5, 0.7), (1.0, 0.2)],
    )


def test_global_progress_inside_first_phase(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=0, phase_progress=0.5)
    # 3 phases, so each is 1/3. First phase midpoint = 1/6.
    assert abs(global_progress(a, s) - (1 / 6)) < 1e-6


def test_global_progress_at_phase_boundary(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0)
    assert abs(global_progress(a, s) - (1 / 3)) < 1e-6


def test_tension_target_interpolates(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0)
    # global_progress = 1/3 ≈ 0.333, curve from (0.0, 0.1) to (0.5, 0.7)
    # interpolated ≈ 0.1 + (0.333/0.5)*(0.7-0.1) ≈ 0.1 + 0.4 = 0.5
    assert abs(tension_target(a, s) - 0.5) < 0.02


def test_tension_gap_uses_recent_observations(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0,
            tension_observed=[(1, 0.2), (2, 0.25), (3, 0.3)])
    # target ≈ 0.5, avg of last 3 observed = 0.25, gap = 0.25
    gap = tension_gap(a, s, window=3)
    assert abs(gap - 0.25) < 0.02


def test_tension_gap_no_observations_returns_target(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0)
    gap = tension_gap(a, s, window=3)
    assert abs(gap - tension_target(a, s)) < 1e-6


def test_advance_phase_resets_progress(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.8)
    b = advance_phase(a, s)
    assert b.current_phase_index == 2
    assert b.phase_progress == 0.0


def test_advance_phase_clamps_at_last(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=2, phase_progress=0.5)
    b = advance_phase(a, s)
    assert b.current_phase_index == 2  # clamped
    assert b.phase_progress == 1.0
```

- [ ] **Step 2: Run — ImportError**

- [ ] **Step 3: Implement `app/craft/arc.py`**

```python
from __future__ import annotations
from .schemas import Arc, Structure


def global_progress(arc: Arc, structure: Structure) -> float:
    n = len(structure.phases)
    if n == 0:
        return 0.0
    phase_size = 1.0 / n
    idx = min(arc.current_phase_index, n - 1)
    start = idx * phase_size
    prog = arc.phase_progress if arc.current_phase_index < n else 1.0
    return start + phase_size * prog


def tension_target(arc: Arc, structure: Structure) -> float:
    pos = global_progress(arc, structure)
    curve = structure.tension_curve
    if not curve:
        return 0.5
    if pos <= curve[0][0]:
        return curve[0][1]
    if pos >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= pos <= x1:
            if x1 == x0:
                return y0
            t = (pos - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return curve[-1][1]


def tension_gap(arc: Arc, structure: Structure, window: int = 3) -> float:
    """Target minus the average of the last `window` observed readings.

    Positive gap = story is lagging behind the target curve.
    Negative gap = story is hotter than the target curve (often fine).
    Returns the full target when there are no observations.
    """
    target = tension_target(arc, structure)
    if not arc.tension_observed:
        return target
    recent = arc.tension_observed[-window:]
    avg = sum(v for _, v in recent) / len(recent)
    return target - avg


def advance_phase(arc: Arc, structure: Structure) -> Arc:
    last_index = len(structure.phases) - 1
    if arc.current_phase_index >= last_index:
        return arc.model_copy(update={
            "current_phase_index": last_index, "phase_progress": 1.0,
        })
    return arc.model_copy(update={
        "current_phase_index": arc.current_phase_index + 1,
        "phase_progress": 0.0,
    })
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add app/craft/arc.py tests/craft/test_arc.py
git commit -m "feat(craft): arc-state utilities (tension target, gap, advance)"
```

---

## Task 8: recommend_tools

**Files:**
- Modify: `app/craft/library.py` (add method)
- Create: `tests/craft/test_recommend.py`

Design: `CraftLibrary.recommend_tools(arc, structure, recent_tool_ids=None, limit=5)` returns candidate tools for this turn.

Scoring (v1, deterministic, no model involvement):

- Start with `structure.phases[arc.current_phase_index].expected_beats`. Every tool id listed there scores +3.
- If a tool is in `arc.required_beats_remaining`, +5.
- If `tension_gap` is significantly positive (>0.15), boost tools in categories `{reversal, tension, pacing}` by +2.
- If `tension_gap` is significantly negative (< -0.15), boost `{scene_sequel, character}` by +1.
- Tools in `recent_tool_ids` (last 2-3 used) get -2 to discourage repetition.
- Any tool with no category match or expected-beat match gets score 0 — not returned.

Return the top `limit` by score, stable order. Ties broken by tool id.

- [ ] **Step 1: Tests**

```python
# tests/craft/test_recommend.py
from pathlib import Path
import pytest
from app.craft.library import CraftLibrary
from app.craft.schemas import Arc


ROOT = Path(__file__).parent.parent.parent / "app" / "craft" / "data"


@pytest.fixture
def lib():
    return CraftLibrary(ROOT)


def _arc(phase_index: int, gap_hint: str = "lagging", phase_progress: float = 0.5):
    """Build an arc whose observed tension forces a given gap polarity."""
    if gap_hint == "lagging":
        observed = [(1, 0.1), (2, 0.15), (3, 0.2)]  # well below most targets
    elif gap_hint == "hot":
        observed = [(1, 0.95), (2, 0.95), (3, 0.95)]  # well above most targets
    else:
        observed = []
    return Arc(id="a", name="a", scale="chapter", structure_id="three_act",
               current_phase_index=phase_index, phase_progress=phase_progress,
               tension_observed=observed)


def test_recommend_pulls_from_phase_expected_beats(lib):
    arc = _arc(phase_index=0, gap_hint="neutral")  # setup phase
    rec = lib.recommend_tools(arc, lib.structure("three_act"))
    ids = [t.id for t in rec]
    # setup expects chekhovs_gun and scene_sequel
    assert "chekhovs_gun" in ids
    assert "scene_sequel" in ids


def test_recommend_boosts_tension_tools_when_lagging(lib):
    arc = _arc(phase_index=2, gap_hint="lagging")  # midpoint, tension too low
    rec = lib.recommend_tools(arc, lib.structure("three_act"))
    ids = [t.id for t in rec]
    # Expect reversal/midpoint_shift/false_victory ranking high
    assert "reversal" in ids or "midpoint_shift" in ids or "false_victory" in ids


def test_recommend_penalizes_recent_tools(lib):
    arc = _arc(phase_index=1, gap_hint="lagging")
    baseline = [t.id for t in lib.recommend_tools(
        arc, lib.structure("three_act"), recent_tool_ids=None, limit=6)]
    with_recent = [t.id for t in lib.recommend_tools(
        arc, lib.structure("three_act"),
        recent_tool_ids=["try_fail_cycle"], limit=6)]
    # try_fail_cycle should drop or disappear
    if "try_fail_cycle" in baseline and "try_fail_cycle" in with_recent:
        assert with_recent.index("try_fail_cycle") > baseline.index("try_fail_cycle")
    else:
        assert "try_fail_cycle" not in with_recent


def test_recommend_honors_required_beats(lib):
    arc = _arc(phase_index=0, gap_hint="neutral")
    arc = arc.model_copy(update={"required_beats_remaining": ["false_victory"]})
    rec = lib.recommend_tools(arc, lib.structure("three_act"))
    assert rec[0].id == "false_victory"


def test_recommend_limit(lib):
    arc = _arc(phase_index=1, gap_hint="lagging")
    rec = lib.recommend_tools(arc, lib.structure("three_act"), limit=2)
    assert len(rec) == 2
```

- [ ] **Step 2: Run — AttributeError**

- [ ] **Step 3: Add `recommend_tools` to `CraftLibrary`**

Append to `app/craft/library.py`:

```python
from .arc import tension_gap
from .schemas import Arc


_HOT_CATEGORIES = {"reversal", "tension", "pacing"}
_COLD_CATEGORIES = {"pacing", "character"}  # cooling / breathing categories


def _score_tool(
    tool,
    phase_expected: set[str],
    required: set[str],
    recent: set[str],
    gap: float,
) -> int:
    score = 0
    if tool.id in phase_expected:
        score += 3
    if tool.id in required:
        score += 5
    if gap > 0.15 and tool.category in _HOT_CATEGORIES:
        score += 2
    if gap < -0.15 and (tool.category in _COLD_CATEGORIES
                        or tool.id == "scene_sequel"):
        score += 1
    if tool.id in recent:
        score -= 2
    return score


class CraftLibrary(CraftLibrary):  # type: ignore[no-redef]
    pass


def _recommend_tools(
    self: "CraftLibrary",
    arc: Arc,
    structure,
    recent_tool_ids: list[str] | None = None,
    limit: int = 5,
):
    phase = structure.phases[min(arc.current_phase_index, len(structure.phases) - 1)]
    expected = set(phase.expected_beats)
    required = set(arc.required_beats_remaining)
    recent = set(recent_tool_ids or [])
    gap = tension_gap(arc, structure)

    scored: list[tuple[int, str, object]] = []
    for tool in self.all_tools():
        score = _score_tool(tool, expected, required, recent, gap)
        if score > 0:
            scored.append((score, tool.id, tool))

    # Required beats come first, ordered by their list position in the arc.
    required_order = {tid: i for i, tid in enumerate(arc.required_beats_remaining)}

    def sort_key(item):
        score, tid, _ = item
        req_rank = required_order.get(tid, len(required_order) + 1)
        return (-score, req_rank, tid)

    scored.sort(key=sort_key)
    return [t for _, _, t in scored[:limit]]


CraftLibrary.recommend_tools = _recommend_tools
```

(The `class CraftLibrary(CraftLibrary)` trick is just a syntactic wrapper so we can tack on the method without rewriting the class body above. Feel free to inline it into the original class if you prefer; both are equivalent.)

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add app/craft/library.py tests/craft/test_recommend.py
git commit -m "feat(craft): recommend_tools scoring (phase expected + gap + recency)"
```

---

## Task 9: Public surface + final smoke test

**Files:**
- Modify: `app/craft/__init__.py`

- [ ] **Step 1: Write**

```python
from .arc import advance_phase, global_progress, tension_gap, tension_target
from .library import CraftLibrary
from .schemas import (
    Arc,
    ArcPhase,
    Example,
    Structure,
    StyleRegister,
    Tool,
)

__all__ = [
    "Arc",
    "ArcPhase",
    "CraftLibrary",
    "Example",
    "Structure",
    "StyleRegister",
    "Tool",
    "advance_phase",
    "global_progress",
    "tension_gap",
    "tension_target",
]
```

- [ ] **Step 2: Verify and run full suite**

```
uv run python -c "from app.craft import CraftLibrary, Arc, Structure, Tool, tension_target; print('ok')"
uv run pytest -v
```

Expected: prints `ok`; all prior tests + new `tests/craft/` tests all PASS.

- [ ] **Step 3: Commit**

```bash
git add app/craft/__init__.py
git commit -m "feat(craft): expose P9.1 public api"
```

---

## Done criteria

- `uv run python -c "from app.craft import CraftLibrary"` resolves.
- `CraftLibrary(Path('app/craft/data'))` loads 3 structures, 6 tools, 10 examples, 2 styles without error.
- `tension_target(arc, structure)` interpolates the structure's curve at the arc's global progress.
- `recommend_tools(arc, structure, recent_tool_ids=...)` returns tools ranked by phase-fit + gap + required beats, with recency penalty.
- Full pytest suite green (existing + new craft tests).

P9.2 will consume this library from the PLAN stage: build an arc-state briefing (current phase, tension target/observed/gap, required beats, recommended tools with examples) and inject it into the user prompt template. The PLAN output schema will gain a `tool_id` per beat that goes into the trace.
