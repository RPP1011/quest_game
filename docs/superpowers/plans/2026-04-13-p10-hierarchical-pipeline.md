# P10 — Hierarchical Planning Pipeline

> **For agentic workers:** Use superpowers:subagent-driven-development to execute. Steps use checkbox syntax.

**Goal:** Replace the flat PLAN stage with a four-layer hierarchy — ARC (rare) → DRAMATIC → EMOTIONAL → CRAFT → WRITE — plus per-layer critics. All in one plan.

**Architecture:** Each layer is a pipeline stage with its own Pydantic output schema, its own Jinja prompt, its own context spec. Each layer consumes the layer above. `Arc` is persisted in SQLite alongside narrative. WRITE consumes the full `CraftScenePlan` rather than a bare beat list. Stub critics (schema-shape validators) are wired in now; LLM critics are left as a follow-on (P10.3 territory — not part of this plan).

**Scope cut vs. the full design doc:**
- No parallel WRITE candidates + rerank in this plan. Single-candidate WRITE with revise-on-critical, same as today. Candidate rerank is P10.4.
- Critics are **schema + heuristic only** (no LLM critic calls) in this plan. Full LLM critics are P10.3.
- ARC directive is generated at quest start and on explicit phase-advance. No auto-detection of phase transitions yet.
- ReaderModel, Narrator, CharacterVoice, MotifRecurrence, InformationAsymmetry — the doc references these richly, but we ship **minimal versions** in this plan (simple dataclasses with 2-3 fields each) so the hierarchy has types to consume without getting buried in schema work. Fleshing these out is later.

**Done criteria:** `quest play` runs the hierarchical pipeline end-to-end against a real `llama-server`. Trace shows `arc → dramatic → emotional → craft → write → extract` stages. Existing tests still pass (the old flat PLAN tests are replaced by tests at the appropriate layer). Full pytest suite green.

---

## Types (shared across tasks)

Exact types — subagents must use these identifiers and field names verbatim.

### `app/planning/schemas.py` — the planning-layer output models

```python
# ---- shared mini-models ----
Intensity = Literal["background", "emerging", "foregrounded", "climactic"]
Urgency = Literal["immediate", "this_phase", "by_phase_end"]
AdvanceType = Literal["progresses", "complicates", "dormant", "resurfaces", "resolves"]
TransitionType = Literal["escalation", "shift", "rupture", "subsidence", "inversion", "complication"]

class ThemePriority(BaseModel):
    theme_id: str
    intensity: Intensity
    method_hint: str | None = None

class PlotObjective(BaseModel):
    description: str
    urgency: Urgency
    plot_thread_id: str | None = None

class CharacterArcDirective(BaseModel):
    character_id: str
    current_state: str
    target_state: str
    key_moment: str | None = None

# ---- ARC layer ----
class ArcDirective(BaseModel):
    current_phase: str                  # phase name from the structure
    phase_assessment: str
    theme_priorities: list[ThemePriority] = []
    plot_objectives: list[PlotObjective] = []
    character_arcs: list[CharacterArcDirective] = []
    tension_range: tuple[float, float] = (0.3, 0.7)
    hooks_to_plant: list[str] = []
    hooks_to_pay_off: list[str] = []
    parallels_to_schedule: list[str] = []

# ---- DRAMATIC layer ----
class ToolSelection(BaseModel):
    tool_id: str
    scene_id: int
    application: str

class ThreadAdvance(BaseModel):
    thread_id: str
    advance_type: AdvanceType
    description: str

class ActionResolution(BaseModel):
    kind: Literal["success", "failure", "partial", "deferred", "invalid"]
    narrative: str                      # one-sentence "what the action achieves"

class DramaticScene(BaseModel):
    scene_id: int
    pov_character_id: str | None = None
    location: str | None = None
    characters_present: list[str] = []
    dramatic_question: str
    outcome: str
    beats: list[str]
    dramatic_function: str
    tools_used: list[str] = []
    tension_target: float = 0.5
    what_can_go_wrong: str | None = None
    theme_ids: list[str] = []
    reveals: list[str] = []
    withholds: list[str] = []

class DramaticPlan(BaseModel):
    action_resolution: ActionResolution
    scenes: list[DramaticScene]
    update_tension_target: float = 0.5
    ending_hook: str
    suggested_choices: list[dict]       # reuse Choice shape: {title, description, tags}
    tools_selected: list[ToolSelection] = []
    thread_advances: list[ThreadAdvance] = []
    questions_opened: list[str] = []
    questions_closed: list[str] = []

# ---- EMOTIONAL layer ----
class CharacterEmotionalState(BaseModel):
    internal: str
    displayed: str
    gap: str | None = None

class EmotionalScenePlan(BaseModel):
    scene_id: int
    primary_emotion: str
    secondary_emotion: str | None = None
    intensity: float = 0.5
    entry_state: str
    exit_state: str
    transition_type: TransitionType
    emotional_source: str
    surface_vs_depth: str | None = None
    character_emotions: dict[str, CharacterEmotionalState] = {}

class EmotionalPlan(BaseModel):
    scenes: list[EmotionalScenePlan]
    update_emotional_arc: str
    contrast_strategy: str

# ---- CRAFT layer ----
class SceneRegister(BaseModel):
    sentence_variance: Literal["low", "medium", "high"] = "medium"
    concrete_abstract_ratio: float = 0.6  # 0..1, higher = more concrete
    interiority_depth: Literal["surface", "medium", "deep"] = "medium"
    sensory_density: Literal["sparse", "moderate", "dense"] = "moderate"
    dialogue_ratio: float = 0.3  # 0..1
    pace: Literal["compressed", "measured", "dilated"] = "measured"

class PassageOverride(BaseModel):
    trigger: str
    new_register: SceneRegister
    duration: str
    reason: str

class MotifInstruction(BaseModel):
    motif_id: str
    placement: str
    semantic_value: str
    intensity: float = 0.5

class VoiceNote(BaseModel):
    character_id: str
    instruction: str
    code_switching_active: str | None = None

class NegativeSpaceInstruction(BaseModel):
    beat_type: str
    what_is_absent: str
    how_to_render: str

class ParallelInstruction(BaseModel):
    parallel_id: str
    source_description: str
    inversion_axis: str
    execution_guidance: str

class TemporalStructure(BaseModel):
    """v1: single field describing the temporal shape of the scene in prose.
    Example: 'present-scene with one brief flashback before the final beat'."""
    description: str = "linear present-scene"

class CraftScenePlan(BaseModel):
    scene_id: int
    temporal: TemporalStructure = Field(default_factory=TemporalStructure)
    register: SceneRegister = Field(default_factory=SceneRegister)
    passage_register_overrides: list[PassageOverride] = []
    motif_instructions: list[MotifInstruction] = []
    narrator_focus: list[str] = []
    narrator_withholding: list[str] = []
    sensory_palette: dict[str, str] = {}
    voice_notes: list[VoiceNote] = []
    parallel_instruction: ParallelInstruction | None = None
    negative_space: list[NegativeSpaceInstruction] = []
    opening_instruction: str | None = None
    closing_instruction: str | None = None

class CraftPlan(BaseModel):
    scenes: list[CraftScenePlan]
```

### `app/planning/world_extensions.py` — minimal new world entities

```python
class Theme(BaseModel):
    id: str
    name: str
    description: str
    stance: str | None = None   # quest's current stance toward the theme

class MotifDef(BaseModel):
    id: str
    name: str
    description: str
    recurrences: list[int] = []  # update numbers where it appeared
```

### `app/world/schema.py` — Arc added as a first-class persisted entity

Already-defined `Arc` (in `app/craft/schemas.py`) gets a peer for persistence:

```python
# Inside app/world/schema.py, append:
class QuestArcState(BaseModel):
    """Persisted arc state (thin — references the craft-level Arc)."""
    arc_id: str                 # matches app.craft.Arc.id
    quest_id: str
    current_phase_index: int = 0
    phase_progress: float = 0.0
    tension_observed: list[tuple[int, float]] = []
    structure_id: str           # e.g. "three_act"
    scale: str                  # "scene" | "chapter" | "campaign" | "saga"
    last_directive: dict | None = None  # JSON-serialized ArcDirective
```

Plus a SQLite table `arcs` keyed by `(quest_id, arc_id)`.

---

## Tasks

### Task 1: Planning schemas + tests

**Files:**
- Create: `app/planning/__init__.py` (empty)
- Create: `app/planning/schemas.py` (exact content above)
- Create: `app/planning/world_extensions.py`
- Create: `tests/planning/__init__.py`
- Create: `tests/planning/test_schemas.py`

Tests must cover: each of ArcDirective / DramaticPlan / EmotionalPlan / CraftPlan round-trips through `.model_dump_json()` / `.model_validate_json()`; SceneRegister defaults are sensible; DramaticScene requires `dramatic_question` + `outcome` + `beats`; CharacterEmotionalState handles `gap=None`; at least one test per top-level schema asserting validation catches bad enum values.

**Commit:** `feat(planning): pydantic schemas for 4-layer hierarchy`

---

### Task 2: Arc persistence (SQLite)

**Files:**
- Modify: `app/world/db.py` — add `arcs` table in `SCHEMA_SQL`:
  ```sql
  CREATE TABLE IF NOT EXISTS arcs (
      quest_id TEXT NOT NULL,
      arc_id TEXT NOT NULL,
      structure_id TEXT NOT NULL,
      scale TEXT NOT NULL,
      current_phase_index INTEGER NOT NULL DEFAULT 0,
      phase_progress REAL NOT NULL DEFAULT 0.0,
      tension_observed TEXT NOT NULL DEFAULT '[]',
      last_directive TEXT,
      PRIMARY KEY (quest_id, arc_id)
  );
  ```
- Modify: `app/world/schema.py` — append `QuestArcState` model
- Modify: `app/world/state_manager.py` — add methods:
  - `upsert_arc(state: QuestArcState) -> None`
  - `get_arc(quest_id: str, arc_id: str) -> QuestArcState` (raises `WorldStateError` if missing)
  - `list_arcs(quest_id: str) -> list[QuestArcState]`
  - `record_tension(quest_id: str, arc_id: str, update_number: int, value: float) -> None` — appends to `tension_observed`
- Create: `tests/world/test_arc_persistence.py` — upsert round-trip; record_tension appends; list returns all for quest.

**Commit:** `feat(world): persist arc state in SQLite`

---

### Task 3: ARC layer — ArcPlanner

**Files:**
- Create: `app/planning/arc_planner.py` — `ArcPlanner(client, renderer)` with `async def plan(self, *, quest_config, arc_state, world_snapshot, structure) -> ArcDirective`.
- Create: `prompts/stages/arc/system.j2` — system prompt: "You are the arc planner. You produce strategic direction only, not scenes or prose." Instructs to emit JSON matching `ArcDirective` schema. Explain what the layer must NOT specify.
- Create: `prompts/stages/arc/user.j2` — context: quest config (genre, premise, themes), structure (current phase + phases list + tension curve), arc state (phase, progress, tension observed), plot-thread summary, chapter-level narrative summaries (last 5).
- Create: `tests/planning/test_arc_planner.py` — with a FakeClient returning a canned ArcDirective JSON, assert `plan()` returns an `ArcDirective` with populated fields; assert malformed output surfaces a `ParseError`.

`ArcPlanner.plan` uses `chat_structured` with a JSON schema derived from `ArcDirective.model_json_schema()`. Tolerant parsing via `OutputParser.parse_json(raw, schema=ArcDirective)`.

**Commit:** `feat(planning): arc layer — strategic directive generation`

---

### Task 4: DRAMATIC layer

**Files:**
- Create: `app/planning/dramatic_planner.py` — `DramaticPlanner(client, renderer, craft_library)`. `async def plan(self, *, directive: ArcDirective, player_action: str, world, arc_state, structure, craft_library) -> DramaticPlan`.
- Create: `prompts/stages/dramatic/system.j2` — emphasize: what happens and why it matters, NOT how it feels or reads. Must include a `dramatic_question` and `outcome` per scene. Use tools from the recommended list and cite them in `tools_used`.
- Create: `prompts/stages/dramatic/user.j2` — inject: arc directive (rendered as prose bullets), player_action, world snapshot (active characters + location + relevant rules), last 2 raw prose segments for continuity, recommended tools from `craft_library.recommend_tools(arc, structure, recent_tool_ids=<last 3 trace tool uses>)`.
- Create: `tests/planning/test_dramatic_planner.py` — FakeClient returning a canned plan; assert schema parses; assert recommended tools appear in `user_prompt`.

**Commit:** `feat(planning): dramatic layer — what-happens-and-why`

---

### Task 5: EMOTIONAL layer

**Files:**
- Create: `app/planning/emotional_planner.py` — `EmotionalPlanner(client, renderer)`. `async def plan(self, *, dramatic: DramaticPlan, world, recent_prose: list[str]) -> EmotionalPlan`.
- Create: `prompts/stages/emotional/system.j2` — emphasize: emotional trajectory per scene, entry/exit states, transitions, subtext via `surface_vs_depth`. Do NOT specify prose, register, or craft.
- Create: `prompts/stages/emotional/user.j2` — dramatic scenes (with questions + outcomes + functions), last 2 prose segments, character roster.
- Create: `tests/planning/test_emotional_planner.py` — fake client, assert scene count matches dramatic scenes, assert transitions parse to the Literal set.

**Commit:** `feat(planning): emotional layer — how this should feel`

---

### Task 6: CRAFT layer

**Files:**
- Create: `app/planning/craft_planner.py` — `CraftPlanner(client, renderer, craft_library)`. `async def plan(self, *, dramatic: DramaticPlan, emotional: EmotionalPlan, craft_library: CraftLibrary, style_register_id: str | None) -> CraftPlan`.
- Create: `prompts/stages/craft/system.j2` — emphasize: translate drama + emotion into prose blueprint (register, temporal, motifs, narrator instructions). This is NOT writing; it's the writer's instructions.
- Create: `prompts/stages/craft/user.j2` — dramatic plan + emotional plan + (if style_register_id set) full style register with voice samples + relevant tool examples from library (for each `tools_used` across scenes, include 1-2 examples).
- Create: `tests/planning/test_craft_planner.py` — fake client; assert every dramatic scene_id is present in craft plan; register defaults propagate; style voice samples appear in user prompt when register id passed.

**Commit:** `feat(planning): craft layer — how to execute on the page`

---

### Task 7: WRITE consumes CraftScenePlan

**Files:**
- Modify: `app/engine/pipeline.py` — `_run_write` now accepts `CraftScenePlan` per scene, loops scenes, concatenates prose.
- Modify: `prompts/stages/write/system.j2` — shorter, tighter: "You execute the craft plan. One scene at a time." List the fields of `CraftScenePlan` the writer will receive and must respect.
- Modify: `prompts/stages/write/user.j2` — inject one scene's craft plan + relevant voice samples + last 300 chars of prior prose for rhythm continuity.
- Modify: `tests/engine/test_pipeline.py` — update `FakeClient` to return canned responses for the 4 new stages.

The `_run_write` method now loops over scenes (one call per scene, thinking OFF). Concatenates results.

**Commit:** `feat(engine): WRITE consumes CraftScenePlan, one scene per call`

---

### Task 8: Stub critics

**Files:**
- Create: `app/planning/critics.py` with four functions:
  - `validate_arc(directive) -> list[ValidationIssue]` — schema-shape checks: non-empty plot_objectives when phase is rising/crisis, tension_range in [0,1].
  - `validate_dramatic(plan, world, active_entity_ids: set[str]) -> list[ValidationIssue]` — every `pov_character_id` and `characters_present` id exists in world; every `tools_used` id exists in the craft library; `suggested_choices` is non-empty.
  - `validate_emotional(plan, dramatic) -> list[ValidationIssue]` — every dramatic scene_id has exactly one emotional plan entry; no duplicates.
  - `validate_craft(plan, dramatic) -> list[ValidationIssue]` — same-shape check; register `concrete_abstract_ratio` in [0,1]; `dialogue_ratio` in [0,1].

These return issue lists, not raise. Pipeline uses them to annotate the trace. Errors block the stage (retry once); warnings only surface in the trace.

- Create: `tests/planning/test_critics.py` — unit tests for each.

**Commit:** `feat(planning): stub critics (schema + heuristic validators)`

---

### Task 9: Pipeline rewrite

**Files:**
- Modify: `app/engine/pipeline.py` — replace `Pipeline.run` with the hierarchical flow:

```
1. Load (or generate if first-run) ArcDirective → _run_arc
2. _run_dramatic(directive, player_action) → DramaticPlan
3. critics.validate_dramatic → if errors, retry once with a critic-annotated prompt
4. _run_emotional(dramatic) → EmotionalPlan
5. critics.validate_emotional → same retry policy
6. _run_craft(dramatic, emotional) → CraftPlan
7. critics.validate_craft → same retry policy
8. _run_write(craft, per-scene) → prose (concatenated)
9. existing CHECK stage on the combined prose
10. existing REVISE if fixable (same as today)
11. COMMIT narrative
12. existing EXTRACT stage (best-effort)
```

ARC is cached on the arc state; it's only regenerated at quest start OR when `--advance-phase` is explicitly requested. For this plan, we only implement "generate at quest start"; phase-advance is a later knob.

- Modify: `tests/engine/test_pipeline.py` and `test_pipeline_branching.py` — the scripted FakeClient now needs to respond to arc / dramatic / emotional / craft / write / check / extract in order. Update stages assertions from `[plan, write, check, ...]` to `[arc?, dramatic, emotional, craft, write, check, ..., extract]`.
- Modify: `tests/engine/test_pipeline_extract.py` — same update.

**Commit:** `feat(engine): hierarchical pipeline replaces flat PLAN`

---

### Task 10: CLI + server compatibility

**Files:**
- Modify: `app/cli/play.py` — no API change required; the Pipeline abstraction hides the new stages.
- Modify: `app/server.py` — `/advance` endpoint unchanged for consumers, but the trace returned by `GET /quests/{id}/traces/{tid}` will now show the new stage sequence. Update chapter's `choices` extraction: pull from the `dramatic` stage's `parsed_output.suggested_choices` (old code reads `plan`).
- Modify: `tests/server/test_api.py` — update `FakeClient` to answer all new structured calls; fix the trace-stage lookup.

**Commit:** `feat(server): surface dramatic-layer choices to API clients`

---

### Task 11: Arc bootstrap at quest creation

**Files:**
- Modify: `app/server.py` — in `create_quest`, after seeding, pick a structure (default `three_act`, overridable via seed `structure_id` field), construct a `QuestArcState` with `current_phase_index=0`, upsert it. Generate the initial `ArcDirective` lazily on first `advance` call.
- Modify: `app/cli/play.py::init` — same.
- Modify: `tests/server/test_api.py` — new test: creating a quest creates an arc row.

**Commit:** `feat(server): bootstrap arc state at quest creation`

---

### Task 12: Public surface + end-to-end smoke

**Files:**
- Modify: `app/planning/__init__.py` — export ArcPlanner, DramaticPlanner, EmotionalPlanner, CraftPlanner, all schemas, critics module.
- Modify: `app/engine/__init__.py` — nothing (pipeline is the public entry point).

**Final verification:**
- `uv run pytest -v` — all pass.
- `uv run python -c "from app.planning import ArcPlanner, DramaticPlanner, EmotionalPlanner, CraftPlanner; print('ok')"` prints ok.

**Commit:** `feat(planning): expose P10 public api`

---

## Test update strategy (read before touching existing tests)

Existing tests that will need edits:

- `tests/engine/test_pipeline.py::test_pipeline_runs_plan_and_write` → assertion changes from `["plan", "write", "check"]` to `["arc", "dramatic", "emotional", "craft", "write", "check", "extract"]` **or** we split scope and the arc stage appears only on first-turn. Decide per the ARC-at-quest-start decision: arc generates at quest creation, so it's **not** in the per-turn stage list. Per-turn stages are `["dramatic", "emotional", "craft", "write", "check", "extract"]`.
- `tests/engine/test_pipeline_branching.py` — four tests, each scripts 3-4 LLM calls. They now need to script 4 more (emotional, craft, write-per-scene). Simplest: extend ScriptedClient helpers to handle the new schema_names.
- `tests/server/test_api.py::test_chapter_summary_includes_choices_from_trace` and `test_advance_writes_chapter_and_trace` — FakeClient needs to answer for all new schema_names. Choices pulled from `dramatic` stage.
- `tests/engine/test_pipeline_extract.py` — same ScriptedClient extension.

**Do not delete tests**; extend the FakeClient/ScriptedClient to cover the new stages. Every existing assertion about outcome, narrative persistence, trace writes, revise/critical flows must continue to pass against the hierarchical pipeline.

---

## Addendum: Wood-gap schemas

Driven by the observation that our decomposition handles dramatic/emotional/craft as separable layers but Wood's central insight is that style, POV, detail, character, and reality are inseparable. The WRITE stage must recompose what the layers separated — and the primary input to WRITE should be a **prose brief** that already embodies the fusion, with the structured plan as backup for critics.

Five additional schemas and one output-format change. These extend the Task 1 schemas, Task 6 CraftPlanner, Task 7 WRITE, and Task 8 critics.

### Addendum schemas (append to `app/planning/schemas.py`)

```python
# ---- Wood gaps ----

class VoicePermeability(BaseModel):
    """How much the narrator's register absorbs the POV character's language.

    Not static. Fluctuates within a scene — pulls close during emotional
    intensity, retreats during transition or description. 0.0 = pure
    narrator, 1.0 = pure character stream-of-consciousness.
    """
    baseline: float = Field(default=0.3, ge=0.0, le=1.0)
    current_target: float = Field(default=0.3, ge=0.0, le=1.0)
    triggers_high: list[str] = []      # "emotional extremity", "moment of decision", ...
    triggers_low: list[str] = []       # "scene transition", "physical description", ...
    bleed_vocabulary: list[str] = []   # character-register words to appear in narration
    excluded_vocabulary: list[str] = []  # narrator words unsuitable during high permeability
    blended_voice_samples: list[str] = []  # few-shot samples of the blend


class PassagePermeability(BaseModel):
    """Per-passage voice permeability instruction."""
    passage_description: str
    target: float = Field(ge=0.0, le=1.0)
    character_id: str
    bleed_words: list[str] = []
    reason: str


class DetailPrinciple(BaseModel):
    """Governs detail selection for a scene — why THIS detail, not a generic one."""
    perceiving_character_id: str
    perceptual_preoccupations: list[str]   # what they're currently biased to notice
    detail_mode: Literal[
        "character_revealing", "world_establishing", "thematic_resonant",
        "mood_setting", "foreshadowing", "ironic",
    ] = "character_revealing"
    triple_duty_targets: list[str] = []    # moments where one detail serves multiple functions


class MetaphorProfile(BaseModel):
    """Metaphor source domains a character draws from. Metaphor IS characterization."""
    character_id: str
    permanent_domains: list[str]           # from life experience, stable
    current_domains: list[str] = []        # activated by current state
    forbidden_domains: list[str] = []      # domains this character has no experience of
    metaphor_density: Literal["sparse", "occasional", "regular", "rich"] = "occasional"
    extends_to_narration: bool = True      # bleeds into narrator during high permeability


class IndirectionInstruction(BaseModel):
    """How to render unconscious motive through behavior and detail, not exposition."""
    character_id: str
    unconscious_motive: str                # what the system knows but won't name
    surface_manifestations: list[str]      # observable behaviors that express the motive
    detail_tells: list[str]                # details whose inclusion carries the subtext
    what_not_to_say: list[str]             # narration to AVOID
    reader_should_infer: str               # the check criterion


class CraftBrief(BaseModel):
    """The prose form of the craft plan, for the WRITE stage.

    A director's note, not a parameter dump. The structured CraftScenePlan
    exists so critics can validate; the brief exists so the writer receives
    a unified creative vision rather than a checklist.
    """
    scene_id: int
    brief: str                             # 100-300 words of prose brief
```

### Addendum changes to `CraftScenePlan`

Extend `CraftScenePlan` with optional fields (all default None / empty so existing tests don't break):

```python
    # --- Wood-gap fields ---
    voice_permeability: VoicePermeability | None = None
    passage_permeabilities: list[PassagePermeability] = []
    detail_principle: DetailPrinciple | None = None
    metaphor_profiles: list[MetaphorProfile] = []
    indirection: list[IndirectionInstruction] = []
```

### Addendum changes to `CraftPlan`

```python
    briefs: list[CraftBrief] = []  # one per scene, parallel to scenes
```

### Addendum changes to Task 6 (CraftPlanner)

The craft planner prompt now must produce **both** the structured `CraftScenePlan` **and** a prose `CraftBrief` per scene. The structured output is for critics; the brief is what WRITE consumes.

System prompt addition: "For each scene, produce (1) the structured craft plan (register, motifs, narrator instructions, voice permeability, detail principle, metaphor profiles, indirection if relevant) and (2) a 100-300 word prose brief that fuses all these into a single creative vision. The brief should read like a director's note. Do not list parameters; weave them into prose."

### Addendum changes to Task 7 (WRITE)

WRITE's user prompt for each scene now leads with the `CraftBrief.brief` as the primary input. The structured `CraftScenePlan` is included as secondary "reference data" — writer can consult but the brief is the vision. Voice samples for the POV character are still included. If `voice_permeability.blended_voice_samples` is non-empty, those are the most important few-shot material — include them prominently.

### Addendum check dimensions (Task 8)

Add five stub validators to `critics.py`:

- `validate_free_indirect_integrity(craft_plan, prose)` — if `voice_permeability` is set for a scene, check that at least one `bleed_vocabulary` word appears in the prose and that `excluded_vocabulary` words do NOT. Heuristic only.
- `validate_detail_characterization(craft_plan, prose)` — if `detail_principle.detail_mode == "character_revealing"`, verify at least one `perceptual_preoccupation` has a lexical footprint in the prose.
- `validate_metaphor_domains(craft_plan, prose)` — if `metaphor_profiles` set for a character, scan for forbidden-domain vocabulary; warn if found. (Permissive — only errors on clearly-forbidden terms.)
- `validate_indirection(craft_plan, prose)` — for each `IndirectionInstruction`, verify no `what_not_to_say` phrase appears verbatim in prose.
- `validate_voice_blend(craft_plan, prose)` — stub: returns empty issue list in v1; real implementation would compare sentence-by-sentence register. Placeholder so the dimension exists in the trace.

All are heuristic string-match validators. Real LLM-based craft critics are P10.3.

---

## What's explicitly out of scope for this plan

- LLM-based critics (replaces stub critics with real reviewer agents) — P10.3.
- Parallel WRITE candidate generation + rerank — P10.4.
- Auto phase-transition detection (when to bump the arc to the next phase automatically) — P10.5.
- Rich ReaderModel (attachment levels, patience counters, expectation stacks) — later.
- Real narrator/character-voice schemas with voice-sample banks — later.
- Real motif recurrence tracking and parallels database — later.

These are referenced in the doc but implementing them all here would double the task count. The plan ships the hierarchy with **minimum viable versions** of each input; later plans flesh them out.
