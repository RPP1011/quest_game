# Quest Engine Harness Spec

**Date:** 2026-04-12
**Status:** Approved design; supersedes the "Quest Engine" section of `2026-04-12-quest-game-design.md`
**Model assumption:** Gemma 4 26B MoE, locally hosted, OpenAI-compatible API (served by the M1 `LlamaServerBackend`).

Everything below is about the software that wraps it.

---

## 1. Core Abstractions

### 1.1 Pipeline Stage

A stage is a unit of LLM work. It has:

```python
@dataclass
class StageConfig:
    name: str                          # "plan", "write", "check", "revise"
    system_prompt_template: str        # Jinja2 template path
    context_spec: ContextSpec          # What to pull from world state + history
    output_schema: dict | None         # JSON schema for structured output (None = free text)
    inference_params: InferenceParams  # temp, top_p, rep_pen, thinking on/off
    max_retries: int = 2
    retry_strategy: str = "bump_temp"  # or "simplify_prompt", "split_task"

@dataclass
class StageResult:
    stage_name: str
    input_prompt: str                  # Full assembled prompt (for diagnostics)
    raw_output: str                    # Raw model output including thinking tokens
    parsed_output: Any                 # Structured data or clean prose
    token_usage: TokenUsage            # input/output/thinking token counts
    latency_ms: int
    retries: int
    errors: list[StageError]           # Parse failures, schema violations, etc.
```

Every stage produces a `StageResult` that gets logged regardless of success or failure. The harness never discards intermediate state.

### 1.2 Pipeline

A pipeline is an ordered sequence of stages with control flow logic between them.

```python
class Pipeline:
    stages: list[StageConfig]
    flow: PipelineFlow                 # DAG of stage transitions + conditions
    context_builder: ContextBuilder
    world_state: WorldStateManager
    trace: PipelineTrace               # Accumulates StageResults

    def run(self, trigger: PlayerAction | QMDirective) -> PipelineOutput:
        """Execute the pipeline, returning final output + full trace."""

    def run_to(self, stage_name: str, trigger) -> PartialPipelineOutput:
        """Run up to and including the named stage, then pause."""

    def resume(self, from_stage: str, overrides: dict) -> PipelineOutput:
        """Resume from a paused stage with optional overrides to prior outputs."""
```

The `run_to` / `resume` pair is what enables QM mode — you can pause after PLAN, inspect the beat sheet, edit it, then resume from WRITE with the edited plan injected.

### 1.3 PipelineFlow

Not a simple linear chain. The flow between stages has conditional logic:

```
PLAN ──▶ WRITE ──▶ CHECK ──▶ ┬─ (no major issues) ──▶ COMMIT
                              │
                              ├─ (fixable issues) ──▶ REVISE ──▶ RECHECK ──▶ ┬─ COMMIT
                              │                                                │
                              │                                                └─ (still broken) ──▶ FLAG_QM
                              │
                              └─ (critical issues) ──▶ REPLAN (back to PLAN with constraints)
```

"Critical issues" = world rule violations, plot-breaking contradictions, plan adherence failures. These can't be fixed by prose revision — the plan itself was bad. REPLAN adds the CHECK output as a negative constraint: "the previous plan produced these issues, avoid them."

REPLAN loops are capped at 1. If the second plan also produces critical issues, the system flags for QM intervention rather than looping forever.

---

## 2. Context Builder

This is the most important component in the harness. It assembles per-stage prompts from world state, narrative history, and configuration. Bad context assembly is the #1 cause of bad output — more than model limitations, more than prompt wording.

### 2.1 ContextSpec

Each stage declares what it needs:

```python
@dataclass
class ContextSpec:
    # What to include from world state
    entities: EntityFilter             # Which entities, how much detail
    relationships: bool                # Include relationship graph?
    rules: RuleFilter                  # Which world rules are relevant
    timeline: TimelineWindow           # How much history

    # What to include from narrative
    narrative_mode: str                # "full" | "summary" | "none"
    narrative_window: int              # Number of recent updates (full mode)
    summary_depth: str                 # "chapter" | "arc" | "full" (summary mode)

    # What to include from pipeline state
    prior_stages: list[str]            # Which prior stage outputs to include

    # Style and config
    include_style: bool                # Include style config?
    include_character_voices: bool     # Include voice samples?
    include_anti_patterns: bool        # Include anti-pattern list?
    include_foreshadowing: bool        # Include foreshadowing tracker?
```

The key insight: **each stage gets a different view of the world.** The PLAN stage gets summaries + world state + arc outline but no style config. The WRITE stage gets full recent prose + character voices + style config but no world rules. The CHECK stage gets everything it needs to evaluate but no style guidance. This separation is what makes focused prompts work.

### 2.2 Context Assembly

```python
class ContextBuilder:
    def build(self, spec: ContextSpec, stage_name: str) -> AssembledContext:
        """
        Returns:
          - system_prompt: str (rendered from template + spec)
          - user_prompt: str (the actual task with injected context)
          - token_estimate: int (for budget tracking)
          - context_manifest: dict (what was included, for diagnostics)
        """
```

The `context_manifest` logs exactly what went into the prompt — which entities were included, which were excluded and why, how many tokens each section consumed. This is essential for debugging "why did the model forget X" — the answer is usually "X wasn't in the context," and the manifest tells you whether that was intentional (filtered by spec) or a bug.

### 2.3 Token Budget Management

```python
@dataclass
class TokenBudget:
    total: int = 200_000               # Conservative limit under 256K
    system_prompt: int = 15_000
    world_state: int = 30_000
    narrative_history: int = 40_000
    style_config: int = 10_000
    prior_stage_outputs: int = 15_000
    generation_headroom: int = 20_000  # Reserved for model output
    safety_margin: int = 10_000

    def remaining(self, used: dict[str, int]) -> int:
        return self.total - sum(used.values()) - self.safety_margin
```

When context exceeds budget, the builder applies a priority-ordered compression strategy:
1. Truncate oldest narrative summaries first
2. Reduce entity detail level (full → abbreviated → name-only)
3. Drop dormant entities (not referenced in last N updates)
4. Compress prior stage outputs

It never drops: current scene state, active character sheets for present characters, world rules relevant to the current action, the prior stage output the current stage directly depends on.

---

## 3. Prompt Template System

Prompts are **not hardcoded**. They're Jinja2 templates stored in a `prompts/` directory, versioned, and hot-reloadable.

```
prompts/
├── stages/
│   ├── plan/
│   │   ├── system.j2
│   │   └── user.j2
│   ├── write/
│   │   ├── system.j2
│   │   └── user.j2
│   ├── check/
│   │   ├── system.j2
│   │   └── user.j2
│   └── revise/
│       ├── system.j2
│       └── user.j2
├── components/
│   ├── character_sheet.j2             # Reusable character rendering
│   ├── location_desc.j2              # Reusable location rendering
│   ├── scene_state.j2                # Current scene summary
│   ├── timeline_entry.j2             # Single timeline event
│   └── foreshadowing_entry.j2        # Single foreshadowing hook
└── style/
    ├── anti_patterns.j2              # Common LLM failure patterns to avoid
    └── voice_guide.j2                # Character voice spec template
```

Why templates matter: **prompt engineering is the primary iteration loop.** You will rewrite these dozens of times. They need to be editable without touching code, diffable in version control, and testable independently. A code change should never be required to adjust how the model is prompted.

Templates receive a typed context object:

```python
@dataclass
class PlanPromptContext:
    player_action: str
    current_scene: SceneState
    active_plot_threads: list[PlotThread]
    arc_outline: ArcOutline | None
    recent_summaries: list[NarrativeSummary]
    foreshadowing_hooks: list[ForeshadowingHook]
    relevant_rules: list[WorldRule]
    relevant_entities: list[Entity]

@dataclass
class WritePromptContext:
    plan: PlanOutput                   # From Stage 1
    character_sheets: list[CharacterSheet]  # With voice samples
    style_config: StyleConfig
    recent_prose: list[str]            # Last N raw narrative segments
    location_details: list[Location]
    anti_patterns: list[str]
```

---

## 4. World State Manager

### 4.1 Schema

SQLite with JSON columns. Entities are typed but extensible:

```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,          -- character, location, faction, item, concept
    name TEXT NOT NULL,
    data JSON NOT NULL,                -- Type-specific fields
    status TEXT DEFAULT 'active',       -- active, dormant, deceased, destroyed
    last_referenced_update INT,        -- For relevance decay
    created_at_update INT,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE relationships (
    source_id TEXT REFERENCES entities(id),
    target_id TEXT REFERENCES entities(id),
    rel_type TEXT NOT NULL,            -- ally, rival, subordinate, parent, owns, located_at, etc.
    data JSON,                         -- Relationship-specific metadata
    established_at_update INT,
    PRIMARY KEY (source_id, target_id, rel_type)
);

CREATE TABLE world_rules (
    id TEXT PRIMARY KEY,
    category TEXT,                     -- magic_system, physics, social, political
    description TEXT NOT NULL,
    constraints JSON,                  -- Machine-readable constraints for the checker
    established_at_update INT
);

CREATE TABLE timeline (
    update_number INT NOT NULL,
    event_index INT NOT NULL,          -- Order within an update
    description TEXT NOT NULL,
    involved_entities JSON,            -- List of entity IDs
    causal_links JSON,                 -- Links to prior timeline entries
    PRIMARY KEY (update_number, event_index)
);

CREATE TABLE narrative (
    update_number INT PRIMARY KEY,
    raw_text TEXT NOT NULL,
    summary TEXT,                      -- Generated after commit
    chapter_id INT,
    state_diff JSON,                   -- What changed in world state
    player_action TEXT,                -- What triggered this update
    pipeline_trace_id TEXT             -- Link to full diagnostic trace
);

CREATE TABLE foreshadowing (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    planted_at_update INT,
    payoff_target TEXT,                -- What this is hinting at
    status TEXT DEFAULT 'planted',     -- planted, referenced, paid_off, abandoned
    paid_off_at_update INT,
    references JSON DEFAULT '[]'       -- Updates where this was referenced
);

CREATE TABLE plot_threads (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT DEFAULT 'active',      -- active, dormant, resolved, abandoned
    involved_entities JSON,
    arc_position TEXT,                 -- rising, climax, falling, denouement
    priority INT DEFAULT 5             -- 1-10, affects context inclusion
);
```

### 4.2 State Transactions

World state changes are **never applied directly by the LLM output**. The pipeline produces a `StateDelta` — a proposed set of changes — which the WorldStateManager validates and applies atomically.

```python
@dataclass
class StateDelta:
    entity_updates: list[EntityUpdate]     # Modify existing entities
    entity_creates: list[EntityCreate]     # New entities introduced
    relationship_changes: list[RelChange]  # New/modified/removed relationships
    timeline_events: list[TimelineEvent]   # Events that occurred
    foreshadowing_updates: list[FSUpdate]  # Hook status changes
    plot_thread_updates: list[PTUpdate]    # Thread status changes

class WorldStateManager:
    def validate_delta(self, delta: StateDelta) -> ValidationResult:
        """
        Check delta against world rules.
        Returns list of violations (if any) and warnings.

        Examples of violations:
        - Attempting to move a character to a location that doesn't exist
        - Killing a character that's already dead
        - Using an item that was destroyed 10 updates ago
        - Violating a world rule (e.g., magic doesn't work in zone X)
        """

    def apply_delta(self, delta: StateDelta, update_number: int) -> None:
        """Apply validated delta atomically. Rolls back on any failure."""

    def rollback(self, to_update: int) -> None:
        """Revert world state to the state after a given update number.
        Used for retcons and QM overrides."""

    def snapshot(self) -> WorldSnapshot:
        """Capture current state for diagnostics/debugging."""
```

The validation step is crucial. The CHECK stage in the pipeline catches narrative inconsistencies, but the WorldStateManager catches *mechanical* inconsistencies — the kind that compound silently if not caught. A character being in two places at once, an item being used after being consumed, a faction acting on information they shouldn't have.

### 4.3 Retcon Support

Retcons are first-class operations:

```python
class WorldStateManager:
    def retcon(self, retcon: RetconSpec) -> RetconResult:
        """
        Apply a retroactive change:
        1. Modify the historical state at the specified update
        2. Cascade the change forward through subsequent updates
        3. Flag any downstream narrative segments that may be invalidated
        4. Update summaries to reflect the retcon

        Does NOT automatically rewrite prose — returns a list of
        affected narrative segments for QM review.
        """
```

---

## 5. Diagnostics & Tracing

This is not a nice-to-have. Without full pipeline tracing, prompt engineering is guesswork.

### 5.1 Pipeline Trace

Every pipeline run produces a complete trace:

```python
@dataclass
class PipelineTrace:
    trace_id: str
    trigger: PlayerAction | QMDirective
    timestamp: datetime
    stages: list[StageResult]          # Full input/output for every stage
    world_state_before: WorldSnapshot
    world_state_after: WorldSnapshot
    state_delta: StateDelta
    total_latency_ms: int
    total_tokens: TokenUsage
    outcome: str                       # "committed", "flagged_qm", "rejected"

    # Derived diagnostics
    context_manifests: dict[str, dict] # Per-stage: what was in context
    check_results: CheckOutput         # From the CHECK stage
    revision_diff: str | None          # What changed between WRITE and REVISE
```

Traces are stored as JSON files, one per pipeline run, in a `traces/` directory. They should be viewable through the diagnostics panel in the frontend.

### 5.2 What the Diagnostics Panel Shows

For the current update:
- **Pipeline flow**: Which stages ran, which branches were taken, any retries
- **Plan rationale**: The beat sheet, with annotations on why each beat was chosen (from the planner's thinking output)
- **Check results**: Every issue found, severity, whether it was auto-fixed or flagged
- **Revision diff**: Side-by-side of pre- and post-revision prose, highlighting changes
- **State diff**: What changed in world state, with before/after values
- **Context manifest**: For each stage, what entities/history/config were in the prompt and how many tokens each consumed
- **Token budget**: How much of the 256K was used per stage, where the pressure points are

Across the quest:
- **Consistency incident log**: All CHECK-flagged issues over time, categorized. Are the same types of errors recurring? That indicates a prompt problem, not a model problem.
- **Foreshadowing tracker**: Visual timeline of planted hooks and payoffs. Hooks that have aged past the threshold are highlighted.
- **Plot thread status**: Active/dormant/resolved threads with last-referenced dates.
- **Tension curve**: The planner's tension_delta values plotted over time. Are you in a rut?
- **Token usage trends**: Are you approaching context limits? How fast is world state growing?

### 5.3 Replay and Comparison

```python
class DiagnosticsManager:
    def replay(self, trace_id: str, stage: str,
               prompt_override: str = None) -> StageResult:
        """
        Re-run a single stage from a historical trace with the same
        context, optionally with a modified prompt template.

        Essential for A/B testing prompt changes without advancing
        the quest state.
        """

    def compare(self, trace_a: str, trace_b: str) -> Comparison:
        """
        Diff two pipeline traces — useful for comparing prompt versions.
        Shows quality differences in check results, prose changes, etc.
        """
```

This is how you iterate on prompts: replay the same scenario with a different template, compare the outputs. Without this, every prompt change requires playing forward and hoping you notice the difference.

---

## 6. Player Interaction Model

### 6.1 Action Types

```python
@dataclass
class PlayerAction:
    action_type: str                   # "choice", "write_in", "dialogue", "meta"
    content: str                       # The action text
    target_entities: list[str] = None  # Explicit entity references

    # "choice": Player selected from presented options
    # "write_in": Player typed a custom action
    # "dialogue": Player wrote specific dialogue for their character
    # "meta": Player wants to do something outside the narrative
    #         (e.g., "I want to examine my inventory", "what do I know about X")
```

Meta actions don't trigger the full pipeline — they query world state directly and return information without advancing the narrative. This is important: players will ask "wait, what's my character's relationship with X again?" and the system should answer without generating a narrative update.

### 6.2 Choice Generation

The PLAN stage outputs `suggested_choices` as part of its beat sheet. These are rendered in the UI after each update. But the player can also write in a custom action.

Write-in handling:
1. Parse the write-in to identify involved entities and intended outcome
2. Validate against world state (can the player actually do this given their location, inventory, relationships?)
3. If valid, feed directly into the PLAN stage as the action
4. If invalid (e.g., "I fly away" in a no-magic setting), return a rejection with explanation and ask for a different action

This validation step is a lightweight LLM call with a focused prompt: "Given this world state and these rules, is this action possible? If not, why not?" It should be fast (small context, structured output, thinking off).

### 6.3 Multi-Player / Vote Mode (Future)

For forum-style play with multiple readers voting:

```python
@dataclass
class VoteRound:
    update_number: int
    options: list[str]                 # QM/system-generated choices
    votes: dict[str, list[str]]        # option → list of voter IDs
    write_ins: list[WriteIn]           # Custom suggestions

    def tally(self, method: str = "plurality") -> VoteResult:
        """Tally votes. Methods: plurality, approval, ranked_choice."""

    def cluster_write_ins(self, model) -> list[WriteInCluster]:
        """Group similar write-ins by semantic similarity."""
```

This is Phase 2. Don't build it until single-player works.

---

## 7. Configuration & Quest Setup

### 7.1 Quest Configuration

```python
@dataclass
class QuestConfig:
    # Identity
    name: str
    genre: str                         # fantasy, scifi, modern, historical, etc.
    tone: ToneSpec                     # dark/light, serious/comedic, etc.

    # Pacing
    target_words_per_update: tuple[int, int]  # (min, max) range
    scenes_per_update: tuple[int, int]

    # Style
    voice_samples: list[str]           # Exemplar prose passages
    anti_patterns: list[str]           # Phrases/patterns to avoid
    narrator_pov: str                  # first, second, third_limited, third_omni

    # World
    protagonist: CharacterSpec
    initial_entities: list[EntitySpec]
    initial_relationships: list[RelSpec]
    world_rules: list[RuleSpec]

    # Arc (optional, can be set later)
    arc_outline: ArcOutline | None

    # Pipeline tuning
    stage_overrides: dict[str, InferenceParams]  # Per-stage param overrides
    check_severity_thresholds: dict    # What counts as major vs minor
    max_revision_loops: int = 2
```

### 7.2 World Initialization

Two paths to starting a quest:

**Seeded**: The player/QM provides a detailed QuestConfig with pre-built world state. The system loads it and starts immediately. Best for players who know what they want.

**Generated**: The player provides a genre, tone, and rough premise. A dedicated WORLDGEN pipeline stage generates the initial world state — characters, locations, factions, rules, initial situation. The player reviews and edits before the quest begins. This is a one-time pipeline call, not part of the per-update loop.

The WORLDGEN stage should produce entities that are immediately usable by the quest pipeline — same schema, same relationship types, same rule format. No manual translation step.

---

## 8. File Structure

```
quest-engine/
├── src/
│   ├── pipeline/
│   │   ├── orchestrator.py            # Pipeline execution, flow control
│   │   ├── stages.py                  # Stage definitions and configs
│   │   ├── context_builder.py         # Prompt assembly from state + spec
│   │   ├── output_parser.py           # Parse structured/free-text output
│   │   └── inference_client.py        # OpenAI-compatible API wrapper
│   ├── world/
│   │   ├── state_manager.py           # CRUD, validation, transactions
│   │   ├── schema.py                  # Entity types, relationships
│   │   ├── rules_engine.py            # World rule validation
│   │   └── migrations.py              # Schema evolution
│   ├── narrative/
│   │   ├── history.py                 # Narrative storage, retrieval
│   │   ├── summarizer.py              # Post-update summarization
│   │   ├── foreshadowing.py           # Hook tracking, payoff scheduling
│   │   └── timeline.py                # Event log management
│   ├── player/
│   │   ├── action_handler.py          # Action parsing, validation
│   │   ├── choice_generator.py        # Post-update choice presentation
│   │   └── meta_queries.py            # Non-narrative queries
│   ├── diagnostics/
│   │   ├── tracer.py                  # Pipeline trace logging
│   │   ├── replay.py                  # Trace replay, A/B comparison
│   │   ├── analytics.py               # Cross-quest trend analysis
│   │   └── export.py                  # Trace export for external analysis
│   └── config/
│       ├── quest_config.py            # Quest setup and configuration
│       └── defaults.py                # Default inference params, budgets
├── prompts/
│   ├── stages/                        # Per-stage system/user templates
│   ├── components/                    # Reusable template fragments
│   ├── style/                         # Style config templates
│   └── worldgen/                      # World generation templates
├── data/
│   ├── quests/                        # Per-quest SQLite databases
│   └── traces/                        # Pipeline trace JSON files
├── tests/
│   ├── test_context_builder.py        # Context assembly tests
│   ├── test_world_state.py            # State validation tests
│   ├── test_output_parser.py          # Output parsing tests
│   └── fixtures/                      # Sample world states, traces
├── cli.py                             # CLI player interface (Phase 1)
└── server.py                          # FastAPI server (Phase 2)
```

Note: existing code lives under `app/runtime/` (M1 runtime layer). The paths above are the engine's target layout; we will reconcile (`src/` vs `app/`) in the first engine plan — likely by adopting `app/` as the root (`app/pipeline/`, `app/world/`, etc.) to keep a single package.

---

## 9. Build Order

1. **`schema.py` + `state_manager.py`**: Define entity types. Write basic CRUD. Write a script that seeds a test world from a JSON file. Validate you can query it sanely.

2. **`inference_client.py`**: Thin wrapper around OpenAI-compatible API. Handles structured output parsing, thinking mode toggle, retry with backoff. Test independently against your Gemma instance. *(M1 already provides a minimal version; this step extends it with structured-output + thinking support.)*

3. **Prompt templates for PLAN and WRITE stages.** Just the templates — no orchestrator yet. Manually assemble a context, paste it into a notebook or CLI, call the model, read the output. Iterate on the templates until PLAN produces usable beat sheets and WRITE produces readable prose. This is where you'll spend the most time and it doesn't require any infrastructure.

4. **`context_builder.py`**: Automate what you were doing manually in step 3. Takes a ContextSpec + world state, produces assembled prompts. Test by comparing its output to your manual assembly.

5. **`output_parser.py`**: Parse structured JSON from PLAN and CHECK stages. Parse free text from WRITE and REVISE. Handle malformed output gracefully (the model WILL produce invalid JSON sometimes).

6. **`orchestrator.py` with linear flow only**: PLAN → WRITE → commit. No CHECK/REVISE yet. Just get the basic loop working: action in, prose out, state updated.

7. **`cli.py`**: Minimal player interface. Print prose, show choices, accept input, run pipeline. Play through 5-10 updates. Evaluate output quality.

8. **CHECK and REVISE stages**: Add the review loop. Compare output quality before/after. This is where you validate whether multi-pass actually helps with your specific prompt templates and model setup.

9. **`tracer.py`**: Instrument everything. Log full traces. Build the replay capability.

10. **Frontend**: Only now. You know the pipeline works, you know what diagnostics matter, you know the interaction patterns. Build the UI around validated workflows, not assumptions.
