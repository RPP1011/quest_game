# Story candidates, virtual-player rollouts, and refinement — design

## Problem

The current pipeline treats every quest as a single blind walk forward: seed → arc directive → dramatic plan → prose, tick by tick. Each dramatic plan invents arc direction on the fly, the player's choices are downstream of a one-shot plan that can't see its own consequences, and there is no notion of whether a given sequence of chapters is actually a good story — structurally, thematically, or in terms of using the seed's pre-authored texture.

Empirically this shows up as:

- **Choices that assume seed knowledge** — the dramatic planner references "Cozme Aflor" in a suggested action before the prose has introduced him, because the planner's context includes the whole seed but the reader has only seen the prose.
- **Under-utilized seeds** — 29 of 49 Pale Lights entities are DORMANT at seed time, and the planner surfaces 0–3 per update somewhat arbitrarily. There's no mechanism ensuring the story actually *uses* the seed's pre-planted foreshadowing, factions, or characters by the end.
- **Unvalidated arcs** — we learn whether a story "works" only by reading it. There's no early signal that chapter 14 is weak, or that the Red Maw hook never gets paid off, until a human reader notices.
- **No re-plan** — once the pipeline commits chapter 3, it can't revisit chapter 3 even if chapter 8 reveals that the earlier beat was a mistake.

This spec proposes a two-phase system: **offline planning with simulated rollouts**, then **presentation of the refined result**. A seed becomes a *space of stories*; the system explores that space, picks the best trajectory, polishes it, and only then presents it to a reader (either as static prose or as scaffolding for interactive play).

## Goals

- A seed supports **multiple candidate stories**. Each candidate commits to a specific arc emphasis — which plot threads are primary, which characters are protagonist-material, which themes drive it.
- Each candidate can be **rolled out** end-to-end multiple times with different virtual-player profiles, producing full-prose trajectories that are evaluated at the chapter level on the existing 8-dim judge.
- Rollouts build a **knowledge base** per quest: which hooks paid off, which entities got used, which character arcs landed, which scenes scored well, which didn't.
- The best trajectory can be **refined** — weak chapters rewritten, pacing gaps filled, foreshadowing payoffs ensured — using the KB as guidance.
- Refined output supports two presentation modes: **static** (read-only polished story) and **scaffolded** (real player agency with quality-guarded fallbacks).
- All rollouts and refinement are incremental-saveable and resumeable. A 15-hour rollout that crashes at rollout 7/9 resumes without losing the first 6.

## Non-goals

- Removing the existing real-time pipeline. Interactive play against a bare seed remains supported; this is additive.
- Generating one refined story per seed. A seed may hold multiple refined trajectories; the picker surfaces them as distinct playable quests.
- Proving that rollouts produce better stories than one-shot generation. That's an empirical question the architecture enables us to answer, not a premise.
- Matching published novels in absolute quality. The judge baseline is Pale Lights Ch 1 (mean 0.75); we aim for rollout-selected stories to beat single-shot by measurable dim deltas, not to hit 0.9+.

## Architecture

```
Seed                                           existing
  │
  ▼
┌──────────────────────────────────────────┐   Phase 1
│ StoryCandidateGenerator                   │
│   seed → 3–5 StoryCandidates              │
│   (synopsis, thread-emphasis, theme-lock) │
└──────────────────────────────────────────┘
  │
  ▼
Pick candidate                                 UI
  │
  ▼
┌──────────────────────────────────────────┐   Phase 2
│ ArcSkeletonGenerator                      │
│   candidate → 10–30-scene skeleton        │
│   (pov, location, dramatic_question,      │
│    required plot beats, target tension)   │
└──────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐   Phase 3
│ RolloutHarness                            │
│   for each virtual-player profile:        │
│     simulate full arc using existing      │
│     Pipeline, routed through action-      │
│     selection driven by profile           │
│   incremental save (resume-safe)          │
└──────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐   Phase 4
│ RolloutScorer + KBExtractor               │
│   8-dim judge per chapter                 │
│   extract: hook payoffs, entity usage,    │
│   character-state trajectories, scenes    │
│   persist to per-quest KB tables          │
└──────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐   Phase 5
│ Refinement                                │
│   select best trajectory                  │
│   rewrite chapters below threshold        │
│   fill pacing gaps                        │
│   splice beats from sibling rollouts      │
└──────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐   Phase 6
│ Presentation                              │
│   static: read-only prose                 │
│   scaffolded: live player with KB-backed  │
│   safety net                              │
└──────────────────────────────────────────┘
```

## Data model

### StoryCandidate

```python
class StoryCandidate(BaseModel):
    id: str                                 # "candidate_01"
    quest_id: str                           # FK
    title: str                              # "Trial-and-survive"
    synopsis: str                           # ~200-word arc outline
    primary_thread_ids: list[str]           # subset of seed plot_threads
    secondary_thread_ids: list[str] = []
    protagonist_character_id: str
    emphasized_theme_ids: list[str]
    climax_description: str
    expected_chapter_count: int
    created_at: datetime
    status: Literal["draft", "picked", "rejected"] = "draft"
```

### ArcSkeleton

```python
class ArcSkeleton(BaseModel):
    id: str
    candidate_id: str                       # FK
    chapters: list[SkeletonChapter]
    theme_arc: list[ThemeBeat]              # when each theme crescendos
    hook_schedule: list[HookPlacement]      # when each seed hook pays off

class SkeletonChapter(BaseModel):
    chapter_index: int                      # 1..N
    pov_character_id: str
    location_constraint: str | None         # e.g. "Bluebell", or None if flexible
    dramatic_question: str
    required_plot_beats: list[str]          # "Tristan meets Cozme", "Abuela's test revealed"
    target_tension: float
    entities_to_surface: list[str] = []     # pre-scheduled DORMANT activations
    theme_emphasis: list[str] = []

class HookPlacement(BaseModel):
    hook_id: str                            # FK to foreshadowing
    planted_by_chapter: int
    paid_off_by_chapter: int
```

### RolloutRun

```python
class VirtualPlayerProfile(BaseModel):
    id: str                                 # "impulsive", "cautious", "scheming"
    description: str
    action_selection_rubric: str            # prompt text for choice-among-choices

class RolloutRun(BaseModel):
    id: str
    candidate_id: str
    skeleton_id: str
    profile_id: str
    seed_nonce: int                         # for reproducibility
    status: Literal["pending", "running", "complete", "failed"]
    chapters_complete: int
    total_chapters_target: int
    started_at: datetime
    completed_at: datetime | None

class RolloutChapter(BaseModel):
    rollout_id: str
    chapter_index: int
    player_action: str
    prose: str
    trace_id: str
    judge_scores: dict[str, float] | None   # 8-dim
    extract_facts: RolloutExtract
```

### RolloutExtract (KB row per chapter)

```python
class RolloutExtract(BaseModel):
    hooks_planted: list[str] = []
    hooks_paid_off: list[str] = []
    entities_introduced: list[str] = []     # DORMANT → ACTIVE this chapter
    entities_removed: list[str] = []        # → DECEASED/DESTROYED
    character_state_deltas: dict[str, dict] = {}
    thread_advances: dict[str, str] = {}    # thread_id → new arc position
    themes_emphasized: list[str] = []
```

## API surface

Per-quest:

- `POST /api/quests/{qid}/candidates/generate` — produce N story candidates
- `GET  /api/quests/{qid}/candidates` — list
- `POST /api/quests/{qid}/candidates/{cid}/pick` — mark picked, triggers skeleton generation
- `GET  /api/quests/{qid}/candidates/{cid}/skeleton` — return arc skeleton
- `POST /api/quests/{qid}/candidates/{cid}/rollouts/start` — launch N rollouts (background)
- `GET  /api/quests/{qid}/candidates/{cid}/rollouts` — list runs with progress
- `GET  /api/quests/{qid}/rollouts/{rid}` — one rollout's chapters + scores
- `POST /api/quests/{qid}/candidates/{cid}/refine` — run refinement passes
- `GET  /api/quests/{qid}/candidates/{cid}/refined` — the picked + refined trajectory

## Storage

New SQLite tables per quest:

```sql
CREATE TABLE story_candidates (id, quest_id, title, synopsis, primary_threads_json,
  secondary_threads_json, protagonist_character_id, emphasized_themes_json,
  climax_description, expected_chapter_count, created_at, status);

CREATE TABLE arc_skeletons (id, candidate_id, chapters_json, theme_arc_json,
  hook_schedule_json);

CREATE TABLE rollout_runs (id, candidate_id, skeleton_id, profile_id, seed_nonce,
  status, chapters_complete, total_chapters_target, started_at, completed_at);

CREATE TABLE rollout_chapters (rollout_id, chapter_index, player_action, prose,
  trace_id, judge_scores_json, extract_json, PRIMARY KEY(rollout_id, chapter_index));

CREATE TABLE kb_hook_payoffs (quest_id, rollout_id, hook_id, planted_at, paid_off_at);
CREATE TABLE kb_entity_usage (quest_id, rollout_id, entity_id, introduced_at, chapter_appearances_json);
CREATE TABLE kb_chapter_scores (rollout_id, chapter_index, dim, score, rationale);
```

## Virtual-player profiles

Initial set (5 profiles, tunable):

| Profile | Bias |
|---|---|
| `impulsive` | Picks the action with the strongest immediate conflict. Avoids "observe first" choices. |
| `cautious` | Prefers information-gathering and fallback actions. Avoids high-tension commitments. |
| `honor_bound` | Picks actions consistent with protagonist's seeded worldview even when suboptimal. |
| `scheming` | Prefers indirect, multi-step actions. Surfaces dormant entities opportunistically. |
| `compassionate` | Resolves conflicts through dialogue/negotiation when available. |

Action-selection implementation: when a chapter completes and produces `suggested_choices`, the virtual player picks one by prompting the same LLM with a short profile-rubric prompt plus the current choices. `~5s per chapter per profile`, negligible overhead.

## Cost model

With current Gemma 4 26B A4B at ~5 min/chapter:

- 3 candidates × 2 profiles × 15 chapters = 90 chapters = ~7.5 hrs per seed
- 5 candidates × 5 profiles × 20 chapters = 500 chapters = ~42 hrs per seed

Phase-3 default: **2 candidates × 2 profiles × 10 chapters = 40 chapters ≈ 3.5 hrs**. Scales up when we swap to a distilled writer LoRA (the plan in `docs/writer-finetune-plan.md`).

## Phasing

### Phase 1 — Story candidates (1–2 days)

- `app/planning/story_candidate_planner.py` — wraps an LLM call that given a seed produces N candidates
- Prompt template `prompts/stages/story_candidate/{system,user}.j2`
- Storage: `story_candidates` table
- API: generate + list + pick
- UI: "Story candidates" view shown after quest creation, player picks one
- Pipeline unchanged — selection writes picked candidate into `config.json`; planners read it as an additional arc constraint

**Exit criterion:** Pale Lights seed produces 3+ candidates with distinct thread-emphasis and protagonist choices, and the pipeline honors the picked candidate's thread-priority in `ArcDirective`.

### Phase 2 — Arc skeleton (2–3 days)

- `app/planning/arc_skeleton_planner.py` — takes a candidate, produces a 10–30-chapter skeleton
- Storage: `arc_skeletons` table
- Dramatic planner consults the skeleton chapter record for the current chapter index, using it as directive override
- Skeleton validator: every seed plot-thread has a scheduled advance, every hook has a planned payoff

**Exit criterion:** dramatic plans for a candidate's chapters reference the skeleton's `required_plot_beats` by id, and entity-surface choices align with `entities_to_surface` pre-scheduled in the skeleton.

### Phase 3 — Rollout harness (1 week)

- `app/rollout/harness.py` — orchestrates N rollouts in sequence (or parallel if GPU allows)
- Virtual-player profiles as YAML under `app/rollout/profiles/`
- `app/rollout/action_selector.py` — profile-driven choice picker
- Incremental save: every chapter commits to `rollout_chapters`; crash resumes from `max(chapter_index)+1`
- CLI: `uv run quest rollout --candidate <cid> --profiles <ids> --chapters <N>`
- Background execution integrated into server: `POST /rollouts/start` returns immediately, progress polled via `GET /rollouts`

**Exit criterion:** a 2×2×10 rollout run completes end-to-end with 40 chapters committed, resume works after SIGTERM, and each chapter has a saved trace and prose.

### Phase 4 — Scoring + KB (3–5 days)

- `app/rollout/scorer.py` — runs the existing 8-dim judge on each chapter post-generation
- `app/rollout/kb_extractor.py` — parses each chapter's trace + prose for hook payoffs, entity usage, thread advances
- Storage: `kb_*` tables
- API: `GET /quests/{qid}/kb` returns aggregated views (per-hook payoff rate, per-entity screen time, dim means per chapter index)
- UI: "Rollout Analytics" tab in the world drawer showing score heatmaps and coverage

**Exit criterion:** for a completed rollout set, we can query "which seed hooks paid off in <50% of rollouts" and "which chapter indices score lowest on subtext_presence" from the UI.

### Phase 5 — Refinement (1+ week)

- `app/rollout/refiner.py` — given a picked "best" trajectory and the KB, performs:
  - **Weak-chapter rewrite**: regenerate any chapter with judge mean < threshold
  - **Hook-payoff fill**: if hook X was supposed to pay off by ch 14 but didn't, insert a payoff beat into the weakest nearby chapter
  - **Cross-rollout splicing**: if a sibling rollout's chapter scored higher on the same skeleton slot, consider adapting
- Each refinement pass is itself recorded as a new rollout for comparability
- Stop criterion: mean dim scores stop improving across passes, or N passes exceeded

**Exit criterion:** a refined trajectory beats its best-sibling-rollout on mean judge score by ≥0.05, with no regression > 0.10 on any single dim.

### Phase 6 — Presentation modes (1+ week)

- **Static mode**: UI renders the refined trajectory as a paginated prose view (chapter 1 → N) with no interactive controls. Useful for sharing / proof of quality.
- **Scaffolded mode**: player plays interactively; every turn, the live pipeline runs but its `ArcDirective` is pinned to the refined skeleton's current chapter. Choices offered favor paths that rolled-out profiles validated. Off-skeleton actions fall through to the existing freestyle pipeline.
- `config.json` gets a `presentation_mode` field.

**Exit criterion:** a real player can play through the refined Pale Lights trajectory in scaffolded mode and finish without the pipeline producing a `failed` outcome.

## Open questions

1. **Rollout parallelism.** Gemma 4 26B on one GPU serializes. Do we benchmark vLLM batching to see if 2–3 rollouts in parallel is feasible, or do we assume serial and plan around it?
2. **Skeleton mutability.** Does the skeleton stay fixed across rollouts of the same candidate, or do rollouts inform a skeleton revision? I'd lock it fixed for Phase 3 and revisit in Phase 5 refinement.
3. **Player-profile calibration.** How do we verify the 5 profiles actually produce distinguishable trajectories? Probably a diff-based eval: two profiles should differ on >30% of chapter-level action choices, else collapse.
4. **Where does choice-quality selection (the original rollout-choices proposal) fit?** It's a narrower case of Phase 3: instead of rolling out 20 chapters, you roll out 1. Implementation can reuse the same harness machinery with a `max_chapters=1` flag, and the lightweight "scratch-rollout" for interactive choice selection becomes a special case. Recommend building that as Phase 3.5 once Phase 3 is in.
5. **Seed density vs rollout cost.** A richer seed (more entities, more hooks) means rollouts have more structure to cover but also more combinatorial possibility. Do we cap seed size for rollout viability, or let it scale?
6. **Distilled writer.** Phase 3 costs drop an order of magnitude if we have the LFM writer LoRA working. Do we gate Phase 3+ on that training being done, or start with Gemma and accept the wall-clock?

## Relationship to existing work

- **Story candidates** are an elaboration of the existing `ArcDirective.theme_priorities` + `plot_objectives` structure. A candidate is effectively a frozen ArcDirective with a synopsis attached.
- **Arc skeleton** is a pre-computed version of the dramatic planner's scene-level output, but for the whole arc. The dramatic planner continues to exist; it consults the skeleton instead of inventing direction from scratch.
- **Rollout harness** reuses the existing `Pipeline` unchanged. The runner built in `app/runner.py` already has the resume-safe chapter loop we need; Phase 3's harness is a thin wrapper that launches N runners with profile-driven action lists.
- **Virtual players** are the mechanism for "action lists" in rollouts — instead of a static YAML list, profile+choices produce the next action at each step.
- **Refinement** can reuse the existing Pipeline's REPLAN/REVISE branches, now fed with KB-derived targeted guidance.

## Recommendation

Build **Phase 1 first**, end-to-end, before committing to later phases. Phase 1 alone delivers:

- A way for a seed to generate multiple playable stories (unlocks the "same seed, different plays" experience without any of the rollout machinery)
- Concrete schemas (`StoryCandidate`, storage, API) that later phases build on
- Evidence that the "story in seed-space" framing produces distinguishable candidates before we invest in rollouts

If Phase 1 candidates are visibly distinct and improve play quality, commit to Phase 2 (skeleton). If Phase 2 gives the dramatic planner enough scaffolding that individual chapters improve measurably, commit to Phase 3 (rollouts). Each phase earns the next.
