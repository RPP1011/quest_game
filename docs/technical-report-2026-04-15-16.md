# Technical Report: World Seed Density → Story-Rollout Architecture

**Dates:** 2026-04-15 to 2026-04-16
**Authors:** Ricky + Claude Opus 4.6
**Scope:** From "the world feels thin" to a six-phase story-generation architecture with empirical validation at each layer.
**Baseline:** 817 tests, per-beat write loop, 3-entity seeds, 2nd-person CYOA narrator.
**Final state:** 910 tests, 5 architecture phases shipped, overnight 2×2×10 rollout running.

---

## 1. Problem statement

The quest-game pipeline was producing structurally correct but narratively thin output. The Pale Lights craft analysis (484 world facts / 44 chapters, 87 named characters, 11 new proper nouns per chapter) revealed the gap: our test seeds had 3 entities. The world felt thin because it *was* thin.

The initial diagnosis identified three layers to fix:
1. **Seed density** — the seed schema accepts entities but nothing pressures the author toward density
2. **Dormant entity activation** — the schema supports `EntityStatus.DORMANT` but nothing in the planner uses it
3. **Runtime world-fact generation** — deferred as a consistency liability

This report covers what was actually built, what worked, what didn't, and what the empirical results show.

---

## 2. Writer fidelity — the first real bug

### 2.1 Diagnosis

Before touching seed density, we ran the existing pipeline against a hand-authored 48-entity Pale Lights seed. Two catastrophic bugs emerged that had nothing to do with density:

**Bug 1: Fortuna rendered as a cat.** The dramatic plan correctly listed Fortuna as `characters_present` with the beat "Fortuna sits on the bed, golden bored eyes." The writer produced: *"She did not stir. She did not meow. She was simply there—a golden-eyed witness... her ears twitching."* The seed described Fortuna across ~200 words (girl-shaped goddess, red dress, gold hair, bellows, pouts, can't touch matter). The writer used one trait (gold eyes) and free-associated a cat.

**Root cause:** The write templates never rendered `{{ entities }}`. The variable was in the template context (loaded by `ContextBuilder`) but neither `system.j2` nor `user.j2` referenced it. The writer had zero access to seeded entity descriptions.

**Bug 2: Narrator POV config ignored.** Seed specified `pov_type: third_limited_alternating` with five third-person voice samples. The writer produced second-person CYOA: *"You slipped through the service entrance."*

**Root cause:** `system.j2` line 1 hardcoded `"SV-quest style: 2nd person past"` regardless of the `Narrator` block in the seed. The narrator config never reached the writer's system prompt.

### 2.2 Fix

Four files changed:
- `system.j2`: conditional narrator block rendering full config (POV, register, worldview, editorial_stance, sensory_bias, attention_bias, knowledge_scope, withholding_tendency, unreliability_axes). Falls back to 2nd-person when no narrator is configured.
- `user.j2`: new "Characters in this scene" block rendering full entity `data` (description, role, voice, skills, worldview, abilities, constraints) for every character the dramatic plan lists as present.
- `pipeline.py`: `_resolve_scene_entities()` helper resolving `characters_present` IDs to full `Entity` objects with `data` blobs.
- `app/cli/play.py`: `narrator` now written to `config.json` so the pipeline can read it.

### 2.3 Before/after

| Metric | V1 (before) | V2 (after) | Pale Lights Ch 1 |
|---|---|---|---|
| Words | 2,389 | 5,437 | 5,438 |
| Time | 16.5 min | 5.2 min | — |
| POV | 2nd person | 3rd person close | 3rd person close |
| Fortuna | Cat | Goddess in red dress | Correct |
| Named world-facts used | 0 | 14 | ~11/chapter |
| Zero "you" sentences | 40+ | 0 | 0 |

The word count matched Pale Lights Ch 1 exactly. Generation time dropped 3× (likely server cache benefit from restart, not code change).

---

## 3. DORMANT entity activation

### 3.1 Design

The seed schema already supported `EntityStatus.DORMANT` but nothing in the planner surfaced dormant entities. Fix:
- `DramaticPlan` gained `entities_to_surface: list[str]`
- Dramatic planner now sees dormant entities in a separate "Available to Surface" section (names + roles, not full descriptions — trimmed to control prompt length)
- `Pipeline._activate_surfaced_entities()` patches DORMANT → ACTIVE after the dramatic plan is committed (not in the extract stage — surfacing is a planning decision, not a prose observation)

### 3.2 Empirical result

v7 rollout: dramatic planner chose `['char:abuela', 'char:cozme', 'char:angharad']` as entities_to_surface. Abuela and Cozme were promoted from DORMANT to ACTIVE in the DB. Abuela appeared 4 times in the prose with seed-fidelity texture ("the casual tilt of her head, the way she had smiled as if she already knew exactly how much blood he would spill").

The plan-vs-execution gap: Cozme appeared in `entities_to_surface` but the writer didn't render him in prose. The activation machinery works; the writer's follow-through on surfaced entities is a separate problem.

---

## 4. Strategy sweep — per-beat wins

We implemented 5 write-stage strategies sharing the same plan:

| Strategy | Words | Secs | Mean 8-dim | vs Pale Lights (0.75) |
|---|---|---|---|---|
| **per_beat** | 5,588 | 82 | **0.762** | +0.012 |
| expand | 3,043 | 50 | 0.675 | -0.075 |
| one_shot | 626 | 7 | 0.525 | -0.225 |
| scene | 828 | 30 | (judge failed) | — |
| refine | 613 | 13 | (judge failed) | — |

**per_beat is the winner** — ties or beats Pale Lights on 7 of 8 dims. The per-beat loop forces the writer to commit to each beat fully, preventing "summary mode" that collapsed one_shot to 626 words.

Key finding: one_shot is faster (7s) but the 26B model produces compressed summaries when given "write the whole chapter." The per-beat loop is the right architecture for this model size.

---

## 5. Check → revise loop enforcement

### 5.1 Bug

The hierarchical post-write flow gated revise on `not has_critical`:
```python
if check_out.has_fixable and not check_out.has_critical and not recheck_done:
    prose = await self._run_revise(...)
```

Critical world-rule violations (e.g. Fortuna touching matter) were flagged but committed unchanged. The detection machinery was working; the correction path was blocked.

### 5.2 Fix

Changed to a loop that fires revise on *any* non-trivial issues (critical OR fixable), up to 2 attempts:
```python
while (check_out.has_critical or check_out.has_fixable) and \
        revise_attempts < MAX_REVISE_ATTEMPTS:
    prose = await self._run_revise(...)
    revise_attempts += 1
    check_out = await self._run_check(...)
```

### 5.3 Cascading bugs discovered

The revise fix exposed three more issues:

**Revise truncation.** `REVISE_MAX_TOKENS=3000` was per-beat sized; revise rewrites the FULL chapter in one call. A 2400-word chapter got truncated to 871 words, ending mid-sentence. Fixed by bumping to 8000.

**Revise meta-commentary leak.** Gemma's thinking mode leaked chain-of-thought into the revise output: "Total = 1600. The word count stays approximately the same!" This was committed as prose. The check stage then saw 5000 words of prose+planning, flagged issues; revise 2 collapsed it to 553 words. Fixed by setting `thinking=False` on the revise call and strengthening the prompt.

**Seed self-contradiction.** Fortuna's description said "Cannot touch the material world; rests her chin on his shoulder" in one sentence. The revise loop fired twice trying to fix the touch violation but the model faithfully reproduced "rests her chin on his shoulder" each time — because the seed said so. Fixed by rewriting the description to make explicit that Fortuna's apparent contact is perceptual (phantom warmth, pressure-without-mass).

### 5.4 Trace outcome persistence

`trace.outcome = outcome` was set after the last `add_stage()`, but the live-trace incremental save only fired inside `add_stage()`. The on-disk trace showed `outcome="running"` even for committed chapters. Fixed by adding `trace.set_outcome()` that triggers the callback.

---

## 6. Story-rollout architecture (Phases 1–5)

### 6.1 Spec

The architecture design (docs/superpowers/specs/2026-04-15-story-rollout-architecture.md) proposes six phases:

1. **Story candidates** — seed → 3–5 candidate arcs
2. **Arc skeleton** — picked candidate → 10–30-chapter outline
3. **Virtual-player rollouts** — synthetic playthroughs with profiles
4. **Scoring + KB** — 8-dim judge + hook/entity extraction
5. **Refinement** — targeted rewrite of weak chapters
6. **Presentation** — static + scaffolded play modes (deferred)

### 6.2 Phase 1: Story candidates

**What it does:** Given a seed, produce N materially distinct candidate story arcs. Each commits to specific primary threads, a protagonist, emphasized themes, a climax, and an expected chapter count. The player picks one before chapter generation starts.

**Implementation:**
- `StoryCandidate` pydantic model + `story_candidates` SQLite table
- `StoryCandidatePlanner` wrapping an LLM call with closed-enum schema constraining thread/character/theme ids to seeded values
- API: `POST /candidates/generate`, `GET /candidates`, `POST /candidates/{cid}/pick`
- UI: candidate picker cards with auto-generation on first quest view, picked banner, "Change story" flow
- Arc planner reads `picked_candidate` from `config.json` and renders it as directive context

**Empirical result — Pale Lights seed produced 3 candidates:**

| Candidate | Protagonist | Primary threads | Length |
|---|---|---|---|
| The Shadow Watch | Tristan | red_maw + hoja_roja + trials | 50 ch |
| The God-Eater's Debt | Tristan | tristans_list + red_maw | 35 ch |
| The Noble's Reckoning | Angharad | angharads_revenge + trials | 42 ch |

**Exit criterion met:** 3+ candidates with distinct thread-emphasis and protagonist choices; pipeline honors picked candidate's thread-priority.

### 6.3 Phase 2: Arc skeleton

**What it does:** Given a picked candidate, produce a chapter-by-chapter outline spanning its expected arc. Each chapter slot carries POV, location constraint, dramatic question, required plot beats, target tension, pre-scheduled DORMANT activations, and theme emphasis. Foreshadowing hooks are scheduled with planted-by / paid-off-by chapter targets.

**Implementation:**
- `ArcSkeleton`, `SkeletonChapter`, `HookPlacement`, `ThemeBeat` schemas + `arc_skeletons` SQLite table
- `ArcSkeletonPlanner` with closed-enum schema constraining all entity/thread/hook references
- `Pipeline._current_skeleton_chapter(update_number)` looks up the chapter for the current tick; wired into both arc and dramatic planner prompts
- UI: "Arc Outline" tab in the world drawer showing per-chapter cards with hook schedule and theme arc

**Empirical result — 30-chapter outline for "The Shadow Watch":**
- POV alternation emerged naturally (Tristan / Angharad per chapter)
- All 10 seeded foreshadowing hooks scheduled (planted_by / paid_off_by)
- `concept:red_maw_truth` peaks at ch 24 (subverting); `concept:trials` peaks at ch 18 (affirming)
- Tension curve rises cleanly: 0.20 → 0.80 → 0.20

**Bug found during verification:** `_load_or_generate_arc()` referenced `update_number` without it being in scope — a Phase 2 wiring error. The arc stage fell back to a minimal directive. The dramatic planner still worked because it consults the skeleton independently.

### 6.4 Phase 3: Virtual-player rollouts

**What it does:** Given a picked candidate + arc skeleton + a virtual-player profile, run the full pipeline chapter-by-chapter with the profile's action-selection rubric choosing among suggested choices at each turn. Each chapter saves incrementally; crashes resume from the next unfinished chapter.

**Implementation:**
- `RolloutRun`, `RolloutChapter`, `RolloutExtract` schemas + SQLite tables
- 3 bundled profiles (`impulsive`, `cautious`, `honor_bound`) as YAML
- `action_selector.select_action()` — structured LLM call picking from choices per profile rubric; graceful fallback to index 0
- `run_rollout()` orchestrator: bootstraps isolated rollout DB (copy main quest DB + wipe narrative), runs pipeline per chapter, incremental save
- CLI: `quest rollout --quest QID --candidate CID --profile PID --chapters N`
- API: `POST /rollouts/start` (fire-and-forget), `GET /rollouts`, `GET /rollouts/{rid}`

**Empirical result — 2-chapter impulsive rollout:**
- Ch 1 action from skeleton beats; Ch 2 action picked by action_selector ("Talk back to Fortuna to see if she can help him read the odds of the next room")
- Prose coherent at 2,430 + 3,630 words
- Resume verified: killed mid-run, restarted, skipped completed chapters

**Trace evaluation revealed issues:**
- Fortuna-touching-matter violation (detected by check but committed due to the revise-loop bug, which we then fixed)
- Ch 2 retreaded Ch 1's heist scene (cross-chapter coherence gap, fixed via prompt)
- Chapter endings cut off mid-sentence (fixed via prompt)

### 6.5 Phase 4: Scoring + KB extraction

**What it does:** Score every rollout chapter on the 8-dim chapter judge (same rubrics as `tools/judge_chapters.py`). Extract hook payoffs, entity usage, and thread advances from each chapter's trace. Aggregate across rollouts via API.

**Implementation:**
- 3 KB tables: `kb_chapter_scores`, `kb_hook_payoffs`, `kb_entity_usage`
- `scorer.py`: `score_chapter()` single batched structured call; `score_and_persist_chapter()` writes per-dim rows + chapter blob; idempotent
- `kb_extractor.py`: parses trace stages for foreshadowing_updates, entity_updates, thread_advances; word-boundary name matching for entity mentions in prose
- Wired into harness: KB extraction always runs (no LLM cost); scoring runs when `score=True`
- API: `GET /kb` (aggregates: payoff_rate, screen_time, dim_means_by_chapter), `GET /rollouts/{rid}/scores`

**Empirical result — v5 chapter scored against Pale Lights:**

| Dim | v5 ch1 | Pale Lights |
|---|---|---|
| tension_execution | 0.70 | 0.90 |
| emotional_trajectory | 0.60 | 0.90 |
| choice_hook_quality | 0.70 | 0.70 |
| update_self_containment | 0.70 | 0.70 |
| voice_distinctiveness | 0.60 | 0.60 |
| thematic_presence | 0.60 | 0.90 |
| subtext_presence | 0.30 | 0.60 |
| interiority_depth | 0.60 | 0.70 |
| **mean** | **0.60** | **0.75** |

Worst gaps: subtext, emotional_trajectory, thematic_presence. These became refinement targets.

### 6.6 Phase 5: Refinement

**What it does:** Three pluggable selectors identify chapters to retry; the framework regenerates each with strategy-specific guidance, scores the result, and commits only when it materially beats the baseline.

**Implementation:**
- `RefinementAttempt` schema + `refinement_attempts` SQLite table
- Framework: `refine_one()` regenerates + scores + accept/reject; `run_refinement_pass()` orchestrates
- Accept thresholds (from spec): mean delta ≥ +0.05 AND no per-dim regression > -0.10
- 3 selectors:
  - **WeakChapterSelector** — chapters below a mean-dim threshold, ranked lowest-first
  - **UnpaidHookSelector** — diffs skeleton hook_schedule against KB payoffs; targets deadline chapters
  - **SiblingOutscoredSelector** — chapters where another rollout scored ≥0.15 higher on any dim; injects sibling prose as reference
- CLI: `quest refine --strategy weak|hooks|sibling|all`
- API: `POST /rollouts/{rid}/refine`, `GET /refinements`

**Empirical result — refining v5 chapter 1:**

| Dim | baseline | refined | Δ |
|---|---|---|---|
| subtext_presence | 0.30 | **0.60** | **+0.30** |
| voice_distinctiveness | 0.60 | **0.90** | **+0.30** |
| interiority_depth | 0.60 | **0.90** | **+0.30** |
| (5 others) | unchanged | unchanged | 0.00 |
| **mean** | **0.60** | **0.713** | **+0.113** |

The targeted dim (subtext_presence, worst at 0.30) doubled. Word count recovered 553 → 3,948 words. Both spec exit criteria met.

---

## 7. UI improvements

Alongside the architecture work, the web frontend got a significant overhaul:

- **Hero panel** for empty quests: premise with serif drop-cap, themes as bullets, cast-at-curtain-up, opening location
- **Candidate picker**: auto-generation, card-based UI with synopsis/threads/protagonist/climax, picked banner with "Change story" flow
- **Arc outline tab** in the world drawer: per-chapter cards (POV, tension, dramatic question, plot beats, entities-to-surface), hook schedule, theme arc, Generate/Regenerate buttons
- **World drawer**: 9 tabs (Characters, Factions, Locations, Items, Concepts, Threads, Hooks, Rules, Motifs) browsing the full seed
- **Starter actions**: opening thread buttons derived from plot threads, clickable to seed the input
- **Live trace progress**: incremental trace save (via `PipelineTrace._on_update` callback) + 2s polling during generation showing stage-by-stage progress
- **Warm earth-tone palette** with serif typography for prose, system sans-serif for chrome

---

## 8. Seed: `seeds/pale_lights.json`

Hand-authored from the Pale Lights Book 1 rollup. Final state:

| Category | Count | Examples |
|---|---|---|
| Characters | 19 | Tristan, Fortuna, Angharad, Fisher, Yaotl, Abuela, Cozme, Yong, Song, Isabel, Augusto, Remund, Tupoc, Lan, Ju, Francho, Vasanti, Crestina, Red Maw |
| Locations | 6 | Sacromonte, Bluebell, Vieja Perdida, Old Fort, Cantica, Rookery |
| Factions | 8 | Watch, Hoja Roja, Red Eye, Guardia, Cordero Sonriente, Cerdan, Ruesta, Villazur |
| Concepts | 7 | Glare, Gloam, contracts, Law of Rats, trials, Iscariot Accords, Signs, Red Maw truth |
| Items | 7 | Spinster's Milk, lodestone extract, Tristan's List, Rhadamanthine pistol, Angharad's saber, Osian's letter, cabinet papers |
| Rules | 7 | Foretelling illegal, Iscariot Accords, Fortuna proximity, Fisher cost, Gloam sickness, sanctuary, Trial of Weeds mechanics |
| Plot threads | 6 | Tristan's List, Angharad's revenge, the trials, Hoja Roja hunts Tristan, Red Maw containment, Yong's husband |
| Foreshadowing | 10 | Red Maw, Fisher's nature, Augusto's betrayal, Isabel's charm, Dominion truth, Cantica lamps, Abuela's hand, Song's Dimming, Tupoc's crew, pistol changes hands |
| Motifs | 4 | pale lights, rats/Law of Rats, hands, water/fishing |
| Themes | 3 | honor vs survival, lost things, contracts as debt |

49 entities total: 20 active, 29 dormant.

---

## 9. Robustness fixes

Several model-interaction issues were diagnosed and fixed during empirical testing:

| Issue | Root cause | Fix |
|---|---|---|
| Dramatic stage fails 3/4 times | llama-server doesn't enforce `required` on nested JSON schema structs | `_repair_missing_scene_ids()` injects 1..N ids by position before validation |
| Dramatic plan truncates at 4k tokens | `max_tokens=4096` too small for rich seeds with 29 dormant entities | Bumped to 8192; trimmed dormant listing to name+role only |
| Revise leaks chain-of-thought | Gemma's thinking mode emits reasoning tokens in output | `thinking=False` on revise call; prompt forbids meta-commentary |
| Revise truncates full-chapter rewrite | `REVISE_MAX_TOKENS=3000` is per-beat sized; revise needs per-chapter | Bumped to 8000 |
| Trace shows `outcome=running` | `trace.outcome = outcome` set after last `add_stage()` but on_update callback only fires inside `add_stage()` | Added `trace.set_outcome()` that triggers the callback |
| Extract rejects DORMANT entity patches | `known_ids` only included ACTIVE entities | Widened to all non-destroyed entities |
| Extract fails on hallucinated hook IDs | Model emits location IDs as if they were hook IDs | `build_delta` now filters unknown hook IDs as warnings, not fatal errors |
| CLI quest_id mismatch | `db.stem = "quest"` vs server's directory name `"pale_lights"` | Both now use `db.parent.name` |

---

## 10. Test coverage

| Baseline | Final | Delta |
|---|---|---|
| 817 | 910 | +93 |

New test modules:
- `tests/world/test_story_candidates.py` (7)
- `tests/world/test_arc_skeletons.py` (5)
- `tests/world/test_rollouts.py` (7)
- `tests/world/test_kb_storage.py` (7)
- `tests/planning/test_story_candidate_planner.py` (3)
- `tests/planning/test_arc_skeleton_planner.py` (5)
- `tests/server/test_candidates_api.py` (2)
- `tests/server/test_skeleton_api.py` (3)
- `tests/server/test_rollouts_api.py` (5)
- `tests/server/test_kb_api.py` (3)
- `tests/engine/test_revise_loop.py` (3)
- `tests/rollout/test_profiles.py` (8)
- `tests/rollout/test_harness.py` (6)
- `tests/rollout/test_harness_phase4.py` (2)
- `tests/rollout/test_scorer.py` (5)
- `tests/rollout/test_kb_extractor.py` (8)
- `tests/refinement/test_framework.py` (4)
- `tests/refinement/test_selectors.py` (10)

---

## 11. What's running now

An overnight 2×2×10 rollout (`tools/overnight_rollout.sh`):
- 2 candidates (auto-picked from 3 generated)
- 2 profiles (`impulsive`, `cautious`)
- 10 chapters each
- 40 total chapters ≈ 3.5–4 hours on Gemma 4 26B A4B
- Scoring + KB extraction on every chapter
- Resume-safe

This will produce the first real multi-rollout dataset for KB aggregation queries ("which hooks paid off in <50% of rollouts", "which chapter indices score lowest on subtext_presence") and cross-rollout refinement (sibling-outscored selector).

---

## 12. What's next

**Phase 6: Presentation modes.** Two modes from the spec:
- Static: paginated read-only prose view of the refined best trajectory
- Scaffolded: real player plays interactively with the arc skeleton as guardrails and the KB as a quality floor

**Quality work:**
- Multi-pass refinement loop (refine, score, refine again until convergence)
- Dialogue ratio improvement (prose is heavy narration, light on speech)
- Metaphor saturation critic (gambling imagery repeated excessively in Tristan POV)
- Writer LoRA training using the scored rollout chapters as (plan, prose) pairs

**Infrastructure:**
- Overnight rollout result analysis
- Cross-rollout refinement with sibling-outscored selector (needs ≥2 rollouts of same candidate)
- Distilled writer model to drop rollout cost by 10×

---

## 13. Artifacts

| Path | Content |
|---|---|
| `seeds/pale_lights.json` | 49-entity Pale Lights seed with full narrator config |
| `docs/superpowers/specs/2026-04-15-story-rollout-architecture.md` | Six-phase design spec |
| `docs/pale-lights-seed-comparison.md` | V1 vs V2 empirical comparison |
| `docs/phase1-story-candidates-result.md` | Phase 1 exit criterion check |
| `docs/phase2-arc-skeleton-result.md` | Phase 2 exit criterion check |
| `docs/phase3-rollout-harness-result.md` | Phase 3 exit criterion check |
| `docs/phase4-kb-scoring-result.md` | Phase 4 exit criterion check |
| `docs/phase5-refinement-result.md` | Phase 5 exit criterion check |
| `tools/strategy_sweep.py` | 5-strategy comparison harness |
| `tools/overnight_rollout.sh` | 2×2×10 rollout launcher |
| `app/planning/story_candidate_planner.py` | Phase 1 planner |
| `app/planning/arc_skeleton_planner.py` | Phase 2 planner |
| `app/rollout/` | Phase 3 harness, profiles, action selector |
| `app/rollout/scorer.py` | Phase 4 chapter judge |
| `app/rollout/kb_extractor.py` | Phase 4 KB extraction |
| `app/refinement/` | Phase 5 framework + 3 selectors |
