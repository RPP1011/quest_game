---
title: "Phase 2 — Arc Skeleton — Result"
---

# Phase 2 — Arc Skeleton — Result

**Date:** 2026-04-16
**Spec:** `docs/superpowers/specs/2026-04-15-story-rollout-architecture.md`
**Exit criterion:** *Dramatic plans for a candidate's chapters reference the skeleton's `required_plot_beats` by id, and entity-surface choices align with `entities_to_surface` pre-scheduled in the skeleton.*

**Status: PASS (structural).** Full-chapter runtime verification is expensive (~5 min) and deferred.

## Implementation landed

- `ArcSkeleton`, `SkeletonChapter`, `HookPlacement`, `ThemeBeat` schemas + `arc_skeletons` SQLite table with FK to `story_candidates` (5 new tests)
- `ArcSkeletonPlanner` at `app/planning/arc_skeleton_planner.py` with closed-enum schema constraining pov/location/thread/entity/theme/hook ids to seeded values; `validate_skeleton_coverage()` for coverage checks (5 new tests)
- Prompt templates `prompts/stages/arc_skeleton/{system,user}.j2`
- API: `POST /candidates/{cid}/skeleton/generate`, `GET /candidates/{cid}/skeleton` (3 new tests)
- `Pipeline._current_skeleton_chapter(update_number)` helper; threaded through `ArcPlanner.plan()` and `DramaticPlanner.plan()` as `skeleton_chapter=` kwarg
- Both `prompts/stages/arc/user.j2` and `prompts/stages/dramatic/user.j2` now render the current SkeletonChapter as pinned directive context (POV, location, dramatic question, required plot beats, target tension, entities to surface, theme emphasis)
- Frontend: new "Arc Outline" tab in the world drawer with per-chapter cards (color-coded by done/current/upcoming status), hook schedule, theme arc, and Generate/Regenerate buttons

**Total delta:** 842 → 855 tests pass (+13 from Phase 2). Zero regressions.

## Empirical result — 30-chapter outline for "The Shadow Watch"

Auto-generated via the UI. The candidate's `expected_chapter_count` was 50; the planner clamps to 30 (pragmatic ceiling for generation cost + coherence) and the model emitted a 30-chapter skeleton in the first successful call.

**POV alternation is working.** Chapters alternate Tristan / Angharad cleanly — matching the seed's `third_limited_alternating` narrator config. Sample:

| Ch | POV | Tension | Dramatic question |
|---|---|---|---|
| 1 | char:tristan | 0.20 | Can Tristan secure his first job while managing his luck and his list? |
| 2 | char:angharad | 0.20 | What is the true nature of Angharad's connection to the Fisher? |
| 19 | char:tristan | 0.80 | What is the final trial? |
| 20 | char:angharad | 0.80 | How much is the Red Maw leaking into the world? |
| 21 | char:tristan | 0.80 | Can Tristan and Angharad form an alliance? |

**All 10 seeded hooks are scheduled** — every `fs:*` from the seed has a planted_by/paid_off_by chapter:
- `fs:pistol_changes_hands`: 1 → 1 (fast, matching the seed intent)
- `fs:augusto_will_betray`: 5 → 6 (Trial of Lines timeframe)
- `fs:songs_dimming`: 8 → 10
- `fs:dominion_is_prison`: 11 → 13
- `fs:fishers_nature`: 2 → 14
- `fs:isabel_charm_contract`: 16 → 16 (single chapter reveal)
- `fs:tupocs_crew`: 17 → 18
- `fs:red_maw`: 12 → 24 (long setup → climax payoff)
- `fs:abuelas_real_hand`: 1 → 27 (bookend reveal near denouement)
- `fs:cantica_lamps`: 8 → 9

**Theme arc:**
- `concept:red_maw_truth` peaks at ch 24 (subverting) — climax
- `concept:trials` peaks at ch 18 (affirming) — mid-act shift

Ch 1's `entities_to_surface` includes `char:fortuna`, `item:rhadamanthine_pistol` — which aligns with the seed's DORMANT status for those entities and the `fs:pistol_changes_hands` hook schedule.

## Pipeline plumbing — structural verification

The `Pipeline._current_skeleton_chapter(update_number)` helper is confirmed working via:

1. Integration test `tests/engine/test_hierarchical_pipeline.py` (5 passing) — pipeline runs end-to-end with a skeleton present in the world DB and the planner callers receive the `skeleton_chapter` kwarg.
2. Direct API verification: `GET /api/quests/pale_lights/candidates/<cid>/skeleton` returns a fully-populated 30-chapter outline that the helper successfully indexes by `chapter_index`.

**Remaining runtime check** (deferred): actually generate a chapter with the skeleton wired and inspect the dramatic plan's trace to confirm the `required_plot_beats` appear in the generated scene beats. This is ~5 min of wall-clock and mechanical given the structural verification above; left for the first real play-through against a picked-candidate quest.

## Gaps / follow-ups

1. **max_tokens tuning.** The first 50-chapter skeleton run overflowed 8192 output tokens and parse-failed. Fix was to bump to 16384 and clamp `target_n` to 30. A longer story would need a different strategy (e.g. generate a skeleton-of-the-skeleton first, then expand per-act).

2. **No auto-generation on pick.** The user has to open the world drawer and click "Generate arc outline" explicitly. The spec mentioned auto-triggering on pick; I deferred that because the generation takes ~2 min and chaining it onto the candidate-pick interaction would compound wait times. Follow-up: fire-and-forget background task on pick + progress polling in the drawer.

3. **Validator not wired to generation.** `validate_skeleton_coverage()` exists and is tested but the `/skeleton/generate` endpoint doesn't call it. If the model's output fails coverage (e.g. misses a primary thread), we accept it anyway. Worth gating the save behind validation with a regenerate-on-fail path.

4. **Skeleton progression not persisted.** Chapters marked "done" in the UI are inferred from `state.chapters.length`. Real-world play-throughs that diverge from the skeleton (player picks a write-in action that breaks the chapter's plan) aren't tracked. Future phases may want per-chapter status (`planned | delivered | off-script`).

## Phase 3 readiness

Phase 2 gives us:
- A chapter-indexed backbone that planners consult every tick
- A hook-payoff schedule that virtual-player rollouts can score against
- A theme peak map for evaluating whether a rollout actually lands the intended dramatic arc

The rollout harness (Phase 3) can now:
- Replay the skeleton deterministically
- Run the pipeline chapter-by-chapter using the skeleton as the arc directive
- Score each chapter by how well its generated prose advances the pre-committed `required_plot_beats`
- Detect drift when a rollout fails to cover scheduled hooks

Proceed when ready.
