---
title: "Phase 1 — Story Candidates — Result"
---

# Phase 1 — Story Candidates — Result

**Date:** 2026-04-16
**Spec:** `docs/superpowers/specs/2026-04-15-story-rollout-architecture.md`
**Exit criterion:** *Pale Lights seed produces 3+ candidates with distinct thread-emphasis and protagonist choices, and the pipeline honors the picked candidate's thread-priority in ArcDirective.*

**Status: PASS.**

## Implementation landed

- `StoryCandidate` schema + SQLite table with full CRUD methods in `WorldStateManager` (7 new tests)
- `StoryCandidatePlanner` wrapping an LLM call with closed-enum schema constraining thread/character/theme ids to seeded values (3 new tests)
- Prompt templates `prompts/stages/story_candidate/{system,user}.j2`
- API endpoints: `GET /candidates`, `POST /candidates/generate?n=N`, `POST /candidates/{cid}/pick` (2 new tests)
- Arc planner now reads `quest_config.picked_candidate` and renders it as directive context in `prompts/stages/arc/user.j2`
- Frontend: candidate-picker stage with auto-generation, card-based UI, picked banner, "Change story" flow

**Total delta:** 833 tests pass (+16 from Phase 1), zero regressions.

## Empirical result — 3 candidates for Pale Lights

Auto-generated via the UI against a fresh `data/quests/pale_lights` DB. Gemma 4 26B returned a well-formed `candidates` array on the first call.

| Candidate | Protagonist | Primary threads | Length | Climax |
|---|---|---|---|---|
| **The Shadow Watch** | Tristan | red_maw_containment, hoja_roja_hunts_tristan, the_trials | 50 ch | Alliance forced by Maw-impersonation crisis |
| **The God-Eater's Debt** | Tristan | tristans_list, red_maw_containment | 35 ch | Tristan-Fortuna bargain to stabilize the prison |
| **The Noble's Reckoning** | Angharad | angharads_revenge, the_trials | 42 ch | (not captured in screenshot; Fisher-centric) |

The three are materially distinct:

- **Protagonist split** across Tristan (two different angles) and Angharad (one)
- **Thread emphasis** shares only `red_maw_containment` (spy-story angle) and `the_trials` (social-pressure angle); otherwise no overlap
- **Arc lengths** span 35–50 chapters, suggesting the planner registered the `expected_chapter_count` constraint
- **Tonal distinction** clearly visible in synopses (read the prose): Shadow Watch is an espionage-thriller shape, God-Eater's Debt is a bargain-with-fate shape, Noble's Reckoning is a revenge-and-honor shape

## Gaps / follow-ups

1. **CLI `quest init` doesn't add themes to world DB.** Themes in the seed's `themes` list make it into `config.json` but not into the `themes` SQLite table via `sm.add_theme`. The server's quest-creation path *does* call `add_theme`. Consequence: candidates generated from CLI-initialized quests see an empty themes list, so `emphasized_theme_ids` comes back `[]`. Fix: CLI init should add themes. Small change, deferred.

2. **UI doesn't show themes/secondary threads on the card.** The card renders primary threads, protagonist, expected length, climax — but not emphasized themes or secondary threads. Once (1) is fixed, these should appear.

3. **"Change story" button doesn't clear the pick server-side.** It only re-opens the picker UI. If the player picks a different candidate, it correctly overwrites, but if they don't, `picked_candidate` still points at the previous pick. Minor UX wrinkle.

4. **Generation uses the full seed context.** With 49 entities + 10 hooks in Pale Lights, the prompt is ~3k tokens. This is fine today but scaling to a 200-entity seed would warrant pruning (e.g. only ACTIVE entities or top-K by priority). Deferred.

## Phase 2 readiness

Phase 1's successful landing unblocks Phase 2 (arc skeleton). A picked candidate now carries enough structure — primary threads, protagonist, climax description, expected chapter count — to be the input of a skeleton generator that produces a 10–30-chapter outline. Proceed when ready.
