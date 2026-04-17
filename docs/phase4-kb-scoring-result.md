# Phase 4 — Scoring + KB Extraction — Result

**Date:** 2026-04-16
**Spec:** `docs/superpowers/specs/2026-04-15-story-rollout-architecture.md`
**Exit criterion:** *for a completed rollout set, we can query "which seed hooks paid off in <50% of rollouts" and "which chapter indices score lowest on subtext_presence" from the UI.*

**Status:** ✅ all storage, API, and pipeline integration complete; UI tab deferred.

## Implementation landed

**Storage** (`app/world/db.py`, `state_manager.py`):
- 3 KB tables: `kb_chapter_scores`, `kb_hook_payoffs`, `kb_entity_usage` with appropriate indices
- WSM CRUD: `save_chapter_scores`, `list_chapter_scores`, `save_hook_payoff`, `list_hook_payoffs`, `save_entity_usage`, `list_entity_usage`
- 7 storage tests

**Scorer** (`app/rollout/scorer.py`):
- `score_chapter()` — single batched structured-output call to the 8-dim chapter judge (rubrics under `prompts/scoring/chapter_dims/`); same dim set as `tools/judge_chapters.py` and `tools/strategy_sweep.py`
- `score_and_persist_chapter()` — also writes per-dim rows to `kb_chapter_scores` and updates the chapter row's `judge_scores` blob; idempotent (no LLM call if scores already present)
- `thinking=False` enforced (Gemma was leaking chain-of-thought tokens that broke JSON parsing)
- Defensive JSON extraction (skip prefix garbage)
- 5 scorer tests

**KB extractor** (`app/rollout/kb_extractor.py`):
- `extract_hook_events()` — pulls hook id + new_status from extract stage's `foreshadowing_updates`
- `extract_entity_introductions()` — DORMANT→ACTIVE entity_updates from extract
- `extract_thread_advances()` — from dramatic plan's `thread_advances`
- `find_entity_mentions()` — word-boundary case-insensitive name match in prose (avoids "Lan" matching "Lanier")
- `persist_chapter_kb()` — orchestrator that writes all of the above into `kb_hook_payoffs` + `kb_entity_usage`
- 8 extractor tests

**Harness wiring** (`app/rollout/harness.py`):
- After every saved chapter: KB extraction always runs (no LLM cost), scorer runs when `score=True` (default)
- Both wrapped in best-effort try/except — extraction or scoring failures never break a rollout
- Harness `score: bool = True` parameter
- 2 harness tests verifying both paths

**API** (`app/server.py`):
- `GET /api/quests/{qid}/kb` — aggregated views: hook payoffs (with payoff_rate across rollouts), entity usage (screen_time across rollouts), per-chapter-index dim means
- `GET /api/quests/{qid}/rollouts/{rid}/scores` — per-chapter dim breakdown with rationales
- 3 API tests

**Total delta:** +25 tests (7 storage + 5 scorer + 8 extractor + 2 harness wiring + 3 API). 871 → 896 tests passing.

## Empirical result — scoring v5's rollout

Scored the existing `ro_dc5f5331` chapter 1 (553 words) against the 8-dim judge. Single LLM call, ~10s.

| Dim | v5 ch1 | Pale Lights ch1 | Δ |
|---|---|---|---|
| tension_execution | 0.70 | 0.90 | -0.20 |
| emotional_trajectory | 0.60 | 0.90 | **-0.30** |
| choice_hook_quality | 0.70 | 0.70 | 0.00 |
| update_self_containment | 0.70 | 0.70 | 0.00 |
| voice_distinctiveness | 0.60 | 0.60 | 0.00 |
| thematic_presence | 0.60 | 0.90 | **-0.30** |
| subtext_presence | 0.30 | 0.60 | **-0.30** |
| interiority_depth | 0.60 | 0.70 | -0.10 |
| **mean** | **0.6** | **0.75** | **-0.15** |

The v5 chapter is materially below Pale Lights on emotional_trajectory, thematic_presence, and subtext_presence. Notable from the rationales:

- *subtext_presence (0.30):* "Most of the tension is expressed through direct sensation and narration, though the ambiguity of whether the 'ticking' is caused by the man or the name provides a small layer of mystery."
- *emotional_trajectory (0.60):* "The chapter moves from the frantic, physical disorientation of the sea to a focused, predatory stillness in the market, pivoting when Tristan decides to stop watching and start moving."
- *thematic_presence (0.60):* "The theme of predator vs. prey is worked through the imagery of rats, mice, needles, and bells."

Several factors pull the score down:
1. **Length.** 553 words leaves no room for the slow build Pale Lights uses for emotional and thematic crescendos. (This was caused by the revise-truncation bug, now fixed.)
2. **Subtext.** The judge correctly identifies that Tristan's tension is rendered as direct physical sensation rather than implication — Pale Lights does this through what characters *don't* say to each other.
3. **Thematic articulation.** The chapter touches the predator/prey theme but doesn't develop it across multiple beats.

## KB extraction findings

For v5 chapter 1:

- **hooks_planted: []** — the dramatic plan didn't plant any new hooks
- **hooks_paid_off: []** — and didn't pay off existing ones
- **entities_introduced: []** — no DORMANT→ACTIVE in extract stage
- **entities_mentioned: ['char:cozme', 'char:fortuna', 'loc:bluebell']** — name-match found these in prose
- **thread_advances: {'pt:tristans_list': ''}** — the "tristans_list" plot thread got an advance but with empty target

The mention-match is the most useful signal in this run; the planner outputs (hooks, introductions, advances) were thin because the chapter was short.

## API verification

```bash
$ curl /api/quests/pale_lights/kb
{"n_rollouts": 2, "hook_payoffs": [], "entity_usage": [
  {"entity_id": "char:cozme", "screen_time": 1, "rows": [...]},
  {"entity_id": "char:fortuna", "screen_time": 1, "rows": [...]},
  {"entity_id": "loc:bluebell", "screen_time": 1, "rows": [...]}
]}

$ curl /api/quests/pale_lights/rollouts/ro_dc5f5331/scores
{"rollout_id": "ro_dc5f5331", "chapters": [{"chapter_index": 1,
  "dims": {"tension_execution": {"score": 0.7, "rationale": "..."}, ...}}]}
```

Both endpoints live and aggregating correctly.

## Gaps / follow-ups

1. **UI tab deferred.** The "Rollout Analytics" drawer tab from the spec is not built. The current API surface is sufficient to drive analytics from CLI / curl; UI addition is a small follow-up pass.

2. **Single rollout = thin KB.** Aggregation queries like "which hook paid off in <50% of rollouts" need many rollouts to be meaningful. Phase 3's exit criterion (2×2×10 overnight) would generate enough data; we deferred that. With one scored chapter the KB is correct but unsatisfying as an analytics view.

3. **Score doesn't compare to baseline automatically.** The result above is hand-compared to Pale Lights. A `/kb/baseline` endpoint that returns the corpus baseline + per-dim deltas would make analytics much more actionable.

4. **No regression tracking.** When the same rollout chapter is re-scored after a model/seed change, we lose the old scores (upsert by primary key). For Phase 5 refinement we'll want to keep a history.

## Phase 5 readiness

Phase 4 gives us:
- A score on every rollout chapter
- Per-chapter, per-dim aggregation across rollouts
- Hook + entity coverage data per rollout

Phase 5 (refinement) can now:
- Identify the weakest chapter index by mean dim score
- Identify hooks that fail to pay off in most rollouts (refine the skeleton)
- Identify entities that never get screen time (drop or surface)
- Drive targeted rewrites of below-threshold chapters using the KB as guidance

Proceed when ready.
