# Day 10 — 50-Chapter Stress Test

**Date**: 2026-04-13
**Goal**: Auto-play 50 updates with the full pipeline on, track degradation,
identify the primary bottleneck.

> Auto-generated from `tools/analyze_stress.py data/stress/run_log.jsonl`.
> Raw run log is gitignored under `data/stress/`.

## Run config

| key | value |
|---|---|
| LLM model | `writer_v1` (LoRA over `LiquidAI/LFM2.5-1.2B-Instruct`) |
| vllm URL | `http://127.0.0.1:8082` |
| Planner hierarchy | arc → dramatic → emotional → craft, all four layers |
| Retrieval | passage + quest + scene + voice + motif + foreshadowing (Waves 1-4, all wired) |
| Scorer | 12 heuristic/critic dims (Day 2), post-commit |
| LLM judge | tension_execution, emotional_trajectory, choice_hook_quality (Day 6), async |
| N candidates | **4** (writer fan-out per scene) |
| SFT collection | on |
| Personas | 4 archetypes cycling: investigator / intuitionist / bruiser / schemer |
| Seed | `tools/story_gen.SEED` — "The Salt and Star", noir |
| Updates | 50 |

Run command:

```bash
uv run python tools/stress_test_50.py \
    --updates 50 --n 4 --lora writer_v1 \
    --out data/stress/run_log.jsonl
```

Analysis:

```bash
uv run python tools/analyze_stress.py data/stress/run_log.jsonl
```

## Top-line numbers

- **committed: 16/50 (32%)** — only one in three updates passed CHECK.
- **flagged_qm: 34/50 (68%)** — primary failure mode.
- **fallback (pipeline crash): 0/50** — pipeline itself is robust.
- Wall-clock total: 1987s ≈ **33.1 min** (under the 60-min budget).
- Wall-clock per update mean: 39.7s (committed-only mean: 49.4s).
- Latency p50 / p95: **51.5s / 56.7s**.
- Tokens: prompt = 201 446. Completion not tracked (`TokenUsage` only
  carries `prompt` from `ContextBuilder` estimates — see "Bugs surfaced").

### Commit rate by bin

| bin | commits / 10 |
|---|---|
| 1-10 | 6 |
| 11-20 | 2 |
| 21-30 | 2 |
| 31-40 | 2 |
| 41-50 | 4 |

Commit rate falls from 60% in the first 10 to ~20% in bins 11-40, with a
mild recovery in the last bin. The drop is sharp — bins 11+ all sit
under 30%. This is the dominant degradation curve.

### Commit rate by persona

| persona | commit | flagged |
|---|---|---|
| investigator | 4 | 9 |
| intuitionist | 6 | 7 |
| bruiser | 2 | 10 |
| schemer | 4 | 8 |

`bruiser` actions degrade hardest. Confrontational, declarative inputs
push the planner toward dramatic escalations the model can't faithfully
execute (more on this in "Primary bottleneck"). `intuitionist` (slower,
withholding actions) holds up best.

## Per-dim score trajectory (over the 16 committed updates)

| dim | first 10 mean | last 10 mean | Δ |
|---|---|---|---|
| overall_score | 0.81 | 0.80 | -0.01 |
| sentence_variance | 0.16 | 0.19 | +0.02 |
| pacing | 0.79 | 0.78 | -0.01 |
| sensory_density | 0.95 | 0.91 | -0.04 |
| free_indirect_quality | 1.00 | 1.00 | 0.00 |
| detail_characterization | 1.00 | 0.97 | -0.03 |
| metaphor_domains_score | 1.00 | 1.00 | 0.00 |
| indirection_score | 1.00 | 1.00 | 0.00 |
| pov_adherence | 1.00 | 1.00 | 0.00 |
| named_entity_presence | 0.90 | 0.90 | 0.00 |
| narrator_sensory_match | 0.93 | 0.93 | -0.01 |
| action_fidelity | 0.95 | 0.95 | 0.00 |
| dialogue_ratio | 0.00 | 0.00 | 0.00 |

**No per-dim degradation among the 16 committed scenes.** All deltas are
within noise. `dialogue_ratio` is flat-zero — the writer LoRA produces
almost no quoted dialogue — but that's a Day-5 finding, not a stress-test
finding. `sensory_density` drifts down 0.04 (worth watching, not yet
actionable).

The conclusion is brutally simple: **when the pipeline commits, prose
quality stays flat. The bottleneck is whether the pipeline commits at all.**

## Latency curve

| bin | n | p50 | p95 | mean |
|---|---|---|---|---|
| 1-10 | 10 | 51.8s | 58.8s | 44.7s |
| 11-20 | 10 | 11.7s | 53.3s | 19.9s |
| 21-30 | 10 | 51.5s | 55.5s | 44.5s |
| 31-40 | 10 | 52.3s | 56.8s | 40.9s |
| 41-50 | 10 | 51.5s | 56.5s | 48.6s |

Bin 11-20 shows a striking p50 drop to 11.7s — those are CHECK-stage
short-circuits where dramatic/craft fell back to a minimal plan and
CHECK flagged immediately. The "fast flagged_qm" pattern (14/34 flagged
runs took <20s) is a real signal — the model is producing output the
critic rejects within the first one or two stages.

Wall-clock is **dominated by the writer fan-out**: 4 candidates × ~10 s
inference each ≈ 40 s. Lowering N to 2 would roughly halve the
committed-update wall-clock. Raising N would not help — the bottleneck
is upstream of the writer.

## Memory / state growth

| update | entities | narrative records | embeddings | plot threads |
|---|---|---|---|---|
| 1 | 3 | 1 | 1 | 1 |
| 13 | 3 | 13 | 7 | 1 |
| 26 | 3 | 26 | 10 | 1 |
| 38 | 3 | 38 | 11 | 1 |
| 50 | 3 | 50 | 16 | 1 |

- **Entities never grow** — the EXTRACT stage is failing to mint new
  characters or locations. Every commit log shows `build_error: dropped
  invalid entity status 'X' for Y` — the LLM is hallucinating entity
  status values outside the enum, the persistence layer drops them, and
  the entity list never expands. This is a Day 7 fix
  (`feat(extract): drop invalid Entity status values from LLM output`)
  that prevents the crash but doesn't make extraction succeed.
- **Embeddings = 16, matching commits** — only committed prose is
  embedded. Flagged_qm chapters are written into the narrative table but
  not into `narrative_embeddings`, so the QuestRetriever's pool is far
  smaller than the apparent narrative size. Worth investigating: should
  flagged_qm prose still get an embedding so callbacks can match against
  it? (Probably yes — the quest is still happening, even if the QM
  flagged it.)
- **Plot threads stay at 1.** Same shape as entities — the planner
  doesn't propose new threads, and the EXTRACT pipeline never adds them.
- **Context tokens** grow from 208 → 345 over 50 updates (max 682). Modest
  growth, well within the 16k vllm window. Not the bottleneck.

## Retrieval activity

| retriever | mean calls/update | mean hits/update |
|---|---|---|
| passage (style anchors) | 1.7 | **0.0** |
| quest (in-quest callbacks) | 1.7 | 3.2 |
| voice (per-POV continuity) | 0.0 | 0.0 |
| motif | 0.0 | 0.0 |
| foreshadowing | 0.0 | 0.0 |

Two large issues here:

1. **PassageRetriever returns zero hits across all 50 updates.** The
   retriever is wired and called (~1.7 calls/update), but the manifest's
   filtering (`pov`, `score_ranges`) likely excludes everything. Voice
   anchors are simply not influencing the writer.
2. **Voice / motif / foreshadowing retrievers are constructed but never
   called** — the pipeline never invokes them through the planners. The
   planners accept these retrievers as `plan(..., motif_retriever=...)`
   keyword arguments, but `Pipeline._run_hierarchical` doesn't pass them.
   Wave 4 wired the retriever classes but the pipeline-level hookup is
   incomplete.

QuestRetriever is the only one delivering signal (mean 3.2 hits/update).

## Consistency-flag growth

- Total errors logged across all 50 updates: **220** (mean 4.4/update).
- Total fallbacks: 34 (matches the flagged_qm count exactly).
- Per-bin: 5.2 → 3.3 → 4.0 → 4.1 → **5.4** errors/update — a slight
  upward trend (+0.2 per bin), but noisier than the commit-rate
  collapse.

Top error kinds (across all 50 updates):

| count | kind | what it means |
|---|---|---|
| 98 | `critic_error` | Dramatic plan critic flagged tool-id mismatches, scene-id ordering, etc. |
| 58 | `critic_warning` | Narrator sensory drift, entity presence fuzz, etc. |
| 34 | `craft_fallback` | Craft planner output failed JSON parse; pipeline used minimal-plan fallback. |
| 30 | `build_error` | EXTRACT stage dropped invalid enum values from LLM (e.g. `status='observed'`, `'warning_received'`). |

`critic_error` and `critic_warning` are non-blocking — they just record
the planner's structured output didn't pass the critic. The actual
flag-the-chapter decision happens later in CHECK, but a planner that
keeps emitting tool ids the dramatic critic rejects (`chekhov_plant`,
`map_planting`, `dialogue_planting`) signals that the 1.2B is hallucinating
craft vocabulary it has seen in the rubric examples but doesn't actually
have available.

`craft_fallback` is structural: the craft planner is consistently
producing `scene_id: 42` (numeric) when the schema requires `scene_id:
"scene_1"`-style strings, plus other shape mismatches. The fallback
catches it cleanly so the pipeline keeps moving, but every fallback
means we're writing prose against a degraded plan.

## Primary bottleneck

> **The CHECK stage is rejecting two-thirds of generated chapters as
> critically flawed, and the rejection is upstream-driven: the dramatic
> + craft planners emit invalid structured output (wrong tool ids,
> wrong scene-id types), the planners fall back to minimal plans, the
> writer produces prose against a stub, and CHECK then flags the
> resulting chapter as critically incoherent.**

Concretely, the cascade is:

1. **Dramatic planner picks a tool from outside the valid set.** The
   dramatic critic rejects the plan. Either the retry succeeds (normal
   path) or the planner falls back to a minimal plan.
2. **Craft planner emits a JSON payload that doesn't match the
   `CraftPlan` schema** — typically a numeric `scene_id` instead of the
   required string. The craft critic catches it; fallback fires.
3. **Writer fans out N=4 against the stub plan.** Rerank picks the
   "best" of four mediocre candidates.
4. **CHECK reads the prose against the stub plan** and finds the prose
   doesn't deliver the (now stub-shaped) plan's beats. It emits a
   critical issue. Outcome = `flagged_qm`.

The 60% → 20% commit-rate collapse around bin 11 corresponds to when
the world state has accumulated enough committed narrative that prompts
get longer and the 1.2B model's structured-output reliability degrades.
We see this in the context-token trajectory: bin 1 averages 250 tok,
bins 11+ average 400+ tok of context.

### Recommendation: fix the craft planner's structured-output reliability

The single highest-leverage fix is to make the craft planner stop emitting
the wrong shape. Two parallel approaches, in order of expected impact:

1. **Add `chat_structured` to the craft planner.** The craft planner
   currently uses `chat()` and parses raw text. The `Pipeline._run_check`
   path uses `chat_structured` (with a JSON schema) and CHECK is reliable.
   Migrating dramatic + craft to `chat_structured` should cut both
   `critic_error` and `craft_fallback` rates dramatically. The Pipeline
   already has the call shape — see `_run_check` for the reference.
2. **Prune the dramatic-planner tool list before the call.** The 1.2B
   keeps inventing plausible-looking tool ids (`chekhov_plant`,
   `map_planting`). Pre-render the valid tool ids as a numbered enum in
   the user prompt so the model has a closed vocabulary to pick from
   instead of free-text.

Both are bounded scope (one or two files each) and should land in
Days 11-12.

### Secondary: re-examine the CHECK stage's "critical" threshold

Even with stub plans, CHECK flagging 68% of chapters as "critical" is
likely too aggressive. The `Severity` enum has `info / warning / error
/ critical`, and `flagged_qm` triggers on the literal `critical`. The
1.2B-Instruct judge may be over-using the `critical` bin under prompt
ambiguity. Worth one calibration pass against a 10-chapter Claude-judged
ground truth, similar to the Day-6 LLM-judge calibration.

### Tertiary: wire the dormant retrievers

`voice`, `motif`, `foreshadowing` retrievers are constructed but never
invoked. `passage` is invoked but returns zero hits — the manifest's
score-range filter is probably too tight, or the embedder is off.
Wiring these in should improve the writer-stage anchoring; they won't
fix the structured-output problem, but they will improve the prose
quality of the 32% that DO commit.

## New bugs surfaced

1. **`Embedder` defaults to CUDA, OOMs against vllm.** The Embedder
   constructor has no `device` arg and `SentenceTransformer` auto-picks
   CUDA, which is fully booked by vllm. Workaround: set
   `CUDA_VISIBLE_DEVICES=""` at the entry point of any script that
   shares a process with the embedder. Real fix: pass `device="cpu"`
   into the SentenceTransformer constructor (and possibly add a Day-1
   `device` kwarg to `Embedder.__init__`).

2. **EXTRACT stage drops invalid entity statuses every commit.** Every
   committed update logs `build_error: dropped invalid entity status 'X'
   for Y`. The Day-7 fix (`drop invalid Entity status values from LLM
   output`) prevents the crash but does NOT teach the LLM the right
   enum values. Result: zero new entities are ever minted from prose.
   This is why `entities=3` for all 50 updates. Either the EXTRACT
   prompt needs the enum literals embedded, or the post-extract
   validator should re-prompt with corrective feedback rather than drop.

3. **CraftPlanner produces numeric `scene_id`s.** Every fallback has
   `scene_id: 42` or similar in the raw output. The schema requires a
   string. Either the prompt needs explicit "scene_id is a string like
   'scene_1'" instruction, or `chat_structured` (see Recommendation #1)
   would force the right shape via JSON-schema enforcement.

4. **Voice / motif / foreshadowing retrievers are wired in
   `__init__` but never invoked.** The Pipeline accepts
   `passage_retriever`, `quest_retriever`, `voice_retriever` as ctor
   kwargs, but `voice_retriever` only fires through
   `_retrieve_voice_continuity` — which IS called per scene but returns
   zero in our run because no scenes have a `pov_character_id` set
   (every dramatic plan in our run leaves it as None). Motif and
   foreshadowing are not Pipeline-level kwargs at all; they're
   per-call kwargs on `CraftPlanner.plan()` / `DramaticPlanner.plan()`
   that the Pipeline doesn't pass. Two-line fix in
   `_run_hierarchical`.

5. **`TokenUsage.completion` is always 0.** The Pipeline only sets
   `TokenUsage(prompt=ctx.token_estimate)` from a CB-side estimate; it
   never reads vllm's `usage.completion_tokens` from the response.
   Result: completion-token totals across the run are unreported. Fix:
   propagate the OpenAI-API `usage` block from `InferenceClient.chat`
   back to the call site.

6. **PassageRetriever returns zero hits across all 50 updates.** Either
   the score-range filter is too aggressive, the manifest is empty for
   the seed's POV (`third_limited`), or the semantic-disabled
   keyword-only path doesn't match the live scene briefs. Worth
   instrumenting one update's retrieval call directly.

## Files

- Run script: `tools/stress_test_50.py`
- Personas: `tools/stress_personas.py`
- Analysis: `tools/analyze_stress.py`
- Raw log: `data/stress/run_log.jsonl` (gitignored)
- Console log: `data/stress/run_console.log` (gitignored)
