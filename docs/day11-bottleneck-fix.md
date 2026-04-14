# Day 11 — Bottleneck fix (Day 10 follow-through)

**Date**: 2026-04-13
**Branch**: `worktree-agent-adfa1ef4`
**Goal**: Address the 68% chapter rejection rate Day 10's stress test
surfaced — dramatic + craft planners emit invalid structured output, fall
back to stubs, and CHECK flags the chapter.

## What changed

| file | change |
|---|---|
| `app/planning/schemas.py` | `_coerce_scene_id` validator on `ToolSelection`, `DramaticScene`, `EmotionalScenePlan`, `CraftScenePlan`, `CraftBrief` — accepts int / float / str-of-int. |
| `app/planning/dramatic_planner.py` | Closed-enum `tool_id` and `tools_used[]` injected into JSON schema as `enum` so xgrammar enforces at decode time. Renumbers scenes 1..N by position post-parse. In-band ParseError retry. `max_tokens=4096`. |
| `app/planning/emotional_planner.py` | Realigns emotional scene_ids to dramatic scene_ids by position; truncates extras. In-band ParseError retry. `max_tokens=4096`. |
| `app/planning/craft_planner.py` | Same scene_id realignment + truncation for scenes and briefs. In-band ParseError retry. `max_tokens=6144`. |
| `prompts/stages/dramatic/system.j2` | Adds explicit "scene_id MUST be a small positive integer" guidance + closed-enum tool_id list. |
| `prompts/stages/dramatic/user.j2` | Adds Valid-tool-ids section before recommended tools. |
| `prompts/stages/emotional/system.j2` | Adds "scene_id MUST match dramatic scene_id" guidance. |
| `prompts/stages/craft/system.j2` | Same scene_id alignment guidance. |
| `app/engine/pipeline.py` | New ctor kwargs `motif_retriever`, `foreshadowing_retriever`, `scene_retriever`; `_run_hierarchical` now passes them (plus `update_number`) to the planners. Day 10's "constructed-but-never-invoked" wiring bug. |
| `tools/stress_test_50.py` | Wires the three retrievers into the Pipeline. |
| `tools/stress_test_5.py` | New 5-update verification harness; thin wrapper over `stress_test_50`. |
| `tests/engine/test_hierarchical_pipeline.py` | Fake planner signatures accept the new retriever kwargs. |

## Verification (5-chapter A/B against Day 10 conditions)

Both runs use `tools/stress_test_5.py` against vllm at 8082 (writer_v1 LoRA),
N=4 candidates, scoring + LLM judge on, the same noir seed and persona cycle.

### Top-line

| metric | before | after |
|---|---|---|
| **commit rate** | **2/5 (40%)** | **5/5 (100%)** |
| flagged_qm | 3/5 | 0/5 |
| pipeline crashes | 0 | 0 |
| wall-clock total | 229.2 s | 152.7 s |
| latency p50 / p95 | 51.0 / 56.6 s | 34.7 / 55.6 s |

Day 10's commit rate over 50 updates was 32%. The 5-chapter A/B sample
runs the same persona cycle Day 10 used for its first five updates,
so the 40% before / 100% after delta is a direct A/B on the same actions.

### Error-kind distribution

| kind | before | after | Δ |
|---|---|---|---|
| `craft_fallback` | 4 | 1 | -3 |
| `critic_error` | 2 | 1 | -1 |
| `critic_warning` | 6 | 7 | +1 |
| `build_error` | 5 | 9 | +4 |

`craft_fallback` (the dominant Day 10 failure mode at 34/50) drops 75%
in this 5-chapter sample. The remaining instance was caught by the
in-band ParseError retry and never escalated to a stub plan.

`build_error` rises because more chapters commit and so EXTRACT runs
five times instead of two — every committed chapter currently logs
"dropped invalid entity status 'X'" at least once. That's a known
Day-7 issue (the EXTRACT prompt doesn't enumerate the valid status
lexicon to the model) and is *not* in scope for Day 11.

### Retrieval activity

Day 10 reported `motif`, `foreshadowing`, and `voice` as 0 calls/update
(constructed but never invoked). After the wiring fix:

| retriever | mean calls/update | mean hits/update |
|---|---|---|
| passage | 1.20 | 0.00 |
| quest | 1.20 | 1.60 |
| voice | 0.00 | 0.00 |
| motif | 1.00 | 0.00 |
| foreshadowing | 2.00 | 0.00 |

`motif` and `foreshadowing` now actually fire on every craft / dramatic
call. They return zero hits in this 5-chapter run because the seed has
no planted hooks yet and motifs haven't accumulated occurrence history.
That's expected behaviour, not a bug — the retrievers wake up as the
quest accumulates state.

`voice` remains 0 — it requires `pov_character_id` on a scene, which
the dramatic planner still leaves None. That's a separate fix
(populate POV by default) and was not in scope.

### World-state growth

Embeddings: 5 vs 2 — every committed chapter is now embedded, and the
QuestRetriever's pool grows in lock-step with commits. Entities still
flat at 3 (Day-7 EXTRACT issue, see above).

## What remains unfixed

1. **EXTRACT entity status drift** (every commit logs `dropped invalid
   entity status 'X' for Y`). Day-7 fix masks the crash but doesn't
   teach the model the enum lexicon. Every commit drops 1-3 of these.
   Worth a Day-12 prompt-tweak to enumerate `EntityStatus` literally.
2. **Voice retriever still 0 calls** — dramatic planner emits scenes
   with `pov_character_id=None`. Adding a deterministic POV picker
   when the model omits one would unlock the voice retriever.
3. **PassageRetriever 0 hits** — manifest filter still excludes the
   live POV. Day 10 surfaced this; no Day 11 work yet.
4. **Sentence variance flat at 0.17** — the LoRA writer still produces
   uniform-length sentences. Phase-2 prose-depth work, not Day 11.

## Why it works

The two structural fixes do most of the lift:

1. **Closed-enum tool_id in JSON schema** stops xgrammar from generating
   `chekhov_plant` / `map_planting` style hallucinations. The 1.2B was
   inventing plausible-looking tool ids it had seen in rubric examples.
2. **Scene_id realignment in all three planner outputs** turns the
   small-model "everything is scene_id 42" pattern into a no-op. The
   structure critic stops firing, the craft fallback stops firing, and
   CHECK gets a coherent plan to evaluate.

The in-band ParseError retry inside each planner (vs. the outer
`_retry_with_critic` loop) avoids one full critic round-trip per
truncation, which is why the bin 1-5 wall-clock dropped 33% (229s →
153s).

## Next

Day 12 is "re-run 20-chapter verification" per the roadmap. With Day 11's
five chapters at 100% commit, the 20-chapter run should land >80% even
with retriever / EXTRACT residue.
