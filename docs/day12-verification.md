# Day 12 — 20-Chapter Verification (Day 11 Fix Regression Test)

**Date**: 2026-04-13
**Branch**: `worktree-agent-a4ce380a`
**Goal**: Verify the Day 11 bottleneck fix (5/5 commit at 5 chapters) still
holds at 20 chapters — the context-growth regression was Day 10's dominant
failure mode.

## TL;DR

**The fix does not hold at 20 chapters.** Commit rate is **5/20 (25%)**,
worse than Day 10's 32% baseline at 50 chapters and a sharp regression
from Day 11's 100% at 5 chapters. Per-bin commit trend collapses with
context (1/5 → 3/5 → 1/5 → **0/5**). Day 11's closed-enum tool_id and
scene_id realignment eliminated `critic_error` (98 → 1) but exposed a
different dominant failure mode.

Per-dim prose quality on the commits that land remains flat (overall
0.79-0.83, within noise). Latency is modestly better (committed-only
mean 38.3s vs Day 10's 49.4s).

## Run config

Same as Day 10 except `--updates 20` and output directory:

```bash
uv run python tools/stress_test_50.py \
    --updates 20 --n 4 --lora writer_v1 \
    --out data/stress/day12-verify/run_log.jsonl
```

`writer_v1` LoRA at `http://127.0.0.1:8082`, N=4 candidates, retrieval +
scoring + LLM judge all on, same noir seed, same persona cycle.

## Top-line comparison

| metric | Day 10 (50ch) | Day 11 (5ch) | **Day 12 (20ch)** |
|---|---|---|---|
| commit rate | 32% (16/50) | 100% (5/5) | **25% (5/20)** |
| flagged_qm | 68% | 0% | **75%** |
| fallback (crash) | 0% | 0% | **0%** |
| wall-clock total | 1987s | 152.7s | **776.1s** |
| per-update mean wall | 39.7s | 30.5s | **38.8s** |
| committed-only mean wall | 49.4s | ~30s | **38.3s** |
| p50 / p95 (s) | 51.5 / 56.7 | 34.7 / 55.6 | **54.4 / 58.2** |
| context tokens (mean / max) | 279 / 682 | — | **424 / 605** |

## Per-bin commit trend (the core regression signal)

| updates | committed | commit rate | ctx mean | wall mean |
|---|---|---|---|---|
| 1-5 | 1 | 20% | 390 tok | 33.8s |
| 6-10 | 3 | 60% | 425 tok | 38.9s |
| 11-15 | 1 | 20% | 423 tok | 49.9s |
| **16-20** | **0** | **0%** | 456 tok | 32.6s |

**Day 10 pattern replicates.** The last bin is zero — the fix did not
prevent the commit-rate collapse with context growth. Bin 6-10 is
anomalously good (3/5 commit); bins 11-20 degrade as before.

## Per-dim score drift on committed scenes

Only 5 committed updates (4, 6, 7, 8, 12), so statistical power is
low. Per-dim scores across those commits:

| dim | mean | range |
|---|---|---|
| overall_score | 0.801 | [0.784, 0.827] |
| action_fidelity | 0.940 | [0.900, 1.000] |
| detail_characterization | 0.900 | [0.800, 1.000] |
| pacing | 0.774 | [0.524, 1.000] |
| sensory_density | 0.950 | flat |
| free_indirect_quality | 1.000 | flat |
| metaphor_domains_score | 1.000 | flat |
| pov_adherence | 1.000 | flat |
| indirection_score | 1.000 | flat |
| narrator_sensory_match | 0.940 | [0.900, 1.000] |
| named_entity_presence | 0.900 | flat |
| sentence_variance | 0.209 | [0.136, 0.336] |
| dialogue_ratio | 0.000 | flat |

**No degradation on the 5 that commit** — overall range 0.784 → 0.827,
the first commit (update 4) is within noise of the last (update 12).
Quality of landed prose holds across the 20-chapter window; the
bottleneck remains whether CHECK lets anything through.

## Error kinds (20-chapter totals)

| kind | Day 10 (50ch) | **Day 12 (20ch)** |
|---|---|---|
| critic_error | 98 | **1** |
| critic_warning | 58 | **29** |
| craft_fallback | 34 | **9** |
| build_error | 30 | **6** |

Day 11's structural fixes held on two fronts — `critic_error` is
effectively eliminated (98 → 1) and `craft_fallback` dropped in
absolute terms (34/50 → 9/20, similar per-update rate). **But
`critic_warning` now accounts for the CHECK-critical escalations.**

## New bugs surfaced at 20 that didn't appear at 5

1. **`narrator_sensory_match` critic_warning is the new dominant failure
   mode.** 11 of 15 flagged updates have messages of the form
   `narrator sensory distribution drift: L1=0.60-1.14 > 0.60`. The
   pattern: `visual: obs≪target`, `interoceptive: obs≫target=0`. The
   writer LoRA over-produces inner-state / bodily sensation and
   under-produces visual detail for the noir narrator persona.
   Threshold is literally `L1 > 0.60` — update 18's flag at `L1=0.60`
   suggests the threshold is too tight or the measurement is noisy at
   short scene lengths.

2. **`detail_mode 'character_revealing' but no perceptual_preoccupation
   phrase found in prose`** fires on 10 of 20 updates. The craft
   planner is setting scene `detail_mode=character_revealing` but the
   writer isn't emitting a matching perceptual phrase. Either the
   planner is over-tagging scenes as character-revealing, or the
   writer prompt doesn't thread the preoccupation phrase through.

3. **`craft_fallback: no JSON found in text`** re-surfaces (9 cases).
   All look structurally well-formed JSON in the message preview but
   the parser can't locate it — probable `chat()` text is prose-padded
   around the JSON, the regex extractor misses the fence. Same
   shape the Day 11 writeup flagged as "one remaining craft_fallback
   caught by in-band retry"; at 20 chapters this recurs frequently.

4. **Wall-clock in bin 16-20 is fast (32.6s mean) with 0/5 commits** —
   the "fast flagged_qm" pattern Day 10 described is back. CHECK is
   short-circuiting around the 30s mark on prose that never had a
   chance.

## Interpretation

Day 11 fixed the **dramatic/craft structural output** (scene_id types,
tool_id hallucinations). That fix holds at 20 chapters. The new
bottleneck is **writer↔plan alignment** on a narrower front:

- The LoRA writer's sensory distribution drifts away from the noir
  narrator profile as context grows (craft plan says "visual / tactile
  noir"; writer produces interior monologue).
- When the craft planner tags `character_revealing`, the writer
  doesn't thread the `perceptual_preoccupation` phrase.

These would have shown up in Day 11's 5-chapter run as `critic_warning`
noise, but CHECK didn't escalate to critical in those 5 — at 20
chapters, the critic's accumulated warnings push more scenes over the
"critical" line.

## Next (Day 13+ scope)

1. Tighten the writer-LoRA sensory profile to match the noir narrator
   registered target (visual=0.40, tactile=0.20, auditory=0.20,
   interoceptive=0.00). Either via a narrator-conditioning prompt
   fragment or a small targeted SFT pass.
2. Audit `detail_mode=character_revealing` logic — either relax the
   critic's "phrase must appear" rule or tighten the planner's
   tagging.
3. Re-examine CHECK's "critical" escalation threshold (Day 10 tertiary
   recommendation). The `narrator_sensory_match L1=0.60` tripwire
   seems over-aggressive for short scenes.
4. Migrate CraftPlanner to `chat_structured` (Day 10 recommendation #1
   was only partially delivered via xgrammar enum). The 9 residual
   `craft_fallback`s suggest structured-mode + in-band retry is still
   leaving parse errors on the table.

## Files

- Raw log: `data/stress/day12-verify/run_log.jsonl` (gitignored)
- Console log: `data/stress/day12-verify/run_console.log` (gitignored)
- Run script: `tools/stress_test_50.py` (already accepts `--updates`)
- Analyzer: `tools/analyze_stress.py`
