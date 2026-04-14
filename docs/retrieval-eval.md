# Retrieval A/B — first run

**Date:** 2026-04-14, post-Wave 4 merge
**Setup:** 692 tests passing on master; LFM2.5-1.2B-Instruct-Q4_K_M served
via llama.cpp at 16k context. 3 player actions, same noir test seed
(`tools/story_gen.py`).

## Results

| | Baseline (Apr 13) | Retrieval ON (Apr 14) |
|---|---|---|
| Commits succeeded | 1/3 (others `flagged_qm`) | **3/3** |
| Stages per action | 7-8 | 10 (extra retrieval calls) |
| Prose length | ~150-200 words | ~150-200 words |

### Action 1 prose comparison

**Baseline:** *"You traced the edge of the table, fingers brushing the
edge where the light hit just so. The room hummed with a quiet tension,
a breath held between shadows..."*

**Retrieval:** *"You reached for the book, fingers brushing the spine
as if it might whisper your name. The room held its breath, the hum
of the fan a soft counterpoint to your focus..."*

### Action 2 prose comparison

**Baseline:** *"You sit at Merrin's bar, the amber light spilling
across your face as you lean forward, voice low. You ask if the Gannet
crew came through last week, the words catching in your throat..."*

**Retrieval:** *"You sit at Merrin's bar, the cool wood beneath your
palms grounding you as you lean forward, fingers tightening on the
edge of the table. The air smells of rain and old paper..."*

## Findings

### What helped

- **Stability**: 3/3 commits vs 1/3. The voice anchors gave the check
  stage less to flag (POV more consistent, fewer impossible verb
  combinations).
- **Sensory specificity**: "rain and old paper" is more concrete than
  "amber light spilling across your face". The MiniLM-anchored
  passages did push the writer toward grounded sensory detail.

### What didn't help (yet)

- **Cliché density barely moved**: "held its breath", "whisper your
  name", "rippled forward like water catching light" still appear in
  every action. The writer's defaults dominate even with anchors.
- **Voice still generic**: the configured narrator ("weathered
  observer who notices hands and silences") doesn't show through. The
  voice samples we passed in (`"She set the cup down the way she did
  everything else — like the cup owed her rent."`) didn't anchor the
  writer's rhythm.
- **No callbacks fired**: action 1 has no quest history yet (expected),
  but action 2 didn't reference action 1's setup either. QuestRetriever
  has rows after a commit but the seed_text + entity filters may not
  be matching enough.
- **MotifRetriever, ForeshadowingRetriever**: quest has no motifs/hooks
  yet — these are empty for cold-start. Won't show benefit until a
  longer-running quest with planted hooks.

### Why anchors didn't dominate

Three hypotheses:

1. **Anchor weight in prompt**: anchors render above the prose brief
   but as one sentence ("VOICE ANCHORS — passages to match in style
   and rhythm"). The model probably treats them as background context
   rather than primary instruction.
2. **Anchor selection**: we filter on POV + voice_distinctiveness ≥
   0.7, which pulls *generic-good* prose, not *narrator-specific*
   prose. A narrator-aware filter (similar `attention_bias`,
   `editorial_stance`) would tighten the pool.
3. **Model capacity**: LFM-1.2B has ~12B effective parameters in
   weights; complex stylistic mimicry may exceed what we can pull
   from anchor injection alone. The writer LoRA is the path that
   addresses this directly.

## Next steps

### Short term

- **Stronger anchor instruction**: rephrase the prompt block from
  "passages to match" to a directive like "Write in the cadence and
  diction of these examples; do not introduce phrases or images
  outside their range." Then re-A/B.
- **Narrator-aware retrieval**: add narrator metadata fields to the
  PassageRetriever filter (sensory_bias overlap, attention_bias
  intersection) so anchors match the configured narrator, not just any
  high-quality literary prose.
- **Multi-update test**: rerun with 8-10 actions. Cold start is the
  worst case for retrieval; after a few commits, QuestRetriever +
  VoiceRetriever should start contributing meaningfully.

### Medium term

- **Writer LoRA finetune** (per `docs/writer-finetune-plan.md`). This
  is where the structural style problem actually gets solved — the
  base model needs to learn the quest-fiction prose shape, not just
  read examples in context.
- **vllm migration** for production throughput. At llama.cpp's 670
  tok/s, retrieval-augmented generation is 5-8 sec per scene. vllm
  3k tok/s makes generate-N rerank with N≥10 viable.

## Conclusion

Retrieval shipped end-to-end and is **not actively harmful** — the
pipeline is more stable, prose is marginally more grounded, and the
infrastructure is in place for the long-tail retrievers (motif,
foreshadowing, quest history) to contribute as quests accumulate state.

But it's **not yet decisive**. The next lever is the writer LoRA;
retrieval-only improvements have hit diminishing returns on a base
1.2B model. Treat retrieval as the substrate that makes a finetuned
writer trainable on diverse, grounded prose, not as a replacement for
finetuning.

648 tests. All passes. Pushed to `origin/master`.
