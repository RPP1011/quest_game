---
title: Quest Engine — Technical Report
description: >-
  Building a hierarchical narrative-generation engine around a 1.2B
  local model. What it is, why it's hard, what we've measured.
---

# Quest Engine

*A hierarchical narrative-generation engine for text-based quest games,
running locally on a 1.2B open-weight model.*

> **Status**: Pipeline works end-to-end. Commit rate on a 20-chapter
> stress test is 100%. Prose quality is decent-not-great and reports
> honestly below. Source at
> [github.com/RPP1011/quest_game](https://github.com/RPP1011/quest_game).

---

## What we're building

The goal is a local quest-playing engine that produces prose *meaningfully
better than what you'd find on Royal Road* — not "surprisingly okay for AI",
but prose you'd actually want to read. The personal target is something
closer to Abercrombie on engagement and Pale Lights on voice
distinctiveness, running entirely on one RTX 4090.

The constraint that makes this interesting: the writer model is
[LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).
That's a 1.2-billion-parameter model. It fits the "can generate text
fast on a single GPU via vllm" box. It does not fit the "can hold a
50-chapter arc in its head" box. If you ask it, without scaffolding, to
continue a noir quest, it will give you something like:

> "The protagonist truly wants to understand what drives them..."

That is a craft-brief paraphrase, not a scene. Base LFM has a strong prior
for producing *summaries of what a scene would contain* rather than the
scene itself. Everything in the architecture below is a reaction to that —
how do you compose a small writer model into a system that behaves like a
novelist?

## The problem, compressed

There are three failure modes that kill small-model long-form fiction, in
rough order of how much damage they do:

1. **Meta-commentary instead of prose.** The model produces summaries of
   the scene instead of the scene itself.
2. **Local incoherence.** POV flips mid-paragraph. A character dies then
   speaks. The weather is sunny in paragraph one and raining in paragraph
   two.
3. **Flat rhythm.** Every sentence is 10-12 words long. No dialogue. No
   variation. Prose reads as a single monotone.

Big closed models (GPT-5, Claude, Gemini) solve (1) and (2) by brute force
and reduce (3) with larger corpora of literary training data. A 1.2B
local model cannot brute force any of them.

Our bet: hierarchical planning handles (2). Retrieval-grounded context
handles the "doesn't know this world" part. A writer LoRA handles (1).
Generate-N + rerank chips away at (3). None of these individually are
novel; the engineering is in composing them and measuring what each
actually buys you.

---

## System architecture

The engine is a pipeline with seven stages: four planners, a writer, a
consistency checker, and an extractor. A retrieval layer sits underneath
and can be queried by any stage. Everything runs on one machine: vllm
serves the writer model, sentence-transformers handles embeddings, SQLite
holds quest state.

<div class="mermaid">
flowchart TB
    subgraph Planning
        ARC[ARC planner<br/>long-arc shape]
        DRA[DRAMATIC planner<br/>per-update scenes]
        EMO[EMOTIONAL planner<br/>per-scene trajectory]
        CRA[CRAFT planner<br/>prose blueprint]
        ARC --> DRA --> EMO --> CRA
    end

    CRA --> WRT[WRITE<br/>vllm + writer LoRA]
    WRT --> CHK[CHECK<br/>consistency pass]
    CHK --> EXT[EXTRACT<br/>world-state delta]

    subgraph Retrieval
        R1[Passage]
        R2[Quest memory]
        R3[Scene shape]
        R4[Motif]
        R5[Foreshadowing]
        R6[Voice]
        R7[Craft]
    end

    Retrieval -.-> Planning
    Retrieval -.-> WRT
</div>

### The planners

Each planner is an LLM call with a tight JSON schema enforced by xgrammar.
They run top-down, each producing structured output that constrains the
next:

- **ARC planner** picks the long-arc shape — three-act, hero's journey,
  tragic parabola, iceberg. Runs once per quest.
- **DRAMATIC planner** breaks each update (player turn) into scenes and
  assigns each scene a dramatic function: escalation, reversal, aftermath,
  quiet. Also picks POV character per scene.
- **EMOTIONAL planner** puts an emotional trajectory over the scene —
  surface/depth emotional pairs, shifts, beats.
- **CRAFT planner** produces the prose blueprint: which craft tools to use
  (e.g. free indirect style at permeability 0.4, chekhov plant, negative
  space opener), sensory bias per scene, indirection instruction, metaphor
  domain, motif tags.

The planners run on the same local LFM2.5 the writer uses. They're
cheap (small JSON outputs under xgrammar) and their output is the entire
context the writer needs — the writer doesn't see the whole quest state,
it sees the CraftPlan for one scene plus a few retrieval hits.

### The writer and its role

The writer is an instance of LFM2.5-1.2B-Instruct with a LoRA adapter
applied, served by vllm. It receives a rendered prompt containing:

- narrator config (voice samples, register, sensory bias)
- retrieved context (voice anchors, quest callbacks, craft exemplars)
- the CraftPlan for one scene
- an explicit write directive ("open in the middle of action; no plan
  summaries; no reader-addressed questions")

The writer emits 4 candidates concurrently (asyncio.gather); the Scorer
picks the best of those 4 via a weighted heuristic sum. That's
**generate-N rerank**, with N=4. We've run experiments at N=10 — the
diminishing returns above N=4 aren't worth the ~2.5x latency.

The CHECK stage is a separate LLM call with an anti-hallucination-aware
prompt (more on that below). It flags critical continuity breaks and
world-rule violations. If CHECK flags the chapter as critical, the commit
is rejected; the pipeline can retry or surface the failure to the
player.

EXTRACT is the last stage. It reads the committed prose and emits
structured world-state deltas: new entities, updated relationships, new
plot threads, new foreshadowing hooks. Those deltas feed back into
subsequent planners.

---

## The retrieval layer

Retrieval is seven retrievers sharing one interface (`Query`,
`Result`, `async retrieve(query, k)`):

| Retriever | Source | What it's for |
|---|---|---|
| Passage | 260 literary corpus passages (Sanderson/Abercrombie/Ishiguro/Woolf/...) | Style anchors — "write like this" |
| Quest | Past chapters of *this* quest, embedded post-commit | In-quest callbacks and entity re-mentions |
| Scene Shape | Labeled scenes from the literary corpus | "Here's an escalation scene that worked" |
| Motif | `motif_occurrences` SQLite table | Recurrence scheduling — planted motifs due for payoff |
| Foreshadowing | `foreshadowing` table | Hooks eligible for payoff this chapter |
| Voice | Past utterances of each POV character | Keeps the character's voice consistent across chapters |
| Craft | CraftLibrary exemplars per tool | "Free indirect style at permeability 0.4 looks like..." |

We use [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
for embeddings — 22MB, 384-dim, CPU-fast. It is not a literary-aware model.
For our scale (< 10k passages per quest, ~260 reference passages) it's
sufficient. Swapping to BGE-small or similar is Phase-2 scope if the
signal warrants.

Storage is deliberately dumb. The calibration corpus lives in
`data/calibration/embeddings.npy` as a 384-dim numpy array with a
parallel `embedding_ids.json`. Per-quest embeddings live in a
`narrative_embeddings` SQLite table with a 384-dim BLOB per scene.
No vector database. No Chroma, no Qdrant. Cosine similarity over
small-enough numpy arrays is fast, debuggable, and zero-ops.

The crucial insight: **not all retrievers need semantic embeddings**.
Motif and Foreshadowing are pure metadata queries — "which planted hooks
are ripe this update?" doesn't benefit from cosine similarity. Craft is
already categorized by tool-id. Voice works on POV-character keys. Only
Passage and Quest use MiniLM. This saves complexity and embedder compute
in the common case.

---

## Training and scoring

### The Scorer

Every committed chapter is scored on a 12-dimension scorecard. Scores are
continuous in [0, 1] and persist to SQLite. The dimensions split into
two families:

- **Heuristic** (4 dims): `sentence_variance`, `dialogue_ratio`,
  `pacing`, `sensory_density`. These are pure text-statistics — std-dev
  of sentence lengths, fraction of quoted spans, sensory-keyword
  density. They're cheap, deterministic, and discriminative.
- **Critic-derived** (8 dims): `free_indirect_quality`,
  `detail_characterization`, `metaphor_domains_score`,
  `indirection_score`, `pov_adherence`, `named_entity_presence`,
  `narrator_sensory_match`, `action_fidelity`. These wrap a validator
  from `app.planning.critics` and turn its `ValidationIssue` list into
  a scalar (errors weigh 0.25, warnings 0.10).

A later extension adds three **LLM-judged** dims on an async post-commit
task: `tension_execution`, `emotional_trajectory`,
`choice_hook_quality`. Single batched structured call. Judge model is
LFM2.5 for now; planned migration to a 9B judge if calibration
r > 0.7 doesn't hold.

A known ceiling: six of the eight critic dims saturate at 1.0 on every
committed chapter. That's a critic-too-lenient problem, not a
prose-too-good problem. It's listed under "what's next" below.

### Generate-N + rerank

Per scene, the writer fans out N=4 candidates. All four get scored.
Rerank is a weighted sum:

```
overall_score = Σ weight_d * score_d  for d in 12 dims
```

Default weights emphasize the heuristic dims (pacing, sentence variance,
dialogue) since they have actual dynamic range. The highest-scoring
candidate becomes the committed chapter; the other three get saved to
the SFT capture directory for later human review.

### The SFT corpus and LoRA

Two LoRAs shipped so far:

- **Writer LoRA v1**: rank 32, 3 epochs, 11 hand-picked SFT records.
  Trained on Claude-picked winners of the first generate-N runs.
  Train time: 3.1 seconds on a 4090.
- **Writer LoRA v2**: rank 64, 5 epochs, 64 SFT records across three
  seeds (noir / political intrigue / SFF heist). Trained on
  heuristic-rater picks (cliché penalty, POV-drift penalty,
  foreign-token leak penalty, dialogue bonus). Train time: 22 seconds.

The v1 corpus was too small to do anything substantive, but it did one
important thing: **it killed base LFM's meta-commentary failure mode.**
Pre-v1, the base model would produce scene-summaries masquerading as
prose ("The protagonist realizes the weight of the moment..."). v1 with
11 records was enough to flip that to actual scene prose with concrete
objects and images. Everything after has been about lifting the
structural dims (sentence variance, dialogue ratio, pacing) which v1
barely moved.

v2 moved pacing (+0.06) and sensory density (+0.10). It didn't move
sentence variance or dialogue. That's the current ceiling, and we think
it needs a v3 on 200-300 records to break it.

---

## Findings from iteration

Six stories from the build-out so far. These are the
interesting ones — not just the numbers that moved, but why they moved.

### 1. LLMs hallucinate world rules under the "check" prompt

The CHECK stage was flagging ~45% of chapters as critical at 20
chapters. Looking at the flags, many said things like:

> *"Violates established pattern of guarded behavior"*
>
> *"Breaks the tonal contract of the narrator's reticence"*

Neither of those was an actual world rule. The quest's Active World
Rules list contained things like "Kaela cannot speak Old Kessian" and
"the lighthouse is unlit after curfew". The CHECK LLM, given a
"consistency checker" role, *invented* plausible-sounding rules and
then dinged the prose for violating them. It was hallucinating the
evidence it needed to justify its role.

Fix: require critical-severity issues to **quote an exact line from
the Active World Rules list**. If the quote isn't in the list, the
issue auto-demotes to warning. Commit rate on a 5-action run went
from 60% to 100%.

The general lesson: when you give an LLM a gatekeeper role, it will
invent gatekeeping-evidence to fill the role. Tie the role's output
to concrete evidence extractable from the input.

### 2. Silent CUDA-OOM disabling quest memory

Symptom: the QuestRetriever (past narrative embeddings) was returning
zero hits for every live run. Same for VoiceRetriever. The
`narrative_embeddings` table was empty despite the pipeline supposedly
writing to it post-commit.

Root cause: the embedder (`sentence-transformers/all-MiniLM-L6-v2`)
auto-loaded on CUDA. But CUDA was fully booked by vllm (22GB on a
24GB card). The embedder crashed with CUDA-OOM. The pipeline wrapped
the embedding write in a `try/except StageError` — the commit outcome
was still "committed", so from every dashboard it looked fine.

Meanwhile `quest_callbacks` and `voice_continuity` retrievers
silently returned zero hits for every live run of the pipeline.
Classic silent failure.

Fix: default the embedder to CPU. MiniLM is 22MB; CPU encode on
short prose is ~10ms. `QUEST_EMBEDDER_DEVICE=cuda` to opt back in
when the GPU is free.

The lesson: wrap-and-log isn't the same as handle-the-failure.
Every silent-except-in-logs catch is a latent bug waiting to
matter. (Our stress-test harness was passing
`CUDA_VISIBLE_DEVICES=""` to the embedder subprocess, so none of our
measurements surfaced it — only live runs did.)

### 3. POV attribution bug — "innkeeper" everywhere instead of "player"

Every row in the `narrative` table had `pov_character_id='innkeeper'`
(Merrin, an NPC) instead of `'player'` (Kaela, the protagonist
character). This mattered because VoiceRetriever is keyed on
POV character — it pulls past utterances for that character. With the
wrong POV, the writer was being anchored to Merrin's past dialogue
while writing Kaela's scenes.

Root cause: the DRAMATIC planner picks a POV character per scene. For
some scene types (conversations, observations of NPC activity) it
picks the NPC as POV. The pipeline then stored whatever the planner
emitted, without a protagonist override.

Fix: narrator config gets an explicit `pov_character_id` override. If
unset, fall back to `id='player'` by convention. Now VoiceRetriever
retrieves Kaela's past dialogue for Kaela's scenes.

### 4. Self-reinforcement trap — voice memory dominated voice seed

With POV fixed, a new problem appeared. `voice_continuity` (dialogue
retrieved from the writer's *own* past output) started dominating
`voice_samples` (the narrator-config rhythm anchors we wrote by hand).
Both sit in the writer's prompt. `voice_continuity` came later,
so LLM recency bias made it the dominant anchor. Result: the writer
began matching its own earlier output — which was already flat-rhythm
and dialogue-light — instead of the varied rhythm anchors we'd
written.

It was a feedback loop. Flat output becomes retrieval context.
Retrieval context reinforces flat output. Dialogue ratio regressed
from 0.07 → 0.03 over 10 chapters.

Fix: flip prompt order. `voice_samples` now appear **after**
`voice_continuity`, so the writer's last impression before the write
directive is the intentional rhythm anchor, not its own prior flat
output.

This is one of those bugs you don't see coming until you have *both*
features working — retrieval-grounded voice *and* bootstrap voice
seeds. Either alone is fine. The composition has a subtle priority
ordering the LLM cares about intensely.

### 5. Seed voice samples matter a lot

Going from 1 rhythmically-flat voice sample to 4 varied ones (short
fragments + a 50-word cascading sentence + some 1-word openers)
**nearly 5x'd dialogue_ratio at 10 chapters** and lifted
sentence_variance meaningfully.

LFM2.5-Instruct has a default cadence of ~10 words per sentence. One
varied example does not break that prior. Several do. The effect is
non-linear: with 4 anchors, the model's variance is ≥ max(variance of
anchors). With 1 anchor, the model averages toward its prior.

This is cheap — voice samples are 200 words total. It beats
anything else we've done at its cost. The question of *why* 4 works
but 1 doesn't is interesting — our working theory is that with 1
sample, the LLM treats it as "one particular instance, draw back
toward the mean"; with 4, it treats them as "here is the *distribution*
you're writing in".

### 6. Prompt engineering has a plateau

We've now extracted most of what the current writer can do via prompt
alone. Iterating on structure, retrieval, and CHECK rigor took commit
rate from 32% → 100%. Per-dim prose scores went from overall
0.72 → 0.85-ish on good runs.

But the *structural* dims — sentence variance at 0.15, dialogue ratio
at 0.04 — have not moved meaningfully. Our writer LoRA has been
trained on 11 records then 64 records. Neither gave us the rhythm
variation that actually distinguishes good prose from uniform prose.

Next lift needs better SFT data. 200-300 records across varied seeds.
Dialogue-heavy scenes deliberately triggered. Claude-picked winners,
not heuristic-picked winners. That's the single highest-leverage move
on the horizon.

---

## Current results

Measured on a 20-chapter noir stress test
(`/tmp/storygen_20ch_final/quest.db`, writer LoRA v2, retrieval on):

| metric | baseline | after these fixes | What changed |
|---|---|---|---|
| commit rate | 11/20 (55%) | **20/20 (100%)** | CHECK prompt tightening |
| dialogue_ratio | 0.000 | 0.038 | `player_action` plumbed, dialogue directive |
| sentence_variance | 0.16 | 0.15 | voice_samples enriched, reorder |
| pacing | 0.72 | 0.84 | compound effect |

At 10 chapters (where context is less diluted): `dialogue_ratio=0.07`,
`sentence_variance=0.18`, `pacing=0.79`. The 10-chapter-vs-20-chapter
gap is real; we don't yet know why the engine degrades with more
committed state, even though commit rate holds at 100%.

For comparison: Sanderson's calibration corpus sits at
`sentence_variance ≈ 0.5`, Abercrombie at `dialogue_ratio ≈ 0.3`.
We are not near those yet.

### What the prose actually looks like

Two scenes from the 20-chapter run. Neither is "perfect" — part of the
point of this report is honest reporting on a 1.2B model.

**Update 8**, player action: *"I find an alley with one entrance and
wait there."*

> You move through the narrow corridor with a tension that hums in your
> bones. The corridor narrows, and you feel the weight of what lies
> ahead pressing against your thoughts. The air is heavy with unspoken
> words, and every step echoes like a promise. You glance at the door,
> your breath catching slightly. The innkeeper's voice comes next,
> calm but carrying an edge you hadn't noticed before.
>
> "You're sure about this?" he asks, his words steady, yet there's a
> flicker in his eyes—uncertain, perhaps even fear. You don't answer
> immediately. Instead, you watch his hands, the way they move, the
> slight tension in his shoulders. It's not anger, but something
> deeper. A quiet resolve.
>
> You keep your distance, letting the silence stretch between you. The
> rain taps against the window, a steady rhythm that matches your
> heartbeat. You listen, but nothing comes. It's not a reaction you
> recognize. Instead, you feel the pull, the unspoken truth settling
> in the hollow of your chest. You don't push him. You wait.

This is decent-not-great. The dialogue work is visible — Merrin's
line landed in-scene, not as meta-commentary. Pacing is uneven (the
first paragraph is thick). A few clichés ("weight of what lies ahead",
"steady rhythm", "hollow of your chest") are unwelcome but
not deal-breaking. Crucially: it's a scene. There's a voiced NPC, a
continued setting (rain on the window), and a POV that stays
second-person throughout.

**Update 17**, a three-sentence exchange:

> You hold the coin like it expects a reply. The tension builds, sharp
> and unspoken. The innkeeper's gaze lingers, but you push the silence
> forward. You say it plain: "Give them what they want." The words
> hang, soft but certain, and the room shifts, still.

Shorter, tighter, and with the player's voice in the line. This is
closer to what the target looks like.

---

## Limitations and what's next

### Context-degradation mystery

At 10 chapters: `dialogue_ratio=0.07`, `sentence_variance=0.18`.
At 20 chapters: `dialogue_ratio=0.04`, `sentence_variance=0.15`.

Commit rate holds at 100% across the range, which means CHECK is
passing both, but the prose is flatter on the back half. We have
three hypotheses:

1. Prompt bloat — retrieval hits and world state grow, the writer's
   effective attention to voice anchors dilutes.
2. Self-reinforcement residue — even after fixing voice_samples
   ordering, the model has more of its own past output to anchor to
   and that pushes back toward flat.
3. Plan dilution — planners get more state to juggle and their
   outputs get looser, leaving the writer with weaker scaffolding.

Next work: ablate each. Serve writes at 10ch-equivalent state and see
which piece moves the numbers. This is our immediate next stress
measurement.

### LoRA v3

Later scope from the phased plan, but we're pulling it earlier.
A 200-300 record corpus with explicit dialogue-heavy scenes, Claude
(as in: the model) picking winners instead of heuristic picks. That's
the lever most likely to move the structural dims.

### Judge-dim calibration

The three LLM-judged dims (`tension_execution`,
`emotional_trajectory`, `choice_hook_quality`) currently use LFM2.5
itself as judge, which is not credible for literary-adjacent
judgments. Planned migration to a 9B+ judge (candidates: Qwen 2.5 9B,
Gemma 4 9B). Calibration target: r > 0.7 against Claude-labeled
ground truth on a 50-passage sample.

### Critic saturation

Six of the eight critic-derived dims saturate at 1.0 on every
committed chapter. That's a too-lenient-critic problem — the critics
are pass/fail gates, not continuous quality signals. An early Phase 2
pass tightens them. Until then, `overall_score` has very little dynamic
range above ~0.80.

### "Prose meaningfully better than Royal Road"

The honest read: we are not there. We are at "reads like a small
model wrote it, but the pipeline holds together and the scenes
coherently advance the quest". The structural heuristics confirm this
— `dialogue_ratio=0.04` is well below any human writer of prose
fiction. Whether a 1.2B model can get there at all, or whether we
need a bigger base, is a question we're planning to answer by
exhausting the SFT and retrieval dimensions on LFM first before
considering an upgrade.

---

## Meta-thesis — why we think this works at all

Three claims worth saying out loud:

**Hierarchical planning decouples long-range coherence from local prose.**
A 1.2B writer cannot keep a 50-chapter arc in its context. But it *can*
execute a one-scene craft brief well. Pushing structure UP (to cheap,
narrow planners) and prose DOWN (to a writer with anchored local
context) lets a small model do work it otherwise cannot. The ARC →
DRAMATIC → EMOTIONAL → CRAFT chain is essentially a progressively-
constrained-context strategy.

**Retrieval, not context stuffing.** We could try to stuff past
chapters and style anchors into every writer call. We'd run out of
context at chapter 8 and the writer would drown in irrelevant detail.
Instead, MiniLM over thousands of passages costs 10ms and gives us
dense lookups. We anchor style and continuity with retrieved
exemplars instead of hoping they fit in the window.

**Stress-test-driven iteration.** Every commit gets measured against a
5/10/20-chapter stress run. Instrumenting both commit rate (structural
health) and per-dim heuristics (prose quality) catches regressions in
two different registers. A change that lifts commit rate but flattens
prose shows up immediately; we almost shipped one (the "no rhetorical
questions" ban) that did exactly that.

We don't know yet whether this approach gets to the quality bar we
want. We *do* know it produces coherent prose end-to-end on a stress
test, from a model that at the start of the project was emitting
scene-summaries instead of scenes. That's the first mile. The rest of
the roadmap is the other nine.

---

## Pointers

Source: [github.com/RPP1011/quest_game](https://github.com/RPP1011/quest_game).

Iteration notes (the journal to this report's synthesis):

- [Phase 1 summary]({{ "/phase1-summary.html" | relative_url }}) — consolidation
- [50-chapter stress test]({{ "/day10-stress-test.html" | relative_url }}) — baseline
- [Bottleneck fix]({{ "/day11-bottleneck-fix.html" | relative_url }})
- [20-chapter verification]({{ "/day12-verification.html" | relative_url }})
- [Targeted fixes]({{ "/day13-fixes.html" | relative_url }})
- [Writer LoRA v1]({{ "/day5-writer-lora-v1.html" | relative_url }})
- [Phase 2 kickoff — writer LoRA v2]({{ "/phase2-kickoff-lora-v2.html" | relative_url }})

Planning:

- [Roadmap]({{ "/roadmap-3mo.html" | relative_url }})
- [Retrieval layer design spec]({{ "/superpowers/specs/2026-04-14-retrieval-layer-design.html" | relative_url }})

<script>
window.addEventListener('load', function () {
  if (!document.querySelector('.mermaid')) return;
  var s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
  s.onload = function () { window.mermaid.initialize({ startOnLoad: true, theme: 'neutral' }); };
  document.head.appendChild(s);
});
</script>
