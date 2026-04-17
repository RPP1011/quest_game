---
title: "Pale Lights — Full Corpus Craft Analysis"
---

# Pale Lights — Full Corpus Craft Analysis

**Date:** 2026-04-15
**Source:** palelights.com full public archive (Book 1 "Lost Things" 45 ch + epilogue; Book 2 "Good Treasons" 78 ch + epilogue through 2025-03), **125 chapters, 1,024,109 words** fetched via `tools/corpus_fetchers/wordpress.py`.
**Method:** Every chapter annotated and judged by Gemma 4 26B A4B (PRISM-PRO-DQ quant, ~5.73 bpw, served via `llama-server`). Annotations via `tools/annotate_chapter.py` (15-field structured JSON schema). Chapter-scale dimension scoring via `tools/judge_chapters.py` with 8 rubric-anchored dims under `prompts/scoring/chapter_dims/`. Full chapters throughout — length is a signal we preserve, not noise we sample away.

## Why this exists

The quest-game pipeline currently produces ~2,000-word updates via a dramatic planner → craft planner → per-beat writer loop. The beat loop works, but the planners emit thin scene sheets and the writer has no pacing comparables. We needed an outside reference — an actual published web serial with strong narrative craft — to see how far our output was from the form, and where in the pipeline the gap actually lives.

Pale Lights was chosen because it's (a) a well-regarded ongoing quest-adjacent serial by a competent author, (b) already listed as a corpus source in `data/calibration/manifest.yaml`, and (c) available under the author's public-web license.

## What was done

1. **Fetch.** `python -m tools.corpus_fetchers.wordpress pale_lights` pulled 125 chapters (Book 1 + current Book 2) to `data/calibration/raw/pale_lights/chap_NNNN.txt`. Two non-chapter TOC anchors (`/summary/`, `/art-maps/`) stripped; file numbering aligned to true chapter index.
2. **Annotation.** `tools/annotate_chapter.py` sends each chapter to Gemma 4 with a 15-field structured JSON schema (summary, POV, scenes[{dramatic_question, outcome, beats[{n, description, beat_type, excerpt}]}], characters_present, relationships_shown, world_facts_established, hooks_planted, hooks_paid_off, arc_movement). A first pass (`gemma4` label, 44 chapters of Book 1) revealed the model was skipping `relationships_shown`; a v2 pass (`gemma4_v2` label, all 125 chapters) with stronger prompt language around that field landed 331 relationships across the corpus (vs. 0 in v1). Both label sets retained for provenance.
3. **Model comparison.** Annotated Ch 1 with both Gemma 4 and LFM 2.5 1.2B. LFM failed on accuracy (hallucinated that Fortuna kills Tristan; rendered POV as "Chapter 3") and JSON validity (malformed output at line 7,592). Gemma 4 produced valid JSON with accurate summary, scenes, and characters in ~17 s. **LFM 1.2B is not viable as annotator** — too small to both recall 5k+ words of plot and render the schema.
4. **Heuristic scoring.** `tools/score_chapters.py` ran `app.calibration.heuristics.run_heuristics` on every full chapter (125 chapters) for the 4 CPU-only dims (sentence_variance, dialogue_ratio, pacing, sensory_density). Output: `chapter_scores.json`.
5. **Chapter-scale judge rubrics.** New rubrics under `prompts/scoring/chapter_dims/` covering 8 dims: `tension_execution`, `emotional_trajectory`, `choice_hook_quality`, `update_self_containment`, `voice_distinctiveness`, `thematic_presence`, `subtext_presence`, `interiority_depth`. These adapt the passage-scale (500–1000w) and scene-scale (2000–4000w) rubrics already in the repo to a chapter-scale (4000–13000w) frame with new anchors.
6. **LLM-judge pass.** `tools/judge_chapters.py` sent each of the 125 chapters through Gemma 4 with a single batched structured-output call returning all 8 dim scores + rationales per chapter. ~20 min wall time at ~10 s/chapter. Three chapters failed on first pass due to a Gemma 4 thinking-mode quirk where the model spent all tokens in reasoning and emitted no content; retried successfully with `max_tokens=8000`.
7. **Rollups.**
   - `tools/rollup_annotations.py` merges per-chapter annotations into a work-level rollup (characters across chapters, relationships with evidence, accreting world facts, planted hooks, arc timeline).
   - `tools/rollup_judge_scores.py` aggregates per-chapter dim scores and diffs chapter means against the manifest's work-level `expected` block (Claude-labeled reference values).

**Gemma-4 bias caveat.** Scoring uses Gemma 4 as both generator and judge. Absolute values are ordinal — trust deltas between variants, not raw numbers. If we ever move to a reward-signal setup, the judge needs ground-truth label validation.

## Corpus stats (full corpus, v2 annotations)

| Metric | Value |
|---|---|
| Chapters | 125 |
| Total words | 1,024,109 |
| Words / chapter | mean 8,193, range 2,861–16,352 |
| Scenes / chapter | mean 4.9 |
| Beats / chapter | mean 24.8 |
| Named characters tracked | 293 |
| Relationships shown | 331 |
| New world facts established | 1,326 |
| Planted hooks | 561 |

## Beat-type distribution

Across 44 × 4.6 scenes × 23.3 beats ≈ 1,027 annotated beats:

| Beat type | Share | Notes |
|---|---|---|
| action | 43 % | The dominant mode. |
| reveal | 34 % | One third of all beats are information exchange — a letter read, a clue found, a secret told, a name learned. |
| decision | 12 % | Clear choice points where the protagonist commits to an action with consequences. |
| dialogue | 7 % | Lower than expected — conversation beats are most often wrapped into action or reveal. |
| interiority | ~0 % | Rendered through action and dialogue with a pact god, not as standalone internal monologue. |
| transition | ~0 % | Rare; scene changes cut hard. |

**Scene openings: 74 % action, 16 % reveal, 4 % dialogue, 2 % decision.** Scenes almost never open on interior thought or exposition.

## Chapter-scale dimension scoring

### Heuristic dims (CPU, `app.calibration.heuristics`) across all 125 chapters

| Dim | PL mean | sd | range | Our Gemma-4 output (2,120w) |
|---|---|---|---|---|
| sentence_variance | 0.747 | 0.140 | 0.352–1.000 | **0.331** |
| dialogue_ratio | 0.264 | 0.081 | 0.028–0.492 | **0.083** |
| pacing | 0.365 | 0.046 | 0.282–0.552 | **0.595** |
| sensory_density | 0.444 | 0.067 | 0.296–0.620 | **0.950** |

Our output reads as: short, uniform-length sentences, light on dialogue, sensory-maximalist. Pale Lights reads as: long + varied sentences, quarter dialogue, specific-but-not-overloaded sensory detail.

### LLM-judge dims (Gemma 4 as judge) across all 125 chapters

| Dim | PL mean | sd | median | min | max | Claude-labeled expected | Δ |
|---|---|---|---|---|---|---|---|
| tension_execution | 0.80 | 0.08 | 0.80 | 0.70 | 0.90 | 0.75 | +0.05 |
| emotional_trajectory | 0.83 | 0.11 | 0.90 | 0.60 | 0.90 | — | — |
| choice_hook_quality | 0.71 | 0.05 | 0.70 | 0.50 | 1.00 | 0.60 | +0.11 |
| update_self_containment | 0.72 | 0.04 | 0.70 | 0.70 | 0.80 | 0.70 | +0.02 |
| voice_distinctiveness | 0.62 | 0.05 | 0.60 | 0.60 | 0.80 | 0.85 | **−0.23** |
| thematic_presence | 0.73 | 0.13 | 0.70 | 0.60 | 0.90 | 0.70 | +0.03 |
| subtext_presence | 0.70 | 0.10 | 0.70 | 0.60 | 0.90 | 0.60 | +0.10 |
| interiority_depth | 0.71 | 0.09 | 0.70 | 0.60 | 0.90 | 0.65 | +0.06 |

**Judge calibration.** 7 of 8 dims land within ±0.11 of Claude-labeled expected — first-pass agreement is stronger than expected for a completely new judge/rubric pipeline. `voice_distinctiveness` is the one real outlier at −0.23; Gemma 4 reads voice more conservatively than Claude did, clustering PL around 0.60 where Claude put it at 0.85. Likely a rubric-anchor mismatch (our rubric reserves 0.9 for "unmistakable"; Claude's implicit rubric was less strict). Worth a second-pass rubric tune later; not a blocker for ordinal comparisons.

**Score compression.** Standard deviations are tight (0.04–0.13). Reflects two things: PL is genuinely consistent at its craft level, and the judge pulls extremes toward center. The real distinguishing signal lives in deltas between generator variants, not in absolute values.

### Our Gemma-4 output vs PL mean (single-chapter direct comparison)

The Mara/Hale tavern chapter (2,120 words, full hierarchical pipeline) judged against the same rubric:

| Dim | Our output | PL mean | Δ |
|---|---|---|---|
| tension_execution | 0.70 | 0.80 | **−0.10** |
| emotional_trajectory | 0.90 | 0.83 | +0.07 |
| choice_hook_quality | 0.70 | 0.71 | ~0 |
| update_self_containment | 0.70 | 0.72 | ~0 |
| voice_distinctiveness | 0.60 | 0.62 | ~0 |
| thematic_presence | 0.60 | 0.73 | **−0.13** |
| subtext_presence | 0.90 | 0.70 | **+0.20** |
| interiority_depth | 0.60 | 0.71 | −0.11 |

n=1 for our output; caveat against reading too much into any single number. But the shape is informative:

- **Where we already match PL:** emotional_trajectory, subtext_presence, update_self_containment, voice_distinctiveness, choice_hook_quality. These are dims where our pipeline's existing machinery (dramatic planner, emotional planner, craft planner, beat-loop writer) is doing its job at chapter scale.
- **Where we trail (interiority_depth, tension_execution, thematic_presence):** also the three dims that need *plot* and *character knowledge* to be done well. Interiority requires a specific POV mind to simulate; tension requires multi-scene escalation (see §7 of Craft Findings); thematic presence requires a theme the planner is actively working across scenes. Our seed has 3 entities and no theme; of course we underperform.
- **Subtext +0.20 is surprising but probably real.** Our Mara/Hale scene is unusually subtext-heavy because Hale's deflection pattern is the scene's whole dynamic — every line means something other than what it says. PL chapters are on average denser in plot events, which lowers the per-word subtext ratio.

## Macro-structure of Book 1

Four identifiable acts, each a movement through a named trial:

| Act | Chapters | Title | Beat of the act |
|---|---|---|---|
| I | 1–5 | Sacromonte origins | Two protagonists introduced separately; each driven from the city by different pressures; both converge on the ship *Bluebell*. |
| II | 6–19 | Trial of Lines | March from the Dominion docks toward the Old Fort. Company politics, cult ambushes, the Fisher trial, Augusto's first betrayal. |
| III | 19–37 | Trial of Ruins | The shrine maze. Crew formation, the Isabel/Ferranda deceptions, aetheric pillar infiltration, the Red Maw revelation, mountain collapse. |
| IV | 38–44+ | Trial of Weeds | Cantica — a voting-to-execute trial that becomes a siege when cultists and devils attack. Protagonists settle personal vendettas. |

**POV discipline.** Each chapter is single-POV, strictly Tristan OR Angharad, alternating. No chapter is shared. Each arc movement is shown twice — once from each protagonist's close-POV — across two consecutive chapters.

**Protagonist chapter presence (top 10 by chapters appeared):**

Angharad Tredegar (20), Tristan Abrascal (19), Fortuna (18), Yong (18), Tupoc Xical (18), Lan (15), Song (14), Augusto Cerdan (14), Isabel Ruesta (13), Cozme Aflor (11).

## Craft findings — what to steal for the planner stack

Each finding pairs an observation from Pale Lights with a concrete intervention point in this codebase.

### 1. Alternating-POV discipline across updates

**Observation.** Pale Lights never blends POVs within a chapter and alternates strictly between two protagonists across chapters.

**Intervention.** Our `DramaticScene` has a `pov_character_id` field but no notion of POV across updates. For multi-protagonist quests, the dramatic planner should: (a) inherit the prior update's POV by default, (b) be able to switch only at update boundaries, never mid-scene. A `pov_inheritance: "continue" | "switch_to:<char>"` field on the update-level dramatic plan would formalize this.

### 2. Scene density is 2–3× ours

**Observation.** Pale Lights: ~4.6 scenes × ~23 beats per chapter, ~7,800 words total. Our current output: 3 scenes × 3 beats, ~2,100 words.

**Intervention.** Two pressures raise this:
- The dramatic-planner user prompt (`prompts/stages/dramatic/user.j2`) does not currently ask for a scene count. Nudging toward 3–5 scenes per update will add structural variety.
- Beat count within a scene should also rise. Current `DramaticScene.beats` tends to emit 2–3; comparables emit 4–7. Add a line to the dramatic system prompt: *"each scene carries 3–6 beats; short scenes that exist to deliver one reveal are permitted."*

### 3. "Reveal" as a first-class beat type

**Observation.** 34 % of Pale Lights beats are reveals — not dialogue, not action, but information flowing to the reader. This is the dominant density multiplier: every chapter the reader learns something concrete about world, plot, or character.

**Intervention.** Our `DramaticScene.beats` is `list[str]`. Introduce a per-beat `beat_type` enum (`action | reveal | decision | dialogue | interiority`) on the dramatic schema and prompt for a target mix (*"at least one reveal per scene; aim for ~30 % reveal beats across the update"*). The writer prompt can then differentiate how to render a reveal beat vs. an action beat.

### 4. Hooks-to-plant as a first-class planner output

**Observation.** Pale Lights plants 4–5 new hooks per chapter. An "hook" is an unresolved promise, threat, or question — named but not yet explored. These are what carry the reader forward.

**Intervention.** The world layer already tracks `foreshadowing` as a persistent table, but the dramatic planner does not actively propose new hooks — it only consumes existing `ripe_hooks` for payoff. Add `hooks_to_plant: list[str]` to the dramatic plan output. The write template then has a natural slot for "name these elements without explaining them yet." Post-commit, persisted as new foreshadowing rows with `planted_at_update = now`.

### 5. World-fact accretion rhythm

**Observation.** Pale Lights averages ~11 new named proper nouns per chapter — factions (Hoja Roja, Cordero Sonriente), spirits (the Fisher, Kshetra), idioms (Law of Rats), institutions (the Watch, the Krypteia), substances (black verity, Spinster's Milk), magic concepts (Gongmin lock, lodestone extract). Our smoke-test seed has 3 named entities total, and the writer introduces nothing new.

**Intervention.** Two paths:
- **Seed-side.** Quest seeds need to ship with a minimum density of named proper nouns — factions, local currencies, idioms, creatures — for the writer to reach for. Our seed JSON schema could enforce minimum counts or warn when too sparse.
- **Planner-side.** Add `new_world_elements: list[str]` to the dramatic plan each update — 1–3 new named things the writer is required to introduce with texture. Post-commit, these enter the world state as entities or rules.

### 6. Pact-god as externalized interiority

**Observation.** Interiority is ~0 % as a standalone beat because Erratic renders it as dialogue with the protagonist's pact god (Fortuna for Tristan, the Fisher for Angharad). This sidesteps the biggest weakness of close-POV quest prose: stretches of unrelieved internal monologue.

**Intervention.** For 3rd-POV quests this is a structural device worth proposing explicitly in the seed — every POV character gets a companion-voice (god, spirit, familiar, ghost). For our default 2nd-person CYOA narrator this is trickier but not impossible (*"you remember what Abuela used to say"* as an externalized voice is a weaker form). Worth prototyping as an optional narrator config flag in v1.

### 7. Scene openings are concrete

**Observation.** 74 % of scene openings are action beats; 16 % reveals. Almost no scene opens on interior thought, reflection, or abstract exposition.

**Intervention.** One line in `prompts/stages/write/system.j2`: *"open each scene on a physical action in space or on a concrete reveal; never on a character thinking about what just happened or on abstract exposition."* This is a trivial change with outsized effect.

## What LFM cannot do (and the implication)

The head-to-head annotation on Ch 1 was decisive: LFM 1.2B cannot hold 5 k words of plot in context *and* emit valid structured JSON *and* be accurate. It hallucinates plot-breaking details and produces malformed JSON even with `strict: true` schema enforcement.

Implication for the training-LoRA plan that motivated this project: the original idea of "LFM annotates cheaply at scale" is dead. The roles must flip. LFM's job in this system is *generation*, not annotation. The Gemma 4 annotations produced here become the training data for an LFM LoRA where:

- **Input:** `(beats + scene context + character state + hooks_to_plant + new_world_elements)`
- **Output:** chapter prose

That is: use Gemma 4 (or a larger annotator) to extract `(plan → prose)` pairs from a reference corpus; train LFM to do the reverse mapping. The bet is that LFM, freed from having to plan, can render a chapter's worth of prose from a high-quality beat sheet — which is exactly the role our current pipeline assigns to it.

## Artifacts produced

| Path | Content |
|---|---|
| `data/calibration/raw/pale_lights/chap_0001.txt` … `chap_0125.txt` | Verbatim prose, 1,024,109 words (Book 1 + current Book 2). |
| `data/calibration/raw/pale_lights/meta.json` | Fetch metadata (source URLs, timestamps, license). |
| `data/calibration/annotations/pale_lights/chap_NNNN.gemma4.json` | Book 1 only, v1 annotation pass (relationships gap). |
| `data/calibration/annotations/pale_lights/chap_NNNN.gemma4_v2.json` | Full corpus, v2 annotation pass with populated relationships. |
| `data/calibration/annotations/pale_lights/rollup.gemma4_v2.json` / `.md` | 293 characters, 331 relationships, 1,326 world facts, 561 hooks, arc timeline. |
| `data/calibration/annotations/pale_lights/chap_NNNN.judge.gemma4.json` | Chapter-scale 8-dim LLM-judge scores + rationales. |
| `data/calibration/annotations/pale_lights/judge_rollup.gemma4.json` / `.md` | Per-dim aggregate (mean/sd/median/min/max) + diff vs manifest expected. |
| `data/calibration/annotations/pale_lights/chapter_scores.json` | Per-chapter heuristic dim scores (4 dims) across all 125 chapters. |
| `prompts/scoring/chapter_dims/*.j2` | 8 new chapter-scale rubrics. |
| `tools/annotate_chapter.py` | Annotation tool. |
| `tools/judge_chapters.py` | Chapter-scale LLM-judge tool (batched structured output). |
| `tools/score_chapters.py` | Per-chapter heuristic scorer. |
| `tools/rollup_annotations.py` | Annotation → rollup. |
| `tools/rollup_judge_scores.py` | Judge scores → rollup + expected diff. |

## Immediate next steps

1. **Planner-stack interventions** (rough leverage × cost order):
   - Scene-opening rule in `write/system.j2` ("open on action or reveal; never on exposition or thought"). Trivial; addresses the 74%-action-opening pattern.
   - Nudge dramatic planner toward 3–5 scenes per update. Trivial.
   - Add per-beat `beat_type` enum + reveal-quota to dramatic schema. Low cost; captures the 34% reveal share.
   - Add `hooks_to_plant` + `new_world_elements` to dramatic plan output; post-commit persistence into world state. Medium cost; highest single-change effect on SV-quest feel.
2. **Voice rubric re-anchor.** Run 5 PL chapters with a looser `voice_distinctiveness` rubric (move 0.85+ threshold from "unmistakable" to "clear fingerprint in 3+ of {diction, rhythm, image family, dialogue tag style}"); check whether the corpus mean rises to ~0.80, matching Claude. Lock if so.
3. **LoRA training-pair format.** Convert the 125 v2 annotations into `(plan_state → chapter_prose)` training pairs. Check plan-side density; if the 24.8 beats/chapter signal is too coarse, add a beat-expansion pass to upsample toward 40–60 micro-beats before training.
4. **Generator variant comparison using the judge as eval.** Once a planner-stack intervention lands, regenerate a sample of smoke-test chapters and score them via `tools/judge_chapters.py` for a delta-to-baseline on all 8 dims. This is the loop we built the judge for.
