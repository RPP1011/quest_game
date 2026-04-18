---
title: "Quest Engine: Full Pipeline Walkthrough"
date: 2026-04-18
layout: default
---

# Quest Engine: Full Pipeline Walkthrough

A labeled example showing how a seed, story history, and player action produce a chapter of prose. All data comes from a real rollout (`ro_529f9e71`, cautious profile, chapter 1) against the Pale Lights seed.

## Glossary

| Term | Definition |
|------|-----------|
| **Seed** | JSON file defining the world: entities, rules, plot threads, foreshadowing hooks, themes, motifs, narrator config |
| **Story candidate** | One of N possible story arcs generated from the seed. The player picks one. |
| **Arc skeleton** | Per-chapter outline generated from the picked candidate: dramatic questions, plot beats, tension targets, entity surfacing schedule |
| **Virtual-player profile** | Personality model (impulsive, cautious, honor_bound) that drives action selection during rollouts |
| **Rollout** | Automated playthrough: profile + candidate + skeleton → 10 chapters of prose |
| **Beat** | One LLM call within the writer stage. A chapter is ~12-16 beats of continuous prose. |
| **Check→revise loop** | Post-write quality gate: LLM checker flags issues, LLM reviser fixes them, up to 4 passes |
| **LLM metaphor critic** | Classifies every figurative use in the chapter by imagery family. Flags when any family exceeds budget. |

## System Overview

```
┌─────────────┐
│    Seed      │ 49 entities, 7 rules, 10 hooks, 6 threads
│ (pale_lights)│ narrator config, themes, motifs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Candidate  │ "The Thief and the Goddess"
│   Picker     │ 3 arcs generated → player picks one
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     Arc      │ 10-chapter skeleton with per-chapter:
│   Skeleton   │ dramatic question, plot beats, tension target,
│              │ POV character, entities to surface
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              Per-Chapter Pipeline                  │
│                                                    │
│  ┌───────┐ ┌──────────┐ ┌───────────┐ ┌───────┐  │
│  │  Arc  │→│ Dramatic  │→│ Emotional │→│ Craft │  │
│  │Planner│ │ Planner   │ │  Planner  │ │Planner│  │
│  └───────┘ └──────────┘ └───────────┘ └───────┘  │
│       │          │             │            │      │
│       └──────────┴─────────────┴────────────┘      │
│                        │                           │
│                        ▼                           │
│              ┌──────────────────┐                  │
│              │  Writer (12-16   │                  │
│              │  beats per scene)│                  │
│              └────────┬─────────┘                  │
│                       │                            │
│                       ▼                            │
│              ┌──────────────────┐                  │
│              │  Check → Revise  │ ← LLM metaphor  │
│              │  (up to 4 loops) │   critic         │
│              └────────┬─────────┘                  │
│                       │                            │
│                       ▼                            │
│              ┌──────────────────┐                  │
│              │    KB Extract    │                  │
│              │  + Score (4 dim) │                  │
│              └──────────────────┘                  │
└──────────────────────────────────────────────────┘
```

## Labeled Example: Chapter 1, Cautious Profile

### Input: Seed (excerpt)

The Pale Lights seed defines 49 entities. The chapter uses 3:

**Tristan Abrascal** (`char:tristan`) — protagonist. A thin, dark-haired rat from the Murk. Lockpicks, a blackjack, a dosing kit. Polite smile, unhurried hands, a "ticking" in his head that is Fortuna reading odds. Voice: plain with gutter slang, short clipped sentences, masked emotion.

**Cozme Aflor** (`char:cozme`) — first name on Tristan's List. Burn scar near his ear, smug, well-spoken. Serves the Villazur procession; has ties to the people who killed Tristan's father.

**Fortuna** (`char:fortuna`) — a girl-shaped goddess of long odds. Visible only to Tristan. CANNOT touch matter — her presence is perceptual only. Red dress, molten-gold hair and eyes.

### Input: Skeleton Chapter 1

| Field | Value |
|-------|-------|
| Dramatic question | What is the nature of Tristan's debt to Fortuna? |
| POV | `char:tristan` |
| Plot beats | `pt:tristans_list`, `fs:pistol_changes_hands` |
| Tension target | 0.3 |
| Entities to surface | `char:fortuna`, `item:rhadamanthine_pistol` |

### Input: Player Action

For chapter 1, the action comes from the skeleton's plot beats:

```
pt:tristans_list fs:pistol_changes_hands
```

### Stage 1: Arc Planner (0ms, cached)

Determines the story's current structural phase:

```json
{
  "current_phase": "setup",
  "phase_assessment": "Establishing the baseline for the two protagonists
    as they converge on the Dominion. The focus is on their individual
    motivations (the List and the Flight) and the contrasting nature
    of their debts..."
}
```

### Stage 2: Dramatic Planner (0ms, cached)

Breaks the chapter into scenes with characters, locations, and beats:

```json
{
  "action_resolution": {
    "kind": "partial",
    "narrative": "Tristan successfully initiates his strike against
      Cozme Aflor, but the ship's motion interrupts the contact..."
  },
  "scenes": [
    {
      "scene_id": 1,
      "pov_character_id": "char:tristan",
      "location": "loc:bluebell",
      "characters_present": ["char:tristan", "char:cozme", "char:fortuna"],
      "dramatic_question": "Does Tristan successfully strike Cozme Aflor?",
      "beats": [
        "Tristan's hand closes around Cozme's neck/collar.",
        "The sudden roll of the Bluebell breaks the contact.",
        "Cozme turns with a suspicious, knowing glance.",
        "Tristan retreats into the role of a clumsy deckhand."
      ],
      "tension_target": 0.7,
      "reveals": ["Cozme is more observant than a typical noble."],
      "withholds": ["Whether Cozme actually felt the hand or just the wind."]
    }
  ]
}
```

### Stage 3: Emotional Planner (0ms, cached)

Sets the emotional trajectory per scene:

```json
{
  "scene_id": 1,
  "primary_emotion": "satisfaction",
  "secondary_emotion": "confidence",
  "intensity": 0.8,
  "entry_state": "anticipation",
  "exit_state": "readiness"
}
```

### Stage 4: Craft Planner (0ms, cached)

Specifies prose register per scene:

```json
{
  "sentence_variance": "medium",
  "concrete_abstract_ratio": 0.7,
  "interiority_depth": "medium",
  "sensory_density": "moderate",
  "dialogue_ratio": 0.1,
  "pace": "measured"
}
```

### Stage 5: Writer (12 beats, 103s total)

The writer receives a system prompt with narrator config + imagery budget, and a user prompt containing:

1. **Characters in this scene** — full entity descriptions from the seed
2. **Craft brief** — synthesized from the craft/emotional/dramatic planners
3. **Prior prose** — empty for ch1, accumulated for subsequent chapters
4. **Beat instruction** — which beat of the scene to write

Each beat produces ~300-400 words of continuous prose. 12 beats → ~4,300 words raw.

**Beat 1 output** (opening):

> Tristan's hand moved like a shadow passing over a coin — quick, quiet, and gone before the eye could settle. He timed it to the heave of the *Bluebell*, catching the rhythm of the timber and the swell of the sea so his own body felt less like a man and more like a piece of the ship's own heavy motion.
>
> His fingers found the coarse, salt-crusted linen of Cozme's collar.

**Beat 12 output** (closing):

> ...drifting toward a destiny that neither could yet name, but both could feel pressing against their skin like the coming of a storm.

### Stage 6: Check (6s)

The LLM checker flags 3 issues:

| Severity | Issue |
|----------|-------|
| error | Prose repeats the same sequence of events multiple times after the "shadow meeting the light" section |
| warning | Highly repetitive metaphors: "ticking in his head", "geometry of the soul", repeated color patterns |
| info | The Indigo/Fisher vision effectively bridges both protagonists' perspectives |

The LLM metaphor critic also runs, classifying all figurative language by family and flagging any family over budget.

### Stage 7: Revise (41s)

The reviser receives the full prose + issue list and produces a corrected version. Key instruction: "Edit surgically — keep paragraphs that aren't broken."

The repetition error is fixed; the metaphor warning is addressed by swapping excess gambling imagery for other registers.

### Stage 8: Re-check (217s)

Second check finds **0 issues**. Loop exits.

### Stage 9: Extract + Score

**KB extraction** identifies hooks planted, entities introduced, and thread advances.

**4-dimension scoring** (logprob-weighted E[score]):

| Dimension | Score |
|-----------|-------|
| Prose execution | 8.9/10 |
| Subtext | 7.8/10 |
| Hook quality | 6.7/10 |
| Metaphor variety | 8.8/10 |

### Final Output

3,911 words of committed prose. The chapter is persisted to the rollout DB with its scores, trace ID, and KB extract.

## Rollout Comparison: Impulsive vs Cautious

Both rollouts use the same seed, candidate, and skeleton. The virtual-player profile drives action selection.

### Action Divergence

| Chapter | Impulsive | Cautious |
|---------|-----------|----------|
| 1 | (same skeleton opening) | (same skeleton opening) |
| 2 | Hide pistol from Yaotl | Shadow Cozme through corridors |
| 3 | Hide pistol from Angharad | Pickpocket a passenger for info |
| 4 | Encounter target on shore | Pickpocket Cozme directly |
| 5 | Frantic scramble through crowd | Use stolen coin to buy supplies |
| 6 | Lockpicking tools in the crowd | Examine new gear |
| 7 | High-tension chase | Test gear during Trial |
| 8 | Try to grab Cozme | Use Glaring Salt in darkness |
| 9 | Intercept Cozme in crowd | Encounter Cozme on deck |
| 10 | Angharad's trial ascension | Shadow Cozme on deck |

**Impulsive** = 10 chapters of escalating chase. **Cautious** = 6 chapters of methodical preparation, then a controlled approach.

### Diversity Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Action Jaccard (mean) | 0.275 | Different actions chosen |
| Prose 4-gram Jaccard | 0.007 | Almost no shared phrases |
| Entity mention Jaccard | 0.501 | Same cast, different emphasis |
| Hook payoff Jaccard | 1.0 | Both pay same hooks (expected) |

### Aggregate Scores (10 chapters each)

| Dimension | Cautious | Impulsive |
|-----------|----------|-----------|
| Prose execution | 0.867 | 0.856 |
| Subtext | 0.782 | 0.774 |
| Hook quality | 0.739 | 0.690 |
| Metaphor variety | 0.757 | 0.708 |
| **Mean** | **0.786** | **0.757** |

Cautious scores higher across every dimension, with the largest gap in hook quality (+0.049) and metaphor variety (+0.049). The methodical pacing gives the writer more room to plant hooks and vary imagery.

## Eval-Validated Prompt Engineering

We used promptfoo to A/B test writer prompt variants against fixed chapter contexts. Each variant ran on 2 test inputs, scored by the LLM metaphor critic.

### Prompt Variants

| Variant | Strategy |
|---------|----------|
| **baseline** | No imagery instruction |
| **imagery_soft** | "Rotate registers, at most 3-4 times per family" |
| **imagery_hard** | Explicit per-family budget with CLOSED/SWITCH rules |

### Results (test case 1: cautious context)

| Variant | Gambling count | Dominant family | Total figurative |
|---------|---------------|----------------|-----------------|
| baseline | 5 | gambling (33%) | 15 |
| imagery_soft | 5 | gambling (29%) | 17 |
| **imagery_hard** | **2** | **textile (22%)** | **9** |

### Results (test case 2: impulsive context)

| Variant | Gambling count | Dominant family | Total figurative |
|---------|---------------|----------------|-----------------|
| baseline | 0 | predator_prey (23%) | 26 |
| imagery_soft | 2 | water_ocean (25%) | 24 |
| **imagery_hard** | **4** | **bodily (28%)** | **18** |

**Finding:** The soft instruction had no measurable effect vs baseline. The hard constraint with explicit budgets and CLOSED language reduced gambling by 60% and rotated the dominant family away from gambling entirely.

## Architecture Decisions

### Why keyword matching failed for metaphor detection

The keyword list (`IMAGERY_FAMILIES`) caught ~5 gambling hits per chapter. The LLM classifier found ~18. The model generates gambling idioms faster than we can enumerate them — "flush", "the house is looking", "a man about to be bled" all escaped the regex.

**Solution:** LLM classification as primary critic, keyword matching as fallback for tests/offline.

### Why revise loops don't fix metaphor density

We tested 2-revise and 4-revise configurations. Both produced ~18-21 gambling metaphors per chapter — the reviser rewrites paragraphs broadly, re-introducing gambling imagery as fast as it removes it.

**Solution:** Fix the problem upstream in the writer prompt, not downstream in revision. The hard constraint prompt reduced gambling from 18→2-4 per scene at generation time.

### Why per-beat writing outperforms one-shot

The writer produces one beat (~300 words) per LLM call, 12-16 calls per chapter. This produces longer, more continuous prose than single-call generation. Each beat receives prior beats as context, maintaining coherence without exhausting the context window on the full chapter.

### Scoring: logprob-weighted E[score]

Each dimension is scored 1-10 with logprob extraction. The model's probability distribution over score tokens gives a continuous E[score] with SD=0.002 on test-retest, eliminating the quantization problem of integer scores.

## Known Limitations

1. **Metaphor variety score ceiling:** The 4-dim scorer rates metaphor variety at 8.8-8.9/10 even when the LLM critic finds 16+ gambling metaphors. The scorer and critic disagree because the scorer evaluates *variety* (many families present) while the critic evaluates *excess* (any family over budget).

2. **Revise loop inefficiency:** 4 revise passes add ~4 minutes per chapter. The hard writer prompt makes most revisions unnecessary — the check→revise loop should be a safety net, not the primary quality mechanism.

3. **Single model:** All generation and scoring uses the same Gemma 4 26B model. Cross-model validation would catch model-specific biases in both writing and scoring.

4. **N=2 rollouts:** Diversity metrics come from 2 profiles. Adding `honor_bound` and more seeds would strengthen the evidence.

5. **No human calibration:** All scores are model-generated. A human calibration set is needed to validate that 8.9/10 prose execution actually maps to human quality perception.
