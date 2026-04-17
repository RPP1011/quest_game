---
title: "Hierarchical Story Planning with Rollout-Based Refinement"
description: >-
  A six-phase architecture for turning a world seed into a refined
  multi-chapter story. Each phase is empirically validated against
  a Pale Lights baseline.
---

# Hierarchical Story Planning with Rollout-Based Refinement

**Dates:** 2026-04-15 to 2026-04-16  
**Model:** Gemma 4 26B A4B (local, llama-server)  
**Reference corpus:** Pale Lights Book 1 (44 chapters, 341k words, 87 characters, 484 world facts)

---

## The problem

A quest-game pipeline produces prose chapter-by-chapter. The pipeline has four planning layers (arc → dramatic → emotional → craft) feeding a per-beat writer. Each layer narrows the decision space for the layer below.

The pipeline's prose was structurally correct but narratively thin. Comparing generated output to the reference corpus:

<pre class="mermaid" markdown="0">
graph LR
    subgraph "Reference corpus"
        PL_E["87 characters"]
        PL_W["484 world facts"]
        PL_H["198 hooks planted"]
        PL_D["~11 new proper nouns / chapter"]
    end
    subgraph "Generated output"
        US_E["3 entities in seed"]
        US_W["0 world facts referenced"]
        US_H["0 hooks planted"]
        US_D["0 new nouns / chapter"]
    end
    PL_E -.- US_E
    PL_W -.- US_W
    PL_H -.- US_H
    PL_D -.- US_D
    style PL_E fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style PL_W fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style PL_H fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style PL_D fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style US_E fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
    style US_W fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
    style US_H fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
    style US_D fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
</pre>

The initial hypothesis was that the seed was too sparse. The actual finding: **even with a rich seed (49 entities, 10 hooks, 6 threads), the pipeline wasn't reading what was already there.** The writer stage had no access to entity descriptions or narrator config. Fixing that wiring brought generated output to parity with Pale Lights on 7 of 8 quality dimensions. The remaining gap — subtext, emotional trajectory, thematic depth — is structural: a single-pass pipeline can't know whether a chapter's hooks will pay off or whether its tension curve serves the arc.

That structural gap motivated the six-phase architecture below.

---

## Architecture

<pre class="mermaid" markdown="0">
flowchart TB
    SEED["Seed<br/>49 entities, 10 hooks<br/>6 threads, 3 themes<br/>full narrator config"]

    P1["Phase 1: Story Candidates<br/>seed → 3-5 candidate arcs<br/>player picks one"]
    P2["Phase 2: Arc Skeleton<br/>picked candidate → 30-chapter outline<br/>hook schedule, POV alternation"]
    P3["Phase 3: Virtual-Player Rollouts<br/>3 profiles × N chapters<br/>resume-safe, isolated world DBs"]
    P4["Phase 4: KB + Scoring<br/>8-dim judge per chapter<br/>hook/entity/thread extraction"]
    P5["Phase 5: Refinement<br/>3 strategies: weak / hooks / sibling<br/>accept if mean +0.05, no regression"]
    P6["Phase 6: Presentation<br/>static + scaffolded play<br/>(deferred)"]

    SEED --> P1 --> P2 --> P3 --> P4 --> P5 --> P6

    P4 -.->|scores + KB| P5
    P5 -.->|refined chapters| P4

    style P1 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P2 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P3 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P4 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P5 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P6 fill:#3a3222,stroke:#5a4a32,color:#c8b897
</pre>

Each phase produces a persistent artifact that the next phase consumes. The pipeline's existing four-layer planner (arc → dramatic → emotional → craft → writer) runs inside Phase 3; the architecture wraps it with a planning layer above (candidate + skeleton) and an evaluation layer below (scoring + refinement).

---

## Phase 1: Story candidates

**Input:** A seed (entities, threads, hooks, themes, narrator config).  
**Output:** 3–5 candidate story arcs, each committing to a protagonist, primary plot threads, emphasized themes, a climax, and an expected chapter count.

The same seed supports materially different stories. The player (or the rollout harness) picks one before chapter generation starts.

**Why this matters:** Without candidates, the arc planner invents direction on the fly each tick — there's no commitment to a specific arc shape. With candidates, the arc planner reads the picked candidate's thread priorities and climax description as directive context, so every chapter serves the same committed shape.

**Implementation:** One structured LLM call with closed-enum constraints on thread/character/theme ids (the model can't hallucinate entities outside the seed). Candidates persist in SQLite; the pick writes to `config.json` so the pipeline reads it.

---

## Phase 2: Arc skeleton

**Input:** A picked candidate.  
**Output:** A chapter-by-chapter outline: POV, location, dramatic question, required plot beats, target tension, pre-scheduled DORMANT entity activations, theme emphasis. Plus a hook schedule (when each seeded hook pays off) and a theme arc (when each theme crescendos).

```
Tension curve (30 chapters):

Ch  1-5:  ▁▁▂▂▃       setup
Ch  6-10: ▃▃▄▄▄       rising
Ch 11-15: ▅▅▅▆▆       midpoint
Ch 16-20: ▆▆▇▇▇       crisis
Ch 21-24: ▇▇█ █       CLIMAX
Ch 25-27: ▇▆▄         falling
Ch 28-30: ▃▂▁         denouement
```

**Why this matters:** The skeleton is the contract between the arc layer and the chapter-by-chapter pipeline. Without it, the dramatic planner makes local decisions that may never pay off hooks or complete character arcs. With it, every tick consults a pre-committed chapter slot: "this chapter's required plot beats are X; this chapter surfaces entity Y; target tension is 0.7."

**Implementation:** One structured LLM call (larger — 16k max_tokens for 30 chapters). The skeleton is stored in SQLite; `Pipeline._current_skeleton_chapter(update_number)` returns the slot for the current tick, which the arc and dramatic prompts render as pinned directive context.

---

## Phase 3: Virtual-player rollouts

**Input:** A skeleton + a virtual-player profile.  
**Output:** A full chapter-by-chapter playthrough with prose, traces, and world-state mutations — all saved incrementally.

<pre class="mermaid" markdown="0">
flowchart LR
    subgraph "Per rollout"
        BOOT["Bootstrap<br/>isolated quest.db"]
        CH1["Chapter 1<br/>action from skeleton"]
        SEL["Action selector<br/>profile rubric × choices"]
        CH2["Chapter 2..N<br/>pipeline.run per chapter"]
        SAVE["Incremental save<br/>resume from last chapter"]
    end

    BOOT --> CH1 --> SEL --> CH2
    CH2 --> SAVE
    SEL -.-> CH2
    SAVE -.->|resume| CH2

    subgraph Profiles
        IMP["Impulsive<br/>'pick highest stakes'"]
        CAU["Cautious<br/>'gather info first'"]
        HON["Honor-bound<br/>'accept the cost'"]
    end

    Profiles --> SEL
</pre>

**Why this matters:** A single playthrough is one sample from the story-space the seed defines. Different virtual-player profiles explore different branches (impulsive players escalate; cautious players gather information). Multiple rollouts give us a distribution of outcomes — which hooks paid off, which characters arced, which chapters scored well — that one playthrough can't.

**Key design decision: isolated world DBs.** Each rollout copies the main quest DB and wipes its narrative history. Entity activations and world-state mutations stay in the rollout's copy, so rollouts don't interfere with each other. The main DB holds only metadata (rollout rows, chapter rows, scores).

**Action selection:** For chapter 1, the action comes from the skeleton's required plot beats. For subsequent chapters, a small structured LLM call picks among the suggested choices using the profile's rubric. ~5 seconds per decision.

---

## Phase 4: Scoring + KB extraction

**Input:** A rollout's chapters (prose + traces).  
**Output:** Per-chapter quality scores on 8 dimensions, plus extracted world events (hook payoffs, entity introductions, thread advances).

<pre class="mermaid" markdown="0">
flowchart TB
    CH["RolloutChapter<br/>prose + trace"]

    CH --> SCORE["8-dim judge<br/>tension, emotion, voice<br/>theme, subtext, interiority<br/>choice_hook, self_containment"]
    CH --> KB["KB extractor<br/>hook payoffs<br/>entity mentions<br/>thread advances"]

    SCORE --> KBS["kb_chapter_scores<br/>per-dim per-chapter"]
    KB --> KBH["kb_hook_payoffs"]
    KB --> KBE["kb_entity_usage"]

    KBS --> AGG["Aggregated views<br/>payoff rates, screen time<br/>dim means by chapter index"]
    KBH --> AGG
    KBE --> AGG
</pre>

**The 8 dimensions** (rubrics under `prompts/scoring/chapter_dims/`):

| Dimension | What it measures |
|---|---|
| tension_execution | Does tension rise, peak, and release within the chapter? |
| emotional_trajectory | Does the emotional state shift meaningfully? |
| voice_distinctiveness | Is the narrator's voice consistent and recognizable? |
| thematic_presence | Do the chapter's events serve a theme? |
| subtext_presence | Is meaning carried by implication, not statement? |
| interiority_depth | How deep is the reader's access to the POV character's mind? |
| choice_hook_quality | Does the chapter end on a hook that makes the reader want to choose? |
| update_self_containment | Can the chapter stand on its own without prior context? |

**KB extraction** is zero-cost (pure trace parsing): which hooks were planted or paid off, which DORMANT entities were promoted, which threads advanced. Aggregation across rollouts answers questions like "which hook paid off in <50% of rollouts?" and "which chapter index scores lowest on subtext?"

---

## Phase 5: Refinement

**Input:** A scored rollout + KB analysis identifying weak points.  
**Output:** Targeted chapter rewrites that demonstrably improve on the original.

<pre class="mermaid" markdown="0">
flowchart TB
    KB["KB scores + analysis"]

    KB --> S1["WeakChapterSelector<br/>mean dim below threshold"]
    KB --> S2["UnpaidHookSelector<br/>skeleton vs actual payoffs"]
    KB --> S3["SiblingOutscoredSelector<br/>another rollout scored higher"]

    S1 --> REGEN["Regenerate chapter<br/>with strategy-specific guidance"]
    S2 --> REGEN
    S3 --> REGEN

    REGEN --> RESCORE["Score refined prose"]

    RESCORE --> DECIDE{"Improved enough?<br/>mean +0.05, no dim regression"}
    DECIDE -->|yes| ACCEPT["Accept<br/>update canonical chapter"]
    DECIDE -->|no| REJECT["Reject<br/>keep original"]

    style ACCEPT fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style REJECT fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
</pre>

**Three selectors, one framework:**

- **WeakChapterSelector** — picks chapters whose mean dim score is below a threshold (default 0.55). Guidance names the worst dim and quotes the judge's rationale.
- **UnpaidHookSelector** — compares the skeleton's hook schedule against actual KB payoffs. If hook X should have paid off by chapter 14 but didn't, target chapter 14 with guidance to land the payoff.
- **SiblingOutscoredSelector** — if another rollout's same chapter scored ≥0.15 higher on any dim, include that sibling's prose as a reference ("look at how this version handled subtext").

**Accept thresholds:** mean improvement ≥ +0.05 AND no per-dim regression > -0.10. This prevents refinement from trading one quality for another.

---

## Empirical results

### Refinement of a weak chapter

| Dim | baseline | refined | Δ | Pale Lights |
|---|---|---|---|---|
| subtext_presence | 0.30 | **0.60** | **+0.30** | 0.60 |
| voice_distinctiveness | 0.60 | **0.90** | **+0.30** | 0.60 |
| interiority_depth | 0.60 | **0.90** | **+0.30** | 0.70 |
| tension_execution | 0.70 | 0.70 | 0.00 | 0.90 |
| emotional_trajectory | 0.60 | 0.60 | 0.00 | 0.90 |
| thematic_presence | 0.60 | 0.60 | 0.00 | 0.90 |
| **mean** | **0.60** | **0.713** | **+0.113** | **0.75** |

The targeted dim (subtext, worst at 0.30) doubled. Voice and interiority improved alongside. The refined chapter is within 0.04 of the Pale Lights baseline mean. Zero per-dim regression.

### What the per-beat writer loop buys

We tested 5 write-stage strategies sharing the same plan:

| Strategy | Words | Time | Mean dim |
|---|---|---|---|
| **per_beat** (1 LLM call per beat, ~15 calls) | 5,588 | 82s | **0.762** |
| expand (sketch + per-scene expansion) | 3,043 | 50s | 0.675 |
| one_shot (single call, full chapter) | 626 | 7s | 0.525 |

**per_beat matches Pale Lights** at 0.762 mean. one_shot collapsed to 626 words — the 26B model produces summaries when given "write everything at once." The per-beat loop forces the writer to commit to each beat in full, preventing compression.

---

## The seed

The system runs on a hand-authored JSON seed. For Pale Lights:

<pre class="mermaid" markdown="0">
pie title "49 entities by type"
    "Characters 19" : 19
    "Locations 6" : 6
    "Factions 8" : 8
    "Concepts 7" : 7
    "Items 7" : 7
</pre>

<pre class="mermaid" markdown="0">
pie title "Entity status at seed time"
    "Active 20" : 20
    "Dormant 29" : 29
</pre>

Plus: 7 world rules, 6 plot threads, 10 foreshadowing hooks, 4 motifs with semantic ranges, 3 themes, and a narrator config block (POV type, register, worldview, editorial stance, sensory bias, attention bias, voice samples, unreliability axes).

DORMANT entities (29 of 49) exist in the world but haven't appeared on-screen. The dramatic planner sees them in a "Dormant Entities (available to surface)" section and can choose 0–3 per update to introduce. The skeleton pre-schedules which dormant entities surface in which chapters — so the world "opens up" on a planned cadence, not randomly.

---

## Key architectural decisions

**Why candidates before chapters?** Without a candidate pick, the arc planner invents direction per-tick. Two runs of the same seed produce incoherent arcs. With a candidate, every tick serves a committed shape — different seeds, different candidates, but each candidate is a coherent story.

**Why skeletons before rollouts?** The skeleton is the contract the pipeline can be held to. Hook payoffs that were "planned but never happened" become detectable failures that the refinement loop can fix. Without a skeleton, there's no ground truth for what the story was supposed to do.

**Why isolated world DBs per rollout?** Entity activations, narrative history, and world-state mutations must not bleed between rollouts. Each rollout starts from the same pristine seed state. This means rollouts can run in parallel (GPU-permitting) and the KB aggregation is well-defined: "which hooks paid off in rollout X?" is a query over rollout X's isolated state.

**Why per-beat writer loop instead of one-shot?** Empirically: per_beat at 0.762 vs one_shot at 0.525 on the same plan. The 26B model compresses aggressively when given a full chapter to write at once. Per-beat forces commitment to each narrative moment. The cost is ~15 LLM calls per chapter instead of 1, but each call is small and fast.

**Why score-gated refinement instead of always-accept?** A refinement can improve subtext while regressing voice. The dual threshold (mean must improve by ≥0.05, no dim may regress by >0.10) prevents quality trading. Rejected refinements are still persisted for analysis.

---

## What's next

**Remaining gap:** emotional_trajectory and thematic_presence are both 0.30 below the Pale Lights baseline even after refinement. These likely require multi-chapter awareness — a single chapter's emotional arc can't be judged without knowing how it sits relative to the surrounding chapters. The skeleton provides this context; the refinement guidance doesn't yet use it.

**Phase 6 (deferred):** Two presentation modes — static (read-only paginated prose) and scaffolded (real player with the skeleton as guardrails and the KB as a quality floor).

**Writer LoRA:** The scored rollout chapters are natural training pairs: (plan, prose). A writer LoRA trained on high-scoring rollout chapters could internalize the patterns that currently require the refinement loop to achieve.

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: false, theme: 'dark' });
  const nodes = document.querySelectorAll('pre.mermaid, div.mermaid');
  for (const n of nodes) {
    try {
      const src = n.textContent.trim();
      const id = 'm' + Math.random().toString(36).slice(2, 9);
      const { svg } = await mermaid.render(id, src);
      const wrap = document.createElement('div');
      wrap.innerHTML = svg;
      n.replaceWith(wrap);
    } catch (e) {
      console.warn('mermaid render failed:', e.message);
      n.style.border = '1px dashed #c78a7a';
      n.style.color = '#c78a7a';
    }
  }
</script>
