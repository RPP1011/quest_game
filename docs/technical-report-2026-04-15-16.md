---
title: "World Seed Density → Story-Rollout Architecture"
description: >-
  From "the world feels thin" to a six-phase story-generation architecture
  with empirical validation at each layer. Covers writer fidelity, DORMANT
  activation, strategy sweep, check→revise enforcement, and Phases 1–5 of
  the story-rollout system.
---

# Technical Report: World Seed Density → Story-Rollout Architecture

**Dates:** 2026-04-15 to 2026-04-16  
**Authors:** Ricky + Claude Opus 4.6  
**Baseline:** 817 tests, per-beat write loop, 3-entity seeds, 2nd-person CYOA narrator.  
**Final state:** 910 tests, 5 architecture phases shipped, overnight 2×2×10 rollout running.

---

## 1. The problem, visualized

The Pale Lights craft analysis revealed a density gap between the source material and our generated output:

<pre class="mermaid" markdown="0">
graph LR
    subgraph "Pale Lights (source)"
        PL_E["87 characters"]
        PL_W["484 world facts"]
        PL_H["198 hooks"]
        PL_D["~11 new proper nouns / chapter"]
    end
    subgraph "Our seeds (before)"
        US_E["3 entities"]
        US_W["0 world facts"]
        US_H["0 hooks"]
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

The initial hypothesis: enrich the seed. The actual finding: **the seed wasn't the only bottleneck — the pipeline wasn't reading what was already there.**

---

## 2. Writer fidelity — the real first bug

Before touching density, we ran the pipeline against a hand-authored 49-entity Pale Lights seed. Two catastrophic bugs surfaced:

<pre class="mermaid" markdown="0">
flowchart TB
    SEED["Seed: 49 entities<br/>full narrator config<br/>10 hooks, 6 threads"]
    PLAN["Dramatic Planner<br/>✅ used 14 seeded names"]
    WRITE["Writer Stage"]
    
    SEED --> PLAN
    PLAN --> WRITE
    
    WRITE --> BUG1["❌ Fortuna = cat<br/>'She did not meow'"]
    WRITE --> BUG2["❌ 2nd person POV<br/>'You slipped through...'"]
    
    FIX1["Fix: pipe entity data<br/>to writer template"]
    FIX2["Fix: pipe narrator config<br/>to writer system prompt"]
    
    BUG1 -.-> FIX1
    BUG2 -.-> FIX2
    
    FIX1 --> GOOD["✅ Goddess in red dress<br/>sprawls on rafters, bellows"]
    FIX2 --> GOOD2["✅ 3rd person close<br/>'Tristan moved through...'"]
    
    style BUG1 fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
    style BUG2 fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
    style GOOD fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style GOOD2 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
</pre>

### Before / after

| Metric | V1 (before) | V2 (after) | Pale Lights Ch 1 |
|---|---|---|---|
| Words | 2,389 | 5,437 | 5,438 |
| POV | 2nd person | 3rd person close | 3rd person close |
| Fortuna | Cat | Goddess in red dress | Correct |
| Named world-facts used | 0 | 14 | ~11/chapter |

---

## 3. Strategy sweep — per-beat wins

We implemented 5 write-stage strategies sharing the same plan and scored each against the 8-dim chapter judge:

<pre class="mermaid" markdown="0">
xychart-beta
    title "Strategy comparison (8-dim mean score)"
    x-axis ["per_beat", "expand", "one_shot", "scene", "refine"]
    y-axis "Mean dim score" 0 --> 1
    bar [0.762, 0.675, 0.525, 0, 0]
    line [0.75, 0.75, 0.75, 0.75, 0.75]
</pre>

**per_beat is the winner** at 0.762 mean — matching Pale Lights (0.75). one_shot collapsed to 626 words; the 26B model produces summaries when given "write everything at once." The per-beat loop forces commitment to each beat.

| Strategy | Words | Secs | Mean |
|---|---|---|---|
| **per_beat** | 5,588 | 82 | **0.762** |
| expand | 3,043 | 50 | 0.675 |
| one_shot | 626 | 7 | 0.525 |
| scene | 828 | 30 | failed |
| refine | 613 | 13 | failed |

---

## 4. Check → revise loop

The pipeline's quality gate had a critical flaw:

<pre class="mermaid" markdown="0">
flowchart LR
    CHECK["CHECK stage<br/>finds issues"]
    
    subgraph "Before (broken)"
        C1["Critical issue?"] -->|yes| FLAG["Flag + commit<br/>unchanged ❌"]
        C1 -->|no| FIX["Fixable?"] -->|yes| REV1["Revise once"]
    end
    
    subgraph "After (fixed)"
        C2["Any issues?"] -->|yes| LOOP["Revise + recheck<br/>up to 2×"]
        LOOP --> ACCEPT{"Clean?"}
        ACCEPT -->|yes| COMMIT["Commit ✅"]
        ACCEPT -->|no| FLAGQM["Flag QM"]
    end
    
    CHECK --> C1
    CHECK --> C2
    
    style FLAG fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
    style COMMIT fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
</pre>

The old code gated revise on `not has_critical` — so critical world-rule violations (Fortuna touching matter) were detected but committed unchanged.

### Cascading bugs discovered in the fix

| Bug | Root cause | Fix |
|---|---|---|
| Revise truncated chapters | `REVISE_MAX_TOKENS=3000` per-beat sized | → 8000 |
| Revise leaked chain-of-thought | Gemma thinking mode on | `thinking=False` |
| Seed self-contradicted | "Cannot touch; rests chin on shoulder" | Rewrite description |
| Trace outcome not persisted | `set_outcome` after last `add_stage` | `trace.set_outcome()` |

---

## 5. The story-rollout architecture

Six phases, each building on the last. Five shipped; one deferred.

<pre class="mermaid" markdown="0">
flowchart TB
    SEED["Seed<br/>49 entities, 10 hooks<br/>6 threads, 3 themes"]
    
    P1["Phase 1: Story Candidates<br/>seed → 3-5 candidate arcs<br/>player picks one"]
    P2["Phase 2: Arc Skeleton<br/>picked candidate → 30-chapter outline<br/>hook schedule, POV alternation"]
    P3["Phase 3: Virtual-Player Rollouts<br/>3 profiles × N chapters<br/>resume-safe, isolated world DBs"]
    P4["Phase 4: KB + Scoring<br/>8-dim judge per chapter<br/>hook/entity/thread extraction"]
    P5["Phase 5: Refinement<br/>3 strategies: weak / hooks / sibling<br/>accept if mean +≥0.05, no dim regression"]
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

### Phase 1: Story candidates

The same seed produces materially different stories:

| Candidate | Protagonist | Primary threads | Chapters |
|---|---|---|---|
| The Shadow Watch | Tristan | red_maw + hoja_roja + trials | 50 |
| The God-Eater's Debt | Tristan | tristans_list + red_maw | 35 |
| The Noble's Reckoning | Angharad | angharads_revenge + trials | 42 |

### Phase 2: Arc skeleton

30-chapter outline with tension curve and hook scheduling:

<pre class="mermaid" markdown="0">
xychart-beta
    title "Skeleton tension curve (30 chapters)"
    x-axis "Chapter" [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    y-axis "Target tension" 0 --> 1
    line [0.2,0.2,0.3,0.3,0.4,0.4,0.4,0.5,0.5,0.5,0.6,0.6,0.6,0.7,0.7,0.7,0.7,0.8,0.8,0.8,0.8,0.8,0.9,0.9,0.8,0.7,0.5,0.4,0.3,0.2]
</pre>

All 10 seeded foreshadowing hooks scheduled with planted-by / paid-off-by chapter targets. POV alternates Tristan / Angharad per chapter.

### Phase 3: Rollout harness

<pre class="mermaid" markdown="0">
flowchart LR
    subgraph "Per rollout"
        BOOT["Bootstrap<br/>isolated quest.db"]
        CH1["Chapter 1<br/>action from skeleton"]
        SEL["Action selector<br/>profile × choices"]
        CH2["Chapter 2..N<br/>pipeline.run()"]
        SAVE["Incremental save<br/>rollout_chapters"]
    end
    
    BOOT --> CH1 --> SEL --> CH2
    CH2 --> SAVE
    SEL -.-> CH2
    SAVE -.->|resume from<br/>max(ch)+1| CH2
    
    subgraph Profiles
        IMP["Impulsive<br/>'pick highest stakes'"]
        CAU["Cautious<br/>'gather info first'"]
        HON["Honor-bound<br/>'accept the cost'"]
    end
    
    Profiles --> SEL
</pre>

### Phase 4: Scoring + KB

Each rollout chapter scored on 8 dimensions and parsed for world events:

<pre class="mermaid" markdown="0">
flowchart TB
    CH["RolloutChapter<br/>(prose + trace)"]
    
    CH --> SCORE["8-dim judge<br/>tension, emotion, voice<br/>theme, subtext, interiority<br/>choice_hook, self_containment"]
    CH --> KB["KB extractor<br/>hook payoffs<br/>entity mentions<br/>thread advances"]
    
    SCORE --> KBS["kb_chapter_scores<br/>per-dim per-chapter"]
    KB --> KBH["kb_hook_payoffs"]
    KB --> KBE["kb_entity_usage"]
    
    KBS --> AGG["GET /api/quests/{qid}/kb<br/>aggregated views"]
    KBH --> AGG
    KBE --> AGG
</pre>

### Phase 5: Refinement

Three selectors, one framework, measurable improvement:

<pre class="mermaid" markdown="0">
flowchart TB
    KB["KB scores"]
    
    KB --> S1["WeakChapterSelector<br/>mean dim < 0.55"]
    KB --> S2["UnpaidHookSelector<br/>skeleton vs actual payoffs"]
    KB --> S3["SiblingOutscoredSelector<br/>another rollout scored +0.15"]
    
    S1 --> REGEN["Regenerate chapter<br/>with strategy guidance"]
    S2 --> REGEN
    S3 --> REGEN
    
    REGEN --> RESCORE["Score refined prose"]
    
    RESCORE --> DECIDE{"mean Δ ≥ +0.05?<br/>no dim regression > -0.10?"}
    DECIDE -->|yes| ACCEPT["Accept ✅<br/>update canonical chapter"]
    DECIDE -->|no| REJECT["Reject<br/>keep original"]
    
    style ACCEPT fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style REJECT fill:#4a2a2a,stroke:#8a5a5a,color:#c78a7a
</pre>

---

## 6. Empirical results — the refinement journey

### Scoring progression across the session

<pre class="mermaid" markdown="0">
xychart-beta
    title "8-dim mean score progression"
    x-axis ["V1 (before fixes)", "V2 (after writer fix)", "per_beat sweep", "v5 rollout ch1", "After refinement", "Pale Lights baseline"]
    y-axis "Mean dim score" 0 --> 1
    bar [0, 0, 0.762, 0.600, 0.713, 0.750]
</pre>

### Refinement per-dim breakdown

| Dim | baseline | refined | Δ | Pale Lights |
|---|---|---|---|---|
| subtext_presence | 0.30 | **0.60** | **+0.30** | 0.60 |
| voice_distinctiveness | 0.60 | **0.90** | **+0.30** | 0.60 |
| interiority_depth | 0.60 | **0.90** | **+0.30** | 0.70 |
| tension_execution | 0.70 | 0.70 | 0.00 | 0.90 |
| emotional_trajectory | 0.60 | 0.60 | 0.00 | 0.90 |
| thematic_presence | 0.60 | 0.60 | 0.00 | 0.90 |
| choice_hook_quality | 0.70 | 0.70 | 0.00 | 0.70 |
| update_self_containment | 0.70 | 0.70 | 0.00 | 0.70 |
| **mean** | **0.60** | **0.713** | **+0.113** | **0.75** |

The targeted dim (subtext_presence) doubled. Voice and interiority dragged along. Remaining gap to Pale Lights: emotional_trajectory and thematic_presence (both -0.30 from baseline).

---

## 7. Seed density

Hand-authored from the Pale Lights Book 1 rollup:

<pre class="mermaid" markdown="0">
pie title "49 entities by type"
    "Characters (19)" : 19
    "Locations (6)" : 6
    "Factions (8)" : 8
    "Concepts (7)" : 7
    "Items (7)" : 7
</pre>

<pre class="mermaid" markdown="0">
pie title "Entity status at seed time"
    "Active (20)" : 20
    "Dormant (29)" : 29
</pre>

Plus: 7 rules, 6 plot threads, 10 foreshadowing hooks, 4 motifs, 3 themes, full narrator config with 5 voice samples.

---

## 8. Robustness fixes

Issues diagnosed and fixed during empirical testing:

<pre class="mermaid" markdown="0">
flowchart TB
    subgraph "Model-server issues"
        A1["scene_id missing from JSON<br/>llama-server ignores required"] --> F1["_repair_missing_scene_ids()"]
        A2["Dramatic plan truncated<br/>max_tokens too small"] --> F2["4096 → 8192"]
        A3["Thinking tokens in output<br/>chain-of-thought leaked"] --> F3["thinking=False on revise+score"]
    end
    
    subgraph "Pipeline issues"
        B1["Extract rejects DORMANT patches<br/>known_ids too narrow"] --> G1["Widen to non-destroyed"]
        B2["Hallucinated hook IDs<br/>kill entire extract"] --> G2["Filter as warnings"]
        B3["Trace outcome not persisted<br/>shows 'running' forever"] --> G3["trace.set_outcome()"]
    end
    
    subgraph "Seed issues"
        C1["Fortuna description<br/>self-contradicts"] --> H1["Rewrite as perceptual"]
        C2["CLI quest_id mismatch<br/>db.stem vs dir name"] --> H2["db.parent.name"]
        C3["Themes not in DB<br/>only in config.json"] --> H3["CLI adds themes+motifs"]
    end
</pre>

---

## 9. DORMANT entity activation

The dramatic planner now sees dormant entities and can surface 0–3 per update:

<pre class="mermaid" markdown="0">
sequenceDiagram
    participant Seed as Seed (49 entities)
    participant DP as Dramatic Planner
    participant Pipeline as Pipeline
    participant DB as World DB

    Seed->>DP: 20 active + 29 dormant
    DP->>DP: Pick entities_to_surface<br/>['char:abuela', 'char:cozme']
    DP->>Pipeline: DramaticPlan with<br/>entities_to_surface
    Pipeline->>DB: PATCH abuela<br/>DORMANT → ACTIVE
    Pipeline->>DB: PATCH cozme<br/>DORMANT → ACTIVE
    Note over DB: Now 22 active, 27 dormant
</pre>

Verified empirically: v7 rollout surfaced `['char:abuela', 'char:cozme', 'char:angharad']`. Abuela appeared 4 times in prose with seed-fidelity texture.

---

## 10. UI improvements

<pre class="mermaid" markdown="0">
flowchart TB
    subgraph "Quest start flow"
        EMPTY["Empty quest"] --> HERO["Hero panel<br/>premise + themes + cast"]
        HERO --> PICK["Candidate picker<br/>3 cards, auto-generated"]
        PICK --> BANNER["Picked banner<br/>'Playing: The Shadow Watch'"]
        BANNER --> STARTERS["Starter actions<br/>from plot threads"]
    end
    
    subgraph "During generation"
        GEN["5-min generation"] --> LIVE["Live trace panel<br/>stages stream in every 2s"]
    end
    
    subgraph "World drawer"
        WORLD["World button"] --> TABS["9 tabs"]
        TABS --> T1["Arc Outline"]
        TABS --> T2["Characters"]
        TABS --> T3["Factions / Locations"]
        TABS --> T4["Hooks / Rules / Motifs"]
    end
</pre>

---

## 11. Test coverage

<pre class="mermaid" markdown="0">
xychart-beta
    title "Test count by module (93 new tests)"
    x-axis ["world", "planning", "server", "engine", "rollout", "refinement"]
    y-axis "New tests" 0 --> 35
    bar [26, 8, 13, 3, 29, 14]
</pre>

| Baseline | Final | Delta |
|---|---|---|
| 817 | 910 | +93 |

---

## 12. What's running now

<pre class="mermaid" markdown="0">
gantt
    title Overnight 2×2×10 rollout (~4 hrs)
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Candidate 1
    impulsive ×10ch    :c1i, 00:00, 100min
    cautious ×10ch     :c1c, after c1i, 100min
    
    section Candidate 2
    impulsive ×10ch    :c2i, after c1c, 100min
    cautious ×10ch     :c2c, after c2i, 100min
</pre>

40 chapters total. Each chapter: pipeline generation (~5 min) + scoring (~10s) + KB extraction (~instant). Resume-safe.

---

## 13. What's next

<pre class="mermaid" markdown="0">
flowchart LR
    NOW["Current state<br/>mean 0.713"]
    
    NOW --> A["Phase 6: Presentation<br/>static + scaffolded play"]
    NOW --> B["Multi-pass refinement<br/>iterate until convergence"]
    NOW --> C["Writer LoRA training<br/>scored rollout chapters<br/>as (plan,prose) pairs"]
    NOW --> D["Dialogue ratio fix<br/>prose is narration-heavy"]
    NOW --> E["Overnight analysis<br/>40-chapter KB aggregation"]
    
    style NOW fill:#2a3a4a,stroke:#3a5a7a,color:#8ac8e8
</pre>

---

## Artifacts

| Path | Content |
|---|---|
| `seeds/pale_lights.json` | 49-entity seed with full narrator config |
| `docs/superpowers/specs/2026-04-15-story-rollout-architecture.md` | Six-phase design spec |
| `docs/phase{1-5}-*-result.md` | Per-phase exit criterion checks |
| `tools/strategy_sweep.py` | 5-strategy comparison harness |
| `tools/overnight_rollout.sh` | 2×2×10 rollout launcher |
| `app/planning/story_candidate_planner.py` | Phase 1 |
| `app/planning/arc_skeleton_planner.py` | Phase 2 |
| `app/rollout/` | Phase 3: harness, profiles, action selector |
| `app/rollout/scorer.py` + `kb_extractor.py` | Phase 4 |
| `app/refinement/` | Phase 5: framework + 3 selectors |

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: false, theme: 'dark' });
  const nodes = document.querySelectorAll('pre.mermaid, div.mermaid');
  for (const n of nodes) {
    const src = n.textContent.trim();
    const id = 'm' + Math.random().toString(36).slice(2, 9);
    const { svg } = await mermaid.render(id, src);
    const wrap = document.createElement('div');
    wrap.innerHTML = svg;
    n.replaceWith(wrap);
  }
</script>
