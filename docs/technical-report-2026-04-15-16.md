---
title: "Hierarchical Story Planning with Rollout-Based Refinement"
description: >-
  A six-phase architecture for turning a world seed into a refined
  multi-chapter story. What works, what's broken, what's next.
---

# Hierarchical Story Planning with Rollout-Based Refinement

**Dates:** 2026-04-15 to 2026-04-17  
**Model:** Gemma 4 26B A4B (local, llama-server)  
**Reference corpus:** Pale Lights Book 1 (44 chapters, 341k words, 87 characters, 484 world facts)

---

## What we built

The existing quest-game pipeline has four planning layers (arc → dramatic → emotional → craft) feeding a per-beat writer. Each layer narrows the decision space for the layer below. The pipeline produces structurally correct prose chapter-by-chapter.

We wrapped this pipeline with a planning layer above and an evaluation layer below:

<pre class="mermaid" markdown="0">
flowchart TB
    SEED["Seed<br/>49 entities, 10 hooks<br/>6 threads, 3 themes<br/>full narrator config"]

    P1["Phase 1: Story Candidates<br/>seed → 3-5 candidate arcs<br/>player picks one"]
    P2["Phase 2: Arc Skeleton<br/>picked candidate → 30-chapter outline<br/>hook schedule, POV alternation"]
    P3["Phase 3: Virtual-Player Rollouts<br/>existing pipeline runs here<br/>3 profiles × N chapters"]
    P4["Phase 4: KB + Scoring<br/>8-dim judge per chapter<br/>hook/entity/thread extraction"]
    P5["Phase 5: Refinement<br/>3 strategies: weak / hooks / sibling<br/>accept if judge says improved"]

    SEED --> P1 --> P2 --> P3 --> P4 --> P5

    P4 -.->|scores + KB| P5
    P5 -.->|refined chapters| P4

    style P1 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P2 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P3 fill:#2a3a4a,stroke:#3a5a7a,color:#8ac8e8
    style P4 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
    style P5 fill:#2a4a2a,stroke:#5a8a5a,color:#8abc7a
</pre>

Phases 1, 2, 4, and 5 are new. Phase 3 is the existing pipeline, unchanged. The architecture is designed so each phase produces a persistent artifact the next phase consumes.

**Phase 1 (Story Candidates):** One LLM call with closed-enum constraints produces 3–5 candidate story arcs from a seed. The player picks one. Without this, the arc planner invents direction per-tick with no commitment to a coherent shape.

**Phase 2 (Arc Skeleton):** One LLM call produces a 30-chapter outline for the picked candidate: POV alternation, tension curve, required plot beats per chapter, pre-scheduled hook payoffs and entity activations. Without this, there's no ground truth for what the story was supposed to do — hooks that "never pay off" are undetectable.

**Phase 3 (Rollouts):** The existing pipeline runs chapter-by-chapter against the skeleton. Virtual-player profiles (impulsive, cautious, honor-bound) drive action selection. Each rollout gets an isolated world DB so mutations don't bleed between playthroughs. Resume-safe: each chapter saves incrementally.

**Phase 4 (Scoring + KB):** Each chapter is scored on 8 dimensions by the same model that generated it. Hook payoffs, entity usage, and thread advances are extracted from traces. Results aggregate across rollouts via SQL.

**Phase 5 (Refinement):** Three selectors identify chapters to retry (weak score, unpaid hook, sibling outscored). The pipeline regenerates with guidance. A gate decides whether to accept the refinement.

---

## What works

**The skeleton shapes output.** When the dramatic planner has a skeleton chapter to consult (required plot beats, target tension, entities to surface), it follows it. Empirically: skeleton ch1 prescribed "Tristan acquires the Rhadamanthine pistol" and "Tristan meets Fortuna" — the dramatic plan's scene 1 was "Can Tristan lift the Rhadamanthine pistol?" with Fortuna as scene 2's focus. POV alternation between Tristan and Angharad emerged from the skeleton without dedicated plumbing.

**Entity fidelity responds to context.** When entity descriptions are piped to the writer, the writer uses them. Before the fix: Fortuna was rendered as a cat. After: "a golden, weightless presence" that "sprawled across the main beam." The narrator config (POV type, register, voice samples) reached the writer and controlled the output — third-person close instead of second-person CYOA.

**Per-beat writer loop outperforms alternatives.** Five strategies tested on the same plan:

| Strategy | Words | Mean dim | Note |
|---|---|---|---|
| **per_beat** | 5,588 | 0.762 | 15 LLM calls, each writes one beat |
| expand | 3,043 | 0.675 | sketch + per-scene expansion |
| one_shot | 626 | 0.525 | single call, full chapter |

The 26B model compresses aggressively in one_shot mode (626 words for a 3-scene plan). Per-beat forces commitment.

**DORMANT activation works.** The dramatic planner surfaces 0–3 dormant entities per chapter. The pipeline activates them in the world DB. Verified: Abuela and Cozme promoted from DORMANT → ACTIVE, Abuela appeared in prose with seed-fidelity texture.

**The check→revise loop catches real violations.** The check stage detected Fortuna touching matter (violating a seeded world rule). After fixing the revise loop to fire on critical issues (it previously skipped them), the loop converges in 2 passes on non-contradictory seeds.

---

## What's broken

### The scoring layer is circular

Every empirical claim in this report is "Gemma 4 26B thinks my Gemma 4 26B output is good." The judge is the same model class as the generator, scoring on subjective dimensions (subtext, voice, interiority), with no external validation.

"Within 0.04 of Pale Lights mean" means the judge model finds them comparably good — which is a statement about the judge's discrimination, not about quality. The Pale Lights baseline is scored by the same judge, so the comparison is relative within the judge's taste, not an external quality assessment.

**What needs to happen:** test-retest variance (score the same chapter 5× to measure the SD per dim), cross-model validation (a second judge model on the same chapters), and a small human-rated calibration set (n≥20) to establish that the judge correlates with what we actually care about.

### The scores are quantized below the acceptance threshold

Score values in the refinement table are all multiples of 0.10 (0.30, 0.60, 0.70, 0.90). The +0.05 acceptance threshold for refinement is below the measurement resolution of a single dim. The mean-over-dims smooths this somewhat, but we're calling improvements at finer precision than the instrument provides.

**Fix in progress:** switch to integer 1–10 scoring with logprob-weighted E[score]. The model's actual distribution over tokens is continuous; the quantization is the sampling layer, not the judgment. Extracting logprobs from llama-server and computing E[score] = Σ(digit × prob)/10 gives continuous values with no extra forward passes. This also exposes judge confidence (spiky distribution = confident; flat = uncertain).

### The refinement result is N=1

One chapter, baseline 0.60 → refined 0.713. That's a case study, not evidence. Open questions:
- What fraction of refinement attempts get accepted?
- What's the conditional mean improvement given acceptance?
- Do untargeted dims drift?
- Is the gate filtering noise into an apparent win?

The overnight 2×2×10 rollout (40 chapters, running) will provide the data for a real distribution.

### The 8 dimensions are probably 3

Subtext, interiority, voice, and thematic presence all read off the same surface properties of prose. An LLM judge will score them as a correlated bundle. The prediction: PCA on scored chapters will show 2–3 components explaining >80% of variance, with tension_execution and choice_hook_quality loading on a different axis than the prose-craft dims. If confirmed, the 8-dim rubric is theater and should be collapsed.

### update_self_containment is miscalibrated

We're building serial fiction modeled on collaborative web narrative. Chapter 14 should leverage setup from chapters 3 and 7 — not stand on its own. Rewarding self-containment pushes the writer toward exposition and recap, which is exactly the AI-fiction failure mode. This dim should be dropped or inverted.

### Comparisons are forced through absolute scoring

Refinement gate, sibling selection, and PL anchoring are all comparison tasks. Absolute scoring is the wrong tool: noisy, quantized, model-taste-dependent. Pairwise comparison ("which is better?") eliminates quantization — the judge just picks — and the literature shows it's more reliable for LLM judges.

**Fix in progress:** pairwise comparisons for refinement gate (refined vs baseline), sibling selection (rollout A vs B), and PL anchoring (our chapter vs Pale Lights excerpt). Logprobs on the A/B token give P(A wins) continuously. Keep absolute scoring only where we need a number across chapters (weak-chapter ranking).

### No cross-chapter evaluation exists

Every dimension is chapter-local. The things readers actually complain about in long-form AI fiction — character drift, plot inconsistency, setups that never pay off, pacing collapse at chapter 20 — the judge can't see any of it, because it never sees two adjacent chapters.

UnpaidHookSelector is the only mechanism touching cross-chapter structure, and it's driven by trace parsing (the planner said it planted a hook), not prose verification (the prose actually contains a legible reference). A hook recorded as "planted" in the trace but not readable in the text would slip through.

**Needed:** A cross-chapter coherence dim scored on chapter pairs. And a prose-level hook verifier: given chapter text and a claimed hook, does an independent read detect the hook reference?

### Rollout diversity is unverified

Three profiles feed choices into the same LLM, which is also generating the choices. Without measuring trajectory divergence (entity-introduction Jaccard across rollouts, hook-payoff overlap, prose n-gram similarity), "different profiles explore different branches" is assumed, not demonstrated. If impulsive and cautious produce 80% identical KB deltas, the 3× cost is buying correlated noise.

---

## The seed

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

Hand-authored from the Pale Lights Book 1 rollup. 49 entities (20 active, 29 dormant), 7 world rules, 6 plot threads, 10 foreshadowing hooks, 4 motifs, 3 themes, full narrator config with 5 voice samples. DORMANT entities surface on a cadence controlled by the skeleton.

---

## Key architectural decisions

**Why candidates before chapters?** Without a candidate pick, two runs of the same seed produce incoherent arcs. With a candidate, every tick serves a committed shape.

**Why skeletons before rollouts?** The skeleton is the contract the pipeline can be held to. "Planned but never happened" becomes a detectable failure.

**Why isolated world DBs per rollout?** Entity activations must not bleed between rollouts. Each starts from pristine seed state.

**Why per-beat writer loop?** Empirically: 0.762 vs 0.525 on the same plan. The model compresses in one-shot; per-beat forces commitment.

**Why score-gated refinement?** A refinement can improve subtext while regressing voice. The gate prevents quality trading. (Though the gate itself needs the scoring fixes described above to be trustworthy.)

---

## Next steps, in priority order

1. **Logprob-weighted scoring.** Integer 1–10, extract E[score] from logprobs, compute confidence. Zero extra forward passes. Makes the +0.05 threshold meaningful.

2. **Pairwise comparison.** Refinement gate, sibling selection, PL anchoring all become A/B picks with P(A wins) from logprobs. Drop absolute scoring where a comparison is the actual question.

3. **Test-retest variance.** Score the same chapter 5× with the logprob method. Report SD per dim. If SD > 0.05, the acceptance threshold is still in noise.

4. **Cross-judge validation.** Run the same comparisons on a second model (Qwen3, Llama 3.3). If judges disagree on >30% of pairs, the rubric measures model taste.

5. **Refinement distribution.** Run refinement on ≥30 chapters from the overnight rollout. Report acceptance rate, conditional effect size, untargeted-dim drift. Plot (ΔmeanScore, maxDimRegression) colored by selector type.

6. **PCA on dim matrix.** Continuous logprob scores make this clean. Collapse correlated dims.

7. **Drop update_self_containment.** It rewards recap in serial fiction.

8. **Rollout diversity check.** Entity-introduction Jaccard + hook-payoff overlap across profiles. One SQL query over the KB tables.

9. **Cross-chapter coherence dim.** Score on chapter pairs, not individuals.

10. **Prose-level hook verifier.** Does the prose actually contain what the trace claims was planted?

11. **Human calibration set.** n≥20 chapters rated by a human. Compute rank correlation with E[score].

12. **Only then: writer LoRA.** Without the above, fine-tuning Gemma on Gemma's judgments is a reward-hacking recipe.

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
