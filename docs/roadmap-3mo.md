# Quest Engine — 3-Month Roadmap (v3)

**Start date**: 2026-04-14
**End date**: 2026-07-14
**Goal**: A personal quest-playing tool that produces prose meaningfully better
than what you'd find on Royal Road. Not "AI-generated fiction that's surprisingly
okay" — fiction you'd actually want to read for its own sake.
**Quality bar**: Beat Sanderson on craft dimensions (voice, FIS, detail,
subtext). Match Abercrombie on engagement dimensions (tension, pacing, hooks).
Approach Pale Lights on voice distinctiveness.
**Velocity**: ~1 "week of normal side-project" per day of actual work. The core
pipeline buildout is ~2 weeks. The remaining 10 weeks are for prose quality
depth, not features.

---

## Phase 1: Close the Loop (Apr 14 – Apr 27)

Get from "pipeline that runs" to "pipeline that measurably improves itself."
No speculative work.

### Day 1: vllm + Retrieval Activation
- [x] Wave 4 retrievers (Motif, Foreshadowing, Voice) — merged 2026-04-14
- [x] Flip retrieval ON; re-run Apr 13 demo — `docs/retrieval-eval.md`
- [ ] vllm setup. Confirm ~3k tok/s on LFM2.5-1.2B.
- [ ] Formal measurement pass: cliché density, POV adherence, voice continuity,
      entity presence. Document baseline numerically.

### Day 2: Heuristic Scoring + Scorecard DB
- [ ] `Scorer` class — 12 heuristic dimensions, 0.0-1.0 continuous.
- [ ] Scorecard + dimension_scores tables.
- [ ] Post-commit hook — every run auto-scores.
- [ ] Score dashboard in web UI.

### Day 3: Generate-N Rerank
- [ ] Fan-out `_run_write`: N=10 candidates per scene.
- [ ] Rerank by weighted heuristic sum.
- [ ] Log all candidates + scores in trace.
- [ ] A/B: N=1 vs N=10 on demo scenario.

### Day 4: SFT Data Collection
- [ ] Auto-save CraftBrief + candidates + scores per run.
- [ ] Claude winner selection batch.
- [ ] Build train.jsonl.
- [ ] Play 15-20 updates across 2-3 seeds.

### Day 5: Writer LoRA v1
- [ ] Train LoRA on LFM2.5-1.2B. 3 epochs, rank 32.
- [ ] A/B: base vs LoRA. Heuristics + Claude pairwise.
- [ ] If wins: default via vllm hot-swap.

### Day 6: LLM Judge (3 dims) + Calibration
- [ ] Judge prompts with anchored scales: emotional_trajectory, tension_execution, choice_hook_quality.
- [ ] Calibrate against corpus (r > 0.7).
- [ ] Wire into async post-commit scoring.

### Day 7: Prompt Optimizer v1
- [ ] identify_weak_dimensions — scan scorecards.
- [ ] propose_mutation — Claude-assisted prompt edits.
- [ ] Replay-based A/B on 5 traces.
- [ ] ExampleCurator v1 — mine high/low scoring outputs.

### Day 8: Model Eval — Gemma 4 vs LoRA-LFM
- [ ] Gemma 4 26B MoE in vllm.
- [ ] Same 10 scenes, both models, full scoring.
- [ ] Gemma 4 as local LLM judge (if r > 0.7, drop API costs).
- [ ] Decision: which model for which role.

### Day 9: Player UX Polish
- [ ] Fix choices rendering. Descriptions + tags + write-in.
- [ ] Streaming writer output to UI.
- [ ] Scene context panel updates live.
- [ ] Trace viewer: per-layer collapsible, craft brief visible.

### Day 10: 50-Chapter Stress Test
- [ ] Voter-rollout personas auto-generate actions. 50 updates overnight.
- [ ] Degradation curves: scores, context tokens, latency, world state size,
      consistency flags, entity presence, motif recurrence, foreshadowing payoff.
- [ ] Identify primary bottleneck.

### Days 11-12: Fix What Broke
- [ ] Address primary bottleneck.
- [ ] Re-run 20-chapter verification.

### Days 13-14: Consolidate
- [ ] Full test suite green. All docs updated. README.
- [ ] Clear picture: per-dimension scores vs. calibration targets.

**Phase 1 exit**: Pipeline with retrieval + N=10 rerank + best available writer
model + auto-improvement loop + 50-chapter stress test passed.

---

## Phase 2: Prose Depth (Apr 28 – Jun 1, ~5 weeks)

Phase 1 gives you a system that works. Phase 2 makes the prose *good* — not
"good for AI" but good enough that you'd keep reading because you want to.

### Week 1 (Apr 28 – May 4): Voice + Free Indirect Style
- [ ] Full 12-dim LLM judge — extend calibration to all subjective dimensions.
- [ ] Blended voice sample authoring — for 3 primary narrator-character pairings.
- [ ] Permeability A/B testing — 0.2, 0.4, 0.6, 0.8 sweet spot.
- [ ] Character voice grounding — populate full schemas for 5-6 characters.
- [ ] Specialized voice critic — split out of monolithic CHECK.

**Target**: free_indirect_quality → 0.5+. voice_distinctiveness → 0.7+.

### Week 2 (May 5 – May 11): Detail + Sensory Texture
- [ ] Perceptual profiles for main characters.
- [ ] Metaphor profile population.
- [ ] Sensory palette tuning.
- [ ] Detail-selection craft examples (Flaubert, Woolf, Ishiguro).
- [ ] A/B: grounded details vs. generic.

**Target**: detail_characterization → 0.6+. controlled sensory_density.

### Week 3 (May 12 – May 18): Subtext + Indirection
- [ ] Unconscious motive population for 3-4 key NPCs.
- [ ] IndirectionInstruction grounding.
- [ ] Negative space examples (Hemingway, Ishiguro, Austen).
- [ ] Surface vs. depth tuning.
- [ ] Subtext scorer calibration.

**Target**: subtext_presence → 0.4+. indirection_leakage at 1.0.

### Week 4 (May 19 – May 25): Theme + Motif + Long-Arc Resonance
- [ ] Theme deepening — 2 themes as full propositions.
- [ ] Motif recurrence testing — 3 motifs across 15 chapters.
- [ ] Parallel delivery — plant 2 structural parallels.
- [ ] Foreshadowing payoff rate measurement.
- [ ] Information asymmetry exploitation.

**Target**: thematic_presence → 0.6+. motif_execution → 0.5+. ≥1 parallel delivered > 0.5.

### Week 5 (May 26 – Jun 1): Writer LoRA v2 + Quality Ceiling
- [ ] LoRA v2 training — full accumulated corpus (500+ pairs).
- [ ] Model tournament: LoRA-v1 vs v2 vs base-LFM vs Gemma-4.
- [ ] Prompt optimization cycle.
- [ ] Example curation cycle.
- [ ] Weight tuning for reranker.

**Phase 2 exit**: Per-dimension scores measured on 20-chapter run across 2 genres.

---

## Phase 3: Play It For Real (Jun 2 – Jul 14, ~6 weeks)

The system is as good as you can make it through engineering. Phase 3 is about
*using* it.

### Week 1-2 (Jun 2 – Jun 15): The Quest You Actually Want to Play
- [ ] Design a quest you're genuinely interested in.
- [ ] Play 30+ chapters. Not automated.
- [ ] Keep a reading journal.
- [ ] Fix things as you find them.

### Week 3-4 (Jun 16 – Jun 29): Chase the Remaining Gaps
Likely candidates:
- [ ] Temporal structure (G11) if flat.
- [ ] Adversarial tension planning (Xie & Riedl) if too predictable.
- [ ] Dedicated dialogue pass if NPC conversations are lifeless.
- [ ] Hierarchical summarization if long-arc coherence degrades.
- [ ] Adversarial planning + asymmetry exploitation if you're bored.

### Week 5-6 (Jun 30 – Jul 14): Polish + Second Quest + Verdict
- [ ] Second quest in different genre.
- [ ] 10-chapter run on quest 2.
- [ ] Final scoring report.
- [ ] Decide what's next: open-source, Chimera, research, or done.

---

## Milestones

| Phase | Duration | Goal | Exit Criterion |
|-------|----------|------|----------------|
| 1 | 2 weeks | Close the loop | Pipeline self-improves. 50-ch stress test passes. |
| 2 | 5 weeks | Prose depth | Beat Sanderson on craft dims. Match Abercrombie on engagement. |
| 3 | 6 weeks | Play it for real | 30+ chapters of a quest you enjoy. Genre generalization tested. |

## Success Criteria at 3 Months

1. You played a 30+ chapter quest and wanted to keep going.
2. A friend reads 3 chapters cold and can't immediately tell it's AI.
3. Scoring shows measurable improvement over time. Auto-improvement loop produced ≥3 accepted mutations.
4. Per-dim scores exceed Sanderson on voice/FIS/detail/subtext, match Abercrombie on tension/pacing/hooks.
5. Clear answer to "what next?" from playing, not speculating.
