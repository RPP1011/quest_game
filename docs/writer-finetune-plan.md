# Writer Fine-tune Plan (LFM2.5-1.2B)

## Goal

Fine-tune LFM2.5-1.2B as the prose-writer in G2's generate-N pipeline. The
model already runs at ~670 tok/s (llama.cpp GPU) / ~3k tok/s (vllm target),
so throwing a lot of candidates at rerank is cheap. Tuning the writer on
quest-quality prose should improve the upper envelope of N candidates.

## Training signal options

### Option A — Best-of-N SFT (recommended)

Generate N=5 candidates per scene using the current pipeline. Claude picks
the best one. Train LFM to produce the winner given (craft plan, dramatic
plan, emotional plan, narrator, POV character voice).

- **Pros**: direct distillation of "what Claude thinks is good prose for
  this scene". Uses existing planning stack as-is. Signal matches the exact
  inference input the trained model will see.
- **Cons**: labeling is slow (Claude reads N candidates per scene). For
  200 scenes × N=5 = 1000 candidate reads. Manageable via parallel subagents.
- **Data scale**: 200 scenes. 200 winner-prose examples is small but LoRA
  on 1.2B can absorb it.

### Option B — Published-prose distillation

Take the 195 labeled passages we already have, and reverse-engineer
synthetic scene plans from them (Claude reads prose, emits what the plan
would have been). Train LFM to reproduce the prose given the reconstructed
plan.

- **Pros**: uses existing labeled data. 195 pairs immediately.
- **Cons**: synthetic plans won't match production plan structure
  (different schema, different level of detail). Prose from Joyce/Woolf is
  very different from quest-fiction defaults. Distillation may teach voice
  but not quest-appropriate behavior.

### Option C — Pairwise preferences over writer output

Generate N=2 candidates per scene. Claude picks A or B. Train with DPO
(Direct Preference Optimization). No need for absolute "is this good"
judgments — just relative preferences.

- **Pros**: mirrors the pairwise scorer work. DPO known to work well on
  small-model preference learning. Each "row" is 2 candidates so easier to
  collect than best-of-N.
- **Cons**: DPO is more training infra to add; peft+trl supports it.
  Marginal gains may be smaller than full SFT.

## Recommendation

**Start with A, add C later.** Concrete steps:

1. **Scene corpus**: take the 65 scenes we already sampled (or a subset)
   and extract (scene prose, craft plan equivalent). Use Claude to emit a
   synthetic plan for each — the plan is the input shape, prose is target.
   This is ~400 LoC of scripting.

2. **Candidate generation**: run the existing pipeline (hierarchical) on
   each synthetic plan with N=5 candidates. Takes ~5s × 5 candidates ×
   200 plans = 5000s = ~90 min on GPU (llama.cpp).

3. **Claude ranking**: dispatch parallel subagents to pick the best
   candidate per scene. ~200 picks × 1 second each = fast.

4. **Training data**: `(synthetic_plan + scene_prefix) → winner_prose`.
   Completion-only loss (same as scorer training). Target length ~1200
   tokens — longer than scorer but within LFM's 32k context.

5. **Eval**: hold out 20 scenes. Generate with trained model. Claude
   picks which of {baseline, finetuned} produces better prose per scene.
   Reports pairwise win rate + per-dim pairwise-scorer ratings.

## Open questions

1. What's the "plan" schema? Does the existing `CraftPlan` + `DramaticPlan`
   suffice, or do we need a stripped-down training-only format?
2. Are we fine-tuning via SFT only, or chaining SFT + DPO?
3. Do we keep the finetuned scorer and writer as separate LoRAs (so we
   can swap independently) or merge into one base?
4. How to handle the output length distribution? The scorer outputs short
   JSON; the writer outputs 1000-2000 word scenes. Same base model, very
   different generation profiles.

## Integration with G2

- Scorer LoRA: adapter_pairwise.safetensors → served via vllm, used in
  `_score_candidate` for subjective dims.
- Writer LoRA: adapter_writer.safetensors → served via vllm (or alternate
  port), used in `_generate_scene_candidates`.
- Both adapters load on top of the same LFM2.5-1.2B base; vllm can host
  multiple adapters simultaneously.
