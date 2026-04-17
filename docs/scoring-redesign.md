---
title: "Scoring Redesign: Logprob-Weighted Absolute + Pairwise Comparison"
---

# Scoring Redesign: Logprob-Weighted Absolute + Pairwise Comparison

**Date:** 2026-04-17
**Motivation:** The current scoring layer has three compounding problems that undermine every empirical claim downstream.

## Problems

**1. Quantization.** The judge emits float tokens ("0.7") which are multi-token sequences with ambiguous logprob attribution. In practice the judge quantizes to {0.3, 0.5, 0.6, 0.7, 0.8, 0.9}. The +0.05 acceptance threshold for refinement is below the instrument's resolution.

**2. Self-judging.** Gemma 4 26B judges Gemma 4 26B output. "Parity with Pale Lights on 7/8 dims" is a statement about the judge's discrimination, not about quality. Cross-model and human calibration are needed but are separate work items.

**3. Wrong tool for comparisons.** Refinement gate, sibling selection, and PL anchoring are all comparison tasks forced through absolute scoring. Absolute scores are noisy, quantized, and model-taste-dependent. Pairwise comparison ("which is better?") eliminates quantization entirely — the judge just picks — and is more reliable for LLM judges (the literature is clear on this).

## Design

Two scoring modes, each used where it fits:

### Mode 1: Logprob-weighted absolute scoring

**Use for:** weak-chapter selection (need a number per chapter to rank within a rollout), KB aggregation (per-dim means across chapters), trend analysis.

**Changes from current:**
- Score type: `integer 1–10` (single token) instead of `float 0.0–1.0` (multi-token)
- Extract top-K logprobs at the score token position via llama-server's `logprobs` + `top_logprobs` params
- Softmax-normalize over digit tokens 1–10
- E[score] = Σ(digit × prob) / 10 → continuous [0,1]
- Confidence = 1 - entropy(distribution) → [0,1], where spiky=confident, flat=uncertain
- Store both E[score] and confidence per dim per chapter

**Implementation:**
- `InferenceClient.chat_with_logprobs()` — new method returning `(text, logprobs_per_token)`
- `app/rollout/scorer.py` — refactor `score_chapter()` to use integer schema + logprob extraction
- `kb_chapter_scores` table gets a `confidence` column
- Weak-chapter selection: rank chapters within a rollout by mean E[score], select bottom quartile (not a fixed threshold)

### Mode 2: Pairwise comparison

**Use for:** refinement gate (refined vs baseline), sibling selection (rollout A ch5 vs rollout B ch5), PL anchoring (our chapter vs a PL excerpt on the same beat).

**Prompt shape:**
```
You are judging two chapter excerpts on [dim]. 
Read both carefully. Which is stronger on [dim]?

CHAPTER A:
<<<
{text_a}
>>>

CHAPTER B:
<<<
{text_b}
>>>

Which chapter is stronger on [dim]? Answer "A" or "B".
```

Single token response. Logprobs at the A/B position give P(A wins) continuously — not a binary pick but a probability. P(A) = 0.73 is a different signal than P(A) = 0.51.

**Where each comparison task uses it:**

| Task | Current approach | New approach |
|---|---|---|
| **Refinement gate** | Score refined prose absolutely, compare mean to baseline absolutely, apply +0.05 threshold | Pairwise: refined vs baseline on each dim. Accept if P(refined wins) > 0.6 on majority of dims AND P(refined wins) > 0.4 on every dim (no dim where baseline is clearly better) |
| **Sibling selection** | Absolute score both, check if delta ≥ 0.15 on any dim | Pairwise: chapter A vs chapter B on each dim. Flag if P(sibling wins) > 0.7 on any dim |
| **PL anchoring** | Score both with same judge, compare numbers | Pairwise: our chapter vs a PL excerpt on the same beat. P(ours wins) per dim gives a calibrated quality signal against an external reference |
| **Weak-chapter selection** | Absolute score below threshold | Rank within rollout by E[score], take bottom quartile. No threshold — the question is "which chapters are weakest relative to this rollout's own quality" not "which are below 0.55" |

**Implementation:**
- `app/rollout/pairwise.py` — `compare_chapters(client, text_a, text_b, dim) → (p_a_wins, logprob_detail)`
- Update `app/refinement/framework.py` — `refine_one()` uses pairwise instead of absolute re-scoring
- Update `app/refinement/selectors.py` — `SiblingOutscoredSelector` uses pairwise
- New: `app/rollout/anchor.py` — compares generated chapters against PL excerpts (requires PL chapter text access)

### What stays absolute

- KB aggregation (per-dim means across chapters need a number, not a comparison)
- Weak-chapter ranking within a rollout (need ordinal ranking, computed from E[score])
- Trend analysis over time (is the pipeline getting better across sessions?)

### What becomes pairwise

- Refinement accept/reject gate
- Sibling-outscored selector
- PL quality anchoring
- Any future A/B comparison (strategy sweep, model comparison, LoRA evaluation)

## Confidence as a first-class signal

Both modes produce confidence:
- Absolute: entropy of the 1–10 distribution. Spiky at 7 = confident score. Spread 4–8 = uncertain.
- Pairwise: distance of P(A wins) from 0.5. P(A)=0.95 = clear winner. P(A)=0.52 = toss-up.

Confidence flows into decisions:
- Don't refine chapters where the baseline score has low confidence (you can't measure improvement against noise)
- Don't accept a refinement where P(refined wins) is near 0.5 on the targeted dim (the improvement is within judge noise)
- Weight KB aggregation by confidence: a 7 with confidence 0.9 should count more than a 7 with confidence 0.3

## Validation protocol (before trusting any of this)

1. **Test-retest:** Score the same chapter 5× with the logprob method. Report mean and SD of E[score] per dim. If SD > 0.05, the +0.05 threshold is still in the noise.
2. **Cross-judge:** Run the same pairwise comparisons on a second model (Qwen3, Llama 3.3). If the two judges disagree on >30% of pairs, the rubric is measuring model taste.
3. **Human calibration:** n=20 chapters rated by a human on 4 simplified dims. Compute rank correlation with E[score]. If ρ < 0.5, the judge isn't measuring what we care about.
4. **PCA on logprob scores:** Now that scores are continuous, PCA will give clean loadings. Collapse correlated dims.
5. **Drop update_self_containment:** It rewards recap and exposition in serial fiction. Remove from both absolute and pairwise dims.

## Priority order

1. Add logprobs to InferenceClient (small, enables everything else)
2. Refactor absolute scorer to integer 1–10 + logprob E[score]
3. Test-retest validation on overnight rollout chapters
4. Build pairwise comparator
5. Rewire refinement gate to pairwise
6. Rewire sibling selector to pairwise
7. PL anchoring (requires PL chapter text access — already in data/calibration/raw/)
8. Cross-judge validation
9. Human calibration
10. PCA + dim collapse
