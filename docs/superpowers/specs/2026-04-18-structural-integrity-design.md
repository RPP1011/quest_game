# Structural Integrity & Scoring Overhaul

**Date:** 2026-04-18
**Status:** Design approved, pending implementation plan

Seven systems that move the pipeline from "individual beats are decent" to "the story keeps its promises and we can prove it." Sequenced: structural integrity before quality polish, measurement before generation changes, cheap before expensive.

## Dependency Graph

```
1. Foreshadow Pool ──────────────┐
2. Typed Edit-Pass ──────────────┤
                                 ├─→ 5. TKG Conflict Detection
3. Cross-Judge (Prometheus) ─────┤
4. Non-LLM Rerankers ───────────┤
                                 ├─→ 6. Suspense Re-ranker
                                 │
7. KL Drift Monitor ────────────┘ (independent)
```

Items 1-4 are independent and can be built in parallel. Items 5-6 benefit from 1-4 being in place. Item 7 is standalone.

---

## 1. CFPG-Style Foreshadow Pool with Prose-Level Verification

### Problem

The pipeline trusts the planner's self-report on foreshadowing. `UnpaidHookSelector` checks whether the trace says a hook was planted — not whether the prose actually contains a legible reference. Foreshadowing planted in the trace but invisible in the prose slips through. Foreshadowing paid off in a way no reader would connect back to the setup also slips through.

### Design

#### Schema

New table `foreshadow_triples`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | `ft_<uuid8>` |
| `hook_id` | TEXT NOT NULL | FK to `foreshadowing.id` |
| `foreshadow_text` | TEXT NOT NULL | Prose-legible description of what was planted |
| `trigger_pred` | JSON NOT NULL | Structured predicate for when to fire payoff |
| `payoff_text` | TEXT NOT NULL | What should happen when trigger fires |
| `status` | TEXT DEFAULT 'planted' | `planted` / `triggered` / `paid_off` / `expired` |
| `planted_chapter` | INT NOT NULL | Chapter where planted |
| `deadline_chapter` | INT | Escalate to planner if unpaid past this |
| `verified_planted` | REAL | Logprob confidence that prose contains legible plant |
| `verified_payoff` | REAL | Logprob confidence that prose contains legible payoff |

#### Trigger Predicate Types

```json
{"type": "chapter_gte", "value": 5}
{"type": "entity_active", "entity_id": "char:cozme"}
{"type": "entity_present", "entity_id": "char:cozme"}
{"type": "event_occurred", "event": "tristan_confronts_cozme"}
{"type": "and", "children": [<pred>, <pred>]}
{"type": "or", "children": [<pred>, <pred>]}
```

Predicates are evaluated deterministically against current world state — no LLM call.

#### Lifecycle

**Plant:** Two sources create triples. (1) The arc skeleton planner's `hook_schedule` — each scheduled hook becomes a triple at skeleton generation time, with `deadline_chapter` set to `paid_off_by_chapter`. (2) The dramatic planner can plant ad-hoc hooks mid-rollout — these become triples with a default deadline of `planted_chapter + 5`. After the beat that should contain the plant, a verification LLM call runs:

```
Does the following prose contain a legible reference to this narrative element?
Element: "{foreshadow_text}"
Prose: "{last_3_beats}"
Answer YES or NO.
```

Logprob on YES token → `verified_planted`. If confidence < 0.6, the triple is flagged and the next beat gets an explicit plant instruction injected into the writer context.

**Trigger:** Before each beat, scan all `planted` triples. Evaluate trigger predicates against current state. When a trigger fires, mark `triggered` and inject the payoff description into the planner's context.

**Payoff:** After the beat that should contain the payoff, same verification call targeting `payoff_text`. If confidence < 0.6, re-inject for next beat. If verified, mark `paid_off`.

**Deadline:** Triples past `deadline_chapter` with status != `paid_off` get injected as mandatory items into the dramatic planner: "OVERDUE — must resolve this chapter."

#### Integration Points

- `app/rollout/harness.py` — trigger check before each beat, verification after each beat
- `app/planning/dramatic_planner.py` — receives triggered/overdue triples in context
- `app/world/state_manager.py` — new CRUD methods for `foreshadow_triples`
- New module: `app/planning/foreshadow_pool.py` — predicate evaluation, verification calls, lifecycle management

---

## 2. Typed Edit-Pass Replacing Open-Ended Revise Loops

### Problem

The reviser receives free-text issues and rewrites paragraphs broadly, re-introducing problems (especially gambling metaphors) as fast as it removes them. The open-ended "fix this" prompt gives the model another chance to reach for its defaults.

### Design

#### Two-Step Process

**Step 1: Detect** — LLM call with structured output. Identifies specific spans and classifies each by edit type:

```json
{
  "edits": [
    {
      "span_start": 1247,
      "span_end": 1302,
      "original_text": "The odds were shifting beneath him like a gambler's last coin",
      "edit_type": "forced_metaphor",
      "reason": "Fourth gambling metaphor in 200 words",
      "replacement": "The ground was tilting beneath him like a floor giving way"
    }
  ]
}
```

**Step 2: Apply** — Deterministic string replacement. Apply edits in reverse position order so offsets don't shift. No LLM call.

#### Edit Type Taxonomy

| Type | Description | Source |
|------|------------|--------|
| `cliche` | Dead metaphor or stock phrase | LAMP |
| `purple_prose` | Overwritten, too many adjectives | LAMP |
| `forced_metaphor` | Metaphor from over-budget imagery family | LAMP + ours |
| `unnecessary_exposition` | Tells what was already shown | LAMP |
| `abrupt_transition` | Scene/mood shift without bridging | LAMP |
| `bland_dialogue` | Generic speech, no character voice | LAMP |
| `weak_closure` | Beat/chapter ending fizzles | LAMP |
| `continuity_break` | Contradicts established fact | ConStory |
| `character_voice_drift` | Character sounds wrong | ConStory |
| `timeline_error` | Events in wrong temporal order | ConStory |
| `entity_contradiction` | Entity state contradicts prior | ConStory |
| `repetition` | Same phrase/idea repeated in short span | Ours |

#### Integration with Existing Check Loop

The existing `CheckIssue` flow stays for world-rule and plan-adherence issues (critical/error severity). The typed edit-pass runs after the check stage specifically for prose-quality issues:

1. Check stage → world-rule violations, continuity breaks (structured issues, may need paragraph-level rewrite)
2. Typed edit detection → prose-quality problems (span-level edits, deterministic application)
3. Critical world-rule fixes still go through the existing reviser
4. Prose-quality fixes apply as deterministic span replacements — no rewrite drift

#### Audit Log

New table `typed_edits`:

| Column | Type |
|--------|------|
| `id` | TEXT PK |
| `trace_id` | TEXT |
| `rollout_id` | TEXT |
| `chapter_index` | INT |
| `edit_type` | TEXT (enum) |
| `original_text` | TEXT |
| `replacement` | TEXT |
| `span_start` | INT |
| `span_end` | INT |
| `reason` | TEXT |

Query: `SELECT edit_type, COUNT(*) FROM typed_edits GROUP BY edit_type ORDER BY 2 DESC` — tells you which failure modes the pipeline generates most.

#### New Module

`app/engine/typed_edits.py` — detect function (LLM call), apply function (string ops), taxonomy constants.

Prompt template: `prompts/stages/typed_edit/system.j2` + `user.j2`.

---

## 3. Different-Family Judge + Swap-and-Average + Per-Dim Independent Scoring

### Problem

Single-model scoring is circular. Joint multi-dim scoring creates anchoring. Position bias in pairwise comparison peaks when candidates are close in quality.

### Design

#### 3a. Per-Dim Independent Scoring

Current `score_chapter()` rates all 4 dims in one prompt. Replace with 4 separate LLM calls, one per dim, each receiving only that dim's rubric:

```python
async def score_chapter_independent(client, prose, prior_context=None):
    tasks = [
        _score_single_dim(client, prose, dim, prior_context)
        for dim in COLLAPSED_DIMS
    ]
    return dict(zip(COLLAPSED_DIMS, await asyncio.gather(*tasks)))
```

Cost: 4 shorter calls instead of 1 long call. Net latency similar with parallel execution.

#### 3b. Swap-and-Average

`compare_chapters_corrected` already runs (A,B) and (B,A) and averages. Audit all comparison call sites in `refinement/framework.py` and `selectors.py` to ensure they use the corrected variant. No new code — just verification.

#### 3c. M-Prometheus-14B Cross-Judge

**Runtime model:** M-Prometheus-14B as GGUF, running on a second llama-server instance on CPU at `http://127.0.0.1:8083`. Runs in-loop — every committed chapter gets scored by both judges.

14B Q4 on CPU: ~5 tok/s. Scoring prompts are ~500 tokens. 4 dims × ~100 output tokens each = ~2s per dim × 4 = ~8s total per chapter. Invisible against the ~5 min chapter generation time.

**New module:** `app/scoring/cross_judge.py`

```python
@dataclass
class JudgePair:
    gemma_scores: dict[str, float]
    prometheus_scores: dict[str, float]
    agreement: float          # mean |gemma - prometheus| across dims
    self_preference_flag: bool  # True if disagreement > 1.5 pts on any dim
```

**Persistence:** New table `cross_judge_scores`:

| Column | Type |
|--------|------|
| `rollout_id` | TEXT |
| `chapter_index` | INT |
| `judge_model` | TEXT |
| `dim` | TEXT |
| `score` | REAL |
| `confidence` | REAL |

**Self-preference tripwire:** If `self_preference_flag` fires, the chapter gets a warning annotation in the trace. Not a block — a signal for later analysis.

---

## 4. Non-LLM Rerankers (Syntactic Template Detector + MTLD + MAUVE)

### Problem

All quality signals come from LLM calls. No independent structural measurement to detect syntactic templating, vocabulary collapse, or style drift from reference prose.

### Design

#### 4a. Syntactic Template Detector

POS-tag with spaCy (`en_core_web_sm`), gzip-compress the POS sequence. Compression ratio = `len(compressed) / len(raw)`.

```python
def syntactic_compression_ratio(prose: str) -> float:
    doc = nlp(prose)
    pos_seq = " ".join(tok.pos_ for tok in doc)
    raw = pos_seq.encode()
    return len(gzip.compress(raw)) / len(raw)
```

Threshold: ratio < 0.35 → flag as syntactically templated.

#### 4b. MTLD (Measure of Textual Lexical Diversity)

Length-robust type-token ratio (McCarthy & Jarvis 2010). ~40 lines of Python, no dependencies beyond tokenization.

Threshold: MTLD < 50 → flag as vocabulary collapse. Literary prose: 70-100+.

#### 4c. MAUVE

Distributional divergence via `mauve-text` package. Compares generated prose distribution against reference corpus (Pale Lights excerpts from `data/calibration/`).

Threshold: MAUVE < 0.7 → significant style drift. Computed per-rollout (needs ~20+ passages).

#### Usage Modes

| Mode | Granularity | Purpose |
|------|------------|---------|
| Per-beat rejection | Each beat | Hard floor: reject beats with CR < 0.30 or MTLD < 40 |
| Per-chapter dashboard | Each chapter | Track metrics over rollout |
| Per-rollout MAUVE | Full rollout | Distributional comparison to reference |

#### Persistence

New columns on `rollout_chapters`: `syntactic_cr`, `mtld`.
New column on `rollout_runs`: `rollout_mauve`.

#### Critical Constraint

These are guardrails (alarms), not optimization targets. Never expose these scores to the generation pipeline. Per Shaib et al.: optimizing for diversity metrics produces bad prose.

#### New Module

`app/scoring/structural_metrics.py` — all three metrics. Dependencies: `spacy`, `mauve-text`.

---

## 5. DOME-Style TKG (Temporal Knowledge Graph) Conflict Detection

### Problem

The check stage sees ~2 prior chapters of context. It cannot detect that chapter 14 contradicts chapter 7. KB extraction logs events but doesn't build a queryable graph of factual claims.

### Design

#### Schema

New table `knowledge_graph`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | `kg_<uuid8>` |
| `subject` | TEXT NOT NULL | Entity id or name |
| `predicate` | TEXT NOT NULL | Enum: `located_at`, `possesses`, `status`, `relationship`, `knows`, `believes` |
| `object` | TEXT NOT NULL | Value |
| `chapter` | INT NOT NULL | When established |
| `superseded` | INT | Chapter where overridden (NULL = current) |

#### Triple Extraction

Structured LLM call after the existing extract stage. Prompt: "List factual claims established or changed in this chapter as (subject, predicate, object) triples." Schema-constrained output with enum predicates.

#### Conflict Detection

Before generating chapter N+1, query the graph for entities in the upcoming scene's `characters_present`. For each entity, pull current (non-superseded) triples. Deterministic checks:

1. **Location conflict** — entity `located_at` X but scene places them at Y without traversal
2. **State conflict** — entity `status` is "dead" but in `characters_present`
3. **Possession conflict** — item possessed by A was already given to B
4. **Relationship conflict** — relationship changed without scene justification

Detected conflicts → injected into check stage as `critical` issues with the contradicting triple quoted.

#### Supersession

When a new triple contradicts an existing one (same subject + predicate), the old triple's `superseded` column gets set to the current chapter. Tracks state evolution without losing history.

#### New Module

`app/world/knowledge_graph.py` — extraction, query, conflict detection. Extraction prompt: `prompts/stages/kg_extract/system.j2` + `user.j2`.

---

## 6. Wilmot-Keller Suspense Re-ranker

### Problem

Default LLM output is low-variance and low-surprise. Per-beat writing produces N=1 candidate. No engagement-oriented signal for selecting among alternatives.

### Design

#### Mechanism

For high-stakes beats, generate N=4 candidates. For each, generate 3 short continuation sketches (~50 tokens). Measure entropy of continuations via logprobs. Higher entropy = more possible futures = more suspense.

```python
async def suspense_score(client, prior_prose, candidate_beat) -> float:
    context = prior_prose + candidate_beat
    continuations = await asyncio.gather(*[
        client.chat_with_logprobs(
            messages=[ChatMessage(role="user", content=f"Continue:\n{context[-1000:]}")],
            max_tokens=60, temperature=0.9,
        )
        for _ in range(3)
    ])
    mean_logprobs = [
        sum(t.logprob for t in c.token_logprobs) / max(len(c.token_logprobs), 1)
        for c in continuations
    ]
    return -sum(mean_logprobs) / len(mean_logprobs)
```

#### Activation Gate

Not every beat — only high-stakes ones:

- `tension_target >= 0.7` → enabled
- `dramatic_function` in `("climax", "reversal", "escalation")` → enabled
- Quiet/aftermath beats → skip, N=1

A 15-beat chapter with 3-4 high-stakes beats → 12-16 extra generation calls.

#### Integration with Non-LLM Rerankers

For N=4 candidate beats, selection combines:
1. Reject candidates below syntactic CR floor (< 0.30) or MTLD floor (< 40)
2. Rank remaining by suspense score

#### Persistence

`suspense_score` field in the write stage's `StageResult.parsed_output` metadata.

#### New Module

`app/engine/suspense.py` — score function, activation gate logic.

---

## 7. Rolling Function-Word KL Drift Monitor

### Problem

No voice drift detection across chapters. The voice tracker tracks metaphor families, not the underlying stylometric signal.

### Design

#### Mechanism

~80 closed-class function words (articles, prepositions, auxiliaries, pronouns). Compute frequency distribution per chapter. KL divergence from baseline.

```python
FUNCTION_WORDS = ["the", "a", "an", "of", "in", "to", "and", "but", ...]  # ~80 total

def function_word_distribution(text: str) -> np.ndarray:
    tokens = text.lower().split()
    counts = Counter(tokens)
    total = sum(counts[w] for w in FUNCTION_WORDS) or 1
    return np.array([counts[w] / total for w in FUNCTION_WORDS])

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon=1e-10) -> float:
    p, q = np.clip(p, epsilon, None), np.clip(q, epsilon, None)
    return float(np.sum(p * np.log(p / q)))
```

#### Baselines

Per-character baselines keyed by `pov_character_id`. KL computed against that character's first chapter, not the global ch1. Prevents false alarms on intended POV switches.

Also compute over a 3-chapter sliding window to detect sudden shifts vs gradual drift.

#### Thresholds

| KL | Interpretation | Action |
|----|---------------|--------|
| < 0.05 | Stable | None |
| 0.05 - 0.15 | Mild drift | Log |
| > 0.15 | Voice drift | Warning injected into planner context |

#### Persistence

New columns on `rollout_chapters`: `fw_kl_baseline`, `fw_kl_window`.

#### Constraint

Diagnostic only — annotates, does not block or revise. The planner gets a warning; the writer prompt doesn't change.

#### New Module

`app/scoring/voice_drift.py` — distribution, KL, baseline management. Dependencies: `numpy` (already present).

---

## New Dependencies

| Package | Used by | Notes |
|---------|---------|-------|
| `spacy` + `en_core_web_sm` | Section 4a (syntactic CR) | `pip install spacy && python -m spacy download en_core_web_sm` |
| `mauve-text` | Section 4c (MAUVE) | `pip install mauve-text` |
| M-Prometheus-14B GGUF | Section 3c | Second llama-server on CPU, port 8083 |

`numpy` and `scipy` already in the project.

## New Tables Summary

| Table | Section | Purpose |
|-------|---------|---------|
| `foreshadow_triples` | 1 | Triple pool with verification scores |
| `typed_edits` | 2 | Audit log of categorized span edits |
| `cross_judge_scores` | 3 | Per-dim scores from both judge models |
| `knowledge_graph` | 5 | Temporal fact triples with supersession |

## New Columns Summary

| Table | Column(s) | Section |
|-------|-----------|---------|
| `rollout_chapters` | `syntactic_cr`, `mtld`, `fw_kl_baseline`, `fw_kl_window` | 4, 7 |
| `rollout_runs` | `rollout_mauve` | 4 |

## New Modules Summary

| Module | Section | Description |
|--------|---------|-------------|
| `app/planning/foreshadow_pool.py` | 1 | Predicate eval, verification, lifecycle |
| `app/engine/typed_edits.py` | 2 | Detect + apply span-level edits |
| `app/scoring/cross_judge.py` | 3 | Dual-model scoring + self-preference detection |
| `app/scoring/structural_metrics.py` | 4 | Syntactic CR, MTLD, MAUVE |
| `app/world/knowledge_graph.py` | 5 | Triple extraction, conflict detection |
| `app/engine/suspense.py` | 6 | Forward-uncertainty suspense scoring |
| `app/scoring/voice_drift.py` | 7 | Function-word KL drift monitor |

## New Prompt Templates Summary

| Template | Section |
|----------|---------|
| `prompts/critics/foreshadow_verify.j2` | 1 |
| `prompts/stages/typed_edit/system.j2` + `user.j2` | 2 |
| `prompts/stages/kg_extract/system.j2` + `user.j2` | 5 |
