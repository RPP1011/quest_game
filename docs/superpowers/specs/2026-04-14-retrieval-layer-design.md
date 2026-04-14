# Retrieval Layer — Unified Design

**Scope:** Introduce retrieval as a first-class substrate across the quest
generation pipeline. Replaces static prompt content with
context-adaptive grounding sourced from a labeled literary corpus and from
each quest's own history.

**Target completion:** overnight (single multi-phase run, parallel
subagents where the interface is stable).

**Non-goals:**
- Cross-quest retrieval (treated as future work; this design is per-quest
  for the quest corpus).
- Replacing the craft library (which already stores per-tool exemplars;
  retrieval augments, not replaces).
- External vector databases (Chroma/Qdrant). SQLite + numpy suffices at
  our scale (<10k records per quest, ~260 reference passages).

---

## 1. Architecture overview

Single new package: `app/retrieval/`.

```
app/retrieval/
    __init__.py
    interface.py          # Query, Result dataclasses; Retriever Protocol
    passage_retriever.py  # Literary-corpus retriever (metadata + embeddings)
    quest_retriever.py    # This-quest retriever (per-quest SQLite embeddings)
    craft_retriever.py    # Craft-tool exemplar retriever (wraps CraftLibrary)
    embeddings.py         # Sentence-transformers embedder, cacheable
    motif_retriever.py    # Scheduled motif recurrence (queries existing motif_occurrences table)
    foreshadowing_retriever.py  # Ripe hooks eligible for payoff
    voice_retriever.py    # Per-character utterance retrieval (phase 4; may slip)
```

Unified consumer interface:

```python
class Retriever(Protocol):
    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]: ...
```

Where `Query` carries the current pipeline context (POV, narrator, target
emotion, dim ranges, semantic seed text, entity filters, etc.) and
`Result` wraps a passage + source metadata + score breakdown.

---

## 2. Storage

### 2.1 Literary corpus
- Source of truth: `data/calibration/manifest.yaml` (per-work expected
  scores), `/tmp/labels_claude_all.json` merged (per-passage ground-truth
  labels), `data/calibration/passages/` (raw text, gitignored).
- Embedding cache: `data/calibration/embeddings.npy` — shape `(N_passages,
  384)` for MiniLM. Rebuilt when passages change; sha-keyed to manifest.
- Sibling `data/calibration/embedding_ids.json` — parallel array of
  `"<work_id>/<passage_id>"` keys identifying each row.

### 2.2 Quest corpus
- New SQLite table alongside `narrative_records`:
  ```sql
  CREATE TABLE narrative_embeddings (
      update_number INTEGER NOT NULL,
      scene_index INTEGER NOT NULL,
      quest_id TEXT NOT NULL,
      embedding BLOB NOT NULL,           -- 384-dim float32 bytes (1.5KB)
      text_preview TEXT NOT NULL,        -- first 200 chars for logging
      PRIMARY KEY (quest_id, update_number, scene_index)
  );
  CREATE INDEX idx_ne_quest ON narrative_embeddings(quest_id);
  ```
- Written post-commit by the extract stage. No deletion on retcon (rows
  superseded by newer update_number).

---

## 3. Retrievers

### 3.1 `PassageRetriever`

Literary-corpus retriever. Supports:
- `Query.filters.pov: Literal["second_person","third_limited","third_omniscient"] | None`
- `Query.filters.is_quest: bool | None`
- `Query.filters.score_ranges: dict[str, tuple[float, float]]` —
  e.g. `{"voice_distinctiveness": (0.7, 1.0), "subtext_presence": (0.4, 1.0)}`
- `Query.filters.exclude_works: set[str]` — don't pull from Joyce for a
  pulp adventure scene
- `Query.seed_text: str | None` — if provided, embed and cosine-rerank
  after metadata filter

Two modes controlled by `enable_semantic: bool` at construction:
- **Metadata-only** (phase 1): returns top-k by filter match score,
  ties broken by metadata proximity (e.g. closest score midpoint).
- **Hybrid** (phase 2+): metadata filter → semantic rerank.

### 3.2 `QuestRetriever`

Per-quest retriever reading the new `narrative_embeddings` table.
- `Query.seed_text: str` — required (callbacks are always semantic).
- `Query.filters.entity_mentions: set[str]` — boost records mentioning
  these entities.
- `Query.filters.max_updates_ago: int | None` — skip ancient records.
- `Query.filters.last_n_records: int | None` — baseline behavior of
  "last N records" falls out as `seed_text=""` + this filter.

### 3.3 `CraftRetriever`

Wraps the existing `CraftLibrary.examples_for_tool`, adds filter dimensions:
- `Query.filters.tool_id: str` — required
- `Query.filters.pov: str | None`
- `Query.filters.scale: str | None`
- `Query.filters.register: str | None`
- No embeddings (examples are short and already categorized).

### 3.4 `MotifRetriever`

Reads `motif_occurrences` table (already populated post-commit by pipeline).
- Returns motifs with `(current_update - last_occurrence_update) >
  target_interval_min` (due) or `> target_interval_max` (overdue).
- Paired with each motif's last 1-2 occurrence contexts for the planner
  to see how it was previously worn.

### 3.5 `ForeshadowingRetriever`

Reads `foreshadowing` table.
- Returns hooks where `planted_at_update <= current_update AND status ==
  PLANTED AND (target_update is None OR target_update >= current_update)`.
- Sorted by "ripeness" — older hooks ranked higher.

### 3.6 `VoiceRetriever` *(phase 4, may slip)*

Per-character past-utterance retrieval. Dependency: we need per-utterance
speaker attribution in narrative records, which we currently don't have.
Two paths:
- **Light**: use scene `pov_character_id` + dialogue-quote heuristic to
  collect quoted lines in scenes where that character is POV. Cheap, noisy.
- **Heavy**: LLM post-pass per commit to attribute utterances to speakers.
  Expensive, clean.

Phase 4 ships the light variant; heavy can come later if signal warrants.

---

## 4. Embedding strategy

- Model: `sentence-transformers/all-MiniLM-L6-v2` (22MB, 384-dim, CPU OK).
  Chosen for fit + speed; swap later if quality is a bottleneck.
- Passage representation: full passage text, truncated to 512 tokens.
  For 500-1000 word passages this truncation loses ~20-40% of tokens but
  retains opening character; acceptable for a first pass.
- Query representation: concatenation of `(scene function, target emotion,
  POV, seed_text)` truncated to 128 tokens.

---

## 5. Pipeline integration

### 5.1 Writer (`_run_write`)

**Input retrieval:** given the current `CraftScenePlan` and `EmotionalScenePlan`,
build a `Query`:
- `filters.pov` = scene POV (default 2nd-person for quests)
- `filters.score_ranges["voice_distinctiveness"] = (0.7, 1.0)` — we want
  stylistically strong anchors
- `filters.score_ranges["free_indirect_quality"]` = range matching the
  scene's permeability target
- `seed_text` = scene prose brief + emotional surface/depth

Retrieve k=3 voice anchors from `PassageRetriever`, k=2 callbacks from
`QuestRetriever` (seed_text same; filtered by entity mentions in scene).

Inject as few-shot block in `prompts/stages/write/user.j2` above the prose
brief, explicitly labeled "VOICE ANCHORS — style to match" and "IN-QUEST
CALLBACKS — continuity to honor".

### 5.2 Craft planner (`CraftPlanner.plan`)

Enrich tool example context: for each tool used by the dramatic plan,
retrieve filtered craft examples matching POV + scale + register.
Replaces (or augments) the fixed-per-tool examples currently loaded.

### 5.3 Dramatic planner (`DramaticPlanner.plan`)

Add scene-shape retrieval: given target dramatic function (escalation,
reversal, aftermath, quiet), retrieve 1-2 arc-scale scenes from the
literary corpus where that function was executed well (scene_coherence >
0.75, appropriate tension_execution). Inject as references.

### 5.4 Extract stage (post-commit)

After `self._world.apply_delta(delta)` commits a new narrative record:
1. Compute embedding of the committed prose.
2. Write row into `narrative_embeddings`.
3. Existing motif-occurrence persistence continues as-is.

### 5.5 Narrator bias (Phase 4 hook)

At write-time, optionally retrieve recent narration from the same quest
matching narrator-register cues to anchor voice drift across chapters.

---

## 6. Testing strategy

- **Unit tests per retriever**: fixture corpus, assert filter correctness
  and ranking order.
- **Integration tests**:
  - Writer retrieves and injects into prompt (assert passage ids in
    rendered prompt).
  - Extract stage writes embedding row.
  - Quest retriever reads back written rows.
- **Regression**: retriever disabled → identical behavior to current
  pipeline (via feature flag in `quest_config`).
- **Eyeball A/B**: 5 scenes, current baseline vs retrieval-augmented.
  Metrics tracked: POV adherence (heuristic + LLM judge), cliché count
  (lexicon-based), voice distinctiveness (pairwise judge if usable).

---

## 7. Phasing and parallelization

Overnight target breaks into 4 waves. Each wave hits a clean interface
boundary so subagents can operate in isolated worktrees without stepping
on each other.

### Wave 1 — Foundations (parallel, ~2h)
1a. `app/retrieval/interface.py` + `embeddings.py` + tests
    — interfaces + embedding cache only; no consumers yet
1b. `app/retrieval/passage_retriever.py` metadata-only
1c. SQLite schema migration + `narrative_embeddings` writer hook in
    extract stage (not called yet, just infrastructure)

### Wave 2 — Writer integration + semantic (parallel, ~2h)
2a. Semantic mode added to `PassageRetriever` (reads cache from 1a)
2b. Writer integration: prompt template update + `_run_write` passes
    retrieved anchors
2c. `CraftRetriever` (wraps existing CraftLibrary)

### Wave 3 — Quest retrieval + dramatic integration (parallel, ~2h)
3a. `QuestRetriever` reads `narrative_embeddings`
3b. Extract-stage hook activated: embeddings written post-commit
3c. Dramatic planner integration: scene-shape retrieval from arc corpus

### Wave 4 — Long-horizon retrievers (parallel, ~2h)
4a. `MotifRetriever` + craft-planner integration
4b. `ForeshadowingRetriever` + arc/dramatic planner integration
4c. `VoiceRetriever` (light variant — POV-scene dialogue heuristic)

### Integration + eval (~1h)
- Final wiring: all retrievers registered in pipeline
- A/B run: 5 scenes, before/after, human read-through
- Commit + push

---

## 8. Dependencies

New dependencies to add (all small):
- `sentence-transformers>=2.7` (pulls torch, which we already have)
- `numpy` (already installed via torch dependency chain)

No external services. No vector DB. Everything local.

---

## 9. Success criteria

- All retrievers ship with tests; full suite stays green.
- Writer retrieval demonstrably changes output: at least 2 of 3 scenes
  show measurable POV-adherence or voice-distinctiveness improvement.
- Quest retrieval recovers at least one in-quest callback per 3-action
  run (concrete entity mention or motif recurrence the prose picks up).
- Feature flag in `quest_config` (`retrieval.enabled: true/false`)
  defaults to `true`; setting to `false` reverts to current behavior
  bit-for-bit.
- Documentation in `docs/retrieval.md` explaining filters, queries, and
  integration points.

---

## 10. Risk register

- **Embedding quality**: MiniLM may not capture literary subtleties
  (voice, tone). Mitigation: phase 1 metadata-only still ships value;
  phase 2 evaluation can justify a swap to BGE-small if needed.
- **Prompt bloat**: injecting 3+2 retrieval results adds ~1500 tokens.
  LFM's 32k context absorbs it fine; watch for slower generation at
  scale. Can cap retrieved-passage length to ~300 words each.
- **Retcon interactions**: quest embeddings for rolled-back narrative
  should be invalidated. Phase 3 includes a cleanup path; if it slips,
  stale embeddings are fine for first ship (they just retrieve less
  relevantly).
- **Parallelization collisions**: Wave 2 and Wave 3 both modify
  pipeline.py. Isolate via clear interface — `_run_write` is 2a's lane,
  `_run_extract` is 3b's lane, `_run_hierarchical` gets a
  `retrievers` kwarg touched by all.
