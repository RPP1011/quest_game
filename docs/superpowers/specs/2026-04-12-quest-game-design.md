# Quest Game — Design Spec

**Date:** 2026-04-12
**Status:** Draft (infra-focused; agent prompt design deferred)

## Summary

A local web app for playing AI-generated forum-style quests (in the spirit of Sufficient Velocity quests). A user-managed library of quests, each driven by a configurable multi-agent pipeline running against a local `llama-server` instance sourced from the user's Hugging Face cache. This spec focuses on **infrastructure**: runtime, storage, orchestration plumbing, API surface, and UI scaffolding. Specific agent prompts, model tuning, and narrative-quality work are explicitly out of scope for v1.

## Goals

- Run entirely locally; single-user, browser-based UI served from `localhost`
- Manage a local `llama-server` subprocess using GGUF models discovered in the HF cache
- Support a library of quests with resume, branch/fork, and export/import
- Provide a pluggable agent pipeline (parallel where possible) with a default set of roles
- Surface a tuning panel exposing per-chapter, per-agent, and A/B feedback
- Let users configure theme, tone, POV (2nd-person CYOA vs 3rd-person narrative), chapter length, choices-per-chapter, write-in, and vote-flavor — via quick prompt, advanced form, or preset templates

## Non-Goals (v1)

- Multi-user / hosted deployment, auth, sharing servers
- Fine-tuning / training / RLHF pipelines (tuning mode only shapes prompts + style guides)
- Image / audio generation
- Mobile UI
- Specific prompt engineering for agents beyond stubs sufficient to prove the plumbing end-to-end

## Stack

- **Backend:** Python 3.11, FastAPI, `uvicorn`, `asyncio`, `httpx` (OpenAI-compat client), `pydantic` v2
- **Storage:** SQLite via `sqlalchemy` + `alembic` migrations; quest content as JSON blobs inside rows
- **Frontend:** Svelte + Vite (small bundle, good DX, minimal ceremony), TypeScript
- **Model runtime:** user-installed `llama-server` binary (from `llama.cpp`); app launches as subprocess
- **Packaging:** `uv` for Python env; single `make dev` starts backend + frontend

## Architecture

Three layers with clear boundaries; each independently testable.

### 1. Model Runtime Layer (`app/runtime/`)

Owns lifecycle of external inference processes. Swappable for future llama-swap/vllm backends via a common interface.

- `ModelCatalog` — scans `~/.cache/huggingface/hub` recursively for `*.gguf` files; extracts metadata (size, quant label from filename, repo-id from path). Cache result; manual refresh endpoint.
- `RuntimeBackend` (protocol) — `start(model_path, params)`, `stop()`, `health()`, `base_url`
- `LlamaServerBackend` — default implementation; launches `llama-server --model … --ctx-size … --port …`, streams logs, waits for readiness via `/health`, handles crash/restart
- `InferenceClient` — thin OpenAI-compatible async client (chat/completions, streaming), retry + timeout policy
- Only one model loaded at a time in v1; model swap = stop + start

### 2. Quest Engine (`app/engine/`)

The orchestration plumbing. Agent-agnostic: it knows about pipelines, steps, context objects, and persistence — not specific prompts.

- **Data model** (see below): Quest, Chapter, Choice, WorldState, StyleGuide, Feedback, Branch
- `QuestStore` — SQLite-backed repository; handles fork/branch (copy-on-write chapter lineage)
- `ContextAssembler` — builds the per-agent prompt context:
  - world-state snapshot, rolling summary of older chapters, verbatim recent N chapters, style guide, quest config, last choice
  - budgeting: fits within configured context window (default 32k); summarization delegated to a **Summarizer agent step** in the pipeline
- `AgentStep` (protocol) — `name`, `run(ctx) -> StepOutput`; pure function of context + LLM client
- `Pipeline` — DAG of AgentSteps with declared dependencies; executes in topological order, runs independent steps in parallel via `asyncio.gather`
- **Default pipeline (stubs only in v1):** ArcPlanner → (ChapterPlanner ∥ Lorekeeper) → Narrator → ConsistencyCritic → ChoiceDesigner → [VoteSimulator]
  - ArcPlanner runs conditionally (first chapter + every N chapters or on large arc drift)
  - ConsistencyCritic can request one Narrator retry with critique notes
- Pipelines are declared in Python; hot-swappable by config. v1 ships the default; tuning panel can toggle which steps run.
- `QuestRunner` — state machine: `setup → chapter_loop → awaiting_choice → ending`; persists progress after each step so crashes resume cleanly
- Feedback is stored alongside the chapter and merged into the style guide / per-agent overrides on next run

### 3. Web UI (`web/`)

- **Library view** — grid of quests, status, last played, branch tree visualizer (WhatIF-inspired, simple SVG tree)
- **Quest creation wizard** — three entry points, all converge on the same QuestConfig:
  - Quick: freeform prompt + model picker
  - Advanced: full form (theme, tone, POV, chapter length tokens, choices-per-chapter, write-in on/off, vote-flavor on/off, context window)
  - Templates: presets (pre-fill advanced form, fully editable)
- **Chapter reader** — streaming prose as it generates; choice list with write-in textbox; vote-flavor rendered as side panel when enabled
- **Tuning panel** (slide-over on any chapter) — four tabs:
  - Per-chapter: thumbs + note → appends to quest style guide
  - Per-agent: shows each agent's output, accept/critique per agent → appends to that agent's overrides
  - A/B: triggers regeneration with 2 variants; user picks → preference stored
  - Style guide: direct edit of accumulated guidance
- **Settings** — model picker, runtime params (ctx size, threads, GPU layers), HF cache path, data dir

### API Surface (selected)

```
GET  /api/models                         list cached GGUF models
POST /api/runtime/start                  {model_path, params}
POST /api/runtime/stop
GET  /api/runtime/status

GET  /api/quests                         library
POST /api/quests                         {config}
GET  /api/quests/{id}
POST /api/quests/{id}/fork               {from_chapter_id}
POST /api/quests/{id}/export             → JSON
POST /api/quests/import                  ← JSON

POST /api/quests/{id}/advance            {choice_id | write_in}  (SSE stream)
POST /api/quests/{id}/chapters/{cid}/regenerate   {variant_count}
POST /api/quests/{id}/chapters/{cid}/feedback     {kind, payload}
GET  /api/quests/{id}/style_guide
PUT  /api/quests/{id}/style_guide
```

## Data Model (SQLite)

- `quests` — id, name, config (JSON), created_at, parent_quest_id (for forks), root_quest_id
- `chapters` — id, quest_id, ordinal, parent_chapter_id, prose, choices (JSON), chosen_choice_id, world_state_snapshot (JSON), summary, agent_outputs (JSON: raw step outputs for the tuning panel), created_at
- `style_guides` — id, quest_id, global_text, per_agent_overrides (JSON)
- `feedback` — id, chapter_id, kind (chapter|agent|ab), payload (JSON), created_at
- `ab_variants` — id, chapter_id, variant_index, prose, agent_outputs, selected (bool)

Forks share ancestry via `parent_chapter_id` and `parent_quest_id`; no content duplication until divergence.

## Error Handling

- Runtime boundary (llama-server crashes, OOM, timeout) → surface to UI with actionable message; auto-restart capped at 3 attempts
- Agent step failures → step-level retry (N=2) with backoff; on final failure, record error in `agent_outputs` and let downstream steps decide (ConsistencyCritic treats missing upstream as skip; Narrator failure aborts chapter with a "retry chapter" UI action)
- Validation at system boundary only (API request schemas, HF cache paths); internal calls trust types

## Testing Strategy

- **Runtime layer:** integration test against a tiny real GGUF (checked into `test/fixtures` or downloaded on CI) — verifies subprocess lifecycle and OpenAI-compat contract. Unit tests with a fake binary for edge cases.
- **Engine:** in-memory SQLite + a `FakeInferenceClient` that returns canned responses keyed by step name. Covers pipeline DAG execution, parallelism, retry-on-critic, fork semantics, context budgeting.
- **API:** FastAPI `TestClient` with fake engine.
- **UI:** component tests with Vitest; one end-to-end smoke with Playwright against fake backend.

## Directory Layout

```
quest_game/
├── app/
│   ├── runtime/           # model catalog, backends, inference client
│   ├── engine/            # pipeline, steps, store, context assembler
│   ├── api/               # FastAPI routers
│   └── main.py
├── web/                   # Svelte + Vite
├── migrations/            # alembic
├── tests/
├── docs/superpowers/specs/
└── pyproject.toml
```

## Build Order (Milestones)

1. **M1 — Runtime skeleton:** HF cache scan, llama-server subprocess management, OpenAI-compat client, smoke test with a real tiny model
2. **M2 — Engine skeleton:** data model, QuestStore, Pipeline/AgentStep abstractions, stub agents that produce placeholder text, end-to-end fake-LLM test
3. **M3 — API + minimal UI:** create quest → advance chapter → render prose → record choice. One quest, one pipeline, no tuning yet.
4. **M4 — Library + forks:** library view, fork/branch, export/import, branch tree visualization
5. **M5 — Tuning panel:** per-chapter, per-agent, A/B, style guide editor; feedback → prompt injection plumbing
6. **M6 — Real agent prompts & polish:** replace stub agents with real prompts informed by StoryWriter / Agents' Room; templates; vote-flavor; settings UI

Each milestone is independently shippable and demoable.

## Open Questions (defer to implementation)

- Streaming granularity for the reader (token-level vs paragraph-level)
- Whether to support concurrent quest advancement (probably no for v1 — single llama-server, single active generation)
- Branch tree visualization library (roll our own SVG vs reuse)
