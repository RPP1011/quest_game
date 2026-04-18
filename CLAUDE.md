# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Locally-hosted web app for playing AI-generated forum-style quests (Sufficient Velocity style). Single-user, runs entirely on localhost against a user-managed `llama-server` (or vLLM) subprocess pointed at GGUF models in the HF cache. Currently using Gemma 4 26B (A4B quantization). Python 3.11 + FastAPI backend, SQLite persistence, interactive web UI in `web/`.

## Commands

```bash
uv sync                              # install deps
uv run pytest                        # unit tests (excludes integration + vllm by default)
uv run pytest -m integration         # integration tests (need real llama-server + GGUF)
uv run pytest tests/engine/test_pipeline.py::test_name   # single test

# CLI (entrypoint: app.cli.play:app)
uv run quest init --db <path> --seed <seed.json>     # bootstrap quest DB from seed
uv run quest play --db <path> --server <url>         # interactive play
uv run quest rollout --quest <id> --candidate <cid> --profile <p> --chapters <n> --server <url>
uv run quest refine --quest <id> --strategy all --server <url>
uv run quest summarize <quest_id> [rollout_ids...]   # extractive chapter summaries
uv run quest serve --quests-dir data/quests --server <url>  # web UI + API

# Evals (promptfoo — needs llama-server running)
cd evals && PROMPTFOO_PYTHON=../.venv/bin/python npx promptfoo eval
```

Pytest config (`pyproject.toml`): `addopts = "-m 'not integration and not vllm'"`. `asyncio_mode = "auto"`.

## Architecture

Three-layer design + rollout/refinement pipeline; see `docs/superpowers/specs/2026-04-12-quest-game-design.md` and `docs/superpowers/specs/2026-04-15-story-rollout-architecture.md`.

### `app/runtime/` — Model Runtime
Owns lifecycle of external inference processes. `InferenceClient` is an OpenAI-compatible async httpx client with logprob extraction (`chat_with_logprobs`, `expected_score`). `ChatMessage` is the message type for all LLM calls.

### `app/engine/` — Quest Engine (orchestration)
`Pipeline` runs a per-chapter DAG: 4 planners → per-beat writer (12-16 LLM calls/chapter) → check→revise loop (up to 4 passes) → typed edit pass → extract. `ContextBuilder` + `TokenBudget` assemble per-step prompt context. `PromptRenderer` renders Jinja templates from `prompts/`. `TraceStore` persists per-step traces. The check→revise loop fires on critical OR fixable issues; `_inject_heuristic_issues` runs the LLM metaphor critic after each check. After the loop, `typed_edits.py` does surgical span-level prose fixes with a 12-type taxonomy (cliche, forced_metaphor, purple_prose, etc.).

### `app/world/` — World State
SQLite-backed (`db.py`, schema in `schema.py`). Key tables: `entities`, `narrative`, `rollout_runs`, `rollout_chapters`, `kb_chapter_scores`, `foreshadow_triples`, `typed_edits`, `cross_judge_scores`, `knowledge_graph` (TKG, planned). `WorldStateManager` is the primary interface. `SeedLoader` bootstraps quests from JSON seeds (see `seeds/pale_lights.json` for the canonical 49-entity seed).

### `app/planning/` — Planners + Critics
Four planners run per chapter: `ArcPlanner` → `DramaticPlanner` → `EmotionalPlanner` → `CraftPlanner`. Additionally:
- `story_candidate_planner.py` — generates N story arcs from seed; player picks one
- `arc_skeleton_planner.py` — generates per-chapter outline (dramatic questions, plot beats, tension targets, entity surfacing)
- `metaphor_critic.py` — LLM-based imagery family classification (primary) with keyword fallback. Detects when any family exceeds budget.
- `voice_tracker.py` — per-character metaphor ring buffer tracking imagery usage across chapters
- `opening_critic.py` — flags repetitive chapter openings
- `foreshadow_pool.py` — CFPG-style (foreshadow, trigger, payoff) triples with prose-level verification via logprob confidence

### `app/rollout/` — Synthetic Playthroughs
`harness.py` orchestrates chapter-by-chapter rollouts against picked story candidates. Each rollout gets an isolated world DB under `data/quests/<qid>/rollouts/<rid>/`. `profiles.py` + `profiles/` define virtual-player personalities (impulsive, cautious, honor_bound). `action_selector.py` picks actions per chapter, optionally skeleton-aware. `scorer.py` runs per-dim independent logprob-weighted scoring (4 parallel calls, no cross-dim anchoring). `diversity.py` measures cross-rollout divergence via Jaccard similarity. Per-chapter harness also runs: foreshadow pool (trigger/verify), cross-judge (M-Prometheus-14B on CPU port 8083), structural metrics (syntactic CR, MTLD), and voice drift monitor (function-word KL divergence).

### `app/refinement/` — Iterative Chapter Improvement
`framework.py` runs `refine_one()` per target: regenerate chapter with strategy-specific guidance, dual-rate against baseline, accept if mean delta ≥ +0.05 with no dim regression > -0.10. Selectors in `selectors.py`: `WeakChapterSelector` (bottom quartile), `UnpaidHookSelector` (skeleton hook gaps), `SiblingOutscoredSelector` (cross-rollout comparison).

### `app/craft/` — Craft Library
Schemas + data (`app/craft/data/`) for narrative structures (three_act, etc.) and techniques referenced by planners.

### `app/retrieval/` — Embedding-based Retrievers
7 retrievers sharing `Query`/`Result` interface: passage, quest, scene, motif, foreshadowing, voice, craft. Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings, CPU by default.

### API + UI
`app/server.py` (factory: `create_app()`) exposes rollout endpoints, scoring, candidate picker, skeleton generator, refinement triggers. `web/` has an interactive frontend with chapter reading, choice selection, trace inspection, and a rollout viewer with side-by-side comparison mode.

### `evals/` — Prompt Engineering
Promptfoo-based eval harness. `provider.py` wraps the local llama-server. `scorers/` contains Python scorers (LLM metaphor classifier, keyword counter). `prompts/` has prompt variants for A/B testing. `dataset.json` has fixed test cases extracted from real traces. Run with `npx promptfoo eval` then `npx promptfoo view` for the dashboard.

## Docs layout

- `docs/superpowers/specs/` — design specs (authoritative)
- `docs/superpowers/plans/` — milestone implementation plans (follow when executing features)
- `docs/technical-report-*.md` — detailed writeups with labeled examples and eval results
- `docs/phase*.md`, `docs/day*.md` — rolling progress

When adding substantial features, check for an existing spec/plan before designing from scratch.

## Conventions

- `from __future__ import annotations` at the top of modules; Pydantic v2 models for API + persistence schemas.
- Prompts are Jinja templates under `prompts/`, rendered via `PromptRenderer`; don't hardcode prompt strings in Python.
- Integration tests must be marked `@pytest.mark.integration` (they spawn real `llama-server` + load GGUFs).
- LLM calls use `ChatMessage` objects, not dicts. Set `thinking=False` on calls that must return structured output (Gemma 4 defaults to thinking mode).
- The server uses a factory pattern (`create_app()`), not a module-level `app` — use `uv run quest serve`, not `uvicorn app.server:app`.
