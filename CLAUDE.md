# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Locally-hosted web app for playing AI-generated forum-style quests (Sufficient Velocity style). Single-user, runs entirely on localhost against a user-managed `llama-server` (or vLLM) subprocess pointed at GGUF models in the HF cache. Python 3.11 + FastAPI backend, SQLite persistence, static web UI in `web/`.

## Commands

```bash
uv sync                              # install deps
uv run pytest                        # unit tests (excludes integration by default)
uv run pytest -m integration         # integration tests (need real llama-server + GGUF)
uv run pytest tests/engine/test_pipeline.py::test_name   # single test
uv run uvicorn app.server:app --reload    # run API server
uv run quest ...                     # CLI entrypoint (app.cli.play:app)
```

Pytest config (`pyproject.toml`) has `addopts = "-m 'not integration'"` — integration tests are opt-in. `asyncio_mode = "auto"`.

## Architecture

Three-layer design; see `docs/superpowers/specs/2026-04-12-quest-game-design.md` for the authoritative spec.

### `app/runtime/` — Model Runtime
Owns lifecycle of external inference processes. `RuntimeBackend` protocol with `LlamaServerBackend` default; `ModelCatalog` scans `~/.cache/huggingface/hub` for GGUFs; `InferenceClient` is an OpenAI-compatible async httpx client. One model loaded at a time (swap = stop + start).

### `app/engine/` — Quest Engine (agent-agnostic orchestration)
`Pipeline` executes a DAG of `AgentStep`s (topological order, parallel via `asyncio.gather`). `ContextBuilder` + `TokenBudget` assemble per-step prompt context within the ctx window; `PromptRenderer` renders Jinja prompts from `prompts/`. `TraceStore` persists per-step traces for the tuning UI. `check.py` / `extract.py` / `stages.py` implement the check-revise loop.

### `app/world/` — World State
SQLite-backed (`db.py`, schema in `schema.py`). `WorldStateManager` applies deltas produced by pipeline steps; `retcon.py` handles edits to past state; `rules_engine.py` enforces world invariants; `output_parser.py` extracts structured deltas from LLM output; `SeedLoader` bootstraps quests.

### Planning / Craft / Retrieval
- `app/planning/` — the concrete agent roles (arc, dramatic, emotional, craft, voice, motives, perception, critics, reader-model). Each is a step plugged into a Pipeline.
- `app/craft/` — the "craft library": schemas + data (`app/craft/data/`) for narrative techniques referenced by planners.
- `app/retrieval/` — embedding-based retrievers (passages, scenes, motifs, voice, foreshadowing, craft, quest). Used by context-building to pull relevant prior content.
- `app/scoring/`, `app/calibration/`, `app/optimization/` — evaluation + tuning infrastructure (calibration runs, A/B, prompt optimization).

### API + UI
`app/server.py` exposes the FastAPI surface (models in `app/api/quest.py`); static frontend in `web/` served directly. CLI in `app/cli/play.py`.

## Tools + Experiments

`tools/` contains one-off scripts for data pipelines (corpus build/fetch, stress tests, story generation, calibration runs, LoRA A/B). `tools/finetune/` and `tools/sft/` are training harnesses for writer-LoRA work (v1→v3; see `docs/writer-finetune-plan.md`, `docs/phase*.md`). Generated run logs land under `data/` (e.g. `data/stress_v3/` is gitignored).

## Docs layout

- `docs/superpowers/specs/` — design specs (authoritative)
- `docs/superpowers/plans/` — milestone implementation plans (follow when executing features)
- `docs/status-*.md`, `docs/phase*.md`, `docs/day*.md` — rolling progress writeups

When adding substantial features, check for an existing spec/plan before designing from scratch.

## Conventions

- `from __future__ import annotations` at the top of modules; Pydantic v2 models for API + persistence schemas.
- Prompts are Jinja templates under `prompts/`, rendered via `PromptRenderer`; don't hardcode prompt strings in Python.
- Integration tests must be marked `@pytest.mark.integration` (they spawn real `llama-server` + load GGUFs).
