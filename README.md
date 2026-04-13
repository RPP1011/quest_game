# Quest Game

Local web app for playing AI-generated forum-style quests.

## Dev setup

```
uv sync
uv run pytest
```

Integration tests (require a real `llama-server` binary and a tiny GGUF model on disk):

```
uv run pytest -m integration
```

See `docs/superpowers/specs/` for design, `docs/superpowers/plans/` for milestone plans.
