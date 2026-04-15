# Quest runner

One CLI for all scripted quest runs (demos, stress tests, SFT corpus
collection). Replaces the five `tools/` scripts that existed before.

## Run a config

```bash
tools/quest_run.py --config tools/configs/runs/<name>.yaml
```

If the run's SQLite DB exists, the runner resumes from the next
update_number. To start fresh, pass `--fresh` (or `rm` the DB file).

## Resume rules

- Resume looks at the run's `db_path` (default
  `/tmp/quest_run/<run_name>.db`).
- If the DB has narrative rows, each is matched against the current
  config's `actions` list at the same index.
- On any mismatch, resume refuses with a named-index error.
  Pass `--fresh` if you intentionally edited the action list.
- Both committed and flagged_qm chapters count as "done" — they're
  persisted attempts, not retried.
- A DB file that exists but is missing the `narrative` table (corrupt
  or pre-schema) raises `CorruptDatabaseError` instead of being wiped.

## Composing a run

Three layers:

- `tools/configs/seeds/<name>.yaml` — world + narrator (entities,
  themes, plot threads, voice samples, sensory bias).
- `tools/configs/actions/<name>.yaml` — list of player input strings.
- `tools/configs/runs/<name>.yaml` — composition + per-run options
  (n_candidates, scoring, sft_collection, run_log, llm_url, llm_model,
  rerank_weights).

`actions` may be either a string (file reference) or an inline list.

## Programmatic use

```python
from pathlib import Path
import asyncio
from app.runner import run_quest
from app.runner_config import load_run_config

cfg = load_run_config(Path("tools/configs/runs/stress-noir-5.yaml"))
result = asyncio.run(run_quest(cfg))
print(result.committed, result.flagged, result.skipped_resume)
```

## Errors

- Exit 2: config load error (missing/typo'd field, unresolvable seed/actions
  reference).
- Exit 3: resume refused (action mismatch, config drift, wrong DB, or
  corrupt DB).

The named exceptions are `ResumeMismatchError`, `ConfigDriftError`,
`WrongDatabaseError` (in `app.runner_resume`) and `CorruptDatabaseError`
(in `app.runner`).
