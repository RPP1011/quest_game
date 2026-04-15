# Unified quest runner with resume — design

## Problem

Five tools in `tools/` drive scripted quest runs (`story_gen.py`,
`stress_test_5.py`, `stress_test_50.py`, `sft/collect_v2_corpus.py`,
`sft/collect_v3_corpus.py`). Each is ~250-500 lines of nearly-identical
boilerplate: bootstrap a `WorldStateManager` from a seed dict, wire
planners + retrievers + scorer, construct a `Pipeline`, loop over an
action list. The seeds and action lists are inline Python dicts/tuples,
duplicated across scripts.

None of the scripts can resume. Each starts with `db_path.unlink()`.
A 60-minute stress run that crashes at chapter 18 of 20 loses
everything. A 4-hour subagent corpus collection that dies in the middle
loses everything.

This spec consolidates the five scripts into one runner with resume
built into its contract from the first commit.

## Goals

- One CLI entrypoint (`tools/quest_run.py`) that takes `--config <yaml>`.
- A library function (`app.runner.run_quest`) that orchestrates the
  per-update loop, used by the CLI and importable by future tooling.
- Three layers of YAML config: **seeds** (world + narrator), **actions**
  (player input lists), **runs** (composition + options).
- Auto-resume on re-invocation: if the run's SQLite db has rows, continue
  from the next update. `--fresh` to start over.
- Strict resume validation: action-list drift refuses to resume with a
  named-index error.

## Non-goals

- Mid-pipeline checkpointing inside a single update. Updates are atomic
  — committed/flagged with a full `narrative` row, or absent.
- Resume for the interactive CLI (`app/cli/play.py`) or the server
  (`app/server.py`). Both are request-driven; no fixed action list.
- Distributed runs or parallel seed execution. One process, one config,
  one quest. Multi-seed collection is multiple invocations.
- Changes to `Pipeline`, `WorldStateManager`, planners, retrievers,
  Scorer, or any other engine internal. The runner is one layer above
  these and uses them unchanged.

## Architecture

```
tools/configs/runs/<name>.yaml      ← what to run (composed)
  └─ references seeds/<name>.yaml   ← world/narrator definition
  └─ references actions/<name>.yaml ← player input list
  └─ sets options                   ← n, sft, scoring, persona, run_log

tools/quest_run.py                  ← thin CLI: load config → call library
  --config <path>
  --fresh                           ← optional override

app/runner.py                       ← library, no CLI, no I/O of its own
  async run_quest(config, *, fresh=False) -> RunResult
app/runner_config.py                ← YAML → RunConfig dataclass + validation
```

`run_quest()` lives under `app/` because it is orchestration logic —
testable, importable, decoupled from CLI concerns. The CLI is ~30 lines
(parse args, load yaml, call `run_quest()`, print summary).

The five existing scripts get deleted after the parity check (step 4
of the migration plan).

## Config schemas

### `tools/configs/seeds/<name>.yaml`

Pure world + narrator definition. No run options.

```yaml
quest_id: noir
genre: low-fantasy noir
entities:
  - {id: inn, entity_type: location, name: The Salt and Star}
  - id: player
    entity_type: character
    name: Kaela
    data:
      voice:
        vocabulary_level: plain
        sentence_length_bias: short_clipped
        directness: 0.8
plot_threads:
  - {id: pt:main, name: The Missing Cargo, ...}
themes: [...]
foreshadowing: [...]
narrator:
  pov_character_id: player
  pov_type: third_limited
  worldview: a weathered observer; notices hands and silences
  sensory_bias: {visual: 0.3, tactile: 0.2, ...}
  voice_samples:
    - "She set the cup down the way she did everything else..."
    - ...
```

### `tools/configs/actions/<name>.yaml`

A list of strings. Nothing else.

```yaml
- I study the room, looking for who's trying too hard not to be noticed.
- I sit at Merrin's bar and ask whether the Gannet crew came through last week.
- ...
```

### `tools/configs/runs/<name>.yaml`

The composed run config. Top-level fields:

```yaml
seed: folk_horror              # → loads tools/configs/seeds/folk_horror.yaml
actions: folk-horror-arrival   # → loads tools/configs/actions/folk-horror-arrival.yaml
                               # OR: actions: ["literal action 1", "literal action 2"]
options:
  n_candidates: 4
  scoring: true
  rerank_weights:              # inline override; default is the b943513 set
    dialogue_ratio: 3.0
    sentence_variance: 2.5
  sft_collection:
    enabled: true
    dir: data/sft/v3/folk_horror
  persona_cycle: false         # only stress tests use this
  run_log: data/stress_v3/folk_horror_run_log.jsonl
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
  db_path: /tmp/quest_run/folk_horror.db   # default: /tmp/quest_run/<run_name>.db
```

Default `db_path` is `/tmp/quest_run/<run_name>.db` if omitted, where
`<run_name>` is the stem of the run config filename (e.g.
`tools/configs/runs/collect-v3-noir.yaml` → `/tmp/quest_run/collect-v3-noir.db`).
This way resume works across invocations of the same config without
anyone passing a path.

`actions` may be either a string (file reference) or a list of strings
(inline). One-off configs that don't deserve a separate file can stay
self-contained.

## Library API

`app/runner.py`:

```python
@dataclass(frozen=True)
class RunResult:
    run_name: str
    db_path: Path
    actions_total: int            # len(config.actions)
    skipped_resume: int           # how many skipped because already in DB
    committed: int                # successful commits this invocation
    flagged: int                  # outcome=flagged_qm
    errors: int                   # exceptions
    wall_clock_seconds: float


class ResumeMismatchError(Exception):
    def __init__(self, index: int, db_action: str, config_action: str): ...


class ConfigDriftError(Exception):
    """Config has fewer actions than the DB has rows."""


class WrongDatabaseError(Exception):
    """DB exists but its quest_id doesn't match the config."""


async def run_quest(
    config: RunConfig,
    *,
    fresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> RunResult:
    """Bootstrap or resume a quest run.

    On entry: if config.options.db_path exists and not fresh, open it,
    validate action-list match, skip already-done actions. Otherwise
    unlink + bootstrap from seed.
    """
```

`RunConfig` is a frozen dataclass that mirrors the YAML structure.
Loading is in `app/runner_config.py`:

```python
def load_run_config(path: Path) -> RunConfig:
    """Parse YAML, resolve seed + actions references, validate, return.

    Raises ConfigError on missing required fields, unknown keys, or
    unresolvable seed/actions references.
    """
```

The `progress_callback(committed_so_far, total, current_action)` exists
so the CLI can print per-update status without `run_quest()` having any
opinion about stdout. Tests pass `None` and assert against `RunResult`.

## Resume contract

On entry to `run_quest()`:

1. **`fresh=True`**: `db_path.unlink(missing_ok=True)`. Bootstrap from
   seed. Done.

2. **`fresh=False` and DB doesn't exist or has zero `narrative` rows**:
   bootstrap from seed, same as fresh. (Empty-DB-exists is just an
   aborted prior run.)

3. **`fresh=False` and DB has rows**: resume path below.

### Resume path

```
SELECT update_number, player_action, outcome FROM narrative
ORDER BY update_number
```

Both `committed` and `flagged_qm` rows count as "done". A flagged row
represents a persisted attempt — re-running it would either crash on a
duplicate `update_number` or get the same flag.

Walk the rows in `update_number` order. For each row at zero-indexed
position `i`:

- If `i >= len(config.actions)`: raise **`ConfigDriftError`** — "DB has
  more committed actions than the current config defines. Pass --fresh
  or extend the action list."
- If `row.player_action != config.actions[i]`: raise
  **`ResumeMismatchError(index=i, db_action=row.player_action,
  config_action=config.actions[i])`**.

If the walk passes, `start_from = max(update_number) + 1`. Run
`config.actions[start_from-1:]`. (Worked example: 3 committed rows with
update_numbers 1, 2, 3 → `start_from = 4` → slice = `actions[3:]` → the
4th action runs first, with `update_number=4`.)

### Other validation

- **DB exists but `quest_id` differs from `config.seed.quest_id`**: raise
  **`WrongDatabaseError`** — caller pointed at the wrong db_path or
  config.
- **DB exists but `narrative` table is missing entirely** (corrupt or
  pre-schema): error pointing the user at `--fresh`.

### Side effects of resume

- Arc state, reader state, motifs, foreshadowing — all live in DB rows.
  They're already populated. Do NOT re-bootstrap.
- SFT collection (if enabled) writes per-update files under
  `data/sft/<quest_id>/`. On resume, prior files exist; we skip those
  updates entirely so the hook never re-fires for them.
- `run_log` (if configured) is JSONL append. We append from `start_from`
  onward — the log will have a per-invocation gap visible by timestamp,
  no special markers.

## Migration plan

Five steps. Each is a separate commit. Each keeps the tree shippable.

### Step 1 — Build new infrastructure side-by-side.
- `app/runner.py`, `app/runner_config.py`, dataclasses + `run_quest()`.
- `tools/quest_run.py` CLI.
- Unit tests in `tests/runner/` (config validation, resume logic with
  fake pipeline, mismatch error paths).
- Existing 5 scripts untouched.

### Step 2 — Author config files.
- `tools/configs/seeds/{noir,intrigue,heist,folk_horror}.yaml`.
- `tools/configs/actions/{noir-investigation,intrigue-court,heist-vault,folk-horror-arrival}.yaml`.
- `tools/configs/runs/{story-gen-noir,stress-noir-5,stress-noir-50,collect-v2-{noir,intrigue,heist},collect-v3-{noir,intrigue,heist,folk-horror}}.yaml`.
- Each existing script's behavior reproducible via one config.

### Step 3 — Verify parity.
- Run new runner on `stress-noir-5.yaml`, compare commit count + per-chapter
  dim means against an old `stress_test_5.py` run.
- Tolerance: same commit rate, mean dims within ±0.02 (LFM is
  non-deterministic at temp 0.8 even with same seed).
- One representative run each of: stress, collect, story-gen.

### Step 4 — Delete old scripts.
- `git rm` `story_gen.py`, `stress_test_5.py`, `stress_test_50.py`,
  `sft/collect_v2_corpus.py`, `sft/collect_v3_corpus.py`.
- Update `docs/` and `README.md` references.
- Append "rerun this with: `tools/quest_run.py --config <path>`" line at
  bottom of the existing per-phase docs (don't rewrite the historical
  narrative).

### Step 5 — Document resume.
- Section in `docs/runner.md`: "Resume = re-run the same `--config`.
  `--fresh` to start over."

If step 3 surfaces a parity issue, fix it before step 4. Don't delete
old scripts until parity holds.

## Testing

### Unit (fast, no LLM, no vllm) — `tests/runner/`
- `test_config_load.py`:
  - valid YAML loads to RunConfig
  - missing required key → ConfigError naming the field
  - unknown key → ConfigError (strict; catches typos)
  - inline `actions: [...]` works alongside file-reference form
- `test_resume_logic.py` (uses an in-memory SQLite + a fake pipeline):
  - empty DB → start_from=1
  - 3 committed rows matching config[:3] → start_from=4
  - 3 rows where row[1].player_action ≠ config[1] → ResumeMismatchError(index=1)
  - 5 rows but config has 3 actions → ConfigDriftError
  - DB with mismatched quest_id → WrongDatabaseError
  - flagged_qm rows count toward "done" same as committed
- `test_run_quest_with_fake_pipeline.py`:
  - fake Pipeline returns canned outputs; assert run_quest() loops over
    the right action slice on resume
  - assert RunResult fields match (committed, flagged, skipped_resume)

### Integration (slow, real pipeline, real LLM, gated `pytest -m vllm`)

Add a new `vllm` marker to `pyproject.toml` `[tool.pytest.ini_options]`
markers list (does not exist today). Tests under this marker do not run
in the default `pytest` invocation; CI and developer workflows opt in
explicitly with `pytest -m vllm`.

- `test_resume_end_to_end.py`:
  - start a 3-action run, kill after action 2 (raise from a callback)
  - re-invoke same config, assert action 3 runs and the final DB has 3
    distinct chapters with consecutive update_numbers

### Parity (one-shot, not pytest)
- `tools/verify_runner_parity.py`:
  - run new runner against `stress-noir-5.yaml`
  - run old `stress_test_5.py`
  - print side-by-side commit-rate + dim-means table
- Used during step 3 of migration. Not retained in CI; LFM is
  non-deterministic.

## Runtime integration

The runner is a library *above* `app/engine/pipeline.py`. It uses but
does not modify any existing engine component:

| Component | Touched? | How |
|---|---|---|
| `app/engine/pipeline.py` | No | Used as-is via `Pipeline()` + `.run()` |
| `app/world/state_manager.py` | No | Used as-is; resume reads `narrative` rows the WSM already writes |
| `app/world/db.py` | No | `open_db()` already creates the schema we read |
| `app/runtime/client.py` | No | `InferenceClient(base_url, model)` constructed identically |
| Planners + retrievers + Scorer | No | Wired identically; YAML drives which |
| `app/server.py` | No | Server is request-driven, single-chapter |
| `app/cli/play.py` | No | Interactive; no fixed action list |
| Training (`tools/finetune/`) | No | Reads jsonl; doesn't care how it was produced |
| A/B harnesses | No | Score existing prose; orthogonal |

What the runner enables next:

- **Subagent checkpoints (next spec)**: an agent dispatched to "collect
  v4 corpus for noir" runs `tools/quest_run.py --config <yaml>`. If the
  agent dies at action 12 of 20, the re-dispatched agent runs the same
  command and resume picks up at 13. The subagent only has to checkpoint
  its own multi-step waypoints (corpus done → train started → train done
  → A/B done), not the per-update quest progress.
- **LoRA reproducibility (third spec)**: when we need to retrain v2 from
  scratch, the train.jsonl is the source of truth; reproducing it is
  `tools/quest_run.py --config tools/configs/runs/collect-v2-{noir,intrigue,heist}.yaml`
  for each seed. Resume makes that practical.
- **Cheap rerun comparisons**: change the writer prompt, rerun the same
  `stress-noir-20.yaml`, diff per-chapter dim means against the prior DB.
  Same config, same actions, real A/B at the run-script level.
