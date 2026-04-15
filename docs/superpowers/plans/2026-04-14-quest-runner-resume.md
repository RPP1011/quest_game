# Unified quest runner with resume — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the five `tools/` quest-run scripts into one CLI + library with auto-resume baked into its contract. Replace inline Python seeds/actions with YAML.

**Architecture:** `app/runner.py` exposes `async run_quest(config, *, fresh) → RunResult`. It sits one layer above `Pipeline`, owns the per-update loop and resume detection, and uses `WorldStateManager` / `Pipeline` / planners / retrievers unchanged. `app/runner_config.py` parses YAML into a Pydantic `RunConfig`. `tools/quest_run.py` is the CLI. Three YAML layers: `seeds/<name>.yaml` (world+narrator), `actions/<name>.yaml` (string list), `runs/<name>.yaml` (composition + options).

**Tech stack:** Python 3.13, Pydantic v2 for `RunConfig`, PyYAML for parsing (both already in `pyproject.toml`), pytest with a new `vllm` marker for the integration test.

**Spec:** `docs/superpowers/specs/2026-04-14-quest-runner-resume-design.md`.

**Schema note:** the `narrative` table does NOT have an `outcome` column. The spec says "both committed and flagged_qm rows count as done" — both kinds of rows are persisted by `Pipeline.write_narrative()` regardless of outcome, so resume just SELECTs all `narrative` rows. No schema migration needed.

---

## Task 1: RunConfig + YAML loader (pure data, no I/O orchestration)

**Files:**
- Create: `app/runner_config.py`
- Test: `tests/runner/__init__.py` (empty marker)
- Test: `tests/runner/test_config_load.py`
- Test: `tests/runner/fixtures/seeds/sample.yaml`
- Test: `tests/runner/fixtures/actions/sample.yaml`
- Test: `tests/runner/fixtures/runs/sample.yaml`

- [ ] **Step 1: Create empty test package**

```bash
mkdir -p tests/runner/fixtures/seeds tests/runner/fixtures/actions tests/runner/fixtures/runs
touch tests/runner/__init__.py
```

- [ ] **Step 2: Write fixture YAML files**

`tests/runner/fixtures/seeds/sample.yaml`:
```yaml
quest_id: sample
genre: test
entities:
  - {id: player, entity_type: character, name: Test Player}
plot_threads: []
themes: []
foreshadowing: []
narrator:
  pov_character_id: player
  pov_type: third_limited
  worldview: a test narrator
  sensory_bias: {visual: 1.0}
  voice_samples:
    - "Test sample one."
    - "Test sample two — slightly longer for variance."
```

`tests/runner/fixtures/actions/sample.yaml`:
```yaml
- I do the first thing.
- I do the second thing.
- I do the third thing.
```

`tests/runner/fixtures/runs/sample.yaml`:
```yaml
seed: sample
actions: sample
options:
  n_candidates: 1
  scoring: false
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
  db_path: /tmp/quest_run_test/sample.db
```

- [ ] **Step 3: Write the failing tests**

`tests/runner/test_config_load.py`:
```python
from pathlib import Path
import pytest
from app.runner_config import (
    RunConfig, ConfigError, load_run_config, load_run_config_from_string,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_basic_run_config():
    cfg = load_run_config(FIXTURES / "runs" / "sample.yaml",
                          base_dir=FIXTURES)
    assert isinstance(cfg, RunConfig)
    assert cfg.run_name == "sample"
    assert cfg.seed.quest_id == "sample"
    assert cfg.seed.narrator.pov_character_id == "player"
    assert cfg.actions == [
        "I do the first thing.",
        "I do the second thing.",
        "I do the third thing.",
    ]
    assert cfg.options.n_candidates == 1
    assert cfg.options.db_path == Path("/tmp/quest_run_test/sample.db")


def test_inline_actions_list_supported():
    yaml_text = """
seed: sample
actions:
  - First inline action.
  - Second inline action.
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    cfg = load_run_config_from_string(yaml_text, run_name="inline-test",
                                      base_dir=FIXTURES)
    assert cfg.actions == ["First inline action.", "Second inline action."]


def test_default_db_path_uses_run_name():
    cfg = load_run_config(FIXTURES / "runs" / "sample.yaml",
                          base_dir=FIXTURES)
    # sample.yaml sets db_path explicitly; check the default code path too
    yaml_text = """
seed: sample
actions: sample
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    cfg2 = load_run_config_from_string(yaml_text, run_name="my-run",
                                       base_dir=FIXTURES)
    assert cfg2.options.db_path == Path("/tmp/quest_run/my-run.db")


def test_missing_required_field_raises_config_error():
    yaml_text = """
seed: sample
# actions intentionally missing
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    with pytest.raises(ConfigError) as excinfo:
        load_run_config_from_string(yaml_text, run_name="bad",
                                    base_dir=FIXTURES)
    assert "actions" in str(excinfo.value)


def test_unknown_key_rejected():
    yaml_text = """
seed: sample
actions: sample
typoo: 4   # typo of 'options' - should fail
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    with pytest.raises(ConfigError) as excinfo:
        load_run_config_from_string(yaml_text, run_name="typo",
                                    base_dir=FIXTURES)
    assert "typoo" in str(excinfo.value)


def test_unresolvable_seed_reference():
    yaml_text = """
seed: nonexistent
actions: sample
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    with pytest.raises(ConfigError) as excinfo:
        load_run_config_from_string(yaml_text, run_name="bad-ref",
                                    base_dir=FIXTURES)
    assert "nonexistent" in str(excinfo.value)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/runner/test_config_load.py -v`
Expected: ImportError on `app.runner_config` — module doesn't exist yet.

- [ ] **Step 5: Implement `app/runner_config.py`**

```python
"""YAML loading and validation for unified quest runner configs.

See docs/superpowers/specs/2026-04-14-quest-runner-resume-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """Raised when a run config is missing fields, has unknown keys, or
    references a seed/actions file that does not exist."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NarratorConfig(_Strict):
    pov_character_id: str | None = None
    pov_type: str
    worldview: str
    editorial_stance: str | None = None
    sensory_bias: dict[str, float] = Field(default_factory=dict)
    attention_bias: list[str] = Field(default_factory=list)
    voice_samples: list[str] = Field(default_factory=list)


class SeedConfig(_Strict):
    quest_id: str
    genre: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    plot_threads: list[dict[str, Any]] = Field(default_factory=list)
    themes: list[dict[str, Any]] = Field(default_factory=list)
    foreshadowing: list[dict[str, Any]] = Field(default_factory=list)
    narrator: NarratorConfig


class SftCollectionConfig(_Strict):
    enabled: bool = False
    dir: str | None = None


class RunOptions(_Strict):
    n_candidates: int = 1
    scoring: bool = False
    rerank_weights: dict[str, float] | None = None
    sft_collection: SftCollectionConfig = Field(default_factory=SftCollectionConfig)
    persona_cycle: bool = False
    run_log: str | None = None
    llm_url: str
    llm_model: str
    db_path: Path | None = None


class RunConfig(_Strict):
    run_name: str
    seed: SeedConfig
    actions: list[str]
    options: RunOptions


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(str(e)) from e


def _resolve_seed(value: Any, base_dir: Path) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ConfigError(f"seed: expected string name or inline dict, got {type(value).__name__}")
    seed_path = base_dir / "seeds" / f"{value}.yaml"
    if not seed_path.is_file():
        raise ConfigError(f"seed reference '{value}' not found at {seed_path}")
    data = _load_yaml(seed_path)
    if not isinstance(data, dict):
        raise ConfigError(f"seed file {seed_path} did not parse as a mapping")
    return data


def _resolve_actions(value: Any, base_dir: Path) -> list[str]:
    if isinstance(value, list):
        return [str(a) for a in value]
    if not isinstance(value, str):
        raise ConfigError(f"actions: expected string name or inline list, got {type(value).__name__}")
    actions_path = base_dir / "actions" / f"{value}.yaml"
    if not actions_path.is_file():
        raise ConfigError(f"actions reference '{value}' not found at {actions_path}")
    data = _load_yaml(actions_path)
    if not isinstance(data, list):
        raise ConfigError(f"actions file {actions_path} did not parse as a list")
    return [str(a) for a in data]


def _build_run_config(raw: dict, run_name: str, base_dir: Path) -> RunConfig:
    if not isinstance(raw, dict):
        raise ConfigError("run config must be a YAML mapping")
    if "actions" not in raw:
        raise ConfigError("required field 'actions' missing from run config")
    if "seed" not in raw:
        raise ConfigError("required field 'seed' missing from run config")
    seed_dict = _resolve_seed(raw["seed"], base_dir)
    actions_list = _resolve_actions(raw["actions"], base_dir)
    options_dict = raw.get("options") or {}
    if not isinstance(options_dict, dict):
        raise ConfigError("'options' must be a mapping if provided")
    # Detect unknown top-level keys (Pydantic catches nested ones)
    allowed_top = {"seed", "actions", "options"}
    unknown = set(raw.keys()) - allowed_top
    if unknown:
        raise ConfigError(f"unknown top-level keys in run config: {sorted(unknown)}")

    if "db_path" not in options_dict:
        options_dict["db_path"] = Path("/tmp/quest_run") / f"{run_name}.db"

    try:
        return RunConfig(
            run_name=run_name,
            seed=seed_dict,
            actions=actions_list,
            options=options_dict,
        )
    except ValidationError as e:
        raise ConfigError(str(e)) from e


def load_run_config(path: Path, *, base_dir: Path | None = None) -> RunConfig:
    """Load a run config from a YAML file.

    The run name is derived from the filename stem.
    ``base_dir`` defaults to the file's grandparent (so refs to
    ``seeds/<name>.yaml`` resolve relative to the configs directory).
    """
    path = Path(path)
    if base_dir is None:
        base_dir = path.parent.parent
    raw = _load_yaml(path)
    return _build_run_config(raw, run_name=path.stem, base_dir=base_dir)


def load_run_config_from_string(
    text: str, *, run_name: str, base_dir: Path
) -> RunConfig:
    """Variant for tests and inline configs."""
    raw = yaml.safe_load(text)
    return _build_run_config(raw, run_name=run_name, base_dir=base_dir)
```

- [ ] **Step 6: Run tests, verify pass**

Run: `.venv/bin/python -m pytest tests/runner/test_config_load.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add app/runner_config.py tests/runner/__init__.py tests/runner/fixtures/ tests/runner/test_config_load.py
git commit -m "feat(runner): RunConfig + YAML loader with strict validation

Pydantic-validated config with seed/actions file-reference resolution
and inline-actions support. Defaults db_path to /tmp/quest_run/<run_name>.db
when omitted. Strict 'extra=forbid' on all nested models so a typo'd
key (e.g. 'sff' instead of 'sft') is caught at load time, not 30
minutes into a corpus collection.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Resume-detection helper (pure function, no real DB)

**Files:**
- Create: `app/runner_resume.py`
- Test: `tests/runner/test_resume_logic.py`

- [ ] **Step 1: Write the failing tests**

`tests/runner/test_resume_logic.py`:
```python
"""Pure-logic tests for resume detection. No real pipeline, no LLM.

The narrative-row reader is injected as a callable so tests can supply
canned rows; production wires it to WorldStateManager.list_narrative.
"""
from app.runner_resume import (
    ResumeDecision,
    ResumeMismatchError,
    ConfigDriftError,
    WrongDatabaseError,
    decide_resume,
)


def _row(update_number, player_action):
    return {"update_number": update_number, "player_action": player_action}


ACTIONS = ["A1", "A2", "A3", "A4", "A5"]


def test_no_rows_means_fresh_start():
    decision = decide_resume(rows=[], actions=ACTIONS,
                             db_quest_id=None, config_quest_id="q")
    assert decision.start_from == 1
    assert decision.skipped == 0


def test_three_committed_rows_resume_at_four():
    rows = [_row(1, "A1"), _row(2, "A2"), _row(3, "A3")]
    decision = decide_resume(rows=rows, actions=ACTIONS,
                             db_quest_id="q", config_quest_id="q")
    assert decision.start_from == 4
    assert decision.skipped == 3


def test_action_drift_at_index_one_raises_mismatch():
    rows = [_row(1, "A1"), _row(2, "DIFFERENT"), _row(3, "A3")]
    try:
        decide_resume(rows=rows, actions=ACTIONS,
                      db_quest_id="q", config_quest_id="q")
    except ResumeMismatchError as e:
        assert e.index == 1
        assert e.db_action == "DIFFERENT"
        assert e.config_action == "A2"
    else:
        raise AssertionError("expected ResumeMismatchError")


def test_more_db_rows_than_config_actions_raises_drift():
    rows = [_row(i, f"A{i}") for i in range(1, 6)]
    short_actions = ["A1", "A2", "A3"]
    try:
        decide_resume(rows=rows, actions=short_actions,
                      db_quest_id="q", config_quest_id="q")
    except ConfigDriftError:
        return
    raise AssertionError("expected ConfigDriftError")


def test_wrong_quest_id_raises():
    rows = [_row(1, "A1")]
    try:
        decide_resume(rows=rows, actions=ACTIONS,
                      db_quest_id="other_quest", config_quest_id="q")
    except WrongDatabaseError as e:
        assert "other_quest" in str(e)
        assert "q" in str(e)
    else:
        raise AssertionError("expected WrongDatabaseError")


def test_quest_id_check_skipped_when_db_has_no_quest_id():
    # An empty DB (no arc/reader rows) reports db_quest_id=None.
    # We should not raise WrongDatabaseError on a fresh-bootstrapped DB.
    rows = []
    decision = decide_resume(rows=rows, actions=ACTIONS,
                             db_quest_id=None, config_quest_id="q")
    assert decision.start_from == 1


def test_resume_skips_using_max_update_number_not_count():
    # If a flagged row is at update_number=5 but only 3 committed rows
    # exist, MAX(update_number)=5 and start_from=6 — pipeline never
    # writes gaps, but document the behavior.
    rows = [_row(1, "A1"), _row(2, "A2"), _row(3, "A3"),
            _row(4, "A4"), _row(5, "A5")]
    decision = decide_resume(rows=rows, actions=ACTIONS,
                             db_quest_id="q", config_quest_id="q")
    assert decision.start_from == 6
    assert decision.skipped == 5
```

- [ ] **Step 2: Run, verify failure**

Run: `.venv/bin/python -m pytest tests/runner/test_resume_logic.py -v`
Expected: ImportError on `app.runner_resume`.

- [ ] **Step 3: Implement `app/runner_resume.py`**

```python
"""Pure resume-detection logic. No I/O — caller supplies the rows.

See docs/superpowers/specs/2026-04-14-quest-runner-resume-design.md
section 'Resume contract'.
"""
from __future__ import annotations

from dataclasses import dataclass


class ResumeMismatchError(Exception):
    def __init__(self, index: int, db_action: str, config_action: str) -> None:
        super().__init__(
            f"resume mismatch at action index {index}: "
            f"DB has {db_action!r}, config has {config_action!r}. "
            f"Pass --fresh or revert the action change."
        )
        self.index = index
        self.db_action = db_action
        self.config_action = config_action


class ConfigDriftError(Exception):
    """Config has fewer actions than the DB has rows."""


class WrongDatabaseError(Exception):
    """DB exists but its quest_id doesn't match the config."""


@dataclass(frozen=True)
class ResumeDecision:
    start_from: int  # 1-based update_number to run next
    skipped: int     # number of actions already done in DB


def decide_resume(
    *,
    rows: list[dict],
    actions: list[str],
    db_quest_id: str | None,
    config_quest_id: str,
) -> ResumeDecision:
    """Decide whether to resume and at what update_number.

    Parameters
    ----------
    rows:
        Narrative rows from the existing DB, ordered by update_number.
        Each row is a dict with at least ``update_number`` (int) and
        ``player_action`` (str).
    actions:
        The current run config's action list.
    db_quest_id:
        ``arc.quest_id`` from the existing DB, or ``None`` if no arc
        rows exist (e.g. truly fresh DB).
    config_quest_id:
        The current config's ``seed.quest_id``.

    Returns
    -------
    ResumeDecision with the 1-based ``start_from`` index for the next
    update and the number of already-done actions to skip.

    Raises
    ------
    WrongDatabaseError
        DB has a quest_id that differs from config's.
    ResumeMismatchError
        A committed action in the DB does not match the corresponding
        action in the current config.
    ConfigDriftError
        DB has more committed rows than the config has actions.
    """
    if not rows:
        # Fresh or empty-DB path. Quest-id check only fires when both
        # sides actually have a value, since a freshly-bootstrapped DB
        # may or may not have an arc row yet depending on caller order.
        return ResumeDecision(start_from=1, skipped=0)

    if db_quest_id is not None and db_quest_id != config_quest_id:
        raise WrongDatabaseError(
            f"DB has quest_id {db_quest_id!r}, config has {config_quest_id!r}"
        )

    if len(rows) > len(actions):
        raise ConfigDriftError(
            f"DB has {len(rows)} committed actions, "
            f"config has only {len(actions)}. Pass --fresh or extend the "
            f"action list."
        )

    for i, row in enumerate(rows):
        if row["player_action"] != actions[i]:
            raise ResumeMismatchError(
                index=i,
                db_action=row["player_action"],
                config_action=actions[i],
            )

    max_update = max(r["update_number"] for r in rows)
    return ResumeDecision(start_from=max_update + 1, skipped=len(rows))
```

- [ ] **Step 4: Run, verify pass**

Run: `.venv/bin/python -m pytest tests/runner/test_resume_logic.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add app/runner_resume.py tests/runner/test_resume_logic.py
git commit -m "feat(runner): pure resume-detection logic with named errors

Decoupled from DB I/O — caller supplies rows. Three named exceptions
(ResumeMismatchError, ConfigDriftError, WrongDatabaseError) carry the
context the CLI needs for actionable error messages.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `run_quest()` orchestration with fake pipeline

**Files:**
- Create: `app/runner.py`
- Test: `tests/runner/test_run_quest.py`

- [ ] **Step 1: Write the failing tests**

`tests/runner/test_run_quest.py`:
```python
"""Integration tests for run_quest() using a fake Pipeline.

The fake skips planners + LLM entirely and writes minimal narrative
rows so resume can find them on re-invocation.
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest

from app.runner import RunResult, run_quest
from app.runner_config import RunConfig, load_run_config_from_string
from app.runner_resume import ResumeMismatchError

FIXTURES = Path(__file__).parent / "fixtures"


def _make_config(actions, db_path, *, run_name="t"):
    yaml_text = f"""
seed: sample
actions:
{chr(10).join("  - " + a for a in actions)}
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
  db_path: {db_path}
"""
    return load_run_config_from_string(yaml_text, run_name=run_name,
                                       base_dir=FIXTURES)


class FakePipeline:
    """Stand-in for app.engine.pipeline.Pipeline.

    Writes a narrative row per call. Supports a kill_after_n hook so
    tests can simulate a crash mid-run.
    """
    def __init__(self, world, kill_after_n=None):
        self.world = world
        self.kill_after_n = kill_after_n
        self.calls = 0

    async def run(self, *, player_action, update_number, **_kw):
        self.calls += 1
        if self.kill_after_n is not None and self.calls > self.kill_after_n:
            raise RuntimeError("simulated crash")
        # Mimic the real pipeline write
        from app.world.schema import NarrativeRecord
        self.world.write_narrative(NarrativeRecord(
            update_number=update_number,
            raw_text=f"prose for action {update_number}",
            player_action=player_action,
            pipeline_trace_id=f"trace-{update_number}",
        ))
        # Mimic the trace shape run_quest reads
        class _Out:
            class _Trace:
                outcome = "committed"
            trace = _Trace()
            prose = f"prose for action {update_number}"
            choices = []
            beats = []
        return _Out()


def test_fresh_run_completes_all_actions(tmp_path):
    db = tmp_path / "fresh.db"
    cfg = _make_config(["A1", "A2", "A3"], db)
    fake_factory = lambda world: FakePipeline(world)
    result = asyncio.run(run_quest(cfg, _pipeline_factory=fake_factory))
    assert result.committed == 3
    assert result.skipped_resume == 0
    assert result.actions_total == 3


def test_resume_after_simulated_crash(tmp_path):
    db = tmp_path / "crash.db"
    cfg = _make_config(["A1", "A2", "A3"], db)

    # First run dies after 2 actions
    fake_factory = lambda world: FakePipeline(world, kill_after_n=2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        asyncio.run(run_quest(cfg, _pipeline_factory=fake_factory))

    # Verify DB has 2 rows
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT update_number, player_action FROM narrative ORDER BY update_number").fetchall()
    assert rows == [(1, "A1"), (2, "A2")]
    conn.close()

    # Re-invoke; should resume at action 3
    fake_factory_ok = lambda world: FakePipeline(world)
    result = asyncio.run(run_quest(cfg, _pipeline_factory=fake_factory_ok))
    assert result.committed == 1
    assert result.skipped_resume == 2
    assert result.actions_total == 3

    # Verify final DB has all 3 rows with consecutive update_numbers
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT update_number, player_action FROM narrative ORDER BY update_number").fetchall()
    assert rows == [(1, "A1"), (2, "A2"), (3, "A3")]
    conn.close()


def test_action_mismatch_refuses_resume(tmp_path):
    db = tmp_path / "drift.db"
    cfg1 = _make_config(["A1", "A2", "A3"], db)
    asyncio.run(run_quest(cfg1, _pipeline_factory=lambda w: FakePipeline(w, kill_after_n=2)))

    # Edit action 1
    cfg2 = _make_config(["A1", "A2-EDITED", "A3"], db)
    with pytest.raises(ResumeMismatchError) as excinfo:
        asyncio.run(run_quest(cfg2, _pipeline_factory=lambda w: FakePipeline(w)))
    assert excinfo.value.index == 1


def test_fresh_flag_overrides_existing_db(tmp_path):
    db = tmp_path / "fresh-override.db"
    cfg = _make_config(["A1", "A2", "A3"], db)
    asyncio.run(run_quest(cfg, _pipeline_factory=lambda w: FakePipeline(w, kill_after_n=1)))

    # Now run fresh — should re-do all 3
    result = asyncio.run(run_quest(cfg, fresh=True,
                                   _pipeline_factory=lambda w: FakePipeline(w)))
    assert result.committed == 3
    assert result.skipped_resume == 0


def test_progress_callback_invoked(tmp_path):
    db = tmp_path / "progress.db"
    cfg = _make_config(["A1", "A2"], db)
    seen = []
    def cb(committed_so_far, total, current_action):
        seen.append((committed_so_far, total, current_action))
    asyncio.run(run_quest(cfg, progress_callback=cb,
                          _pipeline_factory=lambda w: FakePipeline(w)))
    assert seen == [(0, 2, "A1"), (1, 2, "A2")]
```

- [ ] **Step 2: Run, verify failure**

Run: `.venv/bin/python -m pytest tests/runner/test_run_quest.py -v`
Expected: ImportError on `app.runner`.

- [ ] **Step 3: Implement `app/runner.py`**

```python
"""Unified quest runner — orchestration above app/engine/pipeline.py.

See docs/superpowers/specs/2026-04-14-quest-runner-resume-design.md.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.runner_config import RunConfig
from app.runner_resume import (
    ResumeMismatchError,
    ConfigDriftError,
    WrongDatabaseError,
    decide_resume,
)


@dataclass(frozen=True)
class RunResult:
    run_name: str
    db_path: Path
    actions_total: int
    skipped_resume: int
    committed: int
    flagged: int
    errors: int
    wall_clock_seconds: float


ProgressCallback = Callable[[int, int, str], None]


def _peek_db_quest_id(db_path: Path) -> str | None:
    """Return the arc.quest_id from an existing DB, or None if unknown."""
    if not db_path.is_file():
        return None
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        # arcs is the most stable indicator of a bootstrapped quest
        row = conn.execute("SELECT quest_id FROM arcs LIMIT 1").fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        # Table doesn't exist → not yet bootstrapped or schema mismatch
        return None
    finally:
        conn.close()


def _load_existing_rows(db_path: Path) -> list[dict]:
    if not db_path.is_file():
        return []
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT update_number, player_action FROM narrative "
            "ORDER BY update_number"
        )
        return [{"update_number": r[0], "player_action": r[1]} for r in cur]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _bootstrap_world(config: RunConfig):
    """Create a WorldStateManager + apply seed. Returns (sm, conn)."""
    from app.world import SeedLoader
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    from app.world.schema import QuestArcState, ReaderState

    db_path = config.options.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Materialize seed dict to a temp JSON for SeedLoader
    seed_path = db_path.with_suffix(".seed.json")
    seed_path.write_text(json.dumps(_seed_to_dict(config)))

    conn = open_db(db_path)
    sm = WorldStateManager(conn)
    payload = SeedLoader.load(seed_path)
    for rule in payload.rules:
        sm.add_rule(rule)
    for hook in payload.foreshadowing:
        sm.add_foreshadowing(hook)
    for pt in payload.plot_threads:
        sm.add_plot_thread(pt)
    for th in payload.themes:
        sm.add_theme(config.seed.quest_id, th)
    sm.apply_delta(payload.delta, update_number=0)

    sm.upsert_arc(QuestArcState(
        quest_id=config.seed.quest_id, arc_id="main",
        structure_id="three_act", scale="campaign",
        current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=config.seed.quest_id))
    return sm, conn


def _reopen_world(config: RunConfig):
    """Open an existing DB without re-applying seed."""
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    conn = open_db(config.options.db_path)
    return WorldStateManager(conn), conn


def _seed_to_dict(config: RunConfig) -> dict:
    """Materialize the seed Pydantic model back to the dict shape
    SeedLoader expects (it reads JSON from disk)."""
    seed = config.seed
    return {
        "entities": list(seed.entities),
        "plot_threads": list(seed.plot_threads),
        "themes": list(seed.themes),
        "foreshadowing": list(seed.foreshadowing),
        "narrator": seed.narrator.model_dump(),
        "rules": [],
    }


def _build_real_pipeline(world, config: RunConfig):
    """Wire the actual pipeline. Mirrors the boilerplate the deleted
    scripts had. Honors config.options for retrieval/scoring."""
    from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget
    from app.runtime.client import InferenceClient
    from app.craft.library import CraftLibrary
    from app.planning import DramaticPlanner, EmotionalPlanner, CraftPlanner
    from app.planning.arc_planner import ArcPlanner

    REPO = Path(__file__).resolve().parent.parent
    PROMPTS = REPO / "prompts"

    client = InferenceClient(
        base_url=config.options.llm_url,
        retries=1,
        model=config.options.llm_model,
    )
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(world, renderer, TokenBudget())
    craft_library = CraftLibrary(REPO / "app" / "craft" / "data")
    structure = craft_library.structure("three_act")

    planners = {
        "arc_planner": ArcPlanner(client, renderer),
        "dramatic_planner": DramaticPlanner(client, renderer, craft_library),
        "emotional_planner": EmotionalPlanner(client, renderer),
        "craft_planner": CraftPlanner(client, renderer, craft_library),
    }

    scorer = None
    if config.options.scoring or config.options.n_candidates > 1:
        from app.scoring.scorer import Scorer
        scorer = Scorer()

    quest_config = {
        "narrator": config.seed.narrator.model_dump(),
        "genre": config.seed.genre,
        "n_candidates": config.options.n_candidates,
        "retrieval": {"enabled": True},
    }
    if config.options.sft_collection.enabled:
        quest_config["sft_collection"] = {
            "enabled": True,
            "dir": config.options.sft_collection.dir,
        }

    return Pipeline(
        world, cb, client,
        **planners,
        craft_library=craft_library,
        structure=structure,
        scorer=scorer,
        rerank_weights=config.options.rerank_weights,
        quest_config=quest_config,
        quest_id=config.seed.quest_id,
        arc_id="main",
    )


async def run_quest(
    config: RunConfig,
    *,
    fresh: bool = False,
    progress_callback: ProgressCallback | None = None,
    _pipeline_factory: Callable[[Any], Any] | None = None,
) -> RunResult:
    """Bootstrap or resume a quest run.

    On entry: if config.options.db_path exists and not fresh, open it,
    validate action-list match, skip already-done actions. Otherwise
    unlink + bootstrap from seed.

    ``_pipeline_factory`` is for tests — supply a callable that takes
    a WorldStateManager and returns a Pipeline-like object.
    """
    db_path = config.options.db_path
    t0 = time.perf_counter()

    if fresh and db_path.is_file():
        db_path.unlink()

    rows = _load_existing_rows(db_path)
    db_quest_id = _peek_db_quest_id(db_path)
    decision = decide_resume(
        rows=rows,
        actions=config.actions,
        db_quest_id=db_quest_id,
        config_quest_id=config.seed.quest_id,
    )

    if decision.skipped == 0 and not db_path.is_file():
        world, _conn = _bootstrap_world(config)
    elif decision.skipped == 0 and db_path.is_file():
        # Empty existing DB — wipe + bootstrap
        db_path.unlink()
        world, _conn = _bootstrap_world(config)
    else:
        world, _conn = _reopen_world(config)

    if _pipeline_factory is not None:
        pipeline = _pipeline_factory(world)
    else:
        pipeline = _build_real_pipeline(world, config)

    committed = 0
    flagged = 0
    errors = 0
    total = len(config.actions)

    for i, action in enumerate(config.actions[decision.skipped:],
                                start=decision.start_from):
        if progress_callback is not None:
            progress_callback(committed + decision.skipped, total, action)
        try:
            out = await pipeline.run(player_action=action, update_number=i)
        except Exception:
            errors += 1
            raise
        outcome = getattr(out.trace, "outcome", "committed")
        if outcome == "committed":
            committed += 1
        elif outcome == "flagged_qm":
            flagged += 1

    return RunResult(
        run_name=config.run_name,
        db_path=db_path,
        actions_total=total,
        skipped_resume=decision.skipped,
        committed=committed,
        flagged=flagged,
        errors=errors,
        wall_clock_seconds=time.perf_counter() - t0,
    )
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/python -m pytest tests/runner/test_run_quest.py -v`
Expected: 5 passed. If `_build_real_pipeline` import errors fire (they shouldn't — only the `_pipeline_factory` path is exercised), narrow the import to be lazy inside the factory call.

- [ ] **Step 5: Commit**

```bash
git add app/runner.py tests/runner/test_run_quest.py
git commit -m "feat(runner): run_quest() orchestration with resume

run_quest(config, *, fresh=False, progress_callback=None) bootstraps
or resumes a scripted quest run. Handles seed → world bootstrap, real
pipeline construction (planners + retrievers + scorer wired by config),
per-update loop, and resume slicing. Tests use a fake pipeline; real
pipeline factory is only exercised by integration tests + the CLI.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: CLI thin wrapper

**Files:**
- Create: `tools/quest_run.py`

- [ ] **Step 1: Write the CLI**

```python
"""Unified quest runner CLI.

Usage::

    uv run python tools/quest_run.py --config tools/configs/runs/<name>.yaml
    uv run python tools/quest_run.py --config <path> --fresh
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.runner import run_quest
from app.runner_config import ConfigError, load_run_config
from app.runner_resume import (
    ConfigDriftError,
    ResumeMismatchError,
    WrongDatabaseError,
)


def _print_progress(committed_so_far: int, total: int, action: str) -> None:
    print(f"\n[{committed_so_far + 1}/{total}] {action}")


def main() -> int:
    p = argparse.ArgumentParser(description="Run a quest from a YAML config.")
    p.add_argument("--config", required=True, type=Path,
                   help="Path to a runs/<name>.yaml file.")
    p.add_argument("--fresh", action="store_true",
                   help="Delete the existing DB and start over.")
    args = p.parse_args()

    try:
        config = load_run_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    try:
        result = asyncio.run(run_quest(
            config,
            fresh=args.fresh,
            progress_callback=_print_progress,
        ))
    except ResumeMismatchError as e:
        print(f"resume refused: {e}", file=sys.stderr)
        return 3
    except ConfigDriftError as e:
        print(f"resume refused: {e}", file=sys.stderr)
        return 3
    except WrongDatabaseError as e:
        print(f"wrong database: {e}", file=sys.stderr)
        return 3

    print()
    print(f"=== run complete ({result.run_name}) ===")
    print(f"actions:   {result.actions_total}")
    print(f"skipped:   {result.skipped_resume} (resumed)")
    print(f"committed: {result.committed}")
    print(f"flagged:   {result.flagged}")
    print(f"errors:    {result.errors}")
    print(f"db:        {result.db_path}")
    print(f"time:      {result.wall_clock_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test the CLI parses args**

Run: `.venv/bin/python tools/quest_run.py --help`
Expected: prints argparse help. No tracebacks.

- [ ] **Step 3: Verify config-error path**

Run: `.venv/bin/python tools/quest_run.py --config /nonexistent.yaml; echo "exit $?"`
Expected: prints `config error: ...`, exits 2.

- [ ] **Step 4: Commit**

```bash
git add tools/quest_run.py
git commit -m "feat(runner): tools/quest_run.py CLI

Thin wrapper: parse args, load config, run, print summary. Maps the
named runner exceptions to non-zero exit codes (2 = config error,
3 = resume refused / wrong DB).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `vllm` pytest marker

**Files:**
- Modify: `pyproject.toml` (add to `[tool.pytest.ini_options]`)

- [ ] **Step 1: Inspect current marker config**

Run: `grep -A 10 "tool.pytest" pyproject.toml`
Note current `markers` list (if any).

- [ ] **Step 2: Add the `vllm` marker**

Edit `pyproject.toml`. If `[tool.pytest.ini_options]` exists, add to its `markers` list:
```toml
markers = [
    # existing markers...
    "vllm: integration tests that require a running vllm server (opt-in via -m vllm)",
]
```
If `[tool.pytest.ini_options]` doesn't exist, create it:
```toml
[tool.pytest.ini_options]
markers = [
    "vllm: integration tests that require a running vllm server (opt-in via -m vllm)",
]
```

- [ ] **Step 3: Verify pytest sees the marker**

Run: `.venv/bin/python -m pytest --markers | grep vllm`
Expected: line beginning `@pytest.mark.vllm: integration tests that require a running vllm server`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test: add vllm pytest marker for integration tests

Tests under @pytest.mark.vllm are opt-in via 'pytest -m vllm'. Default
'pytest' invocations skip them since they require a running vllm
server with the writer LoRA loaded.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Author seed YAMLs

**Files:**
- Create: `tools/configs/seeds/noir.yaml`
- Create: `tools/configs/seeds/intrigue.yaml`
- Create: `tools/configs/seeds/heist.yaml`
- Create: `tools/configs/seeds/folk_horror.yaml`

- [ ] **Step 1: Extract the noir seed dict from `tools/sft/collect_v2_corpus.py`**

Open `tools/sft/collect_v2_corpus.py`. Locate the `NOIR_SEED` dict (around line 30). Convert to YAML in `tools/configs/seeds/noir.yaml`. Preserve: entities (with the post-`bd6992b` `pov_character_id: player` and 4 voice samples), plot_threads, themes, foreshadowing, narrator block.

Example structure (full content from the Python source):
```yaml
quest_id: noir_v2
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
        emotional_expression: understated
  - id: innkeeper
    entity_type: character
    name: Merrin
    data:
      voice:
        vocabulary_level: coarse
        directness: 0.9
plot_threads:
  - {id: pt:main, name: The Missing Cargo, description: ..., arc_position: rising, priority: 8}
themes:
  - {id: t:trust, proposition: trust is bought with small tells, not words, stance: exploring}
foreshadowing: []
narrator:
  pov_character_id: player
  pov_type: third_limited
  worldview: a weathered observer; notices hands and silences
  editorial_stance: sympathetic but unsentimental
  sensory_bias: {visual: 0.3, tactile: 0.2, auditory: 0.15, kinesthetic: 0.15, interoceptive: 0.15, olfactory: 0.05}
  attention_bias: ["hands", "doorways", "what people don't say"]
  voice_samples:
    - "She set the cup down the way she did everything else — like the cup owed her rent."
    - "He didn't speak. Didn't have to. The silence did the asking, and it was patient."
    - "The room had four people in it when you walked in — the innkeeper behind the bar, two dock-hands at the corner table sharing a bowl of something that had stopped steaming an hour ago, and a woman in a grey coat who was trying not to be a fifth."
    - "Rain. Just rain. The kind that soaks through before you notice it, and by the time you do, there's no point running."
```

- [ ] **Step 2: Repeat for `intrigue.yaml` and `heist.yaml`**

Same process. Source dicts are `INTRIGUE_SEED` and `HEIST_SEED` in `tools/sft/collect_v2_corpus.py`. Both already have the `pov_character_id: player` and 4 varied voice samples per `bd6992b`.

- [ ] **Step 3: Extract `folk_horror.yaml`**

Source: `tools/sft/collect_v3_corpus.py`. Look for the FOLK_HORROR (or equivalent) seed dict. Same extraction.

- [ ] **Step 4: Verify the YAMLs round-trip through Pydantic**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
from app.runner_config import _load_yaml, SeedConfig
for name in ['noir', 'intrigue', 'heist', 'folk_horror']:
    raw = _load_yaml(Path(f'tools/configs/seeds/{name}.yaml'))
    cfg = SeedConfig(**raw)
    print(f'{name}: quest_id={cfg.quest_id}, voice_samples={len(cfg.narrator.voice_samples)}')
"
```
Expected: 4 lines printing each seed's quest_id and voice_samples count (≥ 4 each). No tracebacks.

- [ ] **Step 5: Commit**

```bash
git add tools/configs/seeds/
git commit -m "feat(runner): seed YAMLs extracted from collect_v2/v3 scripts

noir, intrigue, heist, folk_horror — same content as the inline Python
dicts in tools/sft/collect_v2_corpus.py and collect_v3_corpus.py post
the bd6992b voice-sample upgrade. Quest IDs preserve the v2/v3 suffix
(noir_v2 etc) so existing SFT collections don't appear cross-quest.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Author action YAMLs

**Files:**
- Create: `tools/configs/actions/noir-investigation.yaml`
- Create: `tools/configs/actions/intrigue-court.yaml`
- Create: `tools/configs/actions/heist-vault.yaml`
- Create: `tools/configs/actions/folk-horror-arrival.yaml`
- Create: `tools/configs/actions/noir-stress-20.yaml`
- Create: `tools/configs/actions/noir-demo-10.yaml`

- [ ] **Step 1: Extract action lists from each script**

For each existing `*_ACTIONS` tuple in `tools/story_gen.py`, `tools/sft/collect_v2_corpus.py`, `tools/sft/collect_v3_corpus.py`, write a YAML file with one action per `-`-line:

```yaml
- I study the room, looking for who's trying too hard not to be noticed.
- I sit at Merrin's bar and ask whether the Gannet crew came through last week.
- ...
```

Specifically:
- `noir-investigation.yaml` ← `NOIR_ACTIONS` from `collect_v2_corpus.py`
- `intrigue-court.yaml` ← `INTRIGUE_ACTIONS` from `collect_v2_corpus.py`
- `heist-vault.yaml` ← `HEIST_ACTIONS` from `collect_v2_corpus.py`
- `folk-horror-arrival.yaml` ← folk-horror actions from `collect_v3_corpus.py`
- `noir-demo-10.yaml` ← `ACTIONS` from `tools/story_gen.py` (10 actions)
- `noir-stress-20.yaml` ← if `tools/stress_test_50.py` has a static action list, extract it; else write a 20-action list of noir prompts (use the 10 from story_gen.py + 10 follow-up actions about pursuing the Callen lead)

- [ ] **Step 2: Verify each loads**

Run:
```bash
.venv/bin/python -c "
import yaml
from pathlib import Path
for p in sorted(Path('tools/configs/actions').glob('*.yaml')):
    data = yaml.safe_load(p.read_text())
    assert isinstance(data, list), f'{p}: not a list'
    assert all(isinstance(s, str) for s in data), f'{p}: contains non-string'
    print(f'{p.name}: {len(data)} actions')
"
```
Expected: one line per file, no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add tools/configs/actions/
git commit -m "feat(runner): action YAMLs extracted from existing scripts

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Author run YAMLs

**Files:**
- Create: `tools/configs/runs/story-gen-noir.yaml`
- Create: `tools/configs/runs/stress-noir-5.yaml`
- Create: `tools/configs/runs/stress-noir-20.yaml`
- Create: `tools/configs/runs/collect-v2-noir.yaml`
- Create: `tools/configs/runs/collect-v2-intrigue.yaml`
- Create: `tools/configs/runs/collect-v2-heist.yaml`
- Create: `tools/configs/runs/collect-v3-noir.yaml`
- Create: `tools/configs/runs/collect-v3-intrigue.yaml`
- Create: `tools/configs/runs/collect-v3-heist.yaml`
- Create: `tools/configs/runs/collect-v3-folk-horror.yaml`

- [ ] **Step 1: Write the story-gen demo run**

`tools/configs/runs/story-gen-noir.yaml`:
```yaml
seed: noir
actions: noir-demo-10
options:
  n_candidates: 1
  scoring: false
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
```

- [ ] **Step 2: Write the stress runs**

`tools/configs/runs/stress-noir-5.yaml`:
```yaml
seed: noir
actions: noir-investigation   # first 5 of these in practice; or write a noir-stress-5.yaml
options:
  n_candidates: 4
  scoring: true
  rerank_weights:
    dialogue_ratio: 3.0
    sentence_variance: 2.5
    sensory_density: 2.0
    pacing: 1.0
    action_fidelity: 1.0
    free_indirect_quality: 0.1
    detail_characterization: 0.1
    metaphor_domains_score: 0.1
    indirection_score: 0.1
    pov_adherence: 0.1
    named_entity_presence: 0.1
    narrator_sensory_match: 0.1
  persona_cycle: false
  run_log: data/stress/noir-5/run_log.jsonl
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
```

`tools/configs/runs/stress-noir-20.yaml`: same options but `actions: noir-stress-20`.

- [ ] **Step 3: Write the collection runs (v2 and v3)**

For each seed × {v2, v3} combination, the full file:
```yaml
seed: noir            # or intrigue / heist / folk_horror
actions: noir-investigation     # match the seed: intrigue-court / heist-vault / folk-horror-arrival
options:
  n_candidates: 4
  scoring: true
  rerank_weights:
    dialogue_ratio: 3.0
    sentence_variance: 2.5
    sensory_density: 2.0
    pacing: 1.0
    action_fidelity: 1.0
    free_indirect_quality: 0.1
    detail_characterization: 0.1
    metaphor_domains_score: 0.1
    indirection_score: 0.1
    pov_adherence: 0.1
    named_entity_presence: 0.1
    narrator_sensory_match: 0.1
  sft_collection:
    enabled: true
    dir: data/sft/v2/noir       # change suffix per run: v2/intrigue, v2/heist, v3/noir, v3/intrigue, v3/heist, v3/folk_horror
  persona_cycle: false
  run_log: data/sft/v2/noir/run_log.jsonl   # parallel to sft dir
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
```

(The `rerank_weights` block is identical across all runs. Acceptable repetition for now — extracting to a shared default is YAGNI until there's a third reason to share it.)

- [ ] **Step 4: Verify each loads as a `RunConfig`**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
from app.runner_config import load_run_config
for p in sorted(Path('tools/configs/runs').glob('*.yaml')):
    cfg = load_run_config(p)
    print(f'{p.name}: seed={cfg.seed.quest_id}, actions={len(cfg.actions)}, n={cfg.options.n_candidates}')
"
```
Expected: one line per run config, no errors.

- [ ] **Step 5: Commit**

```bash
git add tools/configs/runs/
git commit -m "feat(runner): run YAMLs for the 5 deleted scripts' use cases

story-gen-noir, stress-noir-{5,20}, collect-v2-{noir,intrigue,heist},
collect-v3-{noir,intrigue,heist,folk-horror}. Each one reproduces the
behavior of one previous Python script.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Parity verification script

**Files:**
- Create: `tools/verify_runner_parity.py`

- [ ] **Step 1: Write the parity script**

```python
"""One-shot parity check: new runner vs old script on stress-noir-5.

Runs each side, prints a side-by-side commit-rate + dim-means table.
Used during step 3 of the migration. Not a pytest — LFM is non-
deterministic at temperature 0.8, so the exact prose differs every run.

Tolerance: same commit rate, mean dims within ±0.05.
"""
from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.calibration.heuristics import dialogue_ratio, sentence_variance, pacing
from app.runner import run_quest
from app.runner_config import load_run_config


def _score_db(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT raw_text FROM narrative ORDER BY update_number"
    ).fetchall()
    conn.close()
    if not rows:
        return {"committed": 0}
    dlg = sum(dialogue_ratio(r[0] or "") for r in rows) / len(rows)
    sv = sum(sentence_variance(r[0] or "") for r in rows) / len(rows)
    pc = sum(pacing(r[0] or "") for r in rows) / len(rows)
    return {"committed": len(rows), "dlg": dlg, "sv": sv, "pc": pc}


def main() -> int:
    new_db = Path("/tmp/parity_new.db")
    old_db = Path("/tmp/parity_old/quest.db")
    new_db.unlink(missing_ok=True)
    if old_db.exists():
        old_db.unlink()

    # New runner
    cfg = load_run_config(REPO / "tools/configs/runs/stress-noir-5.yaml")
    cfg = cfg.model_copy(update={"options": cfg.options.model_copy(update={"db_path": new_db})})
    asyncio.run(run_quest(cfg, fresh=True))
    new = _score_db(new_db)

    # Old script (assumes tools/stress_test_5.py still exists at this point
    # of the migration). If you've already deleted it, skip this side.
    old_script = REPO / "tools/stress_test_5.py"
    if old_script.is_file():
        subprocess.run([sys.executable, str(old_script)], check=False)
        # Old script's DB path varies — adjust based on what it writes.
        # If it writes to /tmp/stress_5/quest.db:
        old_default = Path("/tmp/stress_5/quest.db")
        if old_default.is_file():
            old = _score_db(old_default)
        else:
            old = None
    else:
        old = None

    def _row(name, d):
        if d is None:
            return f"{name:8s} | (script not run)"
        return (f"{name:8s} | committed={d['committed']:2d} "
                f"dlg={d.get('dlg', 0):.3f} "
                f"sv={d.get('sv', 0):.3f} "
                f"pc={d.get('pc', 0):.3f}")

    print()
    print("=== parity ===")
    print(_row("new", new))
    print(_row("old", old))
    if old is not None:
        for k in ("dlg", "sv", "pc"):
            delta = abs(new[k] - old[k])
            status = "OK" if delta <= 0.05 else "DRIFT"
            print(f"{k}: |Δ|={delta:.3f} [{status}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit (don't run yet — Task 10 runs it)**

```bash
git add tools/verify_runner_parity.py
git commit -m "tools(runner): parity script for stress-noir-5

One-shot script (not pytest — LFM is non-deterministic). Runs new
runner then old stress_test_5.py, scores both with the heuristic dim
functions, prints side-by-side. Used during migration; deletable
after step 4 lands.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Run parity, fix issues if any

This task is human-judgment, not a TDD loop. Steps:

- [ ] **Step 1: Confirm vllm is up**

Run: `curl -sf http://127.0.0.1:8082/v1/models > /dev/null && echo OK || echo MISSING`
Expected: OK. If MISSING, start vllm before continuing (see `docs/wakeup-note.md` for the launch command).

- [ ] **Step 2: Run the parity script**

Run: `.venv/bin/python tools/verify_runner_parity.py`
Expected: prints two `committed=N dlg=X sv=Y pc=Z` lines and three `|Δ|=...` lines. All three deltas should be `[OK]` (≤ 0.05).

- [ ] **Step 3: If any delta is `[DRIFT]`, debug**

Likely causes:
- Different `n_candidates` in the new run vs old script
- Different rerank_weights (the old script may not have had them)
- Different scorer wiring (new always wires Scorer; old may not)
- Pipeline construction differs (different planner kwargs)

Open the old script and the new `_build_real_pipeline` side by side. Reconcile any mismatch in the run YAML (`tools/configs/runs/stress-noir-5.yaml`). Re-run parity. Loop until OK.

- [ ] **Step 4: No commit needed if no fixes; if fixes were made, commit them with explanation**

```bash
git add tools/configs/runs/stress-noir-5.yaml  # or app/runner.py
git commit -m "fix(runner): reconcile new runner with stress_test_5 pipeline wiring

Parity script flagged a <DIM> drift of <Δ>. Root cause: <difference>.
Fix: <change>. Post-fix parity passes (deltas within ±0.05).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Delete old scripts

**Files:**
- Delete: `tools/story_gen.py`
- Delete: `tools/stress_test_5.py`
- Delete: `tools/stress_test_50.py`
- Delete: `tools/sft/collect_v2_corpus.py`
- Delete: `tools/sft/collect_v3_corpus.py`
- Modify: `docs/phase2-kickoff-lora-v2.md` (append rerun-with-new-CLI line)
- Modify: `docs/phase3-lora-v3-kickoff.md` (append rerun-with-new-CLI line)
- Modify: `README.md` (if it references any of the deleted scripts — check first)

- [ ] **Step 1: Check for any other references**

Run: `grep -rn "story_gen\|stress_test_5\|stress_test_50\|collect_v2_corpus\|collect_v3_corpus" --include="*.py" --include="*.md" --exclude-dir=.venv .`
Note any files that reference the deleted scripts.

- [ ] **Step 2: Append rerun lines to phase docs**

Append to `docs/phase2-kickoff-lora-v2.md`:
```markdown

## Rerun

```bash
tools/quest_run.py --config tools/configs/runs/collect-v2-noir.yaml
tools/quest_run.py --config tools/configs/runs/collect-v2-intrigue.yaml
tools/quest_run.py --config tools/configs/runs/collect-v2-heist.yaml
```
```

Append to `docs/phase3-lora-v3-kickoff.md`:
```markdown

## Rerun

```bash
tools/quest_run.py --config tools/configs/runs/collect-v3-folk-horror.yaml
```
```

Update any tests or other docs surfaced in step 1 to point at the new CLI.

- [ ] **Step 3: Delete the old scripts**

```bash
git rm tools/story_gen.py tools/stress_test_5.py tools/stress_test_50.py \
       tools/sft/collect_v2_corpus.py tools/sft/collect_v3_corpus.py
```

- [ ] **Step 4: Run the test suite**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/retrieval/test_embeddings.py --ignore=tests/retrieval/test_craft_retriever.py`
Expected: 360+ passed (the 352 baseline + the new runner tests). The two retrieval tests with the pre-existing circular-import issue stay ignored.

- [ ] **Step 5: Commit**

```bash
git add docs/phase2-kickoff-lora-v2.md docs/phase3-lora-v3-kickoff.md
# (and any other modified files from step 1)
git commit -m "feat(runner): retire 5 quest-run scripts in favor of unified CLI

story_gen.py, stress_test_5.py, stress_test_50.py, sft/collect_v2_corpus.py,
sft/collect_v3_corpus.py — all behavior reproducible via
'tools/quest_run.py --config tools/configs/runs/<name>.yaml'. Phase
docs updated with rerun commands; historical narrative left intact.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: User-facing docs

**Files:**
- Create: `docs/runner.md`

- [ ] **Step 1: Write the doc**

```markdown
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
from app.runner import run_quest
from app.runner_config import load_run_config

cfg = load_run_config(Path("tools/configs/runs/stress-noir-5.yaml"))
result = await run_quest(cfg)
print(result.committed, result.flagged, result.skipped_resume)
```

## Errors

- Exit 2: config load error (missing/typo'd field, unresolvable seed/actions
  reference).
- Exit 3: resume refused (action mismatch, config drift, or wrong DB).

The named exceptions are `ResumeMismatchError`, `ConfigDriftError`,
`WrongDatabaseError` — all in `app.runner_resume`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runner.md
git commit -m "docs: user-facing quest runner guide

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Integration test (gated, optional)

**Files:**
- Create: `tests/runner/test_resume_e2e.py`

- [ ] **Step 1: Write the gated integration test**

```python
"""End-to-end resume test against a real vllm server.

Skipped by default. Run with: pytest -m vllm tests/runner/test_resume_e2e.py
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest

from app.runner import run_quest
from app.runner_config import load_run_config

pytestmark = pytest.mark.vllm


def test_resume_after_kill_continues_correctly(tmp_path):
    cfg_path = Path("tools/configs/runs/stress-noir-5.yaml")
    cfg = load_run_config(cfg_path)
    cfg = cfg.model_copy(update={
        "options": cfg.options.model_copy(update={"db_path": tmp_path / "e2e.db"}),
    })

    # First run: 3 actions then raise from the progress callback
    raised = []
    def kill_after(n):
        def cb(committed_so_far, total, action):
            if committed_so_far >= n:
                raise RuntimeError("simulated kill")
        return cb

    with pytest.raises(RuntimeError, match="simulated kill"):
        asyncio.run(run_quest(cfg, fresh=True,
                              progress_callback=kill_after(3)))

    # Verify partial DB
    conn = sqlite3.connect(cfg.options.db_path)
    n_rows = conn.execute("SELECT COUNT(*) FROM narrative").fetchone()[0]
    conn.close()
    assert n_rows == 3, f"expected 3 rows after kill at action 4, got {n_rows}"

    # Resume — finish the remaining 2
    result = asyncio.run(run_quest(cfg))
    assert result.skipped_resume == 3
    assert result.committed == 2
    assert result.actions_total == 5

    # Verify final DB has 5 distinct rows
    conn = sqlite3.connect(cfg.options.db_path)
    update_numbers = [r[0] for r in conn.execute(
        "SELECT update_number FROM narrative ORDER BY update_number"
    )]
    conn.close()
    assert update_numbers == [1, 2, 3, 4, 5]
```

- [ ] **Step 2: Verify the marker gates correctly**

Run: `.venv/bin/python -m pytest tests/runner/ -q`
Expected: 18 passed (the existing tests), 1 deselected (the vllm-marked one).

- [ ] **Step 3: Commit**

```bash
git add tests/runner/test_resume_e2e.py
git commit -m "test(runner): e2e resume test gated on @pytest.mark.vllm

Opt-in via 'pytest -m vllm'. Default 'pytest' invocations skip it.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Final check

After all tasks land:

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/retrieval/test_embeddings.py --ignore=tests/retrieval/test_craft_retriever.py`
Expected: 360+ passed (352 baseline + ~8 new unit tests + 5 run_quest tests + 6 config tests).

- [ ] **Step 2: Verify no orphan references**

Run: `grep -rn "story_gen\|stress_test_5\|stress_test_50\|collect_v2_corpus\|collect_v3_corpus" --include="*.py" --include="*.md" --exclude-dir=.venv .`
Expected: only the historical mentions in `docs/day*.md` and `docs/phase*.md` files (don't rewrite history).

- [ ] **Step 3: Push**

```bash
git push origin master
```
