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
    with pytest.raises(RuntimeError, match="simulated crash"):
        asyncio.run(run_quest(cfg1, _pipeline_factory=lambda w: FakePipeline(w, kill_after_n=2)))

    # Edit action 1
    cfg2 = _make_config(["A1", "A2-EDITED", "A3"], db)
    with pytest.raises(ResumeMismatchError) as excinfo:
        asyncio.run(run_quest(cfg2, _pipeline_factory=lambda w: FakePipeline(w)))
    assert excinfo.value.index == 1


def test_fresh_flag_overrides_existing_db(tmp_path):
    db = tmp_path / "fresh-override.db"
    cfg = _make_config(["A1", "A2", "A3"], db)
    with pytest.raises(RuntimeError, match="simulated crash"):
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


def test_corrupt_db_raises_instead_of_wiping(tmp_path):
    """A DB file that exists but lacks the narrative table should error."""
    from app.runner import CorruptDatabaseError
    import sqlite3

    db = tmp_path / "corrupt.db"
    # Create a SQLite file with no narrative table
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()

    cfg = _make_config(["A1", "A2"], db)
    with pytest.raises(CorruptDatabaseError) as excinfo:
        asyncio.run(run_quest(cfg, _pipeline_factory=lambda w: FakePipeline(w)))
    assert "narrative" in str(excinfo.value)
    # File should NOT be unlinked
    assert db.exists()
