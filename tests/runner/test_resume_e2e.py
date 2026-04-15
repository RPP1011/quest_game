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
    # stress-noir-5 uses noir-investigation actions (12 total). Be tolerant
    # about how many of the remaining actions actually committed (the LLM
    # may flag some); just assert at least one committed past the resume
    # boundary and the final DB is correctly ordered.
    assert result.committed >= 1

    # Verify final DB has consecutive update_numbers from 1 onward
    conn = sqlite3.connect(cfg.options.db_path)
    update_numbers = [r[0] for r in conn.execute(
        "SELECT update_number FROM narrative ORDER BY update_number"
    )]
    conn.close()
    assert update_numbers == list(range(1, len(update_numbers) + 1)), \
        f"narrative update_numbers should be consecutive from 1, got {update_numbers}"
    assert len(update_numbers) >= 4, \
        f"expected at least 4 narrative rows after resume, got {len(update_numbers)}"
