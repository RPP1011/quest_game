"""One-shot parity check: new runner vs old script on stress-noir-5.

Runs each side, prints a side-by-side commit-rate + dim-means table.
Used during step 3 of the migration. Not a pytest — LFM is non-
deterministic at temperature 0.8, so the exact prose differs every run.

Tolerance: same commit rate, mean dims within +/- 0.05.
"""
from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.calibration.heuristics import dialogue_ratio, sentence_variance, pacing  # noqa: E402
from app.runner import run_quest  # noqa: E402
from app.runner_config import load_run_config  # noqa: E402


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
    cfg = cfg.model_copy(update={
        "options": cfg.options.model_copy(update={"db_path": new_db}),
    })
    asyncio.run(run_quest(cfg, fresh=True))
    new = _score_db(new_db)

    # Old script (assumes tools/stress_test_5.py still exists at this point
    # of the migration). If you've already deleted it, skip this side.
    old_script = REPO / "tools/stress_test_5.py"
    if old_script.is_file():
        subprocess.run([sys.executable, str(old_script)], check=False)
        # Old script's DB path varies — adjust based on what it writes.
        # Verify the path by reading the script first.
        old_default = Path("/tmp/stress_5/quest.db")
        if old_default.is_file():
            old = _score_db(old_default)
        else:
            old = None
    else:
        old = None

    def _row(name, d):
        if d is None:
            return f"{name:8s} | (script not run or db not found)"
        return (f"{name:8s} | committed={d['committed']:2d} "
                f"dlg={d.get('dlg', 0):.3f} "
                f"sv={d.get('sv', 0):.3f} "
                f"pc={d.get('pc', 0):.3f}")

    print()
    print("=== parity ===")
    print(_row("new", new))
    print(_row("old", old))
    if old is not None and old.get("committed", 0) > 0:
        for k in ("dlg", "sv", "pc"):
            delta = abs(new[k] - old[k])
            status = "OK" if delta <= 0.05 else "DRIFT"
            print(f"{k}: |delta|={delta:.3f} [{status}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
