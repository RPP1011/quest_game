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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runner import CorruptDatabaseError, run_quest  # noqa: E402
from app.runner_config import ConfigError, load_run_config  # noqa: E402
from app.runner_resume import (  # noqa: E402
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
    except CorruptDatabaseError as e:
        print(f"corrupt database: {e}", file=sys.stderr)
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
