"""Day 11: 5-chapter stress test (verification harness for Day 11 fixes).

Thin wrapper around tools/stress_test_50.py with --updates=5. Used to A/B
the Day 11 bottleneck fixes against the Day 10 baseline numbers. Writes
to a separate path so it doesn't clobber the canonical 50-chapter log.

Usage::

    uv run python tools/stress_test_5.py [--out path]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# vllm owns the GPU; force the retrieval embedder onto CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.stress_test_50 import run_updates  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="stress_test_5")
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--lora", type=str,
                    default=os.environ.get("LLM_MODEL", "writer_v1"))
    ap.add_argument("--llm-url", type=str,
                    default=os.environ.get("LLM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/stress_test_5"))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "stress" / "run_log_day11.jsonl")
    ap.add_argument("--scoring", action="store_true", default=True)
    ap.add_argument("--llm-judge", action="store_true", default=True)
    ap.add_argument("--retry-on-fail", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_updates(args))


if __name__ == "__main__":
    main()
