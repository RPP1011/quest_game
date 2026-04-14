"""CLI: `python -m app.calibration {init,run} ...`."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .harness import Harness
from .loader import init_passage_hashes, load_manifest
from .report import to_json, to_text


DEFAULT_MANIFEST = Path("data/calibration/manifest.yaml")


def _cmd_init(args: argparse.Namespace) -> int:
    updated = init_passage_hashes(args.manifest, args.passages_dir)
    total = sum(len(v) for v in updated.values())
    print(f"Updated {total} passage hashes across {len(updated)} works.")
    for wid, entries in updated.items():
        for pid, digest in entries.items():
            print(f"  {wid}/{pid}: {digest[:12]}...")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    client = None
    if args.server_url:
        from app.runtime.client import InferenceClient  # lazy
        client = InferenceClient(base_url=args.server_url, timeout=300.0, retries=0)
    harness = Harness(
        manifest,
        passages_dir=args.passages_dir,
        client=client,
    )
    report = asyncio.run(harness.run())
    if args.json:
        print(to_json(report))
    else:
        print(to_text(report, correlation_threshold=args.threshold))
    failing = report.failing_correlation(args.threshold)
    if args.strict and failing:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="calibrate", description="Calibration harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="Hash passages into manifest")
    pi.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    pi.add_argument("--passages-dir", required=True)
    pi.set_defaults(func=_cmd_init)

    pr = sub.add_parser("run", help="Score passages and emit report")
    pr.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    pr.add_argument("--passages-dir", required=True)
    pr.add_argument("--server-url", default=None,
                    help="If set, call LLM judges at this URL; otherwise heuristics only")
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--threshold", type=float, default=0.7)
    pr.add_argument("--strict", action="store_true",
                    help="Exit nonzero if any dim has r < threshold")
    pr.set_defaults(func=_cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
