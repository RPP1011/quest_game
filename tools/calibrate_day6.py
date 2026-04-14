"""Offline calibration for Day 6 LLM-judge dims.

Walks ``data/calibration/manifest.yaml`` (195 passage-scale) AND
``data/calibration/scenes_manifest.yaml`` (65 scene-scale), calls the
``BatchJudge`` with a real Claude (or local) client for the three Day 6
dims, and reports per-dim correlation + MAE against the Claude-labeled
expected scores from each manifest's ``expected`` block.

Usage::

    uv run python -m tools.calibrate_day6 \\
        --client claude --max-works 5 --out /tmp/day6_cal.json

Treat this as a manual tool — it spends tokens. CI's fixture test lives
at ``tests/calibration/test_day6_judges.py`` and runs without network.

Interpretation:

- ``tension_execution``: labels exist on every work in the passage
  manifest AND every scene in the scene manifest. This dim's r is the
  headline Day 6 number.
- ``choice_hook_quality``: only quest works carry this label. Use
  ``--is-quest`` to restrict. Expect wider scatter than
  ``tension_execution`` because the label pool is smaller (~4 quest
  works).
- ``emotional_trajectory``: has no direct Claude label. We sanity-check
  it via a co-correlation with ``emotional_arc`` (from
  ``/tmp/labels_claude_arc_*.json`` if present), and otherwise print
  raw score distribution so the operator can eyeball cluster shape.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from app.calibration.judges import BatchJudge
from app.calibration.loader import Manifest, Passage, Work, load_manifest
from app.calibration.scorer import aggregate
from app.scoring import LLM_JUDGE_DIMS


log = logging.getLogger("day6_calibrate")


DAY6_DIMS = list(LLM_JUDGE_DIMS)  # tension, emotional_trajectory, choice_hook


@dataclass
class PassageRecord:
    work_id: str
    passage_id: str
    is_quest: bool
    judge_scores: dict[str, float]
    expected: dict[str, float]  # only dims with manifest-level labels


def _iter_passages(
    manifest: Manifest,
    passages_dir: Path,
) -> "list[tuple[Work, Passage, Path]]":
    out: list[tuple[Work, Passage, Path]] = []
    for work in manifest.works:
        for p in work.passages:
            path = passages_dir / work.id / f"{p.id}.txt"
            if path.is_file():
                out.append((work, p, path))
    return out


async def _score_one(
    judge: BatchJudge,
    client: Any,
    work: Work,
    passage: Passage,
    body: str,
) -> PassageRecord:
    judged = await judge.score(
        client=client,
        passage=body,
        work_id=work.id,
        pov=work.pov,
        is_quest=work.is_quest,
        dim_names=DAY6_DIMS,
    )
    return PassageRecord(
        work_id=work.id,
        passage_id=passage.id,
        is_quest=work.is_quest,
        judge_scores={d: float(s.score) for d, s in judged.items()},
        expected={
            d: float(work.expected[d])
            for d in DAY6_DIMS
            if d in work.expected
        },
    )


def _report(records: list[PassageRecord]) -> dict[str, Any]:
    """Build the per-dim correlation report from a list of passage records.

    Scores are aggregated to the work level (mean of passages in the
    work) before correlation, matching the existing
    ``app.calibration.harness`` convention — otherwise within-work noise
    dominates.
    """
    per_dim: dict[str, Any] = {}
    for dim in DAY6_DIMS:
        # Collect per-work mean of judge scores where the manifest has a
        # label for this dim.
        by_work: dict[str, list[float]] = {}
        expected_by_work: dict[str, float] = {}
        for r in records:
            if dim not in r.expected:
                continue
            by_work.setdefault(r.work_id, []).append(r.judge_scores[dim])
            expected_by_work[r.work_id] = r.expected[dim]
        pairs = [
            (mean(scores), expected_by_work[w])
            for w, scores in by_work.items()
        ]
        if not pairs:
            per_dim[dim] = {
                "n_works": 0, "n_passages": 0,
                "pearson": None, "mae": None,
                "note": "no Claude labels available; "
                        "sanity-check via raw score distribution",
                "score_mean": mean(r.judge_scores[dim] for r in records)
                              if records else None,
            }
            continue
        stats = aggregate(pairs)
        per_dim[dim] = {
            "n_works": len(pairs),
            "n_passages": sum(len(v) for v in by_work.values()),
            "pearson": round(stats.pearson, 3),
            "mae": round(stats.mae, 3),
            "rmse": round(stats.rmse, 3),
            "per_work": {
                w: {
                    "judge_mean": round(mean(s), 3),
                    "expected": expected_by_work[w],
                }
                for w, s in by_work.items()
            },
        }
    return per_dim


async def _run(
    manifest_path: Path,
    passages_dir: Path,
    client: Any,
    prompts_dir: Path,
    max_passages: int,
    only_quest: bool,
) -> list[PassageRecord]:
    judge = BatchJudge(prompts_dir)
    manifest = load_manifest(manifest_path)
    candidates = _iter_passages(manifest, passages_dir)
    if only_quest:
        candidates = [c for c in candidates if c[0].is_quest]
    if max_passages > 0:
        candidates = candidates[:max_passages]

    records: list[PassageRecord] = []
    for work, passage, path in candidates:
        try:
            body = path.read_text(encoding="utf-8")
            rec = await _score_one(judge, client, work, passage, body)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s/%s: %s", work.id, passage.id, exc)
            continue
        records.append(rec)
        log.info(
            "scored %s/%s: " + " ".join(
                f"{d}={rec.judge_scores[d]:.2f}" for d in DAY6_DIMS
            ),
            work.id, passage.id,
        )
    return records


def _load_arc_labels() -> dict[tuple[str, str], float]:
    """Look up ``emotional_arc`` Claude labels from /tmp arc label files.

    Returns a map of (work_id, scene_id) -> emotional_arc score. Empty
    map if no files present (runs on the passage manifest still work,
    the emotional_trajectory dim just has no cross-check).
    """
    out: dict[tuple[str, str], float] = {}
    for f in glob.glob("/tmp/labels_claude_arc_*.json"):
        try:
            data = json.loads(Path(f).read_text())
        except Exception:
            continue
        passages = data if isinstance(data, list) else data.get("passages", [])
        for p in passages:
            scores = p.get("scores") or p.get("dimensions") or {}
            if "emotional_arc" not in scores:
                continue
            wid, sid = p.get("work_id"), p.get("scene_id")
            if not wid or not sid:
                continue
            out[(wid, sid)] = float(scores["emotional_arc"])
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="calibrate_day6")
    ap.add_argument("--manifest", default="data/calibration/manifest.yaml")
    ap.add_argument("--passages-dir", default="data/calibration/passages")
    ap.add_argument("--prompts-dir", default="prompts")
    ap.add_argument("--server-url", default=None,
                    help="llama-server URL for local judge")
    ap.add_argument("--anthropic", action="store_true",
                    help="use the anthropic SDK client (reads ANTHROPIC_API_KEY)")
    ap.add_argument("--model", default="claude-3-5-sonnet-latest")
    ap.add_argument("--max-passages", type=int, default=0,
                    help="limit total passages (0 = all)")
    ap.add_argument("--only-quest", action="store_true",
                    help="restrict to quest works (useful for choice_hook_quality)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.anthropic:
        from tools.anthropic_client import AnthropicClient  # local helper
        client: Any = AnthropicClient(model=args.model)
    elif args.server_url:
        from app.runtime.client import InferenceClient
        client = InferenceClient(base_url=args.server_url, timeout=600.0, retries=0)
    else:
        print("must pass --server-url or --anthropic", file=sys.stderr)
        return 2

    records = asyncio.run(_run(
        manifest_path=Path(args.manifest),
        passages_dir=Path(args.passages_dir),
        client=client,
        prompts_dir=Path(args.prompts_dir),
        max_passages=args.max_passages,
        only_quest=args.only_quest,
    ))
    report = _report(records)

    summary = {
        "n_passages": len(records),
        "dims": DAY6_DIMS,
        "per_dim": report,
    }
    out_path = Path(args.out) if args.out else Path("/tmp/day6_cal.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for dim, stats in report.items():
        r = stats.get("pearson")
        pass_fail = "PASS" if (r is not None and r >= 0.7) else "-"
        r_s = f"{r:.3f}" if r is not None else "n/a"
        print(f"  {dim:26s} r={r_s}  {pass_fail}  "
              f"n_works={stats.get('n_works', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
