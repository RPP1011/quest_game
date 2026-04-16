"""Score every chapter of a fetched work with the heuristic dims of
``app.scoring.Scorer`` (no craft_plan, so critic dims are left at their
neutral 1.0). Writes a flat JSON of per-chapter dim scores + aggregate
stats, for use in the craft analysis report.

Usage::

    uv run python tools/score_chapters.py --work pale_lights \\
        --out data/calibration/annotations/pale_lights/chapter_scores.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.calibration.heuristics import run_heuristics  # noqa: E402


RAW_ROOT = Path("data/calibration/raw")


def main() -> None:
    ap = argparse.ArgumentParser(prog="score_chapters")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    chapters_dir = RAW_ROOT / args.work
    files = sorted(chapters_dir.glob("chap_*.txt"))

    rows = []
    for f in files:
        n = int(f.stem.split("_")[-1])
        text = f.read_text()
        words = len(text.split())
        dims = run_heuristics(text)
        rows.append({"chapter": n, "words": words, "dims": dims})

    agg = {}
    if rows:
        dim_names = list(rows[0]["dims"].keys())
        for d in dim_names:
            vals = [r["dims"][d] for r in rows]
            agg[d] = {
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }

    out = {
        "work": args.work,
        "n_chapters": len(rows),
        "total_words": sum(r["words"] for r in rows),
        "aggregate": agg,
        "chapters": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}  ({len(rows)} chapters)")
    for d, s in agg.items():
        print(f"  {d:<20} mean={s['mean']:.3f}  sd={s['sd']:.3f}  "
              f"range={s['min']:.3f}..{s['max']:.3f}")


if __name__ == "__main__":
    main()
