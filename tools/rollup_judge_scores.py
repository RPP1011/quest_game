"""Roll chapter-scale judge scores up to a work-level dim profile.

Reads ``data/calibration/annotations/<work>/chap_*.judge.<label>.json``
and emits:

- ``judge_rollup.<label>.json``: per-chapter table, per-dim aggregate
  (mean, sd, median, min, max), and diff against the work-level
  ``expected`` block in ``data/calibration/manifest.yaml`` when present.
- ``judge_rollup.<label>.md``: human-readable summary.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

ANN_ROOT = Path("data/calibration/annotations")
MANIFEST = Path("data/calibration/manifest.yaml")


def _load_expected(work: str) -> dict[str, float]:
    m = yaml.safe_load(MANIFEST.read_text())
    for w in m.get("works", []):
        if w.get("id") == work:
            return dict(w.get("expected", {}))
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(prog="rollup_judge_scores")
    ap.add_argument("--work", required=True)
    ap.add_argument("--label", default="gemma4")
    args = ap.parse_args()

    root = ANN_ROOT / args.work
    files = sorted(root.glob(f"chap_*.judge.{args.label}.json"))
    if not files:
        raise SystemExit(f"no chapter judge files found for {args.work}/{args.label}")

    rows = []
    dim_set: set[str] = set()
    for f in files:
        d = json.loads(f.read_text())
        scores = {k: v["score"] for k, v in d["scores"].items()}
        dim_set.update(scores.keys())
        rows.append({
            "chapter": d["chapter"],
            "words": d["word_count"],
            "latency_s": d.get("latency_s"),
            "scores": scores,
        })

    dims = sorted(dim_set)
    agg = {}
    for dim in dims:
        vals = [r["scores"].get(dim) for r in rows if dim in r["scores"]]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        agg[dim] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    expected = _load_expected(args.work)
    diffs = {}
    for dim, e in expected.items():
        if dim in agg:
            diffs[dim] = {"expected": e, "observed_mean": agg[dim]["mean"],
                          "delta": agg[dim]["mean"] - e}

    out = {
        "work": args.work,
        "label": args.label,
        "n_chapters": len(rows),
        "dims": dims,
        "aggregate": agg,
        "expected_diff": diffs,
        "chapters": rows,
    }
    out_json = root / f"judge_rollup.{args.label}.json"
    out_json.write_text(json.dumps(out, indent=2))

    lines = [f"# {args.work} — judge rollup ({args.label})\n",
             f"Chapters judged: {len(rows)}\n",
             "## Per-dim aggregate\n",
             "| dim | mean | sd | median | min | max | expected | Δ |",
             "|---|---|---|---|---|---|---|---|"]
    for dim in dims:
        s = agg[dim]
        e = expected.get(dim)
        delta = f"{s['mean'] - e:+.2f}" if e is not None else "—"
        e_str = f"{e:.2f}" if e is not None else "—"
        lines.append(
            f"| {dim} | {s['mean']:.2f} | {s['sd']:.2f} | {s['median']:.2f} | "
            f"{s['min']:.2f} | {s['max']:.2f} | {e_str} | {delta} |"
        )
    lines.append("\n## Per-chapter scores\n")
    lines.append("| ch | words | " + " | ".join(dims) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(dims)) + "|")
    for r in rows:
        cells = [f"{r['scores'].get(d, 0):.2f}" for d in dims]
        lines.append(f"| {r['chapter']} | {r['words']} | " + " | ".join(cells) + " |")

    out_md = root / f"judge_rollup.{args.label}.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_json} and {out_md}")
    for dim in dims:
        s = agg[dim]; e = expected.get(dim)
        diff = f"  Δ={s['mean']-e:+.2f}" if e is not None else ""
        print(f"  {dim:<24} mean={s['mean']:.2f} (sd {s['sd']:.2f}){diff}")


if __name__ == "__main__":
    main()
