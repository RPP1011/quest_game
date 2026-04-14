"""Day 10 analysis: read run_log.jsonl and render ASCII summary.

Usage::

    uv run python tools/analyze_stress.py [path/to/run_log.jsonl]

Prints six blocks to stdout:
  1. Run config (from the first ``_meta`` row).
  2. Per-dim score-over-time ASCII chart, one line per dim.
  3. Latency p50/p95 per 10-update bin.
  4. Consistency-flag growth (cumulative + per-bin).
  5. Retrieval hit-rate over time (per-bin means).
  6. Entity / narrative / embedding growth.
  7. Totals: wall-clock, tokens, fallback rate.

Everything is text — no matplotlib or stdout figures. The output is
suitable for pasting straight into a Markdown doc.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="analyze_stress")
    ap.add_argument("log", type=Path, nargs="?",
                    default=ROOT / "data" / "stress" / "run_log.jsonl")
    ap.add_argument("--bin-size", type=int, default=10,
                    help="Updates per summary bin.")
    return ap.parse_args()


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            else:
                rows.append(obj)
    return meta, rows


def _mean(xs: list[float]) -> float:
    return st.fmean(xs) if xs else float("nan")


def _p(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = max(0, min(len(xs_sorted) - 1, int(round((len(xs_sorted) - 1) * q))))
    return xs_sorted[k]


def _ascii_line(values: list[float], width: int = 40, lo: float = 0.0,
                hi: float = 1.0) -> str:
    """Render a single row of ASCII sparklines for ``values``.

    Empty / NaN values render as `-`. Otherwise we scale into [lo, hi]
    and pick one of 8 block chars.
    """
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # 9 chars incl. space
    out = []
    span = hi - lo if hi > lo else 1.0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append("-")
            continue
        ratio = max(0.0, min(1.0, (float(v) - lo) / span))
        idx = int(round(ratio * (len(blocks) - 1)))
        out.append(blocks[idx])
    return "".join(out)


def bin_indices(n_rows: int, bin_size: int) -> list[tuple[int, int]]:
    """Return [(lo, hi_exclusive), ...] bin boundaries 0-indexed."""
    out: list[tuple[int, int]] = []
    i = 0
    while i < n_rows:
        out.append((i, min(i + bin_size, n_rows)))
        i += bin_size
    return out


def per_dim_over_time(rows: list[dict]) -> dict[str, list[float]]:
    """Collect each dim's score across updates (NaN when missing)."""
    dims_seen: set[str] = set()
    for r in rows:
        dims_seen.update((r.get("dimension_scores") or {}).keys())
    out: dict[str, list[float]] = {}
    for d in sorted(dims_seen):
        out[d] = [
            (r.get("dimension_scores") or {}).get(d, float("nan"))
            for r in rows
        ]
    return out


def latencies(rows: list[dict]) -> list[float]:
    return [float(r.get("wall_clock_seconds", 0.0) or 0.0) for r in rows]


def consistency_count(r: dict) -> int:
    cf = r.get("consistency_flags") or {}
    return int(cf.get("errors", 0))


def fallback_count(r: dict) -> int:
    cf = r.get("consistency_flags") or {}
    return int(cf.get("fallbacks", 0))


def retrieval_hits(r: dict, kind: str) -> int:
    retr = r.get("retrieval") or {}
    return int(retr.get(f"{kind}_hits", 0))


def retrieval_calls(r: dict, kind: str) -> int:
    retr = r.get("retrieval") or {}
    return int(retr.get(f"{kind}_calls", 0))


def render(meta: dict, rows: list[dict], bin_size: int) -> str:
    out: list[str] = []
    n = len(rows)
    out.append("## Run config")
    if meta:
        out.append(f"- model: {meta.get('model')}")
        out.append(f"- n_candidates: {meta.get('n_candidates')}")
        out.append(f"- updates_target: {meta.get('updates_target')}")
        out.append(f"- scoring: {meta.get('scoring')} | llm_judge: {meta.get('llm_judge')}")
        out.append(f"- seed_quest: {meta.get('seed_quest')}")
    out.append(f"- rows_collected: {n}")
    out.append("")

    # ----- top-line numbers -----
    latencies_all = latencies(rows)
    outcomes = [r.get("outcome", "unknown") for r in rows]
    n_committed = sum(1 for o in outcomes if o == "committed")
    n_flagged = sum(1 for o in outcomes if o == "flagged_qm")
    n_fallback = sum(1 for o in outcomes if o == "fallback")
    tok_prompt = sum(int(r.get("prompt_tokens", 0) or 0) for r in rows)
    tok_comp = sum(int(r.get("completion_tokens", 0) or 0) for r in rows)
    wall = sum(latencies_all)
    out.append("## Top-line")
    out.append(f"- committed: {n_committed}/{n}")
    out.append(f"- flagged_qm: {n_flagged}/{n}")
    out.append(f"- fallback (pipeline crash): {n_fallback}/{n}")
    out.append(f"- wall-clock total: {wall:.1f}s ({wall/60:.1f} min)")
    if n:
        out.append(f"- wall-clock per update (mean): {wall/n:.1f}s")
    out.append(f"- latency p50 / p95 (s): {_p(latencies_all, 0.5):.1f} / "
               f"{_p(latencies_all, 0.95):.1f}")
    out.append(f"- tokens: prompt={tok_prompt}  completion={tok_comp}")
    out.append("")

    if not rows:
        return "\n".join(out)

    # ----- per-dim score trajectory -----
    dims = per_dim_over_time(rows)
    out.append("## Per-dim score over time (0.0 ▁..▇ 1.0)")
    out.append(f"legend: one block = 1 update; '-' = missing (crash or no scoring)")
    # First / last bin means to show drift at a glance.
    for name, series in dims.items():
        first_bin = [x for x in series[:bin_size] if not math.isnan(x)]
        last_bin = [x for x in series[-bin_size:] if not math.isnan(x)]
        first_m = _mean(first_bin) if first_bin else float("nan")
        last_m = _mean(last_bin) if last_bin else float("nan")
        delta = (last_m - first_m) if (not math.isnan(first_m)
                                       and not math.isnan(last_m)) else float("nan")
        sparkline = _ascii_line(series)
        out.append(
            f"  {name:28s} {sparkline}  "
            f"first={first_m:.2f} last={last_m:.2f} Δ={delta:+.2f}"
        )
    out.append("")

    # ----- latency per bin -----
    out.append("## Latency by bin")
    out.append(f"bin_size={bin_size} updates")
    out.append(f"  {'bin':>8} {'n':>4} {'p50':>6} {'p95':>6} {'mean':>6}")
    for lo, hi in bin_indices(n, bin_size):
        seg = latencies_all[lo:hi]
        if not seg:
            continue
        out.append(
            f"  {lo+1:>4}-{hi:>3} {len(seg):>4} "
            f"{_p(seg, 0.5):>6.1f} {_p(seg, 0.95):>6.1f} {_mean(seg):>6.1f}"
        )
    out.append("")

    # ----- consistency flag growth -----
    cons = [consistency_count(r) for r in rows]
    falls = [fallback_count(r) for r in rows]
    cum_cons = []
    running = 0
    for c in cons:
        running += c
        cum_cons.append(running)
    out.append("## Consistency flags (errors raised but not blocking)")
    out.append(f"  total errors: {sum(cons)}")
    out.append(f"  total fallbacks: {sum(falls)}")
    out.append(f"  per-bin mean errors:")
    for lo, hi in bin_indices(n, bin_size):
        seg = cons[lo:hi]
        if not seg:
            continue
        out.append(
            f"    {lo+1:>4}-{hi:>3} mean_errors={_mean(seg):.2f} "
            f"cum_end={cum_cons[hi-1]}"
        )
    # Collect kind frequencies across all rows.
    kind_counts: dict[str, int] = {}
    for r in rows:
        for k in (r.get("consistency_flags") or {}).get("kinds", []) or []:
            kind_counts[k] = kind_counts.get(k, 0) + 1
    if kind_counts:
        out.append(f"  top error kinds:")
        for k, v in sorted(kind_counts.items(), key=lambda x: -x[1])[:8]:
            out.append(f"    {v:>4} {k}")
    out.append("")

    # ----- retrieval hit rates -----
    out.append("## Retrieval activity")
    out.append("  kind            bin    mean_calls  mean_hits")
    for kind in ("passage", "quest", "voice", "motif", "foreshadowing"):
        for lo, hi in bin_indices(n, bin_size):
            calls = [retrieval_calls(r, kind) for r in rows[lo:hi]]
            hits = [retrieval_hits(r, kind) for r in rows[lo:hi]]
            if not calls:
                continue
            out.append(
                f"  {kind:14s} {lo+1:>3}-{hi:>3}   {_mean(calls):>6.2f}     "
                f"{_mean(hits):>6.2f}"
            )
    out.append("")

    # ----- world-state growth -----
    out.append("## World state growth (end-of-update snapshot)")
    out.append(f"  {'update':>6} {'entities':>9} {'narrative':>10} "
               f"{'embeddings':>11} {'threads':>8}")
    # Sample a few rows: first, mid, last.
    sample_indices = []
    if n >= 1:
        sample_indices.append(0)
    if n >= 5:
        sample_indices.extend([n // 4, n // 2, 3 * n // 4])
    sample_indices.append(n - 1)
    sample_indices = sorted(set(sample_indices))
    for i in sample_indices:
        ws = rows[i].get("world_state") or {}
        out.append(
            f"  {rows[i].get('update_number', i+1):>6} "
            f"{ws.get('entities', -1):>9} "
            f"{ws.get('narrative_records', -1):>10} "
            f"{ws.get('narrative_embeddings', -1):>11} "
            f"{ws.get('plot_threads', -1):>8}"
        )
    out.append("")

    # ----- context-token growth -----
    ctx_tokens = [int(r.get("context_tokens", 0) or 0) for r in rows]
    out.append("## Context-token trajectory")
    max_ctx = max(ctx_tokens) if ctx_tokens else 0
    out.append(
        f"  {_ascii_line([c / max(max_ctx, 1) for c in ctx_tokens], lo=0.0, hi=1.0)}"
    )
    out.append(f"  first={ctx_tokens[0] if ctx_tokens else 0}  "
               f"last={ctx_tokens[-1] if ctx_tokens else 0}  "
               f"max={max_ctx}")
    out.append("")

    return "\n".join(out)


def main() -> None:
    args = parse_args()
    if not args.log.is_file():
        print(f"[error] no log at {args.log}", file=sys.stderr)
        sys.exit(2)
    meta, rows = load_rows(args.log)
    print(render(meta, rows, args.bin_size))


if __name__ == "__main__":
    main()
