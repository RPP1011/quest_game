"""Report rendering — plain text + JSON."""
from __future__ import annotations

import json
from dataclasses import asdict

from .harness import DimensionStats, Report


def to_dict(report: Report) -> dict:
    return {
        "passages": [asdict(p) for p in report.passages],
        "per_dimension": [asdict(s) for s in report.per_dim],
        "overall": asdict(report.overall),
    }


def to_json(report: Report, *, indent: int = 2) -> str:
    return json.dumps(to_dict(report), indent=indent)


def to_text(report: Report, *, correlation_threshold: float = 0.7) -> str:
    lines: list[str] = []
    lines.append("Calibration Report")
    lines.append("==================")
    lines.append("")
    lines.append(
        f"Overall: MAE={report.overall.mae:.3f}  "
        f"RMSE={report.overall.rmse:.3f}  "
        f"Pearson r={report.overall.pearson:.3f}"
    )
    lines.append("")
    lines.append("Per-dimension")
    lines.append("-------------")
    header = f"{'dimension':<28} {'n':>4} {'MAE':>7} {'RMSE':>7} {'r':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in report.per_dim:
        flag = "  <-- r<thresh" if s.pearson < correlation_threshold else ""
        lines.append(
            f"{s.dimension:<28} {s.n:>4} {s.mae:>7.3f} {s.rmse:>7.3f} "
            f"{s.pearson:>7.3f}{flag}"
        )
    failing = report.failing_correlation(correlation_threshold)
    lines.append("")
    if failing:
        lines.append(
            f"Dimensions with Pearson r < {correlation_threshold}:"
        )
        for s in failing:
            lines.append(f"  - {s.dimension} (r={s.pearson:.3f})")
    else:
        lines.append(f"All dimensions meet r >= {correlation_threshold}.")

    skipped = [p for p in report.passages if p.skipped]
    if skipped:
        lines.append("")
        lines.append(f"Skipped passages: {len(skipped)}")
        for p in skipped[:20]:
            lines.append(f"  - {p.work_id}/{p.passage_id}: {p.skip_reason}")
    return "\n".join(lines)
