"""Promptfoo scorer: fast keyword-based metaphor count.

No LLM call — just regex matching against the imagery families.
Useful as a quick signal alongside the LLM scorer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def get_score(output: str, context: dict) -> dict:
    from app.planning.metaphor_critic import _count_family, IMAGERY_FAMILIES

    counts = {}
    for name, phrases in IMAGERY_FAMILIES.items():
        c = _count_family(output, phrases)
        if c > 0:
            counts[name] = c

    total = sum(counts.values())
    max_per_family = 4
    violations = [f"{k}={v}" for k, v in counts.items() if v > max_per_family]
    score = max(0.0, 1.0 - len(violations) * 0.15)

    return {
        "pass": len(violations) == 0,
        "score": score,
        "reason": f"keyword_total={total}, violations: {', '.join(violations) or 'none'}",
        "componentResults": [
            {
                "pass": count <= max_per_family,
                "score": min(1.0, max_per_family / max(count, 1)),
                "reason": f"{name}: {count} keyword hits",
                "assertion": {"type": "python", "value": f"keyword_{name}"},
            }
            for name, count in sorted(counts.items(), key=lambda x: -x[1])
        ],
    }
