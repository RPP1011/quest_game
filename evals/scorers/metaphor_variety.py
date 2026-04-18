"""Promptfoo scorer: LLM-based metaphor variety classification.

Calls our metaphor critic against the writer output and returns a
composite score based on how many imagery families exceed the threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def get_score(output: str, context: dict) -> dict:
    """Synchronous scorer for promptfoo (calls LLM critic)."""
    import asyncio
    from app.runtime.client import InferenceClient
    from app.planning.metaphor_critic import classify_metaphors_llm

    config = context.get("config", {})
    base_url = config.get("base_url", "http://127.0.0.1:8082")

    client = InferenceClient(base_url=base_url, timeout=120.0, retries=2)

    try:
        classification = asyncio.run(classify_metaphors_llm(client, output))
    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"LLM classification failed: {e}",
        }

    families = classification.get("families", {})
    total = classification.get("total_figurative", 0)
    dominant = classification.get("dominant_family", "none")
    dominant_pct = classification.get("dominant_percentage", 0)

    max_per_family = 4
    violations = []
    for name, data in families.items():
        count = data.get("count", 0)
        if count > max_per_family:
            violations.append(f"{name}={count}")

    # Score: 1.0 if no violations, reduced by 0.15 per violation
    score = max(0.0, 1.0 - len(violations) * 0.15)
    passed = len(violations) == 0

    reason_parts = [
        f"total_figurative={total}",
        f"dominant={dominant} ({dominant_pct:.0f}%)",
    ]
    if violations:
        reason_parts.append(f"violations: {', '.join(violations)}")
    else:
        reason_parts.append("all families within limit")

    # Include per-family breakdown
    family_summary = {
        name: data["count"]
        for name, data in sorted(families.items(), key=lambda x: -x[1]["count"])
    }

    return {
        "pass": passed,
        "score": score,
        "reason": "; ".join(reason_parts),
        "componentResults": [
            {
                "pass": data.get("count", 0) <= max_per_family,
                "score": min(1.0, max_per_family / max(data.get("count", 1), 1)),
                "reason": f"{name}: {data.get('count', 0)} instances",
                "assertion": {"type": "python", "value": f"metaphor_family_{name}"},
            }
            for name, data in sorted(
                families.items(), key=lambda x: -x[1].get("count", 0)
            )
        ],
    }
