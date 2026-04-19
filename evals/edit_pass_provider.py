"""Promptfoo provider that runs the metaphor edit pass on saved prose.

Takes prose from dataset, classifies it, generates targeted edits,
applies them, returns the edited prose.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """The 'prompt' here is the edit strategy instructions (system prompt).
    The prose comes from context.vars.prose."""
    config = options.get("config", {})
    base_url = config.get("base_url", "http://127.0.0.1:8082")
    prose = context.get("vars", {}).get("prose", "")

    if not prose:
        return {"error": "No prose in vars"}

    try:
        result = asyncio.run(_run_edit_pass(base_url, prompt, prose))
        return result
    except Exception as e:
        return {"error": str(e)}


async def _run_edit_pass(base_url: str, strategy_prompt: str, prose: str) -> dict:
    from app.runtime.client import InferenceClient
    from app.planning.metaphor_critic import classify_metaphors_llm
    from app.engine.typed_edits import detect_metaphor_edits, apply_edits

    client = InferenceClient(base_url=base_url, timeout=120.0, retries=2)

    # Step 1: classify the input prose
    pre_classification = await classify_metaphors_llm(client, prose)

    # Step 2: generate targeted edits
    edits = await detect_metaphor_edits(
        client, prose, pre_classification, max_per_family=3,
    )

    # Step 3: apply edits
    edited = apply_edits(prose, edits) if edits else prose

    return {
        "output": edited,
        "metadata": {
            "edits_applied": len(edits),
            "pre_classification": pre_classification,
        },
    }
