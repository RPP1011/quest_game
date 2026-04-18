"""Foreshadow triple pool — CFPG-style promise tracking with prose verification.

Manages (foreshadow, trigger, payoff) triples as first-class objects.
Trigger predicates are evaluated deterministically against world state.
Prose verification uses an LLM call to confirm hooks are legible in text.
"""
from __future__ import annotations

import math
from pathlib import Path

_VERIFY_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "critics" / "foreshadow_verify.j2"


def evaluate_predicate(pred: dict, state: dict) -> bool:
    """Evaluate a trigger predicate against current world state.

    State dict must contain:
        current_chapter: int
        active_entities: list[str]  — entity ids with ACTIVE status
        present_entities: list[str] — entity ids in current scene
        events: list[str]           — KB events logged so far
    """
    ptype = pred["type"]

    if ptype == "chapter_gte":
        return state["current_chapter"] >= pred["value"]

    if ptype == "entity_active":
        return pred["entity_id"] in state["active_entities"]

    if ptype == "entity_present":
        return pred["entity_id"] in state["present_entities"]

    if ptype == "event_occurred":
        return pred["event"] in state["events"]

    if ptype == "and":
        return all(evaluate_predicate(c, state) for c in pred["children"])

    if ptype == "or":
        return any(evaluate_predicate(c, state) for c in pred["children"])

    raise ValueError(f"Unknown predicate type: {ptype}")


async def verify_prose_reference(
    *, client: "InferenceClient", element_text: str, prose: str,
) -> float:
    """Check if prose contains a legible reference to a narrative element.

    Returns confidence score [0, 1] from logprob on the YES token.
    """
    from jinja2 import Template
    from app.runtime.client import ChatMessage

    template = Template(_VERIFY_PROMPT_PATH.read_text())
    prompt = template.render(element_text=element_text, prose=prose[-3000:])

    result = await client.chat_with_logprobs(
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=1,
        temperature=0.0,
        top_logprobs=5,
    )

    # Extract logprob for YES token
    if result.token_logprobs:
        top = result.token_logprobs[0].top_logprobs
        yes_logprob = top.get("YES", top.get("Yes", top.get("yes", -10.0)))
        no_logprob = top.get("NO", top.get("No", top.get("no", -10.0)))
        # Softmax over YES/NO
        yes_prob = math.exp(yes_logprob)
        no_prob = math.exp(no_logprob)
        total = yes_prob + no_prob
        return yes_prob / total if total > 0 else 0.0

    # Fallback: text match
    content = result.content.strip().upper()
    return 0.9 if content.startswith("YES") else 0.1
