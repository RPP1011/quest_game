"""Foreshadow triple pool — CFPG-style promise tracking with prose verification.

Manages (foreshadow, trigger, payoff) triples as first-class objects.
Trigger predicates are evaluated deterministically against world state.
Prose verification uses an LLM call to confirm hooks are legible in text.
"""
from __future__ import annotations


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
