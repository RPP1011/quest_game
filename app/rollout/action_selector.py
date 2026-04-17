"""Profile-driven action selection for virtual-player rollouts.

Given a profile and a list of suggested choices (from a dramatic plan's
``suggested_choices``), pick the one a player of this profile would
plausibly choose. Implemented as a single short structured LLM call;
returns the chosen index and a rationale.

Falls back to ``0`` (the first option) if the LLM call fails — rollouts
should never block on a failed selection.
"""
from __future__ import annotations

import json
from typing import Any

from app.runtime.client import ChatMessage, InferenceClient

from .profiles import VirtualPlayerProfile


def _build_schema(n_choices: int) -> dict:
    return {
        "type": "object",
        "required": ["chosen_index", "rationale"],
        "properties": {
            "chosen_index": {
                "type": "integer", "minimum": 0, "maximum": max(0, n_choices - 1),
            },
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }


def _format_choice(idx: int, choice: Any) -> str:
    if isinstance(choice, str):
        return f"[{idx}] {choice}"
    title = choice.get("title", "") or choice.get("description", "")
    desc = choice.get("description", "")
    tags = choice.get("tags", []) or []
    line = f"[{idx}] {title}"
    if desc and desc != title:
        line += f" — {desc}"
    if tags:
        line += f" (tags: {', '.join(tags)})"
    return line


async def select_action(
    *,
    client: InferenceClient,
    profile: VirtualPlayerProfile,
    choices: list[Any],
    recent_prose_tail: str = "",
) -> tuple[int, str]:
    """Pick an index into ``choices`` that matches the profile.

    Parameters
    ----------
    client:
        Inference client.
    profile:
        The VirtualPlayerProfile whose rubric drives the choice.
    choices:
        A list of suggested choices (string or dict with title/description).
    recent_prose_tail:
        The last ~500 chars of the chapter's prose, for context.

    Returns
    -------
    (chosen_index, rationale) — index is always in [0, len(choices)-1].
    """
    if not choices:
        return (0, "no choices available; defaulting to 0")

    system_prompt = (
        "You are a synthetic reader simulating a particular player's "
        "action-selection style. Follow the rubric strictly. Return JSON "
        "only matching the schema."
    )
    choice_lines = "\n".join(_format_choice(i, c) for i, c in enumerate(choices))
    user_prompt = (
        f"## Player profile: {profile.id}\n"
        f"{profile.description}\n\n"
        f"## Rubric\n{profile.action_selection_rubric}\n\n"
        + (f"## Recent prose (context)\n...{recent_prose_tail}\n\n"
           if recent_prose_tail else "")
        + f"## Choices\n{choice_lines}\n\n"
        "Pick exactly one. Return `chosen_index` (0-based) and a "
        "one-sentence `rationale` grounded in the rubric."
    )

    schema = _build_schema(len(choices))
    try:
        raw = await client.chat_structured(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            json_schema=schema,
            schema_name="ActionSelection",
            temperature=0.4, max_tokens=512,
        )
        parsed = json.loads(raw)
        idx = int(parsed.get("chosen_index", 0))
        if idx < 0 or idx >= len(choices):
            idx = 0
        rationale = str(parsed.get("rationale", ""))
        return (idx, rationale)
    except Exception as e:
        return (0, f"fallback on error: {e}")
