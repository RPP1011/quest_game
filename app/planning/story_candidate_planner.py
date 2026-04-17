"""Story-candidate planner (Phase 1 of story-rollout architecture).

Given a seed's world state, produce N candidate story arcs. Each candidate
commits to a specific arc emphasis — which plot threads are primary, which
character is protagonist, which themes drive the emphasis. The player picks
one before chapter generation starts.

See docs/superpowers/specs/2026-04-15-story-rollout-architecture.md.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from app.engine.prompt_renderer import PromptRenderer
from app.runtime.client import ChatMessage, InferenceClient
from app.world.schema import (
    EntityStatus,
    EntityType,
    StoryCandidate,
    StoryCandidateStatus,
)
from app.world.state_manager import WorldStateManager


def _build_schema(n: int, valid_thread_ids: list[str],
                  valid_character_ids: list[str],
                  valid_theme_ids: list[str]) -> dict:
    """Structured-output schema for the candidates call.

    Constrains thread / character / theme ids to closed enums so the
    model can't hallucinate new ones. ``n`` is enforced via minItems and
    maxItems on the candidates array.
    """
    candidate_item: dict[str, Any] = {
        "type": "object",
        "required": [
            "title", "synopsis",
            "primary_thread_ids", "protagonist_character_id",
            "emphasized_theme_ids", "climax_description",
            "expected_chapter_count",
        ],
        "properties": {
            "title": {"type": "string"},
            "synopsis": {"type": "string"},
            "primary_thread_ids": {
                "type": "array",
                "items": {"type": "string"} if not valid_thread_ids
                else {"type": "string", "enum": valid_thread_ids},
                "minItems": 1,
            },
            "secondary_thread_ids": {
                "type": "array",
                "items": {"type": "string"} if not valid_thread_ids
                else {"type": "string", "enum": valid_thread_ids},
            },
            "protagonist_character_id": (
                {"type": "string", "enum": valid_character_ids}
                if valid_character_ids else {"type": "string"}
            ),
            "emphasized_theme_ids": {
                "type": "array",
                "items": {"type": "string"} if not valid_theme_ids
                else {"type": "string", "enum": valid_theme_ids},
            },
            "climax_description": {"type": "string"},
            "expected_chapter_count": {
                "type": "integer", "minimum": 5, "maximum": 60,
            },
        },
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": n, "maxItems": n,
                "items": candidate_item,
            },
        },
        "additionalProperties": False,
    }


class StoryCandidatePlanner:
    """Wraps an LLM call that generates N StoryCandidate rows from a seed.

    The planner runs once per quest — after seed load, before chapter
    generation. Candidates are persisted via ``WorldStateManager`` so
    subsequent picks and pipeline reads can reference them.
    """

    def __init__(self, client: InferenceClient, renderer: PromptRenderer) -> None:
        self._client = client
        self._renderer = renderer

    async def generate(
        self, *, world: WorldStateManager, quest_id: str,
        quest_config: dict | None = None, n: int = 3,
    ) -> list[StoryCandidate]:
        """Generate N candidates and persist them."""
        entities = [
            e for e in world.list_entities()
            if e.status != EntityStatus.DESTROYED
        ]
        characters = [e for e in entities if e.entity_type == EntityType.CHARACTER]
        plot_threads = world.list_plot_threads()
        try:
            themes = world.list_themes(quest_id)
        except Exception:
            themes = []
        # Load foreshadowing hooks directly via SQL (no list_foreshadowing API)
        try:
            rows = world._conn.execute(
                "SELECT id, description FROM foreshadowing ORDER BY id"
            ).fetchall()
            hooks = [{"id": r[0], "description": r[1]} for r in rows]
        except Exception:
            hooks = []

        valid_thread_ids = [t.id for t in plot_threads]
        valid_character_ids = [c.id for c in characters]
        valid_theme_ids = [t.id for t in themes]

        schema = _build_schema(
            n=n,
            valid_thread_ids=valid_thread_ids,
            valid_character_ids=valid_character_ids,
            valid_theme_ids=valid_theme_ids,
        )

        cfg = quest_config or {}
        system_prompt = self._renderer.render(
            "stages/story_candidate/system.j2",
            {"schema": schema, "n": n},
        )
        user_prompt = self._renderer.render(
            "stages/story_candidate/user.j2",
            {
                "n": n,
                "genre": cfg.get("genre", ""),
                "premise": cfg.get("premise", ""),
                "characters": characters,
                "plot_threads": plot_threads,
                "themes": themes,
                "foreshadowing": hooks,
            },
        )
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            json_schema=schema,
            schema_name="StoryCandidates",
            temperature=0.7, max_tokens=4096,
        )
        parsed = json.loads(raw)
        raw_candidates = parsed.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates is not a list")

        out: list[StoryCandidate] = []
        for i, c in enumerate(raw_candidates[:n]):
            cid = f"sc_{uuid.uuid4().hex[:8]}"
            cand = StoryCandidate(
                id=cid,
                quest_id=quest_id,
                title=str(c.get("title", f"Candidate {i+1}")),
                synopsis=str(c.get("synopsis", "")),
                primary_thread_ids=list(c.get("primary_thread_ids", []) or []),
                secondary_thread_ids=list(c.get("secondary_thread_ids", []) or []),
                protagonist_character_id=c.get("protagonist_character_id"),
                emphasized_theme_ids=list(c.get("emphasized_theme_ids", []) or []),
                climax_description=str(c.get("climax_description", "")),
                expected_chapter_count=int(c.get("expected_chapter_count", 15)),
                status=StoryCandidateStatus.DRAFT,
            )
            world.add_story_candidate(cand)
            out.append(cand)
        return out
