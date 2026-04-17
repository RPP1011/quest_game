"""Arc-skeleton planner (Phase 2 of story-rollout architecture).

Given a picked StoryCandidate and the seeded world, produce an
``ArcSkeleton`` — a chapter-by-chapter outline spanning the candidate's
expected_chapter_count. Each SkeletonChapter carries a POV, location
constraint, dramatic question, required plot beats, target tension,
pre-scheduled DORMANT activations, and theme emphasis.

The skeleton is consulted every tick by the arc/dramatic planners so
chapters don't drift from the committed arc shape.

See docs/superpowers/specs/2026-04-15-story-rollout-architecture.md.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from app.engine.prompt_renderer import PromptRenderer
from app.runtime.client import ChatMessage, InferenceClient
from app.world.schema import (
    ArcSkeleton,
    EntityStatus,
    EntityType,
    HookPlacement,
    SkeletonChapter,
    StoryCandidate,
    ThemeBeat,
)
from app.world.state_manager import WorldStateManager


def _build_schema(
    *, n_chapters: int,
    valid_character_ids: list[str],
    valid_location_ids: list[str],
    valid_thread_ids: list[str],
    valid_entity_ids: list[str],
    valid_theme_ids: list[str],
    valid_hook_ids: list[str],
) -> dict:
    """Structured-output schema for the skeleton call.

    ``n_chapters`` is a soft target — the model is told to produce ~this
    many but the schema allows a range around it so the planner can
    breathe if the story wants 12 instead of 15. Closed-enum constraints
    on id fields prevent hallucination.
    """
    def _enum_or_string(enum: list[str]) -> dict:
        if not enum:
            return {"type": "string"}
        return {"type": "string", "enum": enum}

    def _opt_enum_or_string(enum: list[str]) -> dict:
        # Same as _enum_or_string but allows null.
        if not enum:
            return {"anyOf": [{"type": "string"}, {"type": "null"}]}
        return {"anyOf": [
            {"type": "string", "enum": enum},
            {"type": "null"},
        ]}

    chapter_item: dict[str, Any] = {
        "type": "object",
        "required": [
            "chapter_index", "pov_character_id", "dramatic_question",
            "required_plot_beats", "target_tension",
        ],
        "properties": {
            "chapter_index": {"type": "integer", "minimum": 1},
            "pov_character_id": _opt_enum_or_string(valid_character_ids),
            "location_constraint": _opt_enum_or_string(valid_location_ids),
            "dramatic_question": {"type": "string"},
            "required_plot_beats": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1, "maxItems": 5,
            },
            "target_tension": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "entities_to_surface": {
                "type": "array",
                "items": _enum_or_string(valid_entity_ids),
            },
            "theme_emphasis": {
                "type": "array",
                "items": _enum_or_string(valid_theme_ids),
            },
        },
        "additionalProperties": False,
    }

    hook_item: dict[str, Any] = {
        "type": "object",
        "required": ["hook_id", "paid_off_by_chapter"],
        "properties": {
            "hook_id": _enum_or_string(valid_hook_ids),
            "planted_by_chapter": {"type": "integer", "minimum": 1},
            "paid_off_by_chapter": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
    }

    theme_item: dict[str, Any] = {
        "type": "object",
        "required": ["theme_id", "peak_chapter"],
        "properties": {
            "theme_id": _enum_or_string(valid_theme_ids),
            "peak_chapter": {"type": "integer", "minimum": 1},
            "stance_at_peak": {"type": "string"},
        },
        "additionalProperties": False,
    }

    min_ch = max(5, n_chapters - 5)
    max_ch = n_chapters + 5
    return {
        "type": "object",
        "required": ["chapters", "hook_schedule", "theme_arc"],
        "properties": {
            "chapters": {
                "type": "array",
                "minItems": min_ch, "maxItems": max_ch,
                "items": chapter_item,
            },
            "hook_schedule": {"type": "array", "items": hook_item},
            "theme_arc": {"type": "array", "items": theme_item},
        },
        "additionalProperties": False,
    }


class ArcSkeletonPlanner:
    """Wraps an LLM call that generates an ArcSkeleton for a picked candidate.

    Runs once per pick (re-run on regenerate). The skeleton is persisted
    via WorldStateManager so subsequent pipeline calls can read it as
    directive scaffolding.
    """

    def __init__(self, client: InferenceClient, renderer: PromptRenderer) -> None:
        self._client = client
        self._renderer = renderer

    async def generate(
        self, *, world: WorldStateManager, candidate: StoryCandidate,
    ) -> ArcSkeleton:
        """Generate and persist an ArcSkeleton for the given candidate."""
        entities = [
            e for e in world.list_entities()
            if e.status != EntityStatus.DESTROYED
        ]
        characters = [e for e in entities if e.entity_type == EntityType.CHARACTER]
        locations = [e for e in entities if e.entity_type == EntityType.LOCATION]
        plot_threads = world.list_plot_threads()
        try:
            themes = world.list_themes(candidate.quest_id)
        except Exception:
            themes = []
        try:
            rows = world._conn.execute(
                "SELECT id, description, payoff_target FROM foreshadowing ORDER BY id"
            ).fetchall()
            hooks = [
                {"id": r[0], "description": r[1], "payoff_target": r[2]}
                for r in rows
            ]
        except Exception:
            hooks = []

        # Clamp skeleton length to a tractable range. The candidate's
        # expected_chapter_count is a model estimate and can run long (up to
        # 60); 30 is a pragmatic ceiling for generation cost + coherence.
        target_n = min(30, max(5, candidate.expected_chapter_count))
        schema = _build_schema(
            n_chapters=target_n,
            valid_character_ids=[c.id for c in characters],
            valid_location_ids=[l.id for l in locations],
            valid_thread_ids=[t.id for t in plot_threads],
            valid_entity_ids=[e.id for e in entities],
            valid_theme_ids=[t.id for t in themes],
            valid_hook_ids=[h["id"] for h in hooks],
        )

        system_prompt = self._renderer.render(
            "stages/arc_skeleton/system.j2",
            {"schema": schema, "n_chapters": target_n},
        )
        user_prompt = self._renderer.render(
            "stages/arc_skeleton/user.j2",
            {
                "candidate": candidate,
                "n_chapters": target_n,
                "characters": characters,
                "locations": locations,
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
            schema_name="ArcSkeleton",
            temperature=0.6, max_tokens=16384,
        )
        parsed = json.loads(raw)
        chapter_dicts = parsed.get("chapters", []) or []
        hook_dicts = parsed.get("hook_schedule", []) or []
        theme_dicts = parsed.get("theme_arc", []) or []

        # Renumber chapters by position so we get a clean monotonic 1..N
        # regardless of what the model emitted.
        chapters: list[SkeletonChapter] = []
        for i, c in enumerate(chapter_dicts, start=1):
            c = dict(c)
            c["chapter_index"] = i
            chapters.append(SkeletonChapter(**{
                k: v for k, v in c.items()
                if k in SkeletonChapter.model_fields
            }))

        hooks_out = [
            HookPlacement(**{
                k: v for k, v in h.items() if k in HookPlacement.model_fields
            })
            for h in hook_dicts
        ]
        themes_out = [
            ThemeBeat(**{
                k: v for k, v in t.items() if k in ThemeBeat.model_fields
            })
            for t in theme_dicts
        ]

        skel = ArcSkeleton(
            id=f"sk_{uuid.uuid4().hex[:8]}",
            candidate_id=candidate.id,
            quest_id=candidate.quest_id,
            chapters=chapters,
            theme_arc=themes_out,
            hook_schedule=hooks_out,
        )
        world.save_arc_skeleton(skel)
        return skel


def validate_skeleton_coverage(
    skeleton: ArcSkeleton, candidate: StoryCandidate,
    all_hook_ids: list[str],
) -> list[str]:
    """Return a list of validation issues (empty list = fully valid).

    Checks:
    - every primary thread from the candidate is mentioned in at least one
      chapter's required_plot_beats (soft check: by string containment of
      thread id; we don't require exact tagging)
    - every seeded hook has an entry in hook_schedule (or is explicitly
      absent via empty hook_schedule, which we only allow when there are
      zero seeded hooks)
    """
    issues: list[str] = []
    # Hook coverage
    scheduled = {h.hook_id for h in skeleton.hook_schedule}
    for hid in all_hook_ids:
        if hid not in scheduled:
            issues.append(f"seeded hook {hid!r} has no scheduled payoff")

    # Primary-thread mention (loose: thread id substring in any beat)
    all_beat_text = " ".join(
        b for c in skeleton.chapters for b in c.required_plot_beats
    )
    for tid in candidate.primary_thread_ids:
        if tid not in all_beat_text and tid.replace("pt:", "") not in all_beat_text:
            issues.append(f"primary thread {tid!r} not referenced in any chapter's required_plot_beats")

    return issues
