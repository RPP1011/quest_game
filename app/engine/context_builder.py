from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .context_spec import ContextSpec, EntityScope, NarrativeMode
from .prompt_renderer import PromptRenderer
from .token_budget import TokenBudget, estimate_tokens
from app.world.schema import EntityStatus
from app.world.state_manager import WorldStateManager


@dataclass
class AssembledContext:
    system_prompt: str
    user_prompt: str
    token_estimate: int
    manifest: dict[str, Any] = field(default_factory=dict)


class ContextBuilder:
    def __init__(
        self,
        world: WorldStateManager,
        renderer: PromptRenderer,
        budget: TokenBudget,
    ) -> None:
        self._world = world
        self._renderer = renderer
        self._budget = budget

    def build(
        self,
        *,
        spec: ContextSpec,
        stage_name: str,
        templates: dict[str, str],
        extras: dict[str, Any] | None = None,
    ) -> AssembledContext:
        extras = dict(extras or {})
        manifest: dict[str, Any] = {"stage": stage_name, "compression_applied": False}

        entities = self._select_entities(spec)
        relationships = self._world.list_relationships() if spec.include_relationships else []
        rules = self._world.list_rules() if spec.include_rules else []
        plot_threads = self._world.list_plot_threads() if spec.include_plot_threads else []
        recent_summaries = self._recent_narrative(spec)
        recent_prose = self._recent_prose(spec)

        manifest["entities"] = {"included_count": len(entities)}
        manifest["relationships"] = {"included_count": len(relationships)}
        manifest["plot_threads"] = {"included_count": len(plot_threads)}
        manifest["recent_summaries"] = {"included_count": len(recent_summaries)}

        context = {
            "entities": entities,
            "relationships": relationships,
            "rules": rules,
            "plot_threads": plot_threads,
            "recent_summaries": recent_summaries,
            "recent_prose": recent_prose,
            **extras,
        }

        system_prompt = self._renderer.render(templates["system"], context)
        user_prompt = self._renderer.render(templates["user"], context)
        total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)

        # Compression threshold: user-facing content sections (excludes generation headroom
        # and safety margin).  When world+narrative tokens exceed their combined allocation,
        # iteratively trim content to fit.
        content_budget = self._budget.world_state + self._budget.narrative_history
        if total > content_budget:
            manifest["compression_applied"] = True
            # 1. trim summaries
            while recent_summaries and total > self._budget.total // 2:
                recent_summaries = recent_summaries[1:]
                context["recent_summaries"] = recent_summaries
                user_prompt = self._renderer.render(templates["user"], context)
                total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
            # 2. drop relationships
            if total > self._budget.total // 2 and relationships:
                context["relationships"] = []
                user_prompt = self._renderer.render(templates["user"], context)
                total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
                manifest["relationships"]["dropped"] = True
            # 3. entity name-only
            if total > self._budget.total // 2:
                stripped = [
                    e.model_copy(update={"data": {}}) for e in entities
                ]
                context["entities"] = stripped
                user_prompt = self._renderer.render(templates["user"], context)
                total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
                manifest["entities"]["stripped"] = True

        return AssembledContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            token_estimate=total,
            manifest=manifest,
        )

    def _select_entities(self, spec: ContextSpec):
        all_entities = self._world.list_entities()
        if spec.entity_scope == EntityScope.ALL:
            return [e for e in all_entities if e.status != EntityStatus.DESTROYED]
        if spec.entity_scope == EntityScope.ACTIVE:
            return [e for e in all_entities if e.status == EntityStatus.ACTIVE]
        # RELEVANT: active + recently referenced
        def _relevant(e):
            if e.status != EntityStatus.ACTIVE:
                return False
            return True  # v1: include all active; tighten later
        return [e for e in all_entities if _relevant(e)]

    def _recent_narrative(self, spec: ContextSpec):
        """Return NarrativeRecord objects for summary-mode templates."""
        if spec.narrative_mode == NarrativeMode.NONE:
            return []
        records = self._world.list_narrative(limit=max(spec.narrative_window, 1) * 4)
        return records[-spec.narrative_window:]

    def _recent_prose(self, spec: ContextSpec) -> list[str]:
        """Return raw_text strings for full-mode prose continuity."""
        if spec.narrative_mode != NarrativeMode.FULL:
            return []
        records = self._world.list_narrative(limit=max(spec.narrative_window, 1) * 4)
        recent = records[-spec.narrative_window:]
        return [r.raw_text for r in recent if r.raw_text]
