from __future__ import annotations

from typing import TYPE_CHECKING

from app.craft.library import CraftLibrary
from app.craft.schemas import Arc, Structure
from app.engine.prompt_renderer import PromptRenderer
from app.planning.schemas import ArcDirective, DramaticPlan
from app.retrieval import (
    Query,
    QueryFilters,
    SceneShapeRetriever,
)
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.state_manager import WorldStateManager

if TYPE_CHECKING:
    from app.retrieval.foreshadowing_retriever import ForeshadowingRetriever


class DramaticPlanner:
    """Dramatic-layer planner.

    Consumes an ``ArcDirective`` and decides WHAT happens in this update and
    WHY it matters — scenes with dramatic questions, outcomes, and beats.
    Does NOT specify how scenes feel, how prose reads, register, temporal
    structure, or motif placement; those belong to layers below.
    """

    def __init__(
        self,
        client: InferenceClient,
        renderer: PromptRenderer,
        craft_library: CraftLibrary,
    ) -> None:
        self._client = client
        self._renderer = renderer
        self._craft_library = craft_library

    async def plan(
        self,
        *,
        directive: ArcDirective,
        player_action: str,
        world: WorldStateManager,
        arc: Arc,
        structure: Structure,
        recent_tool_ids: list[str] | None = None,
        quest_id: str | None = None,
        scene_retriever: SceneShapeRetriever | None = None,
        foreshadowing_retriever: ForeshadowingRetriever | None = None,
        update_number: int | None = None,
    ) -> DramaticPlan:
        """Generate a ``DramaticPlan`` for the current update.

        Parameters
        ----------
        directive:
            The ``ArcDirective`` from the arc layer — theme priorities, plot
            objectives, tension envelope, hooks, etc.
        player_action:
            The player's most recent action text.
        world:
            Live ``WorldStateManager`` — characters, plot threads, narrative.
        arc:
            The ``Arc`` object from the craft library, used for tool
            recommendations.
        structure:
            The ``Structure`` governing the arc (e.g. three_act).
        recent_tool_ids:
            Tool ids used in recent pipeline turns (for recency penalty).
        """
        # Pull world context
        characters = world.list_entities()
        plot_threads = world.list_plot_threads()
        all_narrative = world.list_narrative(limit=10_000)
        recent_narrative = all_narrative[-2:] if all_narrative else []

        themes: list = []
        if quest_id is not None:
            try:
                themes = world.list_themes(quest_id)
            except Exception:
                themes = []

        reader_state = (
            world.get_reader_state(quest_id) if quest_id is not None else None
        )

        # Gap G7: compute live information asymmetries for prompt context +
        # tool scoring.
        asymmetries: list = []
        ripe_count = 0
        if quest_id is not None:
            try:
                from app.planning.information_asymmetry import (
                    compute_asymmetries,
                    ripe_asymmetry_count,
                )
                asymmetries = compute_asymmetries(world, quest_id)
                ripe_count = ripe_asymmetry_count(asymmetries)
            except Exception:
                asymmetries = []
                ripe_count = 0

        current_scene_id = None
        recommended_tools = self._craft_library.recommend_tools(
            arc, structure, recent_tool_ids=recent_tool_ids, limit=5,
            themes=themes, current_scene_id=current_scene_id,
            updates_since_major_event=(
                reader_state.updates_since_major_event if reader_state else None
            ),
            ripe_asymmetry_count=ripe_count,
        )
        tools_with_examples = []
        for tool in recommended_tools:
            examples = self._craft_library.examples_for_tool(tool.id)[:2]
            tools_with_examples.append({"tool": tool, "examples": examples})

        # Wave 3c: optional scene-shape exemplar retrieval. When a
        # ``scene_retriever`` is wired in, pull k=2 arc-scale scenes
        # whose tension matches the directive's envelope and whose
        # scene_coherence is strong, so the LLM has concrete references
        # for the kind of scene-shape the arc layer is asking for.
        scene_exemplars: list[dict] = []
        if scene_retriever is not None:
            scene_exemplars = await self._retrieve_scene_exemplars(
                scene_retriever, directive
            )

        # Wave 4b: optional ripe-foreshadowing retrieval. When both a
        # ``foreshadowing_retriever`` and an ``update_number`` are in
        # scope, fetch up to k=3 hooks the dramatic layer could now
        # pay off, so the LLM sees candidates for scene-level payoff
        # selection.
        ripe_hooks: list[dict] = []
        if foreshadowing_retriever is not None and update_number is not None:
            ripe_hooks = await self._retrieve_ripe_hooks(
                foreshadowing_retriever, update_number
            )

        schema = DramaticPlan.model_json_schema()

        system_prompt = self._renderer.render(
            "stages/dramatic/system.j2",
            {"schema": schema},
        )
        user_prompt = self._renderer.render(
            "stages/dramatic/user.j2",
            {
                "directive": directive,
                "player_action": player_action,
                "characters": characters,
                "plot_threads": plot_threads,
                "recent_narrative": recent_narrative,
                "tools_with_examples": tools_with_examples,
                "themes": themes,
                "reader_state": reader_state,
                "information_asymmetries": asymmetries,
                "scene_exemplars": scene_exemplars,
                "ripe_hooks": ripe_hooks,
            },
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        raw = await self._client.chat_structured(
            messages,
            json_schema=schema,
            schema_name="DramaticPlan",
        )

        # ParseError propagates to caller as-is
        return OutputParser.parse_json(raw, schema=DramaticPlan)

    # -- Helpers --------------------------------------------------------

    async def _retrieve_scene_exemplars(
        self,
        scene_retriever: SceneShapeRetriever,
        directive: ArcDirective,
    ) -> list[dict]:
        """Pull arc-scale scene exemplars matching the directive envelope.

        Uses the directive's ``tension_range`` as the primary metadata
        filter and a light ``scene_coherence`` floor so we bias toward
        well-executed scenes. Failures are swallowed — retrieval is
        advisory, never load-bearing for planning.
        """
        lo, hi = directive.tension_range
        score_ranges = {
            "tension_execution": (float(lo), float(hi)),
            "scene_coherence": (0.65, 1.0),
        }
        seed_bits = [directive.current_phase, directive.phase_assessment]
        if directive.plot_objectives:
            seed_bits.extend(obj.description for obj in directive.plot_objectives)
        if directive.hooks_to_plant:
            seed_bits.extend(directive.hooks_to_plant)
        seed_text = "\n".join(b for b in seed_bits if b) or None

        query = Query(
            seed_text=seed_text,
            filters=QueryFilters(score_ranges=score_ranges).to_dict(),
        )
        try:
            results = await scene_retriever.retrieve(query, k=2)
        except Exception:
            return []

        exemplars: list[dict] = []
        for r in results:
            # Template shows first ~200 words. Truncation happens here
            # so the template stays presentation-only.
            words = r.text.split()
            preview = " ".join(words[:200])
            if len(words) > 200:
                preview += " ..."
            exemplars.append(
                {
                    "source_id": r.source_id,
                    "work_id": r.metadata.get("work_id"),
                    "scene_id": r.metadata.get("scene_id"),
                    "dramatic_function": r.metadata.get("dramatic_function"),
                    "tension_execution": r.metadata.get("actual_scores", {}).get(
                        "tension_execution"
                    ),
                    "scene_coherence": r.metadata.get("actual_scores", {}).get(
                        "scene_coherence"
                    ),
                    "preview": preview,
                }
            )
        return exemplars

    async def _retrieve_ripe_hooks(
        self,
        foreshadowing_retriever: ForeshadowingRetriever,
        update_number: int,
    ) -> list[dict]:
        """Pull up to k=3 ripe foreshadowing hooks for the current update.

        Retrieval failures are swallowed — the planner runs without
        ``ripe_hooks`` rather than crashing on an advisory signal.
        """
        query = Query(filters={"current_update": int(update_number)})
        try:
            results = await foreshadowing_retriever.retrieve(query, k=3)
        except Exception:
            return []

        hooks: list[dict] = []
        for r in results:
            meta = r.metadata or {}
            hooks.append(
                {
                    "source_id": r.source_id,
                    "description": r.text,
                    "hook_id": meta.get("hook_id"),
                    "status": meta.get("status"),
                    "planted_at_update": meta.get("planted_at_update"),
                    "target_update_min": meta.get("target_update_min"),
                    "target_update_max": meta.get("target_update_max"),
                    "payoff_description": meta.get("payoff_description"),
                    "ripeness_status": meta.get("ripeness_status"),
                }
            )
        return hooks
