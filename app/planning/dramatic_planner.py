from __future__ import annotations

from app.craft.library import CraftLibrary
from app.craft.schemas import Arc, Structure
from app.engine.prompt_renderer import PromptRenderer
from app.planning.schemas import ArcDirective, DramaticPlan
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.state_manager import WorldStateManager


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
