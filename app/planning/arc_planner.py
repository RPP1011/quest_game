from __future__ import annotations

from app.craft.schemas import Structure
from app.engine.prompt_renderer import PromptRenderer
from app.planning.schemas import ArcDirective
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser, ParseError
from app.world.schema import ParallelStatus, QuestArcState
from app.world.state_manager import WorldStateManager


class ArcPlanner:
    """Strategic arc-layer planner.

    Produces an ``ArcDirective`` — theme priorities, plot objectives, character
    arc directions, tension envelope, hooks to plant/pay off.  Does NOT specify
    scenes, dialogue, prose, or craft register; those are handled by the layers
    below.
    """

    def __init__(self, client: InferenceClient, renderer: PromptRenderer) -> None:
        self._client = client
        self._renderer = renderer

    async def plan(
        self,
        *,
        quest_config: dict,
        arc_state: QuestArcState,
        world_snapshot: WorldStateManager,
        structure: Structure,
    ) -> ArcDirective:
        """Generate an ``ArcDirective`` for the current phase.

        Parameters
        ----------
        quest_config:
            Simple dict with keys ``genre``, ``premise``, ``themes`` (list[str]),
            and any additional quest metadata.
        arc_state:
            The persisted ``QuestArcState`` for this arc (phase index, progress,
            tension history).
        world_snapshot:
            Live ``WorldStateManager`` — used to pull current plot threads and
            the most recent narrative summaries.
        structure:
            The ``Structure`` (e.g. three-act) governing the arc.
        """
        # Resolve current phase from arc_state
        phase_index = arc_state.current_phase_index
        phases = structure.phases  # already sorted by position (model_validator)
        current_phase = phases[phase_index] if phase_index < len(phases) else phases[-1]

        # Pull plot threads and up to 5 most recent narrative records
        plot_threads = world_snapshot.list_plot_threads()
        all_narrative = world_snapshot.list_narrative(limit=10_000)
        narrative_summaries = all_narrative[-5:] if all_narrative else []
        active_parallels = world_snapshot.list_parallels(
            statuses=[ParallelStatus.PLANTED, ParallelStatus.SCHEDULED],
        )

        schema = ArcDirective.model_json_schema()

        system_prompt = self._renderer.render(
            "stages/arc/system.j2",
            {"schema": schema},
        )
        user_prompt = self._renderer.render(
            "stages/arc/user.j2",
            {
                "quest_config": quest_config,
                "structure": structure,
                "arc_state": arc_state,
                "current_phase": current_phase,
                "plot_threads": plot_threads,
                "narrative_summaries": narrative_summaries,
                "active_parallels": active_parallels,
            },
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        raw = await self._client.chat_structured(
            messages,
            json_schema=schema,
            schema_name="ArcDirective",
        )

        # ParseError propagates to caller as-is
        return OutputParser.parse_json(raw, schema=ArcDirective)
