from __future__ import annotations

from app.engine.prompt_renderer import PromptRenderer
from app.planning.schemas import DramaticPlan, EmotionalPlan
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.state_manager import WorldStateManager


class EmotionalPlanner:
    """Emotional-layer planner.

    Consumes a ``DramaticPlan`` and specifies HOW each scene should feel —
    entry/exit emotional states, transitions, subtext, and per-character
    internal vs. displayed emotions.  Does NOT specify prose, register,
    temporal structure, or motif placement; those belong to the craft layer.
    """

    def __init__(self, client: InferenceClient, renderer: PromptRenderer) -> None:
        self._client = client
        self._renderer = renderer

    async def plan(
        self,
        *,
        dramatic: DramaticPlan,
        world: WorldStateManager,
        recent_prose: list[str],
    ) -> EmotionalPlan:
        """Generate an ``EmotionalPlan`` for the current update.

        Parameters
        ----------
        dramatic:
            The ``DramaticPlan`` from the dramatic layer — scenes with
            questions, outcomes, beats, and characters present.
        world:
            Live ``WorldStateManager`` — characters, plot threads, narrative.
        recent_prose:
            Last 2 raw prose segments for emotional continuity.
        """
        characters = world.list_entities()

        schema = EmotionalPlan.model_json_schema()

        system_prompt = self._renderer.render(
            "stages/emotional/system.j2",
            {"schema": schema},
        )
        user_prompt = self._renderer.render(
            "stages/emotional/user.j2",
            {
                "dramatic": dramatic,
                "characters": characters,
                "recent_prose": recent_prose[-2:] if recent_prose else [],
            },
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        raw = await self._client.chat_structured(
            messages,
            json_schema=schema,
            schema_name="EmotionalPlan",
        )

        # ParseError propagates to caller as-is
        return OutputParser.parse_json(raw, schema=EmotionalPlan)
