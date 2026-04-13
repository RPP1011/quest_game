from __future__ import annotations

from app.engine.prompt_renderer import PromptRenderer
from app.planning.schemas import DramaticPlan, EmotionalPlan
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.schema import EmotionalBeat
from app.world.state_manager import WorldStateManager


def detect_monotony(beats: list[EmotionalBeat], window: int = 3) -> bool:
    """True iff the last ``window`` beats all share ``primary_emotion``.

    A cheap heuristic surfaced to the planner prompt so it can bias toward
    contrast. Not a hard rejection; the planner still decides.
    """
    if window < 2 or len(beats) < window:
        return False
    tail = beats[-window:]
    return len({b.primary_emotion for b in tail}) == 1


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
        recent_beats: list[EmotionalBeat] | None = None,
        monotony_flag: bool = False,
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
                "recent_beats": recent_beats or [],
                "monotony_flag": monotony_flag,
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
