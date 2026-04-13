from __future__ import annotations

from app.craft.library import CraftLibrary
from app.engine.prompt_renderer import PromptRenderer
from app.planning.schemas import CraftPlan, DramaticPlan, EmotionalPlan
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.schema import Parallel


class CraftPlanner:
    """Craft-layer planner.

    Consumes a ``DramaticPlan`` and an ``EmotionalPlan`` and produces a
    ``CraftPlan`` — a prose blueprint specifying register, temporal structure,
    motifs, narrator instructions, voice permeability, detail principle,
    metaphor profiles, indirection, and a unified prose brief per scene.

    Does NOT write prose; it specifies HOW prose should be constructed.
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
        dramatic: DramaticPlan,
        emotional: EmotionalPlan,
        style_register_id: str | None = None,
        active_parallels: list[Parallel] | None = None,
    ) -> CraftPlan:
        """Generate a ``CraftPlan`` translating drama + emotion into a prose blueprint.

        Parameters
        ----------
        dramatic:
            The ``DramaticPlan`` from the dramatic layer — scenes with questions,
            outcomes, beats, dramatic functions, and tools used.
        emotional:
            The ``EmotionalPlan`` from the emotional layer — entry/exit states,
            transitions, primary emotion, surface_vs_depth, character emotions.
        style_register_id:
            Optional style register id. When provided, the full ``StyleRegister``
            (including voice samples) is injected into the user prompt.
        """
        # Resolve style register if requested
        style_register = None
        if style_register_id is not None:
            style_register = self._craft_library.style(style_register_id)

        # Collect all tool ids mentioned across dramatic plan
        all_tool_ids: set[str] = set(
            ts.tool_id for ts in dramatic.tools_selected
        )
        for scene in dramatic.scenes:
            all_tool_ids.update(scene.tools_used)

        # Build tool examples dict: tool_id → up to 2 examples
        tool_examples: dict[str, list] = {}
        for tool_id in sorted(all_tool_ids):
            examples = self._craft_library.examples_for_tool(tool_id)[:2]
            if examples:
                tool_examples[tool_id] = examples

        # Build emotional scene lookup for the template
        emotional_by_scene: dict[int, object] = {
            s.scene_id: s for s in emotional.scenes
        }

        schema = CraftPlan.model_json_schema()

        system_prompt = self._renderer.render(
            "stages/craft/system.j2",
            {"schema": schema},
        )
        user_prompt = self._renderer.render(
            "stages/craft/user.j2",
            {
                "dramatic": dramatic,
                "emotional": emotional,
                "emotional_by_scene": emotional_by_scene,
                "style_register": style_register,
                "tool_examples": tool_examples,
                "active_parallels": active_parallels or [],
            },
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        raw = await self._client.chat_structured(
            messages,
            json_schema=schema,
            schema_name="CraftPlan",
        )

        # ParseError propagates to caller as-is
        return OutputParser.parse_json(raw, schema=CraftPlan)
