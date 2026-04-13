from __future__ import annotations

from app.craft.library import CraftLibrary
from app.engine.prompt_renderer import PromptRenderer
from app.planning.schemas import (
    CharacterVoice,
    CraftPlan,
    DramaticPlan,
    EmotionalPlan,
    VoicePermeability,
)
from app.planning.voice import (
    blended_voice_samples_for,
    character_voice_for,
    default_permeability,
)
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.schema import Entity


class CraftPlanner:
    """Craft-layer planner.

    Consumes a ``DramaticPlan`` and an ``EmotionalPlan`` and produces a
    ``CraftPlan`` — a prose blueprint specifying register, temporal structure,
    motifs, narrator instructions, voice permeability, detail principle,
    metaphor profiles, indirection, and a unified prose brief per scene.

    Does NOT write prose; it specifies HOW prose should be constructed.

    Free-indirect grounding (Gap G3): when ``characters`` is supplied, each
    POV character's ``Entity.data["voice"]`` is loaded as a ``CharacterVoice``
    and passed to the template. Per-scene voice permeability is seeded with
    ``default_permeability(narrator, voice)`` so ``bleed_vocabulary`` and
    ``excluded_vocabulary`` come from grounded data, not LLM confabulation.
    Author-curated few-shot samples can be attached to the character entity
    under ``data["blended_voice_samples"]``.
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
        characters: dict[str, Entity] | None = None,
    ) -> CraftPlan:
        """Generate a ``CraftPlan`` translating drama + emotion into a prose blueprint.

        Parameters
        ----------
        dramatic:
            The ``DramaticPlan`` from the dramatic layer.
        emotional:
            The ``EmotionalPlan`` from the emotional layer.
        style_register_id:
            Optional style register id. When provided, the full ``StyleRegister``
            is injected into the user prompt and used as the narrator register
            when seeding voice permeability defaults.
        characters:
            Optional map ``{character_id: Entity}`` for characters referenced
            in the dramatic plan. Used to ground free-indirect-style bleed
            (Gap G3). Each character's voice is read from
            ``entity.data["voice"]`` (schema: ``CharacterVoice``).
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

        # ---- Gap G3: ground free-indirect voice per POV character ----
        characters = characters or {}
        character_voices: dict[str, CharacterVoice] = {}
        blended_samples: dict[str, list[str]] = {}
        scene_permeability_defaults: dict[int, VoicePermeability] = {}

        for scene in dramatic.scenes:
            pov_id = scene.pov_character_id
            if not pov_id:
                continue
            entity = characters.get(pov_id)
            voice = character_voice_for(entity)
            if voice is None:
                continue
            character_voices.setdefault(pov_id, voice)
            samples = blended_voice_samples_for(entity)
            if samples:
                blended_samples.setdefault(pov_id, samples)
            scene_permeability_defaults[scene.scene_id] = default_permeability(
                style_register,
                voice,
                blended_voice_samples=samples,
            )

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
                "character_voices": character_voices,
                "blended_samples": blended_samples,
                "scene_permeability_defaults": scene_permeability_defaults,
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
        plan = OutputParser.parse_json(raw, schema=CraftPlan)

        # Backfill any scene whose LLM-emitted permeability is missing or has
        # empty grounded vocabulary with the grounded defaults. This enforces
        # the invariant that bleed/excluded vocabulary come from character
        # data, not the model.
        for scene in plan.scenes:
            default_vp = scene_permeability_defaults.get(scene.scene_id)
            if default_vp is None:
                continue
            if scene.voice_permeability is None:
                scene.voice_permeability = default_vp
                continue
            vp = scene.voice_permeability
            if not vp.bleed_vocabulary:
                vp.bleed_vocabulary = list(default_vp.bleed_vocabulary)
            if not vp.excluded_vocabulary:
                vp.excluded_vocabulary = list(default_vp.excluded_vocabulary)
            if not vp.blended_voice_samples:
                vp.blended_voice_samples = list(default_vp.blended_voice_samples)

        return plan
