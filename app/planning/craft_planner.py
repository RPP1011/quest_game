from __future__ import annotations

from app.craft.library import CraftLibrary
from app.engine.prompt_renderer import PromptRenderer
from app.planning.motives import (
    UnconsciousMotive,
    pick_primary_motive,
    unconscious_motives_for,
)
from app.planning.schemas import (
    CraftPlan,
    DramaticPlan,
    EmotionalPlan,
    IndirectionInstruction,
)
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.state_manager import (
    EntityNotFoundError,
    WorldStateManager,
)


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
        world: WorldStateManager | None = None,
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
        world:
            Optional world state. When provided, the planner loads each POV
            character's active unconscious motives (G13) and renders them into
            the user prompt so the LLM derives ``IndirectionInstruction`` from
            stored motives rather than inventing new ones. Empty or generic
            indirection entries are backfilled from the stored motive after
            parsing.
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

        # G13: load per-POV-character unconscious motives from world state.
        # Keyed by scene_id -> list[UnconsciousMotive].
        motives_by_scene: dict[int, list[UnconsciousMotive]] = {}
        if world is not None:
            for scene in dramatic.scenes:
                pov = scene.pov_character_id
                if not pov:
                    continue
                try:
                    entity = world.get_entity(pov)
                except EntityNotFoundError:
                    continue
                ms = unconscious_motives_for(entity)
                if ms:
                    motives_by_scene[scene.scene_id] = ms

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
                "motives_by_scene": motives_by_scene,
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

        # G13 backfill: for each scene with grounded motives, ensure the
        # indirection list has a concrete entry derived from the stored
        # motive. If the LLM emitted nothing (or an empty / generic stub)
        # for the POV character, synthesize one from the primary motive.
        if motives_by_scene:
            pov_by_scene = {s.scene_id: s.pov_character_id for s in dramatic.scenes}
            for scene in plan.scenes:
                ms = motives_by_scene.get(scene.scene_id)
                if not ms:
                    continue
                pov = pov_by_scene.get(scene.scene_id)
                if not pov:
                    continue
                primary = pick_primary_motive(ms)
                if primary is None:
                    continue
                _backfill_indirection(scene, pov, primary)

        return plan


def _is_empty_or_generic(instr: IndirectionInstruction) -> bool:
    """Return True if an indirection entry is empty or generic enough to
    warrant replacement with stored-motive content."""
    if not instr.unconscious_motive.strip():
        return True
    if not instr.what_not_to_say and not instr.surface_manifestations:
        return True
    return False


def _backfill_indirection(
    scene,
    pov_character_id: str,
    motive: UnconsciousMotive,
) -> None:
    """Backfill scene.indirection from a stored ``UnconsciousMotive``.

    Rules:
      * If no entry exists for ``pov_character_id``, append one built from
        ``motive``.
      * If an entry exists but is empty / generic, replace its grounded
        fields (``unconscious_motive``, ``surface_manifestations``,
        ``detail_tells``, ``what_not_to_say``) with the stored values,
        preserving any ``reader_should_infer`` the LLM provided.
    """
    grounded = IndirectionInstruction(
        character_id=pov_character_id,
        unconscious_motive=motive.motive,
        surface_manifestations=list(motive.surface_manifestations),
        detail_tells=list(motive.detail_tells),
        what_not_to_say=list(motive.what_not_to_say),
        reader_should_infer=(
            f"The reader should sense that {pov_character_id}'s behaviour is "
            f"shaped by an unnamed motive, without the narration ever naming it."
        ),
    )

    existing_idx = next(
        (i for i, instr in enumerate(scene.indirection)
         if instr.character_id == pov_character_id),
        None,
    )
    if existing_idx is None:
        scene.indirection.append(grounded)
        return

    existing = scene.indirection[existing_idx]
    if _is_empty_or_generic(existing):
        # Preserve reader_should_infer if the LLM provided something concrete.
        reader_should_infer = existing.reader_should_infer or grounded.reader_should_infer
        scene.indirection[existing_idx] = IndirectionInstruction(
            character_id=pov_character_id,
            unconscious_motive=grounded.unconscious_motive,
            surface_manifestations=grounded.surface_manifestations,
            detail_tells=grounded.detail_tells,
            what_not_to_say=grounded.what_not_to_say,
            reader_should_infer=reader_should_infer,
        )
