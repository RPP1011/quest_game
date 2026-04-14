from __future__ import annotations

from app.craft.library import CraftLibrary
from app.craft.schemas import Narrator
from app.engine.prompt_renderer import PromptRenderer
from app.planning.motives import (
    UnconsciousMotive,
    pick_primary_motive,
    unconscious_motives_for,
)
from app.planning.schemas import (
    CharacterMetaphorProfile,
    CharacterVoice,
    CraftPlan,
    DetailPrinciple,
    DramaticPlan,
    EmotionalPlan,
    IndirectionInstruction,
    MetaphorProfile,
    PerceptualProfile,
    VoicePermeability,
)
from app.planning.metaphor import (
    character_metaphor_profile_for,
    default_metaphor_profile,
)
from app.planning.perception import (
    default_detail_principle,
    perceptual_profile_for,
)
from app.planning.voice import (
    blended_voice_samples_for,
    character_voice_for,
    default_permeability,
)
from typing import TYPE_CHECKING

from app.retrieval import MotifRetriever, Query

if TYPE_CHECKING:
    from app.retrieval.foreshadowing_retriever import ForeshadowingRetriever
from app.runtime.client import ChatMessage, InferenceClient
from app.world.output_parser import OutputParser
from app.world.schema import Entity, Parallel
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
        narrator: Narrator | None = None,
        active_parallels: list[Parallel] | None = None,
        characters: dict[str, Entity] | None = None,
        active_motifs: list[dict] | None = None,
        world: WorldStateManager | None = None,
        motif_retriever: MotifRetriever | None = None,
        foreshadowing_retriever: "ForeshadowingRetriever | None" = None,
        update_number: int | None = None,
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
            (Gap G3), detail perception (G9), and metaphor domains (G10).
        world:
            Optional world state. When provided, the planner loads each POV
            character's active unconscious motives (G13) and renders them
            into the user prompt so the LLM derives ``IndirectionInstruction``
            from stored motives rather than inventing new ones.
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

        # ---- Gaps G9/G10: ground detail and metaphor per POV character ----
        perceptual_profiles: dict[str, PerceptualProfile] = {}
        metaphor_profiles_persistent: dict[str, CharacterMetaphorProfile] = {}
        scene_detail_defaults: dict[int, DetailPrinciple] = {}
        scene_metaphor_defaults: dict[int, MetaphorProfile] = {}

        for scene in dramatic.scenes:
            pov_id = scene.pov_character_id
            if not pov_id:
                continue
            entity = characters.get(pov_id)
            voice = character_voice_for(entity)
            if voice is not None:
                character_voices.setdefault(pov_id, voice)
                samples = blended_voice_samples_for(entity)
                if samples:
                    blended_samples.setdefault(pov_id, samples)
                scene_permeability_defaults[scene.scene_id] = default_permeability(
                    style_register,
                    voice,
                    blended_voice_samples=samples,
                )

            # The emotional plan tells us what the scene feels like —
            # used to activate emotion-triggered preoccupations and
            # metaphor source domains.
            emo = emotional_by_scene.get(scene.scene_id)
            primary_emotion = getattr(emo, "primary_emotion", None)
            secondary_emotion = getattr(emo, "secondary_emotion", None)

            perc = perceptual_profile_for(entity)
            if perc is not None:
                perceptual_profiles.setdefault(pov_id, perc)
                scene_detail_defaults[scene.scene_id] = default_detail_principle(
                    pov_id,
                    perc,
                    primary_emotion=primary_emotion,
                    secondary_emotion=secondary_emotion,
                )

            mprof = character_metaphor_profile_for(entity)
            if mprof is not None:
                metaphor_profiles_persistent.setdefault(pov_id, mprof)
                scene_metaphor_defaults[scene.scene_id] = default_metaphor_profile(
                    pov_id,
                    mprof,
                    primary_emotion=primary_emotion,
                    secondary_emotion=secondary_emotion,
                )

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

        due_motifs: list[dict] = []
        if motif_retriever is not None and update_number is not None:
            due_motifs = await self._retrieve_due_motifs(
                motif_retriever, update_number
            )

        ripe_hooks: list[dict] = []
        if foreshadowing_retriever is not None and update_number is not None:
            ripe_hooks = await self._retrieve_ripe_hooks(
                foreshadowing_retriever, update_number
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
                "narrator": narrator,
                "active_parallels": active_parallels or [],
                "character_voices": character_voices,
                "blended_samples": blended_samples,
                "scene_permeability_defaults": scene_permeability_defaults,
                "active_motifs": active_motifs or [],
                "due_motifs": due_motifs,
                "perceptual_profiles": perceptual_profiles,
                "metaphor_profiles_persistent": metaphor_profiles_persistent,
                "scene_detail_defaults": scene_detail_defaults,
                "scene_metaphor_defaults": scene_metaphor_defaults,
                "motives_by_scene": motives_by_scene,
                "ripe_hooks": ripe_hooks,
            },
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        # Day 11: weak models occasionally truncate the structured-output
        # mid-scene (xgrammar can stop early on long deeply-nested
        # objects). One in-band retry catches those cases — much cheaper
        # than letting _retry_with_critic do it because we keep the
        # whole context+schema setup intact. ParseError propagates if
        # the second attempt also fails (caller falls back to stub).
        # ``max_tokens`` is kept conservative so prompt + completion
        # stays under the 16k model_max_len even with the full
        # retriever pile-on; bumping to 12k caused 400 Bad Request
        # responses on long-context updates.
        from app.world.output_parser import ParseError
        raw = await self._client.chat_structured(
            messages,
            json_schema=schema,
            schema_name="CraftPlan",
            max_tokens=6144,
        )
        try:
            plan = OutputParser.parse_json(raw, schema=CraftPlan)
        except ParseError:
            raw = await self._client.chat_structured(
                messages,
                json_schema=schema,
                schema_name="CraftPlan",
                max_tokens=6144,
            )
            plan = OutputParser.parse_json(raw, schema=CraftPlan)

        # Day 11: realign craft scene_ids (and brief scene_ids) to
        # dramatic scene_ids by position. Small models routinely
        # collapse all emitted scene_ids to a single literal value
        # (``scene_id: 42`` was the canonical Day-10 fingerprint) or
        # produce more/fewer scenes than the dramatic plan declared,
        # which trips the cross-layer scene-coverage critic. Aligning
        # by position (and truncating extras) keeps the invariant.
        dramatic_ids = [s.scene_id for s in dramatic.scenes]
        if dramatic_ids:
            n = len(dramatic_ids)
            if len(plan.scenes) > n:
                plan.scenes = plan.scenes[:n]
            for i, scene in enumerate(plan.scenes):
                scene.scene_id = dramatic_ids[i]
            if len(plan.briefs) > n:
                plan.briefs = plan.briefs[:n]
            for i, brief in enumerate(plan.briefs):
                brief.scene_id = dramatic_ids[i]

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

        # ---- Gap G9 backfill: DetailPrinciple grounded preoccupations ----
        for scene in plan.scenes:
            default_dp = scene_detail_defaults.get(scene.scene_id)
            if default_dp is None:
                continue
            if scene.detail_principle is None:
                scene.detail_principle = default_dp
                continue
            dp = scene.detail_principle
            if not dp.perceptual_preoccupations:
                dp.perceptual_preoccupations = list(default_dp.perceptual_preoccupations)
            if not dp.triple_duty_targets:
                dp.triple_duty_targets = list(default_dp.triple_duty_targets)

        # ---- Gap G10 backfill: MetaphorProfile grounded domains ----
        for scene in plan.scenes:
            default_mp = scene_metaphor_defaults.get(scene.scene_id)
            if default_mp is None:
                continue
            # find MetaphorProfile entry for this character (if LLM emitted one)
            pov_entry: MetaphorProfile | None = None
            for mp in scene.metaphor_profiles:
                if mp.character_id == default_mp.character_id:
                    pov_entry = mp
                    break
            if pov_entry is None:
                scene.metaphor_profiles.append(default_mp)
                continue
            if not pov_entry.permanent_domains:
                pov_entry.permanent_domains = list(default_mp.permanent_domains)
            if not pov_entry.current_domains:
                pov_entry.current_domains = list(default_mp.current_domains)
            if not pov_entry.forbidden_domains:
                pov_entry.forbidden_domains = list(default_mp.forbidden_domains)

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

    # -- Helpers --------------------------------------------------------

    async def _retrieve_due_motifs(
        self,
        motif_retriever: MotifRetriever,
        update_number: int,
    ) -> list[dict]:
        """Query ``motif_retriever`` for motifs due/overdue at this update.

        Converts each :class:`~app.retrieval.Result` into a plain dict the
        user template renders. Retrieval failures are swallowed so a
        broken retriever never blocks craft planning.
        """
        query = Query(filters={"current_update": int(update_number)})
        try:
            results = await motif_retriever.retrieve(query, k=3)
        except Exception:
            return []

        due_motifs: list[dict] = []
        for r in results:
            meta = r.metadata or {}
            due_motifs.append(
                {
                    "source_id": r.source_id,
                    "motif_id": meta.get("motif_id"),
                    "name": meta.get("name"),
                    "description": r.text,
                    "last_update_number": meta.get("last_update_number"),
                    "last_semantic_value": meta.get("last_semantic_value"),
                    "intervals_since_last": meta.get("intervals_since_last"),
                    "target_interval_min": meta.get("target_interval_min"),
                    "target_interval_max": meta.get("target_interval_max"),
                    "status": meta.get("status"),
                    "recent_contexts": meta.get("recent_contexts") or [],
                }
            )
        return due_motifs

    async def _retrieve_ripe_hooks(
        self,
        foreshadowing_retriever: "ForeshadowingRetriever",
        update_number: int,
    ) -> list[dict]:
        """Pull up to k=3 ripe foreshadowing hooks for the current update.

        Retrieval failures are swallowed — the planner runs without
        ``ripe_hooks`` rather than crashing on an advisory signal.
        """
        # Lazy import: ``app.retrieval.foreshadowing_retriever`` imports
        # ``app.world.schema`` whose package ``__init__`` pulls seed.py,
        # which re-enters ``app.planning``. Importing at call time
        # breaks that cycle.
        from app.retrieval.interface import Query

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
