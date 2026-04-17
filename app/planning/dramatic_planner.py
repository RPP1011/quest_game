from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING


def _repair_missing_scene_ids(raw: str) -> str:
    """Inject scene_id into scenes that lack it.

    llama-server's JSON-schema enforcement doesn't always honor ``required``
    on nested structs, so the dramatic planner frequently emits
    ``{"beats": [...], ...}`` objects inside ``scenes`` with no ``scene_id``.
    We assign 1..N by position here before validation so downstream layers
    see a clean monotonic sequence. If parsing fails for any reason, return
    the raw string unchanged and let the caller's error path handle it.
    """
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(data, dict):
        return raw
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        return raw
    changed = False
    for i, sc in enumerate(scenes, start=1):
        if isinstance(sc, dict) and "scene_id" not in sc:
            sc["scene_id"] = i
            changed = True
    if not changed:
        return raw
    return json.dumps(data)

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
        skeleton_chapter: "Any | None" = None,
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
        # Pull world context — split active vs dormant so the planner
        # can see what's available to surface this update.
        from app.world.schema import EntityStatus
        all_entities = world.list_entities()
        characters = [e for e in all_entities if e.status == EntityStatus.ACTIVE]
        dormant_entities = [e for e in all_entities if e.status == EntityStatus.DORMANT]
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

        # Day 11: build a closed-enum list of valid tool ids for the
        # prompt + schema. The 1.2B model otherwise hallucinates plausible
        # tool ids ('chekhov_plant', 'map_planting') the critic rejects.
        valid_tool_ids = sorted(t.id for t in self._craft_library.tools())

        schema = DramaticPlan.model_json_schema()
        # Constrain ``ToolSelection.tool_id`` and
        # ``DramaticScene.tools_used[]`` to the valid set via
        # JSON-schema enum so xgrammar enforces it at decode time.
        if valid_tool_ids:
            defs = schema.get("$defs", {})
            ts = defs.get("ToolSelection")
            if ts is not None and "tool_id" in ts.get("properties", {}):
                ts["properties"]["tool_id"]["enum"] = list(valid_tool_ids)
            ds = defs.get("DramaticScene")
            if ds is not None and "tools_used" in ds.get("properties", {}):
                items = ds["properties"]["tools_used"].get("items", {})
                items["enum"] = list(valid_tool_ids)

        system_prompt = self._renderer.render(
            "stages/dramatic/system.j2",
            {"schema": schema, "valid_tool_ids": valid_tool_ids},
        )
        user_prompt = self._renderer.render(
            "stages/dramatic/user.j2",
            {
                "directive": directive,
                "player_action": player_action,
                "characters": characters,
                "dormant_entities": dormant_entities,
                "plot_threads": plot_threads,
                "recent_narrative": recent_narrative,
                "tools_with_examples": tools_with_examples,
                "themes": themes,
                "reader_state": reader_state,
                "information_asymmetries": asymmetries,
                "scene_exemplars": scene_exemplars,
                "ripe_hooks": ripe_hooks,
                "valid_tool_ids": valid_tool_ids,
                "skeleton_chapter": skeleton_chapter,
            },
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        # Day 11: in-band retry on ParseError catches xgrammar
        # truncations cheaply (no validator round-trip) before
        # _retry_with_critic burns its retry budget on a fallback.
        from app.world.output_parser import ParseError as _ParseError
        raw = await self._client.chat_structured(
            messages,
            json_schema=schema,
            schema_name="DramaticPlan",
            max_tokens=8192,
        )
        raw = _repair_missing_scene_ids(raw)
        try:
            plan = OutputParser.parse_json(raw, schema=DramaticPlan)
        except _ParseError:
            raw = await self._client.chat_structured(
                messages,
                json_schema=schema,
                schema_name="DramaticPlan",
                max_tokens=8192,
            )
            raw = _repair_missing_scene_ids(raw)
            plan = OutputParser.parse_json(raw, schema=DramaticPlan)

        # Day 11: Small models routinely emit ``scene_id: 42`` (literal
        # hallucinated default) for every scene. Renumber dramatic scenes
        # 1..N by position so downstream emotional/craft layers see a
        # clean monotonic sequence. The mapping is also applied to
        # ``tools_selected`` so cross-references don't dangle.
        if plan.scenes:
            old_to_new: dict[int, int] = {}
            for i, scene in enumerate(plan.scenes, start=1):
                old_to_new.setdefault(scene.scene_id, i)
                scene.scene_id = i
            for sel in plan.tools_selected:
                sel.scene_id = old_to_new.get(sel.scene_id, sel.scene_id)

            # Day 12: the VoiceRetriever only fires when a scene has
            # ``pov_character_id`` set, but the 1.2B model frequently
            # leaves it ``None``. Post-parse, default any unset POV to
            # the quest's primary character (first CHARACTER entity by
            # id order; this is stable and deterministic). Prompt-based
            # asks are less reliable than this deterministic fill.
            default_pov = self._default_pov_character_id(characters)
            if default_pov is not None:
                for scene in plan.scenes:
                    if not scene.pov_character_id:
                        scene.pov_character_id = default_pov
        return plan

    @staticmethod
    def _default_pov_character_id(characters: list) -> str | None:
        """Pick a deterministic fallback POV when the planner omits one.

        Prefers an entity with id=="player" (the quest-game convention for
        the protagonist). Falls back to the first CHARACTER entity by list
        order. Returns ``None`` when no characters exist yet.
        """
        from app.world.schema import EntityType
        player = None
        first_char = None
        for entity in characters:
            etype = getattr(entity, "entity_type", None)
            is_character = etype == EntityType.CHARACTER or etype == "character"
            if not is_character:
                continue
            if entity.id == "player":
                player = entity.id
                break
            if first_char is None:
                first_char = entity.id
        return player or first_char

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
