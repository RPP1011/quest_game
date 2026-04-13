from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, Union
from app.runtime.client import ChatMessage
from app.world.output_parser import OutputParser, ParseError
from app.world.schema import EmotionalBeat, NarrativeRecord, QuestArcState
from app.world.state_manager import WorldStateManager, WorldStateError
from .check import CHECK_SCHEMA, CheckIssue, CheckOutput
from .context_builder import ContextBuilder
from .context_spec import CHECK_SPEC, EXTRACT_SPEC, PLAN_SPEC, REVISE_SPEC, WRITE_SPEC
from .extract import EXTRACT_SCHEMA, build_delta
from .inference_params import TokenUsage
from .stages import StageError, StageResult
from .trace import PipelineTrace


_BEAT_KEYS = ("beats", "beat_sheet", "beatsheet", "plan", "outline", "steps", "scenes")
_CHOICE_KEYS = ("suggested_choices", "suggestedchoices", "choices", "options", "actions")


def _normalize_beat_sheet(data: dict) -> dict:
    """Accept common key aliases produced by weaker/non-strict models.

    Handles case variants (beatSheet vs beat_sheet), list-of-dicts (extracts a
    string field), nested wrappers (``beat_sheet: {key_actions: [...]}``), and
    falls back to scanning any list-of-strings value.

    ``suggested_choices`` is always emitted as a list of dicts with keys
    ``title``, ``description``, and ``tags``.  Plain strings are coerced to
    ``{title: str, description: "", tags: []}``.
    """
    lowered = {k.lower().replace("_", ""): v for k, v in data.items()}
    beats = _extract_list(_pick(lowered, _BEAT_KEYS), _BEAT_KEYS)
    if not beats:
        for v in data.values():
            beats = _extract_list(v, _BEAT_KEYS)
            if beats:
                break
    raw_choices = _pick(lowered, _CHOICE_KEYS)
    if raw_choices is None:
        # Some models nest suggested_choices inside beat_sheet dict; scan one
        # level into any dict value.
        for v in data.values():
            if isinstance(v, dict):
                inner = {k.lower().replace("_", ""): vv for k, vv in v.items()}
                raw_choices = _pick(inner, _CHOICE_KEYS)
                if raw_choices:
                    break
    choices = _coerce_choice_list(raw_choices)
    return {"beats": beats, "suggested_choices": choices}


def _coerce_choice_list(value) -> list[dict]:
    """Coerce a raw value into a list of choice dicts.

    Each output dict has ``title`` (str), ``description`` (str), and
    ``tags`` (list[str]).  Input items may be plain strings or dicts.
    """
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for item in value:
        if isinstance(item, str):
            out.append({"title": item, "description": "", "tags": []})
        elif isinstance(item, dict):
            # Resolve title from common aliases.
            title = ""
            for key in ("title", "text", "choice", "name"):
                if isinstance(item.get(key), str):
                    title = item[key]
                    break
            if not title:
                # Fall back to concatenating string values.
                parts = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
                title = " -- ".join(parts)
            # Resolve description from common aliases.
            description = ""
            for key in ("description", "flavor", "detail"):
                if isinstance(item.get(key), str):
                    description = item[key]
                    break
            # Resolve tags.
            raw_tags = item.get("tags")
            tags: list[str] = []
            if isinstance(raw_tags, list):
                tags = [t for t in raw_tags if isinstance(t, str)]
            out.append({"title": title, "description": description, "tags": tags})
    return out


def _extract_list(value, nested_keys: tuple[str, ...]) -> list[str]:
    """Coerce value to a list of strings. If value is a dict, look inside it
    for a nested list under any of ``nested_keys``."""
    direct = _coerce_string_list(value)
    if direct:
        return direct
    if isinstance(value, dict):
        inner = {k.lower().replace("_", ""): vv for k, vv in value.items()}
        return _coerce_string_list(_pick(inner, nested_keys + ("keyactions",)))
    return []


def _pick(lowered: dict, keys: tuple[str, ...]):
    for k in keys:
        norm = k.lower().replace("_", "")
        if norm in lowered:
            return lowered[norm]
    return None


def _coerce_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            # Prefer conventional fields, else fall back to concatenating values.
            for key in ("beat", "text", "description", "summary", "name", "content"):
                if isinstance(item.get(key), str):
                    out.append(item[key])
                    break
            else:
                parts = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
                if parts:
                    out.append(" — ".join(parts))
    return out


BEAT_SHEET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "beats": {"type": "array", "items": {"type": "string"}},
        "suggested_choices": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "description"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
    },
    "required": ["beats", "suggested_choices"],
    "additionalProperties": False,
}


def _make_minimal_arc_directive() -> "Any":
    """Return a minimal ArcDirective when no arc planner is available."""
    from app.planning.schemas import ArcDirective
    return ArcDirective(
        current_phase="unknown",
        phase_assessment="No arc planner configured.",
    )


def _make_minimal_dramatic_plan(player_action: str) -> "Any":
    """Synthetic DramaticPlan for when the model's output can't be parsed.
    Produces a single scene that just acknowledges the player action."""
    from app.planning.schemas import (
        ActionResolution, DramaticPlan, DramaticScene,
    )
    scene = DramaticScene(
        scene_id=1,
        dramatic_question=f"How does the protagonist respond to: {player_action}",
        outcome="The action plays out; its consequences ripple forward.",
        beats=[player_action],
        dramatic_function="Advance the story in response to the player's action.",
    )
    return DramaticPlan(
        action_resolution=ActionResolution(kind="partial", narrative=player_action),
        scenes=[scene],
        update_tension_target=0.5,
        ending_hook="What happens next?",
        suggested_choices=[],
    )


def _make_minimal_emotional_plan(dramatic) -> "Any":
    """Synthetic EmotionalPlan with one entry per dramatic scene."""
    from app.planning.schemas import EmotionalPlan, EmotionalScenePlan
    scenes = [
        EmotionalScenePlan(
            scene_id=s.scene_id,
            primary_emotion="focused",
            intensity=0.5,
            entry_state="alert",
            exit_state="resolved",
            transition_type="shift",
            emotional_source="The stakes of the moment.",
        )
        for s in dramatic.scenes
    ]
    return EmotionalPlan(
        scenes=scenes,
        update_emotional_arc="Steady. Tension holds.",
        contrast_strategy="Hold on the moment rather than pushing variety.",
    )


def _make_minimal_craft_plan(dramatic) -> "Any":
    """Synthetic CraftPlan with a minimal brief per scene."""
    from app.planning.schemas import CraftBrief, CraftPlan, CraftScenePlan
    scenes = [CraftScenePlan(scene_id=s.scene_id) for s in dramatic.scenes]
    briefs = [
        CraftBrief(
            scene_id=s.scene_id,
            brief=(
                f"Write this scene in second person, past tense. "
                f"Focus on the dramatic question: {s.dramatic_question}. "
                f"End with the outcome: {s.outcome}. "
                "Prose should be concrete, sensory, unrushed. Trust the moment."
            ),
        )
        for s in dramatic.scenes
    ]
    return CraftPlan(scenes=scenes, briefs=briefs)


class InferenceClientLike(Protocol):
    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str: ...
    async def chat(self, *, messages, **kw) -> str: ...


@dataclass
class PipelineOutput:
    prose: str
    choices: list[dict]
    beats: list[str]
    trace: PipelineTrace


class Pipeline:
    def __init__(
        self,
        world: WorldStateManager,
        context_builder: ContextBuilder,
        client: InferenceClientLike,
        *,
        arc_planner: "Any | None" = None,
        dramatic_planner: "Any | None" = None,
        emotional_planner: "Any | None" = None,
        craft_planner: "Any | None" = None,
        craft_library: "Any | None" = None,
        structure: "Any | None" = None,
        quest_config: dict | None = None,
        quest_id: str | None = None,
        arc_id: str = "main",
        emotional_history_window: int = 10,
        emotional_monotony_window: int = 3,
    ) -> None:
        self._world = world
        self._cb = context_builder
        self._client = client
        self._arc_planner = arc_planner
        self._dramatic_planner = dramatic_planner
        self._emotional_planner = emotional_planner
        self._craft_planner = craft_planner
        self._craft_library = craft_library
        self._structure = structure
        self._quest_config = quest_config or {}
        self._quest_id = quest_id
        self._arc_id = arc_id
        self._narrator = None
        n_raw = self._quest_config.get("narrator") if self._quest_config else None
        if n_raw:
            try:
                from app.craft.schemas import Narrator
                self._narrator = Narrator.model_validate(n_raw)
            except Exception:
                self._narrator = None
        self._last_dramatic: Any | None = None
        self._last_emotional: Any | None = None
        self._emotional_history_window = emotional_history_window
        self._emotional_monotony_window = emotional_monotony_window

    @property
    def is_hierarchical(self) -> bool:
        """True iff all four hierarchical planning layers are wired in."""
        return all([
            self._dramatic_planner is not None,
            self._emotional_planner is not None,
            self._craft_planner is not None,
            self._craft_library is not None,
        ])

    async def run(self, *, player_action: str, update_number: int) -> PipelineOutput:
        trace = PipelineTrace(trace_id=uuid.uuid4().hex, trigger=player_action)

        if self.is_hierarchical:
            craft_plan, plan_like_dict = await self._run_hierarchical(
                trace, player_action, update_number
            )
            narrator_voice = (
                list(self._narrator.voice_samples) if self._narrator else None
            )
            prose = await self._run_write(
                trace, craft_plan, voice_samples=narrator_voice
            )
            # Wood-gap critics on combined prose
            await self._run_craft_critics(trace, craft_plan, prose)
        else:
            # ---- Flat (legacy) flow ----
            plan_like_dict = None
            craft_plan = None

        if not self.is_hierarchical:
            replan_attempts = 0
            recheck_done = False
            critical_feedback: list[CheckIssue] = []

            plan_parsed = await self._run_plan(trace, player_action, critical_feedback)
            prose = await self._run_write(trace, plan_parsed)
            check_out = await self._run_check(trace, plan_parsed, prose)

            # REPLAN branch: critical issues, replan once.
            if check_out.has_critical and replan_attempts < 1:
                replan_attempts += 1
                critical_feedback = list(check_out.issues)
                plan_parsed = await self._run_plan(trace, player_action, critical_feedback)
                prose = await self._run_write(trace, plan_parsed)
                check_out = await self._run_check(trace, plan_parsed, prose)

            # REVISE branch: fixable issues, revise + recheck once.
            if check_out.has_fixable and not check_out.has_critical and not recheck_done:
                prose = await self._run_revise(trace, plan_parsed, prose, check_out.issues)
                recheck_done = True
                check_out = await self._run_check(trace, plan_parsed, prose)

            # Determine outcome.
            if check_out.has_critical:
                outcome = "flagged_qm"
            else:
                outcome = "committed"

            self._world.write_narrative(NarrativeRecord(
                update_number=update_number,
                raw_text=prose,
                player_action=player_action,
                pipeline_trace_id=trace.trace_id,
            ))
            trace.outcome = outcome

            # EXTRACT stage: only when not critical. Best-effort — any failure
            # is recorded in the trace but never breaks the chapter commit.
            if not check_out.has_critical:
                try:
                    await self._run_extract(trace, plan_parsed, prose, update_number)
                except Exception as e:  # pragma: no cover - defensive
                    trace.add_stage(StageResult(
                        stage_name="extract", input_prompt="", raw_output="",
                        errors=[StageError(kind="extract_crash", message=str(e))],
                    ))

            return PipelineOutput(
                prose=prose,
                choices=plan_parsed.get("suggested_choices", []),
                beats=plan_parsed["beats"],
                trace=trace,
            )

        # ---- Hierarchical post-write flow ----
        recheck_done = False
        check_out = await self._run_check(trace, plan_like_dict, prose)

        # REVISE branch (no replan in hierarchical flow for now)
        if check_out.has_fixable and not check_out.has_critical and not recheck_done:
            prose = await self._run_revise(trace, plan_like_dict, prose, check_out.issues)
            recheck_done = True
            check_out = await self._run_check(trace, plan_like_dict, prose)

        if check_out.has_critical:
            outcome = "flagged_qm"
        else:
            outcome = "committed"

        self._world.write_narrative(NarrativeRecord(
            update_number=update_number,
            raw_text=prose,
            player_action=player_action,
            pipeline_trace_id=trace.trace_id,
        ))
        trace.outcome = outcome

        if (outcome == "committed"
                and self._quest_id is not None
                and self._last_dramatic is not None):
            try:
                from app.planning.reader_model import apply_dramatic_plan
                current = self._world.get_reader_state(self._quest_id)
                updated = apply_dramatic_plan(
                    current,
                    self._last_dramatic,
                    update_number=update_number,
                    emotional=self._last_emotional,
                )
                self._world.upsert_reader_state(updated)
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="reader_state", input_prompt="", raw_output="",
                    errors=[StageError(kind="reader_state_crash", message=str(e)[:300])],
                ))

        # Gap G7: post-commit information-state accumulation.
        if (outcome == "committed"
                and self._quest_id is not None
                and self._last_dramatic is not None):
            try:
                from app.planning.information_asymmetry import (
                    apply_dramatic_plan_reveals,
                )
                apply_dramatic_plan_reveals(
                    world=self._world,
                    quest_id=self._quest_id,
                    dramatic=self._last_dramatic,
                    update_number=update_number,
                )
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="information_states", input_prompt="", raw_output="",
                    errors=[StageError(kind="info_states_crash", message=str(e)[:300])],
                ))

        if outcome == "committed":
            self._persist_emotional_beats(plan_like_dict, update_number, trace)

        if outcome == "committed" and craft_plan is not None:
            try:
                self._persist_parallels(craft_plan, update_number)
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="parallels_persist",
                    input_prompt="", raw_output="",
                    errors=[StageError(kind="parallels_persist_crash", message=str(e))],
                ))
            try:
                self._persist_motif_occurrences(craft_plan, update_number)
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="motif_occurrences_persist",
                    input_prompt="", raw_output="",
                    errors=[StageError(kind="motif_occurrences_persist_crash", message=str(e))],
                ))

        if not check_out.has_critical:
            try:
                await self._run_extract(trace, plan_like_dict, prose, update_number)
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="extract", input_prompt="", raw_output="",
                    errors=[StageError(kind="extract_crash", message=str(e))],
                ))

        suggested_choices = plan_like_dict.get("suggested_choices", [])
        beats = plan_like_dict.get("beats", [])
        return PipelineOutput(
            prose=prose,
            choices=suggested_choices,
            beats=beats,
            trace=trace,
        )

    async def _run_hierarchical(
        self,
        trace: PipelineTrace,
        player_action: str,
        update_number: int,
    ) -> "tuple[Any, dict]":
        """Run the 4-layer hierarchy: arc → dramatic → emotional → craft.

        Returns (craft_plan, plan_like_dict) where plan_like_dict has the
        ``beats`` and ``suggested_choices`` keys expected by CHECK/REVISE/EXTRACT.
        """
        from app.planning import critics
        from app.planning.schemas import ArcDirective

        # ---- ARC layer: load or generate ----
        directive = await self._load_or_generate_arc(trace)

        # ---- DRAMATIC layer ----
        dramatic = await self._retry_with_critic(
            trace=trace,
            stage_name="dramatic",
            generator=lambda hint: self._dramatic_planner.plan(
                directive=directive,
                player_action=player_action,
                world=self._world,
                arc=self._get_craft_arc(),
                structure=self._structure,
                recent_tool_ids=None,
                quest_id=self._quest_id,
            ),
            validator=lambda plan: critics.validate_dramatic(
                plan,
                active_entity_ids={e.id for e in self._world.list_entities()},
                valid_tool_ids=self._get_valid_tool_ids(),
            ),
            fallback=lambda: _make_minimal_dramatic_plan(player_action),
        )

        # ---- EMOTIONAL layer ----
        all_narrative = self._world.list_narrative(limit=10_000)
        recent_prose = [r.raw_text for r in all_narrative[-2:]] if all_narrative else []

        # Load recent emotional beats so the planner sees actual trajectory.
        recent_beats: list[EmotionalBeat] = []
        monotony_flag = False
        if self._quest_id is not None:
            from app.planning.emotional_planner import detect_monotony
            recent_beats = self._world.list_recent_emotional_beats(
                self._quest_id, limit=self._emotional_history_window
            )
            monotony_flag = detect_monotony(
                recent_beats, window=self._emotional_monotony_window
            )

        emotional = await self._retry_with_critic(
            trace=trace,
            stage_name="emotional",
            generator=lambda hint: self._emotional_planner.plan(
                dramatic=dramatic,
                world=self._world,
                recent_prose=recent_prose,
                recent_beats=recent_beats,
                monotony_flag=monotony_flag,
            ),
            validator=lambda plan: critics.validate_emotional(plan, dramatic),
            fallback=lambda: _make_minimal_emotional_plan(dramatic),
        )

        # ---- CRAFT layer ----
        craft_plan = await self._retry_with_critic(
            trace=trace,
            stage_name="craft",
            generator=lambda hint: self._craft_planner.plan(
                dramatic=dramatic,
                emotional=emotional,
                style_register_id=None,
                narrator=self._narrator,
                active_parallels=self._world.list_parallels(),
                active_motifs=self._build_motif_context(update_number),
            ),
            validator=lambda plan: critics.validate_craft(plan, dramatic),
            fallback=lambda: _make_minimal_craft_plan(dramatic),
        )

        # Synthesize plan_like_dict for CHECK/REVISE/EXTRACT compat
        beats = [scene.dramatic_question for scene in dramatic.scenes]
        plan_like_dict: dict = {
            "beats": beats,
            "suggested_choices": dramatic.suggested_choices,
            "_emotional_plan": emotional,
        }

        # Stash for post-commit reader_state mutation (Gap G6).
        self._last_dramatic = dramatic
        self._last_emotional = emotional

        return craft_plan, plan_like_dict

    def _persist_emotional_beats(
        self,
        plan_like_dict: dict,
        update_number: int,
        trace: PipelineTrace,
    ) -> None:
        """Write the emotional plan's per-scene targets as observed beats.

        Best-effort: any persistence failure is recorded in the trace but
        never breaks the commit.
        """
        if self._quest_id is None:
            return
        emotional = plan_like_dict.get("_emotional_plan")
        if emotional is None:
            return
        try:
            for scene in emotional.scenes:
                beat = EmotionalBeat(
                    quest_id=self._quest_id,
                    update_number=update_number,
                    scene_index=scene.scene_id,
                    primary_emotion=scene.primary_emotion,
                    secondary_emotion=scene.secondary_emotion,
                    intensity=scene.intensity,
                    source=scene.emotional_source,
                )
                self._world.record_emotional_beat(beat)
        except Exception as e:  # pragma: no cover - defensive
            trace.add_stage(StageResult(
                stage_name="emotional_persist", input_prompt="", raw_output="",
                errors=[StageError(kind="emotional_persist_crash", message=str(e)[:300])],
            ))

    def _persist_parallels(self, craft_plan: "Any", update_number: int) -> None:
        """Post-COMMIT: walk the craft plan and plant/deliver parallels.

        Identity strategy: ``ParallelInstruction.parallel_id`` is the row id.
        If a row with that id exists, mark it ``delivered``. If not, treat the
        instruction as the A-half (plant) and insert a new row — the
        ``source_description`` on the instruction describes the just-established
        half, and ``execution_guidance`` becomes the ``target_description`` the
        B-half should fulfil later.
        """
        from app.world.schema import Parallel, ParallelStatus

        quest_id = self._quest_id or "__ephemeral__"
        for scene in getattr(craft_plan, "scenes", []) or []:
            inst = getattr(scene, "parallel_instruction", None)
            if inst is None:
                continue
            try:
                existing = self._world.get_parallel(inst.parallel_id)
            except WorldStateError:
                existing = None

            if existing is None:
                # Plant the A-half.
                self._world.add_parallel(Parallel(
                    id=inst.parallel_id,
                    quest_id=quest_id,
                    source_update=update_number,
                    source_description=inst.source_description,
                    inversion_axis=inst.inversion_axis,
                    target_description=inst.execution_guidance,
                    status=ParallelStatus.PLANTED,
                ))
            else:
                # Deliver the B-half.
                self._world.update_parallel(inst.parallel_id, {
                    "status": ParallelStatus.DELIVERED,
                    "delivered_at_update": update_number,
                })

    def _persist_motif_occurrences(self, craft_plan: "Any", update_number: int) -> None:
        """Post-COMMIT: record each ``MotifInstruction`` on the craft plan as an
        observed ``MotifOccurrence`` row so recurrence tracking (Gap G5) has
        data on future updates.
        """
        from app.planning.world_extensions import MotifOccurrence

        if self._quest_id is None:
            return
        quest_id = self._quest_id
        known = {m.id for m in self._world.list_motifs(quest_id)}
        for scene in getattr(craft_plan, "scenes", []) or []:
            for inst in getattr(scene, "motif_instructions", []) or []:
                motif_id = getattr(inst, "motif_id", None)
                if not motif_id or motif_id not in known:
                    continue
                occ = MotifOccurrence(
                    motif_id=motif_id,
                    update_number=update_number,
                    context=getattr(inst, "placement", "") or "",
                    semantic_value=getattr(inst, "semantic_value", "") or "",
                    intensity=float(getattr(inst, "intensity", 0.5) or 0.5),
                )
                self._world.record_motif_occurrence(quest_id, occ)

    def _build_motif_context(self, update_number: int) -> list[dict]:
        """Return per-motif recurrence info (last_occurrence_update,
        last_semantic_value, overdue) for the craft planner prompt."""
        if self._quest_id is None:
            return []
        out: list[dict] = []
        for motif in self._world.list_motifs(self._quest_id):
            last = self._world.last_motif_occurrence(self._quest_id, motif.id)
            last_update = last.update_number if last else None
            last_sem = last.semantic_value if last else None
            if last_update is None:
                overdue = update_number >= motif.target_interval_min
            else:
                overdue = (update_number - last_update) > motif.target_interval_max
            out.append({
                "motif": motif,
                "last_occurrence_update": last_update,
                "last_semantic_value": last_sem,
                "overdue": overdue,
            })
        return out

    async def _load_or_generate_arc(self, trace: PipelineTrace) -> "Any":
        """Load persisted ArcDirective or generate one with arc_planner."""
        from app.planning.schemas import ArcDirective

        if self._quest_id is not None:
            try:
                arc_state = self._world.get_arc(self._quest_id, self._arc_id)
                if arc_state.last_directive is not None:
                    directive = ArcDirective.model_validate(arc_state.last_directive)
                    trace.add_stage(StageResult(
                        stage_name="arc",
                        input_prompt="",
                        raw_output=json.dumps(arc_state.last_directive),
                        parsed_output=arc_state.last_directive,
                    ))
                    return directive
            except WorldStateError:
                arc_state = None

            # Generate fresh arc directive
            if self._arc_planner is not None and self._structure is not None:
                if arc_state is None:
                    arc_state = QuestArcState(
                        quest_id=self._quest_id,
                        arc_id=self._arc_id,
                        structure_id=self._structure.id,
                        scale=self._structure.scales[0],
                    )
                try:
                    directive = await self._arc_planner.plan(
                        quest_config=self._quest_config,
                        arc_state=arc_state,
                        world_snapshot=self._world,
                        structure=self._structure,
                    )
                except Exception as e:
                    directive = _make_minimal_arc_directive()
                    trace.add_stage(StageResult(
                        stage_name="arc", input_prompt="", raw_output="",
                        parsed_output=json.loads(directive.model_dump_json()),
                        errors=[StageError(kind="arc_fallback", message=str(e)[:300])],
                    ))
                    directive_dict = json.loads(directive.model_dump_json())
                    arc_state_updated = arc_state.model_copy(update={"last_directive": directive_dict})
                    self._world.upsert_arc(arc_state_updated)
                    return directive
                directive_dict = json.loads(directive.model_dump_json())
                arc_state_updated = arc_state.model_copy(
                    update={"last_directive": directive_dict}
                )
                self._world.upsert_arc(arc_state_updated)
                trace.add_stage(StageResult(
                    stage_name="arc",
                    input_prompt="",
                    raw_output=directive.model_dump_json(),
                    parsed_output=directive_dict,
                ))
                return directive

        # Fallback: generate a minimal arc directive without persistence
        if self._arc_planner is not None and self._structure is not None:
            arc_state = QuestArcState(
                quest_id="__ephemeral__",
                arc_id=self._arc_id,
                structure_id=self._structure.id,
                scale=self._structure.scales[0],
            )
            directive = await self._arc_planner.plan(
                quest_config=self._quest_config,
                arc_state=arc_state,
                world_snapshot=self._world,
                structure=self._structure,
            )
            trace.add_stage(StageResult(
                stage_name="arc",
                input_prompt="",
                raw_output=directive.model_dump_json(),
                parsed_output=json.loads(directive.model_dump_json()),
            ))
            return directive

        # No arc planner — synthesize a minimal directive
        directive = _make_minimal_arc_directive()
        trace.add_stage(StageResult(
            stage_name="arc",
            input_prompt="",
            raw_output="{}",
            parsed_output={},
        ))
        return directive

    async def _retry_with_critic(
        self,
        *,
        trace: PipelineTrace,
        stage_name: str,
        generator,
        validator,
        fallback=None,
    ) -> "Any":
        """Call generator, validate, record StageResult. Retry once on errors.

        If the generator itself raises (ParseError, validation failure against
        the schema, etc.) and a ``fallback`` factory is provided, record the
        crash as a stage error and return ``fallback()``. This keeps weak
        models from breaking the whole pipeline.
        """
        import inspect

        try:
            result = await generator(hint=None)
        except Exception as e:
            if fallback is None:
                raise
            result = fallback()
            trace.add_stage(StageResult(
                stage_name=stage_name, input_prompt="",
                raw_output="",
                parsed_output=json.loads(result.model_dump_json()) if hasattr(result, "model_dump_json") else {},
                errors=[StageError(kind=f"{stage_name}_fallback", message=str(e)[:300])],
            ))
            return result
        issues = validator(result)
        errors = [
            StageError(
                kind="critic_error" if i.severity == "error" else "critic_warning",
                message=i.message,
            )
            for i in issues
        ]
        raw = result.model_dump_json() if hasattr(result, "model_dump_json") else str(result)
        parsed = json.loads(raw) if hasattr(result, "model_dump_json") else {}

        if any(i.severity == "error" for i in issues):
            # Retry once: generator may accept a hint about issues
            hint = "\n".join(f"- [{i.severity}] {i.message}" for i in issues)
            try:
                result2 = await generator(hint=hint)
                issues2 = validator(result2)
                errors2 = [
                    StageError(
                        kind="critic_error" if i.severity == "error" else "critic_warning",
                        message=i.message,
                    )
                    for i in issues2
                ]
                raw2 = result2.model_dump_json() if hasattr(result2, "model_dump_json") else str(result2)
                parsed2 = json.loads(raw2) if hasattr(result2, "model_dump_json") else {}
                trace.add_stage(StageResult(
                    stage_name=stage_name,
                    input_prompt=f"[retry after critic: {hint[:200]}]",
                    raw_output=raw2,
                    parsed_output=parsed2,
                    errors=errors + errors2,
                ))
                return result2
            except Exception:
                pass  # fall through to use original result

        trace.add_stage(StageResult(
            stage_name=stage_name,
            input_prompt="",
            raw_output=raw,
            parsed_output=parsed,
            errors=errors,
        ))
        return result

    def _get_craft_arc(self) -> "Any":
        """Return a minimal craft Arc for tool recommendations."""
        from app.craft.schemas import Arc as CraftArc
        structure_id = self._structure.id if self._structure else "three_act"
        return CraftArc(
            id=self._arc_id,
            name=self._arc_id,
            scale="chapter",
            structure_id=structure_id,
        )

    def _get_valid_tool_ids(self) -> "set[str]":
        """Return all tool ids from the craft library (or empty set if none)."""
        if self._craft_library is None:
            return set()
        try:
            return {t.id for t in self._craft_library.tools()}
        except Exception:
            return set()

    async def _run_craft_critics(
        self, trace: PipelineTrace, craft_plan: "Any", prose: str
    ) -> None:
        """Run Wood-gap critics on prose and append as a single 'craft_critics' StageResult."""
        from app.planning import critics

        all_issues = []
        all_issues.extend(critics.validate_free_indirect_integrity(craft_plan, prose))
        all_issues.extend(critics.validate_detail_characterization(craft_plan, prose))
        all_issues.extend(critics.validate_metaphor_domains(craft_plan, prose))
        all_issues.extend(critics.validate_indirection(craft_plan, prose))
        all_issues.extend(critics.validate_voice_blend(craft_plan, prose))
        all_issues.extend(
            critics.validate_narrator_sensory_distribution(self._narrator, prose)
        )

        errors = [
            StageError(
                kind="critic_error" if i.severity == "error" else "critic_warning",
                message=i.message,
            )
            for i in all_issues
        ]
        trace.add_stage(StageResult(
            stage_name="craft_critics",
            input_prompt="",
            raw_output="",
            errors=errors,
        ))

    async def _run_plan(
        self, trace: PipelineTrace, player_action: str,
        critical_feedback: list[CheckIssue],
    ) -> dict:
        extras: dict[str, Any] = {"player_action": player_action}
        if critical_feedback:
            extras["critical_feedback"] = "\n".join(
                f"- [{i.severity}/{i.category}] {i.message}" for i in critical_feedback
            )
        plan_ctx = self._cb.build(
            spec=PLAN_SPEC,
            stage_name="plan",
            templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
            extras=extras,
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=plan_ctx.system_prompt),
                ChatMessage(role="user", content=plan_ctx.user_prompt),
            ],
            json_schema=BEAT_SHEET_SCHEMA,
            schema_name="BeatSheet",
            temperature=0.4,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        parsed = OutputParser.parse_json(raw)
        if not isinstance(parsed, dict):
            trace.add_stage(StageResult(
                stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
                errors=[StageError(kind="parse_error", message="not a dict")],
                latency_ms=latency,
            ))
            trace.outcome = "failed"
            raise ParseError(f"plan not a dict: {parsed!r}")
        normalized = _normalize_beat_sheet(parsed)
        errors: list[StageError] = []
        if not normalized["beats"]:
            # Weak/non-strict models regularly fail to produce a clean list;
            # fall back to a synthetic plan derived from the player's action so
            # the WRITE stage can still run. Record the parse issue in the trace.
            normalized = {
                "beats": [f"React naturally to: {player_action}"],
                "suggested_choices": normalized.get("suggested_choices", []),
            }
            errors.append(StageError(
                kind="parse_warning",
                message="no beats extracted from plan; using synthetic fallback",
                detail={"raw_sample": raw[:400]},
            ))
        trace.add_stage(StageResult(
            stage_name="plan", input_prompt=plan_ctx.user_prompt, raw_output=raw,
            parsed_output=normalized,
            token_usage=TokenUsage(prompt=plan_ctx.token_estimate),
            latency_ms=latency,
            errors=errors,
        ))
        return normalized

    async def _run_write(
        self,
        trace: PipelineTrace,
        plan: "Union[dict, Any]",  # dict = old path, CraftPlan = new path
        *,
        voice_samples: list[str] | None = None,
        anti_patterns: list[str] | None = None,
    ) -> str:
        """Write prose.

        New path: ``plan`` is a ``CraftPlan`` (has ``.scenes`` and ``.briefs``).
        Old path (backwards-compat shim): ``plan`` is a dict with ``beats`` key.
        """
        # --- detect which path to use ---
        # CraftPlan is a Pydantic model; it won't have a "beats" key via dict
        # access, but it will have a .scenes attribute.
        try:
            scenes = plan.scenes  # CraftPlan
            is_craft_plan = True
        except AttributeError:
            is_craft_plan = False

        if not is_craft_plan:
            # ---- Old backwards-compatible path ----
            plan_text = "\n".join(f"- {b}" for b in plan["beats"])
            write_ctx = self._cb.build(
                spec=WRITE_SPEC,
                stage_name="write",
                templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
                extras={
                    "plan": plan_text,
                    "style": "",
                    "anti_patterns": anti_patterns or [],
                    "brief": None,
                    "scene": None,
                    "voice_samples": voice_samples or [],
                    "recent_prose_tail": "",
                },
            )
            t0 = time.perf_counter()
            raw = await self._client.chat(
                messages=[
                    ChatMessage(role="system", content=write_ctx.system_prompt),
                    ChatMessage(role="user", content=write_ctx.user_prompt),
                ],
                temperature=0.8,
            )
            latency = int((time.perf_counter() - t0) * 1000)
            prose = OutputParser.parse_prose(raw)
            trace.add_stage(StageResult(
                stage_name="write", input_prompt=write_ctx.user_prompt, raw_output=raw,
                parsed_output=prose,
                token_usage=TokenUsage(prompt=write_ctx.token_estimate),
                latency_ms=latency,
            ))
            return prose

        # ---- New CraftPlan path: one call per scene ----
        # Build a brief lookup by scene_id
        briefs_by_scene: dict[int, Any] = {b.scene_id: b for b in plan.briefs}

        prose_parts: list[str] = []
        recent_prose_tail = ""

        for scene in scenes:
            brief_obj = briefs_by_scene.get(scene.scene_id)
            brief_text = brief_obj.brief if brief_obj else None

            # Determine blended voice samples: prefer voice_permeability.blended_voice_samples
            effective_voice_samples = list(voice_samples or [])
            if scene.voice_permeability and scene.voice_permeability.blended_voice_samples:
                # Prepend blended samples as they are most important
                effective_voice_samples = (
                    scene.voice_permeability.blended_voice_samples + effective_voice_samples
                )

            write_ctx = self._cb.build(
                spec=WRITE_SPEC,
                stage_name="write",
                templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
                extras={
                    "brief": brief_text,
                    "scene": scene.model_dump() if hasattr(scene, "model_dump") else scene,
                    "voice_samples": effective_voice_samples,
                    "recent_prose_tail": recent_prose_tail,
                    "anti_patterns": anti_patterns or [],
                    "plan": None,
                    "style": "",
                },
            )
            t0 = time.perf_counter()
            raw = await self._client.chat(
                messages=[
                    ChatMessage(role="system", content=write_ctx.system_prompt),
                    ChatMessage(role="user", content=write_ctx.user_prompt),
                ],
                temperature=0.8,
            )
            latency = int((time.perf_counter() - t0) * 1000)
            scene_prose = OutputParser.parse_prose(raw)
            prose_parts.append(scene_prose)

            # Track last ~300 chars for rhythm continuity on next scene
            full_so_far = "\n\n".join(prose_parts)
            recent_prose_tail = full_so_far[-300:] if len(full_so_far) > 300 else full_so_far

            trace.add_stage(StageResult(
                stage_name="write",
                input_prompt=write_ctx.user_prompt,
                raw_output=raw,
                parsed_output=scene_prose,
                token_usage=TokenUsage(prompt=write_ctx.token_estimate),
                latency_ms=latency,
                detail={"scene_id": scene.scene_id},
            ))

        return "\n\n".join(prose_parts)

    async def _run_check(self, trace: PipelineTrace, plan: dict, prose: str) -> CheckOutput:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        ctx = self._cb.build(
            spec=CHECK_SPEC,
            stage_name="check",
            templates={"system": "stages/check/system.j2", "user": "stages/check/user.j2"},
            extras={"plan": plan_text, "prose": prose},
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            json_schema=CHECK_SCHEMA,
            schema_name="CheckOutput",
            temperature=0.2,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        try:
            parsed = OutputParser.parse_json(raw, schema=CheckOutput)
        except ParseError:
            # Defensive: if we can't parse, treat as clean so we don't block forever.
            parsed = CheckOutput(issues=[])
        trace.add_stage(StageResult(
            stage_name="check", input_prompt=ctx.user_prompt, raw_output=raw,
            parsed_output=parsed.model_dump(),
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))
        return parsed

    async def _run_extract(
        self, trace: PipelineTrace, plan: dict, prose: str, update_number: int,
    ) -> None:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        entities = self._world.list_entities()
        from app.world.schema import EntityStatus
        active_entities = [e for e in entities if e.status == EntityStatus.ACTIVE]
        ctx = self._cb.build(
            spec=EXTRACT_SPEC,
            stage_name="extract",
            templates={
                "system": "stages/extract/system.j2",
                "user": "stages/extract/user.j2",
            },
            extras={"plan": plan_text, "prose": prose, "entities": active_entities},
        )
        t0 = time.perf_counter()
        raw = await self._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            json_schema=EXTRACT_SCHEMA,
            schema_name="StateDelta",
            temperature=0.0,
        )
        latency = int((time.perf_counter() - t0) * 1000)

        from app.world.output_parser import OutputParser, ParseError as _ParseError
        try:
            extracted = OutputParser.parse_json(raw)
        except _ParseError as exc:
            trace.add_stage(StageResult(
                stage_name="extract",
                input_prompt=ctx.user_prompt,
                raw_output=raw,
                errors=[StageError(kind="parse_error", message=str(exc))],
                latency_ms=latency,
            ))
            return

        if not isinstance(extracted, dict):
            trace.add_stage(StageResult(
                stage_name="extract",
                input_prompt=ctx.user_prompt,
                raw_output=raw,
                errors=[StageError(kind="parse_error", message="extract not a dict")],
                latency_ms=latency,
            ))
            return

        known_ids = {e.id for e in active_entities}
        delta, build_issues = build_delta(extracted, update_number, known_ids=known_ids)

        # Also validate via WorldStateManager
        validation = self._world.validate_delta(delta)

        all_errors = (
            [StageError(kind="build_error", message=i.message) for i in build_issues]
            + [
                StageError(kind="validation_error", message=i.message)
                for i in validation.issues
                if i.severity == "error"
            ]
        )

        if all_errors:
            trace.add_stage(StageResult(
                stage_name="extract",
                input_prompt=ctx.user_prompt,
                raw_output=raw,
                parsed_output=extracted,
                errors=all_errors,
                latency_ms=latency,
            ))
            return

        # Apply atomically.
        self._world.apply_delta(delta, update_number)

        # Theme stance evolution (Gap G4). Stance updates are persisted but
        # the LLM-driven assessment that *decides* the new stance is a stub
        # for now: the model may emit updates in the extract JSON, and we
        # write them through. The heuristic/LLM assessor will land later.
        # TODO(G4-stance): replace direct-from-extract writes with a
        # dedicated post-COMMIT "theme assessor" stage that inspects the
        # scene + recent prose and decides whether a stance should shift.
        theme_updates = extracted.get("theme_stance_updates", []) or []
        if theme_updates and self._quest_id is not None:
            _VALID_STANCES = {"exploring", "affirming", "questioning", "subverting"}
            for item in theme_updates:
                tid = item.get("id")
                new_stance = item.get("new_stance")
                if not tid or new_stance not in _VALID_STANCES:
                    continue
                try:
                    self._world.update_theme_stance(self._quest_id, tid, new_stance)
                except Exception:
                    # non-fatal: a bogus theme id shouldn't blow up COMMIT
                    continue

        trace.add_stage(StageResult(
            stage_name="extract",
            input_prompt=ctx.user_prompt,
            raw_output=raw,
            parsed_output=extracted,
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))

    async def _run_revise(
        self, trace: PipelineTrace, plan: dict, prose: str,
        issues: list[CheckIssue],
    ) -> str:
        plan_text = "\n".join(f"- {b}" for b in plan["beats"])
        ctx = self._cb.build(
            spec=REVISE_SPEC,
            stage_name="revise",
            templates={"system": "stages/revise/system.j2", "user": "stages/revise/user.j2"},
            extras={
                "plan": plan_text,
                "prose": prose,
                "issues": [i.model_dump() for i in issues],
                "style": "",
                "anti_patterns": [],
            },
        )
        t0 = time.perf_counter()
        raw = await self._client.chat(
            messages=[
                ChatMessage(role="system", content=ctx.system_prompt),
                ChatMessage(role="user", content=ctx.user_prompt),
            ],
            temperature=0.6,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        revised = OutputParser.parse_prose(raw)
        trace.add_stage(StageResult(
            stage_name="revise", input_prompt=ctx.user_prompt, raw_output=raw,
            parsed_output=revised,
            token_usage=TokenUsage(prompt=ctx.token_estimate),
            latency_ms=latency,
        ))
        return revised
