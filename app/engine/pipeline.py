from __future__ import annotations
import asyncio
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


# Token budget for scene prose. Without this vllm/llama-server falls back to
# the server default (as low as 16), which truncates scenes to a few lines.
# 3000 fits ~2200 words — a bit of headroom above the 2000-word target set in
# the write-stage system prompt.
WRITE_MAX_TOKENS = 3000
REVISE_MAX_TOKENS = 3000


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


DEFAULT_RERANK_WEIGHTS: dict[str, float] = {
    # Heuristic weights. Errors are far worse than warnings; voice/detail
    # grounding are what the hierarchy is built to produce, so they get the
    # largest deductions. POV/action/entity fidelity catch the grossest
    # prose-level failures from small models. Defaults chosen so any single
    # error dominates any stack of warnings; a clean candidate scores 0.0,
    # flawed candidates score negative, and the least-flawed wins.
    "error": -5.0,
    "warning": -1.0,
    # Per-critic scaling: some critics are more load-bearing.
    "narrator_sensory": 1.2,
    "free_indirect": 1.2,
    "detail_characterization": 1.0,
    "metaphor_domains": 0.8,
    "indirection": 0.8,
    "voice_blend": 0.8,
    "pov_adherence": 1.5,
    "named_entity_presence": 1.0,
    "action_fidelity": 1.5,
}


# Day 3: rerank weights for the Scorer-driven path. Keys match
# ``app.scoring.DIMENSION_NAMES``; weights are multiplied against the
# per-dim [0, 1] score and summed for candidate ranking. Default = every
# dim at 1.0 so the rerank objective coincides with the unweighted mean
# that ``Scorer`` returns as ``overall_score``. Callers who want to bias
# rerank toward specific dims (e.g. 2.0 on ``pov_adherence`` because
# small models drift there first) override per-key via the
# ``rerank_weights`` ctor kwarg or ``quest_config["rerank_weights"]``.
DEFAULT_SCORER_RERANK_WEIGHTS: dict[str, float] = {
    "sentence_variance": 1.0,
    "dialogue_ratio": 1.0,
    "pacing": 1.0,
    "sensory_density": 1.0,
    "free_indirect_quality": 1.0,
    "detail_characterization": 1.0,
    "metaphor_domains_score": 1.0,
    "indirection_score": 1.0,
    "pov_adherence": 1.0,
    "named_entity_presence": 1.0,
    "narrator_sensory_match": 1.0,
    "action_fidelity": 1.0,
}


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
        n_candidates: int = 1,
        check_all_candidates: bool = False,
        rerank_weights: dict[str, float] | None = None,
        candidate_base_temperature: float = 0.8,
        candidate_temperature_step: float = 0.15,
        retrieval_embedder: "Any | None" = None,
        passage_retriever: "Any | None" = None,
        quest_retriever: "Any | None" = None,
        voice_retriever: "Any | None" = None,
        motif_retriever: "Any | None" = None,
        foreshadowing_retriever: "Any | None" = None,
        scene_retriever: "Any | None" = None,
        scorer: "Any | None" = None,
        llm_judge_client: "Any | None" = None,
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
        # Target total words per scene, used to give the beat-loop writer a
        # pacing budget. Not a hard cap — each beat ends when the model ends.
        self._scene_target_words: int = int(
            (quest_config or {}).get("scene_target_words", 2000)
        )
        self._emotional_history_window = emotional_history_window
        self._emotional_monotony_window = emotional_monotony_window

        # ---- Gap G2: Generate-N + Rerank ----
        # quest_config takes precedence over ctor default so CLI/server can
        # enable quality runs without re-plumbing constructors.
        cfg = self._quest_config or {}
        self._n_candidates = int(cfg.get("n_candidates", n_candidates))
        if self._n_candidates < 1:
            self._n_candidates = 1
        self._check_all_candidates = bool(
            cfg.get("check_all_candidates", check_all_candidates)
        )
        self._rerank_weights = dict(DEFAULT_RERANK_WEIGHTS)
        if rerank_weights:
            self._rerank_weights.update(rerank_weights)
        cfg_weights = cfg.get("rerank_weights")
        if isinstance(cfg_weights, dict):
            self._rerank_weights.update(cfg_weights)

        # Day 3: Scorer-driven rerank weights. The 12 dim keys are disjoint
        # from the legacy critic keys in ``_rerank_weights`` (critic names
        # vs. dim names — e.g. ``free_indirect`` vs ``free_indirect_quality``),
        # so the same ``rerank_weights`` ctor kwarg can carry both sets and
        # we slice out the dim-shaped entries here. Missing keys default to
        # 1.0; extra keys (legacy critic names, or unknown) are ignored for
        # scorer scoring.
        self._scorer_rerank_weights = dict(DEFAULT_SCORER_RERANK_WEIGHTS)
        if rerank_weights:
            for k, v in rerank_weights.items():
                if k in self._scorer_rerank_weights:
                    self._scorer_rerank_weights[k] = float(v)
        if isinstance(cfg_weights, dict):
            for k, v in cfg_weights.items():
                if k in self._scorer_rerank_weights:
                    self._scorer_rerank_weights[k] = float(v)
        self._candidate_base_temperature = float(
            cfg.get("candidate_base_temperature", candidate_base_temperature)
        )
        self._candidate_temperature_step = float(
            cfg.get("candidate_temperature_step", candidate_temperature_step)
        )

        # Retrieval layer (Wave 1c infra; Wave 3b wiring will supply a real
        # embedder). ``None`` disables the extract-stage embedding write.
        self._retrieval_embedder: Any | None = retrieval_embedder

        # Wave 2b: PassageRetriever for writer voice anchors. ``None``
        # disables retrieval entirely; otherwise retrieval is further
        # gated by ``quest_config["retrieval"]["enabled"]`` (default False)
        # so existing callers that pass a retriever but don't flip the
        # flag see no behavioral change (SUCCESS CRITERIA: default-off).
        self._passage_retriever: Any | None = passage_retriever
        retrieval_cfg = (self._quest_config or {}).get("retrieval") or {}
        self._retrieval_enabled: bool = bool(retrieval_cfg.get("enabled", False))

        # Wave 3b: QuestRetriever for writer in-quest callbacks. Same default-off
        # semantics as the voice-anchor path: ``None`` disables entirely, and
        # ``quest_config["retrieval"]["enabled"]`` further gates it.
        self._quest_retriever: Any | None = quest_retriever

        # Wave 4c: VoiceRetriever — per-POV-character voice continuity.
        # Same default-off semantics: ``None`` disables entirely, and the
        # ``retrieval.enabled`` flag further gates the call. No POV on the
        # scene ⇒ no retrieval, matching the spec (see §3.6).
        self._voice_retriever: Any | None = voice_retriever

        # Day 11: MotifRetriever, ForeshadowingRetriever, SceneShapeRetriever.
        # These were constructed but never invoked in the Day 10 stress test
        # (the pipeline accepted them only as planner.plan() kwargs and
        # _run_hierarchical did not pass them). Now wired.
        self._motif_retriever: Any | None = motif_retriever
        self._foreshadowing_retriever: Any | None = foreshadowing_retriever
        self._scene_retriever: Any | None = scene_retriever

        # Day 2: Scorer for post-commit 12-dim scorecard persistence.
        # Default-off via quest_config["scoring"]["enabled"] — the kwarg
        # may be wired by a server/CLI caller, but if the flag is off the
        # hook stays silent. Also silent when ``quest_id`` is missing,
        # since scorecards are quest-scoped.
        self._scorer: Any | None = scorer
        scoring_cfg = (self._quest_config or {}).get("scoring") or {}
        self._scoring_enabled: bool = bool(scoring_cfg.get("enabled", False))

        # Day 6: optional LLM-judge client for async post-commit scoring of
        # the three anchored dims (tension_execution, emotional_trajectory,
        # choice_hook_quality). Default-off via absence of the kwarg; when
        # both ``scorer`` and ``llm_judge_client`` are present AND the
        # ``scoring.enabled`` flag is truthy, the post-commit hook fires a
        # non-blocking ``asyncio.create_task`` that writes the three extra
        # dim rows onto the already-persisted scorecard when it completes.
        # Tests that want to await the task should read the most recent
        # handle via ``pipeline.last_llm_judge_task``.
        self._llm_judge_client: Any | None = llm_judge_client
        self.last_llm_judge_task: asyncio.Task[None] | None = None

        # Day 4: SFT collection. Default-off via
        # ``quest_config["sft_collection"]["enabled"]``. When enabled AND
        # ``n_candidates > 1`` AND we have a ``quest_id``, dump per-scene
        # {craft brief, all candidates, scorer breakdowns, winner index} to
        # ``data/sft/<quest_id>/u<update>_s<scene>_<trace_id>.json``.
        # ``sft_collection["dir"]`` may override the output root (tests).
        sft_cfg = (self._quest_config or {}).get("sft_collection") or {}
        self._sft_enabled: bool = bool(sft_cfg.get("enabled", False))
        self._sft_dir: str = str(sft_cfg.get("dir", "data/sft"))

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
                trace, craft_plan, voice_samples=narrator_voice,
                player_action=player_action,
                update_number=update_number,
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
            prose = await self._run_write(
                trace, plan_parsed, player_action=player_action,
                update_number=update_number,
            )
            check_out = await self._run_check(trace, plan_parsed, prose)

            # REPLAN branch: critical issues, replan once.
            if check_out.has_critical and replan_attempts < 1:
                replan_attempts += 1
                critical_feedback = list(check_out.issues)
                plan_parsed = await self._run_plan(trace, player_action, critical_feedback)
                prose = await self._run_write(
                    trace, plan_parsed, player_action=player_action,
                    update_number=update_number,
                )
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
                pov_character_id=self._primary_pov_character_id(),
            ))
            trace.outcome = outcome

            # Wave 1c: gated narrative-embedding write. No-op while
            # ``_retrieval_embedder`` is ``None`` (default); Wave 3b activates.
            if outcome == "committed":
                try:
                    self._persist_narrative_embedding(prose, update_number)
                except Exception as e:  # pragma: no cover - defensive
                    trace.add_stage(StageResult(
                        stage_name="narrative_embedding_persist",
                        input_prompt="", raw_output="",
                        errors=[StageError(
                            kind="narrative_embedding_persist_crash",
                            message=str(e),
                        )],
                    ))

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

            # Day 2: post-commit scoring. No-op unless scorer + flag + quest_id.
            if outcome == "committed":
                self._persist_scorecard(
                    trace=trace,
                    prose=prose,
                    craft_plan=None,
                    player_action=player_action,
                    update_number=update_number,
                )

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
            pov_character_id=self._primary_pov_character_id(),
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

        # Wave 1c: infrastructure hook. ``_retrieval_embedder`` is ``None``
        # by default so this is a no-op in every current caller; Wave 3b
        # activates the embedder and flips this on.
        if outcome == "committed":
            try:
                self._persist_narrative_embedding(prose, update_number)
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="narrative_embedding_persist",
                    input_prompt="", raw_output="",
                    errors=[StageError(
                        kind="narrative_embedding_persist_crash",
                        message=str(e),
                    )],
                ))

        if not check_out.has_critical:
            try:
                await self._run_extract(trace, plan_like_dict, prose, update_number)
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="extract", input_prompt="", raw_output="",
                    errors=[StageError(kind="extract_crash", message=str(e))],
                ))

        # Day 2: post-commit scoring. No-op unless scorer + flag + quest_id.
        if outcome == "committed":
            self._persist_scorecard(
                trace=trace,
                prose=prose,
                craft_plan=craft_plan,
                player_action=player_action,
                update_number=update_number,
            )

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
        # Day 11: pass scene_retriever + foreshadowing_retriever +
        # update_number so the dramatic planner actually consults them
        # (Day 10 surfaced them as constructed-but-never-invoked).
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
                scene_retriever=self._scene_retriever,
                foreshadowing_retriever=self._foreshadowing_retriever,
                update_number=update_number,
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
        # Collect POV character entities so the craft planner can ground
        # voice (G3), detail (G9), and metaphor (G10) from entity data.
        pov_ids = {s.pov_character_id for s in dramatic.scenes if s.pov_character_id}
        characters: dict[str, Any] = {}
        for pov_id in pov_ids:
            try:
                characters[pov_id] = self._world.get_entity(pov_id)
            except WorldStateError:
                continue

        # Day 11: pass motif_retriever + foreshadowing_retriever +
        # update_number so the craft planner actually consults them.
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
                characters=characters,
                world=self._world,
                motif_retriever=self._motif_retriever,
                foreshadowing_retriever=self._foreshadowing_retriever,
                update_number=update_number,
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

        # Force player-character POV across all scenes when the narrator
        # config identifies a single protagonist. The dramatic LLM often
        # picks an NPC's POV (e.g. the innkeeper), which puts voice
        # retrieval and narrative-embedding persistence on the wrong
        # character and breaks quest memory. Narrator config convention:
        # ``pov_character_id`` names the entity whose POV the narrator
        # follows. Defaults to id="player" when set.
        narrator_cfg = (self._quest_config or {}).get("narrator") or {}
        forced_pov = narrator_cfg.get("pov_character_id")
        if forced_pov:
            for scene in dramatic.scenes:
                scene.pov_character_id = forced_pov

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

    def _persist_narrative_embedding(self, prose: str, update_number: int) -> None:
        """Post-COMMIT: compute and persist a retrieval embedding for ``prose``.

        Wave 1c — infrastructure hook; Wave 3b activates the embedder.
        Early-exits when no embedder is configured, no quest id is set, or
        the prose is empty. ``scene_index=0`` is used as a pragmatic default
        for the whole-commit embedding (Wave 3b decides per-scene split).
        """
        if self._retrieval_embedder is None or self._quest_id is None:
            return
        if not prose:
            return
        # Compute embedding and upsert. The embedder's primary API is
        # ``embed_one`` (see :class:`app.retrieval.embeddings.Embedder`); we
        # fall back to ``embed`` so test fakes that expose the shorter name
        # still work. The upsert helper normalizes whatever array-likes come
        # back to float32 bytes.
        if hasattr(self._retrieval_embedder, "embed_one"):
            embedding = self._retrieval_embedder.embed_one(prose)
        else:
            embedding = self._retrieval_embedder.embed(prose)
        text_preview = prose[:200]
        self._world.upsert_narrative_embedding(
            quest_id=self._quest_id,
            update_number=update_number,
            scene_index=0,
            embedding=embedding,
            text_preview=text_preview,
        )

    async def _retrieve_voice_anchors(
        self,
        *,
        scene: "Any",
        emotional_scene: "Any | None",
        brief_text: str | None,
    ) -> list[dict]:
        """Wave 2b: retrieve voice-anchor passages for a single scene write.

        Returns ``[]`` when retrieval is disabled, no retriever is wired, or
        the retriever yields no hits. Each returned dict carries the fields
        the prompt template needs: ``text`` (passage body), ``pov``, ``score``,
        and ``source_id`` for provenance.

        The query is built from:
          * ``filters["score_ranges"]`` — a widened ``voice_distinctiveness``
            band and, when craft permeability targets exist, a permeability-
            derived ``free_indirect_quality`` range.
          * ``seed_text`` — the scene brief truncated to 600 chars, with
            the target emotion appended when available (harmless in
            metadata-only mode; useful for Wave 2a semantic rerank).

        Day 12: the POV filter was exact-equality on the corpus ``pov``
        field, but the manifest spells POV as ``third_limited_fis``,
        ``third_limited_multi``, ``first_mixed`` etc. while quest configs
        use the plain ``third_limited`` / ``second`` tokens, so the
        filter dropped nearly every passage. Day 10/11 stress tests
        reported 0 hits across 50 updates as a direct consequence. We
        now omit the POV filter entirely and lean on seed-text semantic
        relevance + ``voice_distinctiveness`` score proximity. POV is
        still surfaced on each returned anchor via ``metadata.pov`` so
        the WRITE stage can steer register if it wants to.
        """
        if self._passage_retriever is None or not self._retrieval_enabled:
            return []

        from app.retrieval.interface import Query

        # ---- Build score range filters ----
        # Day 12: widen ``voice_distinctiveness`` from (0.7, 1.0) to
        # (0.5, 1.0). With the POV filter dropped, the score band is
        # the dominant filter, and the prior range rejected 39/195
        # corpus passages on a dim whose label distribution is long-
        # tailed. (0.5, 1.0) keeps the strong-voice bias while leaving
        # headroom for passages whose actual score dipped just below
        # the work-level expected.
        score_ranges: dict[str, tuple[float, float]] = {
            "voice_distinctiveness": (0.5, 1.0),
        }
        vp = getattr(scene, "voice_permeability", None)
        if vp is not None:
            # Use the permeability baseline/target as the midpoint of a
            # ±0.2 band for free_indirect_quality. Clamp to [0,1].
            target = getattr(vp, "current_target", None)
            if target is None:
                target = getattr(vp, "baseline", 0.3)
            lo = max(0.0, float(target) - 0.2)
            hi = min(1.0, float(target) + 0.2)
            score_ranges["free_indirect_quality"] = (lo, hi)

        # ---- Build seed text from brief + target emotion ----
        seed_parts: list[str] = []
        if brief_text:
            seed_parts.append(brief_text)
        if emotional_scene is not None:
            emo = getattr(emotional_scene, "primary_emotion", None)
            if emo:
                seed_parts.append(f"[emotion: {emo}]")
        seed_text = " ".join(seed_parts)[:600] or None

        query = Query(
            seed_text=seed_text,
            filters={"score_ranges": score_ranges},
        )

        try:
            results = await self._passage_retriever.retrieve(query, k=3)
        except Exception:
            # Best-effort: retrieval failure must not break the write stage.
            return []

        anchors: list[dict] = []
        for r in results:
            meta = getattr(r, "metadata", None) or {}
            anchors.append({
                "source_id": getattr(r, "source_id", ""),
                "text": getattr(r, "text", ""),
                "pov": meta.get("pov", ""),
                "score": float(getattr(r, "score", 0.0)),
            })
        return anchors

    def _collect_scene_entity_mentions(
        self,
        *,
        scene: "Any",
        craft_plan: "Any | None",
    ) -> set[str]:
        """Pull the entity names/ids mentioned by a scene plan.

        Sources (all best-effort, absent fields are skipped):
          * Stashed :attr:`_last_dramatic` plan — POV character id,
            ``characters_present``, and the ``dramatic_question`` (whose
            free-text we scan against the world's entity-name list).
          * Craft scene's ``narrator_focus`` ids.
          * Any ``voice_notes[*].character_id``.

        Returns a string set; caller passes it straight to the retriever
        query under ``filters["entity_mentions"]``.
        """
        mentions: set[str] = set()

        # ---- Craft scene (narrator_focus + voice_notes) ----
        for field in ("narrator_focus", "narrator_withholding"):
            try:
                for name in getattr(scene, field, None) or []:
                    if isinstance(name, str) and name:
                        mentions.add(name)
            except Exception:
                pass
        try:
            for vn in getattr(scene, "voice_notes", None) or []:
                cid = getattr(vn, "character_id", None)
                if isinstance(cid, str) and cid:
                    mentions.add(cid)
        except Exception:
            pass

        # ---- Matching DramaticScene (by scene_id) on the stashed plan ----
        dramatic = getattr(self, "_last_dramatic", None)
        scene_id = getattr(scene, "scene_id", None)
        if dramatic is not None and scene_id is not None:
            try:
                for ds in getattr(dramatic, "scenes", None) or []:
                    if getattr(ds, "scene_id", None) != scene_id:
                        continue
                    pov = getattr(ds, "pov_character_id", None)
                    if isinstance(pov, str) and pov:
                        mentions.add(pov)
                    for cid in getattr(ds, "characters_present", None) or []:
                        if isinstance(cid, str) and cid:
                            mentions.add(cid)
                    # dramatic_question is free text — match any known
                    # entity name as a substring (case-insensitive).
                    dq = getattr(ds, "dramatic_question", None)
                    if isinstance(dq, str) and dq:
                        try:
                            entities = list(self._world.list_entities())
                        except Exception:
                            entities = []
                        dq_low = dq.lower()
                        for e in entities:
                            nm = getattr(e, "name", None)
                            if isinstance(nm, str) and nm and nm.lower() in dq_low:
                                mentions.add(nm)
            except Exception:
                pass

        return mentions

    async def _retrieve_quest_callbacks(
        self,
        *,
        scene: "Any",
        brief_text: str | None,
        craft_plan: "Any | None" = None,
    ) -> list[dict]:
        """Wave 3b: retrieve in-quest callback passages for a single scene write.

        Mirrors the shape of :meth:`_retrieve_voice_anchors` but sources
        candidates from the quest's own ``narrative_embeddings`` table.
        Returns ``[]`` when retrieval is disabled, no retriever is wired,
        or the retriever yields no hits (in all of which the prompt
        template skips the callback block — default-off behavior).

        The query uses:
          * ``seed_text`` — same brief text fed to the voice-anchor
            retriever, so both retrievers see the same scene framing.
          * ``filters["last_n_records"] = 12`` — cap the candidate pool
            to the 12 most-recent records to keep retrieval fast.
          * ``filters["entity_mentions"]`` — every entity name/id the
            scene plan references (see
            :meth:`_collect_scene_entity_mentions`), so callbacks that
            name those entities bubble up.
        """
        if self._quest_retriever is None or not self._retrieval_enabled:
            return []

        from app.retrieval.interface import Query

        seed_text = (brief_text or "")[:600] or None
        if seed_text is None:
            # No seed means nothing to semantically match against; skip.
            return []

        entity_mentions = self._collect_scene_entity_mentions(
            scene=scene, craft_plan=craft_plan,
        )

        query = Query(
            seed_text=seed_text,
            filters={
                "last_n_records": 12,
                "entity_mentions": entity_mentions,
            },
        )

        try:
            results = await self._quest_retriever.retrieve(query, k=2)
        except Exception:
            return []

        callbacks: list[dict] = []
        for r in results:
            meta = getattr(r, "metadata", None) or {}
            callbacks.append({
                "source_id": getattr(r, "source_id", ""),
                "text": getattr(r, "text", ""),
                "score": float(getattr(r, "score", 0.0)),
                "metadata": {
                    "update_number": meta.get("update_number"),
                    "scene_index": meta.get("scene_index"),
                    "quest_id": meta.get("quest_id"),
                },
            })
        return callbacks

    def _resolve_scene_entities(self, scene: "Any") -> list:
        """Look up full Entity objects for a scene's characters_present list.

        Also checks the dramatic plan's characters_present when the craft
        scene doesn't carry its own.  Returns Entity objects with their
        full ``data`` dict so the writer template can render authoritative
        character descriptions.
        """
        char_ids: list[str] = []
        # Try craft scene first
        cp = getattr(scene, "characters_present", None)
        if cp:
            char_ids = list(cp)
        # Fall back to dramatic scene
        if not char_ids:
            dramatic = getattr(self, "_last_dramatic", None)
            scene_id = getattr(scene, "scene_id", None)
            if dramatic and scene_id is not None:
                for ds in getattr(dramatic, "scenes", None) or []:
                    if getattr(ds, "scene_id", None) == scene_id:
                        cp2 = getattr(ds, "characters_present", None)
                        if cp2:
                            char_ids = list(cp2)
                        break
        if not char_ids:
            return []
        entities = []
        for cid in char_ids:
            # Normalize: dramatic plan may use bare name like "tristan"
            # instead of "char:tristan"; try both.
            for lookup in (cid, f"char:{cid}"):
                try:
                    entities.append(self._world.get_entity(lookup))
                    break
                except Exception:
                    continue
        return entities

    def _primary_pov_character_id(self) -> str | None:
        """Choose a representative POV character id for the whole update.

        The ``narrative`` table stores one row per update; a multi-scene
        update may span several POVs. We pick the first scene's POV as
        the "primary" — it matches the beat of what a reader would
        perceive as the chapter's anchor. Falls back to ``None`` when no
        dramatic plan is stashed (flat flow) or no scene carries a POV.
        """
        dramatic = getattr(self, "_last_dramatic", None)
        if dramatic is None:
            return None
        for ds in getattr(dramatic, "scenes", None) or []:
            pov = getattr(ds, "pov_character_id", None)
            if isinstance(pov, str) and pov:
                return pov
        return None

    def _scene_pov_character_id(self, scene: "Any") -> str | None:
        """Resolve the POV character id for ``scene`` from the stashed dramatic plan.

        ``CraftScenePlan`` itself does not carry ``pov_character_id``; the
        dramatic plan does, keyed by the same ``scene_id``. Returns
        ``None`` when no match exists (flat flow, missing plan, or
        unresolved POV).
        """
        dramatic = getattr(self, "_last_dramatic", None)
        scene_id = getattr(scene, "scene_id", None)
        if dramatic is None or scene_id is None:
            return None
        for ds in getattr(dramatic, "scenes", None) or []:
            if getattr(ds, "scene_id", None) == scene_id:
                pov = getattr(ds, "pov_character_id", None)
                if isinstance(pov, str) and pov:
                    return pov
                return None
        return None

    async def _retrieve_voice_continuity(
        self,
        *,
        scene: "Any",
    ) -> list[dict]:
        """Wave 4c: retrieve per-character past utterances for this scene.

        Returns ``[]`` when retrieval is disabled, no retriever is wired,
        no POV can be resolved, or the retriever yields no hits. In all
        those cases the write-prompt template skips the voice-continuity
        block — strict default-off behavior.
        """
        if self._voice_retriever is None or not self._retrieval_enabled:
            return []

        pov_id = self._scene_pov_character_id(scene)
        if not pov_id:
            return []

        from app.retrieval.interface import Query

        query = Query(filters={"character_id": pov_id, "last_n_records": 30})
        try:
            results = await self._voice_retriever.retrieve(query, k=3)
        except Exception:
            # Best-effort: retrieval failure must not break the write stage.
            return []

        out: list[dict] = []
        for r in results:
            meta = getattr(r, "metadata", None) or {}
            out.append({
                "source_id": getattr(r, "source_id", ""),
                "text": getattr(r, "text", ""),
                "character_id": meta.get("character_id", pov_id),
                "source_update_number": meta.get("source_update_number"),
                "seed": bool(meta.get("seed", False)),
            })
        return out

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

    def _score_candidate(
        self,
        *,
        prose: str,
        craft_plan: "Any | None",
        player_action: str | None,
    ) -> "tuple[float, dict[str, Any], list[Any]]":
        """Run heuristic critics on a single candidate and compute a weighted
        score. Returns (weighted_score, per_dimension_breakdown, raw_issues).

        Per-dimension breakdown entries are ``{critic: {"errors": n,
        "warnings": n, "score": float}}``. Weighted score sums
        ``critic_weight * (err_weight * errors + warn_weight * warnings)``
        across critics. A clean candidate scores 0.0; flawed candidates
        score negative (less negative = better).
        """
        from app.planning import critics as _critics

        def _run(name, fn):
            try:
                issues = fn()
            except Exception as e:  # pragma: no cover - defensive
                return name, [], str(e)
            return name, issues, None

        checks: list[tuple[str, Any]] = []
        if craft_plan is not None:
            checks.append(("free_indirect", lambda: _critics.validate_free_indirect_integrity(craft_plan, prose)))
            checks.append(("detail_characterization", lambda: _critics.validate_detail_characterization(craft_plan, prose)))
            checks.append(("metaphor_domains", lambda: _critics.validate_metaphor_domains(craft_plan, prose)))
            checks.append(("indirection", lambda: _critics.validate_indirection(craft_plan, prose)))
            checks.append(("voice_blend", lambda: _critics.validate_voice_blend(craft_plan, prose)))
        checks.append((
            "narrator_sensory",
            lambda: _critics.validate_narrator_sensory_distribution(self._narrator, prose),
        ))
        checks.append(("pov_adherence", lambda: _critics.validate_pov_adherence(prose)))
        entity_names = [e.name for e in self._world.list_entities() if getattr(e, "name", None)]
        checks.append((
            "named_entity_presence",
            lambda: _critics.validate_named_entity_presence(prose, entity_names),
        ))
        if player_action is not None:
            checks.append(("action_fidelity", lambda: _critics.validate_action_fidelity(prose, player_action)))

        breakdown: dict[str, Any] = {}
        all_issues: list[Any] = []
        total = 0.0
        err_w = self._rerank_weights.get("error", -5.0)
        warn_w = self._rerank_weights.get("warning", -1.0)
        for name, fn in checks:
            _, issues, crash = _run(name, fn)
            if crash:
                breakdown[name] = {"errors": 0, "warnings": 0, "score": 0.0, "crash": crash}
                continue
            errs = sum(1 for i in issues if getattr(i, "severity", None) == "error")
            warns = sum(1 for i in issues if getattr(i, "severity", None) == "warning")
            critic_w = self._rerank_weights.get(name, 1.0)
            sub = critic_w * (err_w * errs + warn_w * warns)
            breakdown[name] = {"errors": errs, "warnings": warns, "score": sub}
            total += sub
            all_issues.extend(issues)
        return total, breakdown, all_issues

    def _persist_scorecard(
        self,
        *,
        trace: PipelineTrace,
        prose: str,
        craft_plan: "Any | None",
        player_action: str | None,
        update_number: int,
    ) -> None:
        """Day 2 post-commit hook: score ``prose`` and persist a scorecard.

        No-ops unless the pipeline was constructed with a ``scorer`` AND
        ``quest_config["scoring"]["enabled"]`` is truthy AND a ``quest_id``
        was provided. Any failure is recorded in the trace — the commit
        itself is never rolled back.

        Day 6 extension: if a ``llm_judge_client`` was also wired at
        construction, schedule an ``asyncio.create_task`` that calls
        :meth:`app.scoring.Scorer.score_with_llm_judges` and writes the
        three extra dim rows onto the same scorecard_id. The task handle
        is stashed at :attr:`last_llm_judge_task` for tests / shutdown.
        The commit return path does NOT await the task.
        """
        if self._scorer is None:
            return
        if not self._scoring_enabled:
            return
        if self._quest_id is None:
            return
        try:
            card = self._scorer.score(
                prose,
                craft_plan=craft_plan,
                narrator=self._narrator,
                world=self._world,
                player_action=player_action,
            )
            scorecard_id = self._world.save_scorecard(
                card,
                quest_id=self._quest_id,
                update_number=update_number,
                scene_index=0,
                pipeline_trace_id=trace.trace_id,
            )
            trace.add_stage(StageResult(
                stage_name="scoring",
                input_prompt="", raw_output="",
                parsed_output=None,
                detail={
                    "overall_score": card.overall_score,
                    "dimensions": dict(card.dimension_items()),
                },
            ))
        except Exception as e:  # pragma: no cover - defensive
            trace.add_stage(StageResult(
                stage_name="scoring", input_prompt="", raw_output="",
                errors=[StageError(kind="scoring_crash", message=str(e)[:300])],
            ))
            return

        # Day 6: fire-and-forget LLM-judge scoring. Only when a judge
        # client is wired AND the scorer itself was built to know about
        # it (``has_llm_judge``), so a Scorer constructed without the
        # Day 6 kwarg won't try to call it and crash.
        if (self._llm_judge_client is not None
                and getattr(self._scorer, "has_llm_judge", False)):
            self.last_llm_judge_task = asyncio.create_task(
                self._run_llm_judges_async(
                    prose=prose,
                    scorecard_id=scorecard_id,
                    trace_id=trace.trace_id,
                    craft_plan=craft_plan,
                    player_action=player_action,
                )
            )

    async def _run_llm_judges_async(
        self,
        *,
        prose: str,
        scorecard_id: int,
        trace_id: str,
        craft_plan: "Any | None",
        player_action: str | None,
    ) -> None:
        """Day 6: async body that calls the LLM judge and writes extra dim rows.

        Exceptions are logged to stderr but never propagate — this is a
        fire-and-forget task launched from the commit path. If scoring
        fails, the scorecard header and Day 2 dims are still intact.
        """
        try:
            is_quest = True  # All live chapters are quest scenes.
            pov = "second"
            if self._narrator is not None and getattr(self._narrator, "pov", None):
                pov = str(self._narrator.pov)
            ext = await self._scorer.score_with_llm_judges(  # type: ignore[union-attr]
                prose,
                work_id=self._quest_id or "quest",
                pov=pov,
                is_quest=is_quest,
                craft_plan=craft_plan,
                narrator=self._narrator,
                world=self._world,
                player_action=player_action,
            )
            self._world.append_dimension_scores(
                scorecard_id,
                ext.llm_judge_scores,
            )
        except Exception as e:  # pragma: no cover - defensive
            # Fire-and-forget; emit one warning so test traces show it.
            import sys
            print(
                f"[pipeline] llm-judge async failed for trace {trace_id}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )

    async def _dispatch_candidate(
        self,
        *,
        index: int,
        write_ctx: "Any",
        n: int,
    ) -> "tuple[int, float, float, str, int, float]":
        """Dispatch a single candidate inference call.

        Returns ``(index, temperature, start_ts, raw, latency_ms, end_ts)``.
        ``start_ts`` and ``end_ts`` are ``time.perf_counter()`` values used by
        the concurrency tests to assert that N calls launched near-together.

        Temperature jitter: candidate i uses base_temp + i*step (clamped to
        [0.1, 1.3]). If the client accepts a ``seed`` kwarg we also vary it;
        unrecognised kwargs are handled by falling back to a temperature-only
        call. Clients that forward ``**extra`` (the default InferenceClient,
        and vllm-style batching backends that accept ``seed``) get both.
        """
        temp = self._candidate_base_temperature + index * self._candidate_temperature_step
        temp = max(0.1, min(1.3, temp))
        kw: dict[str, Any] = {"temperature": temp, "max_tokens": WRITE_MAX_TOKENS}
        # Best-effort seed jitter: only pass if the client likely supports it.
        # The default InferenceClient forwards **extra kwargs, so passing
        # seed is safe; fake clients that don't accept it would fail — so
        # we only pass seed when N>1 (dev fake clients use N=1 by default).
        if n > 1:
            kw["seed"] = 1000 + index
        t0 = time.perf_counter()
        try:
            raw = await self._client.chat(
                messages=[
                    ChatMessage(role="system", content=write_ctx.system_prompt),
                    ChatMessage(role="user", content=write_ctx.user_prompt),
                ],
                **kw,
            )
        except TypeError:
            # Client doesn't accept one of our kwargs (e.g. seed). Retry
            # with only temperature — sampling non-determinism still gives
            # us variance between candidates.
            raw = await self._client.chat(
                messages=[
                    ChatMessage(role="system", content=write_ctx.system_prompt),
                    ChatMessage(role="user", content=write_ctx.user_prompt),
                ],
                temperature=temp,
                max_tokens=WRITE_MAX_TOKENS,
            )
        t1 = time.perf_counter()
        latency = int((t1 - t0) * 1000)
        return index, temp, t0, raw, latency, t1

    def _scorer_rerank_candidate(
        self,
        *,
        prose: str,
        craft_plan: "Any | None",
        player_action: str | None,
    ) -> "tuple[float, dict[str, float], float]":
        """Day 3: rerank a single candidate via the ``Scorer``.

        Returns ``(weighted_score, per_dim_scores, overall_score)``. The
        ``weighted_score`` is ``sum(weight_i * dim_i)`` across the 12 dims
        using ``self._scorer_rerank_weights``; ``overall_score`` is the
        unweighted mean produced by the scorer (mirrors what the post-
        commit scorecard would persist). Consumers log both — the weighted
        one drives winner selection, the overall one is the interpretable
        "how good is this candidate overall" number shown in A/B output.
        """
        assert self._scorer is not None
        card = self._scorer.score(
            prose,
            craft_plan=craft_plan,
            narrator=self._narrator,
            world=self._world,
            player_action=player_action,
        )
        dims = dict(card.dimension_items())
        weighted = 0.0
        for name, val in dims.items():
            w = self._scorer_rerank_weights.get(name, 1.0)
            weighted += w * float(val)
        return weighted, dims, float(card.overall_score)

    async def _generate_scene_candidates(
        self,
        *,
        trace: PipelineTrace,
        write_ctx: "Any",
        n: int,
        scene_id: int | None,
        craft_plan: "Any | None",
        player_action: str | None,
        update_number: int | None = None,
        brief_text: str | None = None,
    ) -> "tuple[str, list[dict]]":
        """Generate N candidate prose blocks for one scene, score each,
        return (winner_prose, records). Each record is a dict suitable for
        trace ``detail``.

        Day 3 changes:
        - The N inference calls are dispatched concurrently via
          ``asyncio.gather`` so batching backends like vllm can coalesce
          them into a single forward pass. Per-candidate seed / temperature
          jitter is preserved.
        - If a ``scorer`` was wired on the pipeline, rerank scoring uses
          the Day 2 :class:`~app.scoring.Scorer` (weighted sum over the 12
          dims). Otherwise we fall back to the legacy
          :meth:`_score_candidate` critic-sum — bit-identical behavior for
          callers that don't opt in.
        - A single ``write_rerank`` trace stage logs every candidate, its
          dim-level scores, and the winning index.
        """
        # ---- Concurrent dispatch (Day 3) ----
        # Build N inference coroutines and await them as a single gather.
        # Slow / fast candidates return in source order regardless of
        # completion order, since gather preserves the input list order.
        dispatches = [
            self._dispatch_candidate(index=i, write_ctx=write_ctx, n=n)
            for i in range(n)
        ]
        results = await asyncio.gather(*dispatches)

        use_scorer = self._scorer is not None
        records: list[dict] = []
        for index, temp, _t0, raw, latency, _t1 in results:
            scene_prose = OutputParser.parse_prose(raw)
            if use_scorer:
                try:
                    weighted, dim_scores, overall = self._scorer_rerank_candidate(
                        prose=scene_prose,
                        craft_plan=craft_plan,
                        player_action=player_action,
                    )
                    breakdown: dict[str, Any] = dict(dim_scores)
                    rerank_source = "scorer"
                    score_value = weighted
                    overall_value: float | None = overall
                except Exception as e:  # pragma: no cover - defensive
                    # Scorer crash ⇒ fall back to legacy per-candidate.
                    weighted_leg, leg_breakdown, _ = self._score_candidate(
                        prose=scene_prose,
                        craft_plan=craft_plan,
                        player_action=player_action,
                    )
                    breakdown = dict(leg_breakdown)
                    breakdown["_scorer_error"] = str(e)[:200]
                    rerank_source = "legacy_fallback"
                    score_value = weighted_leg
                    overall_value = None
            else:
                weighted_leg, leg_breakdown, _ = self._score_candidate(
                    prose=scene_prose,
                    craft_plan=craft_plan,
                    player_action=player_action,
                )
                breakdown = dict(leg_breakdown)
                rerank_source = "legacy"
                score_value = weighted_leg
                overall_value = None

            detail: dict[str, Any] = {
                "candidate_index": index,
                "n_candidates": n,
                "temperature": temp,
                "weighted_score": score_value,
                "dimension_scores": breakdown,
                "rerank_source": rerank_source,
            }
            if overall_value is not None:
                detail["overall_score"] = overall_value
            if scene_id is not None:
                detail["scene_id"] = scene_id
            trace.add_stage(StageResult(
                stage_name="write",
                input_prompt=write_ctx.user_prompt,
                raw_output=raw,
                parsed_output=scene_prose,
                token_usage=TokenUsage(prompt=write_ctx.token_estimate),
                latency_ms=latency,
                detail=detail,
            ))
            records.append({
                "index": index,
                "prose": scene_prose,
                "score": score_value,
                "breakdown": breakdown,
                "overall_score": overall_value,
                "rerank_source": rerank_source,
            })

        # Rerank: highest weighted_score wins. Ties broken by lowest index.
        winner = max(records, key=lambda r: (r["score"], -r["index"]))
        ranking = sorted(
            ((r["index"], r["score"]) for r in records),
            key=lambda x: (-x[1], x[0]),
        )
        rerank_detail: dict[str, Any] = {
            "winner_index": winner["index"],
            "winner_score": winner["score"],
            "ranking": [{"index": idx, "score": sc} for idx, sc in ranking],
            "n_candidates": n,
            "rerank_source": (
                "scorer" if use_scorer else "legacy"
            ),
            "candidates": [
                {
                    "index": r["index"],
                    "weighted_score": r["score"],
                    "overall_score": r["overall_score"],
                    "dimension_scores": r["breakdown"],
                }
                for r in records
            ],
        }
        if scene_id is not None:
            rerank_detail["scene_id"] = scene_id
        trace.add_stage(StageResult(
            stage_name="write_rerank",
            input_prompt="",
            raw_output="",
            parsed_output=winner["prose"],
            detail=rerank_detail,
        ))

        # Day 4: SFT collection. Persist (craft brief, all N candidates, per-
        # dim scorer breakdowns, winner index) so we can later mint SFT pairs
        # for the writer LoRA. Gated on the flag AND a quest_id — this is a
        # training-only artifact, separate from the existing trace JSON.
        if (
            self._sft_enabled
            and self._quest_id is not None
            and n > 1
        ):
            try:
                self._persist_sft_record(
                    trace=trace,
                    records=records,
                    winner_index=winner["index"],
                    scene_id=scene_id,
                    update_number=update_number,
                    brief_text=brief_text,
                    rerank_source=("scorer" if use_scorer else "legacy"),
                )
            except Exception as e:  # pragma: no cover - defensive
                trace.add_stage(StageResult(
                    stage_name="sft_collection",
                    input_prompt="", raw_output="",
                    errors=[StageError(
                        kind="sft_collection_crash",
                        message=str(e)[:300],
                    )],
                ))

        return winner["prose"], records

    def _persist_sft_record(
        self,
        *,
        trace: PipelineTrace,
        records: list[dict],
        winner_index: int,
        scene_id: int | None,
        update_number: int | None,
        brief_text: str | None,
        rerank_source: str,
    ) -> None:
        """Write the Day-4 SFT record for this scene to ``data/sft/<quest_id>/``.

        Output path: ``<sft_dir>/<quest_id>/u<update>_s<scene>_<trace_id>.json``.
        Each record contains ``craft_brief``, all N ``candidates`` (index, prose,
        weighted_score, overall_score, dimension_scores), ``winner_index``, and
        pointers (``quest_id``, ``update_number``, ``scene_index``,
        ``pipeline_trace_id``) so downstream tools can link back to traces.
        The record is intentionally focused — we do NOT re-embed the trace
        JSON, which is too big / noisy for training.
        """
        import os as _os
        from pathlib import Path as _Path

        quest_id = self._quest_id or "__ephemeral__"
        update_num = int(update_number) if update_number is not None else 0
        scene_num = int(scene_id) if scene_id is not None else 0

        out_dir = _Path(self._sft_dir) / quest_id
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"u{update_num}_s{scene_num}_{trace.trace_id}.json"
        out_path = out_dir / fname

        payload: dict[str, Any] = {
            "quest_id": quest_id,
            "update_number": update_num,
            "scene_index": scene_num,
            "pipeline_trace_id": trace.trace_id,
            "rerank_source": rerank_source,
            "winner_index": winner_index,
            "craft_brief": brief_text,
            "candidates": [
                {
                    "index": r["index"],
                    "prose": r["prose"],
                    "weighted_score": r["score"],
                    "overall_score": r["overall_score"],
                    "dimension_scores": r["breakdown"],
                    "rerank_source": r["rerank_source"],
                }
                for r in records
            ],
        }

        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str))
        _os.replace(tmp_path, out_path)

        trace.add_stage(StageResult(
            stage_name="sft_collection",
            input_prompt="", raw_output="",
            detail={
                "path": str(out_path),
                "n_candidates": len(records),
                "winner_index": winner_index,
                "scene_index": scene_num,
                "update_number": update_num,
            },
        ))

    async def _run_write(
        self,
        trace: PipelineTrace,
        plan: "Union[dict, Any]",  # dict = old path, CraftPlan = new path
        *,
        voice_samples: list[str] | None = None,
        anti_patterns: list[str] | None = None,
        player_action: str | None = None,
        update_number: int | None = None,
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
                    "voice_anchors": [],
                    "quest_callbacks": [],
                    "voice_continuity": [],
                    "recent_prose_tail": "",
                    "player_action": player_action,
                    "narrator": self._narrator,
                    "scene_entities": [],
                },
            )
            if self._n_candidates == 1:
                t0 = time.perf_counter()
                raw = await self._client.chat(
                    messages=[
                        ChatMessage(role="system", content=write_ctx.system_prompt),
                        ChatMessage(role="user", content=write_ctx.user_prompt),
                    ],
                    temperature=0.8,
                    max_tokens=WRITE_MAX_TOKENS,
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
            prose, _records = await self._generate_scene_candidates(
                trace=trace,
                write_ctx=write_ctx,
                n=self._n_candidates,
                scene_id=None,
                craft_plan=None,
                player_action=player_action,
                update_number=update_number,
                brief_text=None,
            )
            return prose

        # ---- New CraftPlan path: one call per beat, within one scene ----
        # Build a brief lookup by scene_id
        briefs_by_scene: dict[int, Any] = {b.scene_id: b for b in plan.briefs}

        # Build an emotional-plan lookup by scene_id, if one was stashed by
        # the hierarchical run. Used by Wave 2b retrieval to enrich the
        # seed text with target emotion. Missing entries fall back to None.
        emotional_by_scene: dict[int, Any] = {}
        emo_plan = getattr(self, "_last_emotional", None)
        if emo_plan is not None:
            for s in getattr(emo_plan, "scenes", []) or []:
                emotional_by_scene[s.scene_id] = s

        # Build a dramatic-scene lookup so the writer can see the scene's
        # beat sheet. Beats drive the per-beat write loop; when missing or
        # trivial (<=1 beat) we fall back to a single write call.
        dramatic_by_scene: dict[int, Any] = {}
        dram_plan = getattr(self, "_last_dramatic", None)
        if dram_plan is not None:
            for s in getattr(dram_plan, "scenes", []) or []:
                dramatic_by_scene[s.scene_id] = s

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

            # Wave 2b: retrieve voice anchors for this scene. Returns [] when
            # retrieval is disabled, no retriever is wired, or the retriever
            # finds no hits; in all those cases the prompt template skips
            # the anchor block (default-off behavior preserved).
            voice_anchors = await self._retrieve_voice_anchors(
                scene=scene,
                emotional_scene=emotional_by_scene.get(scene.scene_id),
                brief_text=brief_text,
            )

            # Wave 3b: retrieve in-quest callbacks — previously-committed
            # narrative passages that touch the same entities / framing as
            # this scene. Same default-off semantics as voice anchors.
            quest_callbacks = await self._retrieve_quest_callbacks(
                scene=scene,
                brief_text=brief_text,
                craft_plan=plan,
            )

            # Wave 4c: retrieve per-POV-character past utterances for
            # voice continuity. Same default-off semantics as above.
            voice_continuity = await self._retrieve_voice_continuity(scene=scene)

            dramatic_scene = dramatic_by_scene.get(scene.scene_id)
            beats: list[str] = list(
                getattr(dramatic_scene, "beats", None) or []
            )

            scene_entities = self._resolve_scene_entities(scene)

            # Build the write context once per scene; per-beat fields are
            # overlaid in the inner loop below. When the scene has 0 or 1
            # beat, we still call once with beat=None (preserves prior
            # single-shot behavior).
            write_ctx = self._cb.build(
                spec=WRITE_SPEC,
                stage_name="write",
                templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
                extras={
                    "brief": brief_text,
                    "scene": scene.model_dump() if hasattr(scene, "model_dump") else scene,
                    "voice_samples": effective_voice_samples,
                    "voice_anchors": voice_anchors,
                    "quest_callbacks": quest_callbacks,
                    "voice_continuity": voice_continuity,
                    "recent_prose_tail": recent_prose_tail,
                    "anti_patterns": anti_patterns or [],
                    "plan": None,
                    "style": "",
                    "player_action": player_action,
                    "beat": None,
                    "beat_index": None,
                    "total_beats": len(beats),
                    "accumulated_scene_prose": "",
                    "scene_target_words": self._scene_target_words,
                    "words_so_far": 0,
                    "narrator": self._narrator,
                    "scene_entities": scene_entities,
                },
            )

            if self._n_candidates > 1 or len(beats) <= 1:
                # N>1 candidates or no real beat sheet: fall back to the
                # legacy per-scene single-call behavior. Multi-candidate
                # rerank over a beat loop is out of scope for this change.
                if self._n_candidates == 1:
                    t0 = time.perf_counter()
                    raw = await self._client.chat(
                        messages=[
                            ChatMessage(role="system", content=write_ctx.system_prompt),
                            ChatMessage(role="user", content=write_ctx.user_prompt),
                        ],
                        temperature=0.8,
                        max_tokens=WRITE_MAX_TOKENS,
                    )
                    latency = int((time.perf_counter() - t0) * 1000)
                    scene_prose = OutputParser.parse_prose(raw)
                    trace.add_stage(StageResult(
                        stage_name="write",
                        input_prompt=write_ctx.user_prompt,
                        raw_output=raw,
                        parsed_output=scene_prose,
                        token_usage=TokenUsage(prompt=write_ctx.token_estimate),
                        latency_ms=latency,
                        detail={"scene_id": scene.scene_id, "beats": len(beats)},
                    ))
                else:
                    scene_prose, _records = await self._generate_scene_candidates(
                        trace=trace,
                        write_ctx=write_ctx,
                        n=self._n_candidates,
                        scene_id=scene.scene_id,
                        craft_plan=plan,
                        player_action=player_action,
                        update_number=update_number,
                        brief_text=brief_text,
                    )
            else:
                # Beat loop: one LLM call per beat, threading accumulated
                # prose so the next call sees everything written so far in
                # this scene.
                scene_parts: list[str] = []
                accumulated = ""
                for beat_index, beat_text in enumerate(beats):
                    words_so_far = len(accumulated.split())
                    beat_ctx = self._cb.build(
                        spec=WRITE_SPEC,
                        stage_name="write",
                        templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
                        extras={
                            "brief": brief_text,
                            "scene": scene.model_dump() if hasattr(scene, "model_dump") else scene,
                            "voice_samples": effective_voice_samples,
                            "voice_anchors": voice_anchors,
                            "quest_callbacks": quest_callbacks,
                            "voice_continuity": voice_continuity,
                            "recent_prose_tail": recent_prose_tail,
                            "anti_patterns": anti_patterns or [],
                            "plan": None,
                            "style": "",
                            "player_action": player_action if beat_index == 0 else None,
                            "beat": beat_text,
                            "beat_index": beat_index,
                            "total_beats": len(beats),
                            "accumulated_scene_prose": accumulated,
                            "scene_target_words": self._scene_target_words,
                            "words_so_far": words_so_far,
                            "narrator": self._narrator,
                            "scene_entities": scene_entities,
                        },
                    )
                    t0 = time.perf_counter()
                    raw = await self._client.chat(
                        messages=[
                            ChatMessage(role="system", content=beat_ctx.system_prompt),
                            ChatMessage(role="user", content=beat_ctx.user_prompt),
                        ],
                        temperature=0.8,
                        max_tokens=WRITE_MAX_TOKENS,
                    )
                    latency = int((time.perf_counter() - t0) * 1000)
                    beat_prose = OutputParser.parse_prose(raw)
                    scene_parts.append(beat_prose)
                    accumulated = (
                        accumulated + "\n\n" + beat_prose if accumulated else beat_prose
                    )
                    trace.add_stage(StageResult(
                        stage_name="write",
                        input_prompt=beat_ctx.user_prompt,
                        raw_output=raw,
                        parsed_output=beat_prose,
                        token_usage=TokenUsage(prompt=beat_ctx.token_estimate),
                        latency_ms=latency,
                        detail={
                            "scene_id": scene.scene_id,
                            "beat_index": beat_index,
                            "total_beats": len(beats),
                            "words_so_far": words_so_far,
                        },
                    ))
                scene_prose = "\n\n".join(scene_parts)
            prose_parts.append(scene_prose)

            # Track last ~300 chars for rhythm continuity on next scene
            full_so_far = "\n\n".join(prose_parts)
            recent_prose_tail = full_so_far[-300:] if len(full_so_far) > 300 else full_so_far

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
        all_known_entities = [e for e in entities if e.status != EntityStatus.DESTROYED]

        # Gap G4: surface current themes (with proposition + stance) so the
        # extractor can emit theme_stance_updates on material shifts.
        themes: list[dict[str, Any]] = []
        if self._quest_id is not None:
            try:
                for th in self._world.list_themes(self._quest_id):
                    themes.append({
                        "id": th.id,
                        "proposition": th.proposition,
                        "stance": th.stance,
                    })
            except Exception:
                themes = []

        # Gap G13: surface per-character *active* unconscious motives so the
        # extractor can emit motive_resolutions when prose shows a character
        # confronting / moving past one.
        from app.planning.motives import unconscious_motives_for
        character_motives: list[dict[str, Any]] = []
        for e in active_entities:
            try:
                active_motives = unconscious_motives_for(e)
            except Exception:
                active_motives = []
            if not active_motives:
                continue
            character_motives.append({
                "character_id": e.id,
                "character_name": e.name,
                "motives": [
                    {"id": m.id, "motive": m.motive} for m in active_motives
                ],
            })

        ctx = self._cb.build(
            spec=EXTRACT_SPEC,
            stage_name="extract",
            templates={
                "system": "stages/extract/system.j2",
                "user": "stages/extract/user.j2",
            },
            extras={
                "plan": plan_text,
                "prose": prose,
                "entities": active_entities,
                "themes": themes,
                "character_motives": character_motives,
            },
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

        known_ids = {e.id for e in all_known_entities}
        delta, build_issues = build_delta(
            extracted, update_number, known_ids=known_ids, world=self._world,
        )

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

        # Activate DORMANT entities the dramatic planner chose to surface.
        dramatic = getattr(self, "_last_dramatic", None)
        if dramatic is not None:
            to_surface = getattr(dramatic, "entities_to_surface", None) or []
            for eid in to_surface:
                for lookup in (eid, f"char:{eid}"):
                    try:
                        ent = self._world.get_entity(lookup)
                        if ent.status == EntityStatus.DORMANT:
                            self._world.update_entity(lookup, {
                                "status": EntityStatus.ACTIVE.value,
                                "last_referenced_update": update_number,
                            })
                        break
                    except Exception:
                        continue

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
            max_tokens=REVISE_MAX_TOKENS,
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
