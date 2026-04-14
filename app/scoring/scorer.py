"""Day 2: 12-dimension heuristic scorer (+ Day 6 LLM-judge extension).

The ``Scorer`` turns committed prose into a ``Scorecard`` — a fixed set of
twelve [0, 1] dimension scores plus an unweighted-mean ``overall_score``.
Day 6 adds an optional async ``score_with_llm_judges`` that tacks on three
LLM-judged dims (``tension_execution``, ``emotional_trajectory``,
``choice_hook_quality``) via a single batched structured call.

Design choices
--------------
- **Day 2 core stays sync + heuristic-only.** ``Scorer.score()`` does NOT
  call any LLM; pipelines that don't wire an ``llm_judge_client`` see
  bit-identical Day 2 behavior.
- **Day 6 dims are opt-in.** Construct ``Scorer(llm_judge_client=...)`` and
  call ``await scorer.score_with_llm_judges(...)``. Returns an
  ``ExtendedScorecard`` with the original 12 dims plus an
  ``llm_judge_scores`` dict of the three Day 6 dims.
- **Single batched call.** Re-uses ``app.calibration.judges.BatchJudge``,
  restricted to the Day 6 dim names. One structured response per passage.
- **Graceful degradation.** Every dim accepts missing inputs and returns a
  neutral value. A scorer called with ``craft_plan=None`` still produces
  12 dims — the craft-plan-dependent critics simply emit empty issue
  lists and score 1.0.
- **Two families of scores.** Heuristic dims wrap ``app.calibration.heuristics``
  primitives directly; critic dims run a validator from
  ``app.planning.critics`` and convert its ``ValidationIssue`` list to a
  scalar via ``app.calibration.scorer.critic_score`` (errors weigh 0.25,
  warnings 0.10).
"""
from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, Field

from app.calibration import heuristics as _heur
from app.calibration.scorer import critic_score
from app.planning import critics as _critics


DIMENSION_NAMES: tuple[str, ...] = (
    # Heuristic — app.calibration.heuristics
    "sentence_variance",
    "dialogue_ratio",
    "pacing",
    "sensory_density",
    # Critic-derived — app.planning.critics via critic_score()
    "free_indirect_quality",
    "detail_characterization",
    "metaphor_domains_score",
    "indirection_score",
    "pov_adherence",
    "named_entity_presence",
    "narrator_sensory_match",
    "action_fidelity",
)


DIMENSION_SOURCES: dict[str, str] = {
    "sentence_variance": "heuristic",
    "dialogue_ratio": "heuristic",
    "pacing": "heuristic",
    "sensory_density": "heuristic",
    "free_indirect_quality": "critic",
    "detail_characterization": "critic",
    "metaphor_domains_score": "critic",
    "indirection_score": "critic",
    "pov_adherence": "critic",
    "named_entity_presence": "critic",
    "narrator_sensory_match": "critic",
    "action_fidelity": "critic",
}


# ---------------------------------------------------------------------------
# Day 6: LLM-judge dims
# ---------------------------------------------------------------------------

# The three LLM-judged dimensions added on Day 6. Each has an anchored-scale
# prompt at ``prompts/scoring/dims/<name>.j2``; all three are scored in a
# single batched structured call via
# ``app.calibration.judges.BatchJudge.score``. These are persisted alongside
# the Day 2 heuristic/critic dims in the ``dimension_scores`` table — the
# header row carries a single ``overall_score`` that still reflects only
# the Day 2 arithmetic mean (so the dashboard number stays comparable
# across quests regardless of whether the async judge task ran).
LLM_JUDGE_DIMS: tuple[str, ...] = (
    "tension_execution",
    "emotional_trajectory",
    "choice_hook_quality",
)


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))


class Scorecard(BaseModel):
    """12-dimension scorecard for a committed chapter / scene.

    Fields are named to match ``DIMENSION_NAMES``. Each is a float in
    ``[0, 1]``. ``overall_score`` is the unweighted arithmetic mean —
    rerank may apply its own weighting (see
    ``app.engine.pipeline.DEFAULT_RERANK_WEIGHTS``); we persist the raw
    mean for the dashboard.
    """

    # ---- heuristic ----
    sentence_variance: float = Field(ge=0.0, le=1.0)
    """Std-dev of sentence lengths (words) normalized; source: heuristic."""
    dialogue_ratio: float = Field(ge=0.0, le=1.0)
    """Fraction of characters inside quoted-speech spans; source: heuristic."""
    pacing: float = Field(ge=0.0, le=1.0)
    """Sentences per 100 words normalized; source: heuristic."""
    sensory_density: float = Field(ge=0.0, le=1.0)
    """Sensory-keyword hits per 100 words normalized; source: heuristic."""

    # ---- critic-derived ----
    free_indirect_quality: float = Field(ge=0.0, le=1.0)
    """Bleed / excluded vocabulary adherence from VoicePermeability;
    source: critic (validate_free_indirect_integrity)."""
    detail_characterization: float = Field(ge=0.0, le=1.0)
    """Perceptual-preoccupation presence for character_revealing detail
    mode; source: critic (validate_detail_characterization)."""
    metaphor_domains_score: float = Field(ge=0.0, le=1.0)
    """Absence of forbidden metaphor domains; source: critic
    (validate_metaphor_domains)."""
    indirection_score: float = Field(ge=0.0, le=1.0)
    """Absence of what_not_to_say phrases; source: critic
    (validate_indirection)."""
    pov_adherence: float = Field(ge=0.0, le=1.0)
    """Second-person pronoun ratio vs first-person drift; source: critic
    (validate_pov_adherence)."""
    named_entity_presence: float = Field(ge=0.0, le=1.0)
    """At least one active named entity appears in prose; source: critic
    (validate_named_entity_presence)."""
    narrator_sensory_match: float = Field(ge=0.0, le=1.0)
    """Prose sensory distribution vs ``narrator.sensory_bias`` L1 distance;
    source: critic (validate_narrator_sensory_distribution). Future LLM
    judge may replace this dim with a holistic narrator-voice judgement."""
    action_fidelity: float = Field(ge=0.0, le=1.0)
    """Content-word overlap between player action and prose; source: critic
    (validate_action_fidelity)."""

    # ---- aggregate ----
    overall_score: float = Field(ge=0.0, le=1.0)
    """Unweighted arithmetic mean of the 12 dim scores."""

    def dimension_items(self) -> list[tuple[str, float]]:
        """Return (dim_name, score) pairs in ``DIMENSION_NAMES`` order.

        Excludes ``overall_score`` — that's stored in its own column.
        """
        return [(name, getattr(self, name)) for name in DIMENSION_NAMES]


class ExtendedScorecard(BaseModel):
    """Day 6 wrapper: base :class:`Scorecard` + LLM-judge dim scores.

    Produced by :meth:`Scorer.score_with_llm_judges`. The base scorecard is
    the exact Day 2 artifact (same ``overall_score``, same dim fields);
    ``llm_judge_scores`` carries the three additional dim scores with
    rationales. Persistence writes both into ``dimension_scores`` under the
    same ``scorecard_id``.
    """

    base: Scorecard
    llm_judge_scores: dict[str, float] = Field(default_factory=dict)
    llm_judge_rationales: dict[str, str] = Field(default_factory=dict)

    def all_dimension_items(self) -> list[tuple[str, float]]:
        """Return every dim (Day 2 + Day 6) as (name, score) pairs.

        Used by the pipeline's async-persist path to write the full set of
        ``dimension_scores`` rows under one scorecard header.
        """
        return list(self.base.dimension_items()) + [
            (name, self.llm_judge_scores[name])
            for name in LLM_JUDGE_DIMS
            if name in self.llm_judge_scores
        ]


class Scorer:
    """Produce a :class:`Scorecard` from committed prose.

    Construction is cheap — no I/O, no model loads. Reuse one instance
    across calls; it holds no per-chapter state.

    Day 6: pass ``llm_judge_client`` to enable the async
    :meth:`score_with_llm_judges` method. The sync :meth:`score` never
    calls the client regardless — Day 2 semantics are preserved.
    """

    def __init__(
        self,
        *,
        llm_judge_client: Any | None = None,
        prompts_dir: str | Path | None = None,
    ) -> None:
        self._llm_judge_client = llm_judge_client
        self._prompts_dir: Path = (
            Path(prompts_dir) if prompts_dir is not None
            else _default_prompts_dir()
        )
        self._batch_judge: Any | None = None  # lazy

    @property
    def has_llm_judge(self) -> bool:
        return self._llm_judge_client is not None

    def score(
        self,
        prose: str,
        *,
        craft_plan: Any | None = None,
        narrator: Any | None = None,
        world: Any | None = None,
        player_action: str | None = None,
        character_id_for_voice: str | None = None,  # noqa: ARG002 — reserved for LLM-judge dims (Day 6)
    ) -> Scorecard:
        """Score a single prose passage.

        Parameters
        ----------
        prose:
            The committed text to evaluate. Empty / whitespace-only prose
            yields an all-zero scorecard rather than raising.
        craft_plan:
            ``CraftPlan`` for the chapter. When ``None``, craft-derived
            critics (free-indirect, detail, metaphor, indirection) see no
            constraints and score 1.0 (no detectable violation).
        narrator:
            ``Narrator`` instance whose ``sensory_bias`` drives the
            narrator-sensory-match dim. ``None`` ⇒ the critic returns
            empty issues ⇒ score 1.0.
        world:
            ``WorldStateManager`` — used to fetch active entity names for
            the entity-presence dim. ``None`` ⇒ empty candidate list ⇒
            the critic returns [] ⇒ score 1.0.
        player_action:
            The player's input for this update. ``None`` / empty string ⇒
            action-fidelity skipped (score 1.0, treated as "nothing to
            contradict").
        character_id_for_voice:
            Reserved for Day 6 LLM-judge dims (voice-distinctiveness per
            POV character). Ignored in v1.
        """
        prose = prose or ""

        # ---- heuristic dims ----
        sent_var = _clip01(_heur.sentence_variance(prose))
        dia_ratio = _clip01(_heur.dialogue_ratio(prose))
        pace = _clip01(_heur.pacing(prose))
        sens_dens = _clip01(_heur.sensory_density(prose))

        # ---- critic-derived dims ----
        if craft_plan is not None:
            fi_issues = _safe(
                _critics.validate_free_indirect_integrity, craft_plan, prose
            )
            dc_issues = _safe(
                _critics.validate_detail_characterization, craft_plan, prose
            )
            md_issues = _safe(
                _critics.validate_metaphor_domains, craft_plan, prose
            )
            ind_issues = _safe(
                _critics.validate_indirection, craft_plan, prose
            )
        else:
            fi_issues = dc_issues = md_issues = ind_issues = []

        pov_issues = _safe(_critics.validate_pov_adherence, prose)

        if world is not None:
            entity_names = _active_entity_names(world)
        else:
            entity_names = []
        ne_issues = _safe(
            _critics.validate_named_entity_presence, prose, entity_names
        )

        ns_issues = _safe(
            _critics.validate_narrator_sensory_distribution, narrator, prose
        )

        if player_action:
            af_issues = _safe(
                _critics.validate_action_fidelity, prose, player_action
            )
        else:
            af_issues = []

        free_indirect_quality = _clip01(critic_score(fi_issues))
        detail_characterization = _clip01(critic_score(dc_issues))
        metaphor_domains_score = _clip01(critic_score(md_issues))
        indirection_score = _clip01(critic_score(ind_issues))
        pov_adherence = _clip01(critic_score(pov_issues))
        named_entity_presence = _clip01(critic_score(ne_issues))
        narrator_sensory_match = _clip01(critic_score(ns_issues))
        action_fidelity = _clip01(critic_score(af_issues))

        dims: dict[str, float] = {
            "sentence_variance": sent_var,
            "dialogue_ratio": dia_ratio,
            "pacing": pace,
            "sensory_density": sens_dens,
            "free_indirect_quality": free_indirect_quality,
            "detail_characterization": detail_characterization,
            "metaphor_domains_score": metaphor_domains_score,
            "indirection_score": indirection_score,
            "pov_adherence": pov_adherence,
            "named_entity_presence": named_entity_presence,
            "narrator_sensory_match": narrator_sensory_match,
            "action_fidelity": action_fidelity,
        }
        overall = _clip01(fmean(dims.values()))

        return Scorecard(overall_score=overall, **dims)

    async def score_with_llm_judges(
        self,
        prose: str,
        *,
        work_id: str = "quest",
        pov: str = "second",
        is_quest: bool = True,
        craft_plan: Any | None = None,
        narrator: Any | None = None,
        world: Any | None = None,
        player_action: str | None = None,
    ) -> ExtendedScorecard:
        """Day 6: score ``prose`` with heuristic + LLM-judge dims.

        Runs the synchronous :meth:`score` first, then issues ONE batched
        structured call for the three Day 6 dims
        (``tension_execution``, ``emotional_trajectory``,
        ``choice_hook_quality``). The returned :class:`ExtendedScorecard`
        exposes the base scorecard unchanged and the three additional dim
        scores under ``llm_judge_scores``.

        Parameters
        ----------
        work_id, pov, is_quest:
            Metadata that flows into the batch judge prompt. ``is_quest``
            does not gate the Day 6 dims — all three apply to any prose —
            but it is passed through for trace-log fidelity.

        Raises
        ------
        RuntimeError
            If no ``llm_judge_client`` was supplied at construction time.
            Callers wanting heuristic-only scoring should call
            :meth:`score` directly.

        Notes
        -----
        This method is async but does NOT launch tasks; callers who want
        fire-and-forget behavior wrap it with ``asyncio.create_task``.
        The pipeline's post-commit hook does exactly that — the
        scorecard header and heuristic dims land synchronously at commit,
        and the three LLM-judge dim rows are persisted when the async
        task resolves.
        """
        if self._llm_judge_client is None:
            raise RuntimeError(
                "score_with_llm_judges() called on a Scorer constructed "
                "without llm_judge_client"
            )

        base = self.score(
            prose,
            craft_plan=craft_plan,
            narrator=narrator,
            world=world,
            player_action=player_action,
        )

        judge = self._get_batch_judge()
        judge_scores = await judge.score(
            client=self._llm_judge_client,
            passage=prose,
            work_id=work_id,
            pov=pov,
            is_quest=is_quest,
            dim_names=list(LLM_JUDGE_DIMS),
        )
        scores = {d: _clip01(judge_scores[d].score) for d in LLM_JUDGE_DIMS}
        rationales = {d: judge_scores[d].rationale for d in LLM_JUDGE_DIMS}
        return ExtendedScorecard(
            base=base,
            llm_judge_scores=scores,
            llm_judge_rationales=rationales,
        )

    def _get_batch_judge(self) -> Any:
        """Lazy-construct the BatchJudge; cached between calls."""
        if self._batch_judge is None:
            # Local import — the calibration package pulls in yaml/jinja that
            # we'd rather not pay for at cold-start when LLM dims are off.
            from app.calibration.judges import BatchJudge
            self._batch_judge = BatchJudge(self._prompts_dir)
        return self._batch_judge


def _default_prompts_dir() -> Path:
    """Resolve ``prompts/`` relative to the project root.

    ``app/scoring/scorer.py`` -> project root / prompts.
    """
    return Path(__file__).resolve().parent.parent.parent / "prompts"


def _safe(fn, *args, **kwargs) -> list:
    """Call a critic; swallow any exception into an empty issue list.

    Critics are stated to never raise, but a future regression or a
    non-default validator plugged into this spot shouldn't be allowed to
    take down scoring. A crash yields score = 1.0 (no detected issues),
    which is indistinguishable from a clean pass — acceptable for v1 and
    made observable by the pipeline's trace hook.
    """
    try:
        out = fn(*args, **kwargs)
    except Exception:  # pragma: no cover - defensive
        return []
    return list(out) if out is not None else []


def _active_entity_names(world: Any) -> list[str]:
    """Collect entity names from a ``WorldStateManager``-like object.

    Tolerant to partial fakes: anything with ``list_entities()`` returning
    objects that expose a ``name`` attribute works.
    """
    try:
        entities = world.list_entities()
    except Exception:  # pragma: no cover - defensive
        return []
    return [
        getattr(e, "name", None)
        for e in (entities or [])
        if getattr(e, "name", None)
    ]
