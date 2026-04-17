"""Refinement framework (Phase 5 of story-rollout).

Three pluggable selectors identify chapters to retry; the framework
regenerates each one with strategy-specific guidance, scores the new
prose, and accepts the refinement only when it materially beats the
baseline (per spec: mean +≥0.05 AND min per-dim regression > -0.10).

Usage::

    targets = WeakChapterSelector(world).select(quest_id, max_targets=3)
    results = await run_refinement_pass(
        targets=targets, world=world, client=client,
        rollout_dir=quests_dir / qid / "rollouts" / rollout_id,
        quest_id=qid,
    )
"""
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.engine import (
    ContextBuilder, Pipeline, PromptRenderer, TokenBudget, TraceStore,
)
from app.rollout.scorer import compare_chapters_corrected, score_chapter, DEFAULT_DIMS
from app.runtime.client import InferenceClient
from app.world.db import open_db
from app.world.schema import (
    RefinementAttempt, RefinementStrategy, RolloutChapter,
)
from app.world.state_manager import WorldStateManager


# Spec exit criterion thresholds
ACCEPT_MEAN_DELTA = 0.05
REJECT_DIM_REGRESSION = -0.10


@dataclass
class RefinementTarget:
    """One (rollout_id, chapter_index) chosen for refinement.

    ``baseline_scores`` is the per-dim scores the refined prose must
    beat (mean by ACCEPT_MEAN_DELTA, no dim regressing by more than
    -REJECT_DIM_REGRESSION).
    """
    rollout_id: str
    chapter_index: int
    quest_id: str
    strategy: str
    reason: str
    guidance: str
    baseline_scores: dict[str, float]


class RefinementSelector(Protocol):
    """A selector picks targets from world state given a strategy."""

    name: str

    def select(
        self, *, quest_id: str, rollout_id: str | None = None,
        max_targets: int = 3,
    ) -> list[RefinementTarget]: ...


@dataclass
class RefinementResult:
    target: RefinementTarget
    attempt_id: str
    refined_prose: str
    refined_scores: dict[str, float]
    delta_mean: float
    delta_min: float
    accepted: bool
    rejection_reason: str | None = None


def _build_pipeline_for_rollout(
    rollout_dir: Path, client: InferenceClient, quest_id: str,
) -> tuple[Pipeline, WorldStateManager, TraceStore]:
    """Construct a Pipeline pointed at the rollout's isolated world DB.

    Mirrors app.rollout.harness._build_pipeline but exposed so the
    refinement framework can run pipelines against rollout DBs without
    importing the harness.
    """
    REPO = Path(__file__).resolve().parent.parent.parent
    PROMPTS = REPO / "prompts"
    from app.craft.library import CraftLibrary
    from app.planning.arc_planner import ArcPlanner
    from app.planning.craft_planner import CraftPlanner
    from app.planning.dramatic_planner import DramaticPlanner
    from app.planning.emotional_planner import EmotionalPlanner

    renderer = PromptRenderer(PROMPTS)
    rollout_conn = open_db(rollout_dir / "quest.db")
    rollout_sm = WorldStateManager(rollout_conn)
    cb = ContextBuilder(rollout_sm, renderer, TokenBudget())
    craft_library = CraftLibrary(REPO / "app" / "craft" / "data")
    try:
        structure = craft_library.structure("three_act")
    except Exception:
        structure = None

    config_path = rollout_dir / "config.json"
    quest_config: dict = {}
    if config_path.is_file():
        try:
            quest_config = json.loads(config_path.read_text())
        except Exception:
            quest_config = {}

    trace_store = TraceStore(rollout_dir / "traces")

    pipeline = Pipeline(
        rollout_sm, cb, client,
        arc_planner=ArcPlanner(client, renderer),
        dramatic_planner=DramaticPlanner(client, renderer, craft_library),
        emotional_planner=EmotionalPlanner(client, renderer),
        craft_planner=CraftPlanner(client, renderer, craft_library),
        craft_library=craft_library, structure=structure,
        quest_config=quest_config, quest_id=quest_id, arc_id="main",
        live_trace_save=trace_store.save,
    )
    return pipeline, rollout_sm, trace_store


def _evaluate_deltas(
    baseline: dict[str, float], refined: dict[str, float],
) -> tuple[float, float]:
    """Return (mean_delta, min_per_dim_delta).

    mean_delta = mean(refined) - mean(baseline). Computed over the
    intersection of dim names — if refined drops a dim, that dim is not
    counted in the mean (but its absence is itself a regression we'd
    want to flag; for now the strict-min check handles common dims only).
    """
    common = sorted(set(baseline) & set(refined))
    if not common:
        return 0.0, 0.0
    deltas = [refined[d] - baseline[d] for d in common]
    mean_delta = sum(deltas) / len(deltas)
    min_delta = min(deltas)
    return mean_delta, min_delta


async def refine_one(
    *,
    target: RefinementTarget,
    main_world: WorldStateManager,
    rollout_dir: Path,
    client: InferenceClient,
    quest_id: str,
) -> RefinementResult:
    """Regenerate one chapter with strategy guidance, score it, decide
    accept/reject, persist a RefinementAttempt row.
    """
    attempt_id = f"ref_{uuid.uuid4().hex[:8]}"

    # Pull the existing chapter to use its player_action
    chapters = main_world.list_rollout_chapters(target.rollout_id)
    chapter_row = next(
        (c for c in chapters if c.chapter_index == target.chapter_index),
        None,
    )
    if chapter_row is None:
        raise ValueError(
            f"chapter {target.chapter_index} not found for rollout {target.rollout_id!r}"
        )

    # Build the pipeline against the rollout's isolated world. We use the
    # CURRENT world state (post-original-chapter mutations) — this is a
    # v1 simplification; a perfect refinement would snapshot pre-chapter
    # state per chapter.
    pipeline, rollout_sm, trace_store = _build_pipeline_for_rollout(
        rollout_dir, client, quest_id,
    )

    # Inject refinement guidance via the player_action prefix. The arc
    # and dramatic planners read player_action as the directive trigger;
    # prepending refinement guidance gives the planners explicit
    # instructions without wholesale changing the planner schemas.
    refined_action = (
        f"[REFINEMENT GUIDANCE — {target.strategy}] {target.guidance}\n\n"
        f"Original action: {chapter_row.player_action}"
    )

    try:
        out = await pipeline.run(
            player_action=refined_action,
            update_number=target.chapter_index,
        )
        refined_prose = out.prose
        trace_id = out.trace.trace_id
    finally:
        # Close the rollout connection from _build_pipeline_for_rollout
        rollout_sm._conn.close()

    # Compare refined vs original using bias-corrected dual rating.
    # This replaces absolute re-scoring — the dual rating eliminates
    # quantization and position bias issues. Each dim gets a corrected
    # delta and per-chapter scores.
    use_dims = list(DEFAULT_DIMS)
    refined_scores: dict[str, float] = {}
    dim_deltas: dict[str, float] = {}
    for dim in use_dims:
        cmp = await compare_chapters_corrected(
            client=client, text_a=refined_prose,
            text_b=chapter_row.prose, dim=dim,
        )
        refined_scores[dim] = cmp["a_score"]
        dim_deltas[dim] = cmp["delta_corrected"]

    # Accept/reject from the dual-rating deltas
    if dim_deltas:
        mean_delta = sum(dim_deltas.values()) / len(dim_deltas)
        min_delta = min(dim_deltas.values())
    else:
        mean_delta, min_delta = 0.0, 0.0

    if mean_delta >= ACCEPT_MEAN_DELTA and min_delta > REJECT_DIM_REGRESSION:
        accepted = True
        rejection_reason = None
        chapter_row.prose = refined_prose
        chapter_row.judge_scores = refined_scores
        chapter_row.trace_id = trace_id
        main_world.save_rollout_chapter(chapter_row)
        kb_payload = {
            dim: {"score": refined_scores[dim], "rationale": f"refined, delta={dim_deltas[dim]:+.3f}"}
            for dim in use_dims if dim in refined_scores
        }
        main_world.save_chapter_scores(
            target.rollout_id, target.chapter_index, kb_payload,
        )
    else:
        accepted = False
        if mean_delta < ACCEPT_MEAN_DELTA:
            rejection_reason = (
                f"mean delta {mean_delta:+.3f} < threshold {ACCEPT_MEAN_DELTA:+.3f}"
            )
        else:
            rejection_reason = (
                f"min per-dim regression {min_delta:+.3f} <= "
                f"{REJECT_DIM_REGRESSION:+.3f}"
            )

    # Always persist the attempt for audit
    attempt = RefinementAttempt(
        id=attempt_id, quest_id=quest_id, rollout_id=target.rollout_id,
        chapter_index=target.chapter_index, strategy=target.strategy,
        reason=target.reason, guidance=target.guidance,
        baseline_scores=target.baseline_scores,
        refined_prose=refined_prose, refined_scores=refined_scores,
        refined_trace_id=trace_id,
        delta_mean=mean_delta, delta_min=min_delta,
        accepted=accepted, rejection_reason=rejection_reason,
    )
    main_world.save_refinement_attempt(attempt)

    return RefinementResult(
        target=target, attempt_id=attempt_id,
        refined_prose=refined_prose, refined_scores=refined_scores,
        delta_mean=mean_delta, delta_min=min_delta,
        accepted=accepted, rejection_reason=rejection_reason,
    )


async def run_refinement_pass(
    *,
    targets: list[RefinementTarget],
    quests_dir: Path,
    main_world: WorldStateManager,
    client: InferenceClient,
) -> list[RefinementResult]:
    """Run refine_one over a list of targets, sequentially.

    Sequential because each refinement is an LLM-bound chapter generation;
    parallelism gains are limited by the model server. Returns one
    RefinementResult per target.
    """
    results: list[RefinementResult] = []
    for t in targets:
        rollout_dir = quests_dir / t.quest_id / "rollouts" / t.rollout_id
        try:
            r = await refine_one(
                target=t, main_world=main_world,
                rollout_dir=rollout_dir, client=client,
                quest_id=t.quest_id,
            )
        except Exception as e:
            # Don't let one target break the whole pass
            r = RefinementResult(
                target=t, attempt_id="(failed)",
                refined_prose="", refined_scores={},
                delta_mean=0.0, delta_min=0.0,
                accepted=False, rejection_reason=f"refinement crashed: {e}",
            )
        results.append(r)
    return results
