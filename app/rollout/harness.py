"""Rollout harness — synthetic playthroughs of picked story candidates.

Given a picked candidate + a virtual-player profile, orchestrate a full
chapter-by-chapter pipeline run. Each chapter is persisted incrementally
to ``rollout_chapters`` so the harness is resume-safe: a crash at chapter
N+1 re-bootstraps and skips 1..N on restart.

Per rollout, the harness maintains an isolated world DB under
``data/quests/<qid>/rollouts/<rid>/`` so the playthrough's entity
activations and narrative history don't pollute the main quest state.

See Phase 3 of the story-rollout architecture spec.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget, TraceStore
from app.runtime.client import InferenceClient
from app.world.db import open_db
from app.world.schema import (
    RolloutChapter, RolloutExtract, RolloutRun, RolloutStatus, StoryCandidate,
)
from app.world.state_manager import WorldStateManager

from .action_selector import select_action
from .profiles import VirtualPlayerProfile, load_profile


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rollout_dir(quests_dir: Path, qid: str, rid: str) -> Path:
    return quests_dir / qid / "rollouts" / rid


def _opening_action_from_candidate(
    cand: StoryCandidate, skeleton_ch1_beats: list[str] | None,
) -> str:
    """Derive a reasonable opening player-action for chapter 1.

    Prefer the skeleton's first chapter's plot beats (most specific); fall
    back to the candidate's synopsis sliced to a sentence.
    """
    if skeleton_ch1_beats:
        return " ".join(skeleton_ch1_beats)
    # Take the first sentence of the synopsis
    syn = (cand.synopsis or cand.title or "Begin the story.").strip()
    for punct in (". ", "! ", "? "):
        if punct in syn:
            return syn.split(punct, 1)[0] + punct.strip()
    return syn[:240]


def _build_pipeline(
    rollout_sm: WorldStateManager, rollout_dir: Path,
    client: InferenceClient, quest_id: str,
) -> Pipeline:
    """Construct a Pipeline pointed at the rollout's isolated world DB."""
    REPO = Path(__file__).resolve().parent.parent.parent
    PROMPTS = REPO / "prompts"
    from app.craft.library import CraftLibrary
    from app.planning.arc_planner import ArcPlanner
    from app.planning.craft_planner import CraftPlanner
    from app.planning.dramatic_planner import DramaticPlanner
    from app.planning.emotional_planner import EmotionalPlanner

    renderer = PromptRenderer(PROMPTS)
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

    return Pipeline(
        rollout_sm, cb, client,
        arc_planner=ArcPlanner(client, renderer),
        dramatic_planner=DramaticPlanner(client, renderer, craft_library),
        emotional_planner=EmotionalPlanner(client, renderer),
        craft_planner=CraftPlanner(client, renderer, craft_library),
        craft_library=craft_library, structure=structure,
        quest_config=quest_config, quest_id=quest_id, arc_id="main",
        live_trace_save=trace_store.save,
    )


def _bootstrap_rollout_world(
    main_db_path: Path, rollout_dir: Path, main_config_path: Path,
) -> None:
    """Create isolated rollout DB + config.json if they don't already exist.

    Copies the main quest DB (which already contains the seeded entities,
    plot threads, foreshadowing, rules, themes, motifs, story candidates,
    AND the arc skeleton) and clears out any chapters that were committed
    in the main quest. The rollout thus starts from "seed + picked +
    skeleton" — the same state a fresh playthrough would see.
    """
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "traces").mkdir(exist_ok=True)
    db_path = rollout_dir / "quest.db"
    if not db_path.is_file():
        shutil.copyfile(main_db_path, db_path)
        # Reset playthrough state — rollouts start from zero.
        conn = open_db(db_path)
        try:
            conn.execute("DELETE FROM narrative")
            conn.execute("DELETE FROM narrative_embeddings")
            conn.execute("DELETE FROM timeline")
            conn.commit()
        finally:
            conn.close()

    cfg_dest = rollout_dir / "config.json"
    if not cfg_dest.is_file():
        if main_config_path.is_file():
            shutil.copyfile(main_config_path, cfg_dest)
        else:
            cfg_dest.write_text("{}")


async def run_rollout(
    *,
    quests_dir: Path,
    quest_id: str,
    rollout_id: str,
    client: InferenceClient,
) -> RolloutRun:
    """Execute (or resume) a single rollout end-to-end.

    Parameters
    ----------
    quests_dir:
        Root quests dir (e.g. ``data/quests``).
    quest_id:
        The parent quest's id.
    rollout_id:
        The rollout_runs row id. Must already exist.

    Returns the final RolloutRun row (status = COMPLETE or FAILED).
    """
    main_quest_dir = quests_dir / quest_id
    main_db_path = main_quest_dir / "quest.db"
    main_cfg_path = main_quest_dir / "config.json"
    rollout_dir = _rollout_dir(quests_dir, quest_id, rollout_id)

    # Metadata lives in the main quest DB
    main_conn = open_db(main_db_path)
    main_sm = WorldStateManager(main_conn)
    try:
        run = main_sm.get_rollout(rollout_id)
        candidate = main_sm.get_story_candidate(run.candidate_id)
        skeleton = (
            main_sm.get_arc_skeleton(run.skeleton_id)
            if run.skeleton_id else
            main_sm.get_skeleton_for_candidate(run.candidate_id)
        )
        profile: VirtualPlayerProfile = load_profile(run.profile_id)

        # Prepare isolated rollout world (idempotent)
        _bootstrap_rollout_world(main_db_path, rollout_dir, main_cfg_path)

        # Already-committed chapters (resume support)
        completed = main_sm.list_rollout_chapters(rollout_id)
        start_index = max([c.chapter_index for c in completed], default=0) + 1

        main_sm.update_rollout(
            rollout_id, status=RolloutStatus.RUNNING,
            started_at=run.started_at or _utcnow_iso(),
        )

        # Build pipeline against the isolated rollout DB
        rollout_conn = open_db(rollout_dir / "quest.db")
        rollout_sm = WorldStateManager(rollout_conn)
        pipeline = _build_pipeline(rollout_sm, rollout_dir, client, quest_id)

        try:
            prior_choices: list = []
            recent_tail = ""
            # Replay already-committed chapters into scroll state (for
            # action selection context only — the rollout DB already has
            # the committed narrative).
            if completed:
                last = completed[-1]
                recent_tail = (last.prose or "")[-500:]

            for ch_idx in range(start_index, run.total_chapters_target + 1):
                # Decide the action for this chapter
                if ch_idx == 1:
                    sk_ch1 = None
                    if skeleton and skeleton.chapters:
                        sk_ch1 = skeleton.chapters[0]
                    player_action = _opening_action_from_candidate(
                        candidate,
                        sk_ch1.required_plot_beats if sk_ch1 else None,
                    )
                    rationale = "opening action derived from candidate + skeleton ch 1"
                else:
                    # Pull suggested_choices from the last chapter's trace
                    choices = prior_choices
                    chosen_idx, rationale = await select_action(
                        client=client, profile=profile,
                        choices=choices, recent_prose_tail=recent_tail,
                    )
                    if choices:
                        c = choices[chosen_idx]
                        player_action = (
                            c if isinstance(c, str)
                            else c.get("title", "") or c.get("description", "")
                        )
                    else:
                        player_action = "continue"

                # Run the pipeline for this chapter
                out = await pipeline.run(
                    player_action=player_action, update_number=ch_idx,
                )
                prose = out.prose
                recent_tail = (prose or "")[-500:]
                prior_choices = list(out.choices or [])

                # Persist chapter to main DB
                main_sm.save_rollout_chapter(RolloutChapter(
                    rollout_id=rollout_id, chapter_index=ch_idx,
                    player_action=player_action, prose=prose,
                    trace_id=out.trace.trace_id,
                    extract=RolloutExtract(),
                ))
                main_sm.update_rollout(
                    rollout_id, chapters_complete=ch_idx,
                )
        finally:
            rollout_conn.close()

        main_sm.update_rollout(
            rollout_id, status=RolloutStatus.COMPLETE,
            completed_at=_utcnow_iso(),
        )
        return main_sm.get_rollout(rollout_id)

    except Exception as e:
        try:
            main_sm.update_rollout(
                rollout_id, status=RolloutStatus.FAILED,
                error_message=str(e)[:500], completed_at=_utcnow_iso(),
            )
        except Exception:
            pass
        raise
    finally:
        main_conn.close()


def create_rollout_row(
    *, quests_dir: Path, quest_id: str, candidate_id: str,
    profile_id: str, total_chapters_target: int = 10,
    seed_nonce: int = 0,
) -> str:
    """Insert a pending RolloutRun and return its id. Caller then
    invokes ``run_rollout`` to start execution."""
    main_conn = open_db(quests_dir / quest_id / "quest.db")
    try:
        sm = WorldStateManager(main_conn)
        # Validate candidate exists
        cand = sm.get_story_candidate(candidate_id)
        # Attach current skeleton if present
        skel = sm.get_skeleton_for_candidate(candidate_id)
        rid = f"ro_{uuid.uuid4().hex[:8]}"
        sm.create_rollout(RolloutRun(
            id=rid, quest_id=quest_id, candidate_id=candidate_id,
            skeleton_id=(skel.id if skel else None),
            profile_id=profile_id,
            seed_nonce=seed_nonce,
            total_chapters_target=total_chapters_target,
            status=RolloutStatus.PENDING,
        ))
        return rid
    finally:
        main_conn.close()
