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

from app.planning.opening_critic import check_opening_repetition
from app.planning.voice_tracker import CharacterVoiceTracker

from .action_selector import select_action
from .kb_extractor import persist_chapter_kb
from .profiles import VirtualPlayerProfile, load_profile
from .scorer import score_and_persist_chapter


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
    score: bool = True,
) -> RolloutRun:
    """Execute (or resume) a single rollout end-to-end.

    Parameters
    ----------
    quests_dir:
        Root quests dir (e.g. ``data/quests``).
    quest_id:
        The parent quest's id.
    score:
        If True (default), runs the 8-dim chapter judge after each
        committed chapter and persists scores into kb_chapter_scores +
        the chapter row's judge_scores. Adds ~5–15s per chapter.
        Disable for fast smoke tests.
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

        # Voice tracker: per-character metaphor ring buffer
        voice_tracker = CharacterVoiceTracker.from_entities(
            rollout_sm.list_entities(),
        )

        try:
            prior_choices: list = []
            recent_tail = ""
            prior_proses: list[str] = []
            # Replay already-committed chapters into scroll state (for
            # action selection context only — the rollout DB already has
            # the committed narrative).
            if completed:
                last = completed[-1]
                recent_tail = (last.prose or "")[-500:]
                prior_proses = [c.prose for c in completed if c.prose]
                # Replay completed chapters into the voice tracker
                for c in completed:
                    if c.prose:
                        voice_tracker.record_chapter(
                            c.chapter_index, "char:tristan", c.prose,
                        )

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
                    # Pass the next skeleton chapter so the selector can
                    # favor choices that serve the arc's structural needs.
                    next_skel_ch = None
                    if skeleton:
                        next_skel_ch = next(
                            (c for c in skeleton.chapters if c.chapter_index == ch_idx),
                            None,
                        )
                    chosen_idx, rationale = await select_action(
                        client=client, profile=profile,
                        choices=choices, recent_prose_tail=recent_tail,
                        skeleton_chapter=next_skel_ch,
                    )
                    if choices:
                        c = choices[chosen_idx]
                        player_action = (
                            c if isinstance(c, str)
                            else c.get("title", "") or c.get("description", "")
                        )
                    else:
                        player_action = "continue"

                # Determine POV for voice tracking
                pov_id = "char:tristan"
                if skeleton:
                    skel_ch = next(
                        (c for c in skeleton.chapters if c.chapter_index == ch_idx),
                        None,
                    )
                    if skel_ch and skel_ch.pov_character_id:
                        pov_id = skel_ch.pov_character_id

                # Run the pipeline for this chapter
                out = await pipeline.run(
                    player_action=player_action, update_number=ch_idx,
                )
                prose = out.prose
                recent_tail = (prose or "")[-500:]
                prior_choices = list(out.choices or [])

                # Record this chapter's imagery in the voice tracker
                voice_tracker.record_chapter(ch_idx, pov_id, prose or "")

                # Opening-pattern critic
                opening_issues = check_opening_repetition(
                    prose or "", prior_proses[-5:],
                )
                prior_proses.append(prose or "")

                # Persist chapter to main DB
                chapter = RolloutChapter(
                    rollout_id=rollout_id, chapter_index=ch_idx,
                    player_action=player_action, prose=prose,
                    trace_id=out.trace.trace_id,
                    extract=RolloutExtract(),
                )
                main_sm.save_rollout_chapter(chapter)
                main_sm.update_rollout(
                    rollout_id, chapters_complete=ch_idx,
                )

                # Phase 4: KB extraction (always on, no LLM cost)
                try:
                    trace_dict = json.loads(
                        out.trace.model_dump_json()
                    )
                    persist_chapter_kb(
                        world=main_sm, quest_id=quest_id,
                        rollout_id=rollout_id, chapter_index=ch_idx,
                        prose=prose, trace=trace_dict,
                        all_entities=rollout_sm.list_entities(),
                    )
                except Exception:
                    # KB extraction is best-effort; never fail a rollout
                    # because the parser hit something unexpected.
                    pass

                # Phase 4: scoring (opt-in, ~5-15s per chapter)
                if score:
                    try:
                        await score_and_persist_chapter(
                            client=client, world=main_sm,
                            rollout_id=rollout_id, chapter=chapter,
                        )
                    except Exception:
                        # Same: scoring failures don't break the rollout.
                        pass
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
