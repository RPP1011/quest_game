"""Unified quest runner — orchestration above app/engine/pipeline.py.

See docs/superpowers/specs/2026-04-14-quest-runner-resume-design.md.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.runner_config import RunConfig
from app.runner_resume import (
    ResumeMismatchError,
    ConfigDriftError,
    WrongDatabaseError,
    decide_resume,
)


@dataclass(frozen=True)
class RunResult:
    """Summary of a quest run.

    The ``errors`` field is reserved for a future catch-and-continue mode
    and is always ``0`` in the current implementation — pipeline crashes
    propagate to the caller.
    """
    run_name: str
    db_path: Path
    actions_total: int
    skipped_resume: int
    committed: int
    flagged: int
    errors: int
    wall_clock_seconds: float


ProgressCallback = Callable[[int, int, str], None]


class CorruptDatabaseError(Exception):
    """DB file exists but is missing the expected tables."""
    def __init__(self, db_path: Path) -> None:
        super().__init__(
            f"DB at {db_path} is missing the 'narrative' table — "
            f"either corrupt, pre-schema, or not a quest DB. "
            f"Pass --fresh to overwrite, or point --config at a different db_path."
        )
        self.db_path = db_path


def _narrator_for_core(narrator_cfg) -> dict:
    """Strip runner-only fields and None values for app.craft.schemas.Narrator.

    The core Narrator model doesn't define ``pov_character_id`` (runner-only)
    and rejects None for ``editorial_stance``. Apply this anywhere the
    narrator dict crosses into core engine code.
    """
    d = narrator_cfg.model_dump()
    d.pop("pov_character_id", None)
    return {k: v for k, v in d.items() if v is not None}


def _db_state(db_path: Path) -> str:
    """Classify ``db_path`` as one of:
    - ``"absent"``: file does not exist
    - ``"corrupt"``: file exists but ``narrative`` table is missing
    - ``"empty"``: ``narrative`` table exists with zero rows
    - ``"has_rows"``: ``narrative`` has at least one row
    """
    if not db_path.is_file():
        return "absent"
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        try:
            count = conn.execute("SELECT COUNT(*) FROM narrative").fetchone()[0]
        except sqlite3.OperationalError:
            return "corrupt"
        return "has_rows" if count > 0 else "empty"
    finally:
        conn.close()


def _peek_db_quest_id(db_path: Path) -> str | None:
    """Return the arc.quest_id from an existing DB, or None if unknown."""
    if not db_path.is_file():
        return None
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        # arcs is the most stable indicator of a bootstrapped quest
        row = conn.execute("SELECT quest_id FROM arcs LIMIT 1").fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        # Table doesn't exist → not yet bootstrapped or schema mismatch
        return None
    finally:
        conn.close()


def _load_existing_rows(db_path: Path) -> list[dict]:
    if not db_path.is_file():
        return []
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT update_number, player_action FROM narrative "
            "ORDER BY update_number"
        )
        return [{"update_number": r[0], "player_action": r[1]} for r in cur]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _bootstrap_world(config: RunConfig):
    """Create a WorldStateManager + apply seed. Returns (sm, conn)."""
    from app.world import SeedLoader
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    from app.world.schema import QuestArcState, ReaderState

    db_path = config.options.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Materialize seed dict to a temp JSON for SeedLoader
    seed_path = db_path.with_suffix(".seed.json")
    seed_path.write_text(json.dumps(_seed_to_dict(config)))

    conn = open_db(db_path)
    sm = WorldStateManager(conn)
    payload = SeedLoader.load(seed_path)
    for rule in payload.rules:
        sm.add_rule(rule)
    for hook in payload.foreshadowing:
        sm.add_foreshadowing(hook)
    for pt in payload.plot_threads:
        sm.add_plot_thread(pt)
    for th in payload.themes:
        sm.add_theme(config.seed.quest_id, th)
    sm.apply_delta(payload.delta, update_number=0)

    sm.upsert_arc(QuestArcState(
        quest_id=config.seed.quest_id, arc_id="main",
        structure_id="three_act", scale="campaign",
        current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=config.seed.quest_id))
    return sm, conn


def _reopen_world(config: RunConfig):
    """Open an existing DB without re-applying seed."""
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    conn = open_db(config.options.db_path)
    return WorldStateManager(conn), conn


def _seed_to_dict(config: RunConfig) -> dict:
    """Materialize the seed Pydantic model back to the dict shape
    SeedLoader expects (it reads JSON from disk).

    The NarratorConfig carries runner-only fields (``pov_character_id``)
    and permits ``None`` for fields (like ``editorial_stance``) that the
    core ``Narrator`` model requires as strings, so we filter out ``None``
    values and drop runner-only fields before handing to ``SeedLoader``.
    """
    seed = config.seed
    return {
        "entities": list(seed.entities),
        "plot_threads": list(seed.plot_threads),
        "themes": list(seed.themes),
        "foreshadowing": list(seed.foreshadowing),
        "narrator": _narrator_for_core(seed.narrator),
        "rules": [],
    }


def _build_real_pipeline(world, config: RunConfig):
    """Wire the actual pipeline. Mirrors the planner + retriever wiring
    the deleted scripts had. Honors config.options for retrieval/scoring."""
    from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget
    from app.runtime.client import InferenceClient
    from app.craft.library import CraftLibrary
    from app.planning import DramaticPlanner, EmotionalPlanner, CraftPlanner
    from app.planning.arc_planner import ArcPlanner
    from app.retrieval import (
        Embedder, PassageRetriever, QuestRetriever, SceneShapeRetriever,
        MotifRetriever, ForeshadowingRetriever, VoiceRetriever,
    )

    REPO = Path(__file__).resolve().parent.parent
    PROMPTS = REPO / "prompts"

    client = InferenceClient(
        base_url=config.options.llm_url,
        retries=1,
        model=config.options.llm_model,
    )
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(world, renderer, TokenBudget())
    craft_library = CraftLibrary(REPO / "app" / "craft" / "data")
    structure = craft_library.structure("three_act")

    planners = {
        "arc_planner": ArcPlanner(client, renderer),
        "dramatic_planner": DramaticPlanner(client, renderer, craft_library),
        "emotional_planner": EmotionalPlanner(client, renderer),
        "craft_planner": CraftPlanner(client, renderer, craft_library),
    }

    scorer = None
    if config.options.scoring or config.options.n_candidates > 1:
        from app.scoring.scorer import Scorer
        scorer = Scorer()

    CALIB = REPO / "data" / "calibration"
    manifest = CALIB / "manifest.yaml"
    passages_dir = CALIB / "passages"
    scenes_manifest = CALIB / "scenes_manifest.yaml"
    scenes_dir = CALIB / "scenes"

    passage_retriever = None
    if manifest.is_file():
        passage_retriever = PassageRetriever(
            manifest, Path("/tmp"), passages_dir, enable_semantic=False,
        )
    scene_retriever = None
    if scenes_manifest.is_file():
        try:
            scene_retriever = SceneShapeRetriever(
                scenes_manifest, "/tmp/labels_claude_arc_*.json", scenes_dir,
            )
        except Exception:
            scene_retriever = None

    embedder = Embedder()
    quest_retriever = QuestRetriever(world, config.seed.quest_id, embedder=embedder)
    motif_retriever = MotifRetriever(world, config.seed.quest_id)
    foreshadow_retriever = ForeshadowingRetriever(world, config.seed.quest_id)
    voice_retriever = VoiceRetriever(world, config.seed.quest_id)

    quest_config = {
        "narrator": _narrator_for_core(config.seed.narrator),
        "genre": config.seed.genre,
        "n_candidates": config.options.n_candidates,
        "retrieval": {"enabled": True},
    }
    if config.options.sft_collection.enabled:
        quest_config["sft_collection"] = {
            "enabled": True,
            "dir": config.options.sft_collection.dir,
        }

    return Pipeline(
        world, cb, client,
        **planners,
        craft_library=craft_library,
        structure=structure,
        scorer=scorer,
        rerank_weights=config.options.rerank_weights,
        quest_config=quest_config,
        quest_id=config.seed.quest_id,
        arc_id="main",
        passage_retriever=passage_retriever,
        scene_retriever=scene_retriever,
        quest_retriever=quest_retriever,
        motif_retriever=motif_retriever,
        foreshadowing_retriever=foreshadow_retriever,
        voice_retriever=voice_retriever,
        retrieval_embedder=embedder,
    )


async def run_quest(
    config: RunConfig,
    *,
    fresh: bool = False,
    progress_callback: ProgressCallback | None = None,
    _pipeline_factory: Callable[[Any], Any] | None = None,
) -> RunResult:
    """Bootstrap or resume a quest run.

    On entry: if config.options.db_path exists and not fresh, open it,
    validate action-list match, skip already-done actions. Otherwise
    unlink + bootstrap from seed.

    ``_pipeline_factory`` is for tests — supply a callable that takes
    a WorldStateManager and returns a Pipeline-like object.
    """
    db_path = config.options.db_path
    t0 = time.perf_counter()

    if fresh and db_path.is_file():
        db_path.unlink()

    state = _db_state(db_path)
    if state == "corrupt":
        raise CorruptDatabaseError(db_path)
    if state == "absent" or state == "empty":
        if state == "empty":
            db_path.unlink()
        rows = []
        db_quest_id = None
    else:  # has_rows
        rows = _load_existing_rows(db_path)
        db_quest_id = _peek_db_quest_id(db_path)

    decision = decide_resume(
        rows=rows,
        actions=config.actions,
        db_quest_id=db_quest_id,
        config_quest_id=config.seed.quest_id,
    )

    if state in ("absent", "empty"):
        world, _conn = _bootstrap_world(config)
    else:
        world, _conn = _reopen_world(config)

    if _pipeline_factory is not None:
        pipeline = _pipeline_factory(world)
    else:
        pipeline = _build_real_pipeline(world, config)

    committed = 0
    flagged = 0
    total = len(config.actions)

    for i, action in enumerate(config.actions[decision.skipped:],
                                start=decision.start_from):
        if progress_callback is not None:
            progress_callback(committed + decision.skipped, total, action)
        out = await pipeline.run(player_action=action, update_number=i)
        outcome = getattr(out.trace, "outcome", "committed")
        if outcome == "committed":
            committed += 1
        elif outcome == "flagged_qm":
            flagged += 1

    return RunResult(
        run_name=config.run_name,
        db_path=db_path,
        actions_total=total,
        skipped_resume=decision.skipped,
        committed=committed,
        flagged=flagged,
        errors=0,
        wall_clock_seconds=time.perf_counter() - t0,
    )
