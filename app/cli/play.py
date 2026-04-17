from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import typer
from app.engine import (
    ContextBuilder,
    Pipeline,
    PromptRenderer,
    TokenBudget,
    TraceStore,
)
from app.runtime.client import InferenceClient
from app.world import SeedLoader, WorldStateManager
from app.world.db import open_db


app = typer.Typer(help="Quest game CLI.")

PROMPTS = Path(__file__).parent.parent.parent / "prompts"


def _open_world(db_path: Path) -> WorldStateManager:
    conn = open_db(db_path)
    return WorldStateManager(conn)


_DEFAULT_STRUCTURE_ID = "three_act"


@app.command()
def init(
    db: Path = typer.Option(..., help="Path to the quest SQLite DB (will be created)."),
    seed: Path = typer.Option(..., help="Seed JSON file."),
) -> None:
    """Initialize a new quest database from a seed JSON file."""
    import json as _json
    from app.world.schema import QuestArcState

    sm = _open_world(db)
    payload = SeedLoader.load(seed)
    for rule in payload.rules:
        sm.add_rule(rule)
    for hook in payload.foreshadowing:
        sm.add_foreshadowing(hook)
    for pt in payload.plot_threads:
        sm.add_plot_thread(pt)
    sm.apply_delta(payload.delta, update_number=0)

    # Themes + motifs — previously only written to config.json, not the
    # world DB. Story candidate planner reads themes from DB so they must
    # be persisted here too.
    quest_id = db.parent.name
    for th in payload.themes:
        sm.add_theme(quest_id, th)
    for mo in payload.motifs:
        sm.add_motif(quest_id, mo)

    typer.echo(f"Seeded {db} with {len(payload.delta.entity_creates)} entities.")

    # Bootstrap arc state
    try:
        raw_seed = _json.loads(seed.read_text())
    except Exception:
        raw_seed = {}
    structure_id = raw_seed.get("structure_id", _DEFAULT_STRUCTURE_ID)
    quest_id = db.parent.name  # match the server's convention: directory name, not file stem
    arc_state = QuestArcState(
        quest_id=quest_id,
        arc_id="main",
        structure_id=structure_id,
        scale="campaign",
        current_phase_index=0,
        phase_progress=0.0,
        tension_observed=[],
        last_directive=None,
    )
    sm.upsert_arc(arc_state)
    typer.echo(f"Arc state bootstrapped: structure={structure_id}")

    # Write config.json alongside the DB
    quest_config = {
        "genre": raw_seed.get("genre", ""),
        "premise": raw_seed.get("premise", ""),
        "themes": raw_seed.get("themes", []),
        "protagonist": raw_seed.get("protagonist", ""),
        "narrator": raw_seed.get("narrator", {}),
    }
    config_path = db.parent / "config.json"
    config_path.write_text(_json.dumps(quest_config, indent=2))
    typer.echo(f"Quest config written to {config_path}")


@app.command()
def play(
    db: Path = typer.Option(..., help="Path to the quest DB."),
    server: str = typer.Option("http://127.0.0.1:8090", help="llama-server base URL."),
    traces: Path = typer.Option(Path("data/traces"), help="Directory for pipeline trace JSON files."),
    model: str | None = typer.Option(None, help="Model id to request from the server (e.g. a LoRA name)."),
) -> None:
    """Play the quest — reads player actions from stdin, prints prose."""
    import json as _json
    from app.craft.library import CraftLibrary
    from app.planning.arc_planner import ArcPlanner
    from app.planning.craft_planner import CraftPlanner
    from app.planning.dramatic_planner import DramaticPlanner
    from app.planning.emotional_planner import EmotionalPlanner

    sm = _open_world(db)
    client = InferenceClient(base_url=server, retries=1, model=model)
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(sm, renderer, TokenBudget())

    craft_data_dir = Path(__file__).parent.parent / "craft" / "data"
    craft_library = CraftLibrary(craft_data_dir)
    arc_planner = ArcPlanner(client, renderer)
    dramatic_planner = DramaticPlanner(client, renderer, craft_library)
    emotional_planner = EmotionalPlanner(client, renderer)
    craft_planner = CraftPlanner(client, renderer, craft_library)

    quest_id = db.stem
    config_path = db.parent / "config.json"
    quest_config: dict = {}
    if config_path.is_file():
        try:
            quest_config = _json.loads(config_path.read_text())
        except Exception:
            quest_config = {}
    structure = None
    try:
        arc_state = sm.get_arc(quest_id, "main")
        structure = craft_library.structure(arc_state.structure_id)
    except Exception:
        try:
            structure = craft_library.structure(_DEFAULT_STRUCTURE_ID)
        except Exception:
            structure = None

    pipeline = Pipeline(
        sm, cb, client,
        arc_planner=arc_planner,
        dramatic_planner=dramatic_planner,
        emotional_planner=emotional_planner,
        craft_planner=craft_planner,
        craft_library=craft_library,
        structure=structure,
        quest_config=quest_config,
        quest_id=quest_id,
        arc_id="main",
    )
    trace_store = TraceStore(traces)

    typer.echo("Quest started. Type an action and press enter. Ctrl-D to quit.")
    records = sm.list_narrative(limit=10_000)
    update_number = (max((r.update_number for r in records), default=0)) + 1

    for line in sys.stdin:
        action = line.strip()
        if not action:
            continue
        typer.echo("\n--- generating chapter ---\n", err=True)
        try:
            out = asyncio.run(pipeline.run(player_action=action, update_number=update_number))
        except Exception as e:
            typer.echo(f"[error] {e}", err=True)
            continue
        trace_store.save(out.trace)
        typer.echo(out.prose)
        typer.echo(f"\n[trace: {out.trace.trace_id}  outcome: {out.trace.outcome}]", err=True)
        typer.echo("\nChoices:")
        for i, c in enumerate(out.choices, 1):
            typer.echo(f"  {i}. {c}")
        typer.echo()
        update_number += 1


@app.command()
def rollout(
    quests_dir: Path = typer.Option(Path("data/quests"), help="Root directory for quest DBs."),
    quest_id: str = typer.Option(..., "--quest", help="Parent quest id."),
    candidate: str = typer.Option(..., help="Picked candidate id."),
    profile: str = typer.Option("impulsive", help="Virtual-player profile id."),
    chapters: int = typer.Option(5, help="Target chapter count for this rollout."),
    server: str = typer.Option("http://127.0.0.1:8082", help="llama-server base URL."),
    model: str | None = typer.Option(None),
    rollout_id: str | None = typer.Option(None, help="Resume this rollout (creates new if omitted)."),
) -> None:
    """Run a virtual-player rollout against a picked candidate.

    Creates or resumes a RolloutRun, then executes the pipeline
    chapter-by-chapter. Each chapter is saved incrementally; SIGTERM
    or crashes are resumed from the next unfinished chapter on restart.
    """
    from app.rollout.harness import create_rollout_row, run_rollout

    rid = rollout_id or create_rollout_row(
        quests_dir=quests_dir, quest_id=quest_id,
        candidate_id=candidate, profile_id=profile,
        total_chapters_target=chapters,
    )
    typer.echo(f"Rollout: {rid}")
    client = InferenceClient(base_url=server, model=model, timeout=600.0, retries=1)
    try:
        run = asyncio.run(run_rollout(
            quests_dir=quests_dir, quest_id=quest_id,
            rollout_id=rid, client=client,
        ))
        typer.echo(f"Done. Status={run.status.value}  chapters={run.chapters_complete}/{run.total_chapters_target}")
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def refine(
    quests_dir: Path = typer.Option(Path("data/quests"), help="Root directory for quest DBs."),
    quest_id: str = typer.Option(..., "--quest", help="Parent quest id."),
    rollout_id: str | None = typer.Option(None, "--rollout", help="Restrict to this rollout (otherwise all rollouts of the quest)."),
    strategy: str = typer.Option("weak", help="Selector: weak | hooks | sibling | all"),
    max_targets: int = typer.Option(3, help="Cap on the number of refinement targets to attempt."),
    threshold: float = typer.Option(0.55, help="WeakChapterSelector threshold (mean dim score below this triggers refinement)."),
    server: str = typer.Option("http://127.0.0.1:8082"),
    model: str | None = typer.Option(None),
) -> None:
    """Run a refinement pass over a quest's rollouts.

    Picks targets via the chosen selector(s), regenerates each chapter
    with strategy-specific guidance, scores the new prose, and accepts
    only when the result materially beats the baseline.
    """
    from app.refinement.framework import run_refinement_pass
    from app.refinement.selectors import (
        SiblingOutscoredSelector, UnpaidHookSelector, WeakChapterSelector,
    )
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager

    main_conn = open_db(quests_dir / quest_id / "quest.db")
    main_sm = WorldStateManager(main_conn)

    selectors_to_run: list = []
    if strategy in ("weak", "all"):
        selectors_to_run.append(WeakChapterSelector(main_sm, threshold=threshold))
    if strategy in ("hooks", "all"):
        selectors_to_run.append(UnpaidHookSelector(main_sm))
    if strategy in ("sibling", "all"):
        selectors_to_run.append(SiblingOutscoredSelector(main_sm))
    if not selectors_to_run:
        typer.echo(f"unknown strategy: {strategy}", err=True)
        raise typer.Exit(2)

    targets: list = []
    for sel in selectors_to_run:
        sel_targets = sel.select(
            quest_id=quest_id, rollout_id=rollout_id, max_targets=max_targets,
        )
        typer.echo(f"selector {sel.name}: {len(sel_targets)} targets")
        targets.extend(sel_targets)

    if not targets:
        typer.echo("No refinement targets found.")
        return

    client = InferenceClient(base_url=server, model=model, timeout=600.0, retries=1)
    results = asyncio.run(run_refinement_pass(
        targets=targets, quests_dir=quests_dir,
        main_world=main_sm, client=client,
    ))

    n_accepted = sum(1 for r in results if r.accepted)
    typer.echo(f"\n=== Refinement pass complete: {n_accepted}/{len(results)} accepted ===")
    for r in results:
        status = "ACCEPTED" if r.accepted else "REJECTED"
        typer.echo(
            f"  {status:>8}  {r.target.strategy:<18}  "
            f"r={r.target.rollout_id} ch={r.target.chapter_index}  "
            f"Δmean={r.delta_mean:+.3f}  Δmin={r.delta_min:+.3f}"
        )
        if not r.accepted:
            typer.echo(f"           reason: {r.rejection_reason}")


@app.command()
def serve(
    quests_dir: Path = typer.Option(Path("data/quests"), help="Root directory for quest DBs + traces."),
    server: str = typer.Option("http://127.0.0.1:8090", help="llama-server base URL."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the web UI and API."""
    import uvicorn
    from app.server import create_app
    app_obj = create_app(quests_dir=quests_dir, server_url=server)
    uvicorn.run(app_obj, host=host, port=port)
