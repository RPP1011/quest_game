from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.quest import (
    AdvanceRequest,
    AdvanceResponse,
    ChapterSummary,
    Choice,
    QuestSummary,
    SceneContext,
    TraceSummary,
)
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
from app.world.schema import QuestArcState


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
CRAFT_DATA_DIR = Path(__file__).parent / "craft" / "data"
WEB_DIR = Path(__file__).parent.parent / "web"

_DEFAULT_STRUCTURE_ID = "three_act"


class CreateQuestRequest(BaseModel):
    id: str
    seed: dict[str, Any]


def _quest_paths(quests_dir: Path, qid: str) -> dict[str, Path]:
    root = quests_dir / qid
    return {"root": root, "db": root / "quest.db", "traces": root / "traces"}


def _make_client(server_url: str) -> InferenceClient:
    return InferenceClient(base_url=server_url, retries=1)


def _load_craft_library():
    """Load the CraftLibrary from the bundled data directory."""
    from app.craft.library import CraftLibrary
    return CraftLibrary(CRAFT_DATA_DIR)


def _config_from_seed(seed: dict) -> dict:
    """Derive quest_config dict from a seed dict."""
    cfg = {
        "genre": seed.get("genre", ""),
        "premise": seed.get("premise", ""),
        "themes": seed.get("themes", []),
        "protagonist": seed.get("protagonist", ""),
    }
    if "narrator" in seed and seed["narrator"]:
        cfg["narrator"] = seed["narrator"]
    return cfg


def _build_planners(client, renderer, craft_library):
    """Instantiate all four hierarchical planners."""
    from app.planning.arc_planner import ArcPlanner
    from app.planning.dramatic_planner import DramaticPlanner
    from app.planning.emotional_planner import EmotionalPlanner
    from app.planning.craft_planner import CraftPlanner
    return (
        ArcPlanner(client, renderer),
        DramaticPlanner(client, renderer, craft_library),
        EmotionalPlanner(client, renderer),
        CraftPlanner(client, renderer, craft_library),
    )


def create_app(*, quests_dir: Path, server_url: str) -> FastAPI:
    quests_dir = Path(quests_dir)
    quests_dir.mkdir(parents=True, exist_ok=True)
    renderer = PromptRenderer(PROMPTS_DIR)
    client = _make_client(server_url)

    # Load craft library and build planners at startup
    try:
        craft_library = _load_craft_library()
        arc_planner, dramatic_planner, emotional_planner, craft_planner = _build_planners(
            client, renderer, craft_library
        )
    except Exception:
        craft_library = None
        arc_planner = dramatic_planner = emotional_planner = craft_planner = None

    app = FastAPI(title="Quest Game")

    def _open(qid: str) -> tuple[WorldStateManager, TraceStore]:
        paths = _quest_paths(quests_dir, qid)
        if not paths["db"].is_file():
            raise HTTPException(404, f"unknown quest: {qid}")
        sm = WorldStateManager(open_db(paths["db"]))
        store = TraceStore(paths["traces"])
        return sm, store

    @app.get("/api/quests")
    def list_quests() -> list[QuestSummary]:
        out: list[QuestSummary] = []
        for sub in sorted(quests_dir.iterdir()) if quests_dir.is_dir() else []:
            db = sub / "quest.db"
            if not db.is_file():
                continue
            sm = WorldStateManager(open_db(db))
            records = sm.list_narrative(limit=10_000)
            last = records[-1].player_action if records else None
            out.append(QuestSummary(
                id=sub.name, path=str(db.resolve()),
                chapter_count=len(records), last_action=last,
            ))
        return out

    @app.post("/api/quests", status_code=201)
    def create_quest(req: CreateQuestRequest) -> QuestSummary:
        paths = _quest_paths(quests_dir, req.id)
        if paths["db"].exists():
            raise HTTPException(409, f"quest already exists: {req.id}")
        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["traces"].mkdir(parents=True, exist_ok=True)
        # Write seed to tmp file so SeedLoader can read it (simplest path)
        seed_file = paths["root"] / "seed.json"
        seed_file.write_text(json.dumps(req.seed))
        sm = WorldStateManager(open_db(paths["db"]))
        payload = SeedLoader.load(seed_file)
        for rule in payload.rules:
            sm.add_rule(rule)
        for hook in payload.foreshadowing:
            sm.add_foreshadowing(hook)
        for pt in payload.plot_threads:
            sm.add_plot_thread(pt)
        sm.apply_delta(payload.delta, update_number=0)

        # --- Task 11: arc bootstrap ---
        structure_id = req.seed.get("structure_id", _DEFAULT_STRUCTURE_ID)
        arc_state = QuestArcState(
            quest_id=req.id,
            arc_id="main",
            structure_id=structure_id,
            scale="campaign",
            current_phase_index=0,
            phase_progress=0.0,
            tension_observed=[],
            last_directive=None,
        )
        sm.upsert_arc(arc_state)

        # Write config.json with quest metadata derived from seed
        quest_config = _config_from_seed(req.seed)
        config_path = paths["root"] / "config.json"
        config_path.write_text(json.dumps(quest_config, indent=2))

        return QuestSummary(
            id=req.id, path=str(paths["db"].resolve()),
            chapter_count=0, last_action=None,
        )

    def _raw_choices_to_models(raw: list) -> list[Choice]:
        out: list[Choice] = []
        for item in raw:
            if isinstance(item, str):
                out.append(Choice(title=item))
            elif isinstance(item, dict):
                out.append(Choice(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    tags=item.get("tags", []) if isinstance(item.get("tags"), list) else [],
                ))
        return out

    def _extract_choices_from_trace(trace) -> list[Choice]:
        """Pull choices from the dramatic stage; fall back to the old plan stage."""
        # Try dramatic stage first (new hierarchical pipeline)
        for stage in trace.stages:
            if stage.stage_name == "dramatic":
                po = stage.parsed_output
                if isinstance(po, dict):
                    raw = po.get("suggested_choices", []) or []
                    return _raw_choices_to_models(raw)
                break
        # Fall back to old plan stage (legacy traces)
        for stage in trace.stages:
            if stage.stage_name == "plan":
                po = stage.parsed_output
                if isinstance(po, dict):
                    raw = po.get("suggested_choices", []) or []
                    return _raw_choices_to_models(raw)
                break
        return []

    @app.get("/api/quests/{qid}/chapters")
    def list_chapters(qid: str) -> list[ChapterSummary]:
        sm, store = _open(qid)
        results: list[ChapterSummary] = []
        for n in sm.list_narrative(limit=10_000):
            choices: list[Choice] = []
            if n.pipeline_trace_id:
                try:
                    trace = store.load(n.pipeline_trace_id)
                    choices = _extract_choices_from_trace(trace)
                except (FileNotFoundError, Exception):
                    choices = []
            results.append(ChapterSummary(
                update_number=n.update_number, player_action=n.player_action,
                prose=n.raw_text, trace_id=n.pipeline_trace_id,
                choices=choices,
            ))
        return results

    @app.get("/api/quests/{qid}/scene")
    def get_scene(qid: str) -> SceneContext:
        from app.world.schema import EntityStatus, EntityType, ThreadStatus
        sm, _ = _open(qid)
        entities = sm.list_entities()
        active_entities = [e for e in entities if e.status == EntityStatus.ACTIVE]
        # Most recent active location entity
        locations = [e for e in active_entities if e.entity_type == EntityType.LOCATION]
        location_name: str | None = locations[-1].name if locations else None
        # All active characters
        characters = [
            e.name for e in active_entities if e.entity_type == EntityType.CHARACTER
        ]
        # Top-3 active plot threads by priority
        threads = sm.list_plot_threads()
        active_threads = [t for t in threads if t.status == ThreadStatus.ACTIVE]
        top_threads = [t.name for t in active_threads[:3]]
        # Last ~400 chars of most recent narrative
        records = sm.list_narrative(limit=10_000)
        recent_tail = ""
        if records:
            last_prose = records[-1].raw_text or ""
            recent_tail = last_prose[-400:]
        return SceneContext(
            location=location_name,
            present_characters=characters,
            plot_threads=top_threads,
            recent_prose_tail=recent_tail,
        )

    @app.post("/api/quests/{qid}/advance")
    async def advance(qid: str, req: AdvanceRequest) -> AdvanceResponse:
        sm, store = _open(qid)
        cb = ContextBuilder(sm, renderer, TokenBudget())

        # Load quest_config from config.json (written at bootstrap)
        paths = _quest_paths(quests_dir, qid)
        config_path = paths["root"] / "config.json"
        quest_config: dict = {}
        if config_path.is_file():
            try:
                quest_config = json.loads(config_path.read_text())
            except Exception:
                quest_config = {}

        # Resolve structure from arc state
        structure = None
        if craft_library is not None:
            try:
                arc_state = sm.get_arc(qid, "main")
                structure = craft_library.structure(arc_state.structure_id)
            except Exception:
                structure = None
                # Try default structure
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
            quest_id=qid,
            arc_id="main",
        )
        records = sm.list_narrative(limit=10_000)
        update_number = (max((r.update_number for r in records), default=0)) + 1
        try:
            out = await pipeline.run(player_action=req.action, update_number=update_number)
        except Exception as e:
            raise HTTPException(500, f"pipeline failed: {e}")
        store.save(out.trace)
        return AdvanceResponse(
            update_number=update_number,
            prose=out.prose,
            choices=_raw_choices_to_models(out.choices),
            trace_id=out.trace.trace_id,
            outcome=out.trace.outcome,
        )

    @app.get("/api/quests/{qid}/traces")
    def list_traces(qid: str) -> list[TraceSummary]:
        _, store = _open(qid)
        results = []
        for tid in store.list_ids():
            t = store.load(tid)
            results.append(TraceSummary(
                trace_id=t.trace_id, trigger=t.trigger, outcome=t.outcome,
                stages=[s.stage_name for s in t.stages],
                total_latency_ms=t.total_latency_ms,
            ))
        return results

    @app.get("/api/quests/{qid}/traces/{tid}")
    def get_trace(qid: str, tid: str) -> dict:
        _, store = _open(qid)
        try:
            return store.load(tid).model_dump()
        except FileNotFoundError:
            raise HTTPException(404, f"unknown trace: {tid}")

    # Static UI
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app
