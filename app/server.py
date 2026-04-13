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
    QuestSummary,
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


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
WEB_DIR = Path(__file__).parent.parent / "web"


class CreateQuestRequest(BaseModel):
    id: str
    seed: dict[str, Any]


def _quest_paths(quests_dir: Path, qid: str) -> dict[str, Path]:
    root = quests_dir / qid
    return {"root": root, "db": root / "quest.db", "traces": root / "traces"}


def _make_client(server_url: str) -> InferenceClient:
    return InferenceClient(base_url=server_url, retries=1)


def create_app(*, quests_dir: Path, server_url: str) -> FastAPI:
    quests_dir = Path(quests_dir)
    quests_dir.mkdir(parents=True, exist_ok=True)
    renderer = PromptRenderer(PROMPTS_DIR)
    client = _make_client(server_url)

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
        return QuestSummary(
            id=req.id, path=str(paths["db"].resolve()),
            chapter_count=0, last_action=None,
        )

    @app.get("/api/quests/{qid}/chapters")
    def list_chapters(qid: str) -> list[ChapterSummary]:
        sm, store = _open(qid)
        results: list[ChapterSummary] = []
        for n in sm.list_narrative(limit=10_000):
            choices: list[str] = []
            if n.pipeline_trace_id:
                try:
                    trace = store.load(n.pipeline_trace_id)
                    for stage in trace.stages:
                        if stage.stage_name == "plan":
                            po = stage.parsed_output
                            if isinstance(po, dict):
                                choices = po.get("suggested_choices", []) or []
                            break
                except (FileNotFoundError, Exception):
                    choices = []
            results.append(ChapterSummary(
                update_number=n.update_number, player_action=n.player_action,
                prose=n.raw_text, trace_id=n.pipeline_trace_id,
                choices=choices,
            ))
        return results

    @app.post("/api/quests/{qid}/advance")
    async def advance(qid: str, req: AdvanceRequest) -> AdvanceResponse:
        sm, store = _open(qid)
        cb = ContextBuilder(sm, renderer, TokenBudget())
        pipeline = Pipeline(sm, cb, client)
        records = sm.list_narrative(limit=10_000)
        update_number = (max((r.update_number for r in records), default=0)) + 1
        try:
            out = await pipeline.run(player_action=req.action, update_number=update_number)
        except Exception as e:
            raise HTTPException(500, f"pipeline failed: {e}")
        store.save(out.trace)
        return AdvanceResponse(
            update_number=update_number, prose=out.prose, choices=out.choices,
            trace_id=out.trace.trace_id, outcome=out.trace.outcome,
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
