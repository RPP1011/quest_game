from __future__ import annotations
import asyncio
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
        for th in payload.themes:
            sm.add_theme(req.id, th)
        for mo in payload.motifs:
            sm.add_motif(req.id, mo)
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

        # Gap G6: bootstrap an empty reader state for this quest.
        from app.world.schema import ReaderState
        sm.upsert_reader_state(ReaderState(quest_id=req.id))

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

    @app.get("/api/quests/{qid}/candidates")
    def list_candidates(qid: str) -> list[dict]:
        """List story candidates for this quest."""
        sm, _ = _open(qid)
        return [c.model_dump(mode="json") for c in sm.list_story_candidates(qid)]

    @app.post("/api/quests/{qid}/candidates/generate")
    async def generate_candidates(qid: str, n: int = 3) -> list[dict]:
        """Generate N story candidates for this quest.

        Uses the seed's world state as grounding. Persists candidates;
        returns them. If candidates already exist, this appends — the
        caller should check ``list_candidates`` first if they want to
        avoid regeneration.
        """
        from app.planning.story_candidate_planner import StoryCandidatePlanner
        sm, _ = _open(qid)
        paths = _quest_paths(quests_dir, qid)
        config_path = paths["root"] / "config.json"
        quest_config: dict = {}
        if config_path.is_file():
            try:
                quest_config = json.loads(config_path.read_text())
            except Exception:
                quest_config = {}
        planner = StoryCandidatePlanner(client, renderer)
        try:
            cands = await planner.generate(
                world=sm, quest_id=qid, quest_config=quest_config, n=n,
            )
        except Exception as e:
            raise HTTPException(500, f"candidate generation failed: {e}")
        return [c.model_dump(mode="json") for c in cands]

    @app.get("/api/quests/{qid}/kb")
    def get_kb(qid: str) -> dict:
        """Aggregated KB views across all rollouts for this quest.

        Returns:
        - hook_payoffs: per-hook list of {planted_at_chapter, paid_off_at_chapter}
          across rollouts, plus a payoff_rate (paid_off_count / total_rollouts).
        - entity_usage: per-entity {introduced_at_chapter, mention_chapters}
          rows, plus a screen_time count.
        - dim_means_by_chapter: per-chapter-index, per-dim mean scores
          across all rollouts.
        """
        sm, _ = _open(qid)
        rollouts = sm.list_rollouts(quest_id=qid)
        n_rollouts = len(rollouts)

        hooks_raw = sm.list_hook_payoffs(qid)
        # Group by hook_id
        from collections import defaultdict
        hooks_by_id: dict[str, list] = defaultdict(list)
        for r in hooks_raw:
            hooks_by_id[r["hook_id"]].append(r)
        hook_payoffs = []
        for hid, rows in sorted(hooks_by_id.items()):
            paid = sum(1 for r in rows if r["paid_off_at_chapter"] is not None)
            hook_payoffs.append({
                "hook_id": hid,
                "planted_count": sum(1 for r in rows if r["planted_at_chapter"] is not None),
                "paid_off_count": paid,
                "total_rollouts": n_rollouts,
                "payoff_rate": paid / n_rollouts if n_rollouts > 0 else 0.0,
                "rows": rows,
            })

        eu_raw = sm.list_entity_usage(qid)
        eu_by_id: dict[str, list] = defaultdict(list)
        for r in eu_raw:
            eu_by_id[r["entity_id"]].append(r)
        entity_usage = []
        for eid, rows in sorted(eu_by_id.items()):
            total_mentions = sum(len(r["mention_chapters"]) for r in rows)
            entity_usage.append({
                "entity_id": eid,
                "introduced_count": sum(1 for r in rows if r["introduced_at_chapter"] is not None),
                "total_rollouts": n_rollouts,
                "screen_time": total_mentions,
                "rows": rows,
            })

        # Per-(chapter_index, dim) mean across all rollouts for this quest
        dim_means: dict[tuple[int, str], list[float]] = defaultdict(list)
        for r in rollouts:
            for s in sm.list_chapter_scores(r.id):
                dim_means[(s["chapter_index"], s["dim"])].append(s["score"])
        dim_means_by_chapter: list[dict] = []
        for (ch_idx, dim), scores in sorted(dim_means.items()):
            dim_means_by_chapter.append({
                "chapter_index": ch_idx, "dim": dim,
                "mean": sum(scores) / len(scores),
                "n_rollouts_scored": len(scores),
            })

        return {
            "n_rollouts": n_rollouts,
            "hook_payoffs": hook_payoffs,
            "entity_usage": entity_usage,
            "dim_means_by_chapter": dim_means_by_chapter,
        }

    @app.get("/api/quests/{qid}/rollouts/{rid}/scores")
    def get_rollout_scores(qid: str, rid: str) -> dict:
        """Per-chapter dim breakdown for one rollout."""
        sm, _ = _open(qid)
        try:
            sm.get_rollout(rid)
        except Exception:
            raise HTTPException(404, f"unknown rollout: {rid}")
        rows = sm.list_chapter_scores(rid)
        # Group by chapter
        from collections import defaultdict
        by_chapter: dict[int, dict] = defaultdict(dict)
        for r in rows:
            by_chapter[r["chapter_index"]][r["dim"]] = {
                "score": r["score"], "rationale": r["rationale"],
            }
        return {
            "rollout_id": rid,
            "chapters": [
                {"chapter_index": idx, "dims": dims}
                for idx, dims in sorted(by_chapter.items())
            ],
        }

    @app.get("/api/quests/{qid}/rollouts")
    def list_rollouts(qid: str) -> list[dict]:
        sm, _ = _open(qid)
        runs = sm.list_rollouts(quest_id=qid)
        return [r.model_dump(mode="json") for r in runs]

    @app.get("/api/quests/{qid}/rollouts/{rid}")
    def get_rollout(qid: str, rid: str) -> dict:
        sm, _ = _open(qid)
        try:
            run = sm.get_rollout(rid)
        except Exception:
            raise HTTPException(404, f"unknown rollout: {rid}")
        chapters = [
            c.model_dump(mode="json") for c in sm.list_rollout_chapters(rid)
        ]
        return {**run.model_dump(mode="json"), "chapters": chapters}

    @app.post("/api/quests/{qid}/candidates/{cid}/rollouts/start",
              status_code=202)
    async def start_rollout(
        qid: str, cid: str, profile: str = "impulsive",
        chapters: int = 5,
    ) -> dict:
        """Launch a rollout. Returns immediately with the rollout id;
        execution runs in the background.

        Progress is polled via ``GET /rollouts/{rid}``.
        """
        from app.rollout.harness import create_rollout_row, run_rollout
        try:
            rid = create_rollout_row(
                quests_dir=quests_dir, quest_id=qid,
                candidate_id=cid, profile_id=profile,
                total_chapters_target=chapters,
            )
        except Exception as e:
            raise HTTPException(400, f"failed to create rollout: {e}")

        async def _run():
            try:
                await run_rollout(
                    quests_dir=quests_dir, quest_id=qid,
                    rollout_id=rid, client=client,
                )
            except Exception:
                # harness already records FAILED status; just swallow
                pass

        asyncio.create_task(_run())
        return {"rollout_id": rid, "status": "pending"}

    @app.get("/api/rollout-profiles")
    def list_rollout_profiles() -> list[dict]:
        """Return the available virtual-player profiles."""
        from app.rollout.profiles import list_profiles
        return [p.model_dump() for p in list_profiles()]

    @app.get("/api/quests/{qid}/candidates/{cid}/skeleton")
    def get_skeleton(qid: str, cid: str) -> dict:
        """Return the latest arc skeleton for a candidate, or 404 if none."""
        sm, _ = _open(qid)
        # Validate candidate exists
        try:
            sm.get_story_candidate(cid)
        except Exception:
            raise HTTPException(404, f"unknown candidate: {cid}")
        skel = sm.get_skeleton_for_candidate(cid)
        if skel is None:
            raise HTTPException(404, f"no skeleton for candidate: {cid}")
        return skel.model_dump(mode="json")

    @app.post("/api/quests/{qid}/candidates/{cid}/skeleton/generate")
    async def generate_skeleton(qid: str, cid: str) -> dict:
        """Generate an arc skeleton for a picked candidate.

        Uses the candidate's expected_chapter_count as target length.
        Overwrites any prior skeleton for this candidate. Blocks for the
        full LLM call (~1–2 minutes on the current writer). Callers
        should expect a long request.
        """
        from app.planning.arc_skeleton_planner import ArcSkeletonPlanner
        sm, _ = _open(qid)
        try:
            cand = sm.get_story_candidate(cid)
        except Exception:
            raise HTTPException(404, f"unknown candidate: {cid}")
        planner = ArcSkeletonPlanner(client, renderer)
        try:
            skel = await planner.generate(world=sm, candidate=cand)
        except Exception as e:
            raise HTTPException(500, f"skeleton generation failed: {e}")
        return skel.model_dump(mode="json")

    @app.post("/api/quests/{qid}/candidates/{cid}/pick")
    def pick_candidate(qid: str, cid: str) -> dict:
        """Mark a candidate as picked; persist into config.json so the
        pipeline's arc planner can read it as directive input."""
        sm, _ = _open(qid)
        try:
            cand = sm.pick_story_candidate(qid, cid)
        except Exception as e:
            raise HTTPException(404, str(e))
        # Persist pick into config.json for pipeline reads
        paths = _quest_paths(quests_dir, qid)
        config_path = paths["root"] / "config.json"
        cfg: dict = {}
        if config_path.is_file():
            try:
                cfg = json.loads(config_path.read_text())
            except Exception:
                cfg = {}
        cfg["picked_candidate"] = cand.model_dump(mode="json")
        config_path.write_text(json.dumps(cfg, indent=2))
        return cand.model_dump(mode="json")

    @app.get("/api/quests/{qid}/config")
    def get_config(qid: str) -> dict:
        """Quest metadata derived from the seed: genre, premise, themes,
        protagonist, narrator. Used by the UI to render a hero panel for
        empty quests. Returns ``{}`` if no config.json was written."""
        paths = _quest_paths(quests_dir, qid)
        if not paths["db"].is_file():
            raise HTTPException(404, f"unknown quest: {qid}")
        config_path = paths["root"] / "config.json"
        if not config_path.is_file():
            return {}
        try:
            return json.loads(config_path.read_text())
        except Exception:
            return {}

    @app.get("/api/quests/{qid}/world")
    def get_world(qid: str) -> dict:
        """Browseable snapshot of the seeded world.

        Groups entities by type (character, location, faction, item,
        concept), and returns plot threads, foreshadowing hooks, world
        rules, and motifs. Everything the seed shipped, made discoverable
        in the UI without the player having to read a JSON file.
        """
        from app.world.schema import EntityStatus, EntityType
        sm, _ = _open(qid)
        entities = sm.list_entities()
        non_destroyed = [e for e in entities if e.status != EntityStatus.DESTROYED]

        by_type: dict[str, list] = {
            t.value: [] for t in EntityType
        }
        for e in non_destroyed:
            by_type[e.entity_type.value].append({
                "id": e.id, "name": e.name,
                "status": e.status.value,
                "description": e.data.get("description", ""),
                "role": e.data.get("role", ""),
                "data": e.data,
            })

        plot_threads = [
            {
                "id": t.id, "name": t.name, "description": t.description,
                "status": t.status.value, "priority": t.priority,
                "arc_position": t.arc_position.value,
                "involved_entities": t.involved_entities,
            }
            for t in sm.list_plot_threads()
        ]

        try:
            hook_rows = sm._conn.execute(
                "SELECT id, description, status, planted_at_update, payoff_target FROM foreshadowing ORDER BY id"
            ).fetchall()
            hooks = [
                {
                    "id": r[0], "description": r[1], "status": r[2],
                    "planted_at_update": r[3], "payoff_target": r[4],
                }
                for r in hook_rows
            ]
        except Exception:
            hooks = []

        rules = [
            {"id": r.id, "category": r.category, "description": r.description}
            for r in sm.list_rules()
        ]

        try:
            motifs = [
                {
                    "id": m.id, "name": m.name, "description": m.description,
                    "semantic_range": m.semantic_range,
                }
                for m in sm.list_motifs(qid)
            ]
        except Exception:
            motifs = []

        return {
            "entities_by_type": by_type,
            "plot_threads": plot_threads,
            "foreshadowing": hooks,
            "rules": rules,
            "motifs": motifs,
        }

    @app.get("/api/quests/{qid}/starting-actions")
    def get_starting_actions(qid: str) -> list[dict]:
        """Suggested opening actions for chapter 1.

        Derived from the seed's top-priority active plot threads, so the
        player has concrete invitations into the world without having to
        invent an action from the premise alone.
        """
        from app.world.schema import EntityStatus, ThreadStatus
        sm, _ = _open(qid)
        # Only offer suggestions when no chapters have been committed yet
        records = sm.list_narrative(limit=1)
        if records:
            return []
        threads = sm.list_plot_threads()
        active = [t for t in threads if t.status == ThreadStatus.ACTIVE]
        active.sort(key=lambda t: -t.priority)
        out: list[dict] = []
        # Build suggestions from top-3 active threads. We use their
        # description verbatim as the "hook" — the LLM will turn the
        # suggestion into a first action when submitted.
        for t in active[:3]:
            out.append({
                "title": t.name,
                "description": t.description,
                "thread_id": t.id,
            })
        return out

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
            live_trace_save=store.save,
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
