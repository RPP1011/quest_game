"""Generate a chapter with multiple scene-generation strategies against the
same seed + action, then score each with the 12 heuristic dims + 8 chapter
judge dims. Writes a per-strategy prose file and a comparison JSON.

Usage::

    uv run python tools/strategy_sweep.py \\
        --seed seeds/pale_lights.json \\
        --server http://127.0.0.1:8082 \\
        --out data/strategy_sweep/pale_lights \\
        --action "Tristan takes the cabinet job Abuela set him..."

Strategies:
    per_beat      — current production: dramatic -> craft -> beat loop
    scene         — one writer call per scene (all beats concatenated)
    one_shot      — single writer call given the full plan
    refine        — one_shot + critic + targeted revision on weakest scenes
    expand        — two-pass: rough sketch then per-scene expansion

All strategies share the same (arc, dramatic, emotional, craft) plan to
isolate the write-stage differences. Planners are run ONCE; each write
strategy reuses the cached plan.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import ContextBuilder, PromptRenderer, TokenBudget, TraceStore  # noqa: E402
from app.engine.pipeline import Pipeline, PipelineTrace  # noqa: E402
from app.runtime.client import ChatMessage, InferenceClient  # noqa: E402
from app.world import WorldStateManager  # noqa: E402
from app.world.db import open_db  # noqa: E402
from app.craft.library import CraftLibrary  # noqa: E402
from app.planning.arc_planner import ArcPlanner  # noqa: E402
from app.planning.craft_planner import CraftPlanner  # noqa: E402
from app.planning.dramatic_planner import DramaticPlanner  # noqa: E402
from app.planning.emotional_planner import EmotionalPlanner  # noqa: E402
from app.calibration.heuristics import run_heuristics  # noqa: E402

RUBRICS_DIR = ROOT / "prompts" / "scoring" / "chapter_dims"
DEFAULT_JUDGE_DIMS = [
    "tension_execution", "emotional_trajectory", "choice_hook_quality",
    "update_self_containment", "voice_distinctiveness", "thematic_presence",
    "subtext_presence", "interiority_depth",
]


def _load_rubric(dim: str) -> str:
    return (RUBRICS_DIR / f"{dim}.j2").read_text()


def _build_judge_schema(dims: list[str]) -> dict:
    return {
        "type": "object", "required": dims,
        "properties": {
            d: {
                "type": "object", "required": ["score", "rationale"],
                "properties": {
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
            } for d in dims
        },
    }


JUDGE_SYSTEM = """You are a strict literary-quality judge. You score a FULL CHAPTER of a novel or web serial on several dimensions simultaneously. Use the full 0.0-1.0 range and avoid clustering near 0.5. Each rubric below is self-contained and anchored — follow its anchors rather than importing prior priors. Score the chapter on the page, not the situation it describes. For each dimension, emit a score in [0.0, 1.0] and a one-sentence rationale as defined by the rubric. Return ONLY the JSON object matching the response schema — no preamble."""


def _build_judge_user(text: str, dims: list[str]) -> str:
    parts = ["Score the following chapter on each of these dimensions.", "",
             "DIMENSIONS (self-contained rubrics):", ""]
    for d in dims:
        parts.append(f"=== {d} ===")
        parts.append(_load_rubric(d))
        parts.append("")
    parts.append("CHAPTER:")
    parts.append("<<<")
    parts.append(text)
    parts.append(">>>")
    parts.append("")
    parts.append("Return JSON only.")
    return "\n".join(parts)


async def judge_chapter(
    client: InferenceClient, text: str, dims: list[str],
) -> tuple[dict, float]:
    t0 = time.perf_counter()
    raw = await client.chat_structured(
        messages=[
            ChatMessage(role="system", content=JUDGE_SYSTEM),
            ChatMessage(role="user", content=_build_judge_user(text, dims)),
        ],
        json_schema=_build_judge_schema(dims),
        schema_name="chapter_scores",
        temperature=0.2, max_tokens=4000,
    )
    return json.loads(raw), time.perf_counter() - t0


def score_heuristics(text: str) -> dict:
    return run_heuristics(text)


def bootstrap_pipeline(
    db_path: Path, seed_path: Path, server: str, model: str | None,
) -> Pipeline:
    """Fresh DB from seed, return a ready-to-run Pipeline."""
    from app.world import SeedLoader
    from app.world.schema import QuestArcState, ReaderState

    if db_path.is_file():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

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
        sm.add_theme("strategy_sweep", th)
    sm.apply_delta(payload.delta, update_number=0)
    sm.upsert_arc(QuestArcState(
        quest_id="strategy_sweep", arc_id="main",
        structure_id="three_act", scale="campaign",
        current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id="strategy_sweep"))

    raw_seed = json.loads(seed_path.read_text())
    quest_config = {
        "genre": raw_seed.get("genre", ""),
        "premise": raw_seed.get("premise", ""),
        "themes": raw_seed.get("themes", []),
        "narrator": raw_seed.get("narrator", {}),
    }

    client = InferenceClient(base_url=server, timeout=600.0, retries=1, model=model)
    renderer = PromptRenderer(ROOT / "prompts")
    cb = ContextBuilder(sm, renderer, TokenBudget())
    cl = CraftLibrary(ROOT / "app" / "craft" / "data")

    return Pipeline(
        sm, cb, client,
        arc_planner=ArcPlanner(client, renderer),
        dramatic_planner=DramaticPlanner(client, renderer, cl),
        emotional_planner=EmotionalPlanner(client, renderer),
        craft_planner=CraftPlanner(client, renderer, cl),
        craft_library=cl, structure=cl.structure("three_act"),
        quest_config=quest_config, quest_id="strategy_sweep", arc_id="main",
    )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

async def strategy_per_beat(pl: Pipeline, craft_plan, trace, player_action: str,
                            update_number: int) -> str:
    """Current production: dramatic → craft → per-beat writer loop."""
    voice = list(pl._narrator.voice_samples) if pl._narrator else None
    return await pl._run_write(
        trace, craft_plan, voice_samples=voice,
        player_action=player_action, update_number=update_number,
    )


async def strategy_scene(pl: Pipeline, craft_plan, trace, player_action: str,
                          update_number: int) -> str:
    """One writer call per scene. Short-circuits the beat loop by blanking
    the scene's beats list, which triggers Pipeline's legacy per-scene
    single-call branch at len(beats) <= 1."""
    dram = pl._last_dramatic
    saved_beats = {}
    for s in dram.scenes:
        saved_beats[s.scene_id] = list(s.beats or [])
        # Collapse beats to a single synthesized note so writer sees a
        # plan but does not loop; keeps narrative-goal context intact.
        joined = " | ".join(s.beats) if s.beats else s.dramatic_question
        s.beats = [f"Write this scene in full (target ~{pl._scene_target_words} words). Cover: {joined}"]
    try:
        voice = list(pl._narrator.voice_samples) if pl._narrator else None
        return await pl._run_write(
            trace, craft_plan, voice_samples=voice,
            player_action=player_action, update_number=update_number,
        )
    finally:
        for s in dram.scenes:
            s.beats = saved_beats.get(s.scene_id, s.beats)


def _build_one_shot_prompt(pl: Pipeline, craft_plan, player_action: str) -> tuple[str, str]:
    """Assemble a single (system, user) prompt covering the whole update."""
    from app.engine.context_spec import WRITE_SPEC
    # Build a compact plan string for all scenes + briefs
    briefs_by_scene = {b.scene_id: b for b in craft_plan.briefs}
    dram = pl._last_dramatic
    dram_by_scene = {s.scene_id: s for s in dram.scenes}

    plan_parts = ["## Full Plan", ""]
    for scene in craft_plan.scenes:
        sid = scene.scene_id
        brief = briefs_by_scene.get(sid)
        ds = dram_by_scene.get(sid)
        plan_parts.append(f"### Scene {sid} — {ds.location or ''}")
        if ds:
            plan_parts.append(f"POV: {ds.pov_character_id or 'default'}")
            plan_parts.append(f"Dramatic question: {ds.dramatic_question}")
            plan_parts.append(f"Outcome: {ds.outcome}")
            plan_parts.append(f"Characters: {', '.join(ds.characters_present)}")
            plan_parts.append("Beats:")
            for b in ds.beats:
                plan_parts.append(f"  - {b}")
        if brief and getattr(brief, "brief", None):
            plan_parts.append("")
            plan_parts.append(f"Craft brief: {brief.brief}")
        plan_parts.append("")
    plan_text = "\n".join(plan_parts)

    # Resolve all entities referenced across scenes for the writer
    seen = set()
    all_ents = []
    for scene in craft_plan.scenes:
        for e in pl._resolve_scene_entities(scene):
            if e.id not in seen:
                seen.add(e.id)
                all_ents.append(e)

    write_ctx = pl._cb.build(
        spec=WRITE_SPEC, stage_name="write",
        templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
        extras={
            "brief": plan_text,
            "scene": None,
            "voice_samples": list(pl._narrator.voice_samples) if pl._narrator else [],
            "voice_anchors": [], "quest_callbacks": [], "voice_continuity": [],
            "recent_prose_tail": "", "anti_patterns": [],
            "plan": None, "style": "",
            "player_action": player_action,
            "beat": None, "beat_index": None, "total_beats": 0,
            "accumulated_scene_prose": "",
            "scene_target_words": pl._scene_target_words * len(craft_plan.scenes),
            "words_so_far": 0,
            "narrator": pl._narrator,
            "scene_entities": all_ents,
        },
    )
    return write_ctx.system_prompt, write_ctx.user_prompt


async def strategy_one_shot(pl: Pipeline, craft_plan, trace, player_action: str,
                             update_number: int) -> str:
    """Single writer call with the full multi-scene plan."""
    sys_prompt, user_prompt = _build_one_shot_prompt(pl, craft_plan, player_action)
    # Target ~scene_target_words * n_scenes worth of tokens
    n_scenes = max(1, len(craft_plan.scenes))
    max_tokens = min(16384, 2000 * n_scenes)
    raw = await pl._client.chat(
        messages=[
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        temperature=0.8, max_tokens=max_tokens,
    )
    from app.world.output_parser import OutputParser
    return OutputParser.parse_prose(raw)


async def strategy_refine(pl: Pipeline, craft_plan, trace, player_action: str,
                           update_number: int) -> str:
    """one_shot → critic identifies 1 weakest scene → targeted revise.

    The critic is a lightweight second LLM call that picks the scene with
    the weakest emotional / voice execution; the revise call rewrites only
    that scene's prose and splices it in.
    """
    # Pass 1: one-shot
    prose = await strategy_one_shot(pl, craft_plan, trace, player_action, update_number)

    # Pass 2: critic identifies weakest scene
    critic_system = (
        "You are a literary critic reviewing a draft chapter against its plan. "
        "Identify the ONE weakest scene by scene_id (number). Judge on: "
        "voice distinctiveness, emotional trajectory, subtext. "
        "Return JSON: {\"scene_id\": int, \"reason\": string, \"guidance\": string}."
    )
    critic_user = (
        f"## Plan (scene by scene)\n\n"
        + "\n".join(
            f"Scene {s.scene_id}: {getattr(s, 'dramatic_question', '')}"
            for s in pl._last_dramatic.scenes
        )
        + f"\n\n## Draft\n\n{prose}\n\n"
        + "Return JSON only identifying the weakest scene."
    )
    try:
        raw = await pl._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=critic_system),
                ChatMessage(role="user", content=critic_user),
            ],
            json_schema={
                "type": "object",
                "required": ["scene_id", "reason", "guidance"],
                "properties": {
                    "scene_id": {"type": "integer"},
                    "reason": {"type": "string"},
                    "guidance": {"type": "string"},
                },
            },
            schema_name="critic_finding",
            temperature=0.3, max_tokens=1024,
        )
        critique = json.loads(raw)
    except Exception:
        return prose  # degrade gracefully

    # Pass 3: revise targeted scene. We revise the full prose with guidance
    # (rewriting just one scene inline is fragile without structural markers).
    revise_system = (
        "You are revising a chapter draft. Rewrite ONLY the scene identified "
        "by the critic's guidance, keeping all other scenes verbatim. Output "
        "the full revised chapter, not a diff."
    )
    revise_user = (
        f"## Critique\nScene {critique['scene_id']}: {critique['reason']}\n"
        f"Guidance: {critique['guidance']}\n\n"
        f"## Current draft\n\n{prose}\n\n"
        "Rewrite the weak scene to address the guidance; keep other scenes as-is. "
        "Output the full chapter text — no headings, labels, or commentary."
    )
    try:
        raw = await pl._client.chat(
            messages=[
                ChatMessage(role="system", content=revise_system),
                ChatMessage(role="user", content=revise_user),
            ],
            temperature=0.7, max_tokens=16384,
        )
        from app.world.output_parser import OutputParser
        return OutputParser.parse_prose(raw)
    except Exception:
        return prose


async def strategy_expand(pl: Pipeline, craft_plan, trace, player_action: str,
                           update_number: int) -> str:
    """Two-pass: rough 150-word sketch per scene → per-scene expansion.

    Tests whether 'rough draft then polish' produces better voice and
    continuity than per-beat assembly.
    """
    # Pass 1: rough sketches via one LLM call returning {scene_id: sketch}
    dram = pl._last_dramatic
    sketch_system = (
        "You are sketching a chapter from a plan. For each scene below, "
        "write a 100-150-word rough-draft paragraph capturing the scene's "
        "opening and emotional beat. No dialogue, no final prose polish — "
        "a sketch."
    )
    scene_list = "\n".join(
        f"Scene {s.scene_id} ({s.location or 'unspecified'}, POV {s.pov_character_id or 'default'}): "
        f"{s.dramatic_question} -> {s.outcome}"
        for s in dram.scenes
    )
    sketch_user = f"## Scenes\n\n{scene_list}\n\nReturn JSON with sketches keyed by scene_id."
    sketch_schema = {
        "type": "object", "required": ["sketches"],
        "properties": {
            "sketches": {
                "type": "array",
                "items": {
                    "type": "object", "required": ["scene_id", "sketch"],
                    "properties": {
                        "scene_id": {"type": "integer"},
                        "sketch": {"type": "string"},
                    },
                },
            },
        },
    }
    try:
        raw = await pl._client.chat_structured(
            messages=[
                ChatMessage(role="system", content=sketch_system),
                ChatMessage(role="user", content=sketch_user),
            ],
            json_schema=sketch_schema, schema_name="sketches",
            temperature=0.6, max_tokens=4096,
        )
        sketches_obj = json.loads(raw)
        sketches_by_id = {s["scene_id"]: s["sketch"] for s in sketches_obj.get("sketches", [])}
    except Exception:
        sketches_by_id = {}

    # Pass 2: per-scene expansion with sketch as seed
    briefs_by_scene = {b.scene_id: b for b in craft_plan.briefs}
    expansions = []
    for scene in craft_plan.scenes:
        sid = scene.scene_id
        ds = next((d for d in dram.scenes if d.scene_id == sid), None)
        brief = briefs_by_scene.get(sid)
        sketch = sketches_by_id.get(sid, "")
        ents = pl._resolve_scene_entities(scene)
        ent_block = ""
        if ents:
            ent_block = "\n\n## Characters in this scene\n" + "\n".join(
                f"- **{e.name}**: {e.data.get('description', '')}"
                for e in ents
            )
        expand_user = (
            f"## Rough sketch\n{sketch}\n"
            + ent_block
            + f"\n\n## Plan for this scene\n"
            f"POV: {ds.pov_character_id if ds else 'default'}\n"
            f"Location: {ds.location if ds else ''}\n"
            f"Dramatic question: {ds.dramatic_question if ds else ''}\n"
            f"Outcome: {ds.outcome if ds else ''}\n"
            f"Beats: {', '.join(ds.beats) if ds and ds.beats else ''}\n"
            + (f"\n## Craft brief\n{brief.brief}\n" if brief and getattr(brief, 'brief', None) else '')
            + f"\nExpand the sketch into {pl._scene_target_words}-word scene prose. "
              f"Use the characters' descriptions verbatim. Output prose only."
        )
        # Build system prompt via the write system template so narrator config applies
        from app.engine.context_spec import WRITE_SPEC
        wc = pl._cb.build(
            spec=WRITE_SPEC, stage_name="write",
            templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
            extras={
                "brief": None, "scene": None, "voice_samples": [],
                "voice_anchors": [], "quest_callbacks": [], "voice_continuity": [],
                "recent_prose_tail": "", "anti_patterns": [],
                "plan": None, "style": "",
                "player_action": None, "beat": None, "beat_index": None,
                "total_beats": 0, "accumulated_scene_prose": "",
                "scene_target_words": 0, "words_so_far": 0,
                "narrator": pl._narrator, "scene_entities": [],
            },
        )
        try:
            raw = await pl._client.chat(
                messages=[
                    ChatMessage(role="system", content=wc.system_prompt),
                    ChatMessage(role="user", content=expand_user),
                ],
                temperature=0.8, max_tokens=4096,
            )
            from app.world.output_parser import OutputParser
            expansions.append(OutputParser.parse_prose(raw))
        except Exception:
            expansions.append(sketch)

    return "\n\n".join(expansions)


STRATEGIES = {
    "per_beat": strategy_per_beat,
    "scene": strategy_scene,
    "one_shot": strategy_one_shot,
    "refine": strategy_refine,
    "expand": strategy_expand,
}


# ---------------------------------------------------------------------------
# Main sweep orchestration
# ---------------------------------------------------------------------------

async def run_sweep(
    seed_path: Path, action: str, out_dir: Path, strategies: list[str],
    server: str, model: str | None,
) -> list[dict]:
    """Bootstrap ONCE, run planners ONCE, then run each write strategy."""
    db_path = out_dir / "base.db"
    pl = bootstrap_pipeline(db_path, seed_path, server, model)
    trace = PipelineTrace(trace_id=uuid.uuid4().hex, trigger=action)

    print("=== planning (arc/dramatic/emotional/craft) ===", flush=True)
    t_plan = time.time()
    craft_plan, plan_like_dict = await pl._run_hierarchical(trace, action, update_number=1)
    plan_secs = time.time() - t_plan
    print(f"planning done in {plan_secs:.0f}s; {len(craft_plan.scenes)} scenes",
          flush=True)

    trace_store = TraceStore(out_dir / "traces")

    results = []
    for s in strategies:
        print(f"\n=== {s} ===", flush=True)
        fn = STRATEGIES.get(s)
        if fn is None:
            print(f"unknown strategy {s}", flush=True)
            results.append({"strategy": s, "ok": False, "error": "unknown strategy"})
            continue
        t0 = time.time()
        try:
            prose = await fn(pl, craft_plan, trace, action, 1)
            dt = time.time() - t0
            prose_path = out_dir / f"{s}.prose.txt"
            prose_path.write_text(prose)
            heur = score_heuristics(prose)
            results.append({
                "strategy": s, "ok": True, "prose_path": str(prose_path),
                "words": len(prose.split()), "seconds": round(dt, 1),
                "heuristics": heur,
            })
            print(f"  {len(prose.split())}w in {dt:.0f}s", flush=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  FAILED: {e}\n{tb[:500]}", flush=True)
            results.append({
                "strategy": s, "ok": False,
                "seconds": round(time.time() - t0, 1),
                "error": str(e), "traceback": tb[:2000],
            })

    trace_store.save(trace)
    return results


async def main_async(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = Path(args.seed)
    strategies = args.strategies.split(",")

    results = await run_sweep(seed_path, args.action, out_dir, strategies,
                              args.server, args.model)

    # Judge phase
    judge_client = InferenceClient(base_url=args.server, timeout=600.0,
                                    retries=1, model=args.model)
    for r in results:
        if not r.get("ok"):
            continue
        text = Path(r["prose_path"]).read_text()
        print(f"\n=== judging {r['strategy']} ===", flush=True)
        try:
            scores, lat = await judge_chapter(judge_client, text, DEFAULT_JUDGE_DIMS)
            r["judge_scores"] = scores
            r["judge_latency_s"] = round(lat, 2)
            dim_scores = " ".join(
                f"{d[:6]}={scores[d]['score']:.2f}"
                for d in DEFAULT_JUDGE_DIMS if d in scores
            )
            print(f"  {dim_scores}", flush=True)
        except Exception as e:
            r["judge_error"] = str(e)
            print(f"  JUDGE FAILED: {e}", flush=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "seed": str(seed_path), "action": args.action,
        "strategies": strategies, "results": results,
    }, indent=2))
    print(f"\nwrote {summary_path}", flush=True)

    # Headline
    print("\nHEADLINE:")
    print(f"{'strategy':<12}{'words':>7}{'secs':>7}{'tens':>6}"
          f"{'emo':>6}{'voice':>7}{'theme':>7}{'subt':>6}{'int':>6}{'mean':>6}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['strategy']:<12} FAILED: {r.get('error','')[:60]}")
            continue
        js = r.get("judge_scores") or {}
        def g(d): return js.get(d, {}).get("score", 0.0)
        means = [g(d) for d in DEFAULT_JUDGE_DIMS]
        mean = sum(means) / len(means) if means else 0.0
        print(f"{r['strategy']:<12}{r['words']:>7}{r['seconds']:>7.0f}"
              f"{g('tension_execution'):>6.2f}"
              f"{g('emotional_trajectory'):>6.2f}"
              f"{g('voice_distinctiveness'):>7.2f}"
              f"{g('thematic_presence'):>7.2f}"
              f"{g('subtext_presence'):>6.2f}"
              f"{g('interiority_depth'):>6.2f}"
              f"{mean:>6.3f}")


def parse_args():
    ap = argparse.ArgumentParser(prog="strategy_sweep")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default=None)
    ap.add_argument("--strategies", default="per_beat,scene,one_shot,refine,expand",
                    help="comma-separated strategies")
    return ap.parse_args()


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
