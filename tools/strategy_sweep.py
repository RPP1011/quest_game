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
    scene         — same as per_beat but one write call per scene
    one_shot      — single writer call given the full dramatic plan
    refine        — one_shot + critic + targeted revision
    expand        — two-pass: rough sketch then per-scene expansion

Compares outputs against Pale Lights Ch 1 baseline when available.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget, TraceStore  # noqa: E402
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
    "tension_execution",
    "emotional_trajectory",
    "choice_hook_quality",
    "update_self_containment",
    "voice_distinctiveness",
    "thematic_presence",
    "subtext_presence",
    "interiority_depth",
]


def _load_rubric(dim: str) -> str:
    return (RUBRICS_DIR / f"{dim}.j2").read_text()


def _build_judge_schema(dims: list[str]) -> dict:
    return {
        "type": "object",
        "required": dims,
        "properties": {
            d: {
                "type": "object",
                "required": ["score", "rationale"],
                "properties": {
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
            }
            for d in dims
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
        temperature=0.2,
        max_tokens=4000,
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


async def run_strategy(
    name: str, seed_path: Path, action: str, out_dir: Path,
    server: str, model: str | None,
) -> dict:
    """Run one strategy end-to-end, return {name, prose_path, words, seconds,
    outcome, heuristics}."""
    db_path = out_dir / f"{name}.db"
    pl = bootstrap_pipeline(db_path, seed_path, server, model)
    trace_store = TraceStore(out_dir / f"{name}_traces")

    t0 = time.time()
    try:
        out = await pl.run(player_action=action, update_number=1)
        dt = time.time() - t0
        prose_path = out_dir / f"{name}.prose.txt"
        prose_path.write_text(out.prose)
        trace_store.save(out.trace)
        heur = score_heuristics(out.prose)
        return {
            "strategy": name, "ok": True,
            "prose_path": str(prose_path), "words": len(out.prose.split()),
            "seconds": round(dt, 1), "outcome": out.trace.outcome,
            "heuristics": heur,
        }
    except Exception as e:
        return {
            "strategy": name, "ok": False,
            "seconds": round(time.time() - t0, 1), "error": str(e),
        }


async def main_async(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = Path(args.seed)
    strategies = args.strategies.split(",")

    # Phase 1: generate
    results = []
    for s in strategies:
        print(f"\n=== {s} ===", flush=True)
        r = await run_strategy(s, seed_path, args.action, out_dir,
                                args.server, args.model)
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)

    # Phase 2: judge
    judge_client = InferenceClient(base_url=args.server, timeout=600.0,
                                    retries=1, model=args.model)
    for r in results:
        if not r.get("ok"):
            continue
        text = Path(r["prose_path"]).read_text()
        print(f"\n=== judging {r['strategy']} ===", flush=True)
        try:
            scores, lat = await judge_chapter(judge_client, text,
                                              DEFAULT_JUDGE_DIMS)
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

    # Phase 3: summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "seed": str(seed_path), "action": args.action,
        "strategies": strategies, "results": results,
    }, indent=2))
    print(f"\nwrote {summary_path}", flush=True)

    # Headline table
    print("\nHEADLINE:")
    print(f"{'strategy':<12}{'words':>7}{'secs':>7}{'tens':>6}"
          f"{'emo':>6}{'voice':>7}{'theme':>7}{'subt':>6}{'int':>6}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['strategy']:<12} FAILED: {r.get('error','')[:60]}")
            continue
        js = r.get("judge_scores") or {}
        def g(d): return js.get(d, {}).get("score", 0.0)
        print(f"{r['strategy']:<12}{r['words']:>7}{r['seconds']:>7.0f}"
              f"{g('tension_execution'):>6.2f}"
              f"{g('emotional_trajectory'):>6.2f}"
              f"{g('voice_distinctiveness'):>7.2f}"
              f"{g('thematic_presence'):>7.2f}"
              f"{g('subtext_presence'):>6.2f}"
              f"{g('interiority_depth'):>6.2f}")


def parse_args():
    ap = argparse.ArgumentParser(prog="strategy_sweep")
    ap.add_argument("--seed", required=True, help="seed JSON path")
    ap.add_argument("--action", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--server", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default=None)
    ap.add_argument("--strategies", default="per_beat",
                    help="comma-separated strategy names (per_beat is the "
                         "only one implemented so far)")
    return ap.parse_args()


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
