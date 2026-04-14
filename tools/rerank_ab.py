"""Day 3 A/B harness: N=1 vs N=10 rerank on the ``story_gen.py`` seed.

Runs the same 3-action seed from ``tools/story_gen.py`` twice — once with
``n_candidates=1`` (baseline) and once with ``n_candidates=10`` (rerank
active). For each action we capture:

- chosen prose + Scorer overall_score
- per-candidate dim scores from the ``write_rerank`` trace stage
- wall-clock time for the pipeline run
- total tokens (from the trace ``token_usage`` rollup; server log is
  authoritative for tok/s but we don't tail it here)

Writes ``/tmp/rerank_ab.json`` with the structured comparison and prints
a compact per-action summary (N=1 vs best-of-N=10 overall score).

Usage
-----

    python -m tools.rerank_ab                # default: vllm on 8082, Scorer on
    VLLM_URL=http://127.0.0.1:8081 python -m tools.rerank_ab
    SKIP_RETRIEVAL=1 python -m tools.rerank_ab   # faster; same seed

No automatic re-runs if the server is unreachable — an exception is
logged in the JSON so the partial results remain inspectable.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROMPTS = Path(__file__).parent.parent / "prompts"

SEED = {
    "entities": [
        {"id": "inn", "entity_type": "location", "name": "The Salt and Star"},
        {"id": "player", "entity_type": "character", "name": "Kaela",
         "data": {
             "voice": {
                 "vocabulary_level": "plain",
                 "sentence_length_bias": "short_clipped",
                 "directness": 0.8,
                 "emotional_expression": "understated",
             },
         }},
        {"id": "innkeeper", "entity_type": "character", "name": "Merrin",
         "data": {"voice": {"vocabulary_level": "coarse", "directness": 0.9}}},
    ],
    "plot_threads": [
        {"id": "pt:main", "name": "The Missing Cargo",
         "description": "A shipment of silver never reached the port; someone in this inn knows where it went.",
         "arc_position": "rising", "priority": 8},
    ],
    "themes": [
        {"id": "t:trust", "proposition": "trust is bought with small tells, not words",
         "stance": "exploring"},
    ],
    "narrator": {
        "pov_type": "third_limited",
        "worldview": "a weathered observer; notices hands and silences",
        "editorial_stance": "sympathetic but unsentimental",
        "sensory_bias": {"visual": 0.4, "auditory": 0.2, "tactile": 0.2,
                         "kinesthetic": 0.2},
        "attention_bias": ["hands", "doorways", "what people don't say"],
        "voice_samples": [
            "She set the cup down the way she did everything else — like the cup owed her rent.",
        ],
    },
}

ACTIONS = [
    "I study the room, looking for who's trying too hard not to be noticed.",
    "I sit at Merrin's bar and ask whether the Gannet crew came through last week.",
    "I wait for her answer without filling the silence.",
]


def _build_pipeline(tmp_dir: Path, n_candidates: int, vllm_url: str):
    """Construct a full hierarchical pipeline. Mirrors ``story_gen.py`` with
    the single exception of the ``n_candidates`` / ``scorer`` kwargs that
    this harness varies between runs."""
    from app.cli.play import _open_world  # noqa: F401 — parity with story_gen
    from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget
    from app.runtime.client import InferenceClient
    from app.scoring import Scorer
    from app.world import SeedLoader
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    from app.world.schema import QuestArcState, ReaderState

    # Fresh DB per pipeline so the two runs don't cross-contaminate.
    db_path = tmp_dir / "quest.db"
    if db_path.exists():
        db_path.unlink()
    seed_path = tmp_dir / "seed.json"
    seed_path.write_text(json.dumps(SEED))

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
        sm.add_theme("demo", th)
    sm.apply_delta(payload.delta, update_number=0)

    from app.craft.library import CraftLibrary
    craft_library = CraftLibrary(Path(__file__).parent.parent / "app" / "craft" / "data")
    structure = craft_library.structure("three_act")
    sm.upsert_arc(QuestArcState(
        quest_id="demo", arc_id="main",
        structure_id="three_act", scale="campaign",
        current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id="demo"))

    client = InferenceClient(base_url=vllm_url, retries=1)
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(sm, renderer, TokenBudget())

    from app.planning import DramaticPlanner, EmotionalPlanner, CraftPlanner
    from app.planning.arc_planner import ArcPlanner
    dramatic = DramaticPlanner(client, renderer, craft_library)
    emotional = EmotionalPlanner(client, renderer)
    craft = CraftPlanner(client, renderer, craft_library)
    arc = ArcPlanner(client, renderer)

    retrieval_on = os.environ.get("SKIP_RETRIEVAL", "0") != "1"
    passage_retriever = quest_retriever = voice_retriever = None
    embedder = None
    if retrieval_on:
        try:
            from app.retrieval import (
                Embedder, PassageRetriever, QuestRetriever, VoiceRetriever,
            )
            manifest = Path(__file__).parent.parent / "data" / "calibration" / "manifest.yaml"
            passages = Path(__file__).parent.parent / "data" / "calibration" / "passages"
            if manifest.is_file():
                passage_retriever = PassageRetriever(
                    manifest, Path("/tmp"), passages, enable_semantic=False,
                )
            embedder = Embedder()
            quest_retriever = QuestRetriever(sm, "demo", embedder=embedder)
            voice_retriever = VoiceRetriever(sm, "demo")
        except Exception as e:
            print(f"[retrieval skipped: {e}]", file=sys.stderr)

    pipeline = Pipeline(
        sm, cb, client,
        arc_planner=arc,
        dramatic_planner=dramatic,
        emotional_planner=emotional,
        craft_planner=craft,
        craft_library=craft_library,
        structure=structure,
        quest_config={
            "narrator": SEED["narrator"],
            "genre": "low-fantasy noir",
            "retrieval": {"enabled": retrieval_on},
            "scoring": {"enabled": True},
            "n_candidates": n_candidates,
        },
        quest_id="demo",
        arc_id="main",
        passage_retriever=passage_retriever,
        quest_retriever=quest_retriever,
        voice_retriever=voice_retriever,
        retrieval_embedder=embedder,
        scorer=Scorer(),
    )
    return pipeline, conn


def _rollup_tokens(trace) -> int:
    total = 0
    for s in trace.stages:
        try:
            total += int(s.token_usage.prompt or 0)
            total += int(getattr(s.token_usage, "completion", 0) or 0)
        except Exception:
            pass
    return total


def _extract_rerank_detail(trace) -> list[dict[str, Any]]:
    """Pull ``write_rerank`` stage detail objects; skip write_rerank-less runs."""
    out: list[dict[str, Any]] = []
    for s in trace.stages:
        if s.stage_name == "write_rerank":
            out.append({
                "winner_index": s.detail.get("winner_index"),
                "winner_score": s.detail.get("winner_score"),
                "rerank_source": s.detail.get("rerank_source"),
                "candidates": s.detail.get("candidates", []),
                "scene_id": s.detail.get("scene_id"),
            })
    return out


def _extract_scoring_detail(trace) -> dict[str, Any] | None:
    for s in trace.stages:
        if s.stage_name == "scoring":
            return {
                "overall_score": s.detail.get("overall_score"),
                "dimensions": s.detail.get("dimensions", {}),
            }
    return None


async def _run_one(label: str, n_candidates: int, vllm_url: str) -> dict[str, Any]:
    tmp = Path(f"/tmp/rerank_ab_{label}")
    tmp.mkdir(parents=True, exist_ok=True)
    pipeline, conn = _build_pipeline(tmp, n_candidates, vllm_url)
    result: dict[str, Any] = {
        "label": label,
        "n_candidates": n_candidates,
        "per_action": [],
    }
    try:
        update_number = 1
        for action in ACTIONS:
            t0 = time.perf_counter()
            try:
                out = await pipeline.run(
                    player_action=action, update_number=update_number,
                )
                elapsed = time.perf_counter() - t0
                result["per_action"].append({
                    "update_number": update_number,
                    "action": action,
                    "outcome": out.trace.outcome,
                    "prose": out.prose,
                    "choices": out.choices,
                    "wall_ms": int(elapsed * 1000),
                    "total_tokens_prompt": _rollup_tokens(out.trace),
                    "reranks": _extract_rerank_detail(out.trace),
                    "scoring": _extract_scoring_detail(out.trace),
                    "n_stages": len(out.trace.stages),
                })
            except Exception as e:
                elapsed = time.perf_counter() - t0
                result["per_action"].append({
                    "update_number": update_number,
                    "action": action,
                    "wall_ms": int(elapsed * 1000),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                # Continue to the next action — don't let a single
                # chapter bail out of the whole A/B. A real run still
                # surfaces the error in the JSON output.
            update_number += 1
    finally:
        conn.close()
    return result


def _summary(baseline: dict[str, Any], rerank: dict[str, Any]) -> str:
    lines = [
        "",
        "=" * 60,
        f"A/B summary: N={baseline['n_candidates']} vs N={rerank['n_candidates']}",
        "=" * 60,
    ]
    for b, r in zip(baseline["per_action"], rerank["per_action"]):
        upd = b.get("update_number")
        b_score = (b.get("scoring") or {}).get("overall_score")
        r_score = (r.get("scoring") or {}).get("overall_score")
        b_ms = b.get("wall_ms")
        r_ms = r.get("wall_ms")
        # Best-of-N candidate overall score across all rerank stages
        best_candidate_overall: float | None = None
        for rr in r.get("reranks", []) or []:
            for cand in rr.get("candidates", []) or []:
                o = cand.get("overall_score")
                if o is None:
                    continue
                if best_candidate_overall is None or o > best_candidate_overall:
                    best_candidate_overall = o
        lines.append(
            f"  action {upd}: "
            f"N=1 overall={b_score!s:.6s}  wall={b_ms}ms   |   "
            f"N=10 overall={r_score!s:.6s}  best-cand={best_candidate_overall!s:.6s}  wall={r_ms}ms"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


async def main() -> int:
    vllm_url = os.environ.get("VLLM_URL", "http://127.0.0.1:8082")
    out_path = Path(os.environ.get("RERANK_AB_OUT", "/tmp/rerank_ab.json"))

    print(f"[rerank_ab] server: {vllm_url}")
    print(f"[rerank_ab] output: {out_path}")

    baseline = await _run_one("baseline_n1", n_candidates=1, vllm_url=vllm_url)
    rerank = await _run_one("rerank_n10", n_candidates=10, vllm_url=vllm_url)

    bundle = {
        "vllm_url": vllm_url,
        "seed": "tools/story_gen.py::SEED",
        "actions": ACTIONS,
        "baseline": baseline,
        "rerank": rerank,
    }
    out_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"[rerank_ab] wrote {out_path}")

    print(_summary(baseline, rerank))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
