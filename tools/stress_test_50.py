"""Day 10: 50-chapter stress test.

Runs 50 player-action updates against a freshly-seeded noir quest with the
full hierarchical pipeline, all retrievers, Scorer, LLM judge, SFT
collection, and ``N=4`` candidates per scene.

Each update's metrics flush to ``data/stress/run_log.jsonl`` immediately so
a crash mid-run leaves the partial data behind. Pipeline exceptions are
caught and logged as ``fallback`` rows; the run continues.

Usage::

    uv run python tools/stress_test_50.py \\
        [--updates 50] [--n 4] [--lora writer_v1] \\
        [--llm-url http://127.0.0.1:8082] [--out data/stress/run_log.jsonl]

Reuses ``tools/story_gen.SEED`` verbatim so the quest framing is identical
to the canonical dev run, and ``tools/stress_personas`` for action picks.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# vllm owns the GPU; force the retrieval embedder onto CPU so the
# narrative-embedding persist hook doesn't CUDA-OOM every commit. This
# must happen before torch / sentence_transformers is imported, which
# is why it's at module top. Day 10 surfaced this as a persistent
# consistency flag.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.story_gen import SEED as STORY_SEED  # noqa: E402
from tools.stress_personas import pick_action, persona_for  # noqa: E402

PROMPTS = ROOT / "prompts"
CALIB = ROOT / "data" / "calibration"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="stress_test_50")
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument("--n", type=int, default=4,
                    help="Candidates per scene (writer fan-out).")
    ap.add_argument("--lora", type=str,
                    default=os.environ.get("LLM_MODEL", "writer_v1"),
                    help="vllm model name for the LoRA adapter.")
    ap.add_argument("--llm-url", type=str,
                    default=os.environ.get("LLM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--workdir", type=Path,
                    default=Path("/tmp/stress_test_50"))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "stress" / "run_log.jsonl")
    ap.add_argument("--scoring", action="store_true", default=True,
                    help="Enable post-commit Scorer + scorecard persistence.")
    ap.add_argument("--llm-judge", action="store_true", default=True,
                    help="Enable async LLM-judge dims (Day 6).")
    ap.add_argument("--retry-on-fail", type=int, default=0,
                    help="Rarely useful; default 0. We keep going on crash.")
    return ap.parse_args()


class HitCounter:
    """Monkey-patch wrapper for a Retriever — counts calls + nonzero hits."""

    def __init__(self, inner: Any, kind: str) -> None:
        self._inner = inner
        self.kind = kind
        self.calls = 0
        self.total_hits = 0
        self.errors = 0

    async def retrieve(self, query: Any, k: int = 3, **kw: Any) -> list:
        self.calls += 1
        try:
            results = await self._inner.retrieve(query, k=k, **kw)
        except Exception:
            self.errors += 1
            raise
        try:
            n = len(results)
        except Exception:
            n = 0
        self.total_hits += n
        return results

    def __getattr__(self, item: str) -> Any:
        # Forward other attributes (e.g. state, config) to the inner retriever.
        return getattr(self._inner, item)

    def snapshot(self) -> dict[str, int]:
        out = {
            f"{self.kind}_calls": self.calls,
            f"{self.kind}_hits": self.total_hits,
            f"{self.kind}_errors": self.errors,
        }
        self.calls = 0
        self.total_hits = 0
        self.errors = 0
        return out


async def _bootstrap(workdir: Path, lora: str, llm_url: str, n_candidates: int,
                     scoring_on: bool, llm_judge_on: bool):
    """Build the fresh quest world, wire everything, return (pipeline, world, client, counters)."""
    from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget
    from app.runtime.client import InferenceClient
    from app.world import SeedLoader
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    from app.world.schema import QuestArcState, ReaderState
    from app.craft.library import CraftLibrary
    from app.planning import DramaticPlanner, EmotionalPlanner, CraftPlanner
    from app.planning.arc_planner import ArcPlanner
    from app.retrieval import (
        Embedder, PassageRetriever, QuestRetriever, SceneShapeRetriever,
        MotifRetriever, ForeshadowingRetriever, VoiceRetriever,
    )
    from app.scoring import Scorer

    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "quest.db"
    if db_path.exists():
        db_path.unlink()
    seed_path = workdir / "seed.json"
    seed_path.write_text(json.dumps(STORY_SEED))

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
        sm.add_theme("stress", th)
    sm.apply_delta(payload.delta, update_number=0)

    craft_library = CraftLibrary(ROOT / "app" / "craft" / "data")
    structure = craft_library.structure("three_act")
    sm.upsert_arc(QuestArcState(
        quest_id="stress", arc_id="main",
        structure_id="three_act", scale="campaign",
        current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id="stress"))

    client = InferenceClient(
        base_url=llm_url,
        timeout=300.0,
        retries=1,
        model=lora,
    )
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(sm, renderer, TokenBudget())

    dramatic = DramaticPlanner(client, renderer, craft_library)
    emotional = EmotionalPlanner(client, renderer)
    craft = CraftPlanner(client, renderer, craft_library)
    arc = ArcPlanner(client, renderer)

    # Wire every retriever (Waves 1-4).
    manifest = CALIB / "manifest.yaml"
    passages = CALIB / "passages"
    scenes_manifest = CALIB / "scenes_manifest.yaml"
    scenes = CALIB / "scenes"
    passage_retriever = None
    scene_retriever = None
    if manifest.is_file():
        passage_retriever = PassageRetriever(
            manifest, Path("/tmp"), passages, enable_semantic=False,
        )
    if scenes_manifest.is_file():
        try:
            scene_retriever = SceneShapeRetriever(
                scenes_manifest, "/tmp/labels_claude_arc_*.json", scenes,
            )
        except Exception as e:
            print(f"[scene retriever skipped: {e}]")
    embedder = Embedder()
    quest_retriever = QuestRetriever(sm, "stress", embedder=embedder)
    motif_retriever = MotifRetriever(sm, "stress")
    foreshadow_retriever = ForeshadowingRetriever(sm, "stress")
    voice_retriever = VoiceRetriever(sm, "stress")

    # Wrap with hit counters.
    counters: dict[str, HitCounter] = {}
    if passage_retriever is not None:
        passage_retriever = HitCounter(passage_retriever, "passage")
        counters["passage"] = passage_retriever
    quest_retriever = HitCounter(quest_retriever, "quest")
    counters["quest"] = quest_retriever
    voice_retriever = HitCounter(voice_retriever, "voice")
    counters["voice"] = voice_retriever
    motif_retriever_wrapped = HitCounter(motif_retriever, "motif")
    counters["motif"] = motif_retriever_wrapped
    foreshadow_retriever_wrapped = HitCounter(foreshadow_retriever, "foreshadowing")
    counters["foreshadowing"] = foreshadow_retriever_wrapped

    # Scorer (Day 2) + LLM-judge client (Day 6).
    scorer: Any | None = None
    llm_judge_client: Any | None = None
    if scoring_on:
        if llm_judge_on:
            llm_judge_client = client
            scorer = Scorer(llm_judge_client=client)
        else:
            scorer = Scorer()

    pipeline = Pipeline(
        sm, cb, client,
        arc_planner=arc,
        dramatic_planner=dramatic,
        emotional_planner=emotional,
        craft_planner=craft,
        craft_library=craft_library,
        structure=structure,
        quest_config={
            "narrator": STORY_SEED["narrator"],
            "genre": "low-fantasy noir",
            "retrieval": {"enabled": True},
            "sft_collection": {
                "enabled": True,
                "dir": str(workdir / "sft"),
            },
            "scoring": {"enabled": scoring_on},
            "n_candidates": n_candidates,
        },
        quest_id="stress",
        arc_id="main",
        passage_retriever=passage_retriever,
        quest_retriever=quest_retriever,
        voice_retriever=voice_retriever,
        retrieval_embedder=embedder,
        scorer=scorer,
        llm_judge_client=llm_judge_client,
    )
    return pipeline, sm, client, counters


def _world_state_snapshot(sm: Any) -> dict[str, int]:
    """Return size metrics for the mutable world state."""
    try:
        entities = len(sm.list_entities())
    except Exception:
        entities = -1
    try:
        narrative = len(sm.list_narrative(limit=10_000))
    except Exception:
        narrative = -1
    try:
        embeddings = len(sm.list_narrative_embeddings("stress", limit=10_000))
    except Exception:
        embeddings = -1
    try:
        plot_threads = len(sm.list_plot_threads())
    except Exception:
        plot_threads = -1
    return {
        "entities": entities,
        "narrative_records": narrative,
        "narrative_embeddings": embeddings,
        "plot_threads": plot_threads,
    }


def _dim_scores_from_trace(trace: Any) -> dict[str, float]:
    """Pull per-dim scorecard scores from the trace's ``scoring`` stage, if any."""
    for stage in getattr(trace, "stages", []):
        if getattr(stage, "stage_name", "") == "scoring":
            detail = getattr(stage, "detail", None) or {}
            dims = detail.get("dimensions") or {}
            out = {k: float(v) for k, v in dims.items()}
            if "overall_score" in detail:
                out["overall_score"] = float(detail["overall_score"])
            return out
    return {}


def _context_tokens_from_trace(trace: Any) -> int:
    """Largest ``write`` stage prompt-token estimate as a proxy for CB size."""
    best = 0
    for stage in getattr(trace, "stages", []):
        if getattr(stage, "stage_name", "") == "write":
            tu = getattr(stage, "token_usage", None)
            if tu is not None and tu.prompt and tu.prompt > best:
                best = int(tu.prompt)
    return best


def _consistency_flags(trace: Any) -> dict[str, int]:
    """Count error/warning signals across the trace.

    ``errors`` = any ``errors`` list on any stage (critic failures,
    extract crashes, etc.). ``fallbacks`` = any stage whose error kind
    ends in ``_fallback`` (planner fell back to a minimal plan). The
    counts are strictly informational — they don't block the commit.
    """
    n_errors = 0
    n_fallbacks = 0
    kinds: list[str] = []
    messages: list[str] = []
    for stage in getattr(trace, "stages", []):
        for e in getattr(stage, "errors", []) or []:
            n_errors += 1
            k = getattr(e, "kind", "") or ""
            kinds.append(k)
            msg = getattr(e, "message", "") or ""
            if msg:
                messages.append(f"{k}: {msg[:200]}")
            if k.endswith("_fallback"):
                n_fallbacks += 1
    return {
        "errors": n_errors,
        "fallbacks": n_fallbacks,
        "kinds": kinds,
        "messages": messages[:6],  # cap to keep rows small
    }


def _last_prose_tail(sm: Any, chars: int = 400) -> str:
    try:
        rec = sm.list_narrative(limit=1)
    except Exception:
        return ""
    if not rec:
        return ""
    t = rec[-1].raw_text or ""
    return t[-chars:]


async def run_updates(args: argparse.Namespace) -> None:
    pipeline, sm, client, counters = await _bootstrap(
        args.workdir, args.lora, args.llm_url, args.n,
        scoring_on=args.scoring, llm_judge_on=args.llm_judge,
    )

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate prior run log; callers that want to resume should pass a
    # different --out. We use append-only writes after this.
    out_path.write_text("")

    run_meta = {
        "_meta": True,
        "start_ts": time.time(),
        "model": args.lora,
        "llm_url": args.llm_url,
        "n_candidates": args.n,
        "updates_target": args.updates,
        "scoring": args.scoring,
        "llm_judge": args.llm_judge,
        "seed_quest": "stress (noir, The Salt and Star)",
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_meta) + "\n")

    # Ensure the async LLM-judge tasks complete before we exit; we gather
    # their handles so shutdown is clean.
    judge_tasks: list[Any] = []

    for update_number in range(1, args.updates + 1):
        persona, cycle = persona_for(update_number)
        tail = _last_prose_tail(sm)
        action = pick_action(persona, cycle)

        # If the tail is present, append a one-line grounding cue so the
        # action doesn't drift totally loose from the scene. Keep it short
        # so the action-fidelity critic still measures something coherent.
        tail_hint = tail.strip().split(".")[-1].strip() if tail else ""
        if tail_hint:
            action_full = action
        else:
            action_full = action

        print(f"\n[{update_number}/{args.updates}] persona={persona.id} "
              f"action={action_full!r}")

        t0 = time.perf_counter()
        row: dict[str, Any] = {
            "update_number": update_number,
            "persona_id": persona.id,
            "persona_cycle": cycle,
            "action": action_full,
            "tail_chars": len(tail),
        }
        try:
            out = await pipeline.run(
                player_action=action_full, update_number=update_number,
            )
            latency = time.perf_counter() - t0
            trace = out.trace
            row.update({
                "wall_clock_seconds": round(latency, 3),
                "outcome": trace.outcome,
                "n_stages": len(trace.stages),
                "prompt_tokens": trace.total_tokens.prompt,
                "completion_tokens": trace.total_tokens.completion,
                "thinking_tokens": trace.total_tokens.thinking,
                "context_tokens": _context_tokens_from_trace(trace),
                "prose_chars": len(out.prose or ""),
                "n_choices": len(out.choices or []),
                "dimension_scores": _dim_scores_from_trace(trace),
                "world_state": _world_state_snapshot(sm),
                "consistency_flags": _consistency_flags(trace),
            })
            # Snapshot and reset retrieval counters so each row shows
            # per-update retrieval activity, not cumulative.
            ret: dict[str, int] = {}
            for hc in counters.values():
                ret.update(hc.snapshot())
            row["retrieval"] = ret
            row["error"] = None

            # Grab the async judge handle so we can await at the end.
            jt = getattr(pipeline, "last_llm_judge_task", None)
            if jt is not None:
                judge_tasks.append(jt)

            print(
                f"    -> outcome={trace.outcome} "
                f"t={latency:.1f}s ctx={row['context_tokens']}tok "
                f"overall={row['dimension_scores'].get('overall_score', float('nan')):.3f} "
                f"hits={sum(v for k, v in ret.items() if k.endswith('_hits'))}"
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            row.update({
                "wall_clock_seconds": round(latency, 3),
                "outcome": "fallback",
                "n_stages": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "thinking_tokens": 0,
                "context_tokens": 0,
                "prose_chars": 0,
                "n_choices": 0,
                "dimension_scores": {},
                "world_state": _world_state_snapshot(sm),
                "consistency_flags": {"errors": 0, "fallbacks": 0, "kinds": []},
                "retrieval": {},
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=8),
            })
            print(f"    !! ERROR: {type(e).__name__}: {e}")

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # Let any pending LLM-judge tasks finish so their rows actually persist.
    if judge_tasks:
        print(f"\n[awaiting {len(judge_tasks)} pending LLM-judge tasks]")
        results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        fails = [r for r in results if isinstance(r, Exception)]
        if fails:
            print(f"  [warn] {len(fails)} judge tasks raised; see stderr above")

    print(f"\n[done] wrote {out_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_updates(args))


if __name__ == "__main__":
    main()
