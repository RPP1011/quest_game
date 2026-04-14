"""Day-5 A/B runner: base model vs. writer LoRA on the same 3-action seed.

Reuses the story_gen.py SEED + ACTIONS (first 3) by importing them, but
builds a fresh world per side so the two runs don't share state. After
each action commits, scores the committed prose via ``app.scoring.Scorer``
and records per-dim / overall numbers.

Usage (after starting vllm with ``--enable-lora`` and
``--lora-modules writer_v1=data/sft/lora_writer_v1``)::

    uv run python tools/ab_writer_lora.py \\
        --base-model LiquidAI/LFM2.5-1.2B-Instruct \\
        --lora-name writer_v1 \\
        --n-actions 3 \\
        --out data/sft/ab_day5.json

Emits a JSON file with per-scene scorecards for both sides and a summary
diff — that artefact is what `docs/day5-writer-lora-v1.md` consumes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

# Import fixture from story_gen so we write once.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.story_gen import ACTIONS as STORY_ACTIONS, SEED as STORY_SEED  # noqa: E402

PROMPTS = ROOT / "prompts"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ab_writer_lora")
    ap.add_argument("--base-model", type=str,
                    default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--lora-name", type=str, default="writer_v1",
                    help="Model name served by vllm for the LoRA adapter.")
    ap.add_argument("--llm-url", type=str,
                    default=os.environ.get("LLM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--n-actions", type=int, default=3,
                    help="How many actions from story_gen.ACTIONS to run.")
    ap.add_argument("--out", type=Path, default=Path("data/sft/ab_day5.json"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    return ap.parse_args()


async def run_side(*, model_id: str, llm_url: str, n_actions: int,
                   label: str, workdir: Path) -> dict[str, Any]:
    """Run ``n_actions`` updates with ``model_id`` and score each committed
    scene. Returns a dict with per-scene records + summary.
    """
    from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget
    from app.runtime.client import InferenceClient
    from app.world import SeedLoader
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    from app.world.schema import QuestArcState, ReaderState
    from app.craft.library import CraftLibrary
    from app.planning import DramaticPlanner, EmotionalPlanner, CraftPlanner
    from app.planning.arc_planner import ArcPlanner
    from app.scoring import Scorer, DIMENSION_NAMES

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
        sm.add_theme("demo", th)
    sm.apply_delta(payload.delta, update_number=0)

    craft_library = CraftLibrary(ROOT / "app" / "craft" / "data")
    structure = craft_library.structure("three_act")
    sm.upsert_arc(QuestArcState(
        quest_id="demo", arc_id="main",
        structure_id="three_act", scale="campaign",
        current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id="demo"))

    client = InferenceClient(base_url=llm_url, retries=1, model=model_id)
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(sm, renderer, TokenBudget())

    dramatic = DramaticPlanner(client, renderer, craft_library)
    emotional = EmotionalPlanner(client, renderer)
    craft = CraftPlanner(client, renderer, craft_library)
    arc = ArcPlanner(client, renderer)

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
            "retrieval": {"enabled": False},  # keep the two sides comparable
            "sft_collection": {"enabled": False},
            "n_candidates": 1,
        },
        quest_id="demo",
        arc_id="main",
    )

    scorer = Scorer()
    scenes: list[dict[str, Any]] = []

    for i, action in enumerate(STORY_ACTIONS[:n_actions], start=1):
        print(f"[{label}][{i}/{n_actions}] {action!r}")
        try:
            out = await pipeline.run(player_action=action, update_number=i)
        except Exception as e:
            print(f"[{label}][{i}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            scenes.append({"update": i, "action": action, "error": str(e)})
            continue

        prose = out.prose or ""
        # Pull the craft plan from the trace stages if present (for the
        # craft-derived critics). Fall back to None if we can't find it.
        craft_plan = None
        for stage in out.trace.stages:
            if getattr(stage, "stage", "") == "craft" and stage.output:
                craft_plan = stage.output
                break

        card = scorer.score(
            prose,
            craft_plan=craft_plan,
            narrator=None,
            world=sm,
            player_action=action,
        )
        dims = {name: float(getattr(card, name, 0.0)) for name in DIMENSION_NAMES}
        scenes.append({
            "update": i,
            "action": action,
            "prose": prose,
            "outcome": out.trace.outcome,
            "overall_score": float(card.overall_score),
            "dimension_scores": dims,
        })
        print(f"[{label}][{i}] outcome={out.trace.outcome} "
              f"overall={card.overall_score:.3f}")

    scored = [s for s in scenes if "error" not in s]
    summary = {}
    if scored:
        summary["overall_mean"] = mean(s["overall_score"] for s in scored)
        for name in DIMENSION_NAMES:
            summary[name] = mean(
                s["dimension_scores"][name] for s in scored
            )

    return {
        "label": label,
        "model_id": model_id,
        "n_scenes": len(scored),
        "scenes": scenes,
        "summary": summary,
    }


async def main_async() -> None:
    args = parse_args()
    tmp = Path("/tmp/storygen_ab")

    print("=" * 60)
    print("BASE run")
    print("=" * 60)
    base = await run_side(
        model_id=args.base_model,
        llm_url=args.llm_url,
        n_actions=args.n_actions,
        label="base",
        workdir=tmp / "base",
    )

    print()
    print("=" * 60)
    print(f"LORA run ({args.lora_name})")
    print("=" * 60)
    lora = await run_side(
        model_id=args.lora_name,
        llm_url=args.llm_url,
        n_actions=args.n_actions,
        label="lora",
        workdir=tmp / "lora",
    )

    result = {
        "n_actions": args.n_actions,
        "base": base,
        "lora": lora,
    }
    diff = {}
    if base["summary"] and lora["summary"]:
        for k in base["summary"]:
            diff[k] = lora["summary"][k] - base["summary"][k]
    result["delta_lora_minus_base"] = diff

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {args.out}")
    print("\nSummary (lora − base):")
    for k, v in diff.items():
        print(f"  {k:32s}  {v:+.4f}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
