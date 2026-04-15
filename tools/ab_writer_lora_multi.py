"""A/B (or N-way) runner for multiple writer LoRA adapters on the same
seed + actions.

Extension of ``tools/ab_writer_lora.py`` (which compares base vs one LoRA) —
this version takes N named targets and runs each through the same pipeline,
scoring every committed scene.

Usage (after vllm is up with ``--enable-lora`` + ``--lora-modules
writer_v1=… writer_v2=…``)::

    uv run python tools/ab_writer_lora_multi.py \\
        --targets v1=writer_v1 v2=writer_v2 \\
        --n-actions 5 \\
        --out data/sft/ab_v1_vs_v2.json

Emits JSON with per-scene scorecards per target + pairwise summary diffs so
``docs/phase2-kickoff-lora-v2.md`` can pull numbers directly.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from pathlib import Path as _Path

# Load the same data the deleted tools/story_gen.py provided.
_REPO = _Path(__file__).resolve().parent.parent
_NOIR_SEED_YAML = _REPO / "tools" / "configs" / "seeds" / "noir.yaml"
_NOIR_ACTIONS_YAML = _REPO / "tools" / "configs" / "actions" / "noir-demo-10.yaml"

STORY_SEED = yaml.safe_load(_NOIR_SEED_YAML.read_text())
# The seed YAML doesn't have "rules" (Pydantic SeedConfig doesn't expose it);
# add an empty list so SeedLoader.load() finds the field it expects.
STORY_SEED.setdefault("rules", [])
STORY_ACTIONS = list(yaml.safe_load(_NOIR_ACTIONS_YAML.read_text()))

PROMPTS = ROOT / "prompts"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ab_writer_lora_multi")
    ap.add_argument(
        "--targets",
        nargs="+",
        default=["v1=writer_v1"],
        help="Space-separated label=model_id pairs (e.g. 'v1=writer_v1 v2=writer_v2').",
    )
    ap.add_argument("--llm-url", type=str,
                    default=os.environ.get("LLM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--n-actions", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path("data/sft/ab_writer_multi.json"))
    return ap.parse_args()


async def run_side(*, model_id: str, llm_url: str, n_actions: int,
                   label: str, workdir: Path) -> dict[str, Any]:
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
            "retrieval": {"enabled": False},
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
            scenes.append({"update": i, "action": action, "error": str(e)})
            continue

        prose = out.prose or ""
        craft_plan = None
        for stage in out.trace.stages:
            if getattr(stage, "stage", "") == "craft" and stage.output:
                craft_plan = stage.output
                break

        card = scorer.score(
            prose, craft_plan=craft_plan, narrator=None,
            world=sm, player_action=action,
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
        print(f"[{label}][{i}] outcome={out.trace.outcome} overall={card.overall_score:.3f}")

    scored = [s for s in scenes if "error" not in s]
    summary = {}
    if scored:
        summary["overall_mean"] = mean(s["overall_score"] for s in scored)
        for name in DIMENSION_NAMES:
            summary[name] = mean(s["dimension_scores"][name] for s in scored)

    return {
        "label": label,
        "model_id": model_id,
        "n_scenes": len(scored),
        "scenes": scenes,
        "summary": summary,
    }


async def main_async() -> None:
    args = parse_args()
    tmp = Path("/tmp/ab_writer_multi")

    targets: list[tuple[str, str]] = []
    for spec in args.targets:
        if "=" not in spec:
            raise SystemExit(f"bad target spec {spec!r}; expected 'label=model_id'")
        label, model_id = spec.split("=", 1)
        targets.append((label, model_id))

    results: dict[str, dict] = {}
    for label, model_id in targets:
        print("=" * 60)
        print(f"RUN: {label} ({model_id})")
        print("=" * 60)
        results[label] = await run_side(
            model_id=model_id,
            llm_url=args.llm_url,
            n_actions=args.n_actions,
            label=label,
            workdir=tmp / label,
        )

    # Pairwise deltas: for each non-first target, compute target − first.
    deltas: dict[str, dict[str, float]] = {}
    labels = list(results.keys())
    if len(labels) >= 2:
        base_label = labels[0]
        base_sum = results[base_label]["summary"]
        for lab in labels[1:]:
            tgt_sum = results[lab]["summary"]
            if base_sum and tgt_sum:
                deltas[f"{lab}_minus_{base_label}"] = {
                    k: tgt_sum[k] - base_sum.get(k, 0.0) for k in tgt_sum
                }

    out = {
        "n_actions": args.n_actions,
        "targets": [{"label": l, "model_id": m} for l, m in targets],
        "results": results,
        "pairwise_deltas": deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.out}")
    for pair_name, diff in deltas.items():
        print(f"\nΔ {pair_name}:")
        for k, v in diff.items():
            print(f"  {k:32s}  {v:+.4f}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
