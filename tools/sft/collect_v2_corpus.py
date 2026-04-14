"""Phase-2 kickoff: collect the LoRA v2 SFT corpus across three diverse seeds.

Mirrors the structure of ``tools/stress_test_50.py`` (flush-after-every-update,
try/except per update, SFT collection on, N candidates per scene) but runs
against three seeds with distinct narrator configs, quest_ids, and actions.

Output layout::

    data/sft/<quest_id>/<quest_id>/u<u>_s<s>_<trace>.json
    data/stress_v2/<quest_id>_run_log.jsonl

Usage::

    uv run python tools/sft/collect_v2_corpus.py \\
        --seed noir|intrigue|heist --updates 10 --n 8

Or run all three back-to-back::

    uv run python tools/sft/collect_v2_corpus.py --seed all --updates 10 --n 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

# Same as stress_test_50 — force retrieval embedder onto CPU so vllm owns GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS = ROOT / "prompts"
CALIB = ROOT / "data" / "calibration"


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

NOIR_SEED: dict = {
    "entities": [
        {"id": "inn", "entity_type": "location", "name": "The Salt and Star"},
        {"id": "player", "entity_type": "character", "name": "Kaela",
         "data": {"voice": {
             "vocabulary_level": "plain",
             "sentence_length_bias": "short_clipped",
             "directness": 0.8,
             "emotional_expression": "understated",
         }}},
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
        "pov_character_id": "player",
        "pov_type": "third_limited",
        "worldview": "a weathered observer; notices hands and silences",
        "editorial_stance": "sympathetic but unsentimental",
        "sensory_bias": {"visual": 0.3, "tactile": 0.2, "auditory": 0.15,
                         "kinesthetic": 0.15, "interoceptive": 0.15,
                         "olfactory": 0.05},
        "attention_bias": ["hands", "doorways", "what people don't say"],
        "voice_samples": [
            "She set the cup down the way she did everything else — like the cup owed her rent.",
            "He didn't speak. Didn't have to. The silence did the asking, and it was patient.",
            "The room had four people in it when you walked in — the innkeeper behind the bar, two dock-hands at the corner table sharing a bowl of something that had stopped steaming an hour ago, and a woman in a grey coat who was trying not to be a fifth.",
            "Rain. Just rain. The kind that soaks through before you notice it, and by the time you do, there's no point running.",
        ],
    },
}

NOIR_ACTIONS = (
    "I study the room, looking for who's trying too hard not to be noticed.",
    "I sit at Merrin's bar and ask whether the Gannet crew came through last week.",
    "I wait for her answer without filling the silence.",
    "I order a second drink to give her time to talk.",
    "I describe the silver shipment to her — weight, markings, the broker's name.",
    "I let my hand show on the bar, the old burn scar facing up where she'll see it.",
    "\"So who paid you to forget the shipment?\" I ask, keeping my voice level.",
    "I lean in and ask what Merrin saw on the night of the last delivery.",
    "I leave the inn and follow whoever pretends not to follow me.",
    "I find an alley with one entrance and wait there.",
    "When my pursuer rounds the corner, I step out and ask their name.",
    "I make the offer plain: tell me about the cargo or I tell the harbour master what I already know.",
)

# -----------------------------------------------------------------------------

INTRIGUE_SEED: dict = {
    "entities": [
        {"id": "court", "entity_type": "location", "name": "The Iron Gallery"},
        {"id": "player", "entity_type": "character", "name": "Ambassador Vell",
         "data": {"voice": {
             "vocabulary_level": "formal",
             "sentence_length_bias": "measured",
             "directness": 0.4,
             "emotional_expression": "controlled",
         }}},
        {"id": "spymaster", "entity_type": "character", "name": "Ceriel",
         "data": {"voice": {"vocabulary_level": "precise",
                            "directness": 0.3}}},
        {"id": "heir", "entity_type": "character", "name": "Prince Oren",
         "data": {"voice": {"vocabulary_level": "archaic",
                            "directness": 0.6,
                            "emotional_expression": "bitter"}}},
    ],
    "plot_threads": [
        {"id": "pt:main", "name": "The Displaced Crown",
         "description": "Prince Oren was passed over for succession; Vell must decide whose faction the embassy will back before the regent's council convenes.",
         "arc_position": "rising", "priority": 9},
        {"id": "pt:b", "name": "Ceriel's Price",
         "description": "The spymaster has a proposal for the ambassador — the terms are not yet spoken aloud.",
         "arc_position": "rising", "priority": 6},
    ],
    "themes": [
        {"id": "t:power", "proposition": "power is what one can decline to use in public",
         "stance": "exploring"},
        {"id": "t:loyalty", "proposition": "a promise you cannot name is the one you keep",
         "stance": "exploring"},
    ],
    "narrator": {
        "pov_character_id": "player",
        "pov_type": "third_limited",
        "worldview": "an eye for power currents and unsaid promises",
        "editorial_stance": "sardonic, politically literate, amused by rituals",
        "sensory_bias": {"visual": 0.35, "auditory": 0.25, "tactile": 0.1,
                         "kinesthetic": 0.1, "interoceptive": 0.15,
                         "olfactory": 0.05},
        "attention_bias": ["seating arrangements", "what goes unsaid",
                           "half-bowed heads", "eye-lines across a room"],
        "voice_samples": [
            "The regent spoke of tradition the way a gambler speaks of luck — constantly, and only after winning.",
            "Ceriel's silences were the kind you had to pay for; his words were cheaper.",
            "Oren bowed. He didn't mean it. Everyone in the room knew he didn't mean it, and that was the point.",
            "The antechamber was built for three conversations to happen in it at once — one out loud for the regent, one at the window between two dukes, and a third in the corner of the room where the envoy and the spymaster would not appear to have met, though of course they had.",
        ],
    },
}

INTRIGUE_ACTIONS = (
    "I arrive at the Iron Gallery and read the seating before I sit.",
    "I ask Ceriel, without looking at him, what the regent has already agreed to.",
    "\"Your highness,\" I say to Prince Oren, \"I hear you've been waiting to be asked.\"",
    "I let the court announcer finish before I respond, knowing the room is still counting bows.",
    "I offer the regent a single phrase of congratulation and watch who nods back.",
    "I catch Ceriel's eye across the chamber and hold it a beat too long.",
    "I wait at the threshold of the antechamber until one of them breaks the silence.",
    "I sign the introduction letter myself rather than use the embassy seal.",
    "\"Tell me the price, Ceriel,\" I say in the corridor, \"before the price tells itself.\"",
    "I walk past Prince Oren's faction and bow only to the neutral envoy.",
    "I retire to the antechamber and ask my attendant which petitioners stayed longest today.",
    "When Ceriel proposes a private meeting, I accept — but name the hour myself.",
)

# -----------------------------------------------------------------------------

HEIST_SEED: dict = {
    "entities": [
        {"id": "hub", "entity_type": "location", "name": "Dock 7, Spire Harbor"},
        {"id": "player", "entity_type": "character", "name": "Quill",
         "data": {"voice": {
             "vocabulary_level": "slang-technical",
             "sentence_length_bias": "punchy",
             "directness": 0.7,
             "emotional_expression": "wry",
         }}},
        {"id": "rigger", "entity_type": "character", "name": "Sata",
         "data": {"voice": {"vocabulary_level": "jargon-heavy",
                            "directness": 0.9}}},
        {"id": "face", "entity_type": "character", "name": "Nevis",
         "data": {"voice": {"vocabulary_level": "smooth",
                            "directness": 0.3,
                            "emotional_expression": "charming"}}},
        {"id": "muscle", "entity_type": "character", "name": "Bren",
         "data": {"voice": {"vocabulary_level": "sparse",
                            "directness": 1.0,
                            "emotional_expression": "deadpan"}}},
    ],
    "plot_threads": [
        {"id": "pt:main", "name": "The Empyrean Array",
         "description": "A stellar-relay crystal worth a city's breathing air is being auctioned in the arcology — the crew has 18 hours to swap the real thing for Sata's replica.",
         "arc_position": "rising", "priority": 9},
        {"id": "pt:b", "name": "Nevis's Past",
         "description": "Nevis is working with someone the crew hasn't been told about.",
         "arc_position": "rising", "priority": 5},
    ],
    "themes": [
        {"id": "t:trust", "proposition": "trust is a tool, and tools wear out",
         "stance": "exploring"},
        {"id": "t:precision", "proposition": "the job is won in the second nobody's looking",
         "stance": "exploring"},
    ],
    "narrator": {
        "pov_character_id": "player",
        "pov_type": "third_limited",
        "worldview": "dry, amused, notices technical details and interpersonal currents",
        "editorial_stance": "fond of the crew, skeptical of the plan",
        "sensory_bias": {"visual": 0.3, "auditory": 0.15, "tactile": 0.15,
                         "kinesthetic": 0.2, "interoceptive": 0.1,
                         "olfactory": 0.1},
        "attention_bias": ["wiring", "tells", "who laughs first",
                           "who's carrying what", "the hum under the floor"],
        "voice_samples": [
            "Sata's toolbelt clicked when she walked — a six-note chord Quill had learned to read the way other people read a face.",
            "Nevis smiled the way a lock smiles: nothing in it, but a clear way in if you knew the combination.",
            "Bren was late. Bren was always late. The only surprising thing was that the rest of them had stopped flinching about it.",
            "The vault's interior was a cathedral of teeth — interlocking brass plates in four rings that turned against each other like the gears of an impossible clock, each ring carrying the signature of a different lockmaker, each one dead for over a century.",
        ],
    },
}

HEIST_ACTIONS = (
    "I walk Dock 7, counting Arcology drones overhead and noting the two that loiter.",
    "\"Sata,\" I say, \"tell me the replica's mass is within a gram of the real thing.\"",
    "I wait for Nevis to finish his pitch to the auctioneer before I nod to Bren.",
    "I ask Bren how many exits the service corridor has and whether any of them lie about it.",
    "I watch the crowd for the tell that means the mark has picked our replica up.",
    "I slip the cut-timer into my palm and let my sleeve fall back over my wrist.",
    "I let the hum under the deckplates tell me where the power trunk actually runs.",
    "I ask Nevis who else he talked to this morning, and I want the real answer.",
    "I lean on the railing where the camera has a six-second blind spot.",
    "I order one drink and drink half of it so the bartender remembers me as staying.",
    "When the arcology alarm flashes, I step left into the service corridor instead of running.",
    "I meet the crew at the rally point and lay the stone on the table before anyone speaks.",
)


SEEDS: dict[str, dict[str, Any]] = {
    "noir": {"seed": NOIR_SEED, "actions": NOIR_ACTIONS, "quest_id": "noir_v2",
             "genre": "low-fantasy noir"},
    "intrigue": {"seed": INTRIGUE_SEED, "actions": INTRIGUE_ACTIONS,
                 "quest_id": "intrigue_v2", "genre": "political intrigue"},
    "heist": {"seed": HEIST_SEED, "actions": HEIST_ACTIONS,
              "quest_id": "heist_v2", "genre": "science-fantasy heist"},
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def _bootstrap(*, quest_id: str, seed: dict, narrator_cfg: dict,
                     genre: str, workdir: Path, lora: str, llm_url: str,
                     n_candidates: int, sft_root: Path):
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
    seed_path.write_text(json.dumps(seed))

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
        sm.add_theme(quest_id, th)
    sm.apply_delta(payload.delta, update_number=0)

    craft_library = CraftLibrary(ROOT / "app" / "craft" / "data")
    structure = craft_library.structure("three_act")
    sm.upsert_arc(QuestArcState(
        quest_id=quest_id, arc_id="main",
        structure_id="three_act", scale="campaign",
        current_phase_index=0, phase_progress=0.0,
        tension_observed=[], last_directive=None,
    ))
    sm.upsert_reader_state(ReaderState(quest_id=quest_id))

    client = InferenceClient(
        base_url=llm_url, timeout=300.0, retries=1, model=lora,
    )
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(sm, renderer, TokenBudget())

    dramatic = DramaticPlanner(client, renderer, craft_library)
    emotional = EmotionalPlanner(client, renderer)
    craft = CraftPlanner(client, renderer, craft_library)
    arc = ArcPlanner(client, renderer)

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
    quest_retriever = QuestRetriever(sm, quest_id, embedder=embedder)
    motif_retriever = MotifRetriever(sm, quest_id)
    foreshadow_retriever = ForeshadowingRetriever(sm, quest_id)
    voice_retriever = VoiceRetriever(sm, quest_id)

    scorer = Scorer(llm_judge_client=client)

    pipeline = Pipeline(
        sm, cb, client,
        arc_planner=arc,
        dramatic_planner=dramatic,
        emotional_planner=emotional,
        craft_planner=craft,
        craft_library=craft_library,
        structure=structure,
        quest_config={
            "narrator": narrator_cfg,
            "genre": genre,
            "retrieval": {"enabled": True},
            "sft_collection": {
                "enabled": True,
                "dir": str(sft_root / quest_id),
            },
            "scoring": {"enabled": True},
            "n_candidates": n_candidates,
        },
        quest_id=quest_id,
        arc_id="main",
        passage_retriever=passage_retriever,
        quest_retriever=quest_retriever,
        voice_retriever=voice_retriever,
        motif_retriever=motif_retriever,
        foreshadowing_retriever=foreshadow_retriever,
        scene_retriever=scene_retriever,
        retrieval_embedder=embedder,
        scorer=scorer,
        llm_judge_client=client,
    )
    return pipeline, sm, client


async def run_seed(seed_key: str, args: argparse.Namespace) -> dict:
    cfg = SEEDS[seed_key]
    quest_id = cfg["quest_id"]
    sft_root = Path(args.sft_root)
    workdir = Path(args.workdir) / seed_key
    log_path = Path(args.log_root) / f"{quest_id}_run_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    pipeline, sm, client = await _bootstrap(
        quest_id=quest_id,
        seed=cfg["seed"],
        narrator_cfg=cfg["seed"]["narrator"],
        genre=cfg["genre"],
        workdir=workdir,
        lora=args.lora,
        llm_url=args.llm_url,
        n_candidates=args.n,
        sft_root=sft_root,
    )

    actions = cfg["actions"]
    n_updates = min(args.updates, len(actions)) if args.updates > 0 else len(actions)

    run_meta = {
        "_meta": True,
        "seed_key": seed_key,
        "quest_id": quest_id,
        "start_ts": time.time(),
        "model": args.lora,
        "n_candidates": args.n,
        "updates_target": n_updates,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_meta) + "\n")

    committed = 0
    failures = 0

    for update_number in range(1, n_updates + 1):
        action = actions[update_number - 1]
        print(f"\n[{seed_key}][{update_number}/{n_updates}] {action!r}")
        t0 = time.perf_counter()
        row: dict[str, Any] = {
            "seed_key": seed_key,
            "quest_id": quest_id,
            "update_number": update_number,
            "action": action,
        }
        try:
            out = await pipeline.run(
                player_action=action, update_number=update_number,
            )
            latency = time.perf_counter() - t0
            trace = out.trace
            dim_scores = {}
            for s in trace.stages:
                if getattr(s, "stage_name", "") == "scoring":
                    detail = getattr(s, "detail", None) or {}
                    dims = detail.get("dimensions") or {}
                    dim_scores = {k: float(v) for k, v in dims.items()}
                    if "overall_score" in detail:
                        dim_scores["overall_score"] = float(detail["overall_score"])
                    break
            n_sft = sum(
                1 for s in trace.stages
                if getattr(s, "stage_name", "") == "sft_collection"
            )
            row.update({
                "wall_clock_seconds": round(latency, 3),
                "outcome": trace.outcome,
                "n_stages": len(trace.stages),
                "prose_chars": len(out.prose or ""),
                "dimension_scores": dim_scores,
                "sft_records_written": n_sft,
                "error": None,
            })
            if trace.outcome == "committed":
                committed += 1
            print(
                f"    -> outcome={trace.outcome} t={latency:.1f}s "
                f"overall={dim_scores.get('overall_score', float('nan')):.3f} "
                f"sft_records={n_sft}"
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            failures += 1
            row.update({
                "wall_clock_seconds": round(latency, 3),
                "outcome": "fallback",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=6),
            })
            print(f"    !! ERROR: {type(e).__name__}: {e}")

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    return {
        "seed_key": seed_key,
        "quest_id": quest_id,
        "updates_attempted": n_updates,
        "committed": committed,
        "failures": failures,
        "log_path": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="collect_v2_corpus")
    ap.add_argument("--seed", type=str, default="all",
                    choices=list(SEEDS.keys()) + ["all"])
    ap.add_argument("--updates", type=int, default=12,
                    help="Updates per seed (<=0 means all actions).")
    ap.add_argument("--n", type=int, default=8,
                    help="Candidates per scene.")
    ap.add_argument("--lora", type=str,
                    default=os.environ.get("LLM_MODEL", "writer_v1"))
    ap.add_argument("--llm-url", type=str,
                    default=os.environ.get("LLM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--sft-root", type=Path,
                    default=ROOT / "data" / "sft",
                    help="Root of SFT collection output.")
    ap.add_argument("--workdir", type=Path,
                    default=Path("/tmp/collect_v2_corpus"))
    ap.add_argument("--log-root", type=Path,
                    default=ROOT / "data" / "stress_v2")
    return ap.parse_args()


async def main_async() -> None:
    args = parse_args()
    keys = list(SEEDS.keys()) if args.seed == "all" else [args.seed]
    summary: list[dict] = []
    for key in keys:
        try:
            result = await run_seed(key, args)
            summary.append(result)
        except Exception as e:
            print(f"[{key}] HARD FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append({
                "seed_key": key, "quest_id": SEEDS[key]["quest_id"],
                "error": f"{type(e).__name__}: {e}",
            })

    out = Path(args.log_root) / "collect_v2_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {out}")
    for row in summary:
        print("  ", row)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
