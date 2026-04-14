"""One-shot story generation demo.

Wires the full hierarchical pipeline with a local llama-server at port 8081,
a minimal seed, and runs N player actions. Prints the prose for each.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

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


async def main():
    from app.cli.play import _open_world
    from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget
    from app.runtime.client import InferenceClient
    from app.world import SeedLoader
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    from app.world.schema import QuestArcState, ReaderState

    tmp = Path("/tmp/storygen_demo")
    tmp.mkdir(parents=True, exist_ok=True)
    db_path = tmp / "quest.db"
    if db_path.exists():
        db_path.unlink()
    seed_path = tmp / "seed.json"
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

    # Bootstrap arc + reader state
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

    client = InferenceClient(base_url="http://127.0.0.1:8081", retries=1)
    renderer = PromptRenderer(PROMPTS)
    cb = ContextBuilder(sm, renderer, TokenBudget())

    # Wire hierarchical planners
    from app.planning import DramaticPlanner, EmotionalPlanner, CraftPlanner
    from app.planning.arc_planner import ArcPlanner
    dramatic = DramaticPlanner(client, renderer, craft_library)
    emotional = EmotionalPlanner(client, renderer)
    craft = CraftPlanner(client, renderer, craft_library)
    arc = ArcPlanner(client, renderer)

    # Retrieval layer (Waves 1-4): wire up all retrievers + flip the flag.
    import os
    retrieval_on = os.environ.get("RETRIEVAL", "1") != "0"
    passage_retriever = quest_retriever = scene_retriever = None
    motif_retriever = foreshadow_retriever = voice_retriever = None
    embedder = None
    if retrieval_on:
        from app.retrieval import (
            Embedder, PassageRetriever, QuestRetriever, SceneShapeRetriever,
            MotifRetriever, ForeshadowingRetriever, VoiceRetriever,
        )
        manifest = Path(__file__).parent.parent / "data" / "calibration" / "manifest.yaml"
        passages = Path(__file__).parent.parent / "data" / "calibration" / "passages"
        scenes_manifest = Path(__file__).parent.parent / "data" / "calibration" / "scenes_manifest.yaml"
        scenes = Path(__file__).parent.parent / "data" / "calibration" / "scenes"
        if manifest.is_file():
            passage_retriever = PassageRetriever(manifest, Path("/tmp"), passages, enable_semantic=False)
        if scenes_manifest.is_file():
            try:
                scene_retriever = SceneShapeRetriever(scenes_manifest, "/tmp/labels_claude_arc_*.json", scenes)
            except Exception as e:
                print(f"[scene retriever skipped: {e}]")
        embedder = Embedder()
        quest_retriever = QuestRetriever(sm, "demo", embedder=embedder)
        motif_retriever = MotifRetriever(sm, "demo")
        foreshadow_retriever = ForeshadowingRetriever(sm, "demo")
        voice_retriever = VoiceRetriever(sm, "demo")

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
        },
        quest_id="demo",
        arc_id="main",
        passage_retriever=passage_retriever,
        quest_retriever=quest_retriever,
        voice_retriever=voice_retriever,
        retrieval_embedder=embedder,
    )

    update_number = 1
    for action in ACTIONS:
        print(f"\n{'='*60}\n[ACTION {update_number}] {action}\n{'='*60}")
        try:
            out = await pipeline.run(player_action=action, update_number=update_number)
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback; traceback.print_exc()
            break
        print(f"\n--- PROSE ---\n{out.prose}\n")
        print(f"--- CHOICES ---")
        for i, c in enumerate(out.choices, 1):
            print(f"  {i}. {c}")
        print(f"\n[outcome: {out.trace.outcome}  stages: {len(out.trace.stages)}]")
        update_number += 1


if __name__ == "__main__":
    asyncio.run(main())
