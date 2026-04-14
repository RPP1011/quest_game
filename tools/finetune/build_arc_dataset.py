"""Build arc-scale training dataset from scene labels.

Parallel to build_pairwise.py but for arc dimensions (scored on 2000-4000
word scenes, not 500-900 word passages). Emits pairwise pairs for each
arc dim, holding out 1 scene per work as test.
"""
from __future__ import annotations

import glob
import itertools
import json
import random
from pathlib import Path

LABEL_GLOB = "/tmp/labels_claude_arc_*.json"
SCENES_DIR = Path("data/calibration/scenes")
OUT_DIR = Path("data/calibration")

# Dims labeled in the arc pipeline (matches labeler prompts).
ARC_DIMS_COMMON = (
    "tension_execution",
    "thematic_development",
    "scene_coherence",
    "emotional_arc",
    "consequence_weight",
)
ARC_DIMS_QUEST = (
    "choice_hook_quality",
    "update_self_containment",
    "world_state_legibility",
)

DIM_DEFS = {
    "tension_execution": "felt arc of stakes across the scene (build/climax/release)",
    "thematic_development": "how themes resolve/shift across the scene",
    "scene_coherence": "scene functions as a unit with beginning/middle/end",
    "emotional_arc": "emotional trajectory has shape, not flat",
    "consequence_weight": "what happens matters; it changes something",
    "choice_hook_quality": "scene ends creating decision tension",
    "update_self_containment": "scene functions as a readable unit",
    "world_state_legibility": "reader can describe current situation/options",
}

MIN_DELTA = 0.2
MAX_PAIRS_PER_DIM = 400


def load_scene(work_id: str, passage_id: str) -> str:
    path = SCENES_DIR / work_id / f"{passage_id}.txt"
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, _, rest = text[3:].partition("---")
        text = rest.lstrip()
    return text


def merge_labels() -> list[dict]:
    merged: list[dict] = []
    for f in sorted(glob.glob(LABEL_GLOB)):
        data = json.loads(Path(f).read_text())
        passages = data if isinstance(data, list) else data.get("passages", [])
        for p in passages:
            if "scores" in p and "dimensions" not in p:
                p["dimensions"] = p.pop("scores")
            p.setdefault("is_quest", False)
            merged.append(p)
    return merged


def build_pair_row(
    dim: str, pa: dict, text_a: str, pb: dict, text_b: str, winner: str,
) -> dict:
    system = (
        "You are a literary scorer. Given two scenes, decide which has more "
        "of the named dimension. Respond with a single token: `A` or `B`."
    )
    user = (
        f"Dimension: **{dim}** — {DIM_DEFS.get(dim, '')}\n\n"
        f"A:\n---\n{text_a.strip()}\n---\n\n"
        f"B:\n---\n{text_b.strip()}\n---\n\n"
        "Which scene scores higher on this dimension? Respond with only `A` or `B`."
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": winner},
        ],
        "meta": {
            "dim": dim,
            "scope": "scene",
            "A": f"{pa['work_id']}/{pa['passage_id']}",
            "B": f"{pb['work_id']}/{pb['passage_id']}",
            "A_score": pa["dimensions"][dim],
            "B_score": pb["dimensions"][dim],
            "delta": abs(pa["dimensions"][dim] - pb["dimensions"][dim]),
        },
    }


def main():
    merged = merge_labels()
    print(f"raw scene labels: {len(merged)}")

    texts: dict[str, str] = {}
    valid = []
    for p in merged:
        try:
            texts[f"{p['work_id']}/{p['passage_id']}"] = load_scene(p["work_id"], p["passage_id"])
            valid.append(p)
        except FileNotFoundError:
            continue
    merged = valid
    print(f"scenes with text on disk: {len(merged)}")

    # Test-scene holdout: 1 per work (we have only 5 scenes per work)
    by_work: dict[str, list[dict]] = {}
    for p in merged:
        by_work.setdefault(p["work_id"], []).append(p)
    test_keys: set[str] = set()
    for work_id, passages in by_work.items():
        rng = random.Random(f"arc-split:{work_id}")
        passages_sorted = sorted(passages, key=lambda p: p["passage_id"])
        rng.shuffle(passages_sorted)
        if passages_sorted:
            test_keys.add(f"{work_id}/{passages_sorted[0]['passage_id']}")

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    dims_all = list(ARC_DIMS_COMMON) + list(ARC_DIMS_QUEST)
    for dim in dims_all:
        candidates = [p for p in merged if dim in p.get("dimensions", {})]
        pairs_all: list[dict] = []
        for pa, pb in itertools.combinations(candidates, 2):
            sa, sb = pa["dimensions"][dim], pb["dimensions"][dim]
            if abs(sa - sb) < MIN_DELTA:
                continue
            pair_key = (pa["work_id"], pa["passage_id"], pb["work_id"], pb["passage_id"])
            order_rng = random.Random(f"order:{dim}:{pair_key}")
            first, second = (pa, pb) if order_rng.random() < 0.5 else (pb, pa)
            sfirst = first["dimensions"][dim]
            ssecond = second["dimensions"][dim]
            winner = "A" if sfirst > ssecond else "B"
            row = build_pair_row(
                dim, first, texts[f"{first['work_id']}/{first['passage_id']}"],
                second, texts[f"{second['work_id']}/{second['passage_id']}"],
                winner,
            )
            pairs_all.append(row)

        dim_train, dim_test = [], []
        for r in pairs_all:
            touches = r["meta"]["A"] in test_keys or r["meta"]["B"] in test_keys
            (dim_test if touches else dim_train).append(r)
        rng = random.Random(f"cap:{dim}")
        if len(dim_train) > MAX_PAIRS_PER_DIM:
            rng.shuffle(dim_train)
            dim_train = dim_train[:MAX_PAIRS_PER_DIM]
        train_rows.extend(dim_train)
        test_rows.extend(dim_test)
        print(f"{dim:<28} train={len(dim_train):>4}  test={len(dim_test):>4}")

    random.Random(1).shuffle(train_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "arc_pairs_train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    (OUT_DIR / "arc_pairs_test.jsonl").write_text("\n".join(json.dumps(r) for r in test_rows) + "\n")
    print(f"\ntotal train: {len(train_rows)}  test: {len(test_rows)}")


if __name__ == "__main__":
    main()
