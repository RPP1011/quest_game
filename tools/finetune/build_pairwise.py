"""Synthesize pairwise preference dataset from Claude labels.

For each dimension, generate pairs (A, B) from the labeled passage set where
|score_A - score_B| >= MIN_DELTA. Train target is "which passage is higher on
this dim" — a binary preference, matching G2's rerank use case directly.

Output: `data/calibration/pairs_train.jsonl`, `pairs_test.jsonl`.
"""
from __future__ import annotations

import glob
import itertools
import json
import random
from pathlib import Path

LABEL_GLOB = "/tmp/labels_claude_part_*.json"
PASSAGES_DIR = Path("data/calibration/passages")
OUT_DIR = Path("data/calibration")

DIMS = (
    "free_indirect_quality", "interiority_depth", "detail_characterization",
    "sensory_density", "voice_distinctiveness", "thematic_presence",
    "subtext_presence", "clarity",
)

DIM_DEFS = {
    "free_indirect_quality": "narrator adopts character's idiom/perception",
    "interiority_depth": "depth of access to character thought/feeling",
    "detail_characterization": "details reveal the perceiving consciousness",
    "sensory_density": "density of specific sensory perception",
    "voice_distinctiveness": "distinct character/narrator register",
    "thematic_presence": "themes embedded in prose (not announced)",
    "subtext_presence": "important things through what ISN'T said",
    "clarity": "a careful reader can follow sentence-by-sentence",
}

MIN_DELTA = 0.2  # pairs closer than this are ambiguous, skip
MAX_PAIRS_PER_DIM = 600  # cap to avoid one dim dominating


def load_passage(work_id: str, passage_id: str) -> str:
    path = PASSAGES_DIR / work_id / f"{passage_id}.txt"
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, _, rest = text[3:].partition("---")
        text = rest.lstrip()
    return text


def merge_labels() -> list[dict]:
    merged: list[dict] = []
    for f in sorted(glob.glob(LABEL_GLOB)):
        data = json.loads(Path(f).read_text())
        stem = Path(f).stem.replace("labels_claude_part_", "")
        passages = data if isinstance(data, list) else data.get("passages", [])
        for p in passages:
            if "scores" in p and "dimensions" not in p:
                p["dimensions"] = p.pop("scores")
            p.setdefault("work_id", stem)
            merged.append(p)
    return merged


def _content_tokens(text: str) -> set[str]:
    return {w.lower() for w in text.split()}


def dedup(merged: list[dict], threshold: float = 0.5) -> list[dict]:
    by_work: dict[str, list[dict]] = {}
    for p in merged:
        by_work.setdefault(p["work_id"], []).append(p)
    kept = []
    for work_id, passages in by_work.items():
        passages = sorted(passages, key=lambda p: p["passage_id"])
        cache = {}
        for p in passages:
            try:
                cache[p["passage_id"]] = _content_tokens(load_passage(work_id, p["passage_id"]))
            except FileNotFoundError:
                pass
        work_kept = []
        for p in passages:
            if p["passage_id"] not in cache:
                continue
            mine = cache[p["passage_id"]]
            if any(
                len(mine & cache[k["passage_id"]]) / max(1, len(mine | cache[k["passage_id"]])) > threshold
                for k in work_kept
            ):
                continue
            work_kept.append(p)
        kept.extend(work_kept)
    return kept


def build_pair_row(
    dim: str, pa: dict, text_a: str, pb: dict, text_b: str, winner: str,
) -> dict:
    system = (
        "You are a literary scorer. Given two passages, decide which has more "
        "of the named dimension. Respond with a single token: `A` or `B`."
    )
    user = (
        f"Dimension: **{dim}** — {DIM_DEFS[dim]}\n\n"
        f"A:\n---\n{text_a.strip()}\n---\n\n"
        f"B:\n---\n{text_b.strip()}\n---\n\n"
        "Which passage scores higher on this dimension? Respond with only `A` or `B`."
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": winner},
        ],
        "meta": {
            "dim": dim,
            "A": f"{pa['work_id']}/{pa['passage_id']}",
            "B": f"{pb['work_id']}/{pb['passage_id']}",
            "A_score": pa["dimensions"][dim],
            "B_score": pb["dimensions"][dim],
            "delta": abs(pa["dimensions"][dim] - pb["dimensions"][dim]),
        },
    }


def main():
    merged = merge_labels()
    print(f"raw passages: {len(merged)}")
    merged = dedup(merged)
    print(f"after dedup: {len(merged)}")

    # Attach passage text once
    texts: dict[str, str] = {}
    valid = []
    for p in merged:
        try:
            texts[f"{p['work_id']}/{p['passage_id']}"] = load_passage(p["work_id"], p["passage_id"])
            valid.append(p)
        except FileNotFoundError:
            continue
    merged = valid

    # Test-passage holdout: 2 per work, seeded by work_id
    by_work: dict[str, list[dict]] = {}
    for p in merged:
        by_work.setdefault(p["work_id"], []).append(p)
    test_passage_keys: set[str] = set()
    for work_id, passages in by_work.items():
        rng = random.Random(f"split:{work_id}")
        passages_sorted = sorted(passages, key=lambda p: p["passage_id"])
        rng.shuffle(passages_sorted)
        for p in passages_sorted[:2]:
            test_passage_keys.add(f"{work_id}/{p['passage_id']}")

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for dim in DIMS:
        pairs_all: list[dict] = []
        for pa, pb in itertools.combinations(merged, 2):
            if dim not in pa["dimensions"] or dim not in pb["dimensions"]:
                continue
            sa, sb = pa["dimensions"][dim], pb["dimensions"][dim]
            delta = abs(sa - sb)
            if delta < MIN_DELTA:
                continue
            # Randomize presentation order per pair to avoid positional bias
            pair_key = (pa["work_id"], pa["passage_id"], pb["work_id"], pb["passage_id"])
            order_rng = random.Random(f"order:{dim}:{pair_key}")
            if order_rng.random() < 0.5:
                first, second = pa, pb
            else:
                first, second = pb, pa
            sfirst = first["dimensions"][dim]
            ssecond = second["dimensions"][dim]
            winner = "A" if sfirst > ssecond else "B"
            row = build_pair_row(
                dim, first, texts[f"{first['work_id']}/{first['passage_id']}"],
                second, texts[f"{second['work_id']}/{second['passage_id']}"],
                winner,
            )
            pairs_all.append(row)

        # Split: any pair that touches a test passage goes to test
        dim_train, dim_test = [], []
        for r in pairs_all:
            touches_test = r["meta"]["A"] in test_passage_keys or r["meta"]["B"] in test_passage_keys
            (dim_test if touches_test else dim_train).append(r)
        rng = random.Random(f"cap:{dim}")
        if len(dim_train) > MAX_PAIRS_PER_DIM:
            rng.shuffle(dim_train)
            dim_train = dim_train[:MAX_PAIRS_PER_DIM]
        train_rows.extend(dim_train)
        test_rows.extend(dim_test)
        print(f"{dim:<28} train={len(dim_train):>4}  test={len(dim_test):>4}")

    random.Random(1).shuffle(train_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pairs_train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    (OUT_DIR / "pairs_test.jsonl").write_text("\n".join(json.dumps(r) for r in test_rows) + "\n")
    print(f"\ntotal train pairs: {len(train_rows)}  test pairs: {len(test_rows)}")


if __name__ == "__main__":
    main()
