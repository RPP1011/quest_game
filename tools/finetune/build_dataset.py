"""Merge Claude labeler outputs into a training dataset for LoRA finetuning.

Produces `/tmp/labels_claude_all.json` and `data/calibration/{train,test}.jsonl`.

Dimensions in `DIM_RUBRICS` are the passage-scale dims only. Arc-scale dims
(tension_execution, choice_hook_quality, update_self_containment,
choice_meaningfulness, world_state_legibility) are trained via a separate
pipeline — see feat/arc-scale-scoring.
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

LABEL_GLOB = "/tmp/labels_claude_part_*.json"
PASSAGES_DIR = Path("data/calibration/passages")
OUT_DIR = Path("data/calibration")
COMBINED = Path("/tmp/labels_claude_all.json")


DIM_RUBRICS = {
    "free_indirect_quality": "Narrator adopts character's idiom/perception; external & internal voices merging.",
    "interiority_depth": "How deeply we access character thought/feeling vs. surface behavior.",
    "detail_characterization": "Details reveal the perceiving consciousness (not just world-building).",
    "sensory_density": "Density of specific sensory perception (sight/sound/touch/smell/taste/kinesthetic).",
    "voice_distinctiveness": "Distinct character/narrator register recognizable in prose.",
    "thematic_presence": "Thematic concerns embedded in the prose (not announced).",
    "subtext_presence": "Important things working through what ISN'T said.",
    "clarity": "A careful reader can follow sentence-by-sentence without rereading.",
}


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
        for p in data["passages"]:
            if "scores" in p and "dimensions" not in p:
                p["dimensions"] = p.pop("scores")
            p.setdefault("work_id", stem)
            p.setdefault("is_quest", False)
            merged.append(p)
    return merged


def build_row(passage: dict, text: str) -> dict:
    """Batched row: all 8 dims scored in one assistant response.

    Matches production format — the judge is called once per passage and
    returns all dimensions at once. Forces the model to attend to each dim
    name and differentiate scores within a single generation.
    """
    rubric_lines = "\n".join(f"- **{d}**: {r}" for d, r in DIM_RUBRICS.items())
    system = (
        "You are a literary-critic scorer. Score the passage on every listed "
        "dimension independently on a 0.0–1.0 absolute scale; high quality on "
        "one dim does not inflate another. For each dim, return a score and a "
        "one-sentence rationale citing concrete signals from the passage."
    )
    user = (
        f"Dimensions:\n{rubric_lines}\n\n"
        f"Passage:\n---\n{text.strip()}\n---\n\n"
        "Respond with a single JSON object keyed by dimension, each value "
        "`{\"score\": float, \"rationale\": str}`."
    )
    dims = passage["dimensions"]
    rationale = passage.get("rationale", "")
    out: dict[str, dict] = {}
    for d in DIM_RUBRICS:
        if d not in dims:
            continue
        out[d] = {"score": round(float(dims[d]), 2), "rationale": rationale}
    if len(out) != len(DIM_RUBRICS):
        return {}
    assistant = json.dumps(out)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "work_id": passage["work_id"],
            "passage_id": passage["passage_id"],
            "dims": {d: float(dims[d]) for d in DIM_RUBRICS if d in dims},
        },
    }


def main():
    merged = merge_labels()
    COMBINED.write_text(json.dumps({"passages": merged}, indent=2))
    print(f"merged {len(merged)} passage labels -> {COMBINED}")

    rows = []
    for p in merged:
        try:
            text = load_passage(p["work_id"], p["passage_id"])
        except FileNotFoundError:
            continue
        row = build_row(p, text)
        if row:
            rows.append(row)
    print(f"built {len(rows)} batched rows (one per passage, all dims per row)")

    by_passage: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_passage.setdefault((r["meta"]["work_id"], r["meta"]["passage_id"]), []).append(r)
    by_work: dict[str, list[tuple[str, str]]] = {}
    for k in by_passage:
        by_work.setdefault(k[0], []).append(k)

    train_keys: set[tuple[str, str]] = set()
    test_keys: set[tuple[str, str]] = set()
    for work_id, keys in sorted(by_work.items()):
        rng = random.Random(f"split:{work_id}")
        keys_sorted = sorted(keys)
        rng.shuffle(keys_sorted)
        test_keys.update(keys_sorted[:2])
        train_keys.update(keys_sorted[2:])

    train_rows = [r for k, rs in by_passage.items() if k in train_keys for r in rs]
    test_rows = [r for k, rs in by_passage.items() if k in test_keys for r in rs]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    (OUT_DIR / "test.jsonl").write_text("\n".join(json.dumps(r) for r in test_rows) + "\n")
    print(f"train: {len(train_rows)} rows across {len(train_keys)} passages")
    print(f"test:  {len(test_rows)} rows across {len(test_keys)} passages")


if __name__ == "__main__":
    main()
