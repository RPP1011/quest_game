"""Turn ``*.picked.json`` sidecars into ``train.jsonl`` / ``test.jsonl`` for
the writer LoRA.

Each emitted row matches the shape consumed by
``tools/finetune/train_lora.py``::

    {
      "messages": [
        {"role": "system", "content": "<writer system prompt>"},
        {"role": "user", "content": "<craft brief + planning context>"},
        {"role": "assistant", "content": "<chosen candidate prose>"},
      ],
      "meta": {
        "quest_id": "...",
        "update": 5,
        "scene": 1,
        "scorer_overall": 0.73,
        "chosen_index": 2,
        "source": "path/to/scene.picked.json",
      },
    }

Holdout: a seeded ~10% test split drawn at the row level, so runs of this
script produce the same split each time (stable for pipeline A/B).

Usage::

    uv run python -m tools.sft.build_train
    uv run python -m tools.sft.build_train --root data/sft --test-ratio 0.1 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SYSTEM_PROMPT = (
    "You are the prose writer for an interactive-fiction quest engine. "
    "Given the craft brief for one scene, write the scene itself in the "
    "voice and register the brief specifies. Do not add headings, editor "
    "notes, or commentary — only the scene prose."
)


def _iter_picked(root: Path, quest_id: str | None = None) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return
    dirs: list[Path]
    if quest_id:
        d = root / quest_id
        dirs = [d] if d.exists() else []
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        # Phase-2: the SFT collector now writes under
        # ``<sft_root>/<quest_id>/<quest_id>/`` because ``sft_collection.dir``
        # already includes the quest_id. Walk one level deeper to pick up
        # the nested layout while still supporting the legacy flat layout.
        for f in sorted(d.rglob("*.picked.json")):
            yield f


@dataclass
class _Row:
    messages: list[dict]
    meta: dict


def build_row(record: dict, *, source: Path | None = None) -> _Row:
    """Build one training row from a ``*.picked.json`` record.

    Raises :class:`ValueError` when the record is malformed (no candidates,
    no ``claude_pick``, or the chosen index is out of range).
    """
    candidates = record.get("candidates") or []
    if not candidates:
        raise ValueError("record has no candidates")
    pick = record.get("claude_pick")
    if not pick or "chosen_index" not in pick:
        raise ValueError("record has no claude_pick.chosen_index")
    chosen_index = int(pick["chosen_index"])
    chosen = next(
        (c for c in candidates if int(c.get("index", -1)) == chosen_index),
        None,
    )
    if chosen is None:
        raise ValueError(
            f"chosen_index {chosen_index} not present in candidates"
        )
    prose = chosen.get("prose") or ""
    if not prose.strip():
        raise ValueError("chosen candidate has empty prose")

    brief = record.get("craft_brief") or ""
    user_content = (
        f"Craft brief:\n{brief}\n\n"
        f"Write the scene prose now."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": prose},
    ]
    meta = {
        "quest_id": record.get("quest_id"),
        "update": record.get("update_number"),
        "scene": record.get("scene_index"),
        "scorer_overall": chosen.get("overall_score"),
        "chosen_index": chosen_index,
        "pipeline_trace_id": record.get("pipeline_trace_id"),
        "source": str(source) if source else None,
    }
    return _Row(messages=messages, meta=meta)


def split_rows(
    rows: list[_Row],
    *,
    test_ratio: float = 0.1,
    seed: int = 7,
) -> tuple[list[_Row], list[_Row]]:
    """Seeded test-split. Stable: the same ``rows`` + ``seed`` emit the
    same ``(train, test)`` partition.
    """
    if not rows:
        return [], []
    if test_ratio <= 0.0:
        return list(rows), []
    if test_ratio >= 1.0:
        return [], list(rows)
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n_test = max(1, int(round(len(rows) * test_ratio)))
    test_idxs = set(idxs[:n_test])
    train: list[_Row] = []
    test: list[_Row] = []
    for i, r in enumerate(rows):
        (test if i in test_idxs else train).append(r)
    return train, test


def collect_rows(
    root: Path,
    *,
    quest_id: str | None = None,
    strict: bool = False,
) -> tuple[list[_Row], list[Path]]:
    """Walk ``root``, build rows, return ``(rows, skipped_paths)``.

    ``strict=False`` (default): malformed picked-records are skipped and
    reported in ``skipped_paths``. ``strict=True`` re-raises.
    """
    rows: list[_Row] = []
    skipped: list[Path] = []
    for f in _iter_picked(root, quest_id=quest_id):
        try:
            record = json.loads(f.read_text())
            row = build_row(record, source=f)
        except (ValueError, json.JSONDecodeError):
            if strict:
                raise
            skipped.append(f)
            continue
        rows.append(row)
    return rows, skipped


def _write_jsonl(rows: list[_Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps({"messages": r.messages, "meta": r.meta}))
            fp.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="build_train",
        description=(
            "Turn data/sft/**/*.picked.json files into train.jsonl + "
            "test.jsonl for the writer LoRA."
        ),
    )
    ap.add_argument("--root", type=Path, default=Path("data/sft"),
                    help="Root directory for SFT records (default: data/sft)")
    ap.add_argument("--quest", type=str, default=None,
                    help="Restrict to one quest_id subdir.")
    ap.add_argument("--out-train", type=Path, default=None,
                    help="Path for train.jsonl (default: <root>/train.jsonl).")
    ap.add_argument("--out-test", type=Path, default=None,
                    help="Path for test.jsonl (default: <root>/test.jsonl).")
    ap.add_argument("--test-ratio", type=float, default=0.1,
                    help="Fraction of rows held out for eval (default: 0.1).")
    ap.add_argument("--seed", type=int, default=7,
                    help="RNG seed for the test split (default: 7).")
    args = ap.parse_args(argv)

    rows, skipped = collect_rows(args.root, quest_id=args.quest)
    if skipped:
        print(
            f"warn: skipped {len(skipped)} malformed picked record(s):",
            file=sys.stderr,
        )
        for p in skipped[:5]:
            print(f"  {p}", file=sys.stderr)

    train, test = split_rows(
        rows, test_ratio=args.test_ratio, seed=args.seed,
    )
    out_train = args.out_train or (args.root / "train.jsonl")
    out_test = args.out_test or (args.root / "test.jsonl")
    _write_jsonl(train, out_train)
    _write_jsonl(test, out_test)

    print(
        f"Wrote {len(train)} train rows to {out_train} and "
        f"{len(test)} test rows to {out_test}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
