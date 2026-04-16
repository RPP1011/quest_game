"""Roll per-chapter annotations up to arc / character / relationship / hook level.

Reads every ``data/calibration/annotations/<work>/chap_NNNN.<label>.json``
for a given work+label, and emits a single ``rollup.<label>.json`` plus a
human-readable ``rollup.<label>.md`` that track:

- characters across chapters (first appearance, evolving characterization)
- relationships (first shown, how they develop)
- world facts (running glossary, accreting as chapters land)
- planted hooks and when they pay off (if tracked)
- per-chapter arc_movement timeline
- aggregate stats (words, scenes, beats per chapter)

Usage::

    uv run python tools/rollup_annotations.py --work pale_lights --label gemma4
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ANN_ROOT = Path("data/calibration/annotations")
RAW_ROOT = Path("data/calibration/raw")


def _load_chapters(work: str, label: str) -> list[dict]:
    root = ANN_ROOT / work
    files = sorted(root.glob(f"chap_*.{label}.json"))
    out = []
    for f in files:
        d = json.loads(f.read_text())
        out.append(d)
    return out


def _chapter_stats(work: str, chapter: int, annotation: dict) -> dict:
    raw = RAW_ROOT / work / f"chap_{chapter:04d}.txt"
    words = len(raw.read_text().split()) if raw.is_file() else 0
    scenes = annotation.get("scenes") or []
    beats = sum(len(s.get("beats") or []) for s in scenes)
    return {"words": words, "scenes": len(scenes), "beats": beats}


def _merge_characters(chapters: list[dict]) -> dict[str, dict]:
    """Track each character: first chapter, role, accumulated characterizations."""
    out: dict[str, dict] = {}
    for d in chapters:
        ch = d["chapter"]
        for c in (d["annotation"].get("characters_present") or []):
            name = (c.get("name") or "").strip()
            if not name:
                continue
            if name not in out:
                out[name] = {
                    "name": name,
                    "first_chapter": ch,
                    "role": c.get("role"),
                    "characterizations": [],
                    "chapters_present": [],
                }
            out[name]["chapters_present"].append(ch)
            note = c.get("characterization")
            excerpt = c.get("characterization_excerpt")
            if note:
                out[name]["characterizations"].append({
                    "chapter": ch, "note": note, "excerpt": excerpt,
                })
    # Sort entries
    for v in out.values():
        v["chapters_present"].sort()
    return dict(sorted(out.items(), key=lambda kv: kv[1]["first_chapter"]))


def _merge_relationships(chapters: list[dict]) -> list[dict]:
    by_pair: dict[tuple[str, str], dict] = {}
    for d in chapters:
        ch = d["chapter"]
        for r in (d["annotation"].get("relationships_shown") or []):
            a, b = (r.get("a") or "").strip(), (r.get("b") or "").strip()
            if not a or not b:
                continue
            key = tuple(sorted([a, b]))
            if key not in by_pair:
                by_pair[key] = {
                    "a": key[0], "b": key[1],
                    "first_chapter": ch,
                    "observations": [],
                }
            by_pair[key]["observations"].append({
                "chapter": ch, "nature": r.get("nature"),
                "evidence": r.get("evidence"),
            })
    return sorted(by_pair.values(), key=lambda x: x["first_chapter"])


def _merge_world(chapters: list[dict]) -> list[dict]:
    """Dedupe world facts case-insensitively, tracking first appearance."""
    seen: dict[str, dict] = {}
    for d in chapters:
        ch = d["chapter"]
        for fact in (d["annotation"].get("world_facts_established") or []):
            if not fact:
                continue
            key = re.sub(r"\s+", " ", fact.strip().lower())[:80]
            if key in seen:
                continue
            seen[key] = {"chapter": ch, "fact": fact}
    return sorted(seen.values(), key=lambda x: x["chapter"])


def _merge_hooks(chapters: list[dict]) -> dict:
    planted = []
    paid = []
    for d in chapters:
        ch = d["chapter"]
        for h in (d["annotation"].get("hooks_planted") or []):
            planted.append({"chapter": ch, "hook": h})
        for h in (d["annotation"].get("hooks_paid_off") or []):
            paid.append({"chapter": ch, "hook": h})
    return {"planted": planted, "paid_off": paid}


def _arc_timeline(chapters: list[dict]) -> list[dict]:
    return [{
        "chapter": d["chapter"],
        "arc_movement": d["annotation"].get("arc_movement") or "",
        "summary": d["annotation"].get("summary") or "",
    } for d in chapters]


def _write_markdown(work: str, label: str, rollup: dict, out: Path) -> None:
    lines = [f"# {work} — rollup ({label})\n"]
    stats = rollup["stats"]
    lines.append(f"- Chapters: {stats['n_chapters']}")
    lines.append(f"- Total words: {stats['total_words']:,}")
    lines.append(f"- Scenes / chapter: mean {stats['mean_scenes']:.1f}, "
                 f"beats / chapter: mean {stats['mean_beats']:.1f}\n")

    lines.append("## Arc timeline\n")
    for row in rollup["arc_timeline"]:
        lines.append(f"**Ch {row['chapter']}**: {row['summary']}")
        if row["arc_movement"]:
            lines.append(f"  _arc_: {row['arc_movement']}")
        lines.append("")

    lines.append("## Characters\n")
    for name, c in rollup["characters"].items():
        lines.append(f"### {name}  _(first seen ch {c['first_chapter']}, role: {c['role']})_")
        lines.append(f"- Present in chapters: {c['chapters_present']}")
        for note in c["characterizations"][:10]:
            lines.append(f"- **ch {note['chapter']}**: {note['note']}")
            if note.get("excerpt"):
                lines.append(f"    > {note['excerpt']}")
        lines.append("")

    lines.append("## Relationships\n")
    for r in rollup["relationships"]:
        lines.append(f"### {r['a']} ↔ {r['b']}  _(first ch {r['first_chapter']})_")
        for obs in r["observations"][:8]:
            lines.append(f"- **ch {obs['chapter']}**: {obs['nature']}")
            if obs.get("evidence"):
                lines.append(f"    > {obs['evidence']}")
        lines.append("")

    lines.append("## World facts (accreting)\n")
    for f in rollup["world"]:
        lines.append(f"- **ch {f['chapter']}**: {f['fact']}")
    lines.append("")

    lines.append("## Hooks planted\n")
    for h in rollup["hooks"]["planted"]:
        lines.append(f"- **ch {h['chapter']}**: {h['hook']}")
    if rollup["hooks"]["paid_off"]:
        lines.append("\n## Hooks paid off\n")
        for h in rollup["hooks"]["paid_off"]:
            lines.append(f"- **ch {h['chapter']}**: {h['hook']}")

    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(prog="rollup_annotations")
    ap.add_argument("--work", required=True)
    ap.add_argument("--label", default="gemma4")
    args = ap.parse_args()

    chapters = _load_chapters(args.work, args.label)
    if not chapters:
        raise SystemExit(f"no annotations found for {args.work}/{args.label}")

    stats_rows = [
        {"chapter": d["chapter"], **_chapter_stats(args.work, d["chapter"], d["annotation"])}
        for d in chapters
    ]
    total_words = sum(r["words"] for r in stats_rows)
    mean_scenes = sum(r["scenes"] for r in stats_rows) / len(stats_rows)
    mean_beats = sum(r["beats"] for r in stats_rows) / len(stats_rows)

    rollup = {
        "work": args.work,
        "label": args.label,
        "stats": {
            "n_chapters": len(chapters),
            "total_words": total_words,
            "mean_scenes": mean_scenes,
            "mean_beats": mean_beats,
            "per_chapter": stats_rows,
        },
        "arc_timeline": _arc_timeline(chapters),
        "characters": _merge_characters(chapters),
        "relationships": _merge_relationships(chapters),
        "world": _merge_world(chapters),
        "hooks": _merge_hooks(chapters),
    }

    out_json = ANN_ROOT / args.work / f"rollup.{args.label}.json"
    out_json.write_text(json.dumps(rollup, indent=2))
    _write_markdown(args.work, args.label, rollup,
                    ANN_ROOT / args.work / f"rollup.{args.label}.md")
    print(f"Wrote {out_json} and rollup.{args.label}.md")
    print(f"  characters: {len(rollup['characters'])}")
    print(f"  relationships: {len(rollup['relationships'])}")
    print(f"  world facts: {len(rollup['world'])}")
    print(f"  hooks planted: {len(rollup['hooks']['planted'])}")


if __name__ == "__main__":
    main()
