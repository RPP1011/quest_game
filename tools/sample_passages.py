"""Sample mid-chapter passages from raw corpora.

Deterministic: seeded by ``work_id``. Writes passages to
``data/calibration/passages/<work_id>/pNN.txt`` with YAML frontmatter.

Rules:
    * 500-1000 words per passage.
    * Mid-chapter: drop first 20% and last 15% of each chapter's words.
    * Prefer chapters from the middle 80% of the book.
    * When the manifest has 2 passage slots, emit one dialogue-lighter
      and one dialogue-heavier passage when possible (judged by
      :func:`app.calibration.heuristics.dialogue_ratio`).
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from app.calibration.heuristics import dialogue_ratio


RAW_ROOT = Path("data/calibration/raw")
PASSAGES_ROOT = Path("data/calibration/passages")
MANIFEST_PATH = Path("data/calibration/manifest.yaml")

# Split gutenberg ``full.txt`` into chapters. We match common markers
# (CHAPTER N, CHAPTER I, roman numerals on their own line) and fall back
# to dividing the text into equal blocks.
CHAPTER_RE = re.compile(
    r"^\s*(?:CHAPTER|Chapter|Book|BOOK)\s+[IVXLC\d]+[^\n]{0,80}$",
    re.MULTILINE,
)
WORD_RE = re.compile(r"\S+")

TARGET_MIN_WORDS = 500
TARGET_MAX_WORDS = 1000

log = logging.getLogger("sample_passages")


@dataclass
class Chapter:
    index: int
    text: str
    source: str  # filename or "full.txt#N"


def _split_gutenberg(full_text: str) -> list[Chapter]:
    matches = list(CHAPTER_RE.finditer(full_text))
    if len(matches) < 3:
        # Fallback: equal blocks
        words = full_text.split()
        n_blocks = max(10, len(words) // 5000)
        block_size = len(words) // n_blocks
        chapters = []
        for i in range(n_blocks):
            start = i * block_size
            end = len(words) if i == n_blocks - 1 else (i + 1) * block_size
            chapters.append(
                Chapter(i + 1, " ".join(words[start:end]), f"full.txt#{i+1}")
            )
        return chapters
    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chapters.append(
            Chapter(i + 1, full_text[start:end].strip(), f"full.txt#chap{i+1}")
        )
    return chapters


def load_chapters(raw_dir: Path) -> list[Chapter]:
    """Load chapters from a work's raw dir."""
    chap_files = sorted(raw_dir.glob("chap_*.txt"))
    if chap_files:
        return [
            Chapter(i + 1, p.read_text(encoding="utf-8"), p.name)
            for i, p in enumerate(chap_files)
        ]
    full = raw_dir / "full.txt"
    if full.is_file():
        return _split_gutenberg(full.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no raw text in {raw_dir}")


def _mid_slice(chapter_text: str) -> tuple[list[str], int, int]:
    """Return (words, lo, hi) where [lo, hi) is the allowed start-index window."""
    words = chapter_text.split()
    n = len(words)
    lo = int(n * 0.20)
    hi = int(n * 0.85) - TARGET_MIN_WORDS
    return words, lo, max(lo, hi)


def _candidate_passage(
    chapter: Chapter, rng: random.Random, length: int
) -> Optional[tuple[str, int]]:
    """Return (passage_text, start_word_index) or None if no valid window."""
    words, lo, hi = _mid_slice(chapter.text)
    if hi <= lo or len(words) < length + lo:
        return None
    start = rng.randint(lo, hi)
    end = min(len(words), start + length)
    return " ".join(words[start:end]), start


def _pick_chapter_pool(chapters: list[Chapter]) -> list[Chapter]:
    n = len(chapters)
    if n <= 3:
        return chapters
    lo = int(n * 0.10)
    hi = int(n * 0.90)
    pool = chapters[lo:hi] or chapters
    # Require chapter can yield at least one min-length passage plus its
    # 20% mid-slice offset. This is the sampler's hard constraint.
    return [c for c in pool if len(c.text.split()) >= TARGET_MIN_WORDS + 100]


def sample_work(
    work: dict,
    raw_root: Path = RAW_ROOT,
    passages_root: Path = PASSAGES_ROOT,
) -> list[Path]:
    work_id = work["id"]
    raw_dir = raw_root / work_id
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"no raw corpus for {work_id} at {raw_dir}")
    chapters = load_chapters(raw_dir)
    pool = _pick_chapter_pool(chapters)
    if not pool:
        raise RuntimeError(f"no usable chapters for {work_id}")

    slots = work.get("passages") or []
    n_slots = len(slots)
    rng = random.Random(work_id)

    target_len = rng.randint(TARGET_MIN_WORDS + 50, TARGET_MAX_WORDS - 50)
    # Reject any new candidate whose start is within 60% of target_len of
    # an accepted passage from the same chapter — guarantees ≤40% textual
    # overlap. Prior cheap hash(passage[:80]) dedup missed offset-window
    # near-duplicates; this is the structural fix.
    min_offset = int(target_len * 0.6)
    candidates: list[tuple[Chapter, str, float]] = []
    candidate_cap = max(n_slots * 2, 30)
    try_cap = candidate_cap * 6
    tries = 0
    chapter_order = list(pool)
    rng.shuffle(chapter_order)
    for ch in chapter_order:
        if tries > try_cap or len(candidates) >= candidate_cap:
            break
        local_rng = random.Random(f"{work_id}:{ch.index}")
        accepted_starts: list[int] = []
        got = 0
        for _ in range(16):
            tries += 1
            result = _candidate_passage(ch, local_rng, target_len)
            if result is None:
                continue
            passage, start = result
            if any(abs(start - s) < min_offset for s in accepted_starts):
                continue
            accepted_starts.append(start)
            candidates.append((ch, passage, dialogue_ratio(passage)))
            got += 1
            if got >= 4:
                break

    if len(candidates) < n_slots:
        raise RuntimeError(
            f"{work_id}: only found {len(candidates)} viable passages; need {n_slots}"
        )

    # Dialogue-balanced selection when 2 slots.
    selected: list[tuple[Chapter, str]] = []
    if n_slots == 2:
        sorted_cands = sorted(candidates, key=lambda t: t[2])
        light = sorted_cands[0]
        heavy = sorted_cands[-1]
        # Ensure from different chapters if possible
        if light[0].index == heavy[0].index and len(sorted_cands) > 2:
            heavy = sorted_cands[-2]
        selected = [(light[0], light[1]), (heavy[0], heavy[1])]
    else:
        for i in range(n_slots):
            selected.append((candidates[i][0], candidates[i][1]))

    out_dir = passages_root / work_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slot, (ch, text) in zip(slots, selected):
        pid = slot["id"]
        frontmatter = {
            "work": work.get("title", work_id),
            "author": work.get("author", ""),
            "passage_id": pid,
            "source": ch.source,
            "pov_character": "",
        }
        fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        body = f"---\n{fm_yaml}\n---\n\n{text.strip()}\n"
        out_path = out_dir / f"{pid}.txt"
        out_path.write_text(body, encoding="utf-8")
        written.append(out_path)
        log.info("wrote %s (%d words)", out_path, len(text.split()))
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="sample_passages")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--passages-dir", default=str(PASSAGES_ROOT))
    ap.add_argument(
        "--works",
        default=None,
        help="comma-separated work_ids; default = all works with raw data",
    )
    args = ap.parse_args(argv)

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    wanted = set(args.works.split(",")) if args.works else None
    raw_root = Path(args.raw_root)
    passages_root = Path(args.passages_dir)

    errors = 0
    for work in manifest["works"]:
        wid = work["id"]
        if wanted and wid not in wanted:
            continue
        if not (raw_root / wid).is_dir():
            log.info("skip %s: no raw corpus", wid)
            continue
        try:
            sample_work(work, raw_root=raw_root, passages_root=passages_root)
        except Exception as exc:  # pragma: no cover - reported above
            log.error("failed %s: %s", wid, exc)
            errors += 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
