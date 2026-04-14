"""Sample scene-aligned chunks (2000-4000 words) from raw corpora.

Scene-scale analog to ``tools/sample_passages.py``. Used to score arc-level
dimensions (tension_execution, choice_hook_quality, update_self_containment,
choice_meaningfulness, world_state_legibility) that need a whole scene rather
than a snippet.

Strategy
--------
* Novels (Gutenberg ``full.txt``): detect chapter markers. For each selected
  chapter, if its word count is within [2000, 4000] use it whole; if larger,
  trim to the nearest scene break inside the window; if smaller, extend into
  the next chapter(s) until we hit the minimum.
* Quest fiction (``chap_NNNN.txt`` files): each file is one "update". Use the
  whole file if it fits in [2000, 4000]; otherwise trim to a scene break near
  the target size.
* Deterministic: seeded by ``work_id``. Five scenes per work.

Scene-break detection
---------------------
In order of precedence, the following markers split a chapter into scenes:
    1. Explicit section dividers (``***``, ``* * *``, ``---``, ``###``).
    2. ``Scene N`` / ``SCENE N`` headers on their own line.
    3. Paragraph breaks composed of 2+ blank lines (``\\n{3,}``).
If none are found, the chapter is treated as a single scene and trimmed by
word count alone (last-resort split at nearest paragraph boundary).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


RAW_ROOT = Path("data/calibration/raw")
SCENES_ROOT = Path("data/calibration/scenes")
MANIFEST_PATH = Path("data/calibration/manifest.yaml")

TARGET_MIN_WORDS = 2000
TARGET_MAX_WORDS = 4000
SCENES_PER_WORK = 5

CHAPTER_RE = re.compile(
    r"^\s*(?:CHAPTER|Chapter|Book|BOOK)\s+[IVXLC\d]+[^\n]{0,80}$",
    re.MULTILINE,
)

# Scene-break detectors ordered by strength.
_SCENE_DIVIDER_RE = re.compile(
    r"^\s*(?:\*\s*\*\s*\*|\*{3,}|-{3,}|#{3,}|§+|~{3,})\s*$",
    re.MULTILINE,
)
_SCENE_HEADER_RE = re.compile(
    r"^\s*(?:Scene|SCENE)\s+\d+[^\n]{0,60}$",
    re.MULTILINE,
)
_PARAGRAPH_GAP_RE = re.compile(r"\n\s*\n\s*\n+")

log = logging.getLogger("sample_scenes")


@dataclass
class Chapter:
    index: int
    text: str
    source: str  # file name or "full.txt#chapN"


def _split_gutenberg(full_text: str) -> list[Chapter]:
    """Split a ``full.txt`` into chapters via chapter markers, else blocks."""
    matches = list(CHAPTER_RE.finditer(full_text))
    if len(matches) < 3:
        # Fallback: chunk into ~chapter-sized blocks.
        words = full_text.split()
        block_size = 3500
        n_blocks = max(4, len(words) // block_size)
        per = len(words) // n_blocks
        chapters: list[Chapter] = []
        for i in range(n_blocks):
            start = i * per
            end = len(words) if i == n_blocks - 1 else (i + 1) * per
            chapters.append(
                Chapter(i + 1, " ".join(words[start:end]), f"full.txt#block{i+1}")
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


def find_scene_breaks(text: str) -> list[int]:
    """Return sorted character offsets of candidate scene breaks.

    Offsets point at the start of the *following* scene (safe to slice on).
    The list always includes 0 (start of text) and len(text) (end).
    """
    offsets: set[int] = {0, len(text)}
    for rx in (_SCENE_DIVIDER_RE, _SCENE_HEADER_RE):
        for m in rx.finditer(text):
            offsets.add(m.end())
    for m in _PARAGRAPH_GAP_RE.finditer(text):
        offsets.add(m.end())
    return sorted(offsets)


def _words(text: str) -> list[str]:
    return text.split()


def _slice_to_window(
    text: str,
    *,
    min_words: int,
    max_words: int,
) -> str:
    """Trim ``text`` to a scene boundary inside [min_words, max_words].

    Prefer the largest break offset whose prefix has words <= max_words and
    >= min_words. If none qualifies, fall back to a word-count slice at the
    nearest paragraph boundary near max_words.
    """
    breaks = find_scene_breaks(text)
    # Precompute cumulative word counts at each break offset.
    best: tuple[int, int] | None = None  # (offset, word_count)
    for off in breaks:
        if off == 0:
            continue
        wc = len(_words(text[:off]))
        if min_words <= wc <= max_words:
            if best is None or wc > best[1]:
                best = (off, wc)
    if best is not None:
        return text[: best[0]].rstrip()

    # Fall back: slice by word count at a paragraph boundary.
    words = _words(text)
    if len(words) <= max_words:
        return text.rstrip()
    # Approximate target offset by characters.
    approx_char = int(len(text) * (max_words / len(words)))
    # Look for a \n\n within +/- 10% of target.
    window = max(200, int(len(text) * 0.1))
    lo = max(0, approx_char - window)
    hi = min(len(text), approx_char + window)
    search = text[lo:hi]
    para = search.rfind("\n\n")
    if para >= 0:
        cut = lo + para
        if len(_words(text[:cut])) >= min_words:
            return text[:cut].rstrip()
    # Hard cut by word count as last resort.
    return " ".join(words[:max_words])


def build_scene(
    chapters: list[Chapter],
    start_idx: int,
    *,
    min_words: int = TARGET_MIN_WORDS,
    max_words: int = TARGET_MAX_WORDS,
) -> tuple[str, list[int]]:
    """Build one scene starting at ``chapters[start_idx]``.

    Returns (scene_text, source_chapter_indexes).
    """
    if start_idx >= len(chapters):
        raise IndexError("start_idx past end of chapters")
    acc = chapters[start_idx].text
    sources = [chapters[start_idx].index]
    # Extend into subsequent chapters until we hit the minimum.
    i = start_idx + 1
    while len(_words(acc)) < min_words and i < len(chapters):
        acc = acc.rstrip() + "\n\n" + chapters[i].text
        sources.append(chapters[i].index)
        i += 1
    # Trim to a scene boundary under max_words.
    scene = _slice_to_window(acc, min_words=min_words, max_words=max_words)
    return scene, sources


def _select_starts(chapters: list[Chapter], rng: random.Random, n: int) -> list[int]:
    """Pick ``n`` chapter indexes (0-based) spread across the middle 80%."""
    total = len(chapters)
    if total == 0:
        return []
    lo = int(total * 0.10)
    hi = max(lo + 1, int(total * 0.90))
    pool = list(range(lo, hi))
    if not pool:
        pool = list(range(total))
    if len(pool) <= n:
        return pool[:n]
    # Deterministic even-ish spread, then jitter with rng.
    step = len(pool) / n
    picks: list[int] = []
    for i in range(n):
        center = int(i * step + step / 2)
        jitter = rng.randint(-1, 1)
        idx = max(0, min(len(pool) - 1, center + jitter))
        picks.append(pool[idx])
    # Deduplicate preserving order.
    seen: set[int] = set()
    uniq = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    # Fill if we lost some to dedup.
    j = 0
    while len(uniq) < n and j < len(pool):
        if pool[j] not in seen:
            uniq.append(pool[j])
            seen.add(pool[j])
        j += 1
    return uniq[:n]


def sample_work(
    work: dict,
    raw_root: Path = RAW_ROOT,
    scenes_root: Path = SCENES_ROOT,
    n_scenes: int = SCENES_PER_WORK,
) -> list[Path]:
    work_id = work["id"]
    raw_dir = raw_root / work_id
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"no raw corpus for {work_id} at {raw_dir}")
    chapters = load_chapters(raw_dir)
    if not chapters:
        raise RuntimeError(f"{work_id}: no chapters loaded")

    rng = random.Random(f"scenes:{work_id}")
    starts = _select_starts(chapters, rng, n_scenes)

    out_dir = scenes_root / work_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, start_idx in enumerate(starts, start=1):
        try:
            scene_text, sources = build_scene(chapters, start_idx)
        except Exception as exc:
            log.warning("%s: skipping scene %d: %s", work_id, i, exc)
            continue
        wc = len(_words(scene_text))
        sid = f"s{i:02d}"
        frontmatter = {
            "work": work.get("title", work_id),
            "author": work.get("author", ""),
            "passage_id": sid,
            "kind": "scene",
            "word_count": wc,
            "source_chapters": sources,
            "notes": (
                "scene-aligned chunk; see tools/sample_scenes.py. "
                f"start_chapter_idx={start_idx}."
            ),
        }
        fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        body = f"---\n{fm_yaml}\n---\n\n{scene_text.strip()}\n"
        out_path = out_dir / f"{sid}.txt"
        out_path.write_text(body, encoding="utf-8")
        written.append(out_path)
        log.info("wrote %s (%d words, chapters=%s)", out_path, wc, sources)
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="sample_scenes")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--scenes-dir", default=str(SCENES_ROOT))
    ap.add_argument("--n-scenes", type=int, default=SCENES_PER_WORK)
    ap.add_argument("--works", default=None)
    args = ap.parse_args(argv)

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    wanted = set(args.works.split(",")) if args.works else None
    raw_root = Path(args.raw_root)
    scenes_root = Path(args.scenes_dir)

    errors = 0
    for work in manifest["works"]:
        wid = work["id"]
        if wanted and wid not in wanted:
            continue
        if not (raw_root / wid).is_dir():
            log.info("skip %s: no raw corpus", wid)
            continue
        try:
            sample_work(
                work,
                raw_root=raw_root,
                scenes_root=scenes_root,
                n_scenes=args.n_scenes,
            )
        except Exception as exc:  # pragma: no cover
            log.error("failed %s: %s", wid, exc)
            errors += 1
    return 1 if errors else 0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
