"""End-to-end corpus orchestrator: fetch -> sample -> init hashes.

Only covers works whose license permits automated download: the public-
domain novels on Gutenberg and the four public web serials. The six
modern copyrighted novels (``sun_also_rises`` is borderline; it is grouped
with the PD set only if the user has verified its Gutenberg id) must be
supplied manually.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from tools.corpus_fetchers import gutenberg, sv_forum, standard_ebooks, wordpress
from tools.corpus_fetchers._http import FetchError
from tools.sample_passages import PASSAGES_ROOT, RAW_ROOT, sample_work


MANIFEST_PATH = Path("data/calibration/manifest.yaml")
SOURCES_PATH = Path(__file__).parent / "corpus_fetchers" / "sources.yaml"

AUTOMATABLE = {
    "pride_and_prejudice",
    "madame_bovary",
    "brothers_karamazov",
    "ulysses",
    "mrs_dalloway",
    "sun_also_rises",
    "marked_for_death",
    "forge_of_destiny",
    "practical_guide_evil",
    "pale_lights",
}

COPYRIGHTED_MANUAL = {
    "blood_meridian",
    "left_hand_of_darkness",
    "get_shorty",
    "remains_of_the_day",
    "blade_itself",
    "way_of_kings",
}

log = logging.getLogger("build_corpus")


def _write_manual_stub(work_id: str, work_meta: dict, raw_root: Path) -> None:
    """Write a README stub into raw/<work_id>/ for copyrighted works."""
    out_dir = raw_root / work_id
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.txt"
    if readme.exists():
        return
    title = work_meta.get("title", work_id)
    author = work_meta.get("author", "")
    readme.write_text(
        f"{title}"
        + (f" — {author}\n" if author else "\n")
        + "\n"
        "This work is copyrighted and cannot be fetched automatically.\n"
        "\n"
        f"To include it in the calibration corpus, place a plain-text copy\n"
        f"of the full work at:\n"
        f"    {out_dir / 'full.txt'}\n"
        "\n"
        "Then run:\n"
        f"    uv run python -m tools.sample_passages {work_id}\n"
        f"    uv run python -m app.calibration init \\\n"
        f"        --passages-dir data/calibration/passages\n",
        encoding="utf-8",
    )


FETCHERS = {
    "gutenberg": gutenberg.fetch,
    "sv_forum": sv_forum.fetch,
    "wordpress": wordpress.fetch,
    "standard_ebooks": standard_ebooks.fetch,
}


def fetch_one(work_id: str, sources: dict, raw_root: Path) -> bool:
    entry = sources["works"].get(work_id)
    if not entry:
        log.error("no source entry for %s", work_id)
        return False
    kind = entry["type"]
    fn = FETCHERS.get(kind)
    if not fn:
        log.error("unknown source type %s for %s", kind, work_id)
        return False
    try:
        fn(work_id, raw_root=raw_root)
        return True
    except FetchError as exc:
        log.warning("primary fetch failed for %s: %s", work_id, exc)
        fb = entry.get("fallback")
        if fb:
            fb_fn = FETCHERS.get(fb["type"])
            if fb_fn:
                try:
                    fb_fn(work_id, raw_root=raw_root)
                    return True
                except FetchError as exc2:
                    log.error("fallback also failed for %s: %s", work_id, exc2)
        if entry.get("skip_if_unavailable"):
            log.warning("skipping %s: marked skip_if_unavailable", work_id)
            return False
        return False


def run(
    work_ids: list[str],
    manifest_path: Path = MANIFEST_PATH,
    raw_root: Path = RAW_ROOT,
    passages_root: Path = PASSAGES_ROOT,
    sources_path: Path = SOURCES_PATH,
    run_init: bool = True,
) -> int:
    sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest_by_id = {w["id"]: w for w in manifest["works"]}

    errors = 0
    for wid in work_ids:
        if wid in COPYRIGHTED_MANUAL:
            _write_manual_stub(wid, manifest_by_id.get(wid, {}), raw_root)
            log.warning(
                "%s is copyrighted; wrote README stub to %s/%s/ — supply full.txt manually",
                wid, raw_root, wid,
            )
            continue
        if wid not in AUTOMATABLE:
            log.error("unknown work id: %s", wid)
            errors += 1
            continue
        if not fetch_one(wid, sources, raw_root):
            errors += 1
            continue
        work = manifest_by_id.get(wid)
        if not work:
            log.error("no manifest entry for %s", wid)
            errors += 1
            continue
        try:
            sample_work(work, raw_root=raw_root, passages_root=passages_root)
        except Exception as exc:
            log.error("sampling failed for %s: %s", wid, exc)
            errors += 1

    if run_init and any((passages_root / w).is_dir() for w in work_ids):
        log.info("running calibrate init")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "app.calibration",
                "init",
                "--manifest",
                str(manifest_path),
                "--passages-dir",
                str(passages_root),
            ],
            check=False,
        )
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="build_corpus")
    ap.add_argument("--works", default=None, help="comma-separated work_ids")
    ap.add_argument("--all-public", action="store_true",
                    help="process all 10 automatable works")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--passages-dir", default=str(PASSAGES_ROOT))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--no-init", action="store_true")
    args = ap.parse_args(argv)

    if args.all_public:
        work_ids = sorted(AUTOMATABLE)
    elif args.works:
        work_ids = [w.strip() for w in args.works.split(",") if w.strip()]
    else:
        ap.error("pass --works or --all-public")

    for w in COPYRIGHTED_MANUAL:
        if w in work_ids:
            log.warning(
                "%s is copyrighted — cannot auto-fetch; supply manually", w
            )

    return run(
        work_ids,
        manifest_path=Path(args.manifest),
        raw_root=Path(args.raw_root),
        passages_root=Path(args.passages_dir),
        run_init=not args.no_init,
    )


if __name__ == "__main__":
    sys.exit(main())
