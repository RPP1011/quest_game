"""Standard Ebooks plain-text fallback fetcher.

Standard Ebooks publishes a ``<slug>_plain-text.txt`` asset per book. We
pull that single file and write it as ``full.txt`` with a ``meta.json``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from ._http import FetchError, get_text


RAW_ROOT = Path("data/calibration/raw")
SOURCES = Path(__file__).parent / "sources.yaml"

log = logging.getLogger("corpus_fetchers.standard_ebooks")


def _candidate_urls(slug: str) -> list[str]:
    # Standard Ebooks download URL pattern: the "plain text" download file
    # is at /ebooks/<slug>/downloads/<flat-slug>.txt where flat-slug is
    # slug with '/' replaced by '_'.
    flat = slug.replace("/", "_")
    base = f"https://standardebooks.org/ebooks/{slug}"
    return [
        f"{base}/downloads/{flat}.txt",
        f"{base}/text/single-page",  # HTML fallback (not ideal; just a probe)
    ]


def fetch(
    work_id: str,
    *,
    slug: Optional[str] = None,
    raw_root: Path = RAW_ROOT,
    sources_path: Path = SOURCES,
    http_get=get_text,
) -> Path:
    if slug is None:
        cfg = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
        entry = cfg["works"].get(work_id) or {}
        fb = entry.get("fallback") or {}
        slug = fb.get("slug") or entry.get("slug")
        if not slug:
            raise FetchError(f"no standard_ebooks slug for {work_id}")

    out_dir = raw_root / work_id
    full_path = out_dir / "full.txt"
    if full_path.is_file() and full_path.stat().st_size > 1000:
        log.info("already fetched: %s", full_path)
        return full_path
    out_dir.mkdir(parents=True, exist_ok=True)

    last_err: Optional[Exception] = None
    for url in _candidate_urls(slug):
        try:
            text = http_get(url, allow_404=True)
        except FetchError as exc:
            last_err = exc
            continue
        if text is None or len(text) < 1000:
            continue
        full_path.write_text(text, encoding="utf-8")
        meta = {
            "work_id": work_id,
            "source": "standard_ebooks",
            "slug": slug,
            "source_url": url,
            "license": "public_domain_us",
            "fetched_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00","Z"),
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return full_path
    raise FetchError(f"standard_ebooks fetch failed for {work_id}: {last_err}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="standard_ebooks")
    ap.add_argument("work_id")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    args = ap.parse_args(argv)
    try:
        fetch(args.work_id, raw_root=Path(args.raw_root))
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
