"""Manifest loader and passage-hash bookkeeping."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PENDING_SHA = "PENDING"


@dataclass
class Passage:
    id: str
    sha256: str
    expected_high: list[str] = field(default_factory=list)
    expected_low: list[str] = field(default_factory=list)


@dataclass
class Work:
    id: str
    title: str
    author: str
    year: int
    pov: str
    is_quest: bool
    expected: dict[str, float]
    passages: list[Passage]


@dataclass
class Manifest:
    version: int
    scoring: dict[str, float]
    works: list[Work]

    def work(self, work_id: str) -> Work:
        for w in self.works:
            if w.id == work_id:
                return w
        raise KeyError(work_id)


def _parse_work(raw: dict[str, Any]) -> Work:
    passages = [
        Passage(
            id=p["id"],
            sha256=p.get("sha256", PENDING_SHA),
            expected_high=list(p.get("expected_high", []) or []),
            expected_low=list(p.get("expected_low", []) or []),
        )
        for p in raw.get("passages", []) or []
    ]
    return Work(
        id=raw["id"],
        title=raw["title"],
        author=raw["author"],
        year=int(raw["year"]),
        pov=raw["pov"],
        is_quest=bool(raw.get("is_quest", False)),
        expected={k: float(v) for k, v in (raw.get("expected") or {}).items()},
        passages=passages,
    )


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Manifest(
        version=int(raw.get("version", 1)),
        scoring={k: float(v) for k, v in (raw.get("scoring") or {}).items()},
        works=[_parse_work(w) for w in raw["works"]],
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def init_passage_hashes(
    manifest_path: str | Path,
    passages_dir: str | Path,
) -> dict[str, dict[str, str]]:
    """Walk ``passages_dir/<work_id>/<passage_id>.txt`` and fill in sha256s.

    Rewrites ``manifest_path`` in place, preserving YAML shape as best we can.
    Returns a map of {work_id: {passage_id: sha256}} of what was updated.
    """
    manifest_path = Path(manifest_path)
    passages_dir = Path(passages_dir)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    updated: dict[str, dict[str, str]] = {}
    for w in raw["works"]:
        work_id = w["id"]
        for p in w.get("passages", []) or []:
            pid = p["id"]
            candidate = passages_dir / work_id / f"{pid}.txt"
            if not candidate.is_file():
                continue
            digest = sha256_file(candidate)
            if p.get("sha256") != digest:
                p["sha256"] = digest
                updated.setdefault(work_id, {})[pid] = digest

    manifest_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return updated


def verify_passage(path: Path, expected_sha: str) -> tuple[bool, str]:
    """Return (matches, actual_sha)."""
    actual = sha256_file(path)
    return actual == expected_sha, actual
