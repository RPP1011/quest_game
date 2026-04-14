import hashlib
from pathlib import Path

import yaml

from app.calibration.loader import (
    init_passage_hashes,
    load_manifest,
    sha256_file,
    verify_passage,
)


def _tiny_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "m.yaml"
    data = {
        "version": 1,
        "scoring": {"critic_error_weight": 0.25, "critic_warning_weight": 0.10},
        "works": [
            {
                "id": "demo",
                "title": "Demo",
                "author": "Anon",
                "year": 2020,
                "pov": "first",
                "is_quest": False,
                "expected": {"clarity": 0.9},
                "passages": [
                    {"id": "p01", "sha256": "PENDING",
                     "expected_high": ["clarity"], "expected_low": []},
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(data))
    return path


def test_sha256_roundtrip(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert sha256_file(f) == expected


def test_verify_passage_mismatch(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    ok, actual = verify_passage(f, "0" * 64)
    assert not ok and actual != "0" * 64


def test_init_fills_hashes(tmp_path):
    mpath = _tiny_manifest(tmp_path)
    pdir = tmp_path / "passages" / "demo"
    pdir.mkdir(parents=True)
    (pdir / "p01.txt").write_text("the passage body")

    updated = init_passage_hashes(mpath, tmp_path / "passages")
    assert "demo" in updated and "p01" in updated["demo"]

    m = load_manifest(mpath)
    assert m.works[0].passages[0].sha256 != "PENDING"
    assert len(m.works[0].passages[0].sha256) == 64


def test_init_skips_missing_files(tmp_path):
    mpath = _tiny_manifest(tmp_path)
    updated = init_passage_hashes(mpath, tmp_path / "no_such_dir")
    assert updated == {}
    # Manifest unchanged (still PENDING).
    m = load_manifest(mpath)
    assert m.works[0].passages[0].sha256 == "PENDING"
