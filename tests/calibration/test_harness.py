import json
from pathlib import Path

import pytest
import yaml

from app.calibration.harness import Harness
from app.calibration.judges import BatchJudge, COMMON_LLM_DIMS, QUEST_LLM_DIMS
from app.calibration.loader import init_passage_hashes, load_manifest


PROMPTS = Path("prompts")


class _StubClient:
    def __init__(self, scores_by_dim: dict[str, float]):
        self.scores_by_dim = scores_by_dim

    async def chat_structured(self, *, messages, json_schema, **kwargs) -> str:
        data = {
            d: {"score": self.scores_by_dim.get(d, 0.5), "rationale": "stub"}
            for d in json_schema["properties"].keys()
        }
        return json.dumps(data)


def _mini_manifest(tmp_path: Path, expected: dict[str, float],
                   is_quest: bool = False) -> Path:
    path = tmp_path / "m.yaml"
    data = {
        "version": 1,
        "scoring": {"critic_error_weight": 0.25, "critic_warning_weight": 0.10},
        "works": [{
            "id": "demo",
            "title": "Demo",
            "author": "Anon",
            "year": 2020,
            "pov": "first",
            "is_quest": is_quest,
            "expected": expected,
            "passages": [
                {"id": "p01", "sha256": "PENDING",
                 "expected_high": ["clarity"], "expected_low": []},
            ],
        }],
    }
    path.write_text(yaml.safe_dump(data))
    return path


def _write_passage(tmp_path: Path, work_id: str, passage_id: str, body: str) -> Path:
    p = tmp_path / "passages" / work_id
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{passage_id}.txt"
    f.write_text(body)
    return f


async def test_harness_scores_and_rolls_up(tmp_path):
    expected = {d: 0.5 for d in COMMON_LLM_DIMS}
    expected.update({"sentence_variance": 0.3, "dialogue_ratio": 0.0, "pacing": 0.5})
    mpath = _mini_manifest(tmp_path, expected, is_quest=False)
    _write_passage(tmp_path, "demo", "p01",
                   "He ran. She ran. They ran together without stopping.")
    init_passage_hashes(mpath, tmp_path / "passages")
    manifest = load_manifest(mpath)

    client = _StubClient({d: 0.5 for d in COMMON_LLM_DIMS})
    h = Harness(manifest, tmp_path / "passages", client=client,
                judge=BatchJudge(PROMPTS))
    report = await h.run()
    assert len(report.passages) == 1
    ps = report.passages[0]
    assert not ps.skipped
    assert not ps.errors
    # LLM dims arrived.
    for d in COMMON_LLM_DIMS:
        assert d in ps.dimensions
    # Heuristic dims arrived.
    assert "sentence_variance" in ps.dimensions
    # Per-dim stats populated.
    by_dim = {s.dimension: s for s in report.per_dim}
    assert "clarity" in by_dim
    assert by_dim["clarity"].n == 1


async def test_harness_skips_missing_passage(tmp_path):
    """Work dir exists but the individual passage file is absent."""
    expected = {d: 0.5 for d in COMMON_LLM_DIMS}
    mpath = _mini_manifest(tmp_path, expected)
    # The mini manifest fixture's work id is `woolf`; create its dir empty.
    import yaml as _yaml
    work_id = _yaml.safe_load(mpath.read_text())["works"][0]["id"]
    (tmp_path / "passages" / work_id).mkdir(parents=True)
    h = Harness(load_manifest(mpath), tmp_path / "passages", client=None,
                judge=BatchJudge(PROMPTS))
    report = await h.run()
    assert report.passages[0].skipped
    assert "missing" in report.passages[0].skip_reason


async def test_harness_silently_skips_works_without_dir(tmp_path):
    expected = {d: 0.5 for d in COMMON_LLM_DIMS}
    mpath = _mini_manifest(tmp_path, expected)
    (tmp_path / "passages").mkdir()
    h = Harness(load_manifest(mpath), tmp_path / "passages", client=None,
                judge=BatchJudge(PROMPTS))
    report = await h.run()
    assert report.passages == []


async def test_harness_sha_mismatch_raises(tmp_path):
    expected = {d: 0.5 for d in COMMON_LLM_DIMS}
    mpath = _mini_manifest(tmp_path, expected)
    _write_passage(tmp_path, "demo", "p01", "one body")
    init_passage_hashes(mpath, tmp_path / "passages")
    # Change file content after hashing.
    _write_passage(tmp_path, "demo", "p01", "changed body")
    h = Harness(load_manifest(mpath), tmp_path / "passages", client=None,
                judge=BatchJudge(PROMPTS))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        await h.run()


async def test_quest_vs_novel_dim_gating(tmp_path):
    # Novel: judges called with COMMON only.
    expected_novel = {d: 0.5 for d in COMMON_LLM_DIMS}
    mpath = _mini_manifest(tmp_path, expected_novel, is_quest=False)
    _write_passage(tmp_path, "demo", "p01", "A sentence here.")
    init_passage_hashes(mpath, tmp_path / "passages")
    client = _StubClient({d: 0.5 for d in COMMON_LLM_DIMS})
    h = Harness(load_manifest(mpath), tmp_path / "passages", client=client,
                judge=BatchJudge(PROMPTS))
    report = await h.run()
    dims = set(report.passages[0].dimensions)
    for d in QUEST_LLM_DIMS:
        assert d not in dims

    # Quest: judges called with COMMON + QUEST.
    tmp2 = tmp_path / "quest"
    tmp2.mkdir()
    expected_quest = {d: 0.5 for d in list(COMMON_LLM_DIMS) + list(QUEST_LLM_DIMS)}
    mpath2 = _mini_manifest(tmp2, expected_quest, is_quest=True)
    _write_passage(tmp2, "demo", "p01", "A sentence here.")
    init_passage_hashes(mpath2, tmp2 / "passages")
    client2 = _StubClient({d: 0.5 for d in list(COMMON_LLM_DIMS) + list(QUEST_LLM_DIMS)})
    h2 = Harness(load_manifest(mpath2), tmp2 / "passages", client=client2,
                 judge=BatchJudge(PROMPTS))
    report2 = await h2.run()
    dims2 = set(report2.passages[0].dimensions)
    for d in QUEST_LLM_DIMS:
        assert d in dims2
