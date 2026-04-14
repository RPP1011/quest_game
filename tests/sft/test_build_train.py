"""Day 4: round-trip from ``*.picked.json`` → ``train.jsonl`` row.

Plus shape-tests on the ``claude_pick_winners`` file walker (per scope:
don't test the Claude subagent path, just that the walker picks up the
right inputs and produces the right sidecars).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sft import build_train, claude_pick_winners as picker


# ---------------------------------------------------------------------------
# Fixtures: synthesize a miniature data/sft tree.


def _make_sft_record(
    tmp_path: Path,
    *,
    quest_id: str = "q-demo",
    update: int = 5,
    scene: int = 1,
    trace: str = "abcdef123",
    brief: str = "Write the protagonist entering the hall.",
    candidates: list[dict] | None = None,
    winner_index: int = 1,
) -> Path:
    quest_dir = tmp_path / quest_id
    quest_dir.mkdir(parents=True, exist_ok=True)
    if candidates is None:
        candidates = [
            {
                "index": 0,
                "prose": "you hesitate at the threshold.",
                "weighted_score": 5.1,
                "overall_score": 0.42,
                "dimension_scores": {"pov_adherence": 0.5},
                "rerank_source": "scorer",
            },
            {
                "index": 1,
                "prose": "you step into the cold hall and listen.",
                "weighted_score": 6.9,
                "overall_score": 0.58,
                "dimension_scores": {"pov_adherence": 0.9},
                "rerank_source": "scorer",
            },
            {
                "index": 2,
                "prose": "you drift forward through the shadows.",
                "weighted_score": 5.8,
                "overall_score": 0.51,
                "dimension_scores": {"pov_adherence": 0.7},
                "rerank_source": "scorer",
            },
        ]
    rec = {
        "quest_id": quest_id,
        "update_number": update,
        "scene_index": scene,
        "pipeline_trace_id": trace,
        "rerank_source": "scorer",
        "winner_index": winner_index,
        "craft_brief": brief,
        "candidates": candidates,
    }
    path = quest_dir / f"u{update}_s{scene}_{trace}.json"
    path.write_text(json.dumps(rec))
    return path


def _write_pick(src: Path, chosen_index: int, rationale: str = "test") -> Path:
    rec = json.loads(src.read_text())
    rec["claude_pick"] = {
        "chosen_index": chosen_index,
        "rationale": rationale,
        "model": "fake",
    }
    dst = src.with_suffix("").with_suffix(".picked.json")
    dst.write_text(json.dumps(rec))
    return dst


# ---------------------------------------------------------------------------
# build_row


def test_build_row_basic_roundtrip(tmp_path):
    src = _make_sft_record(tmp_path)
    picked = _write_pick(src, chosen_index=1, rationale="clean POV, concrete.")
    record = json.loads(picked.read_text())

    row = build_train.build_row(record, source=picked)

    # messages: system → user (contains brief) → assistant (chosen prose).
    assert [m["role"] for m in row.messages] == ["system", "user", "assistant"]
    assert "You are the prose writer" in row.messages[0]["content"]
    assert "Write the protagonist entering the hall." in row.messages[1]["content"]
    assert row.messages[2]["content"] == (
        "you step into the cold hall and listen."
    )
    # meta carries pointers back to the record.
    assert row.meta["quest_id"] == "q-demo"
    assert row.meta["update"] == 5
    assert row.meta["scene"] == 1
    assert row.meta["chosen_index"] == 1
    assert row.meta["scorer_overall"] == pytest.approx(0.58)
    assert row.meta["pipeline_trace_id"] == "abcdef123"
    assert row.meta["source"].endswith(".picked.json")


def test_build_row_raises_when_chosen_index_invalid(tmp_path):
    src = _make_sft_record(tmp_path)
    rec = json.loads(src.read_text())
    rec["claude_pick"] = {"chosen_index": 99, "rationale": "", "model": "fake"}
    with pytest.raises(ValueError):
        build_train.build_row(rec)


def test_build_row_raises_when_no_pick(tmp_path):
    src = _make_sft_record(tmp_path)
    rec = json.loads(src.read_text())  # no claude_pick
    with pytest.raises(ValueError):
        build_train.build_row(rec)


def test_build_row_raises_on_empty_candidates(tmp_path):
    src = _make_sft_record(tmp_path, candidates=[])
    rec = json.loads(src.read_text())
    rec["claude_pick"] = {"chosen_index": 0, "rationale": "", "model": "fake"}
    with pytest.raises(ValueError):
        build_train.build_row(rec)


# ---------------------------------------------------------------------------
# collect_rows + split_rows


def test_collect_rows_skips_malformed_in_non_strict(tmp_path):
    good = _make_sft_record(tmp_path, scene=1, trace="good")
    _write_pick(good, chosen_index=1)
    # bad: picked file with missing chosen_index
    bad = _make_sft_record(tmp_path, scene=2, trace="bad")
    bad_pick = bad.with_suffix("").with_suffix(".picked.json")
    bad_pick.write_text(json.dumps({"quest_id": "q-demo", "candidates": []}))

    rows, skipped = build_train.collect_rows(tmp_path)
    assert len(rows) == 1
    assert len(skipped) == 1
    assert skipped[0] == bad_pick


def test_collect_rows_strict_raises(tmp_path):
    bad = _make_sft_record(tmp_path, scene=2, trace="bad")
    bad_pick = bad.with_suffix("").with_suffix(".picked.json")
    bad_pick.write_text(json.dumps({"quest_id": "q-demo", "candidates": []}))
    with pytest.raises(ValueError):
        build_train.collect_rows(tmp_path, strict=True)


def test_split_rows_is_seeded(tmp_path):
    # 20 rows → 2 in test split at 0.1
    rows = []
    for i in range(20):
        src = _make_sft_record(
            tmp_path, scene=i, trace=f"t{i:02d}",
        )
        picked = _write_pick(src, chosen_index=1)
        rec = json.loads(picked.read_text())
        rows.append(build_train.build_row(rec, source=picked))

    train_a, test_a = build_train.split_rows(rows, test_ratio=0.1, seed=7)
    train_b, test_b = build_train.split_rows(rows, test_ratio=0.1, seed=7)
    assert [r.meta["scene"] for r in test_a] == [r.meta["scene"] for r in test_b]
    assert len(test_a) == 2
    assert len(train_a) == 18
    # No overlap
    train_scenes = {r.meta["scene"] for r in train_a}
    test_scenes = {r.meta["scene"] for r in test_a}
    assert train_scenes.isdisjoint(test_scenes)


def test_split_rows_empty_edge_cases():
    train, test = build_train.split_rows([], test_ratio=0.1, seed=1)
    assert train == [] and test == []


# ---------------------------------------------------------------------------
# full main() round-trip


def test_main_writes_train_and_test_jsonl(tmp_path):
    # 10 scenes, seeded 10% → 1 test row, 9 train rows.
    for i in range(10):
        src = _make_sft_record(
            tmp_path, scene=i, trace=f"t{i:02d}",
        )
        _write_pick(src, chosen_index=1)

    out_train = tmp_path / "train.jsonl"
    out_test = tmp_path / "test.jsonl"
    rc = build_train.main([
        "--root", str(tmp_path),
        "--out-train", str(out_train),
        "--out-test", str(out_test),
        "--test-ratio", "0.1",
        "--seed", "7",
    ])
    assert rc == 0
    assert out_train.exists()
    assert out_test.exists()
    train_lines = out_train.read_text().splitlines()
    test_lines = out_test.read_text().splitlines()
    assert len(train_lines) == 9
    assert len(test_lines) == 1
    # Every emitted row is a valid JSON object with messages + meta.
    for line in train_lines + test_lines:
        row = json.loads(line)
        assert "messages" in row
        assert "meta" in row
        assert [m["role"] for m in row["messages"]] == [
            "system", "user", "assistant",
        ]


# ---------------------------------------------------------------------------
# claude_pick_winners file walker — shape only (per scope: don't test the
# actual Claude subagent path).


def test_iter_sft_records_skips_picked_and_tmp(tmp_path):
    good = _make_sft_record(tmp_path, scene=1, trace="t01")
    # Adjacent noise:
    # - .picked.json sidecar must be skipped
    picked = good.with_suffix("").with_suffix(".picked.json")
    picked.write_text("{}")
    # - .json.tmp from a crashed write must be skipped
    tmp = good.with_suffix(".json.tmp")
    tmp.write_text("{}")

    found = list(picker.iter_sft_records(tmp_path))
    assert found == [good]


def test_iter_sft_records_respects_quest_filter(tmp_path):
    a = _make_sft_record(tmp_path, quest_id="q-a", scene=1, trace="t01")
    b = _make_sft_record(tmp_path, quest_id="q-b", scene=1, trace="t02")
    # All quests
    assert set(picker.iter_sft_records(tmp_path)) == {a, b}
    # Filter
    assert list(picker.iter_sft_records(tmp_path, quest_id="q-a")) == [a]
    assert list(picker.iter_sft_records(tmp_path, quest_id="q-b")) == [b]
    # Non-existent quest ⇒ empty
    assert list(picker.iter_sft_records(tmp_path, quest_id="q-z")) == []


def test_iter_sft_records_missing_root(tmp_path):
    assert list(picker.iter_sft_records(tmp_path / "nope")) == []


def test_pick_path_suffix_rewrite(tmp_path):
    src = tmp_path / "q" / "u1_s2_trace.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("{}")
    dst = picker.pick_path(src)
    assert dst.name == "u1_s2_trace.picked.json"


def test_is_stale_newer_src_stale(tmp_path):
    src = _make_sft_record(tmp_path, scene=1, trace="t01")
    dst = picker.pick_path(src)
    # dst missing ⇒ stale
    assert picker.is_stale(src, dst) is True
    dst.write_text("{}")
    # Touch src AFTER dst so it looks newer
    import os, time
    t0 = dst.stat().st_mtime
    os.utime(src, (t0 + 2, t0 + 2))
    assert picker.is_stale(src, dst) is True


def test_apply_rating_writes_sidecar(tmp_path):
    src = _make_sft_record(tmp_path, scene=1, trace="t01")
    dst = picker.pick_path(src)

    class _FakeRater:
        def rate(self, inp):
            # Ignore the brief; mechanically pick candidate 2.
            return picker.RatingResult(
                chosen_index=2,
                rationale="fake rater chose last",
                model="fake-rater/1",
            )

    picker.apply_rating(src, dst, _FakeRater())
    assert dst.exists()
    rec = json.loads(dst.read_text())
    assert rec["claude_pick"] == {
        "chosen_index": 2,
        "rationale": "fake rater chose last",
        "model": "fake-rater/1",
    }
    # Original fields preserved.
    assert rec["quest_id"] == "q-demo"
    assert len(rec["candidates"]) == 3


def test_walk_and_rate_skips_fresh_picks(tmp_path):
    src = _make_sft_record(tmp_path, scene=1, trace="t01")
    dst = picker.pick_path(src)

    class _FakeRater:
        def __init__(self):
            self.calls = 0

        def rate(self, inp):
            self.calls += 1
            return picker.RatingResult(
                chosen_index=0, rationale="", model="fake",
            )

    rater = _FakeRater()
    # First pass: picks the record.
    picker.walk_and_rate(tmp_path, rater=rater)
    assert rater.calls == 1
    assert dst.exists()

    # Second pass: sidecar fresh, no new rating (unless --force).
    picker.walk_and_rate(tmp_path, rater=rater)
    assert rater.calls == 1
    picker.walk_and_rate(tmp_path, rater=rater, force=True)
    assert rater.calls == 2


def test_heuristic_rater_picks_top_weighted_score(tmp_path):
    src = _make_sft_record(tmp_path, scene=1, trace="t01")
    record = json.loads(src.read_text())
    rater = picker.HeuristicFallbackRater()
    result = rater.rate(picker.RatingInput(
        brief=record["craft_brief"],
        candidates=record["candidates"],
    ))
    # Candidate 1 has the highest weighted_score in the fixture.
    assert result.chosen_index == 1
    assert result.model.startswith("heuristic/")


def test_main_walk_end_to_end_writes_picked(tmp_path, monkeypatch):
    _make_sft_record(tmp_path, scene=1, trace="t01")
    _make_sft_record(tmp_path, scene=2, trace="t02")

    rc = picker.main([
        "--root", str(tmp_path),
        "--rater", "heuristic",
    ])
    assert rc == 0
    picks = sorted((tmp_path / "q-demo").glob("*.picked.json"))
    assert len(picks) == 2
    for p in picks:
        rec = json.loads(p.read_text())
        assert "claude_pick" in rec
        assert isinstance(rec["claude_pick"]["chosen_index"], int)
