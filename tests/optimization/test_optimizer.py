"""Day 7 tests for :class:`app.optimization.PromptOptimizer`.

All tests are sync and stub the mutation proposer + pipeline factory —
no real inference, no real pipeline reconstruction. The fixture seeds
scorecards with a known weak dim (``free_indirect_quality``) so the
detector has something unambiguous to latch onto.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.optimization import (
    ABResult,
    Mutation,
    PromptOptimizer,
    WeakDim,
)


# ---------------------------------------------------------------------------
# identify_weak_dimensions
# ---------------------------------------------------------------------------


def test_identify_weak_dimensions_returns_lowest_mean(seeded_world):
    opt = PromptOptimizer(seeded_world)
    weak = opt.identify_weak_dimensions(quest_id="qA", threshold=0.5, n=3)
    assert isinstance(weak, list)
    assert len(weak) == 3
    # free_indirect_quality should be the weakest (mean = (0.1+0.2+0.3+0.85+0.95)/5 = 0.48)
    names = [w.dimension for w in weak]
    assert names[0] == "free_indirect_quality"
    # All returned WeakDim entries carry a computed mean and sample size.
    for w in weak:
        assert isinstance(w, WeakDim)
        assert 0.0 <= w.mean_score <= 1.0
        assert w.sample_size > 0


def test_identify_weak_dimensions_returns_snippets(seeded_world):
    opt = PromptOptimizer(seeded_world)
    weak = opt.identify_weak_dimensions(quest_id="qA", threshold=0.5, n=1)
    w = weak[0]
    assert w.dimension == "free_indirect_quality"
    assert len(w.recent_examples) >= 1
    # Snippets should be the prose we seeded, ordered worst-first.
    # Worst free_indirect_quality is 0.1 @ trace tr-1 -> update 1.
    assert "trace tr-1" in w.recent_examples[0]


def test_identify_weak_dimensions_no_scorecards_returns_empty(world):
    opt = PromptOptimizer(world)
    assert opt.identify_weak_dimensions(quest_id="nope") == []


def test_identify_weak_dimensions_pads_when_fewer_than_n_below_threshold(
    seeded_world,
):
    # With threshold=0.0, nothing is "below", but we still want n results.
    opt = PromptOptimizer(seeded_world)
    weak = opt.identify_weak_dimensions(quest_id="qA", threshold=0.0, n=2)
    assert len(weak) == 2


def test_identify_weak_dimensions_all_quests_when_unscoped(seeded_world):
    opt = PromptOptimizer(seeded_world)
    weak = opt.identify_weak_dimensions(quest_id=None, threshold=0.5, n=1)
    assert weak[0].dimension == "free_indirect_quality"


# ---------------------------------------------------------------------------
# propose_mutation
# ---------------------------------------------------------------------------


def _stub_proposer_fn(dim: str, prompt: str, examples):
    # Returns an "edited" prompt that tags the dim + example count.
    new = prompt + f"\n\n# BIAS TOWARD {dim.upper()} (n_examples={len(examples)})"
    rationale = f"stubbed for {dim}"
    return new, rationale


def test_propose_mutation_reads_prompt_and_returns_mutation(
    tmp_path: Path, seeded_world,
):
    prompt_path = tmp_path / "write.j2"
    prompt_path.write_text("Original prompt body.")

    opt = PromptOptimizer(
        seeded_world,
        mutation_proposer=_stub_proposer_fn,
        prompts_root=tmp_path,
    )
    weak = opt.identify_weak_dimensions(quest_id="qA", n=1)[0]
    mutation = opt.propose_mutation(weak, "write.j2")
    assert isinstance(mutation, Mutation)
    assert mutation.dimension == weak.dimension
    assert mutation.before_text == "Original prompt body."
    assert "BIAS TOWARD FREE_INDIRECT_QUALITY" in mutation.after_text
    assert "stubbed for" in mutation.rationale


def test_propose_mutation_without_proposer_raises(
    tmp_path: Path, seeded_world,
):
    prompt_path = tmp_path / "write.j2"
    prompt_path.write_text("x")
    opt = PromptOptimizer(seeded_world, prompts_root=tmp_path)
    weak = WeakDim(dimension="free_indirect_quality", mean_score=0.2, sample_size=3)
    with pytest.raises(RuntimeError, match="mutation_proposer"):
        opt.propose_mutation(weak, "write.j2")


# ---------------------------------------------------------------------------
# replay_ab (stubbed pipeline factory)
# ---------------------------------------------------------------------------


def test_replay_ab_returns_structure_and_accept_when_improved(seeded_world):
    opt = PromptOptimizer(seeded_world)
    mutation = Mutation(
        dimension="free_indirect_quality",
        prompt_path="stages/write/user.j2",
        before_text="before",
        after_text="after",
        rationale="test",
    )

    def factory(ctx, mutated_prompt):
        assert mutated_prompt == "after"
        # Baseline for free_indirect_quality on tr-1 is 0.1; return a big bump.
        return {"free_indirect_quality": 0.9}

    result = opt.replay_ab(
        mutation,
        trace_ids=["tr-1", "tr-2", "tr-3"],
        pipeline_factory=factory,
    )
    assert isinstance(result, ABResult)
    assert result.n_replays == 3
    assert result.mean_before < result.mean_after
    assert result.accept_recommended is True


def test_replay_ab_rejects_when_no_improvement(seeded_world):
    opt = PromptOptimizer(seeded_world)
    mutation = Mutation(
        dimension="free_indirect_quality",
        prompt_path="stages/write/user.j2",
        before_text="before",
        after_text="after",
    )

    # Replay returns the same score we started with.
    def factory(ctx, mutated_prompt):
        return {"free_indirect_quality": ctx.baseline_dim_score or 0.0}

    result = opt.replay_ab(
        mutation, trace_ids=["tr-1", "tr-2"], pipeline_factory=factory,
    )
    assert result.n_replays == 2
    assert result.accept_recommended is False


def test_replay_ab_skips_unknown_traces(seeded_world):
    opt = PromptOptimizer(seeded_world)
    mutation = Mutation(
        dimension="free_indirect_quality",
        prompt_path="stages/write/user.j2",
        before_text="before",
        after_text="after",
    )
    calls = []

    def factory(ctx, _prompt):
        calls.append(ctx.trace_id)
        return {"free_indirect_quality": 0.9}

    result = opt.replay_ab(
        mutation,
        trace_ids=["tr-1", "ghost-id"],
        pipeline_factory=factory,
    )
    # Only tr-1 has a baseline -> n_replays=1
    assert result.n_replays == 1
    # factory only runs for traces with baseline + context.
    assert calls == ["tr-1"]
    assert "skipped" in result.notes


def test_replay_ab_handles_factory_exceptions(seeded_world):
    opt = PromptOptimizer(seeded_world)
    mutation = Mutation(
        dimension="free_indirect_quality",
        prompt_path="stages/write/user.j2",
        before_text="b",
        after_text="a",
    )

    def factory(ctx, _prompt):
        raise RuntimeError("boom")

    result = opt.replay_ab(
        mutation, trace_ids=["tr-1", "tr-2"], pipeline_factory=factory,
    )
    assert result.n_replays == 0
    assert result.accept_recommended is False
    assert "no successful replays" in result.notes


# ---------------------------------------------------------------------------
# apply_mutation
# ---------------------------------------------------------------------------


def test_apply_mutation_backs_up_and_writes(tmp_path: Path, seeded_world):
    prompt = tmp_path / "write.j2"
    prompt.write_text("original")
    opt = PromptOptimizer(seeded_world, prompts_root=tmp_path)
    mutation = Mutation(
        dimension="free_indirect_quality",
        prompt_path="write.j2",
        before_text="original",
        after_text="edited",
    )
    backup = opt.apply_mutation(mutation)
    assert backup.is_file()
    assert backup.read_text() == "original"
    assert prompt.read_text() == "edited"


def test_apply_mutation_missing_prompt_raises(tmp_path: Path, seeded_world):
    opt = PromptOptimizer(seeded_world, prompts_root=tmp_path)
    mutation = Mutation(
        dimension="free_indirect_quality",
        prompt_path="does_not_exist.j2",
        before_text="a",
        after_text="b",
    )
    with pytest.raises(FileNotFoundError):
        opt.apply_mutation(mutation)
