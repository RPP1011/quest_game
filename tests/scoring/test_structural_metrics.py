from __future__ import annotations
import pytest
from app.scoring.structural_metrics import (
    syntactic_compression_ratio, mtld, mtld_forward,
)


def test_syntactic_cr_repetitive():
    prose = ". ".join(["She walked", "She talked", "She smiled", "She frowned"] * 10)
    cr = syntactic_compression_ratio(prose)
    assert cr < 0.40


def test_syntactic_cr_varied():
    prose = (
        "The rain fell steadily. Under the awning, two men argued about "
        "the price of salt. 'It's robbery,' said the first, a short man "
        "with calloused hands. His companion, taller by a head and broader "
        "by a life of hauling nets, simply shrugged. What could you do? "
        "The caravans set the rates. The fishermen paid them."
    )
    cr = syntactic_compression_ratio(prose)
    assert cr > 0.35


def test_mtld_low_diversity():
    prose = " ".join(["the dog sat on the mat"] * 20)
    score = mtld(prose)
    assert score < 30


def test_mtld_high_diversity():
    prose = (
        "Tristan navigated the labyrinthine corridors beneath the citadel. "
        "Every junction presented a bifurcation: left toward the armory's "
        "flickering braziers, right toward the subterranean cisterns where "
        "moisture beaded on ancient stonework. He chose instinctively, "
        "following the draft that carried the metallic tang of freshly "
        "forged iron, a scent as familiar as his own heartbeat."
    )
    score = mtld(prose)
    assert score > 50


def test_mtld_forward_basic():
    words = ["the", "dog", "sat", "on", "the", "mat", "the", "cat"]
    score = mtld_forward(words, ttr_threshold=0.72)
    assert score > 0
