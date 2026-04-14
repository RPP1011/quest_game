"""Calibration corpus harness.

Measures how closely our scoring dimensions match a hand-curated ground truth
across 16 works of fiction (12 novels + 4 quests). Independent of the rerank
scorer — does not share weights or formulas with ``app.engine.pipeline``.

Public API:
    load_manifest(path) -> Manifest
    Harness(manifest, passages_dir, client=None).run() -> Report
"""
from .loader import Manifest, Work, Passage, load_manifest, init_passage_hashes
from .harness import Harness, PassageScore, Report
from .scorer import critic_score, clip01

__all__ = [
    "Manifest",
    "Work",
    "Passage",
    "load_manifest",
    "init_passage_hashes",
    "Harness",
    "PassageScore",
    "Report",
    "critic_score",
    "clip01",
]
