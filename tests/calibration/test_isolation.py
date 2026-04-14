"""Enforce that app.calibration does not depend on app.engine.pipeline.

Keeps the calibration scorer decoupled from the rerank scorer so weight
changes to one don't silently drift the other.
"""
from __future__ import annotations

import importlib
import sys


def test_calibration_does_not_import_pipeline():
    for name in list(sys.modules):
        if name == "app.engine.pipeline" or name.startswith("app.engine.pipeline."):
            del sys.modules[name]

    importlib.import_module("app.calibration")
    importlib.import_module("app.calibration.harness")
    importlib.import_module("app.calibration.scorer")
    importlib.import_module("app.calibration.heuristics")
    importlib.import_module("app.calibration.judges")
    importlib.import_module("app.calibration.loader")
    importlib.import_module("app.calibration.report")

    leaked = [
        name for name in sys.modules
        if name == "app.engine.pipeline" or name.startswith("app.engine.pipeline.")
    ]
    assert not leaked, (
        f"app.calibration transitively imports pipeline: {leaked}. "
        "Keep the rerank scorer and calibration scorer decoupled."
    )
