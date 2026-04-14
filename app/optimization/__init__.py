"""Day 7: Prompt Optimizer + Example Curator.

A closed self-improvement loop on top of the Day 2 scorecard substrate.

- :class:`PromptOptimizer` scans accumulated scorecards, identifies the
  weakest dimension(s), proposes a single targeted prompt mutation, runs
  a replay-based A/B against historical traces, and (when asked) applies
  the winning mutation with a timestamped backup.
- :class:`ExampleCurator` mines the best- and worst-scoring prose from
  the scorecard history into good-pattern (``data/craft/examples``) and
  anti-pattern (``data/craft/anti_patterns``) banks that the craft
  library and writer prompts can pull from.

Everything here is default-off: no automatic prompt edits, no auto-fire
on commit. A caller (CLI or test) invokes the read paths; only
:meth:`PromptOptimizer.apply_mutation` touches the filesystem, and only
when the human operator decides to.
"""
from __future__ import annotations

from .optimizer import (
    ABResult,
    Mutation,
    PromptOptimizer,
    WeakDim,
)
from .curator import (
    ExampleCandidate,
    ExampleCurator,
)

__all__ = [
    "ABResult",
    "ExampleCandidate",
    "ExampleCurator",
    "Mutation",
    "PromptOptimizer",
    "WeakDim",
]
