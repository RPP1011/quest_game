"""Foreshadowing retriever (Wave 4b).

Surfaces ripe foreshadowing hooks to the dramatic and craft planners so
a scene can either pay off a planted-and-waiting hook or weave its
payoff into prose. No embeddings: ripeness is a structured computation
over the ``foreshadowing`` rows owned by the quest.

Ranking
-------
Every candidate hook passes through a two-stage pipeline:

1. **Status filter.** Only hooks in status ``PLANTED`` or ``REFERENCED``
   are eligible — ``PAID_OFF`` and ``ABANDONED`` hooks have nothing more
   to contribute. (``REFERENCED`` is the local equivalent of the spec's
   ``PARTIAL`` — a hook the narrative has touched but not yet resolved.)
2. **Ripeness bucket.** Using ``current_update`` from the query filters:

   * **Overdue** (score 1.0) — ``current_update`` has passed the hook's
     ``target_update_max`` window (when a window exists).
   * **Ripe** (score 0.7) — ``current_update`` sits inside
     ``[target_update_min, target_update_max]`` (when both bounds exist).
   * **Aging** (score 0.3) — no target window *and* the hook has been
     planted for at least 5 updates. Eligible but not urgent.
   * **Fresh** — dropped. Too early to tee up a payoff.

Ties inside a bucket are broken by ``planted_at_update`` ascending
(oldest planted hook wins) and then by ``id`` so ordering is
deterministic.

Schema note
-----------
The current :class:`app.world.schema.ForeshadowingHook` model does not
store ``target_update_min``/``target_update_max`` fields directly, but
the retriever reads them via :func:`getattr` so that if the schema
later grows these columns the ripeness logic picks them up without
another wave. Hooks missing a target window naturally fall into the
*Aging* branch once they're old enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .interface import Query, Result

if TYPE_CHECKING:
    from app.world.schema import ForeshadowingHook
    from app.world.state_manager import WorldStateManager


# Eligible (non-terminal) hook statuses. Spec wording "PLANTED/PARTIAL"
# maps onto the live enum as PLANTED/REFERENCED — the latter is the
# project's name for a hook that's been narratively touched but not yet
# paid off. The actual ``HookStatus`` values are read lazily inside
# :func:`_eligible_statuses` so this module stays import-cycle clean.
def _eligible_statuses() -> frozenset:
    from app.world.schema import HookStatus

    return frozenset({HookStatus.PLANTED, HookStatus.REFERENCED})

# Minimum age (in updates) for a hook with no target window to count as
# "aging" — i.e. ripe enough for a planner to consider paying off.
_AGING_THRESHOLD = 5

# Per-bucket ripeness scores (higher = more urgent).
_SCORE_OVERDUE = 1.0
_SCORE_RIPE = 0.7
_SCORE_AGING = 0.3


@dataclass(frozen=True)
class _ScoredHook:
    hook: Any  # ForeshadowingHook — typed weakly to avoid a top-level import cycle.
    score: float
    ripeness_status: str
    target_update_min: int | None
    target_update_max: int | None


class ForeshadowingRetriever:
    """Serve ripe foreshadowing hooks from a quest's state.

    Parameters
    ----------
    world:
        The :class:`WorldStateManager` backing the current quest.
    quest_id:
        The quest scope — only used when building :class:`Result`
        ``source_id`` values (the local ``foreshadowing`` table does
        not carry a ``quest_id`` column today).
    """

    def __init__(self, world: "WorldStateManager", quest_id: str) -> None:
        self._world = world
        self._quest_id = quest_id

    # -- Public API ----------------------------------------------------

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        """Return up to ``k`` ripe hooks ranked by ripeness then age.

        ``query.filters["current_update"]`` drives the ripeness window;
        if absent, ``0`` is assumed (which effectively drops every hook
        into the *Fresh* bucket so nothing surfaces — planners should
        always pass the current update number).
        """
        current_update = int(query.filters.get("current_update", 0) or 0)
        eligible = _eligible_statuses()

        hooks = self._load_hooks()
        scored: list[_ScoredHook] = []
        for hook in hooks:
            if hook.status not in eligible:
                continue
            entry = self._score(hook, current_update)
            if entry is None:
                continue
            scored.append(entry)

        scored.sort(
            key=lambda s: (
                -s.score,
                s.hook.planted_at_update,
                s.hook.id,
            )
        )

        return [self._to_result(s) for s in scored[:k]]

    # -- Internals -----------------------------------------------------

    def _load_hooks(self) -> list[Any]:
        """Read every foreshadowing hook from the world store.

        The live state manager has no ``list_foreshadowing`` helper, so
        we lean on :meth:`WorldStateManager.snapshot` which already
        materialises every hook in deterministic id order.
        """
        try:
            snap = self._world.snapshot()
        except Exception:
            return []
        return list(snap.foreshadowing)

    @staticmethod
    def _target_bounds(
        hook: Any,
    ) -> tuple[int | None, int | None]:
        """Read optional ``target_update_min``/``target_update_max`` fields.

        The current schema does not declare these attributes, so
        ``getattr`` returns ``None`` and the hook drops into the
        "no target window" branch. A future schema that adds these
        columns will light up the *Overdue*/*Ripe* branches without
        any changes here.
        """
        tmin = getattr(hook, "target_update_min", None)
        tmax = getattr(hook, "target_update_max", None)
        return tmin, tmax

    @classmethod
    def _score(
        cls, hook: Any, current_update: int
    ) -> _ScoredHook | None:
        """Bucket a single hook by ripeness; return ``None`` if *Fresh*."""
        tmin, tmax = cls._target_bounds(hook)

        if tmax is not None and current_update > tmax:
            return _ScoredHook(
                hook=hook,
                score=_SCORE_OVERDUE,
                ripeness_status="overdue",
                target_update_min=tmin,
                target_update_max=tmax,
            )

        if (
            tmin is not None
            and tmax is not None
            and tmin <= current_update <= tmax
        ):
            return _ScoredHook(
                hook=hook,
                score=_SCORE_RIPE,
                ripeness_status="ripe",
                target_update_min=tmin,
                target_update_max=tmax,
            )

        no_window = tmin is None and tmax is None
        if no_window and current_update >= hook.planted_at_update + _AGING_THRESHOLD:
            return _ScoredHook(
                hook=hook,
                score=_SCORE_AGING,
                ripeness_status="aging",
                target_update_min=None,
                target_update_max=None,
            )

        return None

    def _to_result(self, scored: _ScoredHook) -> Result:
        hook = scored.hook
        status_value = (
            hook.status.value if hasattr(hook.status, "value") else hook.status
        )
        metadata: dict[str, Any] = {
            "hook_id": hook.id,
            "status": status_value,
            "planted_at_update": hook.planted_at_update,
            "target_update_min": scored.target_update_min,
            "target_update_max": scored.target_update_max,
            # ``payoff_description`` is the spec's name; the local schema
            # calls it ``payoff_target``. Expose both keys so template
            # code and future schema alignment both work.
            "payoff_description": hook.payoff_target,
            "payoff_target": hook.payoff_target,
            "ripeness_status": scored.ripeness_status,
        }
        return Result(
            source_id=f"hook/{self._quest_id}/{hook.id}",
            text=hook.description,
            score=scored.score,
            metadata=metadata,
        )
