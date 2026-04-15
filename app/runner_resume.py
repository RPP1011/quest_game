"""Pure resume-detection logic. No I/O — caller supplies the rows.

See docs/superpowers/specs/2026-04-14-quest-runner-resume-design.md
section 'Resume contract'.
"""
from __future__ import annotations

from dataclasses import dataclass


class ResumeMismatchError(Exception):
    def __init__(self, index: int, db_action: str, config_action: str) -> None:
        super().__init__(
            f"resume mismatch at action index {index}: "
            f"DB has {db_action!r}, config has {config_action!r}. "
            f"Pass --fresh or revert the action change."
        )
        self.index = index
        self.db_action = db_action
        self.config_action = config_action


class ConfigDriftError(Exception):
    """Config has fewer actions than the DB has rows."""
    def __init__(self, db_row_count: int, config_action_count: int) -> None:
        super().__init__(
            f"DB has {db_row_count} committed actions, "
            f"config has only {config_action_count}. "
            f"Pass --fresh or extend the action list."
        )
        self.db_row_count = db_row_count
        self.config_action_count = config_action_count


class WrongDatabaseError(Exception):
    """DB exists but its quest_id doesn't match the config."""
    def __init__(self, db_quest_id: str, config_quest_id: str) -> None:
        super().__init__(
            f"DB has quest_id {db_quest_id!r}, "
            f"config has {config_quest_id!r}. "
            f"Pass --fresh or point at a different --config."
        )
        self.db_quest_id = db_quest_id
        self.config_quest_id = config_quest_id


@dataclass(frozen=True)
class ResumeDecision:
    start_from: int  # 1-based update_number to run next
    skipped: int     # number of actions already done in DB


def decide_resume(
    *,
    rows: list[dict],
    actions: list[str],
    db_quest_id: str | None,
    config_quest_id: str,
) -> ResumeDecision:
    """Decide whether to resume and at what update_number.

    Parameters
    ----------
    rows:
        Narrative rows from the existing DB, ordered by update_number.
        Each row is a dict with at least ``update_number`` (int) and
        ``player_action`` (str).
    actions:
        The current run config's action list.
    db_quest_id:
        ``arc.quest_id`` from the existing DB, or ``None`` if no arc
        rows exist (e.g. truly fresh DB).
    config_quest_id:
        The current config's ``seed.quest_id``.

    Returns
    -------
    ResumeDecision with the 1-based ``start_from`` index for the next
    update and the number of already-done actions to skip.

    Examples
    --------
    >>> rows = [
    ...     {"update_number": 1, "player_action": "A1"},
    ...     {"update_number": 2, "player_action": "A2"},
    ...     {"update_number": 3, "player_action": "A3"},
    ... ]
    >>> actions = ["A1", "A2", "A3", "A4", "A5"]
    >>> decision = decide_resume(rows=rows, actions=actions,
    ...                          db_quest_id="q", config_quest_id="q")
    >>> decision.start_from
    4
    >>> decision.skipped
    3
    >>> # Caller runs ``actions[decision.skipped:]`` (i.e. ``actions[3:]``)
    >>> # with the next ``update_number`` set to ``decision.start_from``.

    Raises
    ------
    WrongDatabaseError
        DB has a quest_id that differs from config's.
    ResumeMismatchError
        A committed action in the DB does not match the corresponding
        action in the current config.
    ConfigDriftError
        DB has more committed rows than the config has actions.
    """
    if not rows:
        # Fresh or empty-DB path. Quest-id check only fires when both
        # sides actually have a value, since a freshly-bootstrapped DB
        # may or may not have an arc row yet depending on caller order.
        return ResumeDecision(start_from=1, skipped=0)

    if db_quest_id is not None and db_quest_id != config_quest_id:
        raise WrongDatabaseError(db_quest_id, config_quest_id)

    if len(rows) > len(actions):
        raise ConfigDriftError(len(rows), len(actions))

    for i, row in enumerate(rows):
        if row["player_action"] != actions[i]:
            raise ResumeMismatchError(
                index=i,
                db_action=row["player_action"],
                config_action=actions[i],
            )

    max_update = max(r["update_number"] for r in rows)
    return ResumeDecision(start_from=max_update + 1, skipped=len(rows))
