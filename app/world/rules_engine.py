"""Rule-engine: evaluates WorldRule.constraints against a StateDelta."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .delta import StateDelta, ValidationIssue
from .schema import WorldRule

if TYPE_CHECKING:
    from .state_manager import WorldStateManager


class Constraint(Protocol):
    def evaluate(
        self, delta: StateDelta, sm: "WorldStateManager"
    ) -> list[ValidationIssue]: ...


# ---------------------------------------------------------------------------
# Built-in constraint implementations
# ---------------------------------------------------------------------------

class _ForbidRelationship:
    """Emits an error for any RelChange(action='add') whose rel_type matches."""

    def __init__(self, rule_id: str, rel_type: str) -> None:
        self._rule_id = rule_id
        self._rel_type = rel_type

    def evaluate(self, delta: StateDelta, sm: "WorldStateManager") -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for op in delta.relationship_changes:
            if op.action == "add" and op.relationship.rel_type == self._rel_type:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"rule {self._rule_id}: forbidden relationship {self._rel_type}"
                        ),
                        subject=f"{op.relationship.source_id}->{op.relationship.target_id}",
                    )
                )
        return issues


class _ForbidStatusTransition:
    """Emits an error for EntityUpdates that patch status from `from_` to `to_`."""

    def __init__(self, rule_id: str, from_: str, to_: str) -> None:
        self._rule_id = rule_id
        self._from = from_
        self._to = to_

    def evaluate(self, delta: StateDelta, sm: "WorldStateManager") -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for op in delta.entity_updates:
            if "status" not in op.patch:
                continue
            new_status = op.patch["status"]
            if hasattr(new_status, "value"):
                new_status = new_status.value
            if str(new_status) != self._to:
                continue
            # Look up current status
            try:
                entity = sm.get_entity(op.id)
            except Exception:
                continue
            current_status = entity.status
            if hasattr(current_status, "value"):
                current_status = current_status.value
            if str(current_status) == self._from:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"rule {self._rule_id}: forbidden status transition "
                            f"{self._from} -> {self._to}"
                        ),
                        subject=op.id,
                    )
                )
        return issues


class _UniqueEntityName:
    """Emits an error if any EntityCreate uses a name already present."""

    def __init__(self, rule_id: str) -> None:
        self._rule_id = rule_id

    def evaluate(self, delta: StateDelta, sm: "WorldStateManager") -> list[ValidationIssue]:
        if not delta.entity_creates:
            return []
        existing_names = {e.name for e in sm.list_entities()}
        issues: list[ValidationIssue] = []
        for op in delta.entity_creates:
            if op.entity.name in existing_names:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"rule {self._rule_id}: duplicate entity name '{op.entity.name}'"
                        ),
                        subject=op.entity.id,
                    )
                )
        return issues


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_constraint(rule: WorldRule) -> Constraint | None:
    """Return a Constraint for the rule, or None if the type is unknown/missing."""
    constraints = rule.constraints
    constraint_type = constraints.get("type")
    if constraint_type == "forbid_relationship":
        return _ForbidRelationship(rule.id, rel_type=constraints["rel_type"])
    if constraint_type == "forbid_status_transition":
        return _ForbidStatusTransition(
            rule.id,
            from_=constraints["from"],
            to_=constraints["to"],
        )
    if constraint_type == "unique_entity_name":
        return _UniqueEntityName(rule.id)
    return None


# ---------------------------------------------------------------------------
# RulesEngine
# ---------------------------------------------------------------------------

class RulesEngine:
    def __init__(self, rules: list[WorldRule]) -> None:
        self._rules = rules

    def evaluate(
        self, delta: StateDelta, sm: "WorldStateManager"
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for rule in self._rules:
            if "type" not in rule.constraints:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"rule {rule.id}: missing 'type' key in constraints",
                        subject=rule.id,
                    )
                )
                continue
            constraint = _build_constraint(rule)
            if constraint is None:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=(
                            f"rule {rule.id}: unknown constraint type "
                            f"'{rule.constraints['type']}'"
                        ),
                        subject=rule.id,
                    )
                )
                continue
            issues.extend(constraint.evaluate(delta, sm))
        return issues
