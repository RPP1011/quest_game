from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from typing import Any
from .schema import (
    ArcPosition,
    Entity,
    EntityType,
    Expectation,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    OpenQuestion,
    PlotThread,
    QuestArcState,
    ReaderState,
    Relationship,
    ThreadStatus,
    TimelineEvent,
    WorldRule,
)
from .delta import (
    FSUpdate,
    PTUpdate,
    RelChange,
    StateDelta,
    ValidationIssue,
    ValidationResult,
)
from .rules_engine import RulesEngine
from .retcon import RetconSpec, RetconResult


class WorldStateError(Exception):
    pass


class EntityNotFoundError(WorldStateError):
    pass


class RelationshipNotFoundError(WorldStateError):
    pass


class InvalidDeltaError(WorldStateError):
    def __init__(self, result: ValidationResult) -> None:
        super().__init__("; ".join(i.message for i in result.issues if i.severity == "error"))
        self.result = result


@dataclass(frozen=True)
class WorldSnapshot:
    entities: list[Entity]
    relationships: list[Relationship]
    rules: list[WorldRule]
    timeline: list[TimelineEvent]
    narrative: list[NarrativeRecord]
    foreshadowing: list[ForeshadowingHook]
    plot_threads: list[PlotThread]


def _row_to_entity(row: sqlite3.Row) -> Entity:
    return Entity(
        id=row["id"],
        entity_type=row["entity_type"],
        name=row["name"],
        data=json.loads(row["data"]),
        status=row["status"],
        last_referenced_update=row["last_referenced_update"],
        created_at_update=row["created_at_update"],
    )


def _row_to_narrative(row: sqlite3.Row) -> NarrativeRecord:
    return NarrativeRecord(
        update_number=row["update_number"],
        raw_text=row["raw_text"],
        summary=row["summary"],
        chapter_id=row["chapter_id"],
        state_diff=json.loads(row["state_diff"]),
        player_action=row["player_action"],
        pipeline_trace_id=row["pipeline_trace_id"],
    )


def _row_to_hook(row: sqlite3.Row) -> ForeshadowingHook:
    return ForeshadowingHook(
        id=row["id"],
        description=row["description"],
        planted_at_update=row["planted_at_update"],
        payoff_target=row["payoff_target"],
        status=row["status"],
        paid_off_at_update=row["paid_off_at_update"],
        references=json.loads(row["refs"]),
    )


def _row_to_thread(row: sqlite3.Row) -> PlotThread:
    return PlotThread(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        status=row["status"],
        involved_entities=json.loads(row["involved_entities"]),
        arc_position=row["arc_position"],
        priority=row["priority"],
    )


class WorldStateManager:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ---- entities ----

    def create_entity(self, entity: Entity) -> None:
        self._conn.execute(
            "INSERT INTO entities(id, entity_type, name, data, status, "
            "last_referenced_update, created_at_update) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entity.id,
                entity.entity_type.value,
                entity.name,
                json.dumps(entity.data),
                entity.status.value,
                entity.last_referenced_update,
                entity.created_at_update,
            ),
        )
        self._conn.commit()

    def get_entity(self, entity_id: str) -> Entity:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            raise EntityNotFoundError(entity_id)
        return _row_to_entity(row)

    def list_entities(self, entity_type: EntityType | None = None) -> list[Entity]:
        if entity_type is None:
            rows = self._conn.execute("SELECT * FROM entities ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? ORDER BY id",
                (entity_type.value,),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def update_entity(self, entity_id: str, patch: dict[str, Any]) -> None:
        current = self.get_entity(entity_id)  # raises if missing
        merged_data = current.data
        if "data" in patch:
            merged_data = {**current.data, **patch["data"]}
        fields = {
            "name": patch.get("name", current.name),
            "status": patch.get("status", current.status.value if hasattr(current.status, "value") else current.status),
            "last_referenced_update": patch.get(
                "last_referenced_update", current.last_referenced_update
            ),
            "data": json.dumps(merged_data),
        }
        self._conn.execute(
            "UPDATE entities SET name=?, status=?, last_referenced_update=?, data=?, "
            "modified_at=CURRENT_TIMESTAMP WHERE id=?",
            (fields["name"], fields["status"], fields["last_referenced_update"],
             fields["data"], entity_id),
        )
        self._conn.commit()

    # ---- relationships ----

    def add_relationship(self, rel: Relationship) -> None:
        self._conn.execute(
            "INSERT INTO relationships(source_id, target_id, rel_type, data, established_at_update) "
            "VALUES (?, ?, ?, ?, ?)",
            (rel.source_id, rel.target_id, rel.rel_type, json.dumps(rel.data),
             rel.established_at_update),
        )
        self._conn.commit()

    def remove_relationship(self, source_id: str, target_id: str, rel_type: str) -> None:
        cur = self._conn.execute(
            "DELETE FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
            (source_id, target_id, rel_type),
        )
        if cur.rowcount == 0:
            raise RelationshipNotFoundError(f"{source_id} -{rel_type}-> {target_id}")
        self._conn.commit()

    def modify_relationship(
        self, source_id: str, target_id: str, rel_type: str, data_patch: dict[str, Any]
    ) -> None:
        row = self._conn.execute(
            "SELECT data FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
            (source_id, target_id, rel_type),
        ).fetchone()
        if row is None:
            raise RelationshipNotFoundError(f"{source_id} -{rel_type}-> {target_id}")
        merged = {**json.loads(row["data"]), **data_patch}
        self._conn.execute(
            "UPDATE relationships SET data=? WHERE source_id=? AND target_id=? AND rel_type=?",
            (json.dumps(merged), source_id, target_id, rel_type),
        )
        self._conn.commit()

    def list_relationships(self, source_id: str | None = None) -> list[Relationship]:
        if source_id is None:
            rows = self._conn.execute("SELECT * FROM relationships").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM relationships WHERE source_id=?", (source_id,)
            ).fetchall()
        return [
            Relationship(
                source_id=r["source_id"],
                target_id=r["target_id"],
                rel_type=r["rel_type"],
                data=json.loads(r["data"]),
                established_at_update=r["established_at_update"],
            )
            for r in rows
        ]

    # ---- world rules ----

    def add_rule(self, rule: WorldRule) -> None:
        self._conn.execute(
            "INSERT INTO world_rules(id, category, description, constraints, established_at_update) "
            "VALUES (?, ?, ?, ?, ?)",
            (rule.id, rule.category, rule.description, json.dumps(rule.constraints),
             rule.established_at_update),
        )
        self._conn.commit()

    def list_rules(self) -> list[WorldRule]:
        rows = self._conn.execute("SELECT * FROM world_rules ORDER BY id").fetchall()
        return [
            WorldRule(
                id=r["id"], category=r["category"], description=r["description"],
                constraints=json.loads(r["constraints"]),
                established_at_update=r["established_at_update"],
            )
            for r in rows
        ]

    # ---- timeline ----

    def append_timeline_event(self, event: TimelineEvent) -> None:
        self._conn.execute(
            "INSERT INTO timeline(update_number, event_index, description, involved_entities, causal_links) "
            "VALUES (?, ?, ?, ?, ?)",
            (event.update_number, event.event_index, event.description,
             json.dumps(event.involved_entities), json.dumps(event.causal_links)),
        )
        self._conn.commit()

    def list_timeline(self, update_number: int | None = None) -> list[TimelineEvent]:
        if update_number is None:
            rows = self._conn.execute(
                "SELECT * FROM timeline ORDER BY update_number, event_index"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM timeline WHERE update_number=? ORDER BY event_index",
                (update_number,),
            ).fetchall()
        return [
            TimelineEvent(
                update_number=r["update_number"],
                event_index=r["event_index"],
                description=r["description"],
                involved_entities=json.loads(r["involved_entities"]),
                causal_links=[tuple(x) for x in json.loads(r["causal_links"])],
            )
            for r in rows
        ]

    # ---- narrative ----

    def write_narrative(self, record: NarrativeRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO narrative"
            "(update_number, raw_text, summary, chapter_id, state_diff, player_action, pipeline_trace_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.update_number, record.raw_text, record.summary, record.chapter_id,
             json.dumps(record.state_diff), record.player_action, record.pipeline_trace_id),
        )
        self._conn.commit()

    def get_narrative(self, update_number: int) -> NarrativeRecord:
        row = self._conn.execute(
            "SELECT * FROM narrative WHERE update_number=?", (update_number,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no narrative for update {update_number}")
        return _row_to_narrative(row)

    def list_narrative(self, limit: int = 50) -> list[NarrativeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM narrative ORDER BY update_number LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_narrative(r) for r in rows]

    # ---- foreshadowing ----

    def add_foreshadowing(self, hook: ForeshadowingHook) -> None:
        self._conn.execute(
            "INSERT INTO foreshadowing(id, description, planted_at_update, payoff_target, "
            "status, paid_off_at_update, refs) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (hook.id, hook.description, hook.planted_at_update, hook.payoff_target,
             hook.status.value, hook.paid_off_at_update, json.dumps(hook.references)),
        )
        self._conn.commit()

    def get_foreshadowing(self, hook_id: str) -> ForeshadowingHook:
        row = self._conn.execute(
            "SELECT * FROM foreshadowing WHERE id=?", (hook_id,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no foreshadowing hook {hook_id}")
        return _row_to_hook(row)

    def update_foreshadowing(self, hook_id: str, patch: dict[str, Any]) -> None:
        current = self.get_foreshadowing(hook_id)
        new_status = patch.get("status", current.status)
        if hasattr(new_status, "value"):
            new_status = new_status.value
        self._conn.execute(
            "UPDATE foreshadowing SET status=?, paid_off_at_update=?, refs=? WHERE id=?",
            (
                new_status,
                patch.get("paid_off_at_update", current.paid_off_at_update),
                json.dumps(patch.get("references", current.references)),
                hook_id,
            ),
        )
        self._conn.commit()

    # ---- plot threads ----

    def add_plot_thread(self, pt: PlotThread) -> None:
        self._conn.execute(
            "INSERT INTO plot_threads(id, name, description, status, involved_entities, "
            "arc_position, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pt.id, pt.name, pt.description, pt.status.value,
             json.dumps(pt.involved_entities), pt.arc_position.value, pt.priority),
        )
        self._conn.commit()

    def get_plot_thread(self, pt_id: str) -> PlotThread:
        row = self._conn.execute(
            "SELECT * FROM plot_threads WHERE id=?", (pt_id,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no plot thread {pt_id}")
        return _row_to_thread(row)

    def update_plot_thread(self, pt_id: str, patch: dict[str, Any]) -> None:
        current = self.get_plot_thread(pt_id)

        def _v(x: Any) -> Any:
            return x.value if hasattr(x, "value") else x

        self._conn.execute(
            "UPDATE plot_threads SET name=?, description=?, status=?, involved_entities=?, "
            "arc_position=?, priority=? WHERE id=?",
            (
                patch.get("name", current.name),
                patch.get("description", current.description),
                _v(patch.get("status", current.status)),
                json.dumps(patch.get("involved_entities", current.involved_entities)),
                _v(patch.get("arc_position", current.arc_position)),
                patch.get("priority", current.priority),
                pt_id,
            ),
        )
        self._conn.commit()

    def list_plot_threads(self) -> list[PlotThread]:
        rows = self._conn.execute(
            "SELECT * FROM plot_threads ORDER BY priority DESC, id"
        ).fetchall()
        return [_row_to_thread(r) for r in rows]

    def validate_delta(self, delta: StateDelta) -> ValidationResult:
        issues: list[ValidationIssue] = []

        existing_ids = {
            row["id"] for row in self._conn.execute("SELECT id FROM entities").fetchall()
        }
        planned_new_ids: set[str] = set()

        for op in delta.entity_creates:
            if op.entity.id in existing_ids or op.entity.id in planned_new_ids:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"entity create conflicts with existing id: {op.entity.id}",
                    subject=op.entity.id,
                ))
            else:
                planned_new_ids.add(op.entity.id)

        valid_ids = existing_ids | planned_new_ids

        for op in delta.entity_updates:
            if op.id not in existing_ids:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"entity update references missing id: {op.id}",
                    subject=op.id,
                ))
                continue
            row = self._conn.execute(
                "SELECT status FROM entities WHERE id=?", (op.id,)
            ).fetchone()
            if row and row["status"] in ("deceased", "destroyed") and "status" not in op.patch:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=f"acting on {row['status']} entity: {op.id}",
                    subject=op.id,
                ))

        for op in delta.relationship_changes:
            r = op.relationship
            if op.action == "add":
                for eid in (r.source_id, r.target_id):
                    if eid not in valid_ids:
                        issues.append(ValidationIssue(
                            severity="error",
                            message=f"relationship endpoint missing: {eid}",
                            subject=eid,
                        ))
            else:  # remove / modify
                exists = self._conn.execute(
                    "SELECT 1 FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
                    (r.source_id, r.target_id, r.rel_type),
                ).fetchone() is not None
                if not exists:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=f"relationship not found: {r.source_id} -{r.rel_type}-> {r.target_id}",
                        subject=f"{r.source_id}->{r.target_id}",
                    ))

        for op in delta.foreshadowing_updates:
            if self._conn.execute(
                "SELECT 1 FROM foreshadowing WHERE id=?", (op.id,)
            ).fetchone() is None:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"foreshadowing hook not found: {op.id}",
                    subject=op.id,
                ))

        for op in delta.plot_thread_updates:
            if self._conn.execute(
                "SELECT 1 FROM plot_threads WHERE id=?", (op.id,)
            ).fetchone() is None:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"plot thread not found: {op.id}",
                    subject=op.id,
                ))

        engine = RulesEngine(self.list_rules())
        issues.extend(engine.evaluate(delta, self))

        return ValidationResult(issues=issues)

    def apply_delta(self, delta: StateDelta, update_number: int) -> None:
        result = self.validate_delta(delta)
        if not result.ok:
            raise InvalidDeltaError(result)

        touched_ids: set[str] = set()
        try:
            self._conn.execute("BEGIN")

            for op in delta.entity_creates:
                e = op.entity
                e_with_update = e.model_copy(update={
                    "created_at_update": e.created_at_update or update_number,
                    "last_referenced_update": e.last_referenced_update or update_number,
                })
                self._conn.execute(
                    "INSERT INTO entities(id, entity_type, name, data, status, "
                    "last_referenced_update, created_at_update) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        e_with_update.id, e_with_update.entity_type.value, e_with_update.name,
                        json.dumps(e_with_update.data), e_with_update.status.value,
                        e_with_update.last_referenced_update, e_with_update.created_at_update,
                    ),
                )
                touched_ids.add(e.id)

            for op in delta.entity_updates:
                current = self.get_entity(op.id)
                merged_data = {**current.data, **op.patch.get("data", {})}
                new_status = op.patch.get("status", current.status)
                if hasattr(new_status, "value"):
                    new_status = new_status.value
                self._conn.execute(
                    "UPDATE entities SET name=?, status=?, last_referenced_update=?, data=?, "
                    "modified_at=CURRENT_TIMESTAMP WHERE id=?",
                    (
                        op.patch.get("name", current.name),
                        new_status,
                        update_number,
                        json.dumps(merged_data),
                        op.id,
                    ),
                )
                touched_ids.add(op.id)

            for op in delta.relationship_changes:
                r = op.relationship
                if op.action == "add":
                    self._conn.execute(
                        "INSERT INTO relationships(source_id, target_id, rel_type, data, established_at_update) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (r.source_id, r.target_id, r.rel_type, json.dumps(r.data),
                         r.established_at_update or update_number),
                    )
                elif op.action == "remove":
                    self._conn.execute(
                        "DELETE FROM relationships WHERE source_id=? AND target_id=? AND rel_type=?",
                        (r.source_id, r.target_id, r.rel_type),
                    )
                else:  # modify
                    self._conn.execute(
                        "UPDATE relationships SET data=? WHERE source_id=? AND target_id=? AND rel_type=?",
                        (json.dumps(r.data), r.source_id, r.target_id, r.rel_type),
                    )
                touched_ids.update([r.source_id, r.target_id])

            for op in delta.timeline_events:
                ev = op.event
                self._conn.execute(
                    "INSERT INTO timeline(update_number, event_index, description, involved_entities, causal_links) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ev.update_number, ev.event_index, ev.description,
                     json.dumps(ev.involved_entities), json.dumps(ev.causal_links)),
                )
                touched_ids.update(ev.involved_entities)

            for op in delta.foreshadowing_updates:
                current = self.get_foreshadowing(op.id)
                refs = list(current.references)
                if op.add_reference is not None and op.add_reference not in refs:
                    refs.append(op.add_reference)
                self._conn.execute(
                    "UPDATE foreshadowing SET status=?, paid_off_at_update=?, refs=? WHERE id=?",
                    (op.new_status, op.paid_off_at_update or current.paid_off_at_update,
                     json.dumps(refs), op.id),
                )

            for op in delta.plot_thread_updates:
                current = self.get_plot_thread(op.id)

                def _v(x: Any) -> Any:
                    return x.value if hasattr(x, "value") else x

                self._conn.execute(
                    "UPDATE plot_threads SET name=?, description=?, status=?, involved_entities=?, "
                    "arc_position=?, priority=? WHERE id=?",
                    (
                        op.patch.get("name", current.name),
                        op.patch.get("description", current.description),
                        _v(op.patch.get("status", current.status)),
                        json.dumps(op.patch.get("involved_entities", current.involved_entities)),
                        _v(op.patch.get("arc_position", current.arc_position)),
                        op.patch.get("priority", current.priority),
                        op.id,
                    ),
                )

            # Bump last_referenced_update for entities touched but not explicitly updated
            for eid in touched_ids:
                self._conn.execute(
                    "UPDATE entities SET last_referenced_update=? "
                    "WHERE id=? AND (last_referenced_update IS NULL OR last_referenced_update < ?)",
                    (update_number, eid, update_number),
                )

            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def rollback(self, to_update: int) -> None:
        c = self._conn
        try:
            c.execute("BEGIN")
            c.execute("DELETE FROM timeline WHERE update_number > ?", (to_update,))
            c.execute("DELETE FROM narrative WHERE update_number > ?", (to_update,))
            c.execute("DELETE FROM entities WHERE created_at_update > ?", (to_update,))
            c.execute("DELETE FROM relationships WHERE established_at_update > ?", (to_update,))

            # Recalculate last_referenced_update for remaining entities.
            # For each entity, find the max timeline reference <= to_update,
            # falling back to created_at_update.
            for row in c.execute("SELECT id, created_at_update FROM entities").fetchall():
                eid = row["id"]
                tl_row = c.execute(
                    "SELECT MAX(update_number) AS mx FROM timeline "
                    "WHERE update_number <= ? AND involved_entities LIKE ?",
                    (to_update, f'%"{eid}"%'),
                ).fetchone()
                mx = tl_row["mx"] if tl_row and tl_row["mx"] is not None else None
                new_ref = mx if mx is not None else row["created_at_update"]
                c.execute(
                    "UPDATE entities SET last_referenced_update=? WHERE id=?",
                    (new_ref, eid),
                )

            # Foreshadowing: revert payoffs that happened after to_update
            for row in c.execute("SELECT id, paid_off_at_update, refs FROM foreshadowing").fetchall():
                refs = json.loads(row["refs"])
                filtered_refs = [r for r in refs if r <= to_update]
                paid_off = row["paid_off_at_update"]
                if paid_off is not None and paid_off > to_update:
                    # Revert payoff: restore to planted (pre-payoff state)
                    c.execute(
                        "UPDATE foreshadowing SET status=?, paid_off_at_update=NULL, refs=? WHERE id=?",
                        ("planted", json.dumps(filtered_refs), row["id"]),
                    )
                elif filtered_refs != refs:
                    c.execute(
                        "UPDATE foreshadowing SET refs=? WHERE id=?",
                        (json.dumps(filtered_refs), row["id"]),
                    )

            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def retcon(self, spec: RetconSpec) -> RetconResult:
        # Determine the new update number (retcons go at the end)
        row = self._conn.execute(
            "SELECT MAX(update_number) AS mx FROM timeline"
        ).fetchone()
        tl_max = row["mx"] if row and row["mx"] is not None else 0

        nar_row = self._conn.execute(
            "SELECT MAX(update_number) AS mx FROM narrative"
        ).fetchone()
        nar_max = nar_row["mx"] if nar_row and nar_row["mx"] is not None else 0

        new_update_number = max(tl_max, nar_max, spec.target_update) + 1

        # Apply the delta (raises InvalidDeltaError if invalid)
        self.apply_delta(spec.delta, update_number=new_update_number)

        # Collect touched entity names (for substring match) and touched entity ids
        # (for relationship source/target matching)
        touched_entity_ids: set[str] = set()
        for op in spec.delta.entity_creates:
            touched_entity_ids.add(op.entity.id)
        for op in spec.delta.entity_updates:
            touched_entity_ids.add(op.id)
        for op in spec.delta.relationship_changes:
            touched_entity_ids.add(op.relationship.source_id)
            touched_entity_ids.add(op.relationship.target_id)

        # Resolve entity names for touched ids
        touched_names: set[str] = set()
        for eid in touched_entity_ids:
            try:
                entity = self.get_entity(eid)
                touched_names.add(entity.name)
            except EntityNotFoundError:
                pass

        # Build set of (source_id, target_id) pairs from changed relationships
        touched_rel_pairs: set[tuple[str, str]] = {
            (op.relationship.source_id, op.relationship.target_id)
            for op in spec.delta.relationship_changes
        }

        # Find narrative records >= target_update that mention touched entities
        affected_update_numbers: list[int] = []
        all_narrative = self._conn.execute(
            "SELECT update_number, raw_text FROM narrative WHERE update_number >= ?",
            (spec.target_update,),
        ).fetchall()
        for nar_row in all_narrative:
            raw_text = nar_row["raw_text"] or ""
            # Check entity name substring match
            if any(name in raw_text for name in touched_names):
                affected_update_numbers.append(nar_row["update_number"])
                continue
            # Check relationship source/target id substring match
            if any(sid in raw_text or tid in raw_text for sid, tid in touched_rel_pairs):
                affected_update_numbers.append(nar_row["update_number"])

        # Create a timeline event describing the retcon
        # Find next event_index for the new update
        ei_row = self._conn.execute(
            "SELECT MAX(event_index) AS mx FROM timeline WHERE update_number=?",
            (new_update_number,),
        ).fetchone()
        next_event_index = (ei_row["mx"] + 1) if ei_row and ei_row["mx"] is not None else 0

        from .schema import TimelineEvent
        retcon_event = TimelineEvent(
            update_number=new_update_number,
            event_index=next_event_index,
            description=f"[retcon] {spec.reason}",
            involved_entities=list(touched_entity_ids),
            causal_links=[],
        )
        self.append_timeline_event(retcon_event)

        return RetconResult(
            applied_update=spec.target_update,
            new_update_number=new_update_number,
            affected_narrative=sorted(affected_update_numbers),
        )

    # ---- arcs ----

    def upsert_arc(self, state: QuestArcState) -> None:
        self._conn.execute(
            "INSERT INTO arcs(quest_id, arc_id, structure_id, scale, current_phase_index, "
            "phase_progress, tension_observed, last_directive) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(quest_id, arc_id) DO UPDATE SET "
            "structure_id=excluded.structure_id, "
            "scale=excluded.scale, "
            "current_phase_index=excluded.current_phase_index, "
            "phase_progress=excluded.phase_progress, "
            "tension_observed=excluded.tension_observed, "
            "last_directive=excluded.last_directive",
            (
                state.quest_id,
                state.arc_id,
                state.structure_id,
                state.scale,
                state.current_phase_index,
                state.phase_progress,
                json.dumps(state.tension_observed),
                json.dumps(state.last_directive) if state.last_directive is not None else None,
            ),
        )
        self._conn.commit()

    def get_arc(self, quest_id: str, arc_id: str) -> QuestArcState:
        row = self._conn.execute(
            "SELECT * FROM arcs WHERE quest_id=? AND arc_id=?", (quest_id, arc_id)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no arc {arc_id!r} for quest {quest_id!r}")
        return QuestArcState(
            quest_id=row["quest_id"],
            arc_id=row["arc_id"],
            structure_id=row["structure_id"],
            scale=row["scale"],
            current_phase_index=row["current_phase_index"],
            phase_progress=row["phase_progress"],
            tension_observed=[tuple(x) for x in json.loads(row["tension_observed"])],
            last_directive=json.loads(row["last_directive"]) if row["last_directive"] is not None else None,
        )

    def list_arcs(self, quest_id: str) -> list[QuestArcState]:
        rows = self._conn.execute(
            "SELECT * FROM arcs WHERE quest_id=? ORDER BY arc_id", (quest_id,)
        ).fetchall()
        return [
            QuestArcState(
                quest_id=r["quest_id"],
                arc_id=r["arc_id"],
                structure_id=r["structure_id"],
                scale=r["scale"],
                current_phase_index=r["current_phase_index"],
                phase_progress=r["phase_progress"],
                tension_observed=[tuple(x) for x in json.loads(r["tension_observed"])],
                last_directive=json.loads(r["last_directive"]) if r["last_directive"] is not None else None,
            )
            for r in rows
        ]

    def record_tension(self, quest_id: str, arc_id: str, update_number: int, value: float) -> None:
        state = self.get_arc(quest_id, arc_id)
        new_tension = list(state.tension_observed) + [(update_number, value)]
        self._conn.execute(
            "UPDATE arcs SET tension_observed=? WHERE quest_id=? AND arc_id=?",
            (json.dumps(new_tension), quest_id, arc_id),
        )
        self._conn.commit()

    # ---- reader state ----

    def get_reader_state(self, quest_id: str) -> ReaderState:
        """Load reader state for a quest; returns a fresh default if absent."""
        row = self._conn.execute(
            "SELECT * FROM reader_state WHERE quest_id=?", (quest_id,)
        ).fetchone()
        if row is None:
            return ReaderState(quest_id=quest_id)
        return ReaderState(
            quest_id=row["quest_id"],
            known_fact_ids=json.loads(row["known_fact_ids"]),
            open_questions=[OpenQuestion.model_validate(q) for q in json.loads(row["open_questions"])],
            expectations=[Expectation.model_validate(e) for e in json.loads(row["expectations"])],
            attachment_levels=json.loads(row["attachment_levels"]),
            current_emotional_valence=row["current_emotional_valence"],
            updates_since_major_event=row["updates_since_major_event"],
            updates_since_revelation=row["updates_since_revelation"],
            updates_since_emotional_peak=row["updates_since_emotional_peak"],
        )

    def upsert_reader_state(self, state: ReaderState) -> None:
        self._conn.execute(
            "INSERT INTO reader_state(quest_id, known_fact_ids, open_questions, "
            "expectations, attachment_levels, current_emotional_valence, "
            "updates_since_major_event, updates_since_revelation, updates_since_emotional_peak) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(quest_id) DO UPDATE SET "
            "known_fact_ids=excluded.known_fact_ids, "
            "open_questions=excluded.open_questions, "
            "expectations=excluded.expectations, "
            "attachment_levels=excluded.attachment_levels, "
            "current_emotional_valence=excluded.current_emotional_valence, "
            "updates_since_major_event=excluded.updates_since_major_event, "
            "updates_since_revelation=excluded.updates_since_revelation, "
            "updates_since_emotional_peak=excluded.updates_since_emotional_peak",
            (
                state.quest_id,
                json.dumps(state.known_fact_ids),
                json.dumps([q.model_dump() for q in state.open_questions]),
                json.dumps([e.model_dump(mode="json") for e in state.expectations]),
                json.dumps(state.attachment_levels),
                state.current_emotional_valence,
                state.updates_since_major_event,
                state.updates_since_revelation,
                state.updates_since_emotional_peak,
            ),
        )
        self._conn.commit()

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            entities=self.list_entities(),
            relationships=self.list_relationships(),
            rules=self.list_rules(),
            timeline=self.list_timeline(),
            narrative=self.list_narrative(limit=10_000),
            foreshadowing=[
                _row_to_hook(r)
                for r in self._conn.execute("SELECT * FROM foreshadowing ORDER BY id").fetchall()
            ],
            plot_threads=self.list_plot_threads(),
        )
