from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from typing import Any
from app.planning.world_extensions import Motif, MotifOccurrence, Theme
from .schema import (
    ArcPosition,
    EmotionalBeat,
    Entity,
    EntityType,
    Expectation,
    ForeshadowingHook,
    HookStatus,
    InformationState,
    KnowledgeEntry,
    NarrativeRecord,
    OpenQuestion,
    Parallel,
    ParallelStatus,
    PlotThread,
    ArcSkeleton,
    HookPlacement,
    QuestArcState,
    ReaderState,
    Relationship,
    SkeletonChapter,
    StoryCandidate,
    StoryCandidateStatus,
    ThemeBeat,
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
    parallels: tuple[Parallel, ...] = ()


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
    # ``pov_character_id`` was added in Wave 4c. Legacy rows (and rows
    # written before the additive migration ran) surface as ``None``.
    try:
        pov_character_id = row["pov_character_id"]
    except (IndexError, KeyError):
        pov_character_id = None
    return NarrativeRecord(
        update_number=row["update_number"],
        raw_text=row["raw_text"],
        summary=row["summary"],
        chapter_id=row["chapter_id"],
        state_diff=json.loads(row["state_diff"]),
        player_action=row["player_action"],
        pipeline_trace_id=row["pipeline_trace_id"],
        pov_character_id=pov_character_id,
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


def _row_to_parallel(row: sqlite3.Row) -> Parallel:
    return Parallel(
        id=row["id"],
        quest_id=row["quest_id"],
        source_update=row["source_update"],
        source_description=row["source_description"],
        inversion_axis=row["inversion_axis"],
        target_description=row["target_description"],
        status=row["status"],
        target_update_range_min=row["target_update_range_min"],
        target_update_range_max=row["target_update_range_max"],
        theme_ids=json.loads(row["theme_ids"]),
        delivered_at_update=row["delivered_at_update"],
    )


def _row_to_motif(row: sqlite3.Row) -> Motif:
    return Motif(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        theme_ids=json.loads(row["theme_ids"]),
        semantic_range=json.loads(row["semantic_range"]),
        target_interval_min=row["target_interval_min"],
        target_interval_max=row["target_interval_max"],
    )


def _row_to_info_state(row: sqlite3.Row) -> InformationState:
    raw = json.loads(row["known_by"]) if row["known_by"] else {}
    known_by = {k: KnowledgeEntry.model_validate(v) for k, v in raw.items()}
    return InformationState(
        id=row["id"],
        quest_id=row["quest_id"],
        fact=row["fact"],
        ground_truth=bool(row["ground_truth"]),
        known_by=known_by,
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


def _row_to_arc_skeleton(row: sqlite3.Row) -> ArcSkeleton:
    return ArcSkeleton(
        id=row["id"],
        candidate_id=row["candidate_id"],
        quest_id=row["quest_id"],
        chapters=[SkeletonChapter(**c) for c in json.loads(row["chapters"])],
        theme_arc=[ThemeBeat(**t) for t in json.loads(row["theme_arc"])],
        hook_schedule=[HookPlacement(**h) for h in json.loads(row["hook_schedule"])],
    )


def _row_to_story_candidate(row: sqlite3.Row) -> StoryCandidate:
    return StoryCandidate(
        id=row["id"],
        quest_id=row["quest_id"],
        title=row["title"],
        synopsis=row["synopsis"],
        primary_thread_ids=json.loads(row["primary_thread_ids"]),
        secondary_thread_ids=json.loads(row["secondary_thread_ids"]),
        protagonist_character_id=row["protagonist_character_id"],
        emphasized_theme_ids=json.loads(row["emphasized_theme_ids"]),
        climax_description=row["climax_description"],
        expected_chapter_count=row["expected_chapter_count"],
        status=row["status"],
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
            "(update_number, raw_text, summary, chapter_id, state_diff, player_action, pipeline_trace_id, pov_character_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (record.update_number, record.raw_text, record.summary, record.chapter_id,
             json.dumps(record.state_diff), record.player_action, record.pipeline_trace_id,
             record.pov_character_id),
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

    # ---- parallels ----

    def add_parallel(self, p: Parallel) -> None:
        self._conn.execute(
            "INSERT INTO parallels(id, quest_id, source_update, source_description, "
            "inversion_axis, target_description, status, target_update_range_min, "
            "target_update_range_max, theme_ids, delivered_at_update) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                p.id, p.quest_id, p.source_update, p.source_description,
                p.inversion_axis, p.target_description,
                p.status.value if hasattr(p.status, "value") else p.status,
                p.target_update_range_min, p.target_update_range_max,
                json.dumps(p.theme_ids), p.delivered_at_update,
            ),
        )
        self._conn.commit()

    def get_parallel(self, parallel_id: str) -> Parallel:
        row = self._conn.execute(
            "SELECT * FROM parallels WHERE id=?", (parallel_id,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no parallel {parallel_id}")
        return _row_to_parallel(row)

    def update_parallel(self, parallel_id: str, patch: dict[str, Any]) -> None:
        current = self.get_parallel(parallel_id)

        def _v(x: Any) -> Any:
            return x.value if hasattr(x, "value") else x

        self._conn.execute(
            "UPDATE parallels SET status=?, target_description=?, inversion_axis=?, "
            "target_update_range_min=?, target_update_range_max=?, theme_ids=?, "
            "delivered_at_update=? WHERE id=?",
            (
                _v(patch.get("status", current.status)),
                patch.get("target_description", current.target_description),
                patch.get("inversion_axis", current.inversion_axis),
                patch.get("target_update_range_min", current.target_update_range_min),
                patch.get("target_update_range_max", current.target_update_range_max),
                json.dumps(patch.get("theme_ids", current.theme_ids)),
                patch.get("delivered_at_update", current.delivered_at_update),
                parallel_id,
            ),
        )
        self._conn.commit()

    def list_parallels(
        self, quest_id: str | None = None, statuses: list[ParallelStatus] | None = None
    ) -> list[Parallel]:
        q = "SELECT * FROM parallels"
        clauses: list[str] = []
        args: list[Any] = []
        if quest_id is not None:
            clauses.append("quest_id=?")
            args.append(quest_id)
        if statuses:
            placeholders = ",".join("?" * len(statuses))
            clauses.append(f"status IN ({placeholders})")
            args.extend(s.value if hasattr(s, "value") else s for s in statuses)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY source_update, id"
        rows = self._conn.execute(q, tuple(args)).fetchall()
        return [_row_to_parallel(r) for r in rows]

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

            # Parallels: revert deliveries that happened after to_update, and
            # delete parallels whose source was planted after to_update.
            c.execute(
                "DELETE FROM parallels WHERE source_update > ?", (to_update,)
            )
            for row in c.execute(
                "SELECT id, delivered_at_update FROM parallels"
            ).fetchall():
                d = row["delivered_at_update"]
                if d is not None and d > to_update:
                    c.execute(
                        "UPDATE parallels SET status=?, delivered_at_update=NULL WHERE id=?",
                        ("planted", row["id"]),
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

    # ---- themes ----

    def add_theme(self, quest_id: str, theme: Theme) -> None:
        self._conn.execute(
            "INSERT INTO themes(id, quest_id, proposition, stance, motif_ids, "
            "thesis_character_ids, key_scenes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                theme.id,
                quest_id,
                theme.proposition,
                theme.stance,
                json.dumps(theme.motif_ids),
                json.dumps(theme.thesis_character_ids),
                json.dumps(theme.key_scenes),
            ),
        )
        self._conn.commit()

    def get_theme(self, quest_id: str, theme_id: str) -> Theme:
        row = self._conn.execute(
            "SELECT * FROM themes WHERE quest_id=? AND id=?", (quest_id, theme_id)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no theme {theme_id!r} for quest {quest_id!r}")
        return Theme(
            id=row["id"],
            proposition=row["proposition"],
            stance=row["stance"],
            motif_ids=json.loads(row["motif_ids"]),
            thesis_character_ids=json.loads(row["thesis_character_ids"]),
            key_scenes=json.loads(row["key_scenes"]),
        )

    def list_themes(self, quest_id: str) -> list[Theme]:
        rows = self._conn.execute(
            "SELECT * FROM themes WHERE quest_id=? ORDER BY id", (quest_id,)
        ).fetchall()
        return [
            Theme(
                id=r["id"],
                proposition=r["proposition"],
                stance=r["stance"],
                motif_ids=json.loads(r["motif_ids"]),
                thesis_character_ids=json.loads(r["thesis_character_ids"]),
                key_scenes=json.loads(r["key_scenes"]),
            )
            for r in rows
        ]

    def update_theme_stance(self, quest_id: str, theme_id: str, new_stance: str) -> None:
        """Persist a stance change for a theme. Caller is responsible for
        deciding *when* the stance should evolve (e.g. based on post-COMMIT
        assessment)."""
        cur = self._conn.execute(
            "UPDATE themes SET stance=? WHERE quest_id=? AND id=?",
            (new_stance, quest_id, theme_id),
        )
        if cur.rowcount == 0:
            raise WorldStateError(f"no theme {theme_id!r} for quest {quest_id!r}")
        self._conn.commit()

    # ---- motifs (Gap G5) ----

    def add_motif(self, quest_id: str, motif: Motif) -> None:
        self._conn.execute(
            "INSERT INTO motifs(id, quest_id, name, description, theme_ids, "
            "semantic_range, target_interval_min, target_interval_max) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                motif.id,
                quest_id,
                motif.name,
                motif.description,
                json.dumps(motif.theme_ids),
                json.dumps(motif.semantic_range),
                motif.target_interval_min,
                motif.target_interval_max,
            ),
        )
        self._conn.commit()

    def get_motif(self, quest_id: str, motif_id: str) -> Motif:
        row = self._conn.execute(
            "SELECT * FROM motifs WHERE quest_id=? AND id=?", (quest_id, motif_id)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no motif {motif_id!r} for quest {quest_id!r}")
        return _row_to_motif(row)

    def list_motifs(self, quest_id: str) -> list[Motif]:
        rows = self._conn.execute(
            "SELECT * FROM motifs WHERE quest_id=? ORDER BY id", (quest_id,)
        ).fetchall()
        return [_row_to_motif(r) for r in rows]

    def record_motif_occurrence(self, quest_id: str, occ: MotifOccurrence) -> int:
        cur = self._conn.execute(
            "INSERT INTO motif_occurrences"
            "(motif_id, quest_id, update_number, context, semantic_value, intensity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                occ.motif_id,
                quest_id,
                occ.update_number,
                occ.context,
                occ.semantic_value,
                occ.intensity,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_motif_occurrences(
        self, quest_id: str, motif_id: str | None = None
    ) -> list[MotifOccurrence]:
        if motif_id is None:
            rows = self._conn.execute(
                "SELECT * FROM motif_occurrences WHERE quest_id=? "
                "ORDER BY update_number ASC, id ASC",
                (quest_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM motif_occurrences WHERE quest_id=? AND motif_id=? "
                "ORDER BY update_number ASC, id ASC",
                (quest_id, motif_id),
            ).fetchall()
        return [
            MotifOccurrence(
                motif_id=r["motif_id"],
                update_number=r["update_number"],
                context=r["context"],
                semantic_value=r["semantic_value"],
                intensity=r["intensity"],
            )
            for r in rows
        ]

    def last_motif_occurrence(
        self, quest_id: str, motif_id: str
    ) -> MotifOccurrence | None:
        row = self._conn.execute(
            "SELECT * FROM motif_occurrences WHERE quest_id=? AND motif_id=? "
            "ORDER BY update_number DESC, id DESC LIMIT 1",
            (quest_id, motif_id),
        ).fetchone()
        if row is None:
            return None
        return MotifOccurrence(
            motif_id=row["motif_id"],
            update_number=row["update_number"],
            context=row["context"],
            semantic_value=row["semantic_value"],
            intensity=row["intensity"],
        )

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

    # ---- emotional beats ----

    def record_emotional_beat(self, beat: EmotionalBeat) -> int:
        """Persist an observed emotional beat. Returns the row id."""
        cur = self._conn.execute(
            "INSERT INTO emotional_beats"
            "(quest_id, update_number, scene_index, primary_emotion, "
            "secondary_emotion, intensity, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                beat.quest_id,
                beat.update_number,
                beat.scene_index,
                beat.primary_emotion,
                beat.secondary_emotion,
                beat.intensity,
                beat.source,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_recent_emotional_beats(
        self, quest_id: str, limit: int = 10
    ) -> list[EmotionalBeat]:
        """Return the most recent N beats for the quest, ordered oldest->newest.

        Selects the last ``limit`` beats by (update_number, scene_index) and
        returns them in ascending order so the planner sees a left-to-right
        trajectory (e.g. "dread -> dread -> grief -> dread").
        """
        rows = self._conn.execute(
            "SELECT * FROM (SELECT * FROM emotional_beats WHERE quest_id=? "
            "ORDER BY update_number DESC, scene_index DESC, id DESC LIMIT ?) "
            "ORDER BY update_number ASC, scene_index ASC, id ASC",
            (quest_id, limit),
        ).fetchall()
        return [
            EmotionalBeat(
                id=r["id"],
                quest_id=r["quest_id"],
                update_number=r["update_number"],
                scene_index=r["scene_index"],
                primary_emotion=r["primary_emotion"],
                secondary_emotion=r["secondary_emotion"],
                intensity=r["intensity"],
                source=r["source"],
            )
            for r in rows
        ]

    # ---- information states (Gap G7) ----

    def upsert_information_state(self, state: InformationState) -> None:
        self._conn.execute(
            "INSERT INTO information_states(id, quest_id, fact, ground_truth, known_by) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "quest_id=excluded.quest_id, fact=excluded.fact, "
            "ground_truth=excluded.ground_truth, known_by=excluded.known_by",
            (
                state.id,
                state.quest_id,
                state.fact,
                1 if state.ground_truth else 0,
                json.dumps({k: v.model_dump() for k, v in state.known_by.items()}),
            ),
        )
        self._conn.commit()

    def get_information_state(self, id: str) -> InformationState:
        row = self._conn.execute(
            "SELECT * FROM information_states WHERE id=?", (id,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no information_state {id!r}")
        return _row_to_info_state(row)

    def list_information_states(self, quest_id: str) -> list[InformationState]:
        rows = self._conn.execute(
            "SELECT * FROM information_states WHERE quest_id=? ORDER BY id",
            (quest_id,),
        ).fetchall()
        return [_row_to_info_state(r) for r in rows]

    # ---- narrative embeddings (retrieval layer, Wave 1c) ----

    def upsert_narrative_embedding(
        self,
        quest_id: str,
        update_number: int,
        scene_index: int,
        embedding: "Any",
        text_preview: str,
    ) -> None:
        """Persist (or replace) a narrative embedding row.

        Embedding is stored as raw float32 bytes; reconstruct via
        ``np.frombuffer(..., dtype=np.float32)``.
        """
        import numpy as np  # local import to keep import graph lean
        arr = np.asarray(embedding, dtype=np.float32)
        blob = arr.tobytes()
        self._conn.execute(
            "INSERT INTO narrative_embeddings"
            "(quest_id, update_number, scene_index, embedding, text_preview) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(quest_id, update_number, scene_index) DO UPDATE SET "
            "embedding=excluded.embedding, text_preview=excluded.text_preview",
            (quest_id, update_number, scene_index, blob, text_preview),
        )
        self._conn.commit()

    def list_narrative_embeddings(
        self, quest_id: str, limit: int | None = None,
    ) -> list[dict]:
        """Return embedding rows for ``quest_id`` newest-first.

        Each dict has ``update_number``, ``scene_index``,
        ``embedding`` (``np.ndarray`` of float32), and ``text_preview``.
        """
        import numpy as np
        sql = (
            "SELECT update_number, scene_index, embedding, text_preview "
            "FROM narrative_embeddings WHERE quest_id=? "
            "ORDER BY update_number DESC, scene_index DESC"
        )
        params: tuple[Any, ...] = (quest_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (quest_id, int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "update_number": r["update_number"],
                "scene_index": r["scene_index"],
                "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
                "text_preview": r["text_preview"],
            }
            for r in rows
        ]

    # ---- scorecards (Day 2 — 12-dim scoring) ----

    def save_scorecard(
        self,
        scorecard: "Any",
        *,
        quest_id: str,
        update_number: int,
        scene_index: int = 0,
        pipeline_trace_id: str | None = None,
    ) -> int:
        """Persist a :class:`app.scoring.Scorecard` and its dimension rows.

        Returns the new ``scorecards.id``. A single transaction writes the
        header row plus one ``dimension_scores`` row per dim — partial
        writes can't occur.
        """
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO scorecards"
                "(quest_id, update_number, scene_index, pipeline_trace_id, overall_score) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    quest_id,
                    int(update_number),
                    int(scene_index),
                    pipeline_trace_id,
                    float(scorecard.overall_score),
                ),
            )
            scorecard_id = int(cur.lastrowid)
            self._conn.executemany(
                "INSERT INTO dimension_scores(scorecard_id, dimension, score) "
                "VALUES (?, ?, ?)",
                [
                    (scorecard_id, name, float(score))
                    for name, score in scorecard.dimension_items()
                ],
            )
        return scorecard_id

    def append_dimension_scores(
        self,
        scorecard_id: int,
        dims: "dict[str, float]",
    ) -> int:
        """Append extra ``dimension_scores`` rows onto an existing scorecard.

        Day 6 uses this to land the three LLM-judge dim scores onto the
        scorecard header row written at commit time. ``INSERT OR REPLACE``
        semantics are used so a retry/rerun overwrites a prior value for
        the same ``(scorecard_id, dimension)`` rather than raising on the
        primary-key conflict. Returns the number of rows written.
        """
        rows = [
            (int(scorecard_id), str(name), float(score))
            for name, score in dims.items()
        ]
        if not rows:
            return 0
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO dimension_scores"
                "(scorecard_id, dimension, score) VALUES (?, ?, ?)",
                rows,
            )
        return len(rows)

    def list_dimension_scores(
        self, scorecard_id: int,
    ) -> "dict[str, float]":
        """Return the full ``dimension -> score`` map for one scorecard.

        Used by the dashboard and by Day 6 pipeline tests that need to
        verify the async LLM-judge rows landed under the same header.
        """
        rows = self._conn.execute(
            "SELECT dimension, score FROM dimension_scores WHERE scorecard_id=?",
            (int(scorecard_id),),
        ).fetchall()
        return {r["dimension"]: float(r["score"]) for r in rows}

    def list_scorecards(
        self, quest_id: str, limit: int | None = None,
    ) -> "list[Any]":
        """Return scorecards for ``quest_id`` oldest-first.

        Each element is a :class:`app.scoring.Scorecard` reconstructed from
        its persisted dim rows. Ordering is
        ``(update_number, scene_index, id)`` ascending so callers get a
        natural time series for the dashboard.
        """
        from app.scoring import Scorecard  # local import to keep cold-start light

        sql = (
            "SELECT id, overall_score FROM scorecards WHERE quest_id=? "
            "ORDER BY update_number ASC, scene_index ASC, id ASC"
        )
        params: tuple[Any, ...] = (quest_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (quest_id, int(limit))
        header_rows = self._conn.execute(sql, params).fetchall()
        if not header_rows:
            return []

        ids = [r["id"] for r in header_rows]
        placeholders = ",".join("?" for _ in ids)
        dim_rows = self._conn.execute(
            f"SELECT scorecard_id, dimension, score FROM dimension_scores "
            f"WHERE scorecard_id IN ({placeholders})",
            tuple(ids),
        ).fetchall()

        by_card: dict[int, dict[str, float]] = {sid: {} for sid in ids}
        for r in dim_rows:
            by_card[r["scorecard_id"]][r["dimension"]] = float(r["score"])

        out: list[Any] = []
        for r in header_rows:
            dims = by_card.get(r["id"], {})
            try:
                out.append(Scorecard(
                    overall_score=float(r["overall_score"]),
                    **dims,
                ))
            except Exception:
                # Partial / legacy row — skip rather than fail the listing.
                continue
        return out

    # ---- story candidates (Phase 1: story-rollout architecture) ----
    def add_story_candidate(self, cand: StoryCandidate) -> None:
        self._conn.execute(
            "INSERT INTO story_candidates(id, quest_id, title, synopsis, "
            "primary_thread_ids, secondary_thread_ids, protagonist_character_id, "
            "emphasized_theme_ids, climax_description, expected_chapter_count, "
            "status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cand.id, cand.quest_id, cand.title, cand.synopsis,
                json.dumps(cand.primary_thread_ids),
                json.dumps(cand.secondary_thread_ids),
                cand.protagonist_character_id,
                json.dumps(cand.emphasized_theme_ids),
                cand.climax_description, cand.expected_chapter_count,
                cand.status.value if hasattr(cand.status, "value") else cand.status,
            ),
        )
        self._conn.commit()

    def list_story_candidates(self, quest_id: str) -> list[StoryCandidate]:
        rows = self._conn.execute(
            "SELECT * FROM story_candidates WHERE quest_id=? ORDER BY id",
            (quest_id,),
        ).fetchall()
        return [_row_to_story_candidate(r) for r in rows]

    def get_story_candidate(self, candidate_id: str) -> StoryCandidate:
        row = self._conn.execute(
            "SELECT * FROM story_candidates WHERE id=?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no story candidate {candidate_id!r}")
        return _row_to_story_candidate(row)

    def set_candidate_status(
        self, candidate_id: str, status: StoryCandidateStatus,
    ) -> None:
        cur = self._conn.execute(
            "UPDATE story_candidates SET status=? WHERE id=?",
            (status.value if hasattr(status, "value") else status, candidate_id),
        )
        if cur.rowcount == 0:
            raise WorldStateError(f"no story candidate {candidate_id!r}")
        self._conn.commit()

    def pick_story_candidate(
        self, quest_id: str, candidate_id: str,
    ) -> StoryCandidate:
        """Mark one candidate as PICKED and all others as REJECTED.

        Returns the picked candidate. Idempotent: picking an already-picked
        candidate is a no-op; picking a different one swaps the pick.
        """
        target = self.get_story_candidate(candidate_id)
        if target.quest_id != quest_id:
            raise WorldStateError(
                f"candidate {candidate_id!r} belongs to quest "
                f"{target.quest_id!r}, not {quest_id!r}"
            )
        self._conn.execute(
            "UPDATE story_candidates SET status=? WHERE quest_id=? AND status=?",
            (StoryCandidateStatus.REJECTED.value, quest_id,
             StoryCandidateStatus.PICKED.value),
        )
        self._conn.execute(
            "UPDATE story_candidates SET status=? WHERE id=?",
            (StoryCandidateStatus.PICKED.value, candidate_id),
        )
        self._conn.commit()
        return self.get_story_candidate(candidate_id)

    def get_picked_candidate(self, quest_id: str) -> StoryCandidate | None:
        row = self._conn.execute(
            "SELECT * FROM story_candidates WHERE quest_id=? AND status=? LIMIT 1",
            (quest_id, StoryCandidateStatus.PICKED.value),
        ).fetchone()
        return _row_to_story_candidate(row) if row else None

    # ---- arc skeletons (Phase 2: story-rollout architecture) ----
    def save_arc_skeleton(self, skeleton: ArcSkeleton) -> None:
        """Upsert a skeleton by id. Used by the planner on (re)generation."""
        self._conn.execute(
            "INSERT INTO arc_skeletons(id, candidate_id, quest_id, chapters, "
            "theme_arc, hook_schedule) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "chapters=excluded.chapters, theme_arc=excluded.theme_arc, "
            "hook_schedule=excluded.hook_schedule",
            (
                skeleton.id, skeleton.candidate_id, skeleton.quest_id,
                json.dumps([c.model_dump() for c in skeleton.chapters]),
                json.dumps([t.model_dump() for t in skeleton.theme_arc]),
                json.dumps([h.model_dump() for h in skeleton.hook_schedule]),
            ),
        )
        self._conn.commit()

    def get_arc_skeleton(self, skeleton_id: str) -> ArcSkeleton:
        row = self._conn.execute(
            "SELECT * FROM arc_skeletons WHERE id=?", (skeleton_id,),
        ).fetchone()
        if row is None:
            raise WorldStateError(f"no arc skeleton {skeleton_id!r}")
        return _row_to_arc_skeleton(row)

    def get_skeleton_for_candidate(
        self, candidate_id: str,
    ) -> ArcSkeleton | None:
        """Return the most-recent skeleton for a candidate, or None."""
        row = self._conn.execute(
            "SELECT * FROM arc_skeletons WHERE candidate_id=? "
            "ORDER BY created_at DESC LIMIT 1",
            (candidate_id,),
        ).fetchone()
        return _row_to_arc_skeleton(row) if row else None

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
            parallels=tuple(self.list_parallels()),
        )
