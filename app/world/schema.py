from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    FACTION = "faction"
    ITEM = "item"
    CONCEPT = "concept"


class EntityStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    DECEASED = "deceased"
    DESTROYED = "destroyed"


class HookStatus(str, Enum):
    PLANTED = "planted"
    REFERENCED = "referenced"
    PAID_OFF = "paid_off"
    ABANDONED = "abandoned"


class ThreadStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ArcPosition(str, Enum):
    RISING = "rising"
    CLIMAX = "climax"
    FALLING = "falling"
    DENOUEMENT = "denouement"


class Entity(BaseModel):
    id: str
    entity_type: EntityType
    name: str
    data: dict[str, Any] = Field(default_factory=dict)
    status: EntityStatus = EntityStatus.ACTIVE
    last_referenced_update: int | None = None
    created_at_update: int | None = None


class Relationship(BaseModel):
    source_id: str
    target_id: str
    rel_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    established_at_update: int | None = None


class WorldRule(BaseModel):
    id: str
    category: str
    description: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    established_at_update: int | None = None


class TimelineEvent(BaseModel):
    update_number: int
    event_index: int
    description: str
    involved_entities: list[str] = Field(default_factory=list)
    causal_links: list[tuple[int, int]] = Field(default_factory=list)


class NarrativeRecord(BaseModel):
    update_number: int
    raw_text: str
    summary: str | None = None
    chapter_id: int | None = None
    state_diff: dict[str, Any] = Field(default_factory=dict)
    player_action: str | None = None
    pipeline_trace_id: str | None = None
    # Wave 4c: POV character for this update (best-effort; may be ``None``
    # for legacy rows, flat-flow writes, or scenes without a POV).
    pov_character_id: str | None = None


class ForeshadowingHook(BaseModel):
    id: str
    description: str
    planted_at_update: int
    payoff_target: str
    status: HookStatus = HookStatus.PLANTED
    paid_off_at_update: int | None = None
    references: list[int] = Field(default_factory=list)


class ParallelStatus(str, Enum):
    PLANTED = "planted"
    SCHEDULED = "scheduled"
    DELIVERED = "delivered"
    ABANDONED = "abandoned"


class Parallel(BaseModel):
    """A structural/parallel rhyme: an A-half (source) mirrored by a B-half
    (target) elsewhere in the narrative with some inversion axis between them.
    """
    id: str
    quest_id: str
    source_update: int
    source_description: str
    inversion_axis: str
    target_description: str
    status: ParallelStatus = ParallelStatus.PLANTED
    target_update_range_min: int | None = None
    target_update_range_max: int | None = None
    theme_ids: list[str] = Field(default_factory=list)
    delivered_at_update: int | None = None


class PlotThread(BaseModel):
    id: str
    name: str
    description: str
    status: ThreadStatus = ThreadStatus.ACTIVE
    involved_entities: list[str] = Field(default_factory=list)
    arc_position: ArcPosition
    priority: int = Field(default=5, ge=1, le=10)


class ExpectationStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SUBVERTED = "subverted"
    ABANDONED = "abandoned"


class OpenQuestion(BaseModel):
    id: str
    text: str
    priority: int = Field(default=5, ge=1, le=10)
    opened_at_update: int


class Expectation(BaseModel):
    id: str
    text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ExpectationStatus = ExpectationStatus.PENDING
    set_at_update: int


class ReaderState(BaseModel):
    """Model of what the reader currently knows / expects / feels.

    One row per quest. Mutated post-commit from the dramatic plan so that
    subsequent dramatic planning runs have a live picture of open questions,
    live expectations, and patience counters.
    """
    quest_id: str
    known_fact_ids: list[str] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    expectations: list[Expectation] = Field(default_factory=list)
    attachment_levels: dict[str, float] = Field(default_factory=dict)
    current_emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    updates_since_major_event: int = 0
    updates_since_revelation: int = 0
    updates_since_emotional_peak: int = 0


class KnowledgeEntry(BaseModel):
    """A single entity's knowledge of a fact.

    Keyed by entity_id (or the literal ``"reader"`` / ``"narrator"``) in
    :attr:`InformationState.known_by`.
    """
    learned_at_update: int
    learned_how: str = ""
    believes: bool = True
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class InformationState(BaseModel):
    """A tracked fact and a ledger of who knows it.

    One row per fact in the world. Mutated post-commit from each
    ``DramaticScene.reveals`` so that subsequent planning can detect dramatic
    irony, mystery, secret, and false-belief asymmetries.
    """
    id: str
    quest_id: str
    fact: str
    ground_truth: bool = True
    known_by: dict[str, KnowledgeEntry] = Field(default_factory=dict)


class AsymmetryKind(str, Enum):
    DRAMATIC_IRONY = "dramatic_irony"
    MYSTERY = "mystery"
    SECRET = "secret"
    FALSE_BELIEF = "false_belief"


class InformationAsymmetry(BaseModel):
    """Derived, in-memory asymmetry computed from :class:`InformationState`.

    Not persisted. Produced by
    :func:`app.planning.information_asymmetry.compute_asymmetries`.
    """
    kind: AsymmetryKind
    fact_id: str
    fact: str
    # Who knows / who doesn't (entity ids; may include "reader" / "narrator").
    knowers: list[str] = Field(default_factory=list)
    unaware: list[str] = Field(default_factory=list)
    # For false_belief: the id of the entity whose belief disagrees with
    # ground truth.
    believer_id: str | None = None
    # Heuristic score in [0, 1] — how ripe this is for narrative use.
    tension_potential: float = 0.0
    # How many updates since the knowing party learned the fact (for dramatic
    # irony freshness; None if not applicable).
    updates_standing: int | None = None


class EmotionalBeat(BaseModel):
    """Observed per-scene emotional target, persisted post-commit.

    Written from the emotional planner's ``EmotionalScenePlan`` targets after
    a quest update commits. Provides the observed-history analogue to
    ``QuestArcState.tension_observed``: lets the emotional planner detect
    monotony and schedule contrast against actual trajectory instead of
    flying blind each tick.
    """
    id: int | None = None
    quest_id: str
    update_number: int
    scene_index: int
    primary_emotion: str
    secondary_emotion: str | None = None
    intensity: float = Field(ge=0.0, le=1.0)
    source: str


class StoryCandidateStatus(str, Enum):
    DRAFT = "draft"
    PICKED = "picked"
    REJECTED = "rejected"


class StoryCandidate(BaseModel):
    """A candidate story arc for a given seed.

    A single seed supports multiple candidates (Phase 1 of the story-rollout
    architecture). Each candidate commits to a specific arc emphasis —
    which plot threads are primary, which character is protagonist, which
    themes drive the emphasis — so the same seed can produce materially
    different stories. The player picks one; subsequent planning honors
    the pick.
    """
    id: str
    quest_id: str
    title: str
    synopsis: str
    primary_thread_ids: list[str] = Field(default_factory=list)
    secondary_thread_ids: list[str] = Field(default_factory=list)
    protagonist_character_id: str | None = None
    emphasized_theme_ids: list[str] = Field(default_factory=list)
    climax_description: str = ""
    expected_chapter_count: int = 15
    status: StoryCandidateStatus = StoryCandidateStatus.DRAFT


class QuestArcState(BaseModel):
    """Persisted arc state (thin — references the craft-level Arc)."""
    arc_id: str                 # matches app.craft.Arc.id
    quest_id: str
    current_phase_index: int = 0
    phase_progress: float = 0.0
    tension_observed: list[tuple[int, float]] = Field(default_factory=list)
    structure_id: str           # e.g. "three_act"
    scale: str                  # "scene" | "chapter" | "campaign" | "saga"
    last_directive: dict | None = None  # JSON-serialized ArcDirective
