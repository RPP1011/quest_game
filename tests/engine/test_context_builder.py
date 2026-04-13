from pathlib import Path
import pytest
from app.engine.context_builder import AssembledContext, ContextBuilder
from app.engine.context_spec import PLAN_SPEC, WRITE_SPEC
from app.engine.prompt_renderer import PromptRenderer
from app.engine.token_budget import TokenBudget
from app.world import (
    ArcPosition,
    Entity,
    EntityType,
    NarrativeRecord,
    PlotThread,
    Relationship,
    StateDelta,
    WorldStateManager,
)
from app.world.delta import EntityCreate, RelChange
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(
        entity_creates=[
            EntityCreate(entity=Entity(id="alice", entity_type=EntityType.CHARACTER,
                                        name="Alice", data={"description": "A tavern keeper."})),
            EntityCreate(entity=Entity(id="tavern", entity_type=EntityType.LOCATION,
                                        name="The Broken Anchor")),
        ],
        relationship_changes=[RelChange(
            action="add",
            relationship=Relationship(source_id="alice", target_id="tavern", rel_type="owns"),
        )],
    ), update_number=1)
    sm.add_plot_thread(PlotThread(
        id="pt:1", name="Mystery", description="A strange ship docked.",
        involved_entities=["alice"], arc_position=ArcPosition.RISING,
    ))
    sm.write_narrative(NarrativeRecord(
        update_number=1, raw_text="Alice wiped down the bar.",
        summary="Alice tidies the tavern.",
    ))
    yield sm
    conn.close()


def test_plan_context_includes_entities_and_action(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    ctx = cb.build(
        spec=PLAN_SPEC,
        stage_name="plan",
        templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
        extras={"player_action": "Greet the stranger."},
    )
    assert isinstance(ctx, AssembledContext)
    assert "Alice" in ctx.user_prompt
    assert "Greet the stranger." in ctx.user_prompt
    assert "Mystery" in ctx.user_prompt
    assert ctx.token_estimate > 0
    assert "entities" in ctx.manifest
    assert ctx.manifest["entities"]["included_count"] == 2


def test_write_context_includes_style_from_extras(world):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    ctx = cb.build(
        spec=WRITE_SPEC,
        stage_name="write",
        templates={"system": "stages/write/system.j2", "user": "stages/write/user.j2"},
        extras={
            "plan": "Scene: tavern. Beats: 1) Alice greets customer.",
            "style": "Literary, spare prose.",
            "anti_patterns": ["purple prose", "overused adverbs"],
        },
    )
    assert "Literary" in ctx.system_prompt
    assert "purple prose" in ctx.system_prompt
    assert "Scene: tavern" in ctx.user_prompt
    assert "Alice wiped" in ctx.user_prompt  # full narrative mode includes raw_text


def test_manifest_records_drops_under_pressure(world):
    tight = TokenBudget(total=400, system_prompt=100, world_state=50,
                        narrative_history=50, style_config=10,
                        prior_stage_outputs=10, generation_headroom=50,
                        safety_margin=10)
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), tight)
    ctx = cb.build(
        spec=PLAN_SPEC,
        stage_name="plan",
        templates={"system": "stages/plan/system.j2", "user": "stages/plan/user.j2"},
        extras={"player_action": "Look around."},
    )
    # Compression should have been applied
    assert ctx.manifest["compression_applied"] is True
