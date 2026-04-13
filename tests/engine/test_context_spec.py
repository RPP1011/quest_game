from app.engine.context_spec import ContextSpec, EntityScope, NarrativeMode


def test_defaults_are_conservative():
    spec = ContextSpec()
    assert spec.entity_scope == EntityScope.RELEVANT
    assert spec.include_relationships is True
    assert spec.include_rules is True
    assert spec.narrative_mode == NarrativeMode.SUMMARY
    assert spec.narrative_window == 3
    assert spec.include_style is False
    assert spec.include_anti_patterns is False


def test_spec_serializable():
    spec = ContextSpec(
        entity_scope=EntityScope.ACTIVE,
        narrative_mode=NarrativeMode.FULL,
        narrative_window=5,
        include_style=True,
        include_character_voices=True,
    )
    dumped = spec.model_dump()
    assert dumped["entity_scope"] == "active"
    assert dumped["narrative_mode"] == "full"


def test_plan_preset_values():
    from app.engine.context_spec import PLAN_SPEC
    assert PLAN_SPEC.include_style is False
    assert PLAN_SPEC.include_rules is True


def test_write_preset_values():
    from app.engine.context_spec import WRITE_SPEC
    assert WRITE_SPEC.include_style is True
    assert WRITE_SPEC.include_character_voices is True
    assert WRITE_SPEC.narrative_mode.value == "full"
