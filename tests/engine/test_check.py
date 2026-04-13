from app.engine.check import CHECK_SCHEMA, CheckIssue, CheckOutput
from app.engine.context_spec import CHECK_SPEC, REVISE_SPEC


def test_check_output_summary_classifies_severity():
    out = CheckOutput(issues=[
        CheckIssue(severity="info", category="prose_quality", message="meh"),
        CheckIssue(severity="warning", category="continuity", message="x"),
        CheckIssue(severity="critical", category="world_rule", message="magic banned"),
    ])
    assert out.has_critical is True
    assert out.has_fixable is True   # warning counts as fixable
    assert out.all_trivial is False


def test_check_output_empty_is_clean():
    out = CheckOutput(issues=[])
    assert out.has_critical is False
    assert out.has_fixable is False
    assert out.all_trivial is True


def test_check_schema_is_json_schema_shape():
    assert CHECK_SCHEMA["type"] == "object"
    assert "issues" in CHECK_SCHEMA["properties"]


def test_check_spec_conservative_defaults():
    # CHECK needs to see rules, plot threads, recent prose — not style
    assert CHECK_SPEC.include_rules is True
    assert CHECK_SPEC.include_style is False


def test_revise_spec_like_write_plus_check():
    assert REVISE_SPEC.include_style is True
    assert "check" in REVISE_SPEC.prior_stages
