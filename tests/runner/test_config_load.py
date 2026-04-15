from pathlib import Path
import pytest
from app.runner_config import (
    RunConfig, ConfigError, load_run_config, load_run_config_from_string,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_basic_run_config():
    cfg = load_run_config(FIXTURES / "runs" / "sample.yaml",
                          base_dir=FIXTURES)
    assert isinstance(cfg, RunConfig)
    assert cfg.run_name == "sample"
    assert cfg.seed.quest_id == "sample"
    assert cfg.seed.narrator.pov_character_id == "player"
    assert cfg.actions == [
        "I do the first thing.",
        "I do the second thing.",
        "I do the third thing.",
    ]
    assert cfg.options.n_candidates == 1
    assert cfg.options.db_path == Path("/tmp/quest_run_test/sample.db")


def test_inline_actions_list_supported():
    yaml_text = """
seed: sample
actions:
  - First inline action.
  - Second inline action.
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    cfg = load_run_config_from_string(yaml_text, run_name="inline-test",
                                      base_dir=FIXTURES)
    assert cfg.actions == ["First inline action.", "Second inline action."]


def test_default_db_path_uses_run_name():
    cfg = load_run_config(FIXTURES / "runs" / "sample.yaml",
                          base_dir=FIXTURES)
    # sample.yaml sets db_path explicitly; check the default code path too
    yaml_text = """
seed: sample
actions: sample
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    cfg2 = load_run_config_from_string(yaml_text, run_name="my-run",
                                       base_dir=FIXTURES)
    assert cfg2.options.db_path == Path("/tmp/quest_run/my-run.db")


def test_missing_required_field_raises_config_error():
    yaml_text = """
seed: sample
# actions intentionally missing
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    with pytest.raises(ConfigError) as excinfo:
        load_run_config_from_string(yaml_text, run_name="bad",
                                    base_dir=FIXTURES)
    assert "actions" in str(excinfo.value)


def test_unknown_key_rejected():
    yaml_text = """
seed: sample
actions: sample
typoo: 4   # typo of 'options' - should fail
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    with pytest.raises(ConfigError) as excinfo:
        load_run_config_from_string(yaml_text, run_name="typo",
                                    base_dir=FIXTURES)
    assert "typoo" in str(excinfo.value)


def test_unresolvable_seed_reference():
    yaml_text = """
seed: nonexistent
actions: sample
options:
  llm_url: http://127.0.0.1:8082
  llm_model: writer_v1
"""
    with pytest.raises(ConfigError) as excinfo:
        load_run_config_from_string(yaml_text, run_name="bad-ref",
                                    base_dir=FIXTURES)
    assert "nonexistent" in str(excinfo.value)
