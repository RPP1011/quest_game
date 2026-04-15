"""YAML loading and validation for unified quest runner configs.

See docs/superpowers/specs/2026-04-14-quest-runner-resume-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """Raised when a run config is missing fields, has unknown keys, or
    references a seed/actions file that does not exist."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NarratorConfig(_Strict):
    pov_character_id: str | None = None
    pov_type: str
    worldview: str
    editorial_stance: str | None = None
    sensory_bias: dict[str, float] = Field(default_factory=dict)
    attention_bias: list[str] = Field(default_factory=list)
    voice_samples: list[str] = Field(default_factory=list)


class SeedConfig(_Strict):
    quest_id: str
    genre: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    plot_threads: list[dict[str, Any]] = Field(default_factory=list)
    themes: list[dict[str, Any]] = Field(default_factory=list)
    foreshadowing: list[dict[str, Any]] = Field(default_factory=list)
    narrator: NarratorConfig


class SftCollectionConfig(_Strict):
    enabled: bool = False
    dir: str | None = None


class RunOptions(_Strict):
    n_candidates: int = 1
    scoring: bool = False
    rerank_weights: dict[str, float] | None = None
    sft_collection: SftCollectionConfig = Field(default_factory=SftCollectionConfig)
    llm_url: str
    llm_model: str
    db_path: Path | None = None


class RunConfig(_Strict):
    run_name: str
    seed: SeedConfig
    actions: list[str]
    options: RunOptions


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(str(e)) from e


def _resolve_seed(value: Any, base_dir: Path) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ConfigError(f"seed: expected string name or inline dict, got {type(value).__name__}")
    seed_path = base_dir / "seeds" / f"{value}.yaml"
    if not seed_path.is_file():
        raise ConfigError(f"seed reference '{value}' not found at {seed_path}")
    data = _load_yaml(seed_path)
    if not isinstance(data, dict):
        raise ConfigError(f"seed file {seed_path} did not parse as a mapping")
    return data


def _resolve_actions(value: Any, base_dir: Path) -> list[str]:
    if isinstance(value, list):
        return [str(a) for a in value]
    if not isinstance(value, str):
        raise ConfigError(f"actions: expected string name or inline list, got {type(value).__name__}")
    actions_path = base_dir / "actions" / f"{value}.yaml"
    if not actions_path.is_file():
        raise ConfigError(f"actions reference '{value}' not found at {actions_path}")
    data = _load_yaml(actions_path)
    if not isinstance(data, list):
        raise ConfigError(f"actions file {actions_path} did not parse as a list")
    return [str(a) for a in data]


def _build_run_config(raw: dict, run_name: str, base_dir: Path) -> RunConfig:
    if not isinstance(raw, dict):
        raise ConfigError("run config must be a YAML mapping")
    if "actions" not in raw:
        raise ConfigError("required field 'actions' missing from run config")
    if "seed" not in raw:
        raise ConfigError("required field 'seed' missing from run config")
    seed_dict = _resolve_seed(raw["seed"], base_dir)
    actions_list = _resolve_actions(raw["actions"], base_dir)
    options_dict = raw.get("options") or {}
    if not isinstance(options_dict, dict):
        raise ConfigError("'options' must be a mapping if provided")
    # Detect unknown top-level keys (Pydantic catches nested ones)
    allowed_top = {"seed", "actions", "options"}
    unknown = set(raw.keys()) - allowed_top
    if unknown:
        raise ConfigError(f"unknown top-level keys in run config: {sorted(unknown)}")

    if "db_path" not in options_dict:
        options_dict["db_path"] = Path("/tmp/quest_run") / f"{run_name}.db"

    try:
        return RunConfig(
            run_name=run_name,
            seed=seed_dict,
            actions=actions_list,
            options=options_dict,
        )
    except ValidationError as e:
        raise ConfigError(str(e)) from e


def load_run_config(path: Path, *, base_dir: Path | None = None) -> RunConfig:
    """Load a run config from a YAML file.

    The run name is derived from the filename stem.
    ``base_dir`` defaults to the file's grandparent (so refs to
    ``seeds/<name>.yaml`` resolve relative to the configs directory).
    """
    path = Path(path)
    if base_dir is None:
        base_dir = path.parent.parent
    raw = _load_yaml(path)
    return _build_run_config(raw, run_name=path.stem, base_dir=base_dir)


def load_run_config_from_string(
    text: str, *, run_name: str, base_dir: Path
) -> RunConfig:
    """Variant for tests and inline configs."""
    raw = yaml.safe_load(text)
    return _build_run_config(raw, run_name=run_name, base_dir=base_dir)
