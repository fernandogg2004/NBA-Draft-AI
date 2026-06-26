"""Typed configuration loader.

All run-controlling knobs live in ``config/*.yaml`` and are validated here with pydantic.
Code should never read raw YAML directly — load through :func:`load_config` so that every
value is type-checked and defaults are explicit.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Repo root resolved relative to this file: src/nba_draft/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class Paths(BaseModel):
    data_raw: str
    data_interim: str
    data_processed: str
    data_external: str
    artifacts: str
    experiments: str


class Horizon(BaseModel):
    primary_window_years: int = Field(gt=0)


class ValidationConfig(BaseModel):
    n_holdout_years: int = Field(ge=1)
    min_train_years: int = Field(ge=1)
    val_horizon_years: int = Field(ge=1)
    step_years: int = Field(ge=1)
    expanding: bool = True


class Config(BaseModel):
    seed: int = 42
    paths: Paths
    horizon: Horizon
    validation: ValidationConfig

    # Resolved at load time, not from YAML.
    repo_root: Path = REPO_ROOT

    def path(self, key: str) -> Path:
        """Return an absolute path for a configured key (e.g. ``"data_processed"``)."""
        rel: str = getattr(self.paths, key)
        return (self.repo_root / rel).resolve()


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the central config.

    Args:
        path: Optional explicit path to a config YAML; defaults to ``config/config.yaml``.

    Returns:
        A validated :class:`Config` instance.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(**raw)
