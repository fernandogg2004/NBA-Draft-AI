"""Typed source registry loaded from config/sources.yaml.

Every source must declare its license and the result of a robots/ToS review. The registry is
the gate: ingestion code asks for a :class:`Source` by id and must respect ``enabled`` and
``scraping_allowed``; a disabled source cannot be fetched.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from nba_draft.config import REPO_ROOT

DEFAULT_SOURCES_PATH = REPO_ROOT / "config" / "sources.yaml"

SourceKind = str  # one of: "api", "open_dataset", "scrape"
_VALID_KINDS = {"api", "open_dataset", "scrape"}


class Source(BaseModel):
    """One vetted data source."""

    id: str
    name: str
    kind: SourceKind
    base_url: str
    license: str
    tos_url: str = ""
    robots_summary: str = ""
    scraping_allowed: bool = False
    requires_key: bool = False
    rate_limit_per_min: int = Field(default=0, ge=0)
    attribution_required: bool = True
    enabled: bool = False
    coverage: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _validate(self) -> Source:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Source {self.id!r}: kind must be one of {_VALID_KINDS}.")
        if self.license.strip().upper() in {"", "UNKNOWN"}:
            raise ValueError(f"Source {self.id!r}: license must be declared (not UNKNOWN).")
        if self.enabled and self.rate_limit_per_min == 0 and self.kind != "open_dataset":
            raise ValueError(
                f"Source {self.id!r}: enabled network source must set a positive "
                f"rate_limit_per_min."
            )
        return self

    @property
    def min_interval_s(self) -> float:
        """Minimum seconds between requests implied by the rate limit (0 if unlimited)."""
        if self.rate_limit_per_min <= 0:
            return 0.0
        return 60.0 / self.rate_limit_per_min


class _SourcesFile(BaseModel):
    sources: list[Source]

    @model_validator(mode="after")
    def _unique_ids(self) -> _SourcesFile:
        ids = [s.id for s in self.sources]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate source ids in registry: {sorted(dupes)}")
        return self


def load_sources(path: str | Path | None = None) -> dict[str, Source]:
    """Load and validate the source registry, keyed by source id."""
    cfg_path = Path(path) if path is not None else DEFAULT_SOURCES_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Sources registry not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parsed = _SourcesFile(**raw)
    return {s.id: s for s in parsed.sources}


def get_source(source_id: str, path: str | Path | None = None) -> Source:
    """Fetch a single source by id, raising if unknown."""
    sources = load_sources(path)
    if source_id not in sources:
        raise KeyError(f"Unknown source id {source_id!r}. Known: {sorted(sources)}")
    return sources[source_id]
