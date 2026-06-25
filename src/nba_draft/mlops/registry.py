"""A lightweight, file-based model registry.

Each registered model is saved under ``artifacts/models/<name>/<version>/`` with the serialized
estimator plus a ``meta.json`` (metrics, feature columns, data version, stage, timestamp). A
top-level ``registry.json`` indexes versions and tracks which is ``latest`` and each stage.
Kept dependency-light (joblib + json) so model lineage is auditable without a server.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

DEFAULT_ROOT = Path("artifacts/models")


@dataclass
class ModelMeta:
    name: str
    version: str
    created_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    data_version: str | None = None
    stage: str = "none"          # none | staging | production | archived
    extra: dict[str, Any] = field(default_factory=dict)


def _index_path(root: Path, name: str) -> Path:
    return root / name / "registry.json"


def _read_index(root: Path, name: str) -> dict[str, Any]:
    p = _index_path(root, name)
    if p.exists():
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data
    return {"name": name, "latest": None, "stages": {}, "versions": {}}


def _write_index(root: Path, name: str, index: dict[str, Any]) -> None:
    p = _index_path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def register_model(
    model: Any,
    *,
    name: str,
    version: str,
    metrics: dict[str, float] | None = None,
    feature_cols: list[str] | None = None,
    data_version: str | None = None,
    extra: dict[str, Any] | None = None,
    root: str | Path = DEFAULT_ROOT,
) -> Path:
    """Serialize a model + metadata and update the registry index. Returns the version dir."""
    root = Path(root)
    vdir = root / name / version
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, vdir / "model.joblib")
    meta = ModelMeta(
        name=name,
        version=version,
        created_at=datetime.now(UTC).isoformat(),
        metrics=metrics or {},
        feature_cols=feature_cols or [],
        data_version=data_version,
        extra=extra or {},
    )
    (vdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2, sort_keys=True), "utf-8")

    index = _read_index(root, name)
    index["versions"][version] = asdict(meta)
    index["latest"] = version
    _write_index(root, name, index)
    return vdir


def load_model(name: str, version: str = "latest", *, root: str | Path = DEFAULT_ROOT) -> Any:
    """Load a registered model (by version, a stage name, or ``"latest"``)."""
    root = Path(root)
    index = _read_index(root, name)
    resolved: str | None = version
    if version == "latest":
        resolved = index.get("latest")
    elif version in index.get("stages", {}):
        resolved = index["stages"][version]
    if not resolved or resolved not in index.get("versions", {}):
        raise KeyError(f"No model {name!r} version/stage {version!r} in registry.")
    return joblib.load(root / name / resolved / "model.joblib")


def list_models(name: str, *, root: str | Path = DEFAULT_ROOT) -> list[dict[str, Any]]:
    """List all registered versions' metadata for a model name (newest first by created_at)."""
    index = _read_index(Path(root), name)
    versions = list(index.get("versions", {}).values())
    return sorted(versions, key=lambda m: m.get("created_at", ""), reverse=True)


def promote_model(
    name: str, version: str, stage: str, *, root: str | Path = DEFAULT_ROOT
) -> None:
    """Assign a version to a stage (e.g. 'production'); updates the index and the version meta."""
    if stage not in {"none", "staging", "production", "archived"}:
        raise ValueError(f"Unknown stage {stage!r}.")
    root = Path(root)
    index = _read_index(root, name)
    if version not in index.get("versions", {}):
        raise KeyError(f"Version {version!r} not registered for {name!r}.")
    index["stages"][stage] = version
    index["versions"][version]["stage"] = stage
    _write_index(root, name, index)
