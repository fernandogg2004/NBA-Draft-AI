"""Local, content-addressed file cache for fetched artifacts.

Caching is mandatory (instructions.md Phase 1): re-runs must not re-hit a source. Each entry
is stored under ``data/raw/<source_id>/<key>.<ext>`` with a ``.prov.json`` sidecar carrying
its :class:`~nba_draft.ingestion.provenance.Provenance`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nba_draft.ingestion.provenance import Provenance


def cache_key(url: str, params: dict[str, str] | None = None) -> str:
    """Stable filesystem-safe key for a (url, params) request."""
    canonical = url
    if params:
        canonical += "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class FileCache:
    """Filesystem cache rooted at a directory (typically ``data/raw``)."""

    def __init__(self, root: str | Path, *, ext: str = "bin") -> None:
        self.root = Path(root)
        self.ext = ext.lstrip(".")

    def _paths(self, source_id: str, key: str) -> tuple[Path, Path]:
        base = self.root / source_id
        return base / f"{key}.{self.ext}", base / f"{key}.prov.json"

    def has(self, source_id: str, key: str) -> bool:
        payload, _ = self._paths(source_id, key)
        return payload.exists()

    def get(self, source_id: str, key: str) -> bytes | None:
        """Return cached payload bytes, or ``None`` if absent."""
        payload, _ = self._paths(source_id, key)
        if not payload.exists():
            return None
        return payload.read_bytes()

    def put(self, source_id: str, key: str, data: bytes, provenance: Provenance) -> Path:
        """Write payload + provenance sidecar; return the payload path."""
        payload, prov = self._paths(source_id, key)
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(data)
        prov.write_text(provenance.to_json(), encoding="utf-8")
        return payload
