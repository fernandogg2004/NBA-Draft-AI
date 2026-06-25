"""Provenance records: every fetched artifact carries where/when/under-what-license it came.

Written as a JSON sidecar next to each cached payload so the master dataset's lineage is
auditable and licenses are never lost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte payload (content addressing + integrity)."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Provenance:
    """Lineage for one fetched artifact."""

    source_id: str
    url: str
    retrieved_at: str            # ISO-8601 UTC
    license: str
    sha256: str
    n_bytes: int
    attribution_required: bool
    params: dict[str, str] | None = None

    @staticmethod
    def create(
        *,
        source_id: str,
        url: str,
        data: bytes,
        license: str,
        attribution_required: bool,
        params: dict[str, str] | None = None,
    ) -> Provenance:
        return Provenance(
            source_id=source_id,
            url=url,
            retrieved_at=datetime.now(UTC).isoformat(),
            license=license,
            sha256=sha256_bytes(data),
            n_bytes=len(data),
            attribution_required=attribution_required,
            params=params,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
