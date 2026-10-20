"""Data manifests: source URL, pull timestamp, counts and SHA256 for every download."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class DataManifest:
    """Provenance record written next to every raw download."""

    source: str
    source_url: str
    pulled_at: str
    files: list[dict[str, Any]] = field(default_factory=list)
    file_count: int = 0
    row_count: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        source: str,
        source_url: str,
        files: list[Path],
        row_count: int | None = None,
        notes: dict[str, Any] | None = None,
        asset_ids: dict[str, str] | None = None,
    ) -> DataManifest:
        """Hash ``files`` and assemble a manifest stamped with the current UTC time."""
        ids = asset_ids or {}
        entries = [
            {
                "path": p.as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "ee_asset_id": ids.get(p.name),
            }
            for p in files
        ]
        return cls(
            source=source,
            source_url=source_url,
            pulled_at=datetime.now(UTC).isoformat(timespec="seconds"),
            files=entries,
            file_count=len(entries),
            row_count=row_count,
            notes=notes or {},
        )

    def write(self, path: Path) -> Path:
        """Write the manifest as pretty JSON and return its path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path
