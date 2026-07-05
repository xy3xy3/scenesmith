"""Lookup merged asset-library annotations by HSSD id.

The default lookup artifact is the generated HSSD-default asset-library
annotation layer bundled with this critic package.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

DEFAULT_LOOKUP = (
    Path(__file__).resolve().parent
    / "asset_annotation_data"
    / "hssd_annotation_lookup.json.gz"
)


def normalize_hssd_id(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("hssd:"):
        value = value.split(":", 1)[1]
    return value


class AssetLibraryAnnotationStore:
    """In-process search library for canonical-front/relation/DOF annotations."""

    def __init__(self, lookup_path: str | Path = DEFAULT_LOOKUP) -> None:
        self.lookup_path = Path(lookup_path)
        self._records: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._records is None:
            opener = gzip.open if self.lookup_path.suffix == ".gz" else open
            with opener(self.lookup_path, "rt", encoding="utf-8") as f:
                self._records = json.load(f)
        return self._records

    def get(self, hssd_id: str) -> dict[str, Any] | None:
        return self._load().get(normalize_hssd_id(hssd_id))

    def require(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        record = self.get(normalized)
        if record is None:
            raise KeyError(f"HSSD id not found in annotation lookup: {normalized}")
        return record

    def search_category(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        q = str(query or "").strip().lower().replace("_", " ")
        if not q:
            return []
        matches = []
        for record in self._load().values():
            haystack = " ".join(
                str(record.get(key) or "")
                for key in ("category", "category_key", "asset_uid")
            ).lower().replace("_", " ")
            if q in haystack:
                matches.append(record)
                if len(matches) >= limit:
                    break
        return matches


_DEFAULT_STORE: AssetLibraryAnnotationStore | None = None


def get_hssd_asset_annotations(hssd_id: str) -> dict[str, Any] | None:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = AssetLibraryAnnotationStore()
    return _DEFAULT_STORE.get(hssd_id)
