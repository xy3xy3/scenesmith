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
DEFAULT_CLEARANCE_DIR = Path(__file__).resolve().parent / "clearance_data"


def normalize_hssd_id(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("hssd:"):
        value = value.split(":", 1)[1]
    return value


class AssetLibraryAnnotationStore:
    """In-process search library for asset-library and clearance annotations."""

    def __init__(
        self,
        lookup_path: str | Path = DEFAULT_LOOKUP,
        clearance_dir: str | Path = DEFAULT_CLEARANCE_DIR,
    ) -> None:
        self.lookup_path = Path(lookup_path)
        self.clearance_dir = Path(clearance_dir)
        self._records: dict[str, dict[str, Any]] | None = None
        self._nonartic_clearance: dict[str, dict[str, Any]] | None = None
        self._artic_clearance: dict[str, dict[str, Any]] | None = None
        self._functional_partners: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._records is None:
            opener = gzip.open if self.lookup_path.suffix == ".gz" else open
            with opener(self.lookup_path, "rt", encoding="utf-8") as f:
                self._records = json.load(f)
        return self._records

    def _load_clearance_items(self, filename: str) -> dict[str, dict[str, Any]]:
        with (self.clearance_dir / filename).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("items", {})

    def _nonartic(self) -> dict[str, dict[str, Any]]:
        if self._nonartic_clearance is None:
            self._nonartic_clearance = self._load_clearance_items(
                "nonartic_clearance_index.json"
            )
        return self._nonartic_clearance

    def _artic(self) -> dict[str, dict[str, Any]]:
        if self._artic_clearance is None:
            self._artic_clearance = self._load_clearance_items(
                "artic_clearance_index.json"
            )
        return self._artic_clearance

    def _partners(self) -> dict[str, dict[str, Any]]:
        if self._functional_partners is None:
            self._functional_partners = self._load_clearance_items(
                "functional_partners_index.json"
            )
        return self._functional_partners

    def get_clearance_annotations(self, hssd_id: str) -> dict[str, Any]:
        """Return clearance-session annotations keyed by HSSD id.

        This is the asset-level lookup side of the interaction_clearance work:
        human-anchored non-artic keep-clear, articulated swept-volume envelopes,
        and functional-dependency partner categories used for exclusion.
        """
        normalized = normalize_hssd_id(hssd_id)
        nonartic = self._nonartic().get(normalized)
        artic = self._artic().get(normalized)
        partners = self._partners().get(normalized)
        return {
            "metric": "interaction_clearance",
            "source_session": "clearance-plan-execution-w1-w2",
            "asset_id": normalized,
            "has_keep_clear": nonartic is not None or artic is not None,
            "has_nonarticulated_keep_clear": nonartic is not None,
            "has_articulated_swept_volume": artic is not None,
            "has_functional_partners": partners is not None,
            "nonarticulated_keep_clear": nonartic,
            "articulated_swept_volume": artic,
            "functional_partners": partners,
        }

    def get(self, hssd_id: str) -> dict[str, Any] | None:
        normalized = normalize_hssd_id(hssd_id)
        record = self._load().get(normalized)
        if record is None:
            return None
        out = dict(record)
        out["interaction_clearance"] = self.get_clearance_annotations(normalized)
        return out

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
