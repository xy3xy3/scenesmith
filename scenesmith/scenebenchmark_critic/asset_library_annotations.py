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
DEFAULT_UNIFIED_AFFORDANCE_DIR = Path(
    "/data/share/ud4scenesmith/ud4scenesmith_3_2_final_20260513/"
    "affordance_annotations/unified_layer_v0_1"
)
DEFAULT_OPERATION_SPACE_DIR = Path(
    "/data/share/ud4scenesmith/ud4scenesmith_3_2_final_20260513/"
    "affordance_annotations/operation_space_hssd_official"
)
DEFAULT_NONARTIC_CLEARANCE_V2 = Path(
    "/data/250010098/NONARTIC_CLEARANCE_ANNOTATIONS_V2.json"
)
DEFAULT_OFFICIAL_COMBINED_CLEARANCE = Path(
    "/data/share/ud4scenesmith/clearance_fullrun_20260606/"
    "official_combined_clearance.json"
)
DEFAULT_HSSD_ARTICULATION_CLEARANCE_RUN = Path(
    "/data/250010098/clearance_retrieval_pilot_20260609/"
    "hssd_clearance_run_results.json"
)
DEFAULT_HSSD_CLEARANCE_VOXEL_RESULTS = Path(
    "/data/250010098/clearance_retrieval_pilot_20260609/"
    "hssd_clearance_voxel_results.json"
)


def normalize_hssd_id(value: str) -> str:
    value = str(value or "").strip()
    if value.lower().startswith("hssd:"):
        value = value.split(":", 1)[1]
    return value


class AssetLibraryAnnotationStore:
    """In-process search library for asset-library and clearance annotations."""

    def __init__(
        self,
        lookup_path: str | Path = DEFAULT_LOOKUP,
        clearance_dir: str | Path = DEFAULT_CLEARANCE_DIR,
        unified_affordance_dir: str | Path = DEFAULT_UNIFIED_AFFORDANCE_DIR,
        operation_space_dir: str | Path = DEFAULT_OPERATION_SPACE_DIR,
        nonartic_clearance_v2_path: str | Path = DEFAULT_NONARTIC_CLEARANCE_V2,
        official_combined_clearance_path: str | Path = DEFAULT_OFFICIAL_COMBINED_CLEARANCE,
        hssd_articulation_clearance_run_path: str | Path = DEFAULT_HSSD_ARTICULATION_CLEARANCE_RUN,
        hssd_clearance_voxel_results_path: str | Path = DEFAULT_HSSD_CLEARANCE_VOXEL_RESULTS,
    ) -> None:
        self.lookup_path = Path(lookup_path)
        self.clearance_dir = Path(clearance_dir)
        self.unified_affordance_dir = Path(unified_affordance_dir)
        self.operation_space_dir = Path(operation_space_dir)
        self.nonartic_clearance_v2_path = Path(nonartic_clearance_v2_path)
        self.official_combined_clearance_path = Path(official_combined_clearance_path)
        self.hssd_articulation_clearance_run_path = Path(hssd_articulation_clearance_run_path)
        self.hssd_clearance_voxel_results_path = Path(hssd_clearance_voxel_results_path)
        self._records: dict[str, dict[str, Any]] | None = None
        self._nonartic_clearance: dict[str, dict[str, Any]] | None = None
        self._artic_clearance: dict[str, dict[str, Any]] | None = None
        self._functional_partners: dict[str, dict[str, Any]] | None = None
        self._unified_affordance_index: dict[str, Path] | None = None
        self._nonartic_clearance_v2: dict[str, dict[str, Any]] | None = None
        self._official_combined_clearance: dict[str, dict[str, Any]] | None = None
        self._hssd_articulation_clearance_run: dict[str, dict[str, Any]] | None = None
        self._hssd_clearance_voxel_results_cache: dict[str, dict[str, Any]] | None = None

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

    def _load_json_if_present(self, path: Path) -> Any:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _unified_index(self) -> dict[str, Path]:
        if self._unified_affordance_index is None:
            index_path = self.unified_affordance_dir / "index.jsonl"
            records: dict[str, Path] = {}
            if index_path.exists():
                with index_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        asset_id = normalize_hssd_id(row.get("asset_id", ""))
                        records[asset_id] = self.unified_affordance_dir / row["record"]
            self._unified_affordance_index = records
        return self._unified_affordance_index

    def _nonartic_v2(self) -> dict[str, dict[str, Any]]:
        if self._nonartic_clearance_v2 is None:
            data = self._load_json_if_present(self.nonartic_clearance_v2_path) or {}
            annotations = data.get("annotations") or []
            self._nonartic_clearance_v2 = {
                str(item.get("object_id")): item
                for item in annotations
                if isinstance(item, dict) and item.get("object_id")
            }
        return self._nonartic_clearance_v2

    def _official_combined(self) -> dict[str, dict[str, Any]]:
        if self._official_combined_clearance is None:
            data = self._load_json_if_present(self.official_combined_clearance_path) or {}
            self._official_combined_clearance = data if isinstance(data, dict) else {}
        return self._official_combined_clearance

    def _hssd_articulation_run(self) -> dict[str, dict[str, Any]]:
        if self._hssd_articulation_clearance_run is None:
            data = self._load_json_if_present(self.hssd_articulation_clearance_run_path) or []
            records: dict[str, dict[str, Any]] = {}
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    asset_id = normalize_hssd_id(item.get("hssd_id", ""))
                    if asset_id:
                        records[asset_id] = item
            self._hssd_articulation_clearance_run = records
        return self._hssd_articulation_clearance_run

    def _hssd_clearance_voxel_results(self) -> dict[str, dict[str, Any]]:
        if self._hssd_clearance_voxel_results_cache is None:
            data = self._load_json_if_present(self.hssd_clearance_voxel_results_path) or []
            records: dict[str, dict[str, Any]] = {}
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    asset_id = normalize_hssd_id(item.get("hssd_id", ""))
                    if asset_id:
                        records[asset_id] = item
            self._hssd_clearance_voxel_results_cache = records
        return self._hssd_clearance_voxel_results_cache

    def get_unified_affordance_annotations(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        record_path = self._unified_index().get(normalized)
        if record_path is None or not record_path.exists():
            return {
                "available": False,
                "source_layer": "unified_layer_v0_1",
                "asset_id": normalized,
            }
        record = self._load_json_if_present(record_path)
        return {
            "available": record is not None,
            "source_layer": "unified_layer_v0_1",
            "asset_id": normalized,
            "record_path": str(record_path),
            "record": record,
        }

    def get_operation_space_annotations(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        record_path = self.operation_space_dir / "records" / f"{normalized}.json"
        record = self._load_json_if_present(record_path)
        summary = self._load_json_if_present(self.operation_space_dir / "SUMMARY.json")
        return {
            "available": record is not None,
            "source_layer": "operation_space_hssd_official",
            "asset_id": normalized,
            "record_path": str(record_path) if record_path.exists() else None,
            "summary": summary,
            "record": record,
        }

    def get_clearance_region_annotations(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        return {
            "asset_id": normalized,
            "nonartic_clearance_v2": self._nonartic_v2().get(normalized),
            "official_combined_clearance": self._official_combined().get(normalized),
            "hssd_articulation_clearance_run": self._hssd_articulation_run().get(normalized),
            "hssd_clearance_voxel_metrics": self._hssd_clearance_voxel_results().get(normalized),
            "sources": {
                "nonartic_clearance_v2": str(self.nonartic_clearance_v2_path),
                "official_combined_clearance": str(self.official_combined_clearance_path),
                "hssd_articulation_clearance_run": str(self.hssd_articulation_clearance_run_path),
                "hssd_clearance_voxel_metrics": str(self.hssd_clearance_voxel_results_path),
            },
        }

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
        articulation_run = self._hssd_articulation_run().get(normalized)
        voxel_metrics = self._hssd_clearance_voxel_results().get(normalized)
        return {
            "metric": "interaction_clearance",
            "source_session": "clearance-plan-execution-w1-w2",
            "asset_id": normalized,
            "has_keep_clear": nonartic is not None
            or artic is not None
            or articulation_run is not None
            or voxel_metrics is not None,
            "has_nonarticulated_keep_clear": nonartic is not None,
            "has_articulated_swept_volume": artic is not None,
            "has_hssd_articulation_clearance_run": articulation_run is not None,
            "has_hssd_clearance_voxel_metrics": voxel_metrics is not None,
            "has_functional_partners": partners is not None,
            "nonarticulated_keep_clear": nonartic,
            "articulated_swept_volume": artic,
            "hssd_articulation_clearance_run": articulation_run,
            "hssd_clearance_voxel_metrics": voxel_metrics,
            "functional_partners": partners,
        }

    def get(self, hssd_id: str) -> dict[str, Any] | None:
        normalized = normalize_hssd_id(hssd_id)
        record = self._load().get(normalized)
        # The bundled lookup now carries inline interaction_clearance and
        # post_replacement, so the record is self-sufficient for portability.
        # External layers below are OPTIONAL enrichment: when their absolute
        # source paths are missing (e.g. on a fresh clone) each getter returns
        # an ``available: False`` stub rather than failing, and the inline
        # bundled fields remain authoritative.
        interaction_clearance = self.get_clearance_annotations(normalized)
        ud4_affordance = self.get_unified_affordance_annotations(normalized)
        operation_space = self.get_operation_space_annotations(normalized)
        clearance_regions = self.get_clearance_region_annotations(normalized)
        if record is None:
            has_clearance_region = any(
                clearance_regions.get(key) is not None
                for key in (
                    "nonartic_clearance_v2",
                    "official_combined_clearance",
                    "hssd_articulation_clearance_run",
                    "hssd_clearance_voxel_metrics",
                )
            )
            if (
                not interaction_clearance["has_keep_clear"]
                and not interaction_clearance["has_functional_partners"]
                and not has_clearance_region
                and not ud4_affordance["available"]
                and not operation_space["available"]
            ):
                return None
            articulation_run = clearance_regions.get("hssd_articulation_clearance_run") or {}
            category = articulation_run.get("cat") or articulation_run.get("pnm_cat")
            out = {
                "asset_uid": f"hssd:{normalized}",
                "source_id": normalized,
                "source": "auxiliary_annotation_layers",
                "source_scope": "hssd_auxiliary_annotation_only",
                "schema_version": "auxiliary_annotation_lookup_v0",
                "category": category,
                "category_key": category,
                "provenance_meta": {
                    "note": (
                        "This HSSD id is not present in the generated asset-library "
                        "policy lookup, but auxiliary clearance/affordance layers "
                        "contain annotations for it."
                    )
                },
            }
        else:
            out = dict(record)
        # Authoritative interaction_clearance/post_replacement are the bundled
        # inline fields (portable). Only fill from external enrichment when the
        # bundled record lacks them (e.g. the auxiliary-only branch above).
        if not out.get("interaction_clearance"):
            out["interaction_clearance"] = interaction_clearance
        else:
            out["interaction_clearance_external"] = interaction_clearance
        out.setdefault("post_replacement",
                       {"articulated": False, "realization_kind": "static_only"})
        out["ud4_affordance"] = ud4_affordance
        out["operation_space"] = operation_space
        out["clearance_regions"] = clearance_regions
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
