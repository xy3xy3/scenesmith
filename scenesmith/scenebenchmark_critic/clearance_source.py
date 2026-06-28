"""Functional-clearance provider for the embedded SceneBenchmark critic.

This module is the core of the "clearance service": it loads the pre-computed
clearance annotations (human-anchored non-articulated clearance + articulated
swept-volume opening envelopes) keyed by HSSD ``asset_id`` and projects an
object's *local* keep-clear region into a *world-frame* axis-aligned box that
downstream checks can test for intrusion.

Design notes
------------
* Pure Python (json + math only) so it imports without the heavy SceneSmith
  runtime (Drake / Blender / trimesh) and stays unit-testable in isolation.
* Two data sources, both keyed by the 40-char HSSD asset hash:
    - ``nonartic_clearance_index.json``: human-anchored clearance for
      non-articulated objects (6092). Fields: type/dir/depth/width/height/conf.
    - ``artic_clearance_index.json``: articulated swept-volume opening
      envelopes (2120). ``expand`` = swept/static extent ratio per axis.
* World frame convention matches the critic geometry (``bbox_world``):
  X/Y = floor plane, Z = up. Object local "front" is -Y (annotation convention),
  rotated by the object's yaw and snapped to the nearest world axis.
"""

from __future__ import annotations

import json
import logging
import math

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

console_logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "clearance_data"
_NONARTIC_FILE = _DATA_DIR / "nonartic_clearance_index.json"
_ARTIC_FILE = _DATA_DIR / "artic_clearance_index.json"

# Clearance types whose keep-clear region is *above* the object footprint
# (vertical headroom) rather than extending out a side.
_VERTICAL_TYPES = {"上方站立", "above", "overhead"}
# Symmetric / no-front objects: keep-clear ring on all four horizontal sides.
_RING_DIRECTIONS = {"四周", "ring", "all"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        console_logger.warning("clearance index not found: %s", path)
        return {"items": {}}
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        console_logger.warning("failed to read clearance index %s: %s", path, exc)
        return {"items": {}}


@lru_cache(maxsize=1)
def _nonartic_index() -> dict[str, Any]:
    return _load_json(_NONARTIC_FILE).get("items", {})


@lru_cache(maxsize=1)
def _artic_index() -> dict[str, Any]:
    return _load_json(_ARTIC_FILE).get("items", {})


def available() -> bool:
    """True when at least one clearance index is loaded with entries."""
    return bool(_nonartic_index()) or bool(_artic_index())


def stats() -> dict[str, int]:
    """Coverage stats — used by the HTTP service / diagnostics."""
    return {
        "nonarticulated": len(_nonartic_index()),
        "articulated": len(_artic_index()),
    }


def get_clearance(asset_id: str | None) -> dict[str, Any] | None:
    """Return the unified clearance record for an HSSD ``asset_id``.

    Looks up the non-articulated human-anchored clearance first; if the asset
    is articulated, attaches the swept-volume opening envelope as well. Returns
    ``None`` when the asset has no clearance requirement on record.
    """
    if not asset_id:
        return None
    key = str(asset_id)
    na = _nonartic_index().get(key)
    ar = _artic_index().get(key)
    if na is None and ar is None:
        return None

    record: dict[str, Any] = {"asset_id": key}
    if na is not None:
        record.update(
            {
                "kind": "nonarticulated",
                "clearance_type": na.get("type"),
                "direction": na.get("dir"),
                "depth_m": float(na.get("depth") or 0.0),
                "width_m": float(na.get("width") or 0.0),
                "height_m": float(na.get("height") or 0.0),
                "confidence": na.get("conf"),
                "inherits_from_support": bool(na.get("inherits")),
                "object_bbox_m": list(na.get("bbox") or []),
                "category": na.get("cat"),
            }
        )
    if ar is not None:
        record["articulated"] = {
            "kind": "articulated",
            "category": ar.get("cat"),
            "tier": ar.get("tier"),
            "object_bbox_m": list(ar.get("bbox") or []),
            "expand": list(ar.get("expand") or [1.0, 1.0, 1.0]),
            "vol_bloat": float(ar.get("bloat") or 1.0),
            "n_movable": ar.get("n_movable"),
        }
        if na is None:
            record["kind"] = "articulated"
    return record


# SceneSmith scene objects carry the 40-char HSSD asset hash under
# ``hssd_mesh_id`` (asset_source == "hssd"); some code paths instead use
# ``asset_id``/``object_id``. Resolve against the real keys in that order so the
# clearance lookup actually fires on generated scenes (verified against the
# critic_probe outputs: hssd_mesh_id hits, asset_id never does).
_ASSET_ID_KEYS = ("hssd_mesh_id", "asset_id", "object_id")


def asset_id_from_metadata(metadata: Any) -> str | None:
    """Extract the HSSD asset id a clearance record is keyed by, if present."""
    if not isinstance(metadata, dict):
        return None
    for key in _ASSET_ID_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def get_clearance_for_metadata(metadata: Any) -> dict[str, Any] | None:
    """Look up the clearance record for a scene object's ``metadata`` dict."""
    return get_clearance(asset_id_from_metadata(metadata))


def _front_world_axis(yaw_deg: float) -> tuple[int, int]:
    """Map object-local front (-Y) through yaw, snap to nearest world axis.

    Returns ``(axis, sign)`` where axis is 0 (X) or 1 (Y) and sign is +1/-1,
    i.e. the world-frame outward direction the keep-clear region extends toward.
    """
    theta = math.radians(yaw_deg or 0.0)
    # local front (0, -1) rotated by yaw about +Z
    wx = math.sin(theta)
    wy = -math.cos(theta)
    if abs(wx) >= abs(wy):
        return 0, (1 if wx >= 0 else -1)
    return 1, (1 if wy >= 0 else -1)


def _expand_side(
    bmin: list[float], bmax: list[float], axis: int, sign: int, depth: float
) -> tuple[list[float], list[float]]:
    """Grow an AABB outward by ``depth`` on one horizontal side."""
    lo = list(bmin)
    hi = list(bmax)
    if sign >= 0:
        lo[axis] = hi[axis]
        hi[axis] = hi[axis] + depth
    else:
        hi[axis] = lo[axis]
        lo[axis] = lo[axis] - depth
    return lo, hi


def project_keep_clear(
    record: dict[str, Any] | None,
    bbox_world: dict[str, Any] | None,
    yaw_deg: float = 0.0,
) -> list[dict[str, Any]]:
    """Project a clearance record into world-frame keep-clear AABB(s).

    Each returned region is ``{"min": [x,y,z], "max": [x,y,z], "side": str}``.
    Returns an empty list when there is nothing to reserve (e.g. clearance is
    inherited from a supporting surface, or geometry is missing).
    """
    if not record or not isinstance(bbox_world, dict):
        return []
    if record.get("inherits_from_support"):
        # Small supported items reserve no independent box (gating rule).
        return []
    bmin = bbox_world.get("min")
    bmax = bbox_world.get("max")
    if not (isinstance(bmin, (list, tuple)) and isinstance(bmax, (list, tuple))):
        return []
    bmin = [float(v) for v in bmin[:3]]
    bmax = [float(v) for v in bmax[:3]]

    ctype = record.get("clearance_type")
    direction = record.get("direction") or ""
    depth = float(record.get("depth_m") or 0.0)
    height = float(record.get("height_m") or 0.0)

    regions: list[dict[str, Any]] = []

    # Vertical headroom (rug / mat / step stool): box sits above the footprint.
    if ctype in _VERTICAL_TYPES:
        top = bmax[2]
        regions.append(
            {
                "min": [bmin[0], bmin[1], top],
                "max": [bmax[0], bmax[1], top + (height or 1.9)],
                "side": "above",
            }
        )
        return regions

    if depth <= 0.0:
        return []

    # Symmetric ring: reserve on all four horizontal sides.
    if direction in _RING_DIRECTIONS or "四周" in direction:
        for axis, sign, name in ((0, 1, "+x"), (0, -1, "-x"), (1, 1, "+y"), (1, -1, "-y")):
            lo, hi = _expand_side(bmin, bmax, axis, sign, depth)
            if height:
                hi[2] = lo[2] + height
            regions.append({"min": lo, "max": hi, "side": name})
        return regions

    # Directional (front / operate / sit): reserve on the front side.
    axis, sign = _front_world_axis(yaw_deg)
    lo, hi = _expand_side(bmin, bmax, axis, sign, depth)
    if height:
        hi[2] = lo[2] + height
    regions.append({"min": lo, "max": hi, "side": f"front:{'+-'[sign<0]}{'xy'[axis]}"})
    return regions


def aabb_overlap_volume(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Axis-aligned box intersection volume (0 when disjoint)."""
    amin, amax = a["min"], a["max"]
    bmin, bmax = b["min"], b["max"]
    vol = 1.0
    for i in range(3):
        lo = max(float(amin[i]), float(bmin[i]))
        hi = min(float(amax[i]), float(bmax[i]))
        if hi <= lo:
            return 0.0
        vol *= hi - lo
    return vol


def aabb_volume(a: dict[str, Any]) -> float:
    amin, amax = a["min"], a["max"]
    vol = 1.0
    for i in range(3):
        vol *= max(0.0, float(amax[i]) - float(amin[i]))
    return vol


_CONFIDENCE_SCORE = {"high": 0.9, "med": 0.6, "medium": 0.6, "low": 0.3}


def _confidence_score(value: Any) -> float:
    return _CONFIDENCE_SCORE.get(str(value or "").strip().lower(), 0.5)


def intrusions(
    keep_clear: Iterable[dict[str, Any]],
    others: Iterable[dict[str, Any]],
    *,
    min_overlap_m3: float = 1e-4,
) -> list[dict[str, Any]]:
    """Find objects intruding into any keep-clear region.

    ``others`` is an iterable of ``{"id": str, "bbox": {"min","max"}}``. Returns
    a list of ``{"object_id", "side", "overlap_m3", "overlap_frac"}`` for each
    intruding (region, object) pair above ``min_overlap_m3``.
    """
    keep_clear = list(keep_clear)
    hits: list[dict[str, Any]] = []
    for region in keep_clear:
        region_vol = aabb_volume(region) or 1.0
        for other in others:
            bbox = other.get("bbox")
            if not isinstance(bbox, dict):
                continue
            overlap = aabb_overlap_volume(region, bbox)
            if overlap > min_overlap_m3:
                hits.append(
                    {
                        "object_id": other.get("id"),
                        "side": region.get("side"),
                        "overlap_m3": round(overlap, 5),
                        "overlap_frac": round(overlap / region_vol, 4),
                    }
                )
    return hits


# ---------------------------------------------------------------------------
# Critic integration: build clearance checks from a case_pack and score them.
# Kept here (rather than in checks.py / vendor.rules) so the geometry logic
# stays importable without the heavy SceneSmith runtime, hence unit-testable.
# ---------------------------------------------------------------------------


def _object_clearance_record(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a clearance record for a case_pack object dict.

    Prefers the record mirrored into metadata during asset annotation; falls
    back to a direct provider lookup by ``asset_id``.
    """
    meta = obj.get("metadata") or {}
    record = meta.get("clearance")
    if isinstance(record, dict):
        return record
    return get_clearance(meta.get("asset_id"))


def build_clearance_checks(objects: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one clearance check per object that reserves a keep-clear region.

    ``objects`` maps object id -> case_pack geometry dict (with ``bbox_world``,
    ``yaw_deg``, ``metadata``). The keep-clear region and intrusion verdict are
    computed here and embedded in the check so the rule evaluator is a trivial,
    deterministic passthrough (no VLM).
    """
    world_boxes = [
        {"id": oid, "bbox": obj.get("bbox_world")}
        for oid, obj in objects.items()
        if isinstance(obj.get("bbox_world"), dict)
    ]
    checks: list[dict[str, Any]] = []
    for oid, obj in objects.items():
        record = _object_clearance_record(obj)
        if not record:
            continue
        keep_clear = project_keep_clear(
            record, obj.get("bbox_world"), float(obj.get("yaw_deg") or 0.0)
        )
        if not keep_clear:
            continue
        others = [box for box in world_boxes if box["id"] != oid]
        hits = intrusions(keep_clear, others)
        blockers = sorted({h["object_id"] for h in hits if h.get("object_id")})
        label = "fail" if blockers else "pass"
        ctype = record.get("clearance_type") or record.get("kind") or "clearance"
        name = obj.get("name") or oid
        checks.append(
            {
                "check_id": f"clearance__{oid}",
                "metric": "clearance",
                "subject_id": oid,
                "target_ids": blockers,
                "clearance_type": ctype,
                "priority_weight": 0.8,
                "scoring_tier": "core",
                "question": (
                    f"Is the {ctype} clearance around {name} kept unobstructed?"
                ),
                "evidence_refs": ["scene_geometry", "clearance_index"],
                "clearance_result": {
                    "label": label,
                    "blocking_objects": blockers,
                    "keep_clear": keep_clear,
                    "intrusions": hits,
                    "confidence": _confidence_score(record.get("confidence")),
                    "clearance_type": ctype,
                    "direction": record.get("direction"),
                },
            }
        )
    return checks


def evaluate_clearance(check: dict[str, Any]) -> dict[str, Any]:
    """Reshape the embedded clearance verdict into a critic result row."""
    cr = check.get("clearance_result") or {}
    label = str(cr.get("label") or "unknown")
    blockers = list(cr.get("blocking_objects") or [])
    if label == "pass":
        reason = "Functional clearance zone is unobstructed."
    elif blockers:
        reason = (
            f"{len(blockers)} object(s) intrude into the "
            f"{cr.get('clearance_type') or 'clearance'} zone: "
            f"{', '.join(str(b) for b in blockers)}."
        )
    else:
        reason = "Clearance could not be determined."
    return {
        "check_id": check.get("check_id"),
        "metric": "clearance",
        "label": label,
        "primary_object": check.get("subject_id"),
        "blocking_objects": blockers,
        "confidence": float(cr.get("confidence") or 0.0),
        "reason": reason,
        "diagnostics": {
            "clearance_type": cr.get("clearance_type"),
            "direction": cr.get("direction"),
            "keep_clear": cr.get("keep_clear"),
            "intrusions": cr.get("intrusions"),
        },
        "scoring_tier": check.get("scoring_tier", "core"),
    }
