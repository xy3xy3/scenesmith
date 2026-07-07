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
# Functional-dependency sidecar (exported from the UD4/funeval source layer's
# ``functional_dependencies`` field, used_with + requires, keyed by HSSD hash).
# Used for partner-exclusion: an object's keep-clear is not "intruded" by a
# category it is *meant* to be used with (chair<->table, sofa<->coffee table).
_PARTNER_FILE = _DATA_DIR / "functional_partners_index.json"

# Clearance types whose keep-clear region is *above* the object footprint
# (vertical headroom) rather than extending out a side.
_VERTICAL_TYPES = {"上方站立", "above", "overhead"}
# Symmetric / no-front objects: keep-clear ring on all four horizontal sides.
_RING_DIRECTIONS = {"四周", "ring", "all"}
_STRUCTURAL_BLOCKER_CATEGORIES = {"floor", "wall", "ceiling", "door", "window"}


# ---------------------------------------------------------------------------
# Asset-level clearance policy (placement-aware annotation, ZERO scene logic).
#
# The raw per-asset annotations over-reserve in two ways that block good,
# *intended* placements. Both are fixed here at the asset layer — by adjusting
# what keep-clear an asset advertises — so the scene check stays a dumb
# AABB-intrusion test and never needs to reason about relations or walls:
#
#   1. Seating (落座) reserves a deep front zone (0.6 m) pointing at the very
#      surface it pairs with (chair -> dining table, sofa -> coffee table,
#      swivel_chair even reserves a full ring). But that front face is a
#      *tuck/shared* zone: idle the seat tucks under the surface; occupied, the
#      person tucks their legs under and needs almost no front gap. The deep
#      front zone makes the paired table read as an intruder -> false fail.
#      Fix: collapse to a single front side with a token "tuck" depth, so any
#      normal table/desk gap (>= the tuck depth) no longer counts as intrusion.
#
#   2. Beds / daybeds / cribs reserve a four-side *ring*, so a bed correctly
#      pushed against a wall is flagged as intruded. A bed is anchored
#      furniture meant to back against walls on up to three sides; it needs one
#      accessible side, not a ring. Fix: collapse the ring to a single front
#      access side (false-fail rate from 4/4 sides down to at most 1).
#
# These adjustments depend only on (category, clearance type, direction) — they
# are properties of the asset, not of any scene.
# ---------------------------------------------------------------------------

# Seating clearance type: front is a tuck/shared face with the paired surface.
_SEATING_TYPES = {"落座"}
# "面前几乎不需要净空" — a token tuck gap. A paired table/desk/coffee-table at
# any normal distance (>= this) is no longer treated as a clearance intrusion;
# only a seat literally jammed within this gap of its surface still trips.
_SEATING_FRONT_DEPTH_M = 0.10

# Categories whose raw clearance is dominated by false positives in real
# layouts, so no floor keep-clear is reserved at all:
#   * Anchored sleeping furniture — designed to back against walls on up to
#     three sides, with nightstands intentionally abutting. The get-in side is
#     scene-dependent, and a single asset-frame box cannot express "any one side
#     free", so reserving any side mostly produces false fails (measured on real
#     scenes: the chosen front side lands on a wall or the bed's own nightstand).
#   * Wall-mounted decor — a painting / mirror / sconce is not a floor-layout
#     constraint; furniture below it is normal, and its approach zone projects
#     into the wall it hangs on (front convention is wall-ward for wall-mounted
#     assets), so it can only ever flag that wall.
# These are asset properties (anchored vs free-standing, wall-mounted vs floor),
# independent of any scene.
_SUPPRESS_FLOOR_CLEARANCE_CATS = {
    # anchored sleeping furniture
    "bed", "double_bed", "king_bed", "queen_bed", "twin_bed", "single_bed",
    "bunk_bed", "daybed", "round_daybed", "trundle_bed", "toddler_bed", "crib",
    # wall-mounted decor
    "wall_art", "wall_mirror", "wall_lamp", "wall_sconce", "wall_shelf",
    "wall_hook_rack", "wall_clock", "mirror", "picture_frame", "painting",
    "wall_decor", "window_curtain", "curtain",
}


def _apply_asset_clearance_policy(na: dict[str, Any]) -> dict[str, Any]:
    """Refine a raw non-articulated annotation into a placement-aware one.

    Pure asset-level: depends only on ``(cat, type, dir)``. Returns a
    shallow-copied, adjusted record; the human-readable list of adjustments is
    attached under ``_policy`` for auditability (empty/absent = unchanged).
    """
    typ = na.get("type")
    cat = str(na.get("cat") or "").lower()
    direction = str(na.get("dir") or "")
    raw_depth = float(na.get("depth") or 0.0)
    out = dict(na)
    notes: list[str] = []

    if cat in _SUPPRESS_FLOOR_CLEARANCE_CATS:
        out["dir"] = ""
        out["depth"] = 0.0
        notes.append(
            f"{cat}: floor clearance suppressed "
            f"(anchored/wall-mounted; normal adjacency, not a layout constraint)"
        )
    elif typ in _SEATING_TYPES:
        if direction != "前":
            out["dir"] = "前"
            notes.append(f"seating dir {direction or '∅'!r}->'前' (drop ring/back)")
        tuck_depth = min(raw_depth, _SEATING_FRONT_DEPTH_M)
        if tuck_depth != raw_depth:
            out["depth"] = tuck_depth
            notes.append(f"seating depth {raw_depth}->{tuck_depth} (front=tuck zone)")

    if notes:
        out["_policy"] = notes
    return out


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


@lru_cache(maxsize=1)
def _partner_index() -> dict[str, Any]:
    return _load_json(_PARTNER_FILE).get("items", {})


# Coarse furniture families. Functional dependencies name specific categories
# (chair -> dining_table/desk; sofa -> coffee_table); to apply partner-exclusion
# robustly across the differing vocabularies of the clearance index, the scene,
# and the synset targets, both the partner targets and a candidate intruder's
# category are normalized to one of these families before matching.
_CAT_FAMILY: dict[str, str] = {}
for _fam, _members in {
    "seat": (
        "chair armchair sofa couch loveseat sectional settee swivel_chair "
        "office_chair desk_chair dining_chair side_chair accent_chair "
        "lounge_chair rocking_chair recliner bench stool bar_stool counter_stool "
        "ottoman footstool pouf pouffe beanbag_chair ball_chair"
    ),
    "surface": (
        "table dining_table coffee_table end_table side_table console_table "
        "conference_table accent_table nightstand bedside_table desk writing_desk "
        "computer_desk dressing_table vanity countertop counter kitchen_island "
        "bar sideboard buffet credenza"
    ),
    "bed": (
        "bed double_bed king_bed queen_bed twin_bed single_bed bunk_bed daybed "
        "round_daybed trundle_bed toddler_bed crib"
    ),
}.items():
    for _m in _members.split():
        _CAT_FAMILY[_m] = _fam


def _family(cat: Any) -> str:
    """Normalize a category token (synset / instance id stripped) to a family."""
    token = str(cat or "").strip().lower()
    token = token.split(".")[0]  # strip ".n.01" synset suffix
    token = token.replace(" ", "_")  # funeval categories use spaces, the map uses _
    token = token.rsplit("_", 1)[0] if token[-1:].isdigit() else token  # drop _0
    return _CAT_FAMILY.get(token, token)


def _partner_families(asset_id: str | None) -> set[str]:
    """Families an asset is *meant* to be used with (from functional deps)."""
    rec = _partner_index().get(str(asset_id or ""))
    if not rec:
        return set()
    return {_family(p) for p in rec.get("partners", [])}


def _category_family_for_metadata(metadata: Any) -> str | None:
    """Resolve a scene object's coarse family from its HSSD hash / metadata."""
    aid = asset_id_from_metadata(metadata)
    rec = _partner_index().get(str(aid or "")) if aid else None
    cat = rec.get("cat") if rec else None
    if not cat and isinstance(metadata, dict):
        cat = metadata.get("category") or metadata.get("cat")
    return _family(cat) if cat else None


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
        na = _apply_asset_clearance_policy(na)
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
                "policy_applied": na.get("_policy") or [],
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

    # Functional partners: categories this asset is *meant* to be used with, so
    # their presence in its keep-clear zone is intended, not an intrusion.
    record["partner_families"] = sorted(_partner_families(key))
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
    """Map the object's facing direction through yaw, snap to nearest world axis.

    SceneSmith places objects with their *facing* direction along local +Y (its
    pose convention; verified against SceneSmith's own top-down GT renders, where
    the facing arrow points along local +Y). The clearance index's "-Y" note
    describes the raw HSSD mesh frame, but the keep-clear region must extend in
    the direction the placed object actually faces, so we project local +Y
    through the scene yaw here.

    Returns ``(axis, sign)`` where axis is 0 (X) or 1 (Y) and sign is +1/-1,
    i.e. the world-frame outward direction the keep-clear region extends toward.
    """
    theta = math.radians(yaw_deg or 0.0)
    # facing = local (0, +1) rotated by yaw about +Z
    wx = -math.sin(theta)
    wy = math.cos(theta)
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
    back to a direct provider lookup by the SceneSmith HSSD metadata key.
    """
    meta = obj.get("metadata") or {}
    record = meta.get("clearance")
    if isinstance(record, dict):
        return record
    # 2026-07-07: HSSD furniture stores the clearance join key as hssd_mesh_id
    # in normal SceneSmith scenes; use the shared resolver so direct critic runs
    # work even when asset_annotation has not mirrored metadata.clearance yet.
    return get_clearance_for_metadata(meta)


def _is_clearance_blocker_candidate(obj: dict[str, Any]) -> bool:
    """True for scene objects that can physically block an asset clearance zone."""
    # 2026-07-07: Floor/wall/ceiling are room structure, not movable intruders.
    # Keeping them as blockers made every floor-touching keep-clear region fail.
    category = str(obj.get("category_norm") or obj.get("category") or "").lower()
    object_type = str(obj.get("object_type") or "").lower()
    if category in _STRUCTURAL_BLOCKER_CATEGORIES:
        return False
    return object_type not in _STRUCTURAL_BLOCKER_CATEGORIES


def build_clearance_checks(objects: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one clearance check per object that reserves a keep-clear region.

    ``objects`` maps object id -> case_pack geometry dict (with ``bbox_world``,
    ``yaw_deg``, ``metadata``). The keep-clear region and intrusion verdict are
    computed here and embedded in the check so the rule evaluator is a trivial,
    deterministic passthrough (no VLM).
    """
    world_boxes = [
        {
            "id": oid,
            "bbox": obj.get("bbox_world"),
            "family": _category_family_for_metadata(obj.get("metadata") or {}),
        }
        for oid, obj in objects.items()
        if isinstance(obj.get("bbox_world"), dict)
        and _is_clearance_blocker_candidate(obj)
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
        # Exclude functional partners: an object placed where this asset is meant
        # to be used with it (chair at its table, sofa by its coffee table) is an
        # intended adjacency, not a clearance violation.
        partners = set(record.get("partner_families") or [])
        others = [
            box
            for box in world_boxes
            if box["id"] != oid and (box.get("family") not in partners)
        ]
        hits = intrusions(keep_clear, others)
        blockers = sorted({h["object_id"] for h in hits if h.get("object_id")})
        label = "fail" if blockers else "pass"
        ctype = record.get("clearance_type") or record.get("kind") or "clearance"
        name = obj.get("name") or oid
        checks.append(
            {
                "check_id": f"clearance__{oid}",
                "metric": "interaction_clearance",
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
        "metric": "interaction_clearance",
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
