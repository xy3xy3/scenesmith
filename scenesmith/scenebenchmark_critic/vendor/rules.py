"""Compact geometry-only evaluators for v1 SceneBenchmark critic metrics."""

from __future__ import annotations

import math

from typing import Any

LABEL_TO_SCORE = {"pass": 1.0, "degraded": 0.5, "fail": 0.0}


def run_case_pack_checks(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = case_pack.get("scene_geometry") or {}
    objects = {
        str(obj.get("id")): obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    }
    rooms = geometry.get("rooms") or []
    results: list[dict[str, Any]] = []
    for check in case_pack.get("checks") or []:
        metric = check.get("metric")
        if metric == "spatial_accessibility":
            results.append(_evaluate_spatial_accessibility(check, objects, rooms))
        elif metric == "functional_dependency":
            results.append(_evaluate_functional_dependency(check, objects))
    return results


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_metric: dict[str, dict[str, Any]] = {}
    scene = _new_bucket()
    for result in results:
        metric = str(result.get("metric") or "unknown")
        bucket = by_metric.setdefault(metric, _new_bucket())
        label = str(result.get("label") or "unknown")
        _accumulate(bucket, label)
        _accumulate(scene, label)
    return {
        "scene_summary": _finish_bucket(scene),
        "metric_summary": {
            metric: _finish_bucket(bucket)
            for metric, bucket in sorted(by_metric.items())
        },
    }


def _evaluate_spatial_accessibility(
    check: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    rooms: list[dict[str, Any]],
) -> dict[str, Any]:
    subject_id = str(check.get("subject_id") or "")
    subject = objects.get(subject_id)
    if subject is None:
        return _result(check, "unknown", "Subject object is missing.", confidence=0.0)

    room = _room_for_subject(subject, rooms)
    footprint = _bbox_xy(subject)
    if footprint is None:
        return _result(
            check, "unknown", "Subject has no world bounding box.", confidence=0.0
        )

    sx0, sy0, sx1, sy1 = footprint
    cx, cy = ((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0)
    affordance = str(check.get("affordance") or "")
    clearance = 0.55 if affordance in {"sittable", "openable"} else 0.45
    yaw = math.radians(float(subject.get("yaw_deg") or 0.0))
    front = (math.cos(yaw), math.sin(yaw))
    side = (-front[1], front[0])
    radius_x = max((sx1 - sx0) / 2.0, 0.2)
    radius_y = max((sy1 - sy0) / 2.0, 0.2)
    access_points = [
        (
            cx + front[0] * (radius_x + clearance),
            cy + front[1] * (radius_y + clearance),
        ),
        (cx + side[0] * (radius_x + clearance), cy + side[1] * (radius_y + clearance)),
        (cx - side[0] * (radius_x + clearance), cy - side[1] * (radius_y + clearance)),
    ]

    blockers: list[str] = []
    open_points = 0
    for point in access_points:
        if room and not _point_in_polygon(point, room.get("floor_polygon") or []):
            continue
        blocked_by = _blocking_object_at(point, subject_id, objects)
        if blocked_by is None:
            open_points += 1
        else:
            blockers.append(blocked_by)

    if open_points >= 2:
        return _result(
            check,
            "pass",
            "At least two approach zones are open.",
            blocking_objects=sorted(set(blockers)),
        )
    if open_points == 1:
        return _result(
            check,
            "degraded",
            "Only one approach zone is open.",
            blocking_objects=sorted(set(blockers)),
        )
    return _result(
        check,
        "fail",
        "No sampled approach zone is open.",
        blocking_objects=sorted(set(blockers)),
    )


def _evaluate_functional_dependency(
    check: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    subject_id = str(check.get("subject_id") or "")
    target_ids = [str(item) for item in check.get("target_ids") or []]
    subject = objects.get(subject_id)
    if subject is None:
        return _result(check, "unknown", "Subject object is missing.", confidence=0.0)
    targets = [objects[target_id] for target_id in target_ids if target_id in objects]
    if not targets:
        return _result(
            check, "unknown", "No target object is available.", confidence=0.0
        )

    best: tuple[str, float, str, dict[str, Any]] | None = None
    for target in targets:
        label, score, reason, evidence = _support_score(subject, target, check)
        if best is None or score > best[1]:
            best = (label, score, reason, evidence)

    assert best is not None
    label, _score, reason, evidence = best
    return _result(
        check,
        label,
        reason,
        confidence=0.85,
        selected_related_objects=target_ids[:1],
        evidence=evidence,
    )


def _support_score(
    subject: dict[str, Any], target: dict[str, Any], check: dict[str, Any]
) -> tuple[str, float, str, dict[str, Any]]:
    sb = _bbox(subject) or {}
    tb = _bbox(target) or {}
    smin = sb.get("min") or [0.0, 0.0, 0.0]
    smax = sb.get("max") or [0.0, 0.0, 0.0]
    sx0, sy0, sx1, sy1 = _bbox_xy(subject) or (0.0, 0.0, 0.0, 0.0)
    subject_area = max((sx1 - sx0) * (sy1 - sy0), 1e-6)

    region_id = str((check.get("evidence") or {}).get("parent_surface_id") or "")
    regions = target.get("support_regions") or []
    if region_id:
        regions = [
            r for r in regions if str(r.get("region_id")) == region_id
        ] or regions

    best_overlap = 0.0
    best_height_delta = float("inf")
    best_region: dict[str, Any] | None = None
    for region in regions:
        polygon = region.get("polygon_world_xy") or []
        rb = _polygon_bounds(polygon)
        if rb is None:
            continue
        overlap = _rect_overlap_area((sx0, sy0, sx1, sy1), rb) / subject_area
        height = region.get("height_world_z")
        if height is None:
            continue
        height_delta = abs(float(smin[2]) - float(height))
        if overlap > best_overlap or (
            math.isclose(overlap, best_overlap) and height_delta < best_height_delta
        ):
            best_overlap = overlap
            best_height_delta = height_delta
            best_region = region

    if best_region is None and tb:
        tx0, ty0, tx1, ty1 = _bbox_xy(target) or (0.0, 0.0, 0.0, 0.0)
        best_overlap = (
            _rect_overlap_area((sx0, sy0, sx1, sy1), (tx0, ty0, tx1, ty1))
            / subject_area
        )
        best_height_delta = abs(float(smin[2]) - float((tb.get("max") or [0, 0, 0])[2]))

    evidence = {
        "overlap_ratio": best_overlap,
        "height_delta_m": (
            None if best_height_delta == float("inf") else best_height_delta
        ),
        "support_surface_id": (best_region or {}).get("region_id"),
    }
    if best_overlap >= 0.55 and best_height_delta <= 0.18:
        return (
            "pass",
            1.0,
            "Subject footprint and height match the support evidence.",
            evidence,
        )
    if best_overlap >= 0.25 and best_height_delta <= 0.30:
        return (
            "degraded",
            0.5,
            "Subject has partial or height-marginal support evidence.",
            evidence,
        )
    return (
        "fail",
        0.0,
        "Subject is not sufficiently aligned with target support evidence.",
        evidence,
    )


def _result(
    check: dict[str, Any],
    label: str,
    reason: str,
    *,
    confidence: float = 0.8,
    blocking_objects: list[str] | None = None,
    selected_related_objects: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check.get("check_id"),
        "metric": check.get("metric"),
        "label": label,
        "reason": reason,
        "confidence": confidence,
        "blocking_objects": blocking_objects or [],
        "primary_object": check.get("subject_id"),
        "related_objects": check.get("target_ids") or [],
        "selected_related_objects": selected_related_objects or [],
        "evidence": evidence or {},
        "evaluation_source": f"rule_{check.get('metric')}",
    }


def _room_for_subject(
    subject: dict[str, Any], rooms: list[dict[str, Any]]
) -> dict[str, Any] | None:
    room_id = subject.get("room")
    for room in rooms:
        if room.get("id") == room_id:
            return room
    return rooms[0] if rooms else None


def _blocking_object_at(
    point: tuple[float, float], subject_id: str, objects: dict[str, dict[str, Any]]
) -> str | None:
    px, py = point
    for obj_id, obj in objects.items():
        if obj_id == subject_id or _is_ignored_blocker(obj):
            continue
        bounds = _bbox_xy(obj)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        if x0 <= px <= x1 and y0 <= py <= y1:
            return obj_id
    return None


def _is_ignored_blocker(obj: dict[str, Any]) -> bool:
    object_type = obj.get("object_type")
    if object_type in {
        "manipuland",
        "thin_covering",
        "wall_mounted",
        "ceiling_mounted",
    }:
        return True
    bbox = _bbox(obj)
    if not bbox:
        return True
    size = bbox.get("size") or [0.0, 0.0, 0.0]
    bmin = bbox.get("min") or [0.0, 0.0, 0.0]
    if float(size[2]) < 0.35:
        return True
    if float(bmin[2]) > 0.35:
        return True
    return False


def _bbox(obj: dict[str, Any]) -> dict[str, Any] | None:
    bbox = obj.get("bbox_world")
    return bbox if isinstance(bbox, dict) else None


def _bbox_xy(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = _bbox(obj)
    bmin = (bbox or {}).get("min")
    bmax = (bbox or {}).get("max")
    if not (
        isinstance(bmin, list)
        and isinstance(bmax, list)
        and len(bmin) >= 2
        and len(bmax) >= 2
    ):
        return None
    return float(bmin[0]), float(bmin[1]), float(bmax[0]), float(bmax[1])


def _polygon_bounds(
    polygon: list[list[float]],
) -> tuple[float, float, float, float] | None:
    points = [point for point in polygon if isinstance(point, list) and len(point) >= 2]
    if not points:
        return None
    return (
        min(float(point[0]) for point in points),
        min(float(point[1]) for point in points),
        max(float(point[0]) for point in points),
        max(float(point[1]) for point in points),
    )


def _rect_overlap_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return True
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        pj = polygon[j]
        xi, yi = float(pi[0]), float(pi[1])
        xj, yj = float(pj[0]), float(pj[1])
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _new_bucket() -> dict[str, Any]:
    return {
        "total_checks": 0,
        "pass": 0,
        "degraded": 0,
        "fail": 0,
        "unknown": 0,
        "score_sum": 0.0,
    }


def _accumulate(bucket: dict[str, Any], label: str) -> None:
    bucket["total_checks"] += 1
    if label not in {"pass", "degraded", "fail", "unknown"}:
        label = "unknown"
    bucket[label] += 1
    score = LABEL_TO_SCORE.get(label)
    if score is not None:
        bucket["score_sum"] += score


def _finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    effective = bucket["pass"] + bucket["degraded"] + bucket["fail"]
    return {
        **bucket,
        "effective_checks": effective,
        "effective_pass_rate": bucket["pass"] / effective if effective else None,
        "score": bucket["score_sum"] / effective if effective else None,
    }
