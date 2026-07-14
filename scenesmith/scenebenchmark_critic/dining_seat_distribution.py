"""General dining-chair distribution checks for rectangular tables."""

from __future__ import annotations

import math
from typing import Any

from scenesmith.scenebenchmark_critic.dining_place_setting_alignment import (
    _associated_discrete_seats,
)
from scenesmith.scenebenchmark_critic.manipuland_completeness import (
    _is_dining_table,
    _object_identity_text,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.geometry import (
    bbox_center_xy,
)

RELATION_TYPE = "dining_seat_distribution"


def evaluate_dining_seat_distribution(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that chairs are centered or evenly spaced along each table edge."""
    objects = [
        obj
        for obj in ((case_pack.get("scene_geometry") or {}).get("objects") or [])
        if isinstance(obj, dict) and obj.get("id")
    ]
    objects_by_id = {str(obj["id"]): obj for obj in objects}
    results: list[dict[str, Any]] = []
    for table in objects:
        if not _is_dining_table(table) or _is_round_table(table):
            continue
        result = _evaluate_table(table, _associated_discrete_seats(table, objects_by_id))
        if result is not None:
            results.append(result)
    return results


def _evaluate_table(
    table: dict[str, Any], seats: list[dict[str, Any]]
) -> dict[str, Any] | None:
    center = bbox_center_xy(table)
    size = (table.get("bbox_world") or {}).get("size") or []
    seats = [seat for seat in seats if "bench" not in _object_identity_text(seat)]
    if center is None or len(size) < 2 or not seats:
        return None
    width, depth = float(size[0]), float(size[1])
    if min(width, depth) <= 1e-6:
        return None
    yaw = math.radians(float(table.get("yaw_deg") or 0.0))
    tangent_x = (math.cos(yaw), math.sin(yaw))
    tangent_y = (-math.sin(yaw), math.cos(yaw))
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {
        "left": [], "right": [], "front": [], "back": []
    }
    for seat in seats:
        seat_center = bbox_center_xy(seat)
        if seat_center is None:
            continue
        dx, dy = seat_center[0] - center[0], seat_center[1] - center[1]
        local_x = dx * tangent_x[0] + dy * tangent_x[1]
        local_y = dx * tangent_y[0] + dy * tangent_y[1]
        edge = min(
            (
                (abs(local_x + width / 2), "left", local_y),
                (abs(local_x - width / 2), "right", local_y),
                (abs(local_y + depth / 2), "front", local_x),
                (abs(local_y - depth / 2), "back", local_x),
            ),
            key=lambda row: (row[0], row[1]),
        )
        grouped[edge[1]].append((seat, edge[2]))

    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    for edge, members in grouped.items():
        if not members:
            continue
        edge_length = depth if edge in {"left", "right"} else width
        chair_spans = [_seat_tangent_span(seat, edge, yaw) for seat, _ in members]
        perpendicular_length = width if edge in {"left", "right"} else depth
        # 2026-07-13 修改原因：只扣半个椅宽会把长边端部槽位推到桌角，导致
        # 座椅同时侵占相邻短边。至少扣除半个垂直边长，使槽位明确属于当前边。
        margin = max(max(chair_spans) / 2, perpendicular_length / 2)
        usable_span = max(0.0, edge_length - 2 * margin)
        count = len(members)
        # 2026-07-13 修改原因：槽位由桌边长度和该边实际座椅数推导；单椅取
        # 中点，多椅在扣除椅宽边距后等距分布，避免固定四人桌坐标。
        slots = (
            [0.0]
            if count == 1
            else [-usable_span / 2 + i * usable_span / (count - 1) for i in range(count)]
        )
        actual = sorted(members, key=lambda row: (row[1], str(row[0]["id"])))
        for (seat, position), slot, chair_span in zip(actual, slots, sorted(chair_spans)):
            deviation = abs(position - slot)
            allowed = max(0.08, min(0.35 * chair_span, 0.08 * edge_length))
            passed = deviation <= allowed
            # 2026-07-14 修改原因：同一长边多椅的历史分布检查只负责槽位，
            # 不应因旧测试/布局没有逐椅朝向数据而改变其语义；单椅边位才要求
            # 严格正对桌心，正好覆盖四边各一把 dining chair 的场景。
            facing_error = _seat_facing_error_deg(seat, center) if count == 1 else None
            facing_passed = facing_error is None or facing_error <= 10.0
            diagnostics.append({
                "seat_id": str(seat["id"]), "edge": edge,
                "tangent_position_m": round(position, 4),
                "target_position_m": round(slot, 4),
                "deviation_m": round(deviation, 4),
                "allowed_deviation_m": round(allowed, 4), "aligned": passed,
                "facing_error_deg": round(facing_error, 2) if facing_error is not None else None,
                "facing_allowed_error_deg": 10.0,
                "facing_aligned": facing_passed,
            })
            if not passed:
                direction = "positive" if slot > position else "negative"
                failures.append(
                    f"`{seat['id']}` on the {edge} edge is {deviation:.2f}m from "
                    f"its evenly distributed slot; move it in the {direction} edge direction"
                )
            if not facing_passed:
                failures.append(
                    f"`{seat['id']}` on the {edge} edge is rotated {facing_error:.1f}° "
                    "away from the table center; align its front normal to the table"
                )
    if not diagnostics:
        return None
    table_id = str(table["id"])
    related = sorted(str(seat["id"]) for seat in seats)
    failed = bool(failures)
    reason = (
        "Dining chairs on each rectangular table edge must be centered when alone "
        "and evenly distributed when multiple chairs share the edge. "
        "For a dining chair, use an exact table-local slot and do not use generic "
        "center snapping or shift the chair along the edge normal to resolve a "
        "door conflict; move the table or door-compatible layout instead. "
        + "; ".join(failures)
        if failed else
        "Dining chairs are centered or evenly distributed along their respective table edges."
    )
    return {
        "check_id": f"fd_{table_id}_{RELATION_TYPE}",
        "metric": "functional_dependency", "label": "fail" if failed else "pass",
        "confidence": 0.93 if failed else 0.89, "primary_object": table_id,
        "related_objects": related, "selected_related_objects": related,
        "blocking_objects": [], "relation_type": RELATION_TYPE, "reason": reason,
        "diagnostics": {"seat_slots": diagnostics},
        "evidence": {"distribution": "table_local_edge_slots"},
        "evaluation_source": "scenesmith_dining_seat_distribution", "scoring_tier": "core",
    }


def _seat_tangent_span(seat: dict[str, Any], edge: str, table_yaw: float) -> float:
    size = (seat.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return 0.45
    axis = (math.cos(table_yaw), math.sin(table_yaw)) if edge in {"front", "back"} else (-math.sin(table_yaw), math.cos(table_yaw))
    return max(0.2, abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1]))


def _seat_facing_error_deg(
    seat: dict[str, Any], table_center: tuple[float, float] | None
) -> float | None:
    """Return angular error between chair front (+local Y) and table center."""
    # 2026-07-14 修改原因：check_facing_tool 的宽松通过阈值会把约 13° 的
    # dining_chair_2 偏角判为正确；餐桌座位检查需要更严格的 10° 误差。
    if table_center is None:
        return None
    center = bbox_center_xy(seat)
    if center is None or "yaw_deg" not in seat:
        return None
    dx = float(table_center[0]) - float(center[0])
    dy = float(table_center[1]) - float(center[1])
    if abs(dx) + abs(dy) <= 1e-6:
        return None
    desired = math.degrees(math.atan2(-dx, dy))
    actual = float(seat.get("yaw_deg") or 0.0)
    return abs((actual - desired + 180.0) % 360.0 - 180.0)


def _is_round_table(table: dict[str, Any]) -> bool:
    text = _object_identity_text(table)
    return any(token in text for token in ("round", "circular", "oval", "ellipse"))
