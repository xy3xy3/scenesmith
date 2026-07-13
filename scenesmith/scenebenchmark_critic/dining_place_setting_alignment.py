"""Dining place-setting alignment checks for the embedded critic."""

from __future__ import annotations

import functools
import math
import re

from typing import Any

from scenesmith.scenebenchmark_critic.manipuland_completeness import (
    CUTLERY_GROUPS,
    _bbox_gap_xy,
    _footprint_short_side,
    _is_dining_seat,
    _is_dining_table,
    _matches_item_group,
    _object_identity_text,
    _object_text,
    _placement_surface_id,
    _required_groups,
    _scene_object_type,
    _surface_owner_map,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.geometry import (
    bbox_center_xy,
    front_vector,
)

RELATION_TYPE = "dining_place_setting_alignment"


def evaluate_dining_place_setting_alignment(
    case_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that each place setting lies in its assigned dining seat's front lane."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    if not objects or not _prompt_requests_place_settings(case_pack):
        return []

    objects_by_id = {str(obj["id"]): obj for obj in objects}
    surface_owner = _surface_owner_map(objects)
    results: list[dict[str, Any]] = []
    for table in objects:
        if not _is_dining_table(table):
            continue
        table_id = str(table["id"])
        surface_ids = {
            surface_id
            for surface_id, owner_id in surface_owner.items()
            if owner_id == table_id
        }
        if not surface_ids:
            continue
        surface_items = [
            obj
            for obj in objects
            if _scene_object_type(obj) == "manipuland"
            and _placement_surface_id(obj) in surface_ids
        ]
        anchors = [obj for obj in surface_items if _is_place_anchor(obj)]
        seats = _associated_discrete_seats(table, objects_by_id)
        # 2026-07-13 修改原因：只有离散座位与餐位锚点可一对一对应时，
        # “椅子正前方”才有唯一、可执行的几何含义。长凳或数量不一致场景交给
        # completeness/视觉 critic，避免强行套用四人餐桌拓扑。
        if len(anchors) < 2 or len(anchors) != len(seats):
            continue
        result = _evaluate_table_alignment(
            table=table,
            seats=seats,
            anchors=anchors,
            companions=[obj for obj in surface_items if _is_place_companion(obj)],
        )
        if result is not None:
            results.append(result)
    return results


def _evaluate_table_alignment(
    *,
    table: dict[str, Any],
    seats: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    companions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    assignment = _minimum_distance_assignment(seats, anchors)
    if assignment is None:
        return None
    table_center = bbox_center_xy(table)
    table_short_side = _short_side(table)
    if table_center is None or table_short_side is None:
        return None

    anchor_to_seat: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    failures: list[str] = []
    for seat, anchor in assignment:
        seat_center = bbox_center_xy(seat)
        anchor_center = bbox_center_xy(anchor)
        if seat_center is None or anchor_center is None:
            return None
        forward = _usable_seat_front(seat, seat_center, table_center)
        lateral_axis = (-forward[1], forward[0])
        lateral_offset = abs(
            (anchor_center[0] - seat_center[0]) * lateral_axis[0]
            + (anchor_center[1] - seat_center[1]) * lateral_axis[1]
        )
        longitudinal = (
            (anchor_center[0] - seat_center[0]) * forward[0]
            + (anchor_center[1] - seat_center[1]) * forward[1]
        )
        allowed = _front_lane_half_width(seat, anchor, lateral_axis)
        seat_id = str(seat["id"])
        anchor_id = str(anchor["id"])
        anchor_to_seat[anchor_id] = seat
        aligned = longitudinal > 0.0 and lateral_offset <= allowed
        diagnostics.append(
            {
                "seat_id": seat_id,
                "anchor_id": anchor_id,
                "lateral_offset_m": round(lateral_offset, 4),
                "allowed_lateral_offset_m": round(allowed, 4),
                "longitudinal_offset_m": round(longitudinal, 4),
                "aligned": aligned,
                "companion_ids": [],
                "misaligned_companion_ids": [],
            }
        )
        if not aligned:
            failures.append(
                f"`{anchor_id}` is not centered in front of `{seat_id}` "
                f"(lateral {lateral_offset:.2f}m > {allowed:.2f}m)"
            )

    rows_by_anchor = {row["anchor_id"]: row for row in diagnostics}
    for companion in companions:
        anchor = _nearest_cluster_anchor(companion, anchors, table_short_side)
        if anchor is None:
            continue
        anchor_id = str(anchor["id"])
        seat = anchor_to_seat.get(anchor_id)
        if seat is None:
            continue
        seat_center = bbox_center_xy(seat)
        companion_center = bbox_center_xy(companion)
        if seat_center is None or companion_center is None:
            continue
        forward = _usable_seat_front(seat, seat_center, table_center)
        lateral_axis = (-forward[1], forward[0])
        lateral_offset = abs(
            (companion_center[0] - seat_center[0]) * lateral_axis[0]
            + (companion_center[1] - seat_center[1]) * lateral_axis[1]
        )
        allowed = _front_lane_half_width(seat, companion, lateral_axis)
        companion_id = str(companion["id"])
        row = rows_by_anchor[anchor_id]
        row["companion_ids"].append(companion_id)
        if lateral_offset > allowed:
            row["misaligned_companion_ids"].append(companion_id)
            failures.append(
                f"`{companion_id}` belonging to `{anchor_id}` is outside "
                f"`{seat['id']}`'s front lane"
            )

    table_id = str(table["id"])
    related_ids = sorted(
        {
            str(obj["id"])
            for obj in [*seats, *anchors, *companions]
            if obj.get("id")
        }
    )
    if failures:
        reason = (
            "Dining place settings must be centered on the front axis of their "
            "one-to-one nearest seats. "
            + "; ".join(failures[:8])
            + ". Move each plate/bowl together with its nearby cutlery, drinkware, "
            "and napkin toward that seat's centerline; do not move it to another "
            "table edge."
        )
        label = "fail"
        confidence = 0.94
    else:
        reason = (
            f"All {len(assignment)} dining place setting(s) are centered in the "
            "front lanes of distinct nearby seats, with companions grouped to the "
            "same seats."
        )
        label = "pass"
        confidence = 0.9
    return {
        "check_id": f"fd_{table_id}_{RELATION_TYPE}",
        "metric": "functional_dependency",
        "label": label,
        "confidence": confidence,
        "primary_object": table_id,
        "related_objects": related_ids,
        "selected_related_objects": related_ids,
        "blocking_objects": [],
        "relation_type": RELATION_TYPE,
        "reason": reason,
        "diagnostics": {"assignments": diagnostics},
        "evidence": {
            "association": "minimum_distance_one_to_one",
            "alignment": "seat_front_lateral_projection",
        },
        "evaluation_source": "scenesmith_dining_place_setting_alignment",
        "scoring_tier": "core",
    }


def _minimum_distance_assignment(
    seats: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    seat_centers = [bbox_center_xy(seat) for seat in seats]
    anchor_centers = [bbox_center_xy(anchor) for anchor in anchors]
    if any(center is None for center in [*seat_centers, *anchor_centers]):
        return None
    count = len(seats)
    costs = [
        [
            math.hypot(
                float(seat_centers[i][0]) - float(anchor_centers[j][0]),
                float(seat_centers[i][1]) - float(anchor_centers[j][1]),
            )
            for j in range(count)
        ]
        for i in range(count)
    ]

    # 2026-07-13 修改原因：逐椅贪心会让相邻座位争用同一餐盘。位掩码动态规划
    # 求全局最短一对一分配，适配长桌、圆桌和非对称座椅布局。
    @functools.lru_cache(maxsize=None)
    def solve(seat_index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if seat_index == count:
            return 0.0, ()
        best = (math.inf, ())
        for anchor_index in range(count):
            if used_mask & (1 << anchor_index):
                continue
            remaining_cost, remaining = solve(
                seat_index + 1, used_mask | (1 << anchor_index)
            )
            candidate = (
                costs[seat_index][anchor_index] + remaining_cost,
                (anchor_index, *remaining),
            )
            if candidate < best:
                best = candidate
        return best

    # Avoid exponential work for unusually large banquet layouts; deterministic
    # nearest-pair assignment remains a safe report-only fallback above 12 seats.
    if count <= 12:
        _cost, indices = solve(0, 0)
    else:
        available = set(range(count))
        picked: list[int] = []
        for seat_index in range(count):
            anchor_index = min(
                available, key=lambda index: (costs[seat_index][index], index)
            )
            available.remove(anchor_index)
            picked.append(anchor_index)
        indices = tuple(picked)
    return [(seats[index], anchors[indices[index]]) for index in range(count)]


def _prompt_requests_place_settings(case_pack: dict[str, Any]) -> bool:
    task = str(case_pack.get("task_instruction") or "")
    required = _required_groups(task)
    return bool(required & {"plate", "bowl"}) or bool(
        re.search(r"\b(?:table|place)\s*settings?\b", task.lower())
    )


def _associated_discrete_seats(
    table: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    table_center = bbox_center_xy(table)
    table_scale = _footprint_short_side(table)
    if table_center is None or table_scale is None:
        return []
    seats: list[dict[str, Any]] = []
    for seat in objects_by_id.values():
        if not _is_dining_seat(seat) or "bench" in _object_identity_text(seat):
            continue
        seat_center = bbox_center_xy(seat)
        seat_scale = _footprint_short_side(seat)
        gap = _bbox_gap_xy(table, seat)
        if seat_center is None or seat_scale is None or gap is None:
            continue
        fx, fy = front_vector(seat)
        tx, ty = table_center[0] - seat_center[0], table_center[1] - seat_center[1]
        target_distance = math.hypot(tx, ty)
        if target_distance <= 1e-9:
            continue
        front_alignment = (fx * tx + fy * ty) / target_distance
        # 2026-07-13 修改原因：餐位归属不仅看中心距离，还要求座椅朝向该桌；
        # 允许一个座椅短边的桌椅间隙，以覆盖正常拉椅空间但排除相邻桌组。
        association_gap = max(seat_scale, 0.25 * table_scale)
        if gap <= association_gap and front_alignment >= 0.5:
            seats.append(seat)
    return sorted(seats, key=lambda item: str(item.get("id") or ""))


def _is_place_anchor(obj: dict[str, Any]) -> bool:
    text = _object_text(obj)
    return _matches_item_group("plate", text) or _matches_item_group("bowl", text)


def _is_place_companion(obj: dict[str, Any]) -> bool:
    text = _object_text(obj)
    return _matches_item_group("drinkware", text) or _matches_item_group(
        "napkin", text
    ) or any(_matches_item_group(group, text) for group in CUTLERY_GROUPS)


def _usable_seat_front(
    seat: dict[str, Any],
    seat_center: tuple[float, float],
    table_center: tuple[float, float],
) -> tuple[float, float]:
    fx, fy = front_vector(seat)
    tx, ty = table_center[0] - seat_center[0], table_center[1] - seat_center[1]
    target_norm = math.hypot(tx, ty)
    if target_norm <= 1e-9:
        return fx, fy
    # If front metadata is missing or contradicts the nearby table, the dining-set
    # furniture relation owns that orientation failure. Use the geometric seat-table
    # axis here so manipuland feedback still identifies the correct table edge.
    if fx * tx + fy * ty <= 0.2 * target_norm:
        return tx / target_norm, ty / target_norm
    return fx, fy


def _front_lane_half_width(
    seat: dict[str, Any], item: dict[str, Any], lateral_axis: tuple[float, float]
) -> float:
    seat_span = _projected_span(seat, lateral_axis)
    item_span = _projected_span(item, lateral_axis)
    if seat_span is None:
        seat_span = _short_side(seat) or 0.45
    if item_span is None:
        item_span = _short_side(item) or 0.1
    return 0.5 * seat_span + 0.1 * item_span


def _nearest_cluster_anchor(
    companion: dict[str, Any],
    anchors: list[dict[str, Any]],
    table_short_side: float,
) -> dict[str, Any] | None:
    center = bbox_center_xy(companion)
    if center is None:
        return None
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for anchor in anchors:
        anchor_center = bbox_center_xy(anchor)
        if anchor_center is None:
            continue
        distance = math.hypot(
            center[0] - anchor_center[0], center[1] - anchor_center[1]
        )
        ranked.append((distance, str(anchor.get("id") or ""), anchor))
    if not ranked:
        return None
    distance, _anchor_id, anchor = min(ranked)
    anchor_scale = _short_side(anchor) or 0.2
    cluster_radius = max(1.5 * anchor_scale, 0.18 * table_short_side)
    return anchor if distance <= cluster_radius else None


def _projected_span(
    obj: dict[str, Any], axis: tuple[float, float]
) -> float | None:
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    return abs(axis[0]) * float(size[0]) + abs(axis[1]) * float(size[1])


def _short_side(obj: dict[str, Any]) -> float | None:
    size = (obj.get("bbox_world") or {}).get("size") or []
    if len(size) < 2:
        return None
    positive = [float(value) for value in size[:2] if float(value) > 1e-6]
    return min(positive) if positive else None
