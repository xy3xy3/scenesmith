"""Rule checks for manipuland set completeness."""

from __future__ import annotations

from collections import Counter
from typing import Any


DINING_ITEM_GROUPS = {
    "plate": ("plate", "dinner_plate"),
    "fork": ("fork",),
    "knife": ("knife",),
    "spoon": ("spoon",),
    "napkin": ("napkin",),
}


def evaluate_manipuland_completeness(case_pack: dict[str, Any]) -> list[dict[str, Any]]:
    """Return extra rule results for missing manipulands in established sets."""
    geometry = case_pack.get("scene_geometry") or {}
    objects = [
        obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    ]
    if not objects:
        return []

    surface_owner = _surface_owner_map(objects)
    objects_by_id = {str(obj["id"]): obj for obj in objects}
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
        result = _evaluate_dining_table_setting(
            table=table,
            surface_items=surface_items,
            objects_by_id=objects_by_id,
        )
        if result is not None:
            results.append(result)
    return results


def _evaluate_dining_table_setting(
    *,
    table: dict[str, Any],
    surface_items: list[dict[str, Any]],
    objects_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    counts = Counter()
    item_ids_by_group: dict[str, list[str]] = {key: [] for key in DINING_ITEM_GROUPS}
    for item in surface_items:
        text = _object_text(item)
        for group, tokens in DINING_ITEM_GROUPS.items():
            if any(token in text for token in tokens):
                counts[group] += 1
                item_ids_by_group[group].append(str(item["id"]))
                break

    place_count = max(counts["plate"], _nearby_dining_seat_count(table, objects_by_id))
    # Only enforce this when the scene has clearly started a table setting.
    if place_count < 2 or counts["plate"] < 2:
        return None

    missing: dict[str, int] = {}
    for group in ("fork", "knife", "spoon", "napkin"):
        deficit = place_count - counts[group]
        if deficit > 0:
            missing[group] = deficit

    related = sorted(
        {
            item_id
            for group in ("plate", "fork", "knife", "spoon", "napkin")
            for item_id in item_ids_by_group[group]
        }
    )
    table_id = str(table["id"])
    if missing:
        missing_text = ", ".join(
            f"{group} x{count}" for group, count in sorted(missing.items())
        )
        counts_text = ", ".join(
            f"{group}={counts[group]}"
            for group in ("plate", "fork", "knife", "spoon", "napkin")
        )
        # 2026-07-09 修改原因：critic 通过后物理后处理可能删除餐具/餐巾；
        # 最终规则报告必须显式暴露成组 tabletop manipuland 缺失。
        return {
            "check_id": f"manipuland_completeness__{table_id}__dining_place_setting",
            "metric": "manipuland_completeness",
            "label": "fail",
            "confidence": 0.9,
            "primary_object": table_id,
            "related_objects": related,
            "selected_related_objects": related,
            "blocking_objects": [],
            "reason": (
                f"Dining table has {place_count} place setting(s) implied by "
                f"plates/seats, but required tabletop items are missing: "
                f"{missing_text}. Counts: {counts_text}."
            ),
            "diagnostics": {
                "place_count": place_count,
                "counts": {group: counts[group] for group in DINING_ITEM_GROUPS},
                "missing": missing,
            },
            "evidence": {"surface_item_ids": sorted(str(item["id"]) for item in surface_items)},
            "scoring_tier": "core",
        }

    return {
        "check_id": f"manipuland_completeness__{table_id}__dining_place_setting",
        "metric": "manipuland_completeness",
        "label": "pass",
        "confidence": 0.85,
        "primary_object": table_id,
        "related_objects": related,
        "selected_related_objects": related,
        "blocking_objects": [],
        "reason": (
            f"Dining table place setting is complete for {place_count} "
            "implied place setting(s)."
        ),
        "diagnostics": {
            "place_count": place_count,
            "counts": {group: counts[group] for group in DINING_ITEM_GROUPS},
            "missing": {},
        },
        "evidence": {"surface_item_ids": sorted(str(item["id"]) for item in surface_items)},
        "scoring_tier": "core",
    }


def _surface_owner_map(objects: list[dict[str, Any]]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for obj in objects:
        obj_id = str(obj.get("id") or "")
        for surface in obj.get("support_surfaces") or []:
            surface_id = str(surface.get("surface_id") or surface.get("id") or "")
            if surface_id:
                owners[surface_id] = obj_id
        for region in obj.get("support_regions") or []:
            region_id = str(region.get("region_id") or region.get("surface_id") or "")
            if region_id:
                owners[region_id] = obj_id
    return owners


def _nearby_dining_seat_count(
    table: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]
) -> int:
    table_center = _bbox_center_xy(table)
    if table_center is None:
        return 0
    count = 0
    for obj in objects_by_id.values():
        if not _is_dining_seat(obj):
            continue
        center = _bbox_center_xy(obj)
        if center is None:
            continue
        dx = center[0] - table_center[0]
        dy = center[1] - table_center[1]
        if (dx * dx + dy * dy) ** 0.5 <= 1.8:
            count += 1
    return count


def _bbox_center_xy(obj: dict[str, Any]) -> tuple[float, float] | None:
    center = ((obj.get("bbox_world") or {}).get("center") or [])[:2]
    if len(center) != 2:
        return None
    return float(center[0]), float(center[1])


def _placement_surface_id(obj: dict[str, Any]) -> str:
    placement = obj.get("placement_info") or {}
    return str(placement.get("parent_surface_id") or "")


def _scene_object_type(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    return str(hints.get("scene_object_type") or obj.get("object_type") or "").lower()


def _is_dining_table(obj: dict[str, Any]) -> bool:
    text = _object_text(obj)
    return _scene_object_type(obj) == "furniture" and "dining" in text and "table" in text


def _is_dining_seat(obj: dict[str, Any]) -> bool:
    text = _object_text(obj)
    return _scene_object_type(obj) == "furniture" and (
        "dining_chair" in text or ("dining" in text and "chair" in text)
    )


def _object_text(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    parts = [
        obj.get("id"),
        obj.get("name"),
        obj.get("description"),
        obj.get("category"),
        obj.get("category_norm"),
        hints.get("category_group"),
        " ".join(str(item) for item in hints.get("functional_categories") or []),
    ]
    return " ".join(str(part).lower() for part in parts if part)
