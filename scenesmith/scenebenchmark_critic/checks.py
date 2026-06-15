"""Check construction for the embedded rule critic."""

from __future__ import annotations

from typing import Any

ACCESS_AFFORDANCES = {"sittable", "openable", "supportable", "sleepable"}


def build_checks(
    case_pack: dict[str, Any], metrics: tuple[str, ...] | list[str] | None = None
) -> list[dict[str, Any]]:
    enabled = set(metrics or ("spatial_accessibility", "functional_dependency"))
    geometry = case_pack.get("scene_geometry") or {}
    objects = {
        str(obj.get("id")): obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    }
    checks: list[dict[str, Any]] = []

    if "spatial_accessibility" in enabled:
        for obj in objects.values():
            affordances = set(
                ((obj.get("functional_hints") or {}).get("functional_categories") or [])
            )
            for affordance in sorted(affordances & ACCESS_AFFORDANCES):
                checks.append(
                    {
                        "check_id": f"sa_{obj['id']}_{affordance}",
                        "metric": "spatial_accessibility",
                        "subject_id": obj["id"],
                        "affordance": affordance,
                        "question": (
                            f"Is {obj.get('name') or obj['id']} spatially accessible "
                            f"for {affordance} use?"
                        ),
                        "scoring_tier": "core",
                    }
                )

    if "functional_dependency" in enabled:
        surface_owner = _surface_owner_map(objects)
        for obj in objects.values():
            placement = obj.get("placement_info") or {}
            surface_id = str(placement.get("parent_surface_id") or "")
            target_id = surface_owner.get(surface_id)
            if not surface_id or not target_id:
                continue
            relation_type = _relation_type_for(obj, objects[target_id])
            checks.append(
                {
                    "check_id": f"fd_{obj['id']}_{target_id}_{relation_type}",
                    "metric": "functional_dependency",
                    "subject_id": obj["id"],
                    "target_ids": [target_id],
                    "relation_type": relation_type,
                    "question": (
                        f"Is {obj.get('name') or obj['id']} functionally supported "
                        f"by {objects[target_id].get('name') or target_id}?"
                    ),
                    "evidence": {"parent_surface_id": surface_id},
                    "scoring_tier": "core",
                }
            )
    return checks


def _surface_owner_map(objects: dict[str, dict[str, Any]]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for obj_id, obj in objects.items():
        for region in obj.get("support_regions") or []:
            region_id = region.get("region_id")
            if region_id:
                owners[str(region_id)] = obj_id
    return owners


def _relation_type_for(subject: dict[str, Any], target: dict[str, Any]) -> str:
    category = str(subject.get("category_norm") or subject.get("category") or "")
    target_category = str(target.get("category_norm") or target.get("category") or "")
    if "lamp" in category or "light" in category:
        return "lamp_to_surface"
    if "rug" in category or "mat" in category:
        return "floor_covering_on_floor"
    if "floor" in target_category:
        return "object_on_floor"
    return "object_on_support"
