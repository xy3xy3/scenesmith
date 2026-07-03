"""Agent-aware prompt context for SceneBenchmark critic results."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from scenesmith.agent_utils.room import AgentType, RoomScene
from scenesmith.scenebenchmark_critic.reports import (
    format_prompt_context as format_full_prompt_context,
)

ISSUE_LABELS = {"fail", "degraded", "unknown"}
LABEL_RANK = {"fail": 3, "degraded": 2, "unknown": 1}
ARCHITECTURE_CATEGORIES = {"wall", "floor", "ceiling"}
COMPUTER_PERIPHERAL_CATEGORIES = {
    "keyboard",
    "mouse",
    "trackpad",
    "touchpad",
}
COMPUTER_SCREEN_CATEGORIES = {
    "display",
    "laptop",
    "monitor",
    "projection_screen",
    "screen",
    "tablet",
    "tablet_computer",
}
FURNITURE_RELATIONS = {
    "back_against_wall",
    "furniture_faces_furniture",
    "seat_faces_surface",
    "seating_to_media",
    "seating_to_work_surface",
    "side_or_back_against_wall",
}
MANIPULAND_RELATIONS = {
    "computer_peripheral_faces_screen",
    "display_faces_user",
    "object_on_support",
    "seating_to_media",
    "seating_to_work_surface",
}
WORKSTATION_CATEGORIES = (
    COMPUTER_PERIPHERAL_CATEGORIES
    | COMPUTER_SCREEN_CATEGORIES
    | {"computer", "notebook_computer"}
)


def format_agent_prompt_context(
    payload: dict[str, Any],
    *,
    scene: RoomScene | None = None,
    agent_type: AgentType | str,
    current_furniture_id: str | None = None,
    max_issues: int = 8,
    debug_output_dir: Path | None = None,
) -> str:
    """Format SceneBenchmark issues for the agent that can act on them."""
    filtered = filter_prompt_results_for_agent(
        payload,
        scene=scene,
        agent_type=agent_type,
        current_furniture_id=current_furniture_id,
    )
    if debug_output_dir is not None:
        _write_debug_context(debug_output_dir, payload, filtered, agent_type)
    if not filtered:
        counted = [
            result
            for result in payload.get("results") or []
            if not _is_ignored_scoring_tier(result)
        ]
        return (
            "SceneBenchmark geometry critic: no degraded or failed checks relevant "
            f"to the current {_agent_value(agent_type)} agent in "
            f"{len(counted)} counted rule checks."
        )
    return format_full_prompt_context({"results": filtered}, max_issues=max_issues)


def filter_prompt_results_for_agent(
    payload: dict[str, Any],
    *,
    scene: RoomScene | None = None,
    agent_type: AgentType | str,
    current_furniture_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return issues that are actionable for the current SceneSmith agent."""
    objects = _objects_by_id(payload)
    agent = _agent_value(agent_type)
    current_furniture_id = str(current_furniture_id or "").strip() or None
    scope = _scope_for_agent(objects, agent, current_furniture_id)

    selected: list[dict[str, Any]] = []
    for result in payload.get("results") or []:
        if not _is_prompt_issue(result):
            continue
        if _is_self_relation(result):
            continue
        if agent == AgentType.FURNITURE.value:
            if not _furniture_issue_is_relevant(result, objects, scope):
                continue
        elif agent == AgentType.MANIPULAND.value:
            if not _manipuland_issue_is_relevant(result, objects, scope):
                continue
        else:
            continue
        selected.append(result)

    return _dedupe_and_sort(selected)


def _scope_for_agent(
    objects: dict[str, dict[str, Any]],
    agent: str,
    current_furniture_id: str | None,
) -> dict[str, set[str]]:
    if agent == AgentType.FURNITURE.value:
        furniture_ids = {
            object_id
            for object_id, obj in objects.items()
            if _scene_object_type(obj) == "furniture"
        }
        return {
            "object_ids": furniture_ids,
            "support_object_ids": set(),
            "workstation_ids": set(),
        }

    support_ids = set()
    manipuland_ids = set()
    workstation_ids = set()
    if current_furniture_id and current_furniture_id in objects:
        support_ids.add(current_furniture_id)
        owned_surfaces = _surface_ids(objects[current_furniture_id])
        for object_id, obj in objects.items():
            if _parent_surface_id(obj) in owned_surfaces:
                manipuland_ids.add(object_id)
                if _is_workstation_object(obj):
                    workstation_ids.add(object_id)
    else:
        for object_id, obj in objects.items():
            if _scene_object_type(obj) == "manipuland":
                manipuland_ids.add(object_id)
                if _is_workstation_object(obj):
                    workstation_ids.add(object_id)
            elif _scene_object_type(obj) == "furniture":
                support_ids.add(object_id)

    if workstation_ids:
        for object_id, obj in objects.items():
            if _is_seating(obj) or _is_work_surface(obj):
                support_ids.add(object_id)
    return {
        "object_ids": manipuland_ids | support_ids | workstation_ids,
        "support_object_ids": support_ids,
        "workstation_ids": workstation_ids,
    }


def _furniture_issue_is_relevant(
    result: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    scope: dict[str, set[str]],
) -> bool:
    relation_type = str(result.get("relation_type") or "")
    subject_id = str(result.get("primary_object") or "")
    related_ids = _related_ids(result)
    if result.get("metric") == "spatial_accessibility":
        return subject_id in scope["object_ids"]
    if relation_type and relation_type not in FURNITURE_RELATIONS:
        return False
    if relation_type == "seating_to_media" and not _is_seating(objects.get(subject_id)):
        return False
    if relation_type == "seating_to_work_surface" and not _is_seating(
        objects.get(subject_id)
    ):
        return False
    involved = {subject_id, *related_ids}
    if not involved & scope["object_ids"]:
        return False
    return all(
        _scene_object_type(objects.get(object_id)) != "manipuland"
        for object_id in involved
        if object_id in objects
    )


def _manipuland_issue_is_relevant(
    result: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    scope: dict[str, set[str]],
) -> bool:
    relation_type = str(result.get("relation_type") or "")
    subject_id = str(result.get("primary_object") or "")
    related_ids = _related_ids(result)
    involved = {subject_id, *related_ids}

    if result.get("metric") == "spatial_accessibility":
        return subject_id in scope["object_ids"] and _is_workstation_object(
            objects.get(subject_id)
        )
    if relation_type not in MANIPULAND_RELATIONS:
        return False
    if not involved & scope["object_ids"]:
        return False
    if relation_type in {"seating_to_media", "seating_to_work_surface"}:
        return _is_seating(objects.get(subject_id)) and bool(
            set(related_ids) & (scope["workstation_ids"] | scope["support_object_ids"])
        )
    if relation_type == "object_on_support":
        return subject_id in scope["object_ids"] and bool(
            set(related_ids) & scope["support_object_ids"]
        )
    if relation_type == "computer_peripheral_faces_screen":
        return _is_computer_peripheral(objects.get(subject_id)) and any(
            _is_computer_screen(objects.get(target_id)) for target_id in related_ids
        )
    if relation_type == "display_faces_user":
        return _is_computer_screen(objects.get(subject_id)) and bool(
            set(related_ids) & scope["object_ids"]
        )
    return False


def _dedupe_and_sort(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    for result in results:
        key = (
            str(result.get("metric") or ""),
            str(result.get("relation_type") or ""),
            str(result.get("primary_object") or ""),
            tuple(sorted(_related_ids(result))),
        )
        existing = by_key.get(key)
        if existing is None or _result_rank(result) > _result_rank(existing):
            by_key[key] = result
    return sorted(by_key.values(), key=_sort_key)


def _sort_key(result: dict[str, Any]) -> tuple[int, str, str]:
    return (
        -LABEL_RANK.get(str(result.get("label") or ""), 0),
        str(result.get("metric") or ""),
        str(result.get("check_id") or ""),
    )


def _result_rank(result: dict[str, Any]) -> tuple[int, float]:
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return LABEL_RANK.get(str(result.get("label") or ""), 0), confidence


def _objects_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    geometry = ((payload.get("case_pack") or {}).get("scene_geometry") or {})
    return {
        str(obj.get("id") or ""): obj
        for obj in geometry.get("objects") or []
        if isinstance(obj, dict) and obj.get("id")
    }


def _is_prompt_issue(result: dict[str, Any]) -> bool:
    return (
        result.get("label") in ISSUE_LABELS
        and not _is_ignored_scoring_tier(result)
    )


def _is_ignored_scoring_tier(result: dict[str, Any]) -> bool:
    return str(result.get("scoring_tier") or "").strip().lower() == "ignored"


def _is_self_relation(result: dict[str, Any]) -> bool:
    subject_id = str(result.get("primary_object") or "")
    return bool(subject_id) and subject_id in _related_ids(result)


def _related_ids(result: dict[str, Any]) -> list[str]:
    return [str(item) for item in (result.get("related_objects") or []) if str(item)]


def _scene_object_type(obj: dict[str, Any] | None) -> str:
    if not obj:
        return ""
    hints = obj.get("functional_hints") or {}
    return str(
        hints.get("scene_object_type") or obj.get("object_type") or ""
    ).strip().lower()


def _category(obj: dict[str, Any] | None) -> str:
    return str(
        (obj or {}).get("category_norm") or (obj or {}).get("category") or ""
    ).strip().lower()


def _category_text(obj: dict[str, Any] | None) -> str:
    obj = obj or {}
    return " ".join(
        str(obj.get(key) or "").strip().lower().replace("-", "_").replace(" ", "_")
        for key in ("id", "name", "category", "category_norm", "description")
    )


def _is_seating(obj: dict[str, Any] | None) -> bool:
    hints = (obj or {}).get("functional_hints") or {}
    affordances = {
        str(item).strip().lower()
        for item in (
            hints.get("functional_categories")
            or hints.get("candidate_affordances")
            or []
        )
    }
    return "sittable" in affordances or _category(obj) in {
        "armchair",
        "bench",
        "chair",
        "dining_chair",
        "office_chair",
        "sofa",
        "stool",
    }


def _is_work_surface(obj: dict[str, Any] | None) -> bool:
    hints = (obj or {}).get("functional_hints") or {}
    return str(hints.get("category_group") or "").strip().lower() in {
        "storage_surface",
        "work_surface",
    } or _category(obj) in {"desk", "table", "counter", "island"}


def _is_workstation_object(obj: dict[str, Any] | None) -> bool:
    return _is_computer_screen(obj) or _is_computer_peripheral(obj)


def _is_computer_screen(obj: dict[str, Any] | None) -> bool:
    category = _category(obj)
    text = _category_text(obj)
    return category in COMPUTER_SCREEN_CATEGORIES or any(
        token in text for token in ("computer_monitor", "display", "screen")
    )


def _is_computer_peripheral(obj: dict[str, Any] | None) -> bool:
    category = _category(obj)
    text = _category_text(obj)
    return category in COMPUTER_PERIPHERAL_CATEGORIES or any(
        token in text for token in ("keyboard", "mouse", "trackpad", "touchpad")
    )


def _surface_ids(obj: dict[str, Any]) -> set[str]:
    return {
        str(region.get("region_id") or "")
        for region in obj.get("support_regions") or []
        if isinstance(region, dict) and region.get("region_id")
    }


def _parent_surface_id(obj: dict[str, Any]) -> str:
    placement = obj.get("placement_info") or {}
    return str(placement.get("parent_surface_id") or "")


def _agent_value(agent_type: AgentType | str) -> str:
    if isinstance(agent_type, AgentType):
        return agent_type.value
    return str(agent_type or "").strip().lower()


def _write_debug_context(
    output_dir: Path,
    payload: dict[str, Any],
    filtered: list[dict[str, Any]],
    agent_type: AgentType | str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_issues = [
        result for result in payload.get("results") or [] if _is_prompt_issue(result)
    ]
    filtered_ids = {str(result.get("check_id") or "") for result in filtered}
    filtered_out = [
        {
            "check_id": result.get("check_id"),
            "reason": _debug_filter_reason(result, filtered_ids),
        }
        for result in raw_issues
        if str(result.get("check_id") or "") not in filtered_ids
    ]
    debug_payload = {
        "schema_version": "scenesmith.scenebenchmark_critic.prompt_context_debug.v1",
        "agent_type": _agent_value(agent_type),
        "raw_issue_count": len(raw_issues),
        "filtered_issue_count": len(filtered),
        "raw_issue_ids": [result.get("check_id") for result in raw_issues],
        "filtered_issue_ids": sorted(filtered_ids),
        "filtered_out": filtered_out,
    }
    (output_dir / "scenebenchmark_prompt_context_debug.json").write_text(
        json.dumps(debug_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _debug_filter_reason(result: dict[str, Any], filtered_ids: set[str]) -> str:
    check_id = str(result.get("check_id") or "")
    if check_id in filtered_ids:
        return "kept"
    if _is_self_relation(result):
        return "self_relation"
    relation_type = str(result.get("relation_type") or "")
    if relation_type:
        return "not_relevant_to_current_agent_scope_or_relation_policy"
    return "not_relevant_to_current_agent_scope"
