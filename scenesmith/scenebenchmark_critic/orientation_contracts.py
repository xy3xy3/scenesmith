"""Stable orientation contracts for SceneBenchmark functional dependencies."""

from __future__ import annotations

import logging

from typing import Any

from scenesmith.agent_utils.room import RoomScene
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.geometry import (
    bbox_gap_xy,
    distance_xy,
    object_affordances,
    object_category,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.semantics import (
    _is_actionable_seating_surface_pair,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.profiles import (
    object_function_profile,
)

console_logger = logging.getLogger(__name__)

CONTRACT_CHECK_SOURCE = "scenesmith_orientation_contract"
CONTRACT_ATTR = "_scenebenchmark_orientation_contracts"

SEATING_RELATIONS = {"seating_to_media", "seating_to_work_surface"}
MEDIA_CATEGORIES = {
    "display",
    "display_board",
    "entertainment_center",
    "media_console",
    "monitor",
    "projection_screen",
    "screen",
    "television",
    "tv",
    "tv_stand",
}
MEDIA_TEXT_HINTS = (
    "display",
    "entertainment",
    "media",
    "monitor",
    "projector",
    "screen",
    "television",
    "tv",
)
MEDIA_REJECT_HINTS = ("coffee table", "side table", "end table", "lamp")
MEDIA_INTENT_HINTS = MEDIA_TEXT_HINTS + ("viewing", "watch", "watching")
MEDIA_ROOM_HINTS = ("family", "living", "media", "theater", "tv")
LIVING_SEATING = {"armchair", "chair", "loveseat", "sofa"}
WORK_SURFACE_CATEGORIES = {
    "bar_table",
    "coffee_table",
    "counter",
    "desk",
    "dining_table",
    "island",
    "side_table",
    "table",
}
STANDALONE_WALL_ANCHOR_GAP_M = 0.12
STANDALONE_SURFACE_GAP_M = 0.85


def stabilize_orientation_contracts(
    case_pack: dict[str, Any],
    scene: RoomScene,
    config: CriticConfig,
    *,
    stage: str,
) -> None:
    """Keep seating orientation targets stable across SceneBenchmark stages.

    SceneBenchmark may evaluate the same in-progress scene multiple times. Without
    a stable contract, its FD proposer can pick a fresh target from current geometry
    each time, which is noisy for seating that could reasonably face either a table
    or a media focal point. This stores a room-local contract on the live RoomScene
    object and injects matching FD checks into the current case pack.
    """
    if not _enabled(config):
        return

    geometry = case_pack.get("scene_geometry") or {}
    objects = [item for item in geometry.get("objects") or [] if isinstance(item, dict)]
    objects_by_id = {
        str(item.get("id") or ""): item for item in objects if item.get("id")
    }
    if not objects_by_id:
        return

    memory = getattr(scene, CONTRACT_ATTR, None)
    if not isinstance(memory, dict):
        memory = {}
        setattr(scene, CONTRACT_ATTR, memory)

    task_text = str(case_pack.get("task_instruction") or "")
    room_type = str(case_pack.get("room_type") or "")
    media_focus = _best_media_focus(objects, task_text=task_text, room_type=room_type)
    media_intent = _has_media_intent(task_text, room_type) and media_focus is not None

    checks_added = 0
    for subject in objects:
        subject_id = str(subject.get("id") or "")
        if not subject_id or not _is_seating(subject):
            continue

        existing = memory.get(subject_id)
        if _contract_is_usable(existing, objects_by_id, media_intent, media_focus):
            contract = dict(existing)
            contract["stage_last_seen"] = stage
        else:
            contract = _plan_contract(
                subject,
                objects,
                media_focus=media_focus,
                media_intent=media_intent,
                stage=stage,
            )

        if not contract:
            memory.pop(subject_id, None)
            continue

        memory[subject_id] = contract
        _replace_contract_check(case_pack, subject, contract)
        checks_added += 1

    if checks_added:
        console_logger.info(
            "SceneBenchmark orientation contracts active for %d seating object(s) "
            "at stage %s",
            checks_added,
            stage,
        )


def orientation_contract_subjects(case_pack: dict[str, Any]) -> set[str]:
    """Return subjects whose seating FD target is fixed by a contract check."""
    subjects: set[str] = set()
    for check in case_pack.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("check_source") != CONTRACT_CHECK_SOURCE:
            continue
        if str(check.get("relation_type") or "") not in SEATING_RELATIONS:
            continue
        subject_id = str(check.get("subject_id") or "")
        if subject_id:
            subjects.add(subject_id)
    return subjects


def _enabled(config: CriticConfig) -> bool:
    value = config.extra.get("stable_orientation_contracts", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _contract_is_usable(
    contract: Any,
    objects_by_id: dict[str, dict[str, Any]],
    media_intent: bool,
    media_focus: dict[str, Any] | None,
) -> bool:
    if not isinstance(contract, dict):
        return False
    target_ids = [str(item) for item in contract.get("target_ids") or [] if str(item)]
    if not target_ids or any(
        target_id not in objects_by_id for target_id in target_ids
    ):
        return False
    relation_type = str(contract.get("relation_type") or "")
    if relation_type not in SEATING_RELATIONS:
        return False

    # A newly available semantic focal point is a legitimate topology change.
    if media_intent and media_focus is not None and relation_type != "seating_to_media":
        return False
    return True


def _plan_contract(
    subject: dict[str, Any],
    objects: list[dict[str, Any]],
    *,
    media_focus: dict[str, Any] | None,
    media_intent: bool,
    stage: str,
) -> dict[str, Any] | None:
    if media_intent and media_focus is not None and _should_face_media(subject):
        return _contract(
            subject,
            media_focus,
            relation_type="seating_to_media",
            stage=stage,
            reason=(
                "room/task has a media focal point; seating keeps that facing "
                "target across stages"
            ),
        )

    surface = _nearest_work_surface(subject, objects)
    if surface is None:
        return None
    return _contract(
        subject,
        surface,
        relation_type="seating_to_work_surface",
        stage=stage,
        reason=(
            "no media focal point is active; seating uses the nearest functional "
            "surface"
        ),
    )


def _contract(
    subject: dict[str, Any],
    target: dict[str, Any],
    *,
    relation_type: str,
    stage: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "scenesmith.scenebenchmark_critic.orientation_contract.v1",
        "subject_id": str(subject.get("id") or ""),
        "target_ids": [str(target.get("id") or "")],
        "relation_type": relation_type,
        "stage_created": stage,
        "stage_last_seen": stage,
        "target_category": object_category(target),
        "policy": (
            "lock_primary_target_until_target_removed_or_semantic_focal_point_added"
        ),
        "reason": reason,
    }


def _replace_contract_check(
    case_pack: dict[str, Any],
    subject: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    subject_id = str(contract.get("subject_id") or subject.get("id") or "")
    target_ids = [str(item) for item in contract.get("target_ids") or [] if str(item)]
    relation_type = str(contract.get("relation_type") or "")
    if not subject_id or not target_ids or relation_type not in SEATING_RELATIONS:
        return

    checks = [
        check
        for check in case_pack.get("checks") or []
        if not (
            isinstance(check, dict)
            and check.get("check_source") == CONTRACT_CHECK_SOURCE
            and str(check.get("subject_id") or "") == subject_id
        )
    ]
    check_id = f"fd_contract_{subject_id}_{'_'.join(target_ids)}_{relation_type}"
    checks.append(
        {
            "check_id": check_id,
            "metric": "functional_dependency",
            "subject_id": subject_id,
            "target_ids": target_ids,
            "relation_type": relation_type,
            "expected_use": _expected_use(relation_type),
            "priority_weight": 0.9,
            "question": (
                f"Does the stable orientation contract `{relation_type}` hold "
                f"for `{subject_id}`?"
            ),
            "evidence": {
                "source": CONTRACT_CHECK_SOURCE,
                "policy": contract.get("policy"),
                "reason": contract.get("reason"),
                "stage_created": contract.get("stage_created"),
                "stage_last_seen": contract.get("stage_last_seen"),
            },
            "evidence_refs": ["scene_geometry"],
            "check_source": CONTRACT_CHECK_SOURCE,
            "scoring_tier": "core",
        }
    )
    case_pack["checks"] = checks


def _expected_use(relation_type: str) -> str:
    if relation_type == "seating_to_media":
        return "sit and view the room's chosen media focal point"
    return "sit at and use the chosen table or work surface"


def _best_media_focus(
    objects: list[dict[str, Any]], *, task_text: str, room_type: str
) -> dict[str, Any] | None:
    candidates = [obj for obj in objects if _is_media_target(obj)]
    if not candidates:
        return None
    has_media_intent = _has_media_intent(task_text, room_type)
    if not has_media_intent:
        return None
    candidates.sort(key=lambda obj: (_media_rank(obj), str(obj.get("id") or "")))
    return candidates[0]


def _has_media_intent(task_text: str, room_type: str) -> bool:
    text = f"{task_text} {room_type}".lower()
    return any(hint in text for hint in MEDIA_INTENT_HINTS + MEDIA_ROOM_HINTS)


def _is_media_target(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    text = _object_text(obj)
    if any(hint in text for hint in MEDIA_REJECT_HINTS):
        return False
    return category in MEDIA_CATEGORIES or any(
        hint in text for hint in MEDIA_TEXT_HINTS
    )


def _media_rank(obj: dict[str, Any]) -> tuple[int, float]:
    category = object_category(obj)
    text = _object_text(obj)
    if category in {"television", "tv", "monitor", "screen", "projection_screen"}:
        kind_rank = 0
    elif any(hint in text for hint in ("television", "tv", "monitor", "screen")):
        kind_rank = 1
    elif category in {"entertainment_center", "media_console", "tv_stand"}:
        kind_rank = 2
    else:
        kind_rank = 3
    center = (obj.get("bbox_world") or {}).get("center") or [0.0, 0.0]
    try:
        centrality = abs(float(center[0])) + abs(float(center[1])) * 0.05
    except Exception:
        centrality = 999.0
    return kind_rank, centrality


def _is_seating(obj: dict[str, Any]) -> bool:
    # 2026-07-08 修改原因：asset affordances 会把部分非座椅误标成 sittable，
    # orientation contract 必须使用归一化后的功能画像，避免给桌、灯、小物生成 seating 关系。
    return object_function_profile(obj).is_seating and (
        "sittable" in object_affordances(obj) or object_category(obj) in LIVING_SEATING
    )


def _should_face_media(subject: dict[str, Any]) -> bool:
    category = object_category(subject)
    return _is_seating(subject) and category in LIVING_SEATING


def _nearest_work_surface(
    subject: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidates = [
        obj
        for obj in objects
        if obj.get("id") != subject.get("id")
        and _is_work_surface(obj)
        and _is_actionable_seating_surface_pair(subject, obj)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda obj: _surface_rank(subject, obj))
    if _is_wall_anchored_standalone_seating(subject, candidates[0], objects):
        return None
    return candidates[0]


def _is_work_surface(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    if category in {"bookcase", "bookshelf", "shelf", "wall_shelf"}:
        return False
    if category in WORK_SURFACE_CATEGORIES:
        return True
    return object_function_profile(obj).is_work_surface and not _is_media_target(obj)


def _surface_rank(
    subject: dict[str, Any], target: dict[str, Any]
) -> tuple[float, float, str]:
    gap = bbox_gap_xy(subject, target)
    dist = distance_xy(subject, target)
    return (
        gap if gap is not None else 999.0,
        dist if dist is not None else 999.0,
        str(target.get("id") or ""),
    )


def _is_wall_anchored_standalone_seating(
    subject: dict[str, Any],
    target: dict[str, Any],
    objects: list[dict[str, Any]],
) -> bool:
    wall_gap = _nearest_wall_gap(subject, objects)
    surface_gap = bbox_gap_xy(subject, target)
    if wall_gap is None or surface_gap is None:
        return False
    category = object_category(subject)
    if category not in {"armchair", "chair", "dining_chair", "office_chair"}:
        return False
    # 2026-07-10 修改原因：靠墙空闲椅子不应被稳定 contract 绑定到远处桌面；
    # 只要已经贴墙摆放且最近桌面并不近，就保留为 standalone chair。
    return wall_gap <= STANDALONE_WALL_ANCHOR_GAP_M and surface_gap > STANDALONE_SURFACE_GAP_M


def _nearest_wall_gap(subject: dict[str, Any], objects: list[dict[str, Any]]) -> float | None:
    subject_bbox = (subject.get("bbox_world") or {})
    subject_min = subject_bbox.get("min") or []
    subject_max = subject_bbox.get("max") or []
    if len(subject_min) < 2 or len(subject_max) < 2:
        return None
    best: float | None = None
    for obj in objects:
        if object_category(obj) != "wall":
            continue
        wall_bbox = obj.get("bbox_world") or {}
        wall_min = wall_bbox.get("min") or []
        wall_max = wall_bbox.get("max") or []
        if len(wall_min) < 2 or len(wall_max) < 2:
            continue
        dx = max(float(wall_min[0] - subject_max[0]), float(subject_min[0] - wall_max[0]), 0.0)
        dy = max(float(wall_min[1] - subject_max[1]), float(subject_min[1] - wall_max[1]), 0.0)
        gap = (dx * dx + dy * dy) ** 0.5
        if best is None or gap < best:
            best = gap
    return best


def _object_text(obj: dict[str, Any]) -> str:
    parts = [
        obj.get("id"),
        obj.get("name"),
        obj.get("description"),
        obj.get("category"),
        obj.get("category_norm"),
    ]
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        parts.extend(
            [
                metadata.get("category"),
                metadata.get("asset_category"),
                metadata.get("semantic_label"),
            ]
        )
    return " ".join(str(part).lower() for part in parts if part)
