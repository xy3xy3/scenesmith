"""Check construction for the embedded rule critic."""

from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic import clearance_source
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.geometry import (
    bbox_gap_xy,
    distance_xy,
    is_small_object,
    object_affordances,
    object_category,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.constants import (
    DINING_TABLES,
    LAMP_SUBJECT_REJECT_HINTS,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.relations import (
    _infer_relation_type,
    _relation_target_is_valid,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.semantics import (
    _is_nightstand_target,
    _is_seating_subject,
    _is_work_surface_target,
)

ACCESS_AFFORDANCES = {"sittable", "openable", "supportable", "sleepable", "graspable"}
ACCESS_AFFORDANCE_PRIORITY = (
    "openable",
    "sittable",
    "graspable",
    "supportable",
    "sleepable",
)
SMALL_SA_SUBJECT_HINTS = (
    "book",
    "notebook",
    "pen",
    "pencil",
    "figurine",
    "toy",
    "vase",
    "bottle",
    "cup",
    "mug",
    "plate",
    "bowl",
)
WALL_BACKED_RELATIONS = {"back_against_wall", "side_or_back_against_wall"}


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
    seen_check_ids: set[str] = set()

    if "spatial_accessibility" in enabled:
        for obj in objects.values():
            affordance = _spatial_access_affordance(obj)
            if affordance is None:
                continue
            check_id = f"spatial_accessibility__{obj['id']}"
            target_ids = _spatial_access_target_ids(obj, objects.values(), affordance)
            checks.append(
                {
                    "check_id": check_id,
                    "metric": "spatial_accessibility",
                    "subject_id": obj["id"],
                    "target_ids": target_ids,
                    "affordance": affordance,
                    "functional_categories": sorted(object_affordances(obj)),
                    "priority_weight": _priority_weight(
                        obj, "spatial_accessibility", 1.0
                    ),
                    "question": (
                        f"Is {obj.get('name') or obj['id']} spatially accessible "
                        f"for {affordance} use?"
                    ),
                    "evidence_refs": ["scene_geometry"],
                    "scoring_tier": "core",
                }
            )
            seen_check_ids.add(check_id)

    if "functional_dependency" in enabled:
        for relation in geometry.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            if relation.get("annotation_source") != "metadata":
                continue
            subject_id = str(
                relation.get("subject_id") or relation.get("subject") or ""
            )
            target_ids = [
                str(item)
                for item in (relation.get("target_ids") or [])
                if str(item) in objects
            ]
            if subject_id not in objects or not target_ids:
                continue
            relation_type = _metadata_relation_type(
                relation, objects[subject_id], objects[target_ids[0]]
            )
            check_id = f"fd_{subject_id}_{'_'.join(target_ids)}_{relation_type}"
            if check_id in seen_check_ids:
                continue
            checks.append(
                {
                    "check_id": check_id,
                    "metric": "functional_dependency",
                    "subject_id": subject_id,
                    "target_ids": target_ids,
                    "relation_type": relation_type,
                    "expected_use": _expected_use(relation_type),
                    "priority_weight": _priority_weight(
                        objects[subject_id], "functional_dependency", 0.7
                    ),
                    "question": (
                        f"Does metadata relation `{relation_type}` hold for "
                        f"{objects[subject_id].get('name') or subject_id}?"
                    ),
                    "evidence": {
                        "annotation_source": "metadata",
                        "target_surface_id": relation.get("target_surface_id"),
                        "reason": relation.get("reason"),
                    },
                    "evidence_refs": ["scene_geometry", "object_metadata"],
                    "scoring_tier": str(relation.get("scoring_tier") or "core"),
                }
            )
            seen_check_ids.add(check_id)

        surface_owner = _surface_owner_map(objects)
        for obj in objects.values():
            placement = obj.get("placement_info") or {}
            surface_id = str(placement.get("parent_surface_id") or "")
            target_id = surface_owner.get(surface_id)
            if not surface_id or not target_id:
                continue
            relation_type = _relation_type_for(obj, objects[target_id])
            check_id = f"fd_{obj['id']}_{target_id}_{relation_type}"
            if check_id in seen_check_ids:
                continue
            checks.append(
                {
                    "check_id": check_id,
                    "metric": "functional_dependency",
                    "subject_id": obj["id"],
                    "target_ids": [target_id],
                    "relation_type": relation_type,
                    "expected_use": _expected_use(relation_type),
                    "priority_weight": _priority_weight(
                        obj, "functional_dependency", 0.7
                    ),
                    "question": (
                        f"Is {obj.get('name') or obj['id']} functionally supported "
                        f"by {objects[target_id].get('name') or target_id}?"
                    ),
                    "evidence": {"parent_surface_id": surface_id},
                    "evidence_refs": ["scene_geometry", "placement_info"],
                    "scoring_tier": "core",
                }
            )
            seen_check_ids.add(check_id)

        checks.extend(_build_explicit_target_relation_checks(objects, seen_check_ids))
        checks.extend(_build_dependency_annotation_checks(objects, seen_check_ids))
        checks.extend(
            _build_grouped_functional_dependency_checks(objects, seen_check_ids)
        )

    if "clearance" in enabled:
        for check in clearance_source.build_clearance_checks(objects):
            if check["check_id"] in seen_check_ids:
                continue
            checks.append(check)
            seen_check_ids.add(check["check_id"])
    return checks


def _spatial_access_affordance(obj: dict[str, Any]) -> str | None:
    policy = _accessibility_policy(obj)
    if policy in {"ignored", "optional"}:
        return None
    if _should_drop_small_spatial_accessibility(obj):
        return None
    if _should_exclude_from_spatial_access_subjects(obj):
        return None
    affordances = object_affordances(obj) & ACCESS_AFFORDANCES
    if not affordances:
        return None
    for affordance in ACCESS_AFFORDANCE_PRIORITY:
        if affordance in affordances:
            return affordance
    return sorted(affordances)[0]


def _accessibility_policy(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    policy = _normalize_token(hints.get("accessibility_policy"))
    if policy in {"required", "optional", "ignored"}:
        return policy
    return "required"


def _should_exclude_from_spatial_access_subjects(obj: dict[str, Any]) -> bool:
    """Skip mounted ceiling objects as spatial-accessibility subjects."""
    if _scene_object_type(obj) == "ceiling_mounted":
        return True
    hints = obj.get("functional_hints") or {}
    group = _normalize_token(hints.get("category_group"))
    if group in {"ceiling", "ceiling_mounted"}:
        return True
    category = object_category(obj)
    if "ceiling" in category:
        return True
    if "lamp" in category or "light" in category:
        return _text_has_any(obj, LAMP_SUBJECT_REJECT_HINTS)
    return False


def _scene_object_type(obj: dict[str, Any]) -> str:
    hints = obj.get("functional_hints") or {}
    value = _normalize_token(hints.get("scene_object_type"))
    if value in {"wall_mounted", "manipuland", "ceiling_mounted", "furniture"}:
        return value
    return "unknown"


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _text_has_any(obj: dict[str, Any], hints: tuple[str, ...]) -> bool:
    text = " ".join(
        str(obj.get(key) or "").strip().lower()
        for key in (
            "id",
            "name",
            "category",
            "category_norm",
            "asset_id",
            "description",
        )
    )
    text = text.replace("-", "_").replace(" ", "_")
    return any(hint in text for hint in hints)


def _priority_weight(obj: dict[str, Any], metric: str, default: float) -> float:
    hints = obj.get("functional_hints") or {}
    metric_relevance = hints.get("metric_relevance") or {}
    if isinstance(metric_relevance, dict):
        try:
            value = metric_relevance.get(metric)
            if value:
                return float(value)
        except (TypeError, ValueError):
            pass
    return float(default)


def _spatial_access_target_ids(
    subject: dict[str, Any],
    candidates: Any,
    affordance: str,
) -> list[str]:
    limit = 4 if affordance == "sittable" else 2
    targets = _nearby_targets(
        subject,
        candidates,
        predicate=_is_spatial_access_target,
        max_gap_m=1.6 if affordance == "sittable" else 1.0,
        limit=limit,
    )
    return [str(target.get("id") or "") for target in targets if target.get("id")]


def _is_spatial_access_target(candidate: dict[str, Any]) -> bool:
    category = object_category(candidate)
    if category in {"floor", "wall", "door", "window", "ceiling"}:
        return False
    if _should_drop_small_spatial_accessibility(candidate):
        return False
    affordances = object_affordances(candidate)
    if affordances & {"sittable", "supportable", "openable", "sleepable"}:
        return True
    group = str((candidate.get("functional_hints") or {}).get("category_group") or "")
    return group in {
        "seating",
        "sleeping",
        "storage",
        "storage_surface",
        "work_surface",
    }


def _should_drop_small_spatial_accessibility(obj: dict[str, Any]) -> bool:
    if _scene_object_type(obj) == "manipuland":
        return False
    if is_small_object(obj):
        return True
    text = " ".join(
        str(obj.get(key) or "").strip().lower()
        for key in ("id", "category", "category_norm", "asset_id", "description")
    )
    if any(hint in text for hint in SMALL_SA_SUBJECT_HINTS):
        return True
    hints = obj.get("functional_hints") or {}
    group = str(hints.get("category_group") or "")
    affordances = object_affordances(obj)
    if group == "small_object" or affordances == {"graspable"}:
        return True
    bbox = obj.get("bbox_world") or {}
    size = bbox.get("size") or [0.0, 0.0, 0.0]
    if len(size) >= 3:
        area = max(float(size[0] or 0.0), 0.0) * max(float(size[1] or 0.0), 0.0)
        height = float(size[2] or 0.0)
        if area <= 0.12 and height <= 0.5:
            return True
    category = object_category(obj)
    return category in {
        "book",
        "vase",
        "mug",
        "cup",
        "bottle",
        "remote",
        "laptop",
        "tray",
        "plate",
        "bowl",
    }


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


def _metadata_relation_type(
    relation: dict[str, Any], subject: dict[str, Any], target: dict[str, Any]
) -> str:
    relation_type = str(relation.get("relation_type") or "").strip()
    if relation_type and relation_type not in {
        "functional_dependency",
        "placed_on_surface",
    }:
        return relation_type
    return _relation_type_for(subject, target)


def _build_explicit_target_relation_checks(
    objects: dict[str, dict[str, Any]], seen_check_ids: set[str]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for subject in objects.values():
        target_relations = _explicit_target_relations(subject)
        if not target_relations:
            continue
        targets = _targets_matching_relations(
            subject, objects.values(), target_relations
        )
        if not targets:
            continue
        relation_type = _infer_relation_type(subject, targets[0]) or _relation_type_for(
            subject, targets[0]
        )
        compatible_targets = [
            target
            for target in targets
            if _relation_target_is_valid(subject, target, relation_type)
        ]
        if not compatible_targets:
            compatible_targets = targets[:1]
        target_ids = [
            str(target.get("id") or "")
            for target in compatible_targets
            if target.get("id")
        ]
        check_id = f"fd_{subject['id']}_{'_'.join(target_ids)}_{relation_type}"
        if check_id in seen_check_ids:
            continue
        checks.append(
            {
                "check_id": check_id,
                "metric": "functional_dependency",
                "subject_id": subject["id"],
                "target_ids": target_ids,
                "relation_type": relation_type,
                "expected_use": _expected_use(relation_type),
                "priority_weight": _priority_weight(
                    subject, "functional_dependency", 0.7
                ),
                "question": (
                    f"Does explicit target relation `{relation_type}` hold for "
                    f"{subject.get('name') or subject['id']}?"
                ),
                "evidence": {"explicit_target_relation": target_relations},
                "evidence_refs": ["scene_geometry", "object_metadata"],
                "check_source": "asset_explicit_target_relation",
                "scoring_tier": "core",
            }
        )
        seen_check_ids.add(check_id)
    return checks


def _explicit_target_relations(obj: dict[str, Any]) -> list[str]:
    hints = obj.get("functional_hints") or {}
    raw = hints.get("explicit_target_relation")
    values = raw if isinstance(raw, list) else [raw]
    return [str(item).strip() for item in values if str(item).strip()]


def _targets_matching_relations(
    subject: dict[str, Any],
    candidates: Any,
    target_relations: list[str],
) -> list[dict[str, Any]]:
    targets = _nearby_targets(
        subject,
        candidates,
        predicate=lambda candidate: _matches_target_relation(
            candidate, target_relations
        )
        and _is_useful_explicit_target(subject, candidate, target_relations),
        max_gap_m=2.4,
        limit=4,
    )
    return targets


def _matches_target_relation(
    candidate: dict[str, Any], target_relations: list[str]
) -> bool:
    category = object_category(candidate)
    if not category:
        return False
    affordances = object_affordances(candidate)
    normalized_category = _normalize_relation_token(category)
    for target in target_relations:
        normalized_target = _normalize_relation_token(target)
        if not normalized_target:
            continue
        if normalized_target == "graspable_object" and "graspable" in affordances:
            return True
        if normalized_target == normalized_category:
            return True
        if (
            normalized_target in {"table", "desk"}
            and normalized_target in normalized_category
        ):
            return True
        if normalized_category.startswith(normalized_target + "_"):
            return True
        if normalized_category.endswith("_" + normalized_target):
            return True
    return False


def _normalize_relation_token(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _is_useful_explicit_target(
    subject: dict[str, Any],
    candidate: dict[str, Any],
    target_relations: list[str],
) -> bool:
    if subject.get("id") == candidate.get("id"):
        return False
    if object_category(candidate) in {"wall", "door", "window"}:
        return False
    if _should_drop_small_spatial_accessibility(candidate):
        subject_affordances = object_affordances(subject)
        candidate_affordances = object_affordances(candidate)
        wants_graspable_object = any(
            _normalize_relation_token(relation) == "graspable_object"
            for relation in target_relations
        )
        return "graspable" in subject_affordances or (
            wants_graspable_object and "graspable" in candidate_affordances
        )
    return True


def _build_grouped_functional_dependency_checks(
    objects: dict[str, dict[str, Any]], seen_check_ids: set[str]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for obj in objects.values():
        category = object_category(obj)
        if category in DINING_TABLES:
            checks.extend(_grouped_dining_checks(obj, objects, seen_check_ids))
        elif _is_workstation_surface(obj):
            checks.extend(_grouped_workstation_checks(obj, objects, seen_check_ids))
        elif category == "bed":
            checks.extend(_grouped_bedside_checks(obj, objects, seen_check_ids))
    return checks


def _build_dependency_annotation_checks(
    objects: dict[str, dict[str, Any]], seen_check_ids: set[str]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for subject in objects.values():
        subject_id = str(subject.get("id") or "")
        for source_key, source_name in (
            ("attachment_dependencies", "asset_attachment_dependency"),
            ("orientation_dependencies", "asset_orientation_dependency"),
        ):
            for index, dependency in enumerate(
                _dependency_annotation_items(subject, source_key)
            ):
                relation_type = _normalize_relation_token(
                    dependency.get("relation_type") or dependency.get("type")
                )
                if not relation_type:
                    continue
                if _dependency_conflicts_with_placement(
                    subject, relation_type, source_key
                ):
                    continue
                targets = _dependency_targets(subject, objects.values(), dependency)
                if not targets:
                    continue
                target_ids = [
                    str(target.get("id") or "")
                    for target in targets
                    if target.get("id")
                ]
                if not target_ids:
                    continue
                check_id = (
                    f"fd_{subject_id}_{'_'.join(target_ids)}_"
                    f"{relation_type}_{source_key}_{index}"
                )
                if check_id in seen_check_ids:
                    continue
                checks.append(
                    {
                        "check_id": check_id,
                        "metric": "functional_dependency",
                        "subject_id": subject_id,
                        "target_ids": target_ids,
                        "relation_type": relation_type,
                        "expected_use": _expected_use(relation_type),
                        "priority_weight": _priority_weight(
                            subject, "functional_dependency", 0.8
                        ),
                        "question": (
                            f"Does annotated dependency `{relation_type}` hold for "
                            f"{subject.get('name') or subject_id}?"
                        ),
                        "evidence": {
                            "dependency": dependency,
                            "dependency_key": source_key,
                            "annotation_source": source_name,
                        },
                        "evidence_refs": ["scene_geometry", "object_metadata"],
                        "check_source": source_name,
                        "scoring_tier": str(dependency.get("scoring_tier") or "core"),
                    }
                )
                seen_check_ids.add(check_id)
    return checks


def _dependency_annotation_items(
    obj: dict[str, Any], source_key: str
) -> list[dict[str, Any]]:
    hints = obj.get("functional_hints") or {}
    raw = hints.get(source_key)
    values = raw if isinstance(raw, list) else [raw]
    return [dict(item) for item in values if isinstance(item, dict)]


def _dependency_conflicts_with_placement(
    subject: dict[str, Any], relation_type: str, source_key: str
) -> bool:
    if source_key != "attachment_dependencies" or relation_type != "object_on_floor":
        return False
    placement = subject.get("placement_info") or {}
    parent_surface_id = str(placement.get("parent_surface_id") or "").strip()
    if not parent_surface_id:
        return False
    hints = subject.get("functional_hints") or {}
    placement_class = _normalize_relation_token(hints.get("placement_class"))
    scene_object_type = _scene_object_type(subject)
    return scene_object_type == "manipuland" or placement_class in {
        "surface_object",
        "tabletop_object",
        "shelf_object",
    }


def _dependency_targets(
    subject: dict[str, Any],
    candidates: Any,
    dependency: dict[str, Any],
) -> list[dict[str, Any]]:
    explicit_ids = {
        str(value)
        for value in _as_dependency_target_id_values(dependency)
        if str(value)
    }
    if explicit_ids:
        explicit_targets = [
            candidate
            for candidate in candidates
            if str(candidate.get("id") or "") in explicit_ids
        ]
        if explicit_targets:
            return explicit_targets

    target_kind = _normalize_relation_token(dependency.get("target_kind") or "object")
    target_categories = _dependency_target_categories(dependency)
    if not target_categories:
        return []

    max_distance = _float_value(dependency.get("max_distance_m"), 2.4)
    limit = 6 if target_kind == "architecture" else 4
    nearby_targets = _nearby_targets(
        subject,
        candidates,
        predicate=lambda candidate: _dependency_target_matches(
            candidate, target_categories, target_kind
        ),
        max_gap_m=max_distance,
        limit=limit,
    )
    if nearby_targets:
        return nearby_targets

    relation_type = _normalize_relation_token(
        dependency.get("relation_type") or dependency.get("type")
    )
    if (
        target_kind == "architecture"
        and "wall" in target_categories
        and relation_type in WALL_BACKED_RELATIONS
    ):
        nearest_wall = _nearest_dependency_target(
            subject,
            candidates,
            predicate=lambda candidate: _dependency_target_matches(
                candidate, target_categories, target_kind
            ),
        )
        if nearest_wall is not None:
            return [nearest_wall]
    return []


def _nearest_dependency_target(
    subject: dict[str, Any], candidates: Any, *, predicate: Any
) -> dict[str, Any] | None:
    subject_id = str(subject.get("id") or "")
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    for candidate in candidates:
        target_id = str(candidate.get("id") or "")
        if not target_id or target_id == subject_id:
            continue
        if not predicate(candidate):
            continue
        gap = bbox_gap_xy(subject, candidate)
        if gap is None:
            continue
        distance = distance_xy(subject, candidate)
        ranked.append(
            (gap, distance if distance is not None else 999.0, target_id, candidate)
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def _as_dependency_target_id_values(dependency: dict[str, Any]) -> list[Any]:
    raw = (
        dependency.get("target_ids")
        or dependency.get("targets")
        or dependency.get("target_id")
        or dependency.get("target")
        or dependency.get("object_id")
    )
    return raw if isinstance(raw, list) else [raw]


def _dependency_target_categories(dependency: dict[str, Any]) -> list[str]:
    raw = dependency.get("target_category")
    if raw is None:
        raw = dependency.get("target_categories")
    values = raw if isinstance(raw, list) else [raw]
    categories: list[str] = []
    for value in values:
        normalized = _normalize_relation_token(value)
        if normalized and normalized not in categories:
            categories.append(normalized)
    return categories


def _dependency_target_matches(
    candidate: dict[str, Any], target_categories: list[str], target_kind: str
) -> bool:
    category = _normalize_relation_token(object_category(candidate))
    if not category:
        return False
    if target_kind == "architecture" and category not in {"wall", "floor", "ceiling"}:
        return False
    for target_category in target_categories:
        if target_category == category:
            return True
        if category.startswith(target_category + "_"):
            return True
        if category.endswith("_" + target_category):
            return True
    return False


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _grouped_dining_checks(
    table: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    seen_check_ids: set[str],
) -> list[dict[str, Any]]:
    targets = _nearby_targets(
        table,
        objects.values(),
        predicate=_is_seating_subject,
        max_gap_m=1.8,
        limit=6,
    )
    if not targets:
        return []
    check = _grouped_fd_check(
        table,
        targets,
        relation_type="dining_set",
        seen_check_ids=seen_check_ids,
        expected_use="seating surrounds and faces a dining table",
    )
    if check is None:
        return []
    return [
        check,
    ]


def _grouped_workstation_checks(
    surface: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    seen_check_ids: set[str],
) -> list[dict[str, Any]]:
    targets = _nearby_targets(
        surface,
        objects.values(),
        predicate=_is_seating_subject,
        max_gap_m=1.8,
        limit=4,
    )
    if not targets:
        return []
    check = _grouped_fd_check(
        surface,
        targets,
        relation_type="workstation",
        seen_check_ids=seen_check_ids,
        expected_use="seat and work surface form a usable workstation",
    )
    if check is None:
        return []
    return [
        check,
    ]


def _is_workstation_surface(obj: dict[str, Any]) -> bool:
    category = object_category(obj)
    if category in {"desk", "office_desk", "computer_desk", "writing_desk"}:
        return True
    text = " ".join(
        str(obj.get(key) or "").strip().lower()
        for key in ("id", "name", "description", "category", "category_norm")
    )
    return "desk" in text and _is_work_surface_target(obj)


def _grouped_bedside_checks(
    bed: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    seen_check_ids: set[str],
) -> list[dict[str, Any]]:
    targets = _nearby_targets(
        bed,
        objects.values(),
        predicate=_is_nightstand_target,
        max_gap_m=1.2,
        limit=4,
    )
    if len(targets) < 2:
        return []
    check = _grouped_fd_check(
        bed,
        targets,
        relation_type="bedside_pair",
        seen_check_ids=seen_check_ids,
        expected_use="bed has one or more reachable bedside surfaces",
    )
    if check is None:
        return []
    return [
        check,
    ]


def _nearby_targets(
    subject: dict[str, Any],
    candidates: Any,
    *,
    predicate: Any,
    max_gap_m: float,
    limit: int,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    subject_id = str(subject.get("id") or "")
    for candidate in candidates:
        target_id = str(candidate.get("id") or "")
        if not target_id or target_id == subject_id:
            continue
        if not predicate(candidate):
            continue
        gap = bbox_gap_xy(subject, candidate)
        if gap is None or gap > max_gap_m:
            continue
        distance = distance_xy(subject, candidate)
        ranked.append(
            (gap, distance if distance is not None else 999.0, target_id, candidate)
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [candidate for *_rank, candidate in ranked[:limit]]


def _grouped_fd_check(
    subject: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    relation_type: str,
    seen_check_ids: set[str],
    expected_use: str,
) -> dict[str, Any] | None:
    subject_id = str(subject.get("id") or "")
    target_ids = [str(target.get("id") or "") for target in targets if target.get("id")]
    check_id = f"fd_{subject_id}_{'_'.join(target_ids)}_{relation_type}"
    if check_id in seen_check_ids:
        return None
    seen_check_ids.add(check_id)
    return {
        "check_id": check_id,
        "metric": "functional_dependency",
        "subject_id": subject_id,
        "target_ids": target_ids,
        "relation_type": relation_type,
        "expected_use": expected_use,
        "priority_weight": _priority_weight(subject, "functional_dependency", 0.7),
        "question": (
            f"Do {subject.get('name') or subject_id} and its nearby targets "
            f"form a valid `{relation_type}` relation?"
        ),
        "evidence_refs": ["scene_geometry"],
        "check_source": "scenesmith_grouped_relation",
        "scoring_tier": "core",
    }


def _expected_use(relation_type: str) -> str:
    return {
        "seating_to_work_surface": "sit at and use the nearby work or table surface",
        "seating_to_media": "sit and view the nearby media object",
        "dining_set": "seating surrounds and faces a dining table",
        "workstation": "seat, work surface, and work objects form a usable workstation",
        "bed_to_nightstand": "bed has a reachable bedside surface",
        "bedside_pair": "bed is paired with one or more usable bedside surfaces",
        "object_on_support": "small object is supported by an appropriate surface",
        "lamp_to_surface": "lamp serves a nearby support surface",
        "floor_covering_on_floor": "floor covering rests on the floor",
        "object_on_floor": "object rests on the floor",
        "back_against_wall": "furniture back is close to and faces a wall",
        "side_or_back_against_wall": "furniture side or back is close to a wall",
        "mounted_to_wall": "mounted object is attached to a wall",
        "mounted_to_ceiling": "mounted object is attached to the ceiling",
        "seat_faces_surface": "seat faces a usable work or table surface",
        "furniture_faces_furniture": "furniture face orientation matches its target",
    }.get(relation_type, "nearby objects form a valid use relation")
