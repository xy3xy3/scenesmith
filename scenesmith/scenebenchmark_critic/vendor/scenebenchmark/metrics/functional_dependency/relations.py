from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.geometry import (
    GeometryStore,
    angle_to_target_deg,
    bbox_gap_xy,
    distance_xy,
    object_category,
    seating_angle_to_target_deg,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.constants import *
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.profiles import (
    object_function_profile,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.results import (
    _empty_fd_diagnostics,
    _fd_diagnostics_from_targets,
    _fd_label_rank,
    _relation_label_rank,
    _result_scoring_tier_payload,
    _target_eval_payload,
    _unknown,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.semantics import (
    _category_group,
    _category_token_has_any,
    _is_actionable_seating_surface_pair,
    _is_any_lamp_object,
    _is_lamp_subject,
    _is_media_target,
    _is_nightstand_target,
    _is_seating_subject,
    _is_side_surface_target,
    _is_supported_small_subject,
    _is_work_surface_target,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.support import (
    _eval_object_on_support,
    _is_lamp_surface_target,
    _is_primary_support_target,
    evaluate_support_relation,
)


def evaluate_functional_dependency(
    store: GeometryStore,
    check: dict[str, Any],
) -> dict[str, Any]:
    subject_id = str(check.get("subject_id") or "")
    target_ids = [str(item) for item in (check.get("target_ids") or []) if str(item)]
    subject = store.objects.get(subject_id)
    targets = [
        store.objects[target_id]
        for target_id in target_ids
        if target_id in store.objects
    ]
    if subject is None:
        return _unknown(
            check, f"Rule dependency could not find subject object `{subject_id}`."
        )
    if not targets:
        return _unknown(
            check, f"Rule dependency found no valid target objects for `{subject_id}`."
        )

    relation_type = _normalize_relation_type(
        str(
            check.get("relation_type")
            or _infer_relation_type(subject, targets[0])
            or "generic_relation"
        )
    )
    label, confidence, reason, diagnostics = _eval_relation_over_targets(
        store, subject, targets, relation_type
    )
    selected_related_objects = [
        str(item)
        for item in (diagnostics.get("selected_target_ids") or [])
        if str(item)
    ]
    return {
        "check_id": check.get("check_id"),
        "metric": "functional_dependency",
        "label": label,
        "reason": f"Rule dependency `{relation_type}`: subject `{subject_id}`; {reason}",
        "blocking_objects": [],
        "confidence": confidence,
        "evaluation_source": "rule_functional_dependency",
        "primary_object": subject_id,
        "related_objects": target_ids,
        "selected_related_objects": selected_related_objects,
        "relation_type": relation_type,
        "diagnostics": diagnostics,
        **_result_scoring_tier_payload(check.get("scoring_tier")),
    }


def _infer_relation_type(subject: dict[str, Any], target: dict[str, Any]) -> str | None:
    sc = object_category(subject)
    subject_profile = object_function_profile(subject)
    if (
        _is_seating_subject(subject)
        and _is_work_surface_target(target)
        and _is_actionable_seating_surface_pair(subject, target)
    ):
        return "seating_to_work_surface"
    if _is_seating_subject(subject) and _is_media_target(target):
        return "seating_to_media"
    if (
        sc in BEDS
        or (
            subject_profile.source == "explicit" and subject_profile.is_sleeping_surface
        )
    ) and _is_nightstand_target(target):
        return "bed_to_nightstand"
    if _is_supported_small_subject(subject) and _is_primary_support_target(target):
        return "object_on_support"
    if _is_lamp_subject(subject) and _is_lamp_surface_target(target):
        return "lamp_to_surface"
    return None


def _normalize_relation_type(relation_type: str) -> str:
    return {
        "media_viewing": "seating_to_media",
        "bedside": "bedside_pair",
        "bed_to_nightstands": "bedside_pair",
        "near": "generic_near_relation",
        "generic_relation": "generic_near_relation",
    }.get(relation_type, relation_type)


def _eval_relation_over_targets(
    store: GeometryStore,
    subject: dict[str, Any],
    targets: list[dict[str, Any]],
    relation_type: str,
) -> tuple[str, float, str, dict[str, Any]]:
    if relation_type == "dining_set":
        return _eval_dining_set(subject, targets)
    if relation_type == "workstation":
        return _eval_workstation(subject, targets)
    if relation_type == "bedside_pair":
        return _eval_bedside_pair(subject, targets)

    scored: list[dict[str, Any]] = []
    for target in targets:
        target_relation = relation_type
        eval_subject = subject
        eval_target = target
        direction_note = ""
        if relation_type == "generic_near_relation":
            inferred = _infer_relation_type(subject, target)
            if inferred and _relation_target_is_valid(subject, target, inferred):
                target_relation = inferred
            elif _relation_target_is_valid(target, subject, "object_on_support"):
                target_relation = "object_on_support"
                eval_subject = target
                eval_target = subject
                direction_note = f"interpreted reversed support direction: `{target.get('id')}` is supported by `{subject.get('id')}`; "
        elif not _relation_target_is_valid(subject, target, relation_type):
            inferred = _infer_relation_type(subject, target)
            if inferred and _relation_target_is_valid(subject, target, inferred):
                target_relation = inferred
            elif _relation_target_is_valid(target, subject, "object_on_support"):
                target_relation = "object_on_support"
                eval_subject = target
                eval_target = subject
                direction_note = f"interpreted reversed support direction: `{target.get('id')}` is supported by `{subject.get('id')}`; "
            else:
                scored.append(
                    _target_eval_payload(
                        target,
                        "fail",
                        0.3,
                        "target category is not compatible with relation.",
                        relation_type,
                    )
                )
                continue
        evaluator = {
            "seating_to_work_surface": _eval_seating_to_surface,
            "seating_to_media": _eval_facing_relation,
            "bed_to_nightstand": _eval_bed_to_nightstand,
            "object_on_support": _eval_object_on_support,
            "lamp_to_surface": _eval_object_on_support,
        }.get(target_relation, _eval_generic_near_relation)
        if evaluator is _eval_object_on_support:
            support_result = evaluate_support_relation(
                eval_subject, eval_target, target_relation, store=store
            )
            scored.append(
                _target_eval_payload(
                    target,
                    support_result.label,
                    support_result.confidence,
                    direction_note + support_result.reason,
                    target_relation,
                    evidence=support_result.evidence,
                )
            )
        else:
            label, confidence, reason = evaluator(subject, target, target_relation)
            scored.append(
                _target_eval_payload(target, label, confidence, reason, target_relation)
            )
    if not scored:
        return "unknown", 0.0, "no target could be evaluated.", _empty_fd_diagnostics()

    best = max(
        scored,
        key=lambda item: (
            _relation_label_rank(relation_type, item["label"]),
            item["confidence"],
        ),
    )
    rescue = _maybe_rescue_support_target(
        store, subject, targets, relation_type, scored, best
    )
    if rescue is not None:
        return rescue
    diagnostics = _fd_diagnostics_from_targets(scored, selected=[best["target_id"]])
    return (
        best["label"],
        best["confidence"],
        f"selected `{best['target_id']}`; {best['reason']}",
        diagnostics,
    )


def _maybe_rescue_support_target(
    store: GeometryStore,
    subject: dict[str, Any],
    declared_targets: list[dict[str, Any]],
    relation_type: str,
    scored: list[dict[str, Any]],
    best: dict[str, Any],
) -> tuple[str, float, str, dict[str, Any]] | None:
    if relation_type not in {"object_on_support", "lamp_to_surface"}:
        return None
    if any(item.get("label") == "pass" for item in scored):
        return None
    if best.get("label") == "degraded" and float(best.get("confidence") or 0.0) >= 0.78:
        return None

    subject_id = str(subject.get("id") or "")
    declared_ids = {
        str(target.get("id") or "") for target in declared_targets if target.get("id")
    }
    candidates: list[dict[str, Any]] = []
    for candidate in store.objects.values():
        candidate_id = str(candidate.get("id") or "")
        if (
            not candidate_id
            or candidate_id == subject_id
            or candidate_id in declared_ids
        ):
            continue
        if not _is_support_rescue_candidate(subject, candidate, relation_type):
            continue
        support_result = evaluate_support_relation(
            subject, candidate, relation_type, store=store
        )
        if support_result.label != "pass" or support_result.confidence < 0.80:
            continue
        if not _rescue_is_clear_improvement(best, support_result, relation_type):
            continue
        candidates.append(
            {
                "candidate": candidate,
                "support_result": support_result,
                "rank": _support_rescue_rank(subject, candidate, support_result),
            }
        )

    if not candidates:
        return None
    selected = min(candidates, key=lambda item: item["rank"])
    candidate = selected["candidate"]
    support_result = selected["support_result"]
    candidate_id = str(candidate.get("id") or "")
    original_ids = [
        str(target.get("id") or "") for target in declared_targets if target.get("id")
    ]
    evidence = dict(support_result.evidence)
    evidence.update(
        {
            "support_evaluation_path": "target_rescue",
            "rescue_support_evaluation_path": support_result.evaluation_path,
            "rescue_from_target_ids": original_ids,
            "rescue_selected_target_id": candidate_id,
            "rescue_original_best_label": best.get("label"),
            "rescue_original_best_reason": best.get("reason"),
        }
    )
    rescue_confidence = min(float(support_result.confidence), 0.84)
    rescue_reason = (
        f"target rescue selected `{candidate_id}` after declared target `{best.get('target_id')}` "
        f"scored {best.get('label')}; candidate support: {support_result.reason}"
    )
    rescue_payload = _target_eval_payload(
        candidate,
        "pass",
        rescue_confidence,
        rescue_reason,
        relation_type,
        evidence=evidence,
    )
    rescue_scored = scored + [rescue_payload]
    diagnostics = _fd_diagnostics_from_targets(rescue_scored, selected=[candidate_id])
    diagnostics.update(
        {
            "support_evaluation_path": "target_rescue",
            "rescue_from_target_ids": original_ids,
            "rescue_selected_target_id": candidate_id,
            "rescue_original_best_label": best.get("label"),
            "rescue_original_best_reason": best.get("reason"),
        }
    )
    return (
        "pass",
        rescue_confidence,
        f"selected `{candidate_id}` via target rescue; original target `{best.get('target_id')}` was {best.get('label')}; {support_result.reason}",
        diagnostics,
    )


def _is_support_rescue_candidate(
    subject: dict[str, Any],
    candidate: dict[str, Any],
    relation_type: str,
) -> bool:
    if relation_type == "object_on_support":
        return _is_primary_support_target(candidate)
    if relation_type != "lamp_to_surface":
        return False
    if _lamp_rescue_rejects_target(candidate):
        return False
    if _is_lamp_surface_target(candidate):
        return True
    return _is_rigid_lamp_rescue_platform(candidate)


def _lamp_rescue_rejects_target(candidate: dict[str, Any]) -> bool:
    category = object_category(candidate).lower()
    group = _category_group(candidate)
    if category in BEDS or group == "sleeping":
        return True
    if category in {
        "chair",
        "office_chair",
        "dining_chair",
        "armchair",
        "sofa",
        "loveseat",
    }:
        return True
    if _category_token_has_any(candidate, SOFT_SUPPORT_TARGET_REJECT_HINTS):
        return True
    if _category_token_has_any(
        candidate,
        (
            "art",
            "artwork",
            "painting",
            "picture",
            "poster",
            "mirror",
            "wall_art",
            "wall_mirror",
        ),
    ):
        return True
    if group in {"decor", "lighting"}:
        return True
    return _is_any_lamp_object(candidate)


def _is_rigid_lamp_rescue_platform(candidate: dict[str, Any]) -> bool:
    category = object_category(candidate).lower()
    if category not in {"bench", "stool", "ottoman"} and not _candidate_text_has_any(
        candidate, ("bench", "stool", "ottoman")
    ):
        return False
    bbox = candidate.get("bbox_world") or {}
    size = bbox.get("size") or []
    tmax = bbox.get("max") or []
    if len(size) < 3 or len(tmax) < 3:
        return False
    x_size = abs(float(size[0]))
    y_size = abs(float(size[1]))
    z_size = abs(float(size[2]))
    if max(x_size, y_size) < 0.28 or min(x_size, y_size) < 0.20:
        return False
    return z_size <= 0.75 and float(tmax[2]) <= 0.95


def _candidate_text_has_any(candidate: dict[str, Any], terms: tuple[str, ...]) -> bool:
    hints = candidate.get("functional_hints") or {}
    parts = [
        str(candidate.get("id") or ""),
        str(candidate.get("category") or ""),
        str(candidate.get("category_norm") or ""),
        str(hints.get("category_group") or ""),
    ]
    parts.extend(str(item or "") for item in (hints.get("category_keywords") or []))
    text = " ".join(parts).lower()
    return any(term in text for term in terms)


def _rescue_is_clear_improvement(
    best: dict[str, Any],
    support_result: Any,
    relation_type: str,
) -> bool:
    best_label = str(best.get("label") or "unknown")
    if _relation_label_rank(
        relation_type, support_result.label
    ) <= _relation_label_rank(relation_type, best_label):
        return False
    if best_label == "degraded" and float(best.get("confidence") or 0.0) >= 0.78:
        return support_result.confidence >= float(best.get("confidence") or 0.0) + 0.08
    return True


def _support_rescue_rank(
    subject: dict[str, Any],
    candidate: dict[str, Any],
    support_result: Any,
) -> tuple[float, float, float]:
    gap = bbox_gap_xy(subject, candidate)
    gap_rank = gap if gap is not None else 999.0
    overlap_rank = -_float_evidence(
        support_result.evidence, "support_overlap_ratio", 0.0
    )
    height_rank = _float_evidence(
        support_result.evidence, "support_height_delta_m", 999.0
    )
    return gap_rank, height_rank, overlap_rank


def _float_evidence(evidence: dict[str, Any], key: str, default: float) -> float:
    value = evidence.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _eval_dining_set(
    subject: dict[str, Any], targets: list[dict[str, Any]]
) -> tuple[str, float, str, dict[str, Any]]:
    table = (
        subject
        if object_category(subject) in DINING_TABLES
        else next(
            (target for target in targets if object_category(target) in DINING_TABLES),
            None,
        )
    )
    chairs = [obj for obj in ([subject] + targets) if _is_seating_subject(obj)]
    if table is None or not chairs:
        return (
            "fail",
            0.82,
            "missing dining table or seating targets.",
            _empty_fd_diagnostics(),
        )
    valid: list[dict[str, Any]] = []
    for chair in chairs:
        label, confidence, reason = _eval_seating_to_surface(
            chair, table, "seating_to_work_surface"
        )
        valid.append(
            _target_eval_payload(chair, label, confidence, reason, "dining_set")
        )
    pass_count = sum(1 for item in valid if item["label"] == "pass")
    degraded_count = sum(1 for item in valid if item["label"] in {"pass", "degraded"})
    if pass_count >= 2:
        label, confidence = "pass", 0.9
    elif degraded_count >= 1:
        label, confidence = "degraded", 0.74
    else:
        label, confidence = "fail", 0.84
    diagnostics = _fd_diagnostics_from_targets(
        valid,
        selected=[
            item["target_id"] for item in valid if item["label"] in {"pass", "degraded"}
        ],
    )
    diagnostics["cardinality_score"] = min(len(chairs) / 2.0, 1.0)
    return (
        label,
        confidence,
        f"{pass_count} seats strongly pair with dining table `{table.get('id')}`.",
        diagnostics,
    )


def _eval_workstation(
    subject: dict[str, Any], targets: list[dict[str, Any]]
) -> tuple[str, float, str, dict[str, Any]]:
    surface_candidates = [
        target for target in targets if _is_work_surface_target(target)
    ]
    if _is_work_surface_target(subject):
        surface_candidates.insert(0, subject)
    seat_candidates = [obj for obj in ([subject] + targets) if _is_seating_subject(obj)]
    if not surface_candidates or not seat_candidates:
        return "fail", 0.82, "missing work surface or seat.", _empty_fd_diagnostics()
    scored: list[dict[str, Any]] = []
    for seat in seat_candidates:
        for surface in surface_candidates:
            label, confidence, reason = _eval_seating_to_surface(
                seat, surface, "seating_to_work_surface"
            )
            payload = _target_eval_payload(
                surface, label, confidence, reason, "workstation"
            )
            payload["seat_id"] = seat.get("id")
            scored.append(payload)
    best = max(
        scored, key=lambda item: (_fd_label_rank(item["label"]), item["confidence"])
    )
    diagnostics = _fd_diagnostics_from_targets(scored, selected=[best["target_id"]])
    return (
        best["label"],
        best["confidence"],
        f"selected seat `{best.get('seat_id')}` and work surface `{best['target_id']}`; {best['reason']}",
        diagnostics,
    )


def _eval_bedside_pair(
    subject: dict[str, Any], targets: list[dict[str, Any]]
) -> tuple[str, float, str, dict[str, Any]]:
    bed = subject if object_category(subject) in BEDS else None
    if bed is None:
        return "fail", 0.82, "subject is not a bed.", _empty_fd_diagnostics()
    nightstands = [target for target in targets if _is_nightstand_target(target)]
    if not nightstands:
        return "fail", 0.82, "no nightstand target found.", _empty_fd_diagnostics()
    scored = []
    for target in nightstands:
        label, confidence, reason = _eval_bed_to_nightstand(
            bed, target, "bed_to_nightstand"
        )
        scored.append(
            _target_eval_payload(target, label, confidence, reason, "bedside_pair")
        )
    best = max(
        scored, key=lambda item: (_fd_label_rank(item["label"]), item["confidence"])
    )
    diagnostics = _fd_diagnostics_from_targets(scored, selected=[best["target_id"]])
    diagnostics["cardinality_score"] = min(len(nightstands) / 2.0, 1.0)
    return (
        best["label"],
        best["confidence"],
        f"selected bedside target `{best['target_id']}`; {best['reason']}",
        diagnostics,
    )


def _eval_seating_to_surface(
    subject: dict[str, Any], target: dict[str, Any], _relation_type: str
) -> tuple[str, float, str]:
    gap = bbox_gap_xy(subject, target)
    angle, angle_mode = seating_angle_to_target_deg(subject, target)
    if gap is None or angle is None:
        return "unknown", 0.0, "missing distance or orientation geometry."
    angle_note = ""
    if angle_mode == "depth_axis_fallback":
        angle_note = " using seating depth-axis fallback"
    elif angle_mode == "nearest_surface":
        angle_note = " using target-edge fallback"
    elif angle_mode == "nearest_long_surface":
        angle_note = " using nearest long table-edge fallback"
    elif angle_mode == "front_ray_surface":
        angle_note = " using front-facing table-edge fallback"
    elif angle_mode == "reversed_front_fallback":
        angle_note = " using flipped-front fallback"
    if not _is_actionable_seating_surface_pair(subject, target):
        return (
            "unknown",
            0.0,
            "chair can stand alone here; no nearby usable table or counter relation is required "
            f"(gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}).",
        )
    living_room_pair = (
        object_category(subject) in LIVING_ROOM_SEATING
        and object_category(target) == "coffee_table"
    )
    if _is_side_surface_target(target):
        if gap <= 0.55:
            return (
                "pass",
                0.88,
                f"seat is adjacent to a side surface with gap {gap:.2f}m.",
            )
        if gap <= 0.95:
            return (
                "degraded",
                0.72,
                f"side surface is usable but loose: gap {gap:.2f}m.",
            )
        return "fail", 0.8, f"side surface is too far from the seat: gap {gap:.2f}m."
    if gap <= 0.35 and angle <= 110.0:
        return (
            "pass",
            0.9,
            f"close seating pair with gap {gap:.2f}m and facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.35 and angle <= 150.0:
        return (
            "pass",
            0.86,
            f"seat is tight to the surface despite noisy yaw: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.45 and angle <= 140.0:
        return (
            "pass",
            0.84,
            f"seat remains close enough for paired use: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.8 and angle <= 100.0:
        return (
            "pass",
            0.88,
            f"paired seating relation is close and plausibly oriented: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.3 and angle <= 75.0:
        return (
            "pass",
            0.86,
            f"seat is moderately spaced but well oriented: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if living_room_pair and gap <= 1.35 and angle <= 110.0:
        return (
            "pass",
            0.88,
            f"living-room seating is well paired with the coffee table: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.2 and angle <= 75.0:
        return (
            "pass",
            0.9,
            f"gap {gap:.2f}m and facing angle {angle:.0f}deg{angle_note} support paired use.",
        )
    if angle > 150.0:
        return (
            "fail",
            0.85,
            f"subject is back-facing relative to the target: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if living_room_pair and gap <= 1.35 and angle <= 135.0:
        return (
            "degraded",
            0.78,
            f"living-room pair is usable but loose: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.0 and angle <= 125.0:
        return (
            "pass",
            0.82,
            f"relation is usable despite moderate spacing or yaw: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 0.45 and angle <= 150.0:
        return (
            "degraded",
            0.78,
            f"close pair is usable but rotated: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    if gap <= 1.8 and angle <= 110.0:
        return (
            "degraded",
            0.75,
            f"relation is weak: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
        )
    return (
        "fail",
        0.85,
        f"target is too far or poorly oriented: gap {gap:.2f}m, facing angle {angle:.0f}deg{angle_note}.",
    )


def _eval_facing_relation(
    subject: dict[str, Any], target: dict[str, Any], _relation_type: str
) -> tuple[str, float, str]:
    dist = distance_xy(subject, target)
    angle = angle_to_target_deg(subject, target)
    if dist is None or angle is None:
        return "unknown", 0.0, "missing distance or orientation geometry."
    living_room_media = object_category(
        subject
    ) in LIVING_ROOM_SEATING and _is_media_target(target)
    if 0.8 <= dist <= 5.5 and angle <= 65.0:
        return (
            "pass",
            0.9,
            f"distance {dist:.2f}m and facing angle {angle:.0f}deg support viewing/use.",
        )
    if living_room_media and 0.8 <= dist <= 5.5 and angle <= 100.0:
        return (
            "pass",
            0.84,
            f"living-room seating has usable media view: distance {dist:.2f}m, angle {angle:.0f}deg.",
        )
    if living_room_media and dist <= 6.5 and angle <= 125.0:
        return (
            "degraded",
            0.72,
            f"media relation is usable but oblique: distance {dist:.2f}m, angle {angle:.0f}deg.",
        )
    if dist <= 6.5 and angle <= 100.0:
        return (
            "degraded",
            0.75,
            f"relation is usable but weak: distance {dist:.2f}m, angle {angle:.0f}deg.",
        )
    return (
        "fail",
        0.85,
        f"relation does not support use: distance {dist:.2f}m, angle {angle:.0f}deg.",
    )


def _eval_bed_to_nightstand(
    subject: dict[str, Any], target: dict[str, Any], _relation_type: str
) -> tuple[str, float, str]:
    gap = bbox_gap_xy(subject, target)
    if gap is None:
        return "unknown", 0.0, "missing distance geometry."
    if gap <= 0.45:
        return "pass", 0.9, f"nightstand is adjacent to bed with {gap:.2f}m bbox gap."
    if gap <= 0.9:
        return (
            "degraded",
            0.75,
            f"nightstand is nearby but not tight to the bed: {gap:.2f}m gap.",
        )
    return "fail", 0.85, f"nightstand is too far from the bed: {gap:.2f}m gap."


def _eval_generic_near_relation(
    subject: dict[str, Any], target: dict[str, Any], relation_type: str
) -> tuple[str, float, str]:
    gap = bbox_gap_xy(subject, target)
    if gap is None:
        return "unknown", 0.0, "missing distance geometry."
    if gap <= 0.6:
        return (
            "pass",
            0.75,
            f"generic `{relation_type}` target is nearby with {gap:.2f}m gap.",
        )
    if gap <= 1.2:
        return (
            "degraded",
            0.65,
            f"generic `{relation_type}` target is somewhat far with {gap:.2f}m gap.",
        )
    return (
        "fail",
        0.75,
        f"generic `{relation_type}` target is too far with {gap:.2f}m gap.",
    )


def _preferred_relations_for_subject(subject: dict[str, Any]) -> list[str]:
    profile = object_function_profile(subject)
    if profile.source == "explicit" and profile.is_seating:
        subject_category = object_category(subject)
        if subject_category in LIVING_ROOM_SEATING:
            return ["seating_to_media", "seating_to_work_surface"]
        return ["seating_to_work_surface", "seating_to_media"]
    if profile.source == "explicit" and profile.is_sleeping_surface:
        return ["bed_to_nightstand"]
    if (
        profile.source == "explicit"
        and profile.is_small_placeable
        and not (
            profile.can_support_top
            or profile.has_internal_shelf
            or profile.is_work_surface
            or profile.is_media_target
        )
    ):
        return ["object_on_support"]
    subject_category = object_category(subject)
    if subject_category in SEATING and not _is_seating_subject(subject):
        return []
    if subject_category in LIVING_ROOM_SEATING:
        return ["seating_to_media", "seating_to_work_surface"]
    if subject_category in SEATING:
        return ["seating_to_work_surface", "seating_to_media"]
    if subject_category in BEDS:
        return ["bed_to_nightstand"]
    if _is_lamp_subject(subject):
        return ["lamp_to_surface"]
    if _is_supported_small_subject(subject):
        return ["object_on_support"]
    return []


def _best_template_target(
    subject: dict[str, Any],
    relation_type: str,
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.proposer import (
        _rank_targets_for_relation,
    )

    target_by_id = {
        str(target.get("id") or ""): target for target in objects if target.get("id")
    }
    ranked_ids = _rank_targets_for_relation(subject, relation_type, objects)
    if not ranked_ids:
        return None
    return target_by_id[ranked_ids[0]]


def _angle_penalty(
    subject: dict[str, Any], target: dict[str, Any], relation_type: str
) -> float:
    if relation_type not in {"seating_to_work_surface", "seating_to_media"}:
        return 0.0
    angle = angle_to_target_deg(subject, target)
    if angle is None:
        return 0.0
    return max(angle - 60.0, 0.0) / 60.0


def _relation_target_is_valid(
    subject: dict[str, Any], target: dict[str, Any], relation_type: str
) -> bool:
    relation_type = _normalize_relation_type(relation_type)
    if relation_type == "seating_to_work_surface":
        return _is_seating_subject(subject) and _is_work_surface_target(target)
    if relation_type == "seating_to_media":
        return _is_seating_subject(subject) and _is_media_target(target)
    if relation_type == "bed_to_nightstand":
        profile = object_function_profile(subject)
        return (
            object_category(subject) in BEDS
            or (profile.source == "explicit" and profile.is_sleeping_surface)
        ) and _is_nightstand_target(target)
    if relation_type == "object_on_support":
        return _is_supported_small_subject(subject) and _is_primary_support_target(
            target
        )
    if relation_type == "lamp_to_surface":
        return _is_lamp_subject(subject) and _is_lamp_surface_target(target)
    if relation_type == "dining_set":
        return (
            object_category(subject) in DINING_TABLES and _is_seating_subject(target)
        ) or (_is_seating_subject(subject) and object_category(target) in DINING_TABLES)
    if relation_type == "workstation":
        return (
            _is_work_surface_target(subject)
            or _is_work_surface_target(target)
            or _is_seating_subject(target)
        )
    if relation_type == "bedside_pair":
        profile = object_function_profile(subject)
        return (
            object_category(subject) in BEDS
            or (profile.source == "explicit" and profile.is_sleeping_surface)
        ) and _is_nightstand_target(target)
    if relation_type == "generic_near_relation":
        return True
    return False
