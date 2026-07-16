"""Tests for generalized dining-seat edge distribution."""

from __future__ import annotations

import math

from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.dining_seat import (
    evaluate_dining_seat_distribution,
)


def _obj(object_id: str, category: str, x: float, y: float, sx: float, sy: float, *, yaw: float = 0.0) -> dict:
    return {
        "id": object_id, "category": category, "category_norm": category,
        "yaw_deg": yaw,
        "bbox_world": {"center": [x, y, 0.5], "size": [sx, sy, 1.0],
                       "min": [x - sx / 2, y - sy / 2, 0.0],
                       "max": [x + sx / 2, y + sy / 2, 1.0]},
        "functional_hints": {"scene_object_type": "furniture"},
    }


def _result(objects: list[dict]) -> dict:
    results = evaluate_dining_seat_distribution(
        {"scene_geometry": {"objects": objects}}
    )
    assert len(results) == 1
    return results[0]


def test_single_chair_on_edge_must_be_centered() -> None:
    table = _obj("table", "dining_table", 0, 0, 2.0, 1.0)
    centered = _obj("chair", "dining_chair", 0, -0.75, 0.45, 0.45)
    offset = _obj("chair", "dining_chair", 0.55, -0.75, 0.45, 0.45)

    assert _result([table, centered])["label"] == "pass"
    assert _result([table, offset])["label"] == "fail"


def test_multiple_chairs_use_symmetric_even_slots() -> None:
    table = _obj("table", "dining_table", 0, 0, 3.0, 1.0)
    left = _obj("left", "dining_chair", -1.0, -0.75, 0.45, 0.45)
    right = _obj("right", "dining_chair", 1.0, -0.75, 0.45, 0.45)
    crowded = _obj("right", "dining_chair", -0.5, -0.75, 0.45, 0.45)

    assert _result([table, left, right])["label"] == "pass"
    assert _result([table, left, crowded])["label"] == "fail"


def test_distribution_uses_rotated_table_frame() -> None:
    yaw = 35.0
    angle = math.radians(yaw)
    tangent = (math.cos(angle), math.sin(angle))
    normal = (-math.sin(angle), math.cos(angle))
    table = _obj("table", "dining_table", 2, 3, 2.0, 1.0, yaw=yaw)
    chair = _obj(
        "chair", "dining_chair", 2 - 0.75 * normal[0],
        3 - 0.75 * normal[1], 0.45, 0.45, yaw=yaw,
    )
    # Keep the unused tangent variable explicit: the chair is at tangent coordinate 0.
    assert tangent[0] != 0
    assert _result([table, chair])["label"] == "pass"


def test_pulled_out_short_edge_chairs_keep_edge_and_must_face_table() -> None:
    # 2026-07-15 修改原因：净空修复把短边椅拉到桌角之外时，不能按无限桌边
    # 将其误归长边并跳过朝向；错误的 180° 翻转必须被直接指出。
    table = _obj("table", "dining_table", 0, 0, 1.87, 0.75)
    west_wrong = _obj(
        "west", "dining_chair", -1.30, -0.02, 0.60, 0.46, yaw=90.0
    )
    east_wrong = _obj(
        "east", "dining_chair", 1.30, -0.02, 0.60, 0.46, yaw=-90.0
    )

    wrong = _result([table, west_wrong, east_wrong])
    diagnostics = {
        item["seat_id"]: item for item in wrong["diagnostics"]["seat_slots"]
    }
    assert wrong["label"] == "fail"
    assert diagnostics["west"]["edge"] == "left"
    assert diagnostics["east"]["edge"] == "right"
    assert diagnostics["west"]["facing_error_deg"] > 170.0
    assert diagnostics["east"]["facing_error_deg"] > 170.0

    west_wrong["yaw_deg"] = -90.0
    east_wrong["yaw_deg"] = 90.0
    corrected = _result([table, west_wrong, east_wrong])
    assert corrected["label"] == "pass"


def test_dining_facing_prefers_exported_interaction_front() -> None:
    # 2026-07-15 修改原因：资产标注的 semantic front 可能与通用 yaw 约定不同；
    # critic 应优先采用 adapter 导出的 front interaction face。
    table = _obj("table", "dining_table", 0, 0, 1.6, 0.8)
    chair = _obj("chair", "dining_chair", -1.1, 0, 0.5, 0.5, yaw=90.0)
    chair["interaction_faces"] = [
        {"name": "front", "normal_xy": [1.0, 0.0], "affordances": ["sittable"]}
    ]
    del chair["yaw_deg"]

    result = _result([table, chair])
    diagnostic = result["diagnostics"]["seat_slots"][0]
    assert result["label"] == "pass"
    assert diagnostic["facing_error_deg"] == 0.0


def test_bench_and_round_table_are_not_forced_into_rectangular_slots() -> None:
    table = _obj("table", "round_dining_table", 0, 0, 1.2, 1.2)
    bench = _obj("bench", "dining_bench", 0, -0.9, 1.0, 0.45)

    assert evaluate_dining_seat_distribution(
        {"scene_geometry": {"objects": [table, bench]}}
    ) == []
