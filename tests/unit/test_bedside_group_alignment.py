"""Tests for bed-local bedside-group topology and door coordination."""

from __future__ import annotations

import math

from scenesmith.scenebenchmark_critic.bedside_group_alignment import (
    evaluate_bedside_group_alignment,
)
from scenesmith.scenebenchmark_critic.prompt_context import format_agent_prompt_context


def _obj(
    object_id: str,
    category: str,
    x: float,
    y: float,
    side_size: float,
    front_size: float,
    *,
    yaw: float = 0.0,
) -> dict:
    angle = math.radians(yaw)
    front = (-math.sin(angle), math.cos(angle))
    side = (-front[1], front[0])
    corners = [
        [
            x
            + side_sign * side_size / 2.0 * side[0]
            + front_sign * front_size / 2.0 * front[0],
            y
            + side_sign * side_size / 2.0 * side[1]
            + front_sign * front_size / 2.0 * front[1],
        ]
        for side_sign, front_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "object_type": "furniture",
        "room": "room",
        "yaw_deg": yaw,
        "footprint_world": corners,
        "bbox_world": {
            "center": [x, y, 0.5],
            "size": [max(xs) - min(xs), max(ys) - min(ys), 1.0],
            "min": [min(xs), min(ys), 0.0],
            "max": [max(xs), max(ys), 1.0],
        },
        "functional_hints": {"scene_object_type": "furniture"},
    }


def _case(objects: list[dict], *, doors: list[dict] | None = None) -> dict:
    return {
        "scene_geometry": {
            "rooms": [
                {
                    "id": "room",
                    "bbox": {"min": [-2.5, -2.25, 0.0], "max": [2.5, 2.25, 3.0]},
                }
            ],
            "objects": objects,
            "scene_shell": {"doors": doors or [], "windows": []},
        }
    }


def _result(objects: list[dict], *, doors: list[dict] | None = None) -> dict:
    results = evaluate_bedside_group_alignment(_case(objects, doors=doors))
    assert len(results) == 1
    return results[0]


def _standard_group(*, yaw: float = 0.0) -> list[dict]:
    angle = math.radians(yaw)
    front = (-math.sin(angle), math.cos(angle))
    side = (-front[1], front[0])
    bed = _obj("bed_0", "bed", 0.0, 0.0, 1.5, 2.0, yaw=yaw)
    head_center = (-front[0], -front[1])
    stands = [
        _obj(
            f"nightstand_{index}",
            "nightstand",
            head_center[0] + sign * 1.05 * side[0],
            head_center[1] + sign * 1.05 * side[1],
            0.5,
            0.5,
            yaw=yaw,
        )
        for index, sign in enumerate((-1, 1))
    ]
    return [bed, *stands]


def test_correct_head_end_opposite_sides_passes() -> None:
    floor = _obj("floor_bedroom", "floor", 0.0, 0.0, 5.0, 4.5)
    result = _result([floor, *_standard_group()])

    assert result["label"] == "pass"
    assert result["diagnostics"]["opposite_sides"] is True
    assert all(
        item["at_head_end"] for item in result["diagnostics"]["nightstand_slots"]
    )
    assert result["primary_object"] == "bed_0"


def test_foot_end_tables_fail_even_when_adjacent_and_parallel() -> None:
    bed = _obj("bed_0", "bed", 0.0, 0.0, 1.5, 2.0)
    stands = [
        _obj(f"nightstand_{index}", "nightstand", sign * 1.05, 1.0, 0.5, 0.5)
        for index, sign in enumerate((-1, 1))
    ]

    result = _result([bed, *stands])

    assert result["label"] == "fail"
    assert all(
        not item["at_head_end"] for item in result["diagnostics"]["nightstand_slots"]
    )
    assert "not at the bed head end" in result["reason"]


def test_two_tables_on_same_side_fail() -> None:
    bed = _obj("bed_0", "bed", 0.0, 0.0, 1.5, 2.0)
    stands = [
        _obj("nightstand_0", "nightstand", -1.02, -0.9, 0.5, 0.5),
        _obj("nightstand_1", "nightstand", -1.08, -1.1, 0.5, 0.5),
    ]

    result = _result([bed, *stands])

    assert result["label"] == "fail"
    assert result["diagnostics"]["opposite_sides"] is False
    assert "same side" in result["reason"]


def test_single_nightstand_does_not_require_a_pair() -> None:
    result = _result(_standard_group()[:2])

    assert result["label"] == "pass"
    assert result["diagnostics"]["opposite_sides"] is True


def test_rotated_group_uses_bed_local_coordinates() -> None:
    result = _result(_standard_group(yaw=37.0))

    assert result["label"] == "pass"


def test_door_conflict_requires_whole_group_wall_change() -> None:
    bed = _obj("bed_0", "bed", -1.3, 0.0, 1.5, 2.0, yaw=-90.0)
    stands = [
        _obj("nightstand_0", "nightstand", -2.3, -1.05, 0.5, 0.5, yaw=-90.0),
        _obj("nightstand_1", "nightstand", -2.3, 1.05, 0.5, 0.5, yaw=-90.0),
    ]
    door = {
        "id": "door_west",
        "wall_direction": "west",
        "bbox": {"min": [-2.5, -0.9, 0.0], "max": [-1.7, 0.0, 2.1]},
    }

    result = _result([bed, *stands], doors=[door])

    assert result["label"] == "fail"
    assert result["diagnostics"]["headboard_wall"] == "west"
    assert result["diagnostics"]["actual_door_conflicts"] == ["door_west"]
    assert "one coordinated group" in result["repair_advice"]
    assert "independently" in result["repair_advice"]


def test_foot_tables_expose_door_conflict_in_reconstructed_head_slots() -> None:
    # 2026-07-15 修改原因：复现 batch_003。床沿 west wall 上移后门净空已不再
    # 与当前柜子相交，但把柜子恢复到正确床头槽位会再次撞门，因此必须整组换墙。
    bed = _obj("bed_0", "bed", -1.3, 1.2, 1.5, 2.0, yaw=-90.0)
    stands = [
        _obj("nightstand_0", "nightstand", 0.0, 0.5, 0.5, 0.5, yaw=-90.0),
        _obj("nightstand_1", "nightstand", 0.0, 1.9, 0.5, 0.5, yaw=-90.0),
    ]
    door = {
        "id": "door_west",
        "wall_direction": "west",
        "bbox": {"min": [-2.5, -0.93, 0.0], "max": [-1.7, -0.03, 2.1]},
    }

    result = _result([bed, *stands], doors=[door])

    assert result["label"] == "fail"
    assert result["diagnostics"]["actual_door_conflicts"] == []
    assert result["diagnostics"]["target_slot_door_conflicts"] == ["door_west"]
    assert "Move `bed_0` and all bedside tables" in result["repair_advice"]


def test_passing_group_is_preserved_in_furniture_prompt_context() -> None:
    case_pack = _case(_standard_group())
    payload = {
        "case_pack": case_pack,
        "results": evaluate_bedside_group_alignment(case_pack),
    }

    context = format_agent_prompt_context(payload, agent_type="furniture")

    assert "Authoritative stable bedside-group contracts" in context
    assert "Do not move a passed bed or nightstand independently" in context
