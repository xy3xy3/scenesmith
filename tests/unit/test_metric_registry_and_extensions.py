"""Regression tests for the registry migration and new visual/FD checks."""

from __future__ import annotations

from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.evaluator import run_case_pack_checks
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.workstation_alignment import (
    evaluate_workstation_focal_alignment,
)
from scenesmith.scenebenchmark_critic.metrics.registry import (
    METRIC_REGISTRY,
    get_metric_plugins,
)
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.evaluator import (
    evaluate_visual_clearance,
)


def _object(
    object_id: str,
    category: str,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    scene_type: str = "furniture",
    wall: str | None = None,
    parent_surface: str | None = None,
) -> dict:
    x, y, z = center
    sx, sy, sz = size
    obj = {
        "id": object_id,
        "name": category,
        "category": category,
        "category_norm": category,
        "object_type": scene_type,
        "bbox_world": {
            "min": [x - sx / 2, y - sy / 2, z - sz / 2],
            "max": [x + sx / 2, y + sy / 2, z + sz / 2],
            "center": [x, y, z],
            "size": [sx, sy, sz],
        },
        "functional_hints": {"scene_object_type": scene_type},
    }
    if wall or parent_surface:
        obj["placement_info"] = {
            "parent_surface_id": parent_surface or f"room_{wall}",
        }
    return obj


def _workstation_case(
    *,
    chair_x: float = 0.0,
    chair_yaw: float = 180.0,
    monitor_parent: str = "desk_top",
    monitor_yaw: float = 0.0,
) -> dict:
    desk = _object(
        "desk_0",
        "desk",
        center=(0.0, 0.0, 0.4),
        size=(2.0, 1.0, 0.8),
    )
    desk["support_regions"] = [{"region_id": "desk_top"}]
    chair = _object(
        "office_chair_0",
        "office_chair",
        center=(chair_x, 1.2, 0.45),
        size=(0.6, 0.6, 0.9),
    )
    chair["yaw_deg"] = chair_yaw
    monitor = _object(
        "monitor_0",
        "monitor",
        center=(0.0, 0.1, 1.0),
        size=(0.6, 0.1, 0.4),
        scene_type="manipuland",
        parent_surface=monitor_parent,
    )
    monitor["yaw_deg"] = monitor_yaw
    return {
        "scene_geometry": {"objects": [desk, chair, monitor]},
        "checks": [
            {
                "check_id": "fd_workstation",
                "metric": "functional_dependency",
                "subject_id": "office_chair_0",
                "target_ids": ["desk_0"],
                "relation_type": "seating_to_work_surface",
            }
        ],
    }


def test_registry_has_four_plugins_and_rejects_unknown_metrics() -> None:
    assert set(METRIC_REGISTRY) == {
        "functional_dependency",
        "spatial_accessibility",
        "interaction_clearance",
        "visual_clearance",
    }
    assert len(get_metric_plugins(CriticConfig().metrics)) == 4
    try:
        get_metric_plugins(["not_a_metric"])
    except ValueError as exc:
        assert "Unknown" in str(exc)
    else:
        raise AssertionError("unknown metric must fail configuration")


def test_workstation_focal_alignment_passes_and_prefers_lateral_failure() -> None:
    result = evaluate_workstation_focal_alignment(_workstation_case())[0]
    assert result["label"] == "pass"
    assert result["metric"] == "functional_dependency"

    lateral_failure = evaluate_workstation_focal_alignment(
        _workstation_case(chair_x=1.0)
    )[0]
    assert lateral_failure["label"] == "fail"
    assert lateral_failure["diagnostics"]["priority"] == "lateral_alignment"


def test_workstation_extension_requires_focus_on_the_same_desk() -> None:
    assert evaluate_workstation_focal_alignment(
        _workstation_case(monitor_parent="another_desk_top")
    ) == []


def test_workstation_extension_checks_display_front_toward_user() -> None:
    facing = evaluate_workstation_focal_alignment(
        _workstation_case(monitor_yaw=0.0)
    )
    display_result = next(
        item for item in facing if item["relation_type"] == "display_faces_user"
    )
    assert display_result["label"] == "pass"

    reversed_display = evaluate_workstation_focal_alignment(
        _workstation_case(monitor_yaw=180.0)
    )
    display_result = next(
        item
        for item in reversed_display
        if item["relation_type"] == "display_faces_user"
    )
    assert display_result["label"] == "fail"
    assert display_result["diagnostics"]["priority"] == "orientation"


def test_registry_executes_display_orientation_extension() -> None:
    results = run_case_pack_checks(
        _workstation_case(),
        CriticConfig(enabled=True, metrics=("functional_dependency",)),
    )

    display_results = [
        result for result in results if result.get("relation_type") == "display_faces_user"
    ]
    assert len(display_results) == 1
    assert display_results[0]["label"] == "pass"


def _wall_object(
    object_id: str,
    category: str,
    x0: float,
    x1: float,
    *,
    wall: str = "north",
) -> dict:
    return _object(
        object_id,
        category,
        center=((x0 + x1) / 2, 2.0, 1.5),
        size=(x1 - x0, 0.05, 1.0),
        scene_type="wall_mounted",
        wall=wall,
    )


def _wall_case(objects: list[dict]) -> dict:
    return {
        "scene_geometry": {
            "objects": objects,
            "rooms": [
                {
                    "id": "room",
                    "bbox": {
                        "min": [-2.0, -2.0, 0.0],
                        "max": [2.0, 2.0, 3.0],
                    },
                }
            ],
        }
    }


def test_same_wall_overlap_reports_smaller_primary_and_ratio() -> None:
    result = evaluate_visual_clearance(
        _wall_case(
            [
                _wall_object("clock_0", "clock", 0.0, 1.0),
                _wall_object("mirror_0", "mirror", 0.5, 1.5),
            ]
        )
    )
    overlap = next(item for item in result if item["relation_type"] == "wall_mounted_overlap")
    assert overlap["metric"] == "visual_clearance"
    assert overlap["label"] == "fail"
    assert overlap["primary_object"] == "clock_0"
    assert overlap["diagnostics"]["overlap_ratio"] == 0.5


def test_same_wall_edges_and_different_walls_do_not_report_overlap() -> None:
    touching = _wall_case(
        [
            _wall_object("clock_0", "clock", 0.0, 1.0),
            _wall_object("mirror_0", "mirror", 1.0, 2.0),
        ]
    )
    assert not [
        item for item in evaluate_visual_clearance(touching)
        if item["relation_type"] == "wall_mounted_overlap"
    ]
    different_walls = _wall_case(
        [
            _wall_object("clock_0", "clock", 0.0, 1.0, wall="north"),
            _wall_object("mirror_0", "mirror", 0.0, 1.0, wall="south"),
        ]
    )
    assert not [
        item
        for item in evaluate_visual_clearance(different_walls)
        if item["relation_type"] == "wall_mounted_overlap"
    ]


def test_visual_extensions_and_furniture_occlusion_keep_separate_results() -> None:
    painting = _wall_object("painting_0", "painting", 0.0, 1.0)
    wardrobe = _object(
        "wardrobe_0",
        "wardrobe",
        center=(0.5, 1.6, 1.2),
        size=(1.0, 0.8, 2.4),
    )
    results = evaluate_visual_clearance(_wall_case([painting, wardrobe]))
    assert any(item["relation_type"] == "wall_mounted_visibility" for item in results)
    assert all(item["metric"] == "visual_clearance" for item in results)


def test_registry_evaluator_reports_visual_metric_without_manual_api_branch() -> None:
    case_pack = _wall_case(
        [
            _wall_object("clock_0", "clock", 0.0, 1.0),
            _wall_object("mirror_0", "mirror", 0.5, 1.5),
        ]
    )
    results = run_case_pack_checks(
        case_pack,
        CriticConfig(enabled=True, metrics=("visual_clearance",)),
    )
    assert results
    assert {result["metric"] for result in results} == {"visual_clearance"}
