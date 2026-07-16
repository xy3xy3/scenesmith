"""Tests for wall-display visibility against nearby furniture silhouettes."""

from __future__ import annotations

from scenesmith.scenebenchmark_critic.prompt_context import format_agent_prompt_context
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.furniture_occlusion import (
    evaluate_wall_mounted_visibility,
)


def _obj(
    object_id: str,
    category: str,
    object_type: str,
    *,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    wall: str | None = None,
) -> dict:
    center = [(minimum[index] + maximum[index]) / 2.0 for index in range(3)]
    size = [maximum[index] - minimum[index] for index in range(3)]
    result = {
        "id": object_id,
        "name": category,
        "description": category.replace("_", " "),
        "category": category,
        "category_norm": category,
        "object_type": object_type,
        "room": "room",
        "bbox_world": {
            "min": list(minimum),
            "max": list(maximum),
            "center": center,
            "size": size,
        },
        "functional_hints": {"scene_object_type": object_type},
    }
    if wall:
        result["placement_info"] = {
            "parent_surface_id": f"room_{wall}",
            "placement_method": "wall_placement",
        }
    return result


def _case(objects: list[dict]) -> dict:
    return {
        "scene_geometry": {
            "rooms": [
                {
                    "id": "room",
                    "bbox": {"min": [-2.5, -2.2, 0.0], "max": [2.5, 2.2, 2.7]},
                }
            ],
            "objects": objects,
        }
    }


def _result(objects: list[dict], object_id: str) -> dict:
    results = evaluate_wall_mounted_visibility(_case(objects))
    return next(result for result in results if result["primary_object"] == object_id)


def test_batch_003_painting_occluded_by_wardrobe_fails() -> None:
    # 2026-07-15 修改原因：复现 batch_003，north-wall 画作约 74% 的正面投影
    # 被贴墙衣柜覆盖；仅检查物理碰撞无法发现这种视觉遮挡。
    painting = _obj(
        "painting_0",
        "painting_painting_framed",
        "wall_mounted",
        minimum=(0.995, 2.170, 1.254),
        maximum=(1.703, 2.200, 1.962),
        wall="north",
    )
    wardrobe = _obj(
        "wardrobe_0",
        "wardrobe",
        "furniture",
        minimum=(1.183, 1.273, 0.0),
        maximum=(2.383, 2.045, 2.483),
    )

    result = _result([painting, wardrobe], "painting_0")

    assert result["label"] == "fail"
    assert result["metric"] == "visual_clearance"
    assert result["relation_type"] == "wall_mounted_visibility"
    assert result["blocking_objects"] == ["wardrobe_0"]
    assert result["diagnostics"]["occluded_fraction"] > 0.70
    assert "move `painting_0`" in result["repair_advice"]
    assert "do not move the wardrobe" in result["repair_advice"]


def test_art_above_dresser_with_vertical_gap_passes() -> None:
    painting = _obj(
        "painting_0",
        "wall_art",
        "wall_mounted",
        minimum=(-0.5, 2.17, 1.35),
        maximum=(0.5, 2.2, 2.05),
        wall="north",
    )
    dresser = _obj(
        "dresser_0",
        "dresser",
        "furniture",
        minimum=(-0.6, 1.65, 0.0),
        maximum=(0.6, 2.05, 0.85),
    )

    result = _result([painting, dresser], "painting_0")

    assert result["label"] == "pass"
    assert result["blocking_objects"] == []
    assert result["diagnostics"]["occluded_fraction"] == 0.0


def test_low_bed_does_not_occlude_high_south_wall_mirror() -> None:
    mirror = _obj(
        "mirror_0",
        "mirror_mirror_round",
        "wall_mounted",
        minimum=(0.35, -2.2, 1.2),
        maximum=(0.95, -2.18, 2.0),
        wall="south",
    )
    bed = _obj(
        "bed_0",
        "bed",
        "furniture",
        minimum=(-0.8, -2.13, 0.0),
        maximum=(0.8, -0.27, 0.73),
    )

    assert _result([mirror, bed], "mirror_0")["label"] == "pass"


def test_partial_wall_clock_occlusion_is_degraded() -> None:
    clock = _obj(
        "wall_clock_0",
        "clock",
        "wall_mounted",
        minimum=(2.37, -0.2, 1.6),
        maximum=(2.45, 0.2, 2.0),
        wall="east",
    )
    clock["description"] = "Minimalist wall clock with a readable display"
    cabinet = _obj(
        "cabinet_0",
        "cabinet",
        "furniture",
        minimum=(1.8, 0.13, 0.0),
        maximum=(2.3, 0.5, 2.1),
    )

    result = _result([clock, cabinet], "wall_clock_0")

    assert result["label"] == "degraded"
    assert 0.15 < result["diagnostics"]["occluded_fraction"] < 0.20


def test_furniture_far_from_wall_is_not_a_wall_display_blocker() -> None:
    picture = _obj(
        "picture_0",
        "framed_picture",
        "wall_mounted",
        minimum=(-0.5, 2.17, 1.2),
        maximum=(0.5, 2.2, 2.0),
        wall="north",
    )
    central_partition = _obj(
        "cabinet_0",
        "cabinet",
        "furniture",
        minimum=(-0.5, 0.0, 0.0),
        maximum=(0.5, 0.6, 2.2),
    )

    assert _result([picture, central_partition], "picture_0")["label"] == "pass"


def test_tv_and_floating_shelf_are_outside_decor_visibility_scope() -> None:
    television = _obj(
        "tv_0",
        "television",
        "wall_mounted",
        minimum=(-0.6, 2.17, 1.0),
        maximum=(0.6, 2.2, 1.7),
        wall="north",
    )
    shelf = _obj(
        "shelf_0",
        "shelf",
        "wall_mounted",
        minimum=(-2.45, -0.4, 1.4),
        maximum=(-2.2, 0.4, 1.5),
        wall="west",
    )

    assert evaluate_wall_mounted_visibility(_case([television, shelf])) == []


def test_wall_agent_prompt_receives_visibility_failure() -> None:
    painting = _obj(
        "painting_0",
        "painting",
        "wall_mounted",
        minimum=(1.0, 2.17, 1.25),
        maximum=(1.7, 2.2, 1.95),
        wall="north",
    )
    wardrobe = _obj(
        "wardrobe_0",
        "wardrobe",
        "furniture",
        minimum=(1.2, 1.3, 0.0),
        maximum=(2.4, 2.05, 2.48),
    )
    case_pack = _case([painting, wardrobe])
    payload = {
        "case_pack": case_pack,
        "results": evaluate_wall_mounted_visibility(case_pack),
    }

    context = format_agent_prompt_context(payload, agent_type="wall_mounted")

    assert "wall-mounted display" in context.lower()
    assert "wardrobe_0" in context
    assert "move `painting_0`" in context
