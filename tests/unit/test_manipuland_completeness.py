"""Tests for manipuland completeness critic rules."""

import math

from scenesmith.scenebenchmark_critic.dining_place_setting_alignment import (
    evaluate_dining_place_setting_alignment,
)
from scenesmith.scenebenchmark_critic.manipuland_completeness import (
    evaluate_manipuland_completeness,
)


def test_dining_place_setting_fails_when_required_items_missing() -> None:
    case_pack = _case_pack(
        [
            _table(),
            *[_item(f"dinner_plate_{idx}", "dinner_plate") for idx in range(4)],
            *[_item(f"wine_glass_{idx}", "wine_glass") for idx in range(4)],
            *[_item(f"silver_fork_{idx}", "silver_fork") for idx in range(2)],
            *[_item(f"silver_knife_{idx}", "silver_knife") for idx in range(2)],
            *[_item(f"silver_spoon_{idx}", "silver_spoon") for idx in range(2)],
        ],
        prompt=(
            "A dining table with four place settings including plates, forks, "
            "knives, spoons, napkins, and glasses."
        ),
    )

    results = evaluate_manipuland_completeness(case_pack)

    assert len(results) == 1
    assert results[0]["label"] == "fail"
    assert results[0]["metric"] == "manipuland_completeness"
    assert results[0]["diagnostics"]["missing"] == {
        "fork": 2,
        "knife": 2,
        "napkin": 4,
        "spoon": 2,
    }


def test_dining_place_setting_passes_when_required_items_present() -> None:
    case_pack = _case_pack(
        [
            _table(),
            *[_item(f"dinner_plate_{idx}", "dinner_plate") for idx in range(4)],
            *[_item(f"silver_fork_{idx}", "silver_fork") for idx in range(4)],
            *[_item(f"silver_knife_{idx}", "silver_knife") for idx in range(4)],
            *[_item(f"silver_spoon_{idx}", "silver_spoon") for idx in range(4)],
            *[
                _item(f"white_linen_folded_napkin_{idx}", "white_linen_folded_napkin")
                for idx in range(4)
            ],
        ],
        prompt=(
            "A dining table with four place settings including plates, forks, "
            "knives, spoons, and napkins."
        ),
    )

    results = evaluate_manipuland_completeness(case_pack)

    assert len(results) == 1
    assert results[0]["label"] == "pass"
    assert results[0]["diagnostics"]["missing"] == {}


def test_generic_cutlery_does_not_require_western_full_set_or_napkins() -> None:
    case_pack = _case_pack(
        [
            _table(),
            *[_item(f"dinner_plate_{idx}", "dinner_plate") for idx in range(4)],
            *[_item(f"wine_glass_{idx}", "wine_glass") for idx in range(4)],
            *[_item(f"dining_fork_{idx}", "dining_fork") for idx in range(4)],
        ],
        prompt="Table settings for four including plates, cutlery, and glasses.",
    )

    results = evaluate_manipuland_completeness(case_pack)

    # 2026-07-12 修改原因：未点名 napkin/knife/spoon 时不应把西式全套当硬约束。
    assert len(results) == 1
    assert results[0]["label"] == "pass"
    assert results[0]["diagnostics"]["required_groups"] == [
        "cutlery",
        "drinkware",
        "plate",
    ]


def test_generic_utensil_asset_satisfies_generic_cutlery() -> None:
    case_pack = _case_pack(
        [
            _table(),
            *[_item(f"dinner_plate_{idx}", "dinner_plate") for idx in range(4)],
            *[_item(f"wine_glass_{idx}", "wine_glass") for idx in range(4)],
            *[_item(f"dining_utensil_{idx}", "dining_utensil") for idx in range(4)],
        ],
        prompt="Table settings for four including plates, cutlery, and glasses.",
    )

    result = evaluate_manipuland_completeness(case_pack)[0]

    assert result["label"] == "pass"
    assert result["diagnostics"]["counts"]["utensil"] == 4


def test_glass_vase_does_not_satisfy_requested_drinkware() -> None:
    case_pack = _case_pack(
        [
            _table(),
            *[_item(f"dinner_plate_{idx}", "dinner_plate") for idx in range(4)],
            *[_item(f"dining_fork_{idx}", "dining_fork") for idx in range(4)],
            _item("glass_vase_0", "glass_vase"),
        ],
        prompt="Table settings for four including plates, cutlery, and glasses.",
    )

    result = evaluate_manipuland_completeness(case_pack)[0]

    assert result["label"] == "fail"
    assert result["diagnostics"]["missing"] == {"drinkware": 4}


def test_explicit_setting_count_detects_missing_anchor_items() -> None:
    case_pack = _case_pack(
        [
            _table(),
            *[_item(f"dining_fork_{idx}", "dining_fork") for idx in range(4)],
            *[_item(f"wine_glass_{idx}", "wine_glass") for idx in range(4)],
        ],
        prompt="Table settings for four including plates, cutlery, and glasses.",
    )

    result = evaluate_manipuland_completeness(case_pack)[0]

    assert result["label"] == "fail"
    assert result["diagnostics"]["missing"] == {"plate": 4}


def test_non_western_bowl_chopstick_setting_is_supported() -> None:
    case_pack = _case_pack(
        [
            _table(),
            *[_item(f"rice_bowl_{idx}", "rice_bowl") for idx in range(4)],
            *[_item(f"chopsticks_{idx}", "chopsticks_pair") for idx in range(4)],
            *[_item(f"tea_cup_{idx}", "tea_cup") for idx in range(4)],
        ],
        prompt="Four settings with rice bowls, chopsticks, and tea cups.",
    )

    results = evaluate_manipuland_completeness(case_pack)

    assert len(results) == 1
    assert results[0]["label"] == "pass"


def test_decorative_plates_without_requested_place_settings_are_ignored() -> None:
    case_pack = _case_pack(
        [
            _table(),
            _item("decorative_plate_0", "decorative_plate"),
            _item("decorative_plate_1", "decorative_plate"),
        ],
        prompt="A dining room with decorative ceramics displayed on the table.",
    )

    assert evaluate_manipuland_completeness(case_pack) == []


def test_explicit_place_count_overrides_noisy_nearby_seat_inference() -> None:
    objects = [
        _table(),
        *[_item(f"dinner_plate_{idx}", "dinner_plate") for idx in range(4)],
        *[_item(f"wine_glass_{idx}", "wine_glass") for idx in range(4)],
        *[_item(f"dining_utensil_{idx}", "dining_utensil") for idx in range(4)],
    ]
    for idx in range(5):
        objects.append(_seat(f"dining_chair_{idx}", x=0.0, y=0.0))

    result = evaluate_manipuland_completeness(
        _case_pack(
            objects,
            prompt="Table settings for four including plates, cutlery, and glasses.",
        )
    )[0]

    # 2026-07-12 修改原因：显式四席不能被邻近区域中的额外座位误增为五席。
    assert result["label"] == "pass"
    assert result["diagnostics"]["place_count"] == 4


def test_sideboard_description_does_not_make_it_a_dining_table_or_seat() -> None:
    sideboard = {
        "id": "sideboard_0",
        "name": "sideboard",
        "description": "Storage beside the dining table and dining chairs.",
        "object_type": "furniture",
        "category": "sideboard",
        "bbox_world": {
            "center": [0.0, 0.7, 0.45],
            "size": [1.2, 0.45, 0.9],
            "min": [-0.6, 0.475, 0.0],
            "max": [0.6, 0.925, 0.9],
        },
        "support_surfaces": [{"surface_id": "S_sideboard"}],
        "functional_hints": {
            "scene_object_type": "furniture",
            "category_group": "storage_surface",
            "functional_categories": ["serving surface near dining table"],
        },
    }
    objects = [
        _table(),
        sideboard,
        *[_item(f"dinner_plate_{idx}", "dinner_plate") for idx in range(4)],
        *[_item(f"wine_glass_{idx}", "wine_glass") for idx in range(4)],
        *[_item(f"dining_utensil_{idx}", "dining_utensil") for idx in range(4)],
    ]

    results = evaluate_manipuland_completeness(
        _case_pack(
            objects,
            prompt="Table settings for four including plates, cutlery, and glasses.",
        )
    )

    # 2026-07-12 修改原因：关系描述中的 table/chairs 不能改变家具本体类别。
    assert len(results) == 1
    assert results[0]["primary_object"] == "dining_table_0"
    assert results[0]["label"] == "pass"


def test_dining_place_settings_fail_when_side_seats_use_corner_clusters() -> None:
    objects = [
        _table(),
        _oriented_seat("chair_north", x=-0.4, y=0.98, yaw_deg=180.0),
        _oriented_seat("chair_south", x=0.4, y=-0.98, yaw_deg=0.0),
        _oriented_seat("chair_east", x=1.45, y=0.0, yaw_deg=90.0),
        _oriented_seat("chair_west", x=-1.38, y=0.0, yaw_deg=-90.0),
        _positioned_item("plate_southwest", "dinner_plate", -0.45, -0.28),
        _positioned_item("plate_southeast", "dinner_plate", 0.45, -0.28),
        _positioned_item("plate_northwest", "dinner_plate", -0.45, 0.28),
        _positioned_item("plate_northeast", "dinner_plate", 0.45, 0.28),
        _positioned_item("utensil_southwest", "dining_utensil", -0.62, -0.28),
        _positioned_item("utensil_southeast", "dining_utensil", 0.62, -0.28),
        _positioned_item("utensil_northwest", "dining_utensil", -0.62, 0.28),
        _positioned_item("utensil_northeast", "dining_utensil", 0.62, 0.28),
    ]

    result = evaluate_dining_place_setting_alignment(
        _case_pack(
            objects,
            prompt="Table settings for four including plates and cutlery.",
        )
    )[0]

    # 2026-07-13 修改原因：复现四套餐具统一按桌面四角排布，导致左右椅的
    # 餐位偏离座椅正面。规则应按椅子 front axis，而不是桌面象限判定。
    assert result["label"] == "fail"
    assert result["relation_type"] == "dining_place_setting_alignment"
    by_seat = {
        row["seat_id"]: row for row in result["diagnostics"]["assignments"]
    }
    assert by_seat["chair_east"]["aligned"] is False
    assert by_seat["chair_west"]["aligned"] is False
    assert by_seat["chair_north"]["aligned"] is True
    assert by_seat["chair_south"]["aligned"] is True


def test_dining_place_settings_pass_when_each_cluster_uses_seat_front() -> None:
    objects = [
        _table(),
        _oriented_seat("chair_north", x=-0.4, y=0.98, yaw_deg=180.0),
        _oriented_seat("chair_south", x=0.4, y=-0.98, yaw_deg=0.0),
        _oriented_seat("chair_east", x=1.45, y=0.0, yaw_deg=90.0),
        _oriented_seat("chair_west", x=-1.38, y=0.0, yaw_deg=-90.0),
        _positioned_item("plate_north", "dinner_plate", -0.4, 0.28),
        _positioned_item("plate_south", "dinner_plate", 0.4, -0.28),
        _positioned_item("plate_east", "dinner_plate", 0.62, 0.0),
        _positioned_item("plate_west", "dinner_plate", -0.62, 0.0),
        _positioned_item("utensil_north", "dining_utensil", -0.55, 0.28),
        _positioned_item("utensil_south", "dining_utensil", 0.55, -0.28),
        _positioned_item("utensil_east", "dining_utensil", 0.62, -0.12),
        _positioned_item("utensil_west", "dining_utensil", -0.62, 0.12),
    ]

    result = evaluate_dining_place_setting_alignment(
        _case_pack(
            objects,
            prompt="Table settings for four including plates and cutlery.",
        )
    )[0]

    assert result["label"] == "pass"
    assert all(
        row["aligned"] and not row["misaligned_companion_ids"]
        for row in result["diagnostics"]["assignments"]
    )


def test_dining_place_settings_allow_scale_relative_side_drinkware() -> None:
    objects = [
        _table(),
        _oriented_seat("chair_north", x=0.0, y=0.98, yaw_deg=180.0),
        _oriented_seat("chair_south", x=0.0, y=-0.98, yaw_deg=0.0),
        _oriented_seat("chair_east", x=1.45, y=0.0, yaw_deg=90.0),
        _oriented_seat("chair_west", x=-1.38, y=0.0, yaw_deg=-90.0),
        _positioned_item("plate_north", "dinner_plate", 0.0, 0.28),
        _positioned_item("plate_south", "dinner_plate", 0.0, -0.28),
        _positioned_item("plate_east", "dinner_plate", 0.62, 0.0),
        _positioned_item("plate_west", "dinner_plate", -0.62, 0.0),
        _positioned_item("glass_north", "wine_glass", 0.26, 0.28),
        _positioned_item("glass_south", "wine_glass", -0.26, -0.28),
        _positioned_item("glass_east", "wine_glass", 0.62, 0.24),
        _positioned_item("glass_west", "wine_glass", -0.62, -0.24),
    ]

    result = evaluate_dining_place_setting_alignment(
        _case_pack(objects, prompt="Table settings for four including plates and glasses.")
    )[0]

    # 2026-07-14 修改原因：酒杯/配套物可自然放在盘子侧边；只要仍靠近其
    # 一对一餐盘且在该座位的尺度化可达范围内，不应诱发模型反复移动餐盘。
    assert result["label"] == "pass"
    assert all(not row["misaligned_companion_ids"] for row in result["diagnostics"]["assignments"])


def test_centered_four_side_chairs_reject_four_corner_plate_grid() -> None:
    objects = [
        _table(),
        _oriented_seat("chair_north", x=0.0, y=0.98, yaw_deg=180.0),
        _oriented_seat("chair_south", x=0.0, y=-0.98, yaw_deg=0.0),
        _oriented_seat("chair_east", x=1.45, y=0.0, yaw_deg=90.0),
        _oriented_seat("chair_west", x=-1.38, y=0.0, yaw_deg=-90.0),
        _positioned_item("plate_southwest", "dinner_plate", -0.26, -0.28),
        _positioned_item("plate_southeast", "dinner_plate", 0.26, -0.28),
        _positioned_item("plate_northwest", "dinner_plate", -0.26, 0.28),
        _positioned_item("plate_northeast", "dinner_plate", 0.26, 0.28),
    ]

    result = evaluate_dining_place_setting_alignment(
        _case_pack(objects, prompt="Table settings for four including plates.")
    )[0]

    # 2026-07-13 修改原因：复现真机中座椅已经在四边居中、餐盘仍保持 2x2
    # 四角网格的情况。每个餐盘都应收到其所属座椅中心线的明确二维移动目标。
    assert result["label"] == "fail"
    assignments = result["diagnostics"]["assignments"]
    assert len(assignments) == 4
    assert all(not row["aligned"] for row in assignments)
    assert all(row["allowed_lateral_offset_m"] < 0.1 for row in assignments)
    assert all(
        any(abs(value) > 0.2 for value in row["recommended_translation_xy_m"])
        for row in assignments
    )
    assert "whole place-setting cluster" in result["reason"]


def test_dining_alignment_selects_nearest_segmented_table_surface() -> None:
    table = _table()
    # 2026-07-14 修改原因：真实 HSSD 餐桌会把一个连续桌面拆成数个窄的
    # support surface。北/南端餐位必须指向各自最先接触到的条带，不能退化为
    # 面积最大的中央条带或整张家具 footprint。
    table["support_regions"] = [
        {
            "region_id": "S_center",
            "polygon_world_xy": [[-0.8, -0.1], [0.8, -0.1], [0.8, 0.1], [-0.8, 0.1]],
        },
        {
            "region_id": "S_north",
            "polygon_world_xy": [[-0.8, 0.1], [0.8, 0.1], [0.8, 0.32], [-0.8, 0.32]],
        },
        {
            "region_id": "S_south",
            "polygon_world_xy": [[-0.8, -0.32], [0.8, -0.32], [0.8, -0.1], [-0.8, -0.1]],
        },
    ]
    objects = [
        table,
        _oriented_seat("chair_north", x=0.0, y=0.98, yaw_deg=180.0),
        _oriented_seat("chair_south", x=0.0, y=-0.98, yaw_deg=0.0),
        _oriented_seat("chair_east", x=1.45, y=0.0, yaw_deg=90.0),
        _oriented_seat("chair_west", x=-1.38, y=0.0, yaw_deg=-90.0),
        _positioned_item("plate_north", "dinner_plate", 0.0, 0.22),
        _positioned_item("plate_south", "dinner_plate", 0.0, -0.22),
        _positioned_item("plate_east", "dinner_plate", 0.58, 0.0),
        _positioned_item("plate_west", "dinner_plate", -0.58, 0.0),
    ]

    result = evaluate_dining_place_setting_alignment(
        _case_pack(objects, prompt="Table settings for four including plates.")
    )[0]
    by_seat = {row["seat_id"]: row for row in result["diagnostics"]["assignments"]}

    assert by_seat["chair_north"]["recommended_support_surface_id"] == "S_north"
    assert by_seat["chair_south"]["recommended_support_surface_id"] == "S_south"
    assert by_seat["chair_east"]["recommended_support_surface_id"] == "S_center"
    assert by_seat["chair_west"]["recommended_support_surface_id"] == "S_center"


def test_dining_alignment_scales_with_table_and_plate_sizes() -> None:
    for width, depth, plate_size in ((1.2, 0.7, 0.2), (2.8, 1.2, 0.34)):
        table = _scaled_table(width, depth)
        inset = plate_size / 2.0 + max(0.03, 0.05 * min(width, depth))
        target_y = depth / 2.0 - inset
        north_plate = _positioned_item("plate_north", "dinner_plate", 0.0, target_y)
        south_plate = _positioned_item("plate_south", "dinner_plate", 0.0, -target_y)
        _resize_item(north_plate, plate_size)
        _resize_item(south_plate, plate_size)
        objects = [
            table,
            _oriented_seat(
                "chair_north", x=0.0, y=depth / 2.0 + 0.5, yaw_deg=180.0
            ),
            _oriented_seat(
                "chair_south", x=0.0, y=-depth / 2.0 - 0.5, yaw_deg=0.0
            ),
            north_plate,
            south_plate,
        ]

        result = evaluate_dining_place_setting_alignment(
            _case_pack(objects, prompt="Two dining place settings with plates.")
        )[0]

        # 2026-07-13 修改原因：桌边槽位必须随桌深和盘径变化，不能只通过
        # 当前 1.87m x 0.75m 四人桌的固定坐标回归。
        assert result["label"] == "pass"
        assert all(row["aligned"] for row in result["diagnostics"]["assignments"])


def test_dining_alignment_uses_rotated_table_polygon() -> None:
    yaw_deg = 32.0
    yaw = math.radians(yaw_deg)
    width, depth, plate_size = 2.0, 0.9, 0.27
    table = _scaled_table(width, depth, yaw_deg=yaw_deg)
    inset = plate_size / 2.0 + 0.05 * depth
    target_y = depth / 2.0 - inset

    def rotate(x: float, y: float) -> tuple[float, float]:
        return (
            x * math.cos(yaw) - y * math.sin(yaw),
            x * math.sin(yaw) + y * math.cos(yaw),
        )

    north_xy = rotate(0.0, depth / 2.0 + 0.5)
    south_xy = rotate(0.0, -depth / 2.0 - 0.5)
    north_plate_xy = rotate(0.0, target_y)
    south_plate_xy = rotate(0.0, -target_y)
    north_plate = _positioned_item(
        "plate_north", "dinner_plate", *north_plate_xy
    )
    south_plate = _positioned_item(
        "plate_south", "dinner_plate", *south_plate_xy
    )
    objects = [
        table,
        _oriented_seat(
            "chair_north", x=north_xy[0], y=north_xy[1], yaw_deg=180.0 + yaw_deg
        ),
        _oriented_seat(
            "chair_south", x=south_xy[0], y=south_xy[1], yaw_deg=yaw_deg
        ),
        north_plate,
        south_plate,
    ]

    result = evaluate_dining_place_setting_alignment(
        _case_pack(objects, prompt="Two dining place settings with plates.")
    )[0]

    assert result["label"] == "pass"


def test_dining_alignment_ignores_decorative_tabletop_objects() -> None:
    objects = [
        _table(),
        _oriented_seat("chair_north", x=0.0, y=0.98, yaw_deg=180.0),
        _oriented_seat("chair_south", x=0.0, y=-0.98, yaw_deg=0.0),
        _positioned_item("decorative_plate_0", "decorative_plate", -0.3, 0.0),
        _positioned_item("decorative_plate_1", "decorative_plate", 0.3, 0.0),
    ]

    assert (
        evaluate_dining_place_setting_alignment(
            _case_pack(
                objects,
                prompt="Decorative ceramic plates displayed on the dining table.",
            )
        )
        == []
    )


def _case_pack(objects: list[dict], *, prompt: str) -> dict:
    return {"task_instruction": prompt, "scene_geometry": {"objects": objects}}


def _table() -> dict:
    return {
        "id": "dining_table_0",
        "name": "dining_table",
        "description": "rectangular dining table",
        "object_type": "furniture",
        "category": "dining_table",
        "bbox_world": {
            "center": [0.0, 0.0, 0.75],
            "size": [1.6, 0.9, 0.75],
            "min": [-0.8, -0.45, 0.375],
            "max": [0.8, 0.45, 1.125],
        },
        "footprint_world": [
            [-0.8, -0.45],
            [0.8, -0.45],
            [0.8, 0.45],
            [-0.8, 0.45],
        ],
        "support_surfaces": [{"surface_id": "S_table"}],
        "functional_hints": {"scene_object_type": "furniture"},
    }


def _scaled_table(width: float, depth: float, *, yaw_deg: float = 0.0) -> dict:
    table = _table()
    yaw = math.radians(yaw_deg)
    corners = [
        (-width / 2.0, -depth / 2.0),
        (width / 2.0, -depth / 2.0),
        (width / 2.0, depth / 2.0),
        (-width / 2.0, depth / 2.0),
    ]
    footprint = [
        [
            x * math.cos(yaw) - y * math.sin(yaw),
            x * math.sin(yaw) + y * math.cos(yaw),
        ]
        for x, y in corners
    ]
    xs = [point[0] for point in footprint]
    ys = [point[1] for point in footprint]
    table["yaw_deg"] = yaw_deg
    table["footprint_world"] = footprint
    table["bbox_world"].update(
        {
            "size": [max(xs) - min(xs), max(ys) - min(ys), 0.75],
            "min": [min(xs), min(ys), 0.375],
            "max": [max(xs), max(ys), 1.125],
        }
    )
    return table


def _item(object_id: str, name: str) -> dict:
    return {
        "id": object_id,
        "name": name,
        "description": name.replace("_", " "),
        "object_type": "manipuland",
        "category": name,
        "bbox_world": {"center": [0.0, 0.0, 0.9]},
        "placement_info": {"parent_surface_id": "S_table"},
        "functional_hints": {"scene_object_type": "manipuland"},
    }


def _seat(object_id: str, *, x: float, y: float) -> dict:
    return {
        "id": object_id,
        "name": "dining_chair",
        "description": "dining chair",
        "object_type": "furniture",
        "category": "dining_chair",
        "bbox_world": {
            "center": [x, y, 0.45],
            "size": [0.5, 0.5, 0.9],
            "min": [x - 0.25, y - 0.25, 0.0],
            "max": [x + 0.25, y + 0.25, 0.9],
        },
        "functional_hints": {"scene_object_type": "furniture"},
    }


def _oriented_seat(
    object_id: str, *, x: float, y: float, yaw_deg: float
) -> dict:
    seat = _seat(object_id, x=x, y=y)
    seat["yaw_deg"] = yaw_deg
    return seat


def _positioned_item(object_id: str, name: str, x: float, y: float) -> dict:
    item = _item(object_id, name)
    width = 0.27 if "plate" in name or "bowl" in name else 0.06
    depth = width if width > 0.1 else 0.2
    item["bbox_world"] = {
        "center": [x, y, 0.9],
        "size": [width, depth, 0.04],
        "min": [x - width / 2.0, y - depth / 2.0, 0.88],
        "max": [x + width / 2.0, y + depth / 2.0, 0.92],
    }
    return item


def _resize_item(item: dict, size: float) -> None:
    center = item["bbox_world"]["center"]
    item["bbox_world"].update(
        {
            "size": [size, size, 0.04],
            "min": [center[0] - size / 2.0, center[1] - size / 2.0, 0.88],
            "max": [center[0] + size / 2.0, center[1] + size / 2.0, 0.92],
        }
    )
