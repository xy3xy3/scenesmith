"""Tests for manipuland completeness critic rules."""

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
        _positioned_item("plate_east", "dinner_plate", 0.45, 0.0),
        _positioned_item("plate_west", "dinner_plate", -0.45, 0.0),
        _positioned_item("utensil_north", "dining_utensil", -0.55, 0.28),
        _positioned_item("utensil_south", "dining_utensil", 0.55, -0.28),
        _positioned_item("utensil_east", "dining_utensil", 0.52, -0.12),
        _positioned_item("utensil_west", "dining_utensil", -0.52, 0.12),
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
        "support_surfaces": [{"surface_id": "S_table"}],
        "functional_hints": {"scene_object_type": "furniture"},
    }


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
