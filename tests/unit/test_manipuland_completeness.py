"""Tests for manipuland completeness critic rules."""

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
