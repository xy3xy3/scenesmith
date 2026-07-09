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
        ]
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
        ]
    )

    results = evaluate_manipuland_completeness(case_pack)

    assert len(results) == 1
    assert results[0]["label"] == "pass"
    assert results[0]["diagnostics"]["missing"] == {}


def _case_pack(objects: list[dict]) -> dict:
    return {"scene_geometry": {"objects": objects}}


def _table() -> dict:
    return {
        "id": "dining_table_0",
        "name": "dining_table",
        "description": "rectangular dining table",
        "object_type": "furniture",
        "category": "dining_table",
        "bbox_world": {"center": [0.0, 0.0, 0.75]},
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
