from __future__ import annotations

from scenesmith.scenebenchmark_critic.prompt_context import format_agent_prompt_context
from scenesmith.scenebenchmark_critic.room_center_alignment import (
    evaluate_room_center_alignment,
)


def _obj(object_id: str, category: str, x: float, y: float) -> dict:
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "name": category,
        "description": category,
        "bbox_world": {
            "center": [x, y, 0.4],
            "size": [1.0, 0.6, 0.8],
            "min": [x - 0.5, y - 0.3, 0.0],
            "max": [x + 0.5, y + 0.3, 0.8],
        },
    }


def _case(prompt: str, table_y: float) -> dict:
    return {
        "task_instruction": prompt,
        "scene_geometry": {
            "rooms": [
                {
                    "id": "room",
                    "bbox": {"min": [-2.25, -2.0, 0], "max": [2.25, 2.0, 3]},
                }
            ],
            "objects": [
                _obj("dining_table_0", "dining_table", 0.0, table_y),
                _obj("dining_chair_0", "dining_chair", 0.0, table_y - 0.9),
            ],
        },
    }


def test_explicit_room_center_is_checked_and_reports_group_repair() -> None:
    result = evaluate_room_center_alignment(
        _case(
            "A dining room with a dining table in the center and one chair on each side.",
            1.2,
        )
    )

    assert len(result) == 1
    assert result[0]["label"] == "fail"
    assert result[0]["relation_type"] == "room_center_alignment"
    assert result[0]["related_objects"] == ["dining_chair_0"]
    assert "coordinated group" in result[0]["repair_advice"]


def test_relative_center_constraints_do_not_become_room_center_contracts() -> None:
    result = evaluate_room_center_alignment(
        _case(
            "A dining room with a dining table centered between the sideboard and door.",
            1.2,
        )
    )

    assert result == []


def test_tabletop_centerpiece_does_not_become_room_center_contract() -> None:
    result = evaluate_room_center_alignment(
        _case(
            "A centerpiece vase with flowers sits in the middle of the table, "
            "and a set of coasters sits on the sideboard.",
            1.2,
        )
    )

    assert result == []


def test_central_anchor_and_room_center_clause_are_supported() -> None:
    central = evaluate_room_center_alignment(
        _case("A central dining table has four chairs around it.", 1.2)
    )
    room_contains = evaluate_room_center_alignment(
        _case("The center of the room contains a dining table.", 1.2)
    )

    assert central and central[0]["label"] == "fail"
    assert room_contains and room_contains[0]["label"] == "fail"


def test_furniture_prompt_context_preserves_passed_center_contract() -> None:
    payload = {
        "results": evaluate_room_center_alignment(
            _case("A dining room with a dining table in the center.", 0.0)
        ),
        "case_pack": {"checks": []},
    }

    context = format_agent_prompt_context(
        payload,
        agent_type="furniture",
    )

    assert "room-center placement contracts" in context
    assert "Do not move the anchor alone" in context
