"""Tests for window planning and SceneBenchmark window clearance rules."""

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.scenebenchmark_critic.checks import build_checks
from scenesmith.scenebenchmark_critic.dining_seat_distribution import (
    _seat_facing_error_deg,
)
from scenesmith.scenebenchmark_critic.orientation_contracts import _plan_contract


def test_bedroom_window_budget_preserves_wall_capacity() -> None:
    # 2026-07-14 修改原因：回归卧室最多两个窗，避免 share base 为后续电视/壁挂物
    # 留不出可用墙面。
    layout = HouseLayout()
    tools = FloorPlanTools(layout=layout, mode="room")
    result = tools._generate_room_specs_impl(
        '[{"type":"bedroom","prompt":"bedroom","width":5.0,"depth":4.5}]'
    )
    assert result.success
    assert tools._add_window_impl("A", "center").success
    assert tools._add_window_impl("B", "center").success
    rejected = tools._add_window_impl("C", "center")
    assert not rejected.success
    assert "window budget" in rejected.message


def test_window_clearance_prefers_window_repair() -> None:
    # 2026-07-14 修改原因：窗口被高柜遮挡时，critic 结果必须携带窗口修复优先级。
    case_pack = {
        "scene_geometry": {
            "objects": [
                {
                    "id": "wardrobe_0",
                    "category": "wardrobe",
                    "category_norm": "wardrobe",
                    "bbox_world": {"min": [0.0, 0.0, 0.0], "max": [1.0, 0.5, 2.0]},
                }
            ],
            "scene_shell": {
                "windows": [
                    {
                        "id": "window_0",
                        "sill_height": 0.9,
                        "bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 0.5, 2.1]},
                    }
                ]
            },
            "relations": [],
        }
    }
    checks = build_checks(case_pack, metrics=["interaction_clearance"])
    check = next(item for item in checks if item["check_id"] == "window_clearance__window_0")
    assert check["clearance_result"]["label"] == "fail"
    assert check["clearance_result"]["repair_priority"][:3] == [
        "shrink_window", "move_window", "remove_window"
    ]


def test_minor_sideboard_window_overlap_is_advisory() -> None:
    # 2026-07-14 修改原因：sideboard 仅高出窗台少量时不应触发移动家具循环。
    case_pack = {
        "scene_geometry": {
            "objects": [
                {
                    "id": "sideboard_0",
                    "category": "sideboard",
                    "bbox_world": {"min": [0.0, 0.0, 0.0], "max": [1.0, 0.5, 0.99]},
                }
            ],
            "scene_shell": {
                "windows": [
                    {
                        "id": "window_0",
                        "sill_height": 0.9,
                        "bbox": {"min": [0.0, 0.0, 0.0], "max": [1.0, 0.5, 2.1]},
                    }
                ]
            },
            "relations": [],
        }
    }
    checks = build_checks(case_pack, metrics=["interaction_clearance"])
    check = next(item for item in checks if item["check_id"] == "window_clearance__window_0")
    assert check["clearance_result"]["label"] == "pass"
    assert check["clearance_result"]["advisory_blocking_objects"] == ["sideboard_0"]


def test_dining_chair_contract_precedes_incidental_wall_proximity() -> None:
    # 2026-07-14 修改原因：回归 dining_chair 靠墙后仍绑定 dining_table，避免
    # door clearance 将其错误切换为 back_against_wall。
    chair = {"id": "dining_chair_2", "category": "dining_chair"}
    table = {"id": "dining_table_0", "category": "dining_table"}
    contract = _plan_contract(
        chair,
        [chair, table],
        media_focus=None,
        media_intent=False,
        stage="test",
    )
    assert contract is not None
    assert contract["relation_type"] == "seating_to_work_surface"
    assert contract["target_ids"] == ["dining_table_0"]


def test_dining_chair_strict_facing_detects_lateral_offset() -> None:
    # 2026-07-14 修改原因：13° 左右的偏角不应再被宽松 check_facing 判为完美正对。
    chair = {
        "id": "dining_chair_2",
        "bbox_world": {"center": [-1.45, 0.30], "size": [0.87, 0.67]},
        "yaw_deg": -90.0,
    }
    error = _seat_facing_error_deg(chair, (0.0, 0.0))
    assert error is not None
    assert 10.0 < error < 20.0


def test_resize_window_preserves_wall_and_updates_opening() -> None:
    # 2026-07-14 修改原因：窗口遮挡修复应支持缩小窗口，而不是只能删除或移动家具。
    layout = HouseLayout()
    tools = FloorPlanTools(layout=layout, mode="room")
    result = tools._generate_room_specs_impl(
        '[{"type":"living_room","prompt":"living room","width":5.0,"depth":4.0}]'
    )
    assert result.success
    assert tools._add_window_impl("A", "center", width=1.5).success
    window = layout.windows[0]
    old_center = window.position_along_wall + window.width / 2
    resized = tools._resize_window_impl(window.id, width=0.9)
    assert resized.success
    assert layout.windows[0].width == 0.9
    assert layout.windows[0].position_along_wall + 0.45 == old_center
    wall = next(w for w in layout.placed_rooms[0].walls if w.direction == window.wall_direction)
    opening = next(o for o in wall.openings if o.opening_id == window.id)
    assert opening.width == 0.9
