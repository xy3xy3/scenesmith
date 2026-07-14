"""Tests for window planning and SceneBenchmark window clearance rules."""

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.scenebenchmark_critic.checks import build_checks


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
    assert check["clearance_result"]["repair_priority"][0] == "remove_window_or_move_window"
