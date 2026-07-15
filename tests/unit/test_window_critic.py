"""Tests for window planning and SceneBenchmark window clearance rules."""

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.scenebenchmark_critic.checks import build_checks
from scenesmith.scenebenchmark_critic.dining_seat_distribution import (
    _seat_facing_error_deg,
)
from scenesmith.scenebenchmark_critic.orientation_contracts import _plan_contract
from scenesmith.scenebenchmark_critic.prompt_context import (
    filter_prompt_results_for_agent,
)
from scenesmith.agent_utils.room import AgentType
from scenesmith.scenebenchmark_critic.clearance_source import evaluate_clearance
from scenesmith.scenebenchmark_critic.media_support_alignment import (
    evaluate_media_support_alignment,
)


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


def test_window_clearance_detects_wall_mounted_object_on_same_wall() -> None:
    # 2026-07-14 修改原因：窗口与壁挂电视共享墙面时，完整 3D AABB 可能只在
    # 墙体厚度方向相交，旧规则会误判为无遮挡。critic 应检查墙面二维开口区，
    # 并把窗口修复建议传递给 wall-mounted agent。
    case_pack = {
        "scene_geometry": {
            "objects": [
                {
                    "id": "wall_mounted_television_0",
                    "category": "television",
                    "category_norm": "television",
                    "object_type": "wall_mounted",
                    "bbox_world": {
                        "min": [0.2, 2.16, 1.1],
                        "max": [0.9, 2.2, 1.9],
                    },
                }
            ],
            "scene_shell": {
                "windows": [
                    {
                        "id": "window_0",
                        "wall_direction": "north",
                        "sill_height": 0.9,
                        "bbox": {
                            "min": [0.0, 1.75, 0.0],
                            "max": [1.0, 2.25, 2.1],
                        },
                    }
                ]
            },
            "relations": [],
        }
    }
    check = next(
        item
        for item in build_checks(case_pack, metrics=["interaction_clearance"])
        if item["check_id"] == "window_clearance__window_0"
    )
    assert check["clearance_result"]["label"] == "fail"
    assert check["clearance_result"]["wall_mounted_blocking_objects"] == [
        "wall_mounted_television_0"
    ]

    result = evaluate_clearance(check)
    payload = {"case_pack": case_pack, "results": [result]}
    filtered = filter_prompt_results_for_agent(
        payload, agent_type=AgentType.WALL_MOUNTED
    )
    assert [item["check_id"] for item in filtered] == [
        "window_clearance__window_0"
    ]
    assert "Prefer removing the window" in result["reason"]


def test_window_clearance_ignores_wall_mounted_object_on_other_wall() -> None:
    # 2026-07-14 修改原因：同一房间其他墙上的壁挂物不应因沿墙投影巧合重叠
    # 而触发窗口修复，避免 critic 跨墙移动正常电视/镜子。
    case_pack = {
        "scene_geometry": {
            "objects": [
                {
                    "id": "wall_mounted_television_0",
                    "category": "television",
                    "object_type": "wall_mounted",
                    "bbox_world": {
                        "min": [0.2, 0.0, 1.1],
                        "max": [0.9, 0.04, 1.9],
                    },
                }
            ],
            "scene_shell": {
                "windows": [
                    {
                        "id": "window_0",
                        "wall_direction": "north",
                        "sill_height": 0.9,
                        "bbox": {
                            "min": [0.0, 1.75, 0.0],
                            "max": [1.0, 2.25, 2.1],
                        },
                    }
                ]
            },
            "relations": [],
        }
    }
    check = next(
        item
        for item in build_checks(case_pack, metrics=["interaction_clearance"])
        if item["check_id"] == "window_clearance__window_0"
    )
    assert check["clearance_result"]["label"] == "pass"


def test_window_clearance_uses_declared_wall_type_when_hint_is_furniture() -> None:
    # 2026-07-14 修改原因：回归错误 functional hint 不应覆盖明确的
    # object_type=wall_mounted，确保真实 TV 导出数据仍能触发窗口净空 critic。
    case_pack = {
        "scene_geometry": {
            "objects": [
                {
                    "id": "television_0",
                    "category": "television",
                    "object_type": "wall_mounted",
                    "functional_hints": {"scene_object_type": "furniture"},
                    "bbox_world": {
                        "min": [0.2, 2.16, 1.1],
                        "max": [0.9, 2.2, 1.9],
                    },
                }
            ],
            "scene_shell": {
                "windows": [
                    {
                        "id": "window_0",
                        "wall_direction": "north",
                        "sill_height": 0.9,
                        "bbox": {
                            "min": [0.0, 1.75, 0.0],
                            "max": [1.0, 2.25, 2.1],
                        },
                    }
                ]
            },
            "relations": [],
        }
    }
    check = next(
        item
        for item in build_checks(case_pack, metrics=["interaction_clearance"])
        if item["check_id"] == "window_clearance__window_0"
    )
    assert check["clearance_result"]["label"] == "fail"
    assert check["clearance_result"]["wall_mounted_blocking_objects"] == [
        "television_0"
    ]

    result = evaluate_clearance(check)
    filtered = filter_prompt_results_for_agent(
        {"case_pack": case_pack, "results": [result]},
        agent_type=AgentType.WALL_MOUNTED,
    )
    assert [item["check_id"] for item in filtered] == [
        "window_clearance__window_0"
    ]


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


def test_move_window_preserves_dimensions_and_updates_opening() -> None:
    # 2026-07-14 修改原因：窗口修复顺序包含移动；移动后必须同步 Window 和
    # Wall.openings，下一次 wall surface 提取才会使用新排除区。
    layout = HouseLayout()
    tools = FloorPlanTools(layout=layout, mode="room")
    result = tools._generate_room_specs_impl(
        '[{"type":"living_room","prompt":"living room","width":5.0,"depth":4.0}]'
    )
    assert result.success
    assert tools._add_window_impl("A", "center", width=0.9).success
    window = layout.windows[0]
    old_width = window.width
    moved = tools._move_window_impl(window.id, position_along_wall=0.8)
    assert moved.success
    assert layout.windows[0].position_along_wall == 0.8
    assert layout.windows[0].width == old_width
    wall = next(w for w in layout.placed_rooms[0].walls if w.direction == window.wall_direction)
    opening = next(o for o in wall.openings if o.opening_id == window.id)
    assert opening.position_along_wall == 0.8


def test_wall_media_support_alignment_detects_window_forced_offset() -> None:
    # 2026-07-14 修改原因：TV 可能合法地避开窗口，却被迫偏离 TV stand；critic
    # 必须报告“位于支撑家具正上方”的 functional dependency，而不只检查碰撞。
    case_pack = {
        "scene_geometry": {
            "objects": [
                {
                    "id": "tv_stand_0",
                    "room": "living_room",
                    "category_norm": "tv_stand",
                    "bbox_world": {
                        "min": [-0.7, -2.4, 0.0],
                        "max": [0.7, -1.9, 0.8],
                    },
                },
                {
                    "id": "television_0",
                    "room": "living_room",
                    "category_norm": "television",
                    "object_type": "wall_mounted",
                    "placement_info": {"parent_surface_id": "living_room_south"},
                    "bbox_world": {
                        "min": [-1.9, -2.45, 1.1],
                        "max": [-0.7, -2.4, 1.9],
                    },
                },
            ]
        }
    }
    result = evaluate_media_support_alignment(case_pack)
    assert len(result) == 1
    assert result[0]["label"] == "fail"
    assert result[0]["relation_type"] == "media_over_support_alignment"
    assert result[0]["diagnostics"]["lateral_offset_m"] > 1.0


def test_media_alignment_resolves_support_wall_and_window_before_repair() -> None:
    # 2026-07-15 修改原因：TV 放到任意侧墙时，critic 需要指出 TV stand 所在墙
    # 及其窗口，避免 agent 只沿当前墙横移或用侧墙绕开开口。
    case_pack = {
        "scene_geometry": {
            "objects": [
                {
                    "id": "tv_stand_0",
                    "room": "living_room",
                    "category_norm": "tv_stand",
                    "bbox_world": {
                        "min": [-0.7, -2.4, 0.0],
                        "max": [0.7, -1.9, 0.8],
                    },
                },
                {
                    "id": "south_wall",
                    "room": "living_room",
                    "category_norm": "wall",
                    "object_type": "wall",
                    "bbox_world": {
                        "min": [-3.0, -2.5, 0.0],
                        "max": [3.0, -2.45, 2.7],
                    },
                },
                {
                    "id": "east_wall",
                    "room": "living_room",
                    "category_norm": "wall",
                    "object_type": "wall",
                    "bbox_world": {
                        "min": [2.95, -2.5, 0.0],
                        "max": [3.0, 2.5, 2.7],
                    },
                },
                {
                    "id": "television_0",
                    "room": "living_room",
                    "category_norm": "television",
                    "object_type": "wall_mounted",
                    "placement_info": {"parent_surface_id": "living_room_east"},
                        "bbox_world": {
                            "size": [0.05, 1.0, 0.7],
                            "min": [2.9, -0.5, 1.0],
                            "max": [2.95, 0.5, 1.7],
                        },
                },
            ],
            "scene_shell": {
                "windows": [
                    {
                        "id": "window_south",
                        "wall_direction": "south",
                        "bbox": {
                            "min": [-0.25, -2.5, 0.0],
                            "max": [0.25, -2.0, 2.1],
                        },
                    }
                ]
            },
        }
    }

    result = evaluate_media_support_alignment(case_pack)[0]

    assert result["label"] == "fail"
    assert result["diagnostics"]["target_wall_surface_id"] == "south_wall"
    assert result["diagnostics"]["target_wall_window_ids"] == ["window_south"]
    assert "support is on `south_wall`" in result["reason"]
    assert "window_south" in result["reason"]
