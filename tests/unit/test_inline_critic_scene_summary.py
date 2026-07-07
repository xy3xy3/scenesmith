"""Unit tests for inline critic fallback scene summaries.

2026-07-07: Added to lock the critic inline-retry fix that restricts manipuland
fallback context to the current furniture scope after sideboard/table drift.
"""

import json
from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.room import AgentType, ObjectType, PlacementInfo, SceneObject


class _TestAgent(BaseStatefulAgent):
    @property
    def agent_type(self) -> AgentType:
        return AgentType.MANIPULAND

    def _get_final_scores_directory(self) -> Path:
        return Path(".")

    def _get_critique_prompt_enum(self):
        return None

    def _get_extra_critique_kwargs(self) -> dict:
        return {}

    async def _run_agent_workflow(self) -> None:
        return None

    def _get_planner_prompt_enum(self):
        return None

    def _get_designer_prompt_enum(self):
        return None

    def _get_design_change_prompt_enum(self):
        return None

    def _get_initial_design_prompt_enum(self):
        return None

    def _get_initial_design_prompt_kwargs(self) -> dict:
        return {}

    def _set_placement_noise_profile(self, _mode) -> None:
        return None


class _FakeSurface:
    def __init__(self, surface_id: str):
        self.surface_id = surface_id


class _FakeScene:
    def __init__(self, objects: list[SceneObject]):
        self.objects = {obj.object_id: obj for obj in objects}
        self.room_id = "room_dining_room"

    def get_objects_by_type(self, object_type: ObjectType) -> list[SceneObject]:
        return [
            obj for obj in self.objects.values() if obj.object_type == object_type
        ]

    def get_object(self, object_id: str) -> SceneObject | None:
        return self.objects.get(object_id)


def _make_object(
    object_id: str,
    object_type: ObjectType,
    name: str,
    xyz: tuple[float, float, float],
    placement_info: PlacementInfo | None = None,
    support_surfaces: list[object] | None = None,
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        object_type=object_type,
        name=name,
        description=f"{name} description",
        transform=RigidTransform(p=np.array(xyz)),
        support_surfaces=support_surfaces or [],
        placement_info=placement_info,
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
    )


def test_manipuland_inline_summary_only_includes_current_furniture_scope() -> None:
    sideboard_surface = _FakeSurface("sideboard_top")
    table_surface = _FakeSurface("table_top")

    sideboard = _make_object(
        object_id="sideboard_0",
        object_type=ObjectType.FURNITURE,
        name="Sideboard",
        xyz=(1.0, 2.0, 0.0),
        support_surfaces=[sideboard_surface],
    )
    dining_table = _make_object(
        object_id="dining_table_0",
        object_type=ObjectType.FURNITURE,
        name="Dining Table",
        xyz=(4.0, 5.0, 0.0),
        support_surfaces=[table_surface],
    )
    coaster = _make_object(
        object_id="coaster_0",
        object_type=ObjectType.MANIPULAND,
        name="Coaster",
        xyz=(1.1, 2.1, 0.9),
        placement_info=PlacementInfo(
            parent_surface_id="sideboard_top",
            position_2d=np.array([0.0, 0.0]),
            rotation_2d=0.0,
            placement_method="manual",
        ),
    )
    plate = _make_object(
        object_id="plate_0",
        object_type=ObjectType.MANIPULAND,
        name="Plate",
        xyz=(4.1, 5.1, 0.75),
        placement_info=PlacementInfo(
            parent_surface_id="table_top",
            position_2d=np.array([0.0, 0.0]),
            rotation_2d=0.0,
            placement_method="manual",
        ),
    )

    scene = _FakeScene([sideboard, dining_table, coaster, plate])
    agent = object.__new__(_TestAgent)
    agent.scene = scene
    agent.current_furniture_id = "sideboard_0"
    agent.room_bounds = None
    agent.ceiling_height = None

    summary = agent._build_inline_critic_scene_summary()

    assert summary is not None
    relevant_objects = json.loads(summary)["relevant_objects"]
    relevant_ids = {obj["object_id"] for obj in relevant_objects}

    assert relevant_ids == {"sideboard_0", "coaster_0"}
    assert "dining_table_0" not in relevant_ids
    assert "plate_0" not in relevant_ids
