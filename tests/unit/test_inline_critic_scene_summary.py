"""Unit tests for inline critic fallback scene summaries.

2026-07-07: Added to lock the critic inline-retry fix that restricts manipuland
fallback context to the current furniture scope after sideboard/table drift.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from pydrake.all import RigidTransform

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.room import AgentType, ObjectType, PlacementInfo, SceneObject
from scenesmith.agent_utils.scoring import (
    CategoryScore,
    ManipulandCritiqueWithScores,
)


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
        self.bounding_box_min = np.array([-1.0, -0.5, 0.0])
        self.bounding_box_max = np.array([1.0, 0.5, 0.0])


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
    summary_data = json.loads(summary)
    assert summary_data["current_furniture_id"] == "sideboard_0"
    assert summary_data["valid_surface_ids"] == ["sideboard_top"]
    assert summary_data["current_furniture_surface_count"] == 1
    assert summary_data["current_furniture_manipuland_count"] == 1
    assert summary_data["surface_object_ids"] == {"sideboard_top": ["coaster_0"]}
    coaster = next(
        obj for obj in summary_data["relevant_objects"] if obj["object_id"] == "coaster_0"
    )
    assert coaster["parent_surface_id"] == "sideboard_top"


def _make_critique(text: str) -> ManipulandCritiqueWithScores:
    score = CategoryScore(name="realism", grade=7, comment="stable result")
    return ManipulandCritiqueWithScores(
        critique=text,
        realism=score,
        functionality=CategoryScore("functionality", 7, "stable result"),
        layout=CategoryScore("layout", 7, "stable result"),
        holistic_completeness=CategoryScore(
            "holistic completeness", 7, "stable result"
        ),
        prompt_following=CategoryScore("prompt following", 7, "stable result"),
    )


@pytest.mark.asyncio
async def test_critic_session_is_cleared_when_scene_hash_changes() -> None:
    agent = object.__new__(_TestAgent)
    agent.critic_session = type("Session", (), {})()
    agent.critic_session.clear_session = AsyncMock()
    agent._critic_session_scene_hash = "scene-old"

    await agent._prepare_critic_session("scene-new")
    await agent._prepare_critic_session("scene-new")

    agent.critic_session.clear_session.assert_awaited_once()
    assert agent._critic_session_scene_hash == "scene-new"


def test_invalid_surface_ids_are_detected_against_current_furniture() -> None:
    surface = _FakeSurface("S_a")
    furniture = _make_object(
        object_id="sideboard_0",
        object_type=ObjectType.FURNITURE,
        name="Sideboard",
        xyz=(0.0, 0.0, 0.0),
        support_surfaces=[surface],
    )
    scene = _FakeScene([furniture])
    agent = object.__new__(_TestAgent)
    agent.scene = scene
    agent.current_furniture_id = "sideboard_0"

    invalid = agent._invalid_critic_surface_ids(
        _make_critique("The object is on S_missing, not S_a.")
    )

    assert invalid == {"S_missing"}


def test_same_critique_after_scene_change_is_marked_inconsistent() -> None:
    agent = object.__new__(_TestAgent)
    response = _make_critique("Monitor orientation is correct.")
    fingerprint = agent._critique_fingerprint(response)
    agent._last_critique_scene_hash = "scene-old"
    agent._last_critique_fingerprint = fingerprint

    reason = agent._critique_inconsistency_reason(
        scene_hash="scene-new",
        fingerprint=agent._critique_fingerprint(
            _make_critique("  monitor orientation is   correct.  ")
        ),
        invalid_surface_ids=set(),
    )

    assert reason == "critic returned the same critique fingerprint after the scene changed"
