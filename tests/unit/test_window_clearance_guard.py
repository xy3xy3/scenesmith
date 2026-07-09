"""Tests for manipuland window clearance pre-placement guard."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenesmith.manipuland_agents.tools.window_clearance_guard import (
    window_clearance_placement_error,
)


def test_window_clearance_guard_rejects_tall_object_in_window_zone() -> None:
    scene = _scene_with_window()
    obj = _manipuland("vase_0", z_center=0.94, height=0.12)

    error = window_clearance_placement_error(scene=scene, obj=obj)

    assert error is not None
    assert "window_0" in error
    assert "exceeds window sill" in error


def test_window_clearance_guard_allows_low_object_in_window_zone() -> None:
    scene = _scene_with_window()
    obj = _manipuland("coaster_0", z_center=0.885, height=0.01)

    assert window_clearance_placement_error(scene=scene, obj=obj) is None


def test_window_clearance_guard_allows_tall_object_outside_window_zone() -> None:
    scene = _scene_with_window()
    obj = _manipuland("vase_0", z_center=0.94, height=0.12, x=2.0)

    assert window_clearance_placement_error(scene=scene, obj=obj) is None


def _scene_with_window() -> RoomScene:
    opening = SimpleNamespace(
        opening_id="window_0",
        opening_type="window",
        sill_height=0.90,
        clearance_bbox_min=np.array([-0.5, -0.5, 0.0]),
        clearance_bbox_max=np.array([0.5, 0.5, 2.0]),
    )
    room_geometry = SimpleNamespace(openings=[opening])
    return RoomScene(
        room_geometry=room_geometry,
        scene_dir=Path("/tmp"),
        text_description="test room",
    )


def _manipuland(
    object_id: str, *, z_center: float, height: float, x: float = 0.0
) -> SceneObject:
    half_height = height / 2.0
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.MANIPULAND,
        name=object_id,
        description=object_id,
        transform=RigidTransform(p=[x, 0.0, z_center]),
        bbox_min=np.array([-0.05, -0.05, -half_height]),
        bbox_max=np.array([0.05, 0.05, half_height]),
    )
