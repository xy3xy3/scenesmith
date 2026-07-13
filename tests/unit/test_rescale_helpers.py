"""Tests for shared object rescaling behavior."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pydrake.math import RigidTransform

from scenesmith.agent_utils import rescale_helpers
from scenesmith.agent_utils.room import ObjectType, PlacementInfo, SceneObject, UniqueID


def test_rescale_supported_object_preserves_world_bottom(
    tmp_path: Path, monkeypatch
) -> None:
    sdf_path = tmp_path / "pillow.sdf"
    sdf_path.write_text("<sdf version='1.7'/>")
    pillow = SceneObject(
        object_id=UniqueID("pillow_0"),
        object_type=ObjectType.MANIPULAND,
        name="throw pillow",
        description="throw pillow",
        transform=RigidTransform(p=[0.0, 0.0, 1.0]),
        sdf_path=sdf_path,
        bbox_min=np.array([-0.2, -0.1, -0.15]),
        bbox_max=np.array([0.2, 0.1, 0.15]),
        placement_info=PlacementInfo(
            parent_surface_id=UniqueID("S_0"),
            position_2d=np.array([0.0, 0.0]),
            rotation_2d=0.0,
        ),
    )
    scene = SimpleNamespace(
        objects={pillow.object_id: pillow},
        get_object=lambda object_id: pillow if str(object_id) == "pillow_0" else None,
    )
    monkeypatch.setattr(rescale_helpers, "rescale_sdf", lambda **_kwargs: None)
    before_bottom = pillow.compute_world_bounds()[0][2]

    result = rescale_helpers.rescale_object_common(
        scene=scene,
        object_id="pillow_0",
        scale_factor=1.5,
        object_type_name="manipuland",
    )

    after_bottom = pillow.compute_world_bounds()[0][2]
    assert result.success is True
    assert after_bottom == before_bottom
    assert pillow.transform.translation()[2] > 1.0


def test_rescale_unplaced_object_keeps_transform(
    tmp_path: Path, monkeypatch
) -> None:
    sdf_path = tmp_path / "wall_art.sdf"
    sdf_path.write_text("<sdf version='1.7'/>")
    wall_art = SceneObject(
        object_id=UniqueID("wall_art_0"),
        object_type=ObjectType.WALL_MOUNTED,
        name="wall art",
        description="wall art",
        transform=RigidTransform(p=[0.0, 0.0, 1.5]),
        sdf_path=sdf_path,
        bbox_min=np.array([-0.5, -0.05, -0.4]),
        bbox_max=np.array([0.5, 0.05, 0.4]),
    )
    scene = SimpleNamespace(
        objects={wall_art.object_id: wall_art},
        get_object=lambda object_id: (
            wall_art if str(object_id) == "wall_art_0" else None
        ),
    )
    monkeypatch.setattr(rescale_helpers, "rescale_sdf", lambda **_kwargs: None)

    result = rescale_helpers.rescale_object_common(
        scene=scene,
        object_id="wall_art_0",
        scale_factor=1.5,
        object_type_name="wall-mounted object",
    )

    assert result.success is True
    assert wall_art.transform.translation()[2] == 1.5
