import unittest

from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from omegaconf import OmegaConf
from pydrake.all import RigidTransform

from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenesmith.furniture_agents.tools.snapping_helpers import (
    compute_snap_direction_mesh_to_mesh,
    snap_with_iterative_collision_check,
)


class FakeCollisionManager:
    instances = []

    def __init__(self):
        self.add_calls = []
        self.set_transform_calls = []
        self.min_distance_calls = 0
        FakeCollisionManager.instances.append(self)

    def add_object(self, name, mesh, transform=None):
        self.add_calls.append(name)

    def set_transform(self, name, transform):
        self.set_transform_calls.append(name)

    def min_distance_other(self, other_manager):
        self.min_distance_calls += 1
        if self.min_distance_calls < 3:
            return 1.0
        return 0.0


class TestSnappingHelpers(unittest.TestCase):

    def setUp(self):
        FakeCollisionManager.instances.clear()

    def test_iterative_collision_check_reuses_registered_collision_objects(self):
        cfg = OmegaConf.create(
            {
                "snap_to_object": {
                    "iterative_snap_step_m": 0.1,
                    "max_snap_distance_m": 1.0,
                }
            }
        )
        collision_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        obj = SceneObject(
            object_id=UniqueID("obj_0"),
            object_type=ObjectType.FURNITURE,
            name="obj",
            description="object",
            transform=RigidTransform(p=[0.0, 0.0, 0.0]),
            geometry_path=Path("obj.gltf"),
            sdf_path=Path("obj.sdf"),
        )
        target = SceneObject(
            object_id=UniqueID("target_0"),
            object_type=ObjectType.FURNITURE,
            name="target",
            description="target",
            transform=RigidTransform(p=[5.0, 0.0, 0.0]),
            geometry_path=Path("target.gltf"),
            sdf_path=Path("target.sdf"),
        )

        with (
            patch(
                "scenesmith.furniture_agents.tools.snapping_helpers.load_object_collision_geometry",
                side_effect=[[collision_mesh.copy()], [collision_mesh.copy()]],
            ),
            patch(
                "scenesmith.furniture_agents.tools.snapping_helpers.trimesh.collision.CollisionManager",
                FakeCollisionManager,
            ),
        ):
            movement_vector, distance = snap_with_iterative_collision_check(
                obj=obj,
                target=target,
                direction=np.array([1.0, 0.0, 0.0]),
                cfg=cfg,
            )

        self.assertEqual(len(FakeCollisionManager.instances), 2)
        target_manager, obj_manager = FakeCollisionManager.instances

        self.assertEqual(target_manager.add_calls, ["target_0"])
        self.assertEqual(obj_manager.add_calls, ["obj_0"])
        self.assertGreaterEqual(len(obj_manager.set_transform_calls), 3)
        self.assertAlmostEqual(distance, 0.2)
        np.testing.assert_allclose(movement_vector, np.array([0.2, 0.0, 0.0]))

    def test_compute_snap_direction_prefers_collision_geometry(self):
        cfg = OmegaConf.create({"snap_to_object": {"max_sample_vertices": 2000}})
        obj = SceneObject(
            object_id=UniqueID("obj_0"),
            object_type=ObjectType.FURNITURE,
            name="obj",
            description="object",
            transform=RigidTransform(p=[0.0, 0.0, 0.0]),
            geometry_path=Path("obj.gltf"),
            sdf_path=Path("obj.sdf"),
        )
        target = SceneObject(
            object_id=UniqueID("target_0"),
            object_type=ObjectType.FURNITURE,
            name="target",
            description="target",
            transform=RigidTransform(p=[2.0, 0.0, 0.0]),
            geometry_path=Path("target.gltf"),
            sdf_path=Path("target.sdf"),
        )
        obj_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        target_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        def fake_closest_point(mesh, points):
            closest = points + np.array([2.0, 0.0, 0.0])
            distances = np.full(len(points), 2.0)
            return closest, distances, None

        with (
            patch(
                "scenesmith.furniture_agents.tools.snapping_helpers.load_object_collision_geometry",
                side_effect=[[obj_mesh.copy()], [target_mesh.copy()]],
            ),
            patch(
                "scenesmith.furniture_agents.tools.snapping_helpers.trimesh.proximity.closest_point",
                side_effect=fake_closest_point,
            ),
            patch(
                "scenesmith.furniture_agents.tools.snapping_helpers.trimesh.load",
                side_effect=AssertionError("visual mesh load should not be used"),
            ),
        ):
            direction = compute_snap_direction_mesh_to_mesh(
                obj=obj,
                target=target,
                cfg=cfg,
            )

        np.testing.assert_allclose(direction, np.array([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
