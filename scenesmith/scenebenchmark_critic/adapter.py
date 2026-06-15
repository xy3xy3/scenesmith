"""Adapters from SceneSmith scene objects to SceneBenchmark-style case packs."""

from __future__ import annotations

import math

from typing import Any

import numpy as np

from pydrake.math import RollPitchYaw

from scenesmith.agent_utils.house import HouseScene
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
)
from scenesmith.scenebenchmark_critic.checks import build_checks


def room_scene_to_case_pack(
    scene: RoomScene,
    *,
    stage: str = "adhoc",
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    scene_geometry = _room_scene_geometry(scene)
    case_pack = {
        "schema_version": "scenesmith.scenebenchmark_critic.v1",
        "scene_id": f"{scene.room_id}:{stage}",
        "source_method": "scenesmith_online",
        "task_instruction": scene.text_description,
        "scene_geometry": scene_geometry,
        "checks": [],
    }
    case_pack["checks"] = build_checks(case_pack, metrics=metrics)
    return case_pack


def house_scene_to_case_pack(
    house: HouseScene,
    *,
    stage: str = "adhoc",
    metrics: tuple[str, ...] | list[str] | None = None,
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    rooms: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    for room_id, room in house.rooms.items():
        ox, oy = house._get_room_position(room_id)
        room_offset = np.array([ox, oy, 0.0])
        room_geom = _room_scene_geometry(
            room, room_offset=room_offset, include_object_types=include_object_types
        )
        rooms.extend(room_geom["rooms"])
        objects.extend(room_geom["objects"])
        relations.extend(room_geom["relations"])

    case_pack = {
        "schema_version": "scenesmith.scenebenchmark_critic.v1",
        "scene_id": f"house:{stage}",
        "source_method": "scenesmith_online",
        "task_instruction": (
            getattr(house.layout, "house_prompt", "")
            or getattr(house.layout, "prompt", "")
            or ""
        ),
        "scene_geometry": {
            "unit": "m",
            "rooms": rooms,
            "objects": objects,
            "relations": relations,
            "task_relation_graph": {},
        },
        "checks": [],
    }
    case_pack["checks"] = build_checks(case_pack, metrics=metrics)
    return case_pack


def _room_scene_geometry(
    scene: RoomScene,
    *,
    room_offset: np.ndarray | None = None,
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any]:
    offset = room_offset if room_offset is not None else np.zeros(3)
    objects: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    if scene.room_geometry and scene.room_geometry.floor:
        objects.append(_object_to_geometry(scene.room_geometry.floor, scene, offset))

    included_ids: set[str] = set()
    allowed_types = (
        set(include_object_types) if include_object_types is not None else None
    )
    for obj in scene.objects.values():
        if allowed_types is not None and obj.object_type not in allowed_types:
            continue
        objects.append(_object_to_geometry(obj, scene, offset))
        included_ids.add(str(obj.object_id))
        if obj.placement_info:
            relations.append(
                {
                    "relation_type": "placed_on_surface",
                    "subject_id": str(obj.object_id),
                    "target_surface_id": str(obj.placement_info.parent_surface_id),
                    "placement_method": obj.placement_info.placement_method,
                }
            )

    return {
        "unit": "m",
        "rooms": [_room_geometry_record(scene, offset)],
        "objects": objects,
        "relations": [
            relation
            for relation in relations
            if str(relation.get("subject_id")) in included_ids
        ],
        "task_relation_graph": {},
    }


def _room_geometry_record(scene: RoomScene, offset: np.ndarray) -> dict[str, Any]:
    geom = scene.room_geometry
    length = float(getattr(geom, "length", 0.0) or 0.0)
    width = float(getattr(geom, "width", 0.0) or 0.0)
    height = float(getattr(geom, "wall_height", 2.5) or 2.5)
    x0, x1 = -length / 2.0 + offset[0], length / 2.0 + offset[0]
    y0, y1 = -width / 2.0 + offset[1], width / 2.0 + offset[1]
    return {
        "id": scene.room_id,
        "room_type": scene.room_type,
        "bbox": {"min": [x0, y0, 0.0], "max": [x1, y1, height]},
        "floor_polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def _object_to_geometry(
    obj: SceneObject, scene: RoomScene, offset: np.ndarray
) -> dict[str, Any]:
    bounds = obj.compute_world_bounds()
    if bounds is None:
        world_min = np.array(obj.transform.translation(), dtype=float) + offset
        world_max = world_min.copy()
    else:
        world_min, world_max = bounds
        world_min = np.array(world_min, dtype=float) + offset
        world_max = np.array(world_max, dtype=float) + offset

    center = (world_min + world_max) / 2.0
    size = np.maximum(world_max - world_min, 0.0)
    category = _category_for_object(obj)
    yaw = math.degrees(RollPitchYaw(obj.transform.rotation()).yaw_angle())

    support_regions = []
    for surface in obj.support_surfaces:
        support_regions.append(_support_surface_to_region(surface, offset))

    record: dict[str, Any] = {
        "id": str(obj.object_id),
        "room": scene.room_id,
        "name": obj.name,
        "description": obj.description,
        "object_type": obj.object_type.value,
        "category": category,
        "category_norm": category,
        "yaw_deg": yaw,
        "bbox_world": {
            "center": center.tolist(),
            "size": size.tolist(),
            "min": world_min.tolist(),
            "max": world_max.tolist(),
        },
        "functional_hints": _functional_hints(obj),
        "metadata": dict(obj.metadata),
    }
    if support_regions:
        record["support_regions"] = support_regions
    if obj.placement_info:
        record["placement_info"] = {
            "parent_surface_id": str(obj.placement_info.parent_surface_id),
            "position_2d": obj.placement_info.position_2d.tolist(),
            "rotation_2d": float(obj.placement_info.rotation_2d),
            "placement_method": obj.placement_info.placement_method,
        }
    return record


def _support_surface_to_region(
    surface: SupportSurface, offset: np.ndarray
) -> dict[str, Any]:
    bbox_min = surface.bounding_box_min
    bbox_max = surface.bounding_box_max
    corners_local = np.array(
        [
            [bbox_min[0], bbox_min[1], 0.0],
            [bbox_max[0], bbox_min[1], 0.0],
            [bbox_max[0], bbox_max[1], 0.0],
            [bbox_min[0], bbox_max[1], 0.0],
        ]
    )
    corners_world = np.array([surface.transform @ corner for corner in corners_local])
    corners_world = corners_world + offset
    z_world = float((surface.transform @ np.array([0.0, 0.0, 0.0]))[2] + offset[2])
    return {
        "region_id": str(surface.surface_id),
        "support_kind": "top_surface",
        "height_world_z": z_world,
        "clearance_above_m": None,
        "access_type": "top",
        "area_m2": surface.area,
        "polygon_world_xy": corners_world[:, :2].tolist(),
        "bbox_local": {
            "min": surface.bounding_box_min.tolist(),
            "max": surface.bounding_box_max.tolist(),
        },
    }


def _category_for_object(obj: SceneObject) -> str:
    raw = (
        obj.metadata.get("category")
        or obj.metadata.get("category_norm")
        or obj.metadata.get("asset_category")
        or obj.name
    )
    return str(raw).strip().lower().replace(" ", "_")


def _functional_hints(obj: SceneObject) -> dict[str, Any]:
    text = " ".join(
        [
            obj.name.lower(),
            obj.description.lower(),
            " ".join(str(v).lower() for v in obj.metadata.values()),
        ]
    )
    categories: set[str] = set()
    if _contains_any(text, ("chair", "stool", "sofa", "couch", "bench", "seat")):
        categories.add("sittable")
    if _contains_any(text, ("bed", "mattress")):
        categories.add("sleepable")
    if _contains_any(
        text, ("table", "desk", "shelf", "nightstand", "counter", "cabinet")
    ):
        categories.add("supportable")
    if _contains_any(text, ("cabinet", "drawer", "wardrobe", "door")):
        categories.add("openable")
    if _contains_any(text, ("lamp", "light", "switch")):
        categories.add("toggleable")
    if obj.object_type in {ObjectType.MANIPULAND, ObjectType.THIN_COVERING}:
        categories.add("graspable")

    explicit = obj.metadata.get("functional_categories")
    if isinstance(explicit, list):
        categories.update(str(item) for item in explicit)

    return {
        "functional_categories": sorted(categories),
        "category_group": _category_group(obj, categories),
    }


def _category_group(obj: SceneObject, categories: set[str]) -> str:
    if obj.object_type == ObjectType.WALL_MOUNTED:
        return "decor"
    if obj.object_type == ObjectType.CEILING_MOUNTED:
        return "ceiling"
    if obj.object_type in {ObjectType.MANIPULAND, ObjectType.THIN_COVERING}:
        return "small_object"
    if "supportable" in categories:
        return "work_surface"
    return obj.object_type.value


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
