from __future__ import annotations

import ast
import importlib
import json
import pkgutil

from pathlib import Path
from typing import Any

import lxml.etree as ET
import numpy as np
import pytest
import yaml

from omegaconf import OmegaConf
from pydrake.math import RigidTransform, RollPitchYaw

import scenesmith.agent_utils.base_stateful_agent as base_stateful_agent

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.house import (
    ClearanceOpeningData,
    HouseLayout,
    HouseScene,
    PlacedRoom,
    RoomGeometry,
    RoomSpec,
)
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.agent_utils.scoring import CategoryScore, FurnitureCritiqueWithScores
from scenesmith.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
)
from scenesmith.scenebenchmark_critic import (
    CriticConfig,
    annotate_room_scene,
    evaluate_room_scene,
    format_prompt_context,
    write_house_stage_report,
    write_room_stage_report,
)
from scenesmith.scenebenchmark_critic.prompt_context import (
    filter_prompt_results_for_agent,
    format_agent_prompt_context,
)
from scenesmith.scenebenchmark_critic.adapter import (
    house_scene_to_case_pack,
    room_scene_to_case_pack,
)
from scenesmith.scenebenchmark_critic.checks import build_checks
from scenesmith.scenebenchmark_critic.config import critic_config_from_any
from scenesmith.scenebenchmark_critic.orientation_contracts import (
    CONTRACT_CHECK_SOURCE,
    stabilize_orientation_contracts,
)
from scenesmith.scenebenchmark_critic.reports import format_markdown_report
from scenesmith.scenebenchmark_critic.vendor.rules import (
    aggregate_results,
    run_case_pack_checks,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.models import (
    FunctionalDependencyProposal,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency import (
    proposer as fd_proposer,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.relations import (
    _relation_target_is_valid,
)


def _box_object(
    object_id: str,
    name: str,
    object_type: ObjectType,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_deg: float = 0.0,
) -> SceneObject:
    sx, sy, sz = size
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=name,
        description=name,
        transform=RigidTransform(
            RollPitchYaw(0.0, 0.0, np.deg2rad(yaw_deg)).ToRotationMatrix(),
            list(center),
        ),
        bbox_min=np.array([-sx / 2.0, -sy / 2.0, -sz / 2.0]),
        bbox_max=np.array([sx / 2.0, sy / 2.0, sz / 2.0]),
    )


def _benchmark_obj(
    object_id: str,
    category: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    yaw: float = 0.0,
) -> dict[str, Any]:
    cx, cy, cz = center
    sx, sy, sz = size
    return {
        "id": object_id,
        "category": category,
        "category_norm": category,
        "room": "room",
        "yaw_deg": yaw,
        "bbox_world": {
            "center": [cx, cy, cz],
            "size": [sx, sy, sz],
            "min": [cx - sx / 2, cy - sy / 2, cz - sz / 2],
            "max": [cx + sx / 2, cy + sy / 2, cz + sz / 2],
        },
        "functional_hints": {"functional_categories": []},
    }


def _benchmark_case_pack(
    objects: list[dict[str, Any]], checks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "scene_id": "scene",
        "room_type": "bedroom",
        "task_instruction": "test scene",
        "scene_geometry": {
            "unit": "m",
            "rooms": [
                {
                    "id": "room",
                    "bbox": {"min": [0, 0, 0], "max": [5, 5, 3]},
                    "floor_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
                }
            ],
            "objects": objects,
            "relations": [],
            "task_relation_graph": {},
        },
        "checks": checks or [],
    }


def _run_direct_case_pack(
    case_pack: dict[str, Any],
    *,
    metrics: list[str],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return run_case_pack_checks(
        case_pack,
        {
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": metrics,
                **(extra or {}),
            }
        },
    )


def _scene(tmp_path: Path) -> RoomScene:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sdf_path = tmp_path / "room.sdf"
    sdf_path.write_text("<sdf version='1.7'><world name='default'/></sdf>")
    floor = _box_object(
        "floor_0",
        "floor",
        ObjectType.FLOOR,
        center=(0.0, 0.0, -0.05),
        size=(6.0, 4.0, 0.1),
    )
    geometry = RoomGeometry(
        sdf_tree=ET.ElementTree(ET.Element("sdf")),
        sdf_path=sdf_path,
        walls=[],
        floor=floor,
        width=4.0,
        length=6.0,
        wall_height=3.0,
    )
    scene = RoomScene(
        room_geometry=geometry,
        scene_dir=tmp_path,
        room_id="main",
        room_type="bedroom",
        text_description="A bedroom with a table and mug.",
    )
    table = _box_object(
        "table_0",
        "nightstand table",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.35),
        size=(1.0, 1.0, 0.7),
    )
    surface = SupportSurface(
        surface_id=UniqueID("S_0"),
        bounding_box_min=np.array([-0.5, -0.5, 0.0]),
        bounding_box_max=np.array([0.5, 0.5, 0.0]),
        transform=RigidTransform(p=[0.0, 0.0, 0.7]),
    )
    table.support_surfaces = [surface]
    mug = _box_object(
        "mug_0",
        "coffee mug",
        ObjectType.MANIPULAND,
        center=(0.0, 0.0, 0.8),
        size=(0.18, 0.18, 0.2),
    )
    mug.placement_info = PlacementInfo(
        parent_surface_id=UniqueID("S_0"),
        position_2d=np.array([0.0, 0.0]),
        rotation_2d=0.0,
    )
    scene.add_object(table)
    scene.add_object(mug)
    return scene


def _house(tmp_path: Path) -> HouseScene:
    room = _scene(tmp_path / "room_main")
    layout = HouseLayout(
        house_prompt="A house with one bedroom.",
        room_specs=[
            RoomSpec(
                room_id="main",
                room_type="bedroom",
                prompt=room.text_description,
                width=4.0,
                length=6.0,
            )
        ],
        house_dir=tmp_path,
        placed_rooms=[
            PlacedRoom(room_id="main", position=(-3.0, -2.0), width=6.0, depth=4.0)
        ],
    )
    return HouseScene(layout=layout, rooms={"main": room})


def test_room_scene_adapter_builds_geometry_and_checks(tmp_path: Path) -> None:
    case_pack = room_scene_to_case_pack(
        _scene(tmp_path),
        stage="final_scene",
        metrics=["spatial_accessibility", "functional_dependency"],
    )

    room = case_pack["scene_geometry"]["rooms"][0]
    assert case_pack["room_type"] == "bedroom"
    assert room["floor_polygon"] == [[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]]
    table = next(
        obj for obj in case_pack["scene_geometry"]["objects"] if obj["id"] == "table_0"
    )
    assert table["support_regions"][0]["region_id"] == "S_0"
    assert table["functional_hints"]["support_region_summary"] == {
        "count": 1,
        "support_kinds": ["top_surface"],
        "source": "scenesmith_support_surface",
    }
    assert any(
        check["metric"] == "functional_dependency"
        and check["subject_id"] == "mug_0"
        and check["target_ids"] == ["table_0"]
        for check in case_pack["checks"]
    )
    relation = case_pack["scene_geometry"]["relations"][0]
    assert relation["subject"] == "mug_0"
    assert relation["object"] == "table_0"
    assert relation["target_surface_id"] == "S_0"


def test_room_scene_adapter_uses_scenebenchmark_demo_category_aliases(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    beanbag = _box_object(
        "beanbag_0",
        "beanbag chair",
        ObjectType.FURNITURE,
        center=(-1.0, 0.0, 0.3),
        size=(0.8, 0.8, 0.6),
    )
    fridge = _box_object(
        "fridge_0",
        "small fridge",
        ObjectType.FURNITURE,
        center=(1.0, 0.0, 0.9),
        size=(0.7, 0.7, 1.8),
    )
    nightstand = _box_object(
        "bedroom_nightstand_1_f0_c",
        "asset instance",
        ObjectType.FURNITURE,
        center=(0.0, 1.0, 0.35),
        size=(0.5, 0.5, 0.7),
    )
    scene.add_object(beanbag)
    scene.add_object(fridge)
    scene.add_object(nightstand)

    case_pack = room_scene_to_case_pack(
        scene, stage="final_scene", metrics=["spatial_accessibility"]
    )
    objects = {obj["id"]: obj for obj in case_pack["scene_geometry"]["objects"]}

    beanbag_hints = objects["beanbag_0"]["functional_hints"]
    assert objects["beanbag_0"]["category_norm"] == "beanbag_chair"
    assert "sittable" in beanbag_hints["functional_categories"]
    assert beanbag_hints["category_group"] == "seating"
    assert beanbag_hints["front_hint"] == "front"
    assert beanbag_hints["target_relation"] == ["desk", "table"]
    assert beanbag_hints["metric_relevance"]["spatial_accessibility"] == 0.8
    assert beanbag_hints["mobility_class"] == "movable"
    assert beanbag_hints["accessibility_policy"] == "optional"

    fridge_hints = objects["fridge_0"]["functional_hints"]
    assert objects["fridge_0"]["category_norm"] == "refrigerator"
    assert {"containable", "openable"} <= set(fridge_hints["functional_categories"])
    assert fridge_hints["category_group"] == "appliance_storage"
    assert fridge_hints["front_hint"] == "front"
    assert fridge_hints["target_relation"] == ["wall", "clear_space"]
    assert fridge_hints["metric_relevance"]["spatial_accessibility"] == 1.0

    nightstand_hints = objects["bedroom_nightstand_1_f0_c"]["functional_hints"]
    assert objects["bedroom_nightstand_1_f0_c"]["category_norm"] == "nightstand"
    assert {"openable", "supportable"} <= set(nightstand_hints["functional_categories"])
    assert nightstand_hints["category_group"] == "storage_surface"
    assert "bedside table" in nightstand_hints["category_keywords"]
    assert nightstand_hints["front_hint"] == "top"
    assert nightstand_hints["target_relation"] == ["wall", "clear_space"]
    assert nightstand_hints["metric_relevance"]["spatial_accessibility"] == 1.0

    checked_subjects = {
        check["subject_id"]
        for check in case_pack["checks"]
        if check["metric"] == "spatial_accessibility"
    }
    assert "beanbag_0" not in checked_subjects
    assert {"fridge_0", "bedroom_nightstand_1_f0_c"} <= checked_subjects


def test_room_scene_adapter_preserves_external_dependency_annotations(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    chair = _box_object(
        "chair_0",
        "desk chair",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.45),
        size=(0.6, 0.6, 0.9),
    )
    chair.metadata["functional_hints"] = {
        "mobility_class": "movable",
        "accessibility_policy": "optional",
        "access_sides": ["front"],
        "orientation_dependencies": [
            {
                "relation_type": "seat_faces_surface",
                "target_kind": "object",
                "target_category": ["desk", "table"],
            }
        ],
    }
    scene.add_object(chair)

    case_pack = room_scene_to_case_pack(
        scene,
        stage="final_scene",
        metrics=["spatial_accessibility", "functional_dependency"],
    )
    chair_record = next(
        obj for obj in case_pack["scene_geometry"]["objects"] if obj["id"] == "chair_0"
    )

    hints = chair_record["functional_hints"]
    assert hints["mobility_class"] == "movable"
    assert hints["accessibility_policy"] == "optional"
    assert hints["access_sides"] == ["front"]
    assert hints["orientation_dependencies"][0]["relation_type"] == "seat_faces_surface"
    assert all(
        check["subject_id"] != "chair_0"
        for check in case_pack["checks"]
        if check["metric"] == "spatial_accessibility"
    )


def test_room_scene_adapter_exports_scenebenchmark_geometry_fields(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.room_geometry.openings.append(
        ClearanceOpeningData(
            opening_id="door_main",
            opening_type="door",
            wall_direction="south",
            center_world=[0.0, -2.0, 1.0],
            width=0.9,
            sill_height=0.0,
            height=2.0,
            clearance_bbox_min=[-0.45, -2.0, 0.0],
            clearance_bbox_max=[0.45, -1.2, 2.0],
            wall_start=[-3.0, -2.0],
            wall_end=[3.0, -2.0],
            position_along_wall=3.0,
        )
    )

    case_pack = room_scene_to_case_pack(scene, stage="final_scene")
    geometry = case_pack["scene_geometry"]
    table = next(obj for obj in geometry["objects"] if obj["id"] == "table_0")

    assert len(table["footprint_world"]) == 4
    assert table["interaction_faces"]
    assert table["nav_obstacle_class"] == "blocking"
    assert geometry["scene_shell"]["doors"][0]["opening_id"] == "door_main"
    assert geometry["scene_shell"]["doors"][0]["center"] == [0.0, -2.0, 1.0]


def test_room_scene_adapter_normalizes_workstation_categories(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    monitor = _box_object(
        "computer_monitor_0",
        "computer monitor",
        ObjectType.MANIPULAND,
        center=(0.0, 0.0, 0.9),
        size=(0.45, 0.08, 0.32),
    )
    mouse = _box_object(
        "wireless_mouse_0",
        "wireless mouse",
        ObjectType.MANIPULAND,
        center=(0.3, 0.0, 0.72),
        size=(0.08, 0.13, 0.04),
    )
    tablet = _box_object(
        "tablet_computer_0",
        "tablet computer",
        ObjectType.MANIPULAND,
        center=(-0.3, 0.0, 0.72),
        size=(0.22, 0.14, 0.02),
    )
    scene.add_object(monitor)
    scene.add_object(mouse)
    scene.add_object(tablet)

    case_pack = room_scene_to_case_pack(scene, stage="final_scene")
    objects = {obj["id"]: obj for obj in case_pack["scene_geometry"]["objects"]}

    assert objects["computer_monitor_0"]["category_norm"] == "monitor"
    assert (
        objects["computer_monitor_0"]["functional_hints"]["category_group"] == "media"
    )
    assert objects["wireless_mouse_0"]["category_norm"] == "mouse"
    assert objects["tablet_computer_0"]["category_norm"] == "tablet_computer"


def test_computer_peripheral_faces_screen_accepts_workstation_categories() -> None:
    monitor = _benchmark_obj(
        "computer_monitor_0",
        "monitor",
        (0.0, 0.0, 0.9),
        (0.45, 0.08, 0.32),
    )
    monitor["functional_hints"]["category_group"] = "media"
    mouse = _benchmark_obj(
        "wireless_mouse_0",
        "mouse",
        (0.3, 0.0, 0.72),
        (0.08, 0.13, 0.04),
    )
    laptop = _benchmark_obj(
        "laptop_0",
        "laptop",
        (-0.3, 0.0, 0.75),
        (0.32, 0.22, 0.03),
    )
    laptop["functional_hints"]["category_group"] = "small_object"

    assert _relation_target_is_valid(mouse, monitor, "computer_peripheral_faces_screen")
    assert _relation_target_is_valid(mouse, laptop, "computer_peripheral_faces_screen")


def test_furniture_faces_furniture_requires_directional_subject() -> None:
    sofa = _benchmark_obj("sofa_1", "sofa", (0.0, 0.0, 0.4), (1.6, 0.8, 0.8))
    coffee_table = _benchmark_obj(
        "coffee_table_1", "coffee_table", (0.0, 0.9, 0.25), (1.0, 0.5, 0.5)
    )
    tv_stand = _benchmark_obj(
        "tv_stand_1", "tv_stand", (0.0, 1.2, 0.3), (1.2, 0.35, 0.6)
    )
    television = _benchmark_obj(
        "wall_mounted_television_1", "television", (0.0, 1.8, 1.2), (1.0, 0.08, 0.6)
    )

    assert _relation_target_is_valid(sofa, coffee_table, "furniture_faces_furniture")
    assert _relation_target_is_valid(sofa, tv_stand, "furniture_faces_furniture")
    assert _relation_target_is_valid(television, sofa, "furniture_faces_furniture")
    assert not _relation_target_is_valid(
        coffee_table, sofa, "furniture_faces_furniture"
    )


def test_room_scene_adapter_respects_nonfunctional_asset_annotation(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    decorative_cabinet = _box_object(
        "cabinet_0",
        "decorative cabinet facade",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.6),
        size=(0.8, 0.4, 1.2),
    )
    decorative_cabinet.metadata.update(
        {
            "category_norm": "cabinet",
            "functional_categories": ["openable", "supportable"],
            "classification_source": "asset_annotation",
            "benchmark_relevance": "decorative",
        }
    )
    decorative_table = _box_object(
        "table_0",
        "decorative table sculpture",
        ObjectType.FURNITURE,
        center=(1.2, 0.0, 0.35),
        size=(0.8, 0.4, 0.7),
    )
    decorative_table.metadata.update(
        {
            "category_norm": "table",
            "functional_categories": ["supportable"],
            "asset_annotation_source": "unit_test",
            "benchmark_relevance": "decorative",
        }
    )
    scene.add_object(decorative_cabinet)
    scene.add_object(decorative_table)

    case_pack = room_scene_to_case_pack(scene, stage="scene_after_furniture")
    objects = {obj["id"]: obj for obj in case_pack["scene_geometry"]["objects"]}
    cabinet_record = objects["cabinet_0"]
    table_record = objects["table_0"]

    assert cabinet_record["functional_hints"]["functional_categories"] == []
    assert table_record["functional_hints"]["functional_categories"] == []
    assert not any(
        check["metric"] == "spatial_accessibility"
        and check["subject_id"] in {"cabinet_0", "table_0"}
        for check in case_pack["checks"]
    )


def test_room_scene_adapter_preserves_scenebenchmark_functional_hints(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    shelf = _box_object(
        "shelf_0",
        "high shelf",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 1.6),
        size=(0.6, 0.6, 3.2),
    )
    shelf.metadata.update(
        {
            "functional_categories": ["supportable"],
            "category_keywords": ["storage shelf", "display shelf"],
            "access_type": {"primary": "top"},
            "access_direction": [0.0, -1.0, 0.0],
            "interaction_height_m": {"supportable": 2.8},
            "operation_space": {"front_clearance_m": 0.6},
            "target_relation": ["graspable_object"],
            "explicit_target_relation": ["book"],
            "metric_relevance": {"spatial_accessibility": 1.0},
            "classification_confidence": 0.91,
            "classification_reason": "asset annotation fixture",
            "asset_annotation_source": "unit_test",
            "classification_source": "asset_annotation",
            "front_hint": "left",
        }
    )
    wall_clock = _box_object(
        "clock_0",
        "wall clock",
        ObjectType.WALL_MOUNTED,
        center=(1.0, 0.0, 1.7),
        size=(0.2, 0.08, 0.2),
    )
    scene.add_object(shelf)
    scene.add_object(wall_clock)

    case_pack = room_scene_to_case_pack(scene, stage="scene_after_furniture")
    objects = {obj["id"]: obj for obj in case_pack["scene_geometry"]["objects"]}
    shelf_record = objects["shelf_0"]
    clock_record = objects["clock_0"]

    assert shelf_record["interaction_height_m"] == {"supportable": 2.8}
    assert shelf_record["functional_hints"]["interaction_height_m"] == {
        "supportable": 2.8
    }
    assert shelf_record["functional_hints"]["category_keywords"] == [
        "storage shelf",
        "display shelf",
    ]
    assert shelf_record["functional_hints"]["access_type"] == {"primary": "top"}
    assert shelf_record["functional_hints"]["access_direction"] == [0.0, -1.0, 0.0]
    assert shelf_record["functional_hints"]["operation_space"] == {
        "front_clearance_m": 0.6
    }
    assert shelf_record["functional_hints"]["target_relation"] == ["graspable_object"]
    assert shelf_record["functional_hints"]["explicit_target_relation"] == ["book"]
    assert shelf_record["functional_hints"]["metric_relevance"] == {
        "spatial_accessibility": 1.0
    }
    assert shelf_record["functional_hints"]["classification_confidence"] == 0.91
    assert shelf_record["functional_hints"]["classification_reason"] == (
        "asset annotation fixture"
    )
    assert shelf_record["functional_hints"]["asset_annotation_source"] == "unit_test"
    assert shelf_record["functional_hints"]["classification_source"] == (
        "asset_annotation"
    )
    assert shelf_record["functional_hints"]["front_hint"] == "left"
    assert shelf_record["functional_hints"]["scene_object_type"] == "furniture"
    assert clock_record["functional_hints"]["scene_object_type"] == "wall_mounted"
    shelf_sa_check = next(
        check
        for check in case_pack["checks"]
        if check["metric"] == "spatial_accessibility"
        and check["subject_id"] == "shelf_0"
    )
    assert shelf_sa_check["priority_weight"] == 1.0
    assert shelf_sa_check["evidence_refs"] == ["scene_geometry"]


def test_room_scene_adapter_uses_candidate_affordances_as_functional_categories(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    fixture = _box_object(
        "fixture_0",
        "plain fixture",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.6),
        size=(0.8, 0.4, 1.2),
    )
    fixture.metadata.update(
        {
            "category_norm": "dataset_specific_fixture",
            "candidate_affordances": ["openable"],
        }
    )
    annotated_fixture = _box_object(
        "annotated_fixture_0",
        "dataset fixture",
        ObjectType.FURNITURE,
        center=(1.2, 0.0, 0.6),
        size=(0.8, 0.4, 1.2),
    )
    annotated_fixture.metadata.update(
        {
            "category_norm": "dataset_specific_fixture",
            "affordances": ["supportable"],
            "front_face": "left",
            "asset_annotation_source": "unit_test",
        }
    )
    scene.add_object(fixture)
    scene.add_object(annotated_fixture)

    case_pack = room_scene_to_case_pack(
        scene,
        stage="scene_after_furniture",
        metrics=["spatial_accessibility"],
    )
    fixture_record = next(
        obj
        for obj in case_pack["scene_geometry"]["objects"]
        if obj["id"] == "fixture_0"
    )
    annotated_record = next(
        obj
        for obj in case_pack["scene_geometry"]["objects"]
        if obj["id"] == "annotated_fixture_0"
    )
    sa_check = next(
        check
        for check in case_pack["checks"]
        if check["metric"] == "spatial_accessibility"
        and check["subject_id"] == "fixture_0"
    )
    fixture_front = next(
        face for face in fixture_record["interaction_faces"] if face["name"] == "front"
    )
    annotated_front = next(
        face
        for face in annotated_record["interaction_faces"]
        if face["name"] == "front"
    )

    assert fixture_record["functional_hints"]["functional_categories"] == ["openable"]
    assert fixture_record["functional_hints"]["candidate_affordances"] == ["openable"]
    assert fixture_record["functional_hints"]["anchor_type"] == "front_access"
    assert fixture_front["affordances"] == ["openable"]
    assert annotated_record["functional_hints"]["functional_categories"] == [
        "supportable"
    ]
    assert annotated_record["functional_hints"]["candidate_affordances"] == [
        "supportable"
    ]
    assert annotated_record["functional_hints"]["front_hint"] == "left"
    assert annotated_record["functional_hints"]["anchor_type"] == "top_surface"
    assert annotated_front["affordances"] == ["supportable"]
    assert sa_check["affordance"] == "openable"


def test_room_scene_adapter_infers_scenebenchmark_surface_profiles(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    sideboard = _box_object(
        "sideboard_0",
        "sideboard buffet",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.45),
        size=(1.2, 0.4, 0.9),
    )
    tv_stand = _box_object(
        "tv_stand_0",
        "tv stand",
        ObjectType.FURNITURE,
        center=(1.6, 0.0, 0.3),
        size=(1.0, 0.35, 0.6),
    )
    plant = _box_object(
        "plant_0",
        "plant",
        ObjectType.MANIPULAND,
        center=(0.0, 0.0, 0.95),
        size=(0.18, 0.18, 0.25),
    )
    scene.add_object(sideboard)
    scene.add_object(tv_stand)
    scene.add_object(plant)

    case_pack = room_scene_to_case_pack(scene, stage="scene_after_furniture")
    objects = {obj["id"]: obj for obj in case_pack["scene_geometry"]["objects"]}

    assert objects["sideboard_0"]["functional_hints"]["category_group"] == (
        "storage_surface"
    )
    assert (
        "supportable"
        in objects["sideboard_0"]["functional_hints"]["functional_categories"]
    )
    assert objects["sideboard_0"]["object_function_profile"]["can_support_top"] is True
    assert (
        objects["sideboard_0"]["object_function_profile"]["has_internal_shelf"] is True
    )
    assert objects["sideboard_0"]["object_function_profile"]["is_work_surface"] is True
    assert objects["tv_stand_0"]["object_function_profile"]["is_media_target"] is True
    assert objects["tv_stand_0"]["object_function_profile"]["can_support_top"] is True
    assert objects["plant_0"]["object_function_profile"]["is_small_placeable"] is True


def test_room_scene_adapter_uses_access_direction_for_interaction_face(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    cabinet = _box_object(
        "cabinet_0",
        "storage cabinet",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.5),
        size=(0.8, 0.6, 1.0),
        yaw_deg=0.0,
    )
    cabinet.metadata.update(
        {
            "category_norm": "cabinet",
            "functional_categories": ["openable"],
            "access_direction": [0.0, -1.0, 0.0],
            "classification_source": "asset_annotation",
        }
    )
    scene.add_object(cabinet)

    case_pack = room_scene_to_case_pack(scene, stage="scene_after_furniture")
    cabinet_record = next(
        obj
        for obj in case_pack["scene_geometry"]["objects"]
        if obj["id"] == "cabinet_0"
    )
    front_face = next(
        face for face in cabinet_record["interaction_faces"] if face["name"] == "front"
    )

    assert cabinet_record["functional_hints"]["front_hint"] == "right"
    assert front_face["normal_xy"] == pytest.approx([0.0, -1.0])
    assert front_face["center"][1] == pytest.approx(-0.3)


def test_room_scene_adapter_turns_metadata_dependencies_into_fd_checks(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    mug = scene.objects[UniqueID("mug_0")]
    mug.placement_info = None
    mug.metadata["functional_dependencies"] = [
        {
            "relation_type": "object_on_support",
            "target_surface_id": "S_0",
            "reason": "asset annotation says mug rests on this surface",
        }
    ]

    case_pack = room_scene_to_case_pack(
        scene,
        stage="scene_after_furniture",
        metrics=["functional_dependency"],
    )

    metadata_relation = next(
        relation
        for relation in case_pack["scene_geometry"]["relations"]
        if relation.get("annotation_source") == "metadata"
    )
    fd_check = next(
        check
        for check in case_pack["checks"]
        if check["metric"] == "functional_dependency" and check["subject_id"] == "mug_0"
    )

    assert metadata_relation["target_ids"] == ["table_0"]
    assert metadata_relation["target_surface_id"] == "S_0"
    assert fd_check["target_ids"] == ["table_0"]
    assert fd_check["relation_type"] == "object_on_support"
    assert fd_check["expected_use"] == (
        "small object is supported by an appropriate surface"
    )
    assert fd_check["priority_weight"] == 0.7
    assert fd_check["evidence_refs"] == ["scene_geometry", "object_metadata"]
    assert fd_check["evidence"]["annotation_source"] == "metadata"


def test_room_scene_adapter_reads_nested_functional_hint_dependency(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    mug = scene.objects[UniqueID("mug_0")]
    mug.placement_info = None
    mug.metadata["functional_hints"] = {
        "functional_dependencies": [
            {
                "relation_type": "object_on_support",
                "target_ids": ["table_0"],
                "reason": "asset hint relation",
            }
        ]
    }

    case_pack = room_scene_to_case_pack(
        scene,
        stage="scene_after_furniture",
        metrics=["functional_dependency"],
    )

    metadata_relation = next(
        relation
        for relation in case_pack["scene_geometry"]["relations"]
        if relation.get("annotation_source") == "metadata"
    )
    fd_check = next(
        check
        for check in case_pack["checks"]
        if check["metric"] == "functional_dependency" and check["subject_id"] == "mug_0"
    )

    assert metadata_relation["target_ids"] == ["table_0"]
    assert fd_check["target_ids"] == ["table_0"]
    assert fd_check["relation_type"] == "object_on_support"
    assert fd_check["expected_use"] == (
        "small object is supported by an appropriate surface"
    )
    assert fd_check["evidence"]["reason"] == "asset hint relation"


def test_room_scene_adapter_turns_explicit_target_relation_into_fd_check(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    table = scene.objects[UniqueID("table_0")]
    table.metadata["category_norm"] = "nightstand"
    mug = scene.objects[UniqueID("mug_0")]
    mug.placement_info = None
    mug.metadata.update(
        {
            "category_norm": "mug",
            "functional_categories": ["graspable"],
            "explicit_target_relation": ["nightstand"],
        }
    )

    payload = evaluate_room_scene(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        stage="scene_after_furniture",
    )
    check = next(
        check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "asset_explicit_target_relation"
    )
    result = next(
        result
        for result in payload["results"]
        if result.get("check_id") == check["check_id"]
    )

    assert check["subject_id"] == "mug_0"
    assert check["target_ids"] == ["table_0"]
    assert check["relation_type"] == "object_on_support"
    assert check["expected_use"] == (
        "small object is supported by an appropriate surface"
    )
    assert check["evidence_refs"] == ["scene_geometry", "object_metadata"]
    assert check["evidence"]["explicit_target_relation"] == ["nightstand"]
    assert result["evaluation_source"] == "rule_functional_dependency"
    assert result["relation_type"] == "object_on_support"


def test_explicit_graspable_object_relation_matches_small_placeable_target(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    shelf = _box_object(
        "shelf_0",
        "display shelf",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.5),
        size=(0.8, 0.4, 1.0),
    )
    shelf.metadata.update(
        {
            "category_norm": "shelf",
            "functional_categories": ["supportable"],
            "explicit_target_relation": ["graspable_object"],
        }
    )
    book = _box_object(
        "book_0",
        "book",
        ObjectType.MANIPULAND,
        center=(0.0, 0.0, 1.08),
        size=(0.22, 0.16, 0.12),
    )
    book.metadata.update(
        {
            "category_norm": "book",
            "functional_categories": ["graspable"],
        }
    )
    scene.add_object(shelf)
    scene.add_object(book)

    payload = evaluate_room_scene(
        scene,
        config=CriticConfig(enabled=True, metrics=("functional_dependency",)),
        stage="scene_after_furniture",
    )
    check = next(
        check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "asset_explicit_target_relation"
    )
    result = next(
        result
        for result in payload["results"]
        if result.get("check_id") == check["check_id"]
    )

    assert check["subject_id"] == "shelf_0"
    assert check["target_ids"] == ["book_0"]
    assert check["relation_type"] == "object_on_support"
    assert result["evaluation_source"] == "rule_functional_dependency"
    assert result["label"] in {"pass", "degraded"}


def test_explicit_reverse_support_relation_requires_geometric_support() -> None:
    shelf = _benchmark_obj("shelf_1", "shelf", (0.0, 0.0, 0.5), (0.8, 0.4, 1.0))
    shelf["functional_hints"].update(
        {
            "scene_object_type": "furniture",
            "explicit_target_relation": ["book"],
        }
    )
    shelf["support_regions"] = [
        {
            "region_id": "S_shelf",
            "support_kind": "top_surface",
            "height_world_z": 1.0,
            "polygon_world_xy": [
                [-0.4, -0.2],
                [0.4, -0.2],
                [0.4, 0.2],
                [-0.4, 0.2],
            ],
            "clearance_above_m": 0.6,
            "access_type": "top",
        }
    ]
    far_book = _benchmark_obj("far_book_1", "book", (1.5, 0.0, 1.04), (0.2, 0.16, 0.08))
    near_book = _benchmark_obj(
        "near_book_1", "book", (0.0, 0.0, 1.04), (0.2, 0.16, 0.08)
    )
    far_book["functional_hints"].update(
        {"functional_categories": ["graspable"], "scene_object_type": "manipuland"}
    )
    near_book["functional_hints"].update(
        {"functional_categories": ["graspable"], "scene_object_type": "manipuland"}
    )

    far_checks = build_checks(
        _benchmark_case_pack([shelf, far_book]),
        metrics=["functional_dependency"],
    )
    near_checks = build_checks(
        _benchmark_case_pack([shelf, far_book, near_book]),
        metrics=["functional_dependency"],
    )
    explicit_near = [
        check
        for check in near_checks
        if check.get("check_source") == "asset_explicit_target_relation"
    ]

    assert not any(
        check.get("check_source") == "asset_explicit_target_relation"
        and check.get("subject_id") == "shelf_1"
        for check in far_checks
    )
    assert len(explicit_near) == 1
    assert explicit_near[0]["subject_id"] == "shelf_1"
    assert explicit_near[0]["target_ids"] == ["near_book_1"]


def test_explicit_target_relation_skips_incompatible_floor_lamp_surface() -> None:
    floor_lamp = _benchmark_obj(
        "tripod_floor_lamp_1",
        "tripod_floor_lamp",
        (0.0, 0.0, 0.8),
        (0.35, 0.35, 1.6),
    )
    floor_lamp["functional_hints"].update(
        {
            "category_group": "lighting",
            "explicit_target_relation": ["sofa"],
        }
    )
    sofa = _benchmark_obj("sofa_1", "sofa", (0.8, 0.0, 0.4), (1.6, 0.8, 0.8))

    checks = build_checks(
        _benchmark_case_pack([floor_lamp, sofa]),
        metrics=["functional_dependency"],
    )

    assert not any(
        check.get("check_source") == "asset_explicit_target_relation"
        and check.get("subject_id") == "tripod_floor_lamp_1"
        for check in checks
    )


def test_small_placeable_profile_overrides_noisy_work_surface_category() -> None:
    desk = _benchmark_obj("desk_1", "desk", (2.0, 2.0, 0.4), (1.2, 0.8, 0.8))
    desk["functional_hints"].update(
        {"category_group": "work_surface", "scene_object_type": "furniture"}
    )
    desk["support_regions"] = [
        {
            "region_id": "S_desk",
            "support_kind": "top_surface",
            "height_world_z": 0.8,
            "polygon_world_xy": [
                [1.4, 1.6],
                [2.6, 1.6],
                [2.6, 2.4],
                [1.4, 2.4],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    pen_cup = _benchmark_obj("pen_cup_1", "desk", (2.0, 2.0, 0.85), (0.10, 0.08, 0.10))
    pen_cup["functional_hints"].update(
        {
            "category_group": "object",
            "scene_object_type": "manipuland",
            "placement_class": "surface_object",
            "explicit_target_relation": ["desk"],
            "orientation_dependencies": [
                {
                    "relation_type": "front_faces",
                    "target_kind": "object",
                    "target_category": ["desk"],
                    "subject_face": "front",
                }
            ],
        }
    )
    pen_cup["object_function_profile"] = {
        "can_support_top": True,
        "has_internal_shelf": False,
        "is_small_placeable": True,
        "is_seating": False,
        "is_work_surface": False,
        "is_media_target": False,
        "is_bedside_surface": False,
        "is_sleeping_surface": False,
    }
    pen_cup["placement_info"] = {
        "parent_surface_id": "S_desk",
        "placement_method": "surface_placement",
    }
    chair = _benchmark_obj(
        "office_chair_1", "office_chair", (2.0, 1.15, 0.45), (0.5, 0.5, 0.9)
    )
    chair["functional_hints"]["category_group"] = "seating"

    case_pack = _benchmark_case_pack([desk, pen_cup, chair])
    checks = build_checks(case_pack, metrics=["functional_dependency"])
    check_ids = {check["check_id"] for check in checks}
    results = _run_direct_case_pack(
        _benchmark_case_pack([desk, pen_cup, chair], checks),
        metrics=["functional_dependency"],
    )
    support_result = next(
        result
        for result in results
        if result["check_id"] == "fd_pen_cup_1_desk_1_object_on_support"
    )

    assert "fd_pen_cup_1_desk_1_object_on_support" in check_ids
    assert not any(
        check.get("subject_id") == "pen_cup_1"
        and check.get("relation_type") == "workstation"
        for check in checks
    )
    assert not any(
        check.get("subject_id") == "pen_cup_1"
        and check.get("check_source") == "asset_orientation_dependency"
        for check in checks
    )
    assert support_result["label"] == "pass"


def test_surface_placed_small_placeable_furniture_can_be_supported() -> None:
    dresser = _benchmark_obj("dresser_1", "dresser", (0.0, 0.0, 0.35), (1.4, 0.5, 0.7))
    dresser["functional_hints"].update(
        {"scene_object_type": "furniture", "category_group": "storage"}
    )
    dresser["support_regions"] = [
        {
            "region_id": "S_dresser",
            "support_kind": "top_surface",
            "height_world_z": 0.7,
            "polygon_world_xy": [
                [-0.7, -0.25],
                [0.7, -0.25],
                [0.7, 0.25],
                [-0.7, 0.25],
            ],
            "clearance_above_m": 0.8,
            "access_type": "top",
        }
    ]
    jewelry_box = _benchmark_obj(
        "jewelry_box_1",
        "jewelry_box_jewelry",
        (0.0, 0.0, 0.73),
        (0.16, 0.12, 0.06),
    )
    jewelry_box["functional_hints"].update(
        {
            "category_group": "storage",
            "scene_object_type": "furniture",
            "placement_class": "surface_object",
            "functional_categories": [
                "containable",
                "graspable",
                "openable",
                "supportable",
            ],
        }
    )
    jewelry_box["object_function_profile"] = {
        "can_support_top": False,
        "has_internal_shelf": True,
        "is_small_placeable": True,
        "is_seating": False,
        "is_work_surface": False,
        "is_media_target": False,
        "is_bedside_surface": False,
        "is_sleeping_surface": False,
    }
    jewelry_box["placement_info"] = {
        "parent_surface_id": "S_dresser",
        "placement_method": "surface_placement",
    }
    check = {
        "check_id": "fd_jewelry_box_dresser",
        "metric": "functional_dependency",
        "subject_id": "jewelry_box_1",
        "target_ids": ["dresser_1"],
        "relation_type": "object_on_support",
    }

    result = _run_direct_case_pack(
        _benchmark_case_pack([jewelry_box, dresser], [check]),
        metrics=["functional_dependency"],
    )[0]

    assert _relation_target_is_valid(jewelry_box, dresser, "object_on_support")
    assert result["label"] == "pass"


def test_build_checks_materializes_grouped_fd_relations() -> None:
    dining_table = _benchmark_obj(
        "dining_table_1", "dining_table", (2.5, 2.5, 0.4), (1.2, 0.8, 0.8)
    )
    chair_1 = _benchmark_obj(
        "chair_1", "dining_chair", (2.5, 1.65, 0.45), (0.5, 0.5, 0.9)
    )
    chair_2 = _benchmark_obj(
        "chair_2", "dining_chair", (2.5, 3.35, 0.45), (0.5, 0.5, 0.9)
    )
    desk = _benchmark_obj("desk_1", "desk", (4.0, 2.5, 0.4), (1.2, 0.6, 0.8))
    office_chair = _benchmark_obj(
        "office_chair_1", "office_chair", (4.0, 1.75, 0.45), (0.5, 0.5, 0.9)
    )
    bed = _benchmark_obj("bed_1", "bed", (1.0, 3.0, 0.35), (1.4, 2.0, 0.7))
    nightstand_1 = _benchmark_obj(
        "nightstand_1", "nightstand", (0.05, 2.0, 0.35), (0.45, 0.45, 0.7)
    )
    nightstand_2 = _benchmark_obj(
        "nightstand_2", "nightstand", (1.95, 2.0, 0.35), (0.45, 0.45, 0.7)
    )
    case_pack = _benchmark_case_pack(
        [
            dining_table,
            chair_1,
            chair_2,
            desk,
            office_chair,
            bed,
            nightstand_1,
            nightstand_2,
        ]
    )

    checks = build_checks(case_pack, metrics=["functional_dependency"])
    grouped = {
        check["relation_type"]: check
        for check in checks
        if check.get("check_source") == "scenesmith_grouped_relation"
    }

    assert grouped["dining_set"]["subject_id"] == "dining_table_1"
    assert grouped["dining_set"]["target_ids"][:2] == ["chair_1", "chair_2"]
    assert grouped["workstation"]["subject_id"] == "desk_1"
    assert grouped["workstation"]["target_ids"][0] == "office_chair_1"
    assert grouped["bedside_pair"]["subject_id"] == "bed_1"
    assert set(grouped["bedside_pair"]["target_ids"]) == {
        "nightstand_1",
        "nightstand_2",
    }


def test_grouped_fd_relations_run_through_vendored_evaluator() -> None:
    table = _benchmark_obj(
        "dining_table_1", "dining_table", (2.5, 2.5, 0.4), (1.2, 0.8, 0.8)
    )
    chair_1 = _benchmark_obj(
        "chair_1", "dining_chair", (2.5, 1.65, 0.45), (0.5, 0.5, 0.9)
    )
    chair_2 = _benchmark_obj(
        "chair_2", "dining_chair", (2.5, 3.35, 0.45), (0.5, 0.5, 0.9)
    )
    case_pack = _benchmark_case_pack([table, chair_1, chair_2])
    case_pack["checks"] = build_checks(case_pack, metrics=["functional_dependency"])

    results = run_case_pack_checks(
        case_pack,
        CriticConfig(enabled=True, metrics=("functional_dependency",)),
    )
    dining_result = next(
        result for result in results if result.get("relation_type") == "dining_set"
    )

    assert dining_result["metric"] == "functional_dependency"
    assert dining_result["evaluation_source"] == "rule_functional_dependency"
    assert dining_result["label"] in {"pass", "degraded", "fail"}


def test_spatial_accessibility_checks_keep_nearby_blocker_targets() -> None:
    sofa = _benchmark_obj("sofa_1", "sofa", (2.0, 2.0, 0.45), (1.8, 0.8, 0.9))
    sofa["functional_hints"]["functional_categories"] = ["sittable"]
    coffee_table = _benchmark_obj(
        "coffee_table_1", "coffee_table", (2.0, 1.25, 0.25), (1.0, 0.45, 0.5)
    )
    coffee_table["functional_hints"]["functional_categories"] = ["supportable"]
    mug = _benchmark_obj("mug_1", "mug", (2.0, 1.25, 0.7), (0.12, 0.12, 0.18))
    mug["functional_hints"]["functional_categories"] = ["graspable"]
    case_pack = _benchmark_case_pack([sofa, coffee_table, mug])

    checks = build_checks(case_pack, metrics=["spatial_accessibility"])
    sofa_check = next(check for check in checks if check["subject_id"] == "sofa_1")

    assert sofa_check["target_ids"] == ["coffee_table_1"]


def test_spatial_accessibility_policy_skips_movable_seating() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"] = {
        "functional_categories": ["sittable"],
        "mobility_class": "movable",
        "accessibility_policy": "optional",
    }
    refrigerator = _benchmark_obj(
        "fridge_1", "refrigerator", (3.0, 2.0, 0.9), (0.7, 0.7, 1.8)
    )
    refrigerator["functional_hints"] = {
        "functional_categories": ["containable", "openable"],
        "mobility_class": "fixed",
        "accessibility_policy": "required",
    }

    checks = build_checks(
        _benchmark_case_pack([chair, refrigerator]),
        metrics=["spatial_accessibility"],
    )
    checked_subjects = {check["subject_id"] for check in checks}

    assert "chair_1" not in checked_subjects
    assert "fridge_1" in checked_subjects


def test_spatial_accessibility_skips_ceiling_mounted_lamps() -> None:
    table = _benchmark_obj("table_1", "table", (2.0, 2.0, 0.4), (1.0, 0.8, 0.8))
    table["functional_hints"]["functional_categories"] = ["supportable"]
    ceiling_lamp = _benchmark_obj(
        "ceiling_lamp_1", "ceiling_light", (2.0, 2.0, 2.4), (0.35, 0.35, 0.3)
    )
    ceiling_lamp["functional_hints"].update(
        {
            "functional_categories": ["supportable"],
            "category_group": "lighting",
            "scene_object_type": "ceiling_mounted",
        }
    )
    high_cabinet = _benchmark_obj(
        "high_cabinet_1", "cabinet", (3.2, 2.0, 1.4), (0.8, 0.5, 2.2)
    )
    high_cabinet["functional_hints"].update(
        {"functional_categories": ["openable"], "category_group": "storage"}
    )
    case_pack = _benchmark_case_pack([table, ceiling_lamp, high_cabinet])

    checks = build_checks(case_pack, metrics=["spatial_accessibility"])
    checked_subjects = {check["subject_id"] for check in checks}

    assert "ceiling_lamp_1" not in checked_subjects
    assert "table_1" in checked_subjects
    assert "high_cabinet_1" in checked_subjects


def test_room_scene_adapter_preserves_metadata_support_regions_and_profile(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    cabinet = _box_object(
        "cabinet_0",
        "storage cabinet",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.6),
        size=(0.8, 0.6, 1.2),
    )
    cabinet.metadata["support_regions"] = [
        {
            "region_id": "cabinet_floor",
            "support_kind": "cabinet_base",
            "height_world_z": 0.18,
            "polygon_world_xy": [
                [-0.4, -0.3],
                [0.4, -0.3],
                [0.4, 0.3],
                [-0.4, 0.3],
            ],
            "clearance_above_m": 0.5,
            "access_type": "openable_storage",
        }
    ]
    cabinet.metadata["object_function_profile"] = {
        "can_support_top": False,
        "has_internal_shelf": True,
        "is_work_surface": False,
    }
    scene.add_object(cabinet)

    case_pack = room_scene_to_case_pack(scene, stage="scene_after_furniture")
    cabinet_record = next(
        obj
        for obj in case_pack["scene_geometry"]["objects"]
        if obj["id"] == "cabinet_0"
    )

    assert cabinet_record["support_regions"][0]["region_id"] == "cabinet_floor"
    assert cabinet_record["support_regions"][0]["support_kind"] == "cabinet_base"
    assert cabinet_record["functional_hints"]["support_region_summary"] == {
        "count": 1,
        "support_kinds": ["cabinet_base"],
        "source": None,
    }
    assert cabinet_record["object_function_profile"]["can_support_top"] is False
    assert cabinet_record["object_function_profile"]["has_internal_shelf"] is True
    assert cabinet_record["object_function_profile"]["is_work_surface"] is False


def test_asset_annotation_mock_writes_effective_hints_and_files(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    stage_dir = tmp_path / "scene_states" / "final_scene"
    stage_dir.mkdir(parents=True)
    (stage_dir / "scene_state.json").write_text(
        json.dumps(scene.to_state_dict(), indent=2), encoding="utf-8"
    )
    config = CriticConfig(
        enabled=True,
        asset_annotation={
            "enabled": True,
            "backend": "mock",
            "write_files": True,
            "write_back": True,
        },
    )

    summary = annotate_room_scene(
        scene,
        output_dir=stage_dir,
        config=config,
        stage="final_scene",
    )

    assert summary is not None
    assert summary["object_count"] == 2
    assert (stage_dir / "asset_annotations" / "table_0.yaml").exists()
    table_hints = scene.objects[UniqueID("table_0")].metadata["functional_hints"]
    assert table_hints["classification_source"] == "asset_annotation"
    assert table_hints["asset_annotation_source"] == ("scenesmith_vlm_asset_annotator")
    assert "supportable" in table_hints["functional_categories"]
    assert table_hints["mobility_class"] == "semi_movable"
    assert table_hints["accessibility_policy"] == "required"
    assert "top" in table_hints["access_sides"]
    assert table_hints["attachment_dependencies"][0]["relation_type"] == (
        "object_on_floor"
    )
    table_annotation = yaml.safe_load(
        (stage_dir / "asset_annotations" / "table_0.yaml").read_text(encoding="utf-8")
    )
    assert table_annotation["object_function_profile"]["can_support_top"] is True
    assert table_annotation["effective_annotation"]["mobility_class"] == "semi_movable"
    saved_state = json.loads((stage_dir / "scene_state.json").read_text())
    saved_hints = saved_state["objects"]["table_0"]["metadata"]["functional_hints"]
    assert saved_hints["classification_source"] == "asset_annotation"
    assert saved_hints["accessibility_policy"] == "required"


def test_asset_annotation_reuses_saved_object_function_profile(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    stage_dir = tmp_path / "scene_states" / "final_scene"
    annotation_dir = stage_dir / "asset_annotations"
    annotation_dir.mkdir(parents=True)
    config = CriticConfig(
        enabled=True,
        asset_annotation={
            "enabled": True,
            "backend": "mock",
            "write_files": False,
            "write_back": True,
            "skip_existing": True,
        },
    )
    annotation = {
        "object_id": "table_0",
        "effective_annotation": {
            "category_norm": "table",
            "scene_object_type": "furniture",
            "benchmark_relevance": "functional",
            "affordances": ["supportable"],
            "source": "unit_test",
            "confidence": 0.95,
        },
        "object_function_profile": {
            "can_support_top": False,
            "has_internal_shelf": True,
            "is_small_placeable": False,
            "is_seating": False,
            "is_work_surface": False,
            "is_media_target": False,
            "is_bedside_surface": False,
            "is_sleeping_surface": False,
        },
    }
    (annotation_dir / "table_0.yaml").write_text(
        yaml.safe_dump(annotation, sort_keys=False), encoding="utf-8"
    )

    summary = annotate_room_scene(
        scene,
        output_dir=stage_dir,
        config=config,
        stage="final_scene",
    )

    assert summary is not None
    table_profile = scene.objects[UniqueID("table_0")].metadata[
        "object_function_profile"
    ]
    assert table_profile["can_support_top"] is False
    assert table_profile["has_internal_shelf"] is True


def test_write_room_stage_report_runs_asset_annotation_before_case_pack(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    stage_dir = tmp_path / "scene_states" / "scene_after_furniture"
    stage_dir.mkdir(parents=True)
    config = CriticConfig(
        enabled=True,
        room_stage_hooks=("scene_after_furniture",),
        asset_annotation={
            "enabled": True,
            "backend": "mock",
            "write_files": False,
            "write_scene_state": False,
        },
    )

    payload = write_room_stage_report(
        scene,
        stage_dir,
        config=config,
        stage="scene_after_furniture",
    )

    assert payload is not None
    objects = {
        obj["id"]: obj for obj in payload["case_pack"]["scene_geometry"]["objects"]
    }
    assert objects["table_0"]["functional_hints"]["classification_source"] == (
        "asset_annotation"
    )
    assert objects["table_0"]["functional_hints"]["asset_annotation_source"] == (
        "scenesmith_vlm_asset_annotator"
    )


def test_room_scene_adapter_reads_nested_support_region_hints(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    cabinet = _box_object(
        "cabinet_0",
        "dataset storage asset",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.6),
        size=(0.8, 0.6, 1.2),
    )
    cabinet.metadata["functional_hints"] = {
        "support_region": {
            "region_id": "nested_shelf",
            "support_kind": "open_shelf",
            "height_world_z": 0.5,
            "polygon_world_xy": [
                [-0.35, -0.25],
                [0.35, -0.25],
                [0.35, 0.25],
                [-0.35, 0.25],
            ],
            "clearance_above_m": 0.4,
            "access_type": "openable_storage",
        }
    }
    scene.add_object(cabinet)

    case_pack = room_scene_to_case_pack(scene, stage="scene_after_furniture")
    cabinet_record = next(
        obj
        for obj in case_pack["scene_geometry"]["objects"]
        if obj["id"] == "cabinet_0"
    )

    assert cabinet_record["support_regions"][0]["region_id"] == "nested_shelf"
    assert cabinet_record["support_regions"][0]["support_kind"] == "open_shelf"
    assert cabinet_record["object_function_profile"]["has_internal_shelf"] is True


def test_house_case_pack_offsets_metadata_support_region_polygons(
    tmp_path: Path,
) -> None:
    house = _house(tmp_path)
    house.layout.placed_rooms[0].position = (-2.0, -1.0)
    room = house.rooms["main"]
    room.objects.clear()
    cabinet = _box_object(
        "cabinet_0",
        "storage cabinet",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.6),
        size=(0.8, 0.6, 1.2),
    )
    cabinet.metadata["support_regions"] = [
        {
            "region_id": "cabinet_floor",
            "support_kind": "cabinet_base",
            "height_world_z": 0.18,
            "polygon_world_xy": [[-0.4, -0.3], [0.4, -0.3], [0.4, 0.3], [-0.4, 0.3]],
            "clearance_above_m": 0.5,
            "access_type": "openable_storage",
        }
    ]
    room.add_object(cabinet)

    case_pack = house_scene_to_case_pack(house, stage="combined_house_after_furniture")
    cabinet_record = next(
        obj
        for obj in case_pack["scene_geometry"]["objects"]
        if obj["id"] == "cabinet_0"
    )

    assert cabinet_record["support_regions"][0]["polygon_world_xy"][0] == [0.6, 0.7]


def test_room_scene_adapter_uses_oriented_local_footprint_for_rotated_objects(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    table = _box_object(
        "table_0",
        "table",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.35),
        size=(2.0, 1.0, 0.7),
        yaw_deg=90.0,
    )
    scene.add_object(table)

    case_pack = room_scene_to_case_pack(scene, stage="final_scene")
    table_record = next(
        obj for obj in case_pack["scene_geometry"]["objects"] if obj["id"] == "table_0"
    )

    assert np.allclose(table_record["footprint_world"][0], [0.5, -1.0], atol=1e-6)
    assert np.allclose(table_record["footprint_world"][1], [0.5, 1.0], atol=1e-6)


def test_room_scene_adapter_includes_structural_walls(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    wall = _box_object(
        "north_wall",
        "north wall",
        ObjectType.WALL,
        center=(0.0, 2.0, 1.5),
        size=(6.0, 0.05, 3.0),
    )
    scene.room_geometry.walls.append(wall)

    case_pack = room_scene_to_case_pack(scene, stage="final_scene")
    object_ids = {obj["id"] for obj in case_pack["scene_geometry"]["objects"]}

    assert "north_wall" in object_ids


def test_evaluate_room_scene_returns_rule_results(tmp_path: Path) -> None:
    assert CriticConfig(enabled=True).enabled

    payload = evaluate_room_scene(
        _scene(tmp_path),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["spatial_accessibility", "functional_dependency"],
            }
        },
        stage="final_scene",
    )

    fd_result = next(
        result
        for result in payload["results"]
        if result["metric"] == "functional_dependency"
    )
    assert fd_result["label"] == "pass"
    assert payload["summary"]["scene_summary"]["total_checks"] >= 1


def test_orientation_contract_keeps_living_room_seat_target_stable(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.room_type = "living_room"
    scene.text_description = (
        "A living room with an armchair near a coffee table facing a TV stand."
    )
    scene.add_object(
        _box_object(
            "armchair_0",
            "comfortable armchair",
            ObjectType.FURNITURE,
            center=(1.2, 0.4, 0.4),
            size=(0.8, 0.8, 0.8),
            yaw_deg=90.0,
        )
    )
    scene.add_object(
        _box_object(
            "coffee_table_0",
            "coffee table",
            ObjectType.FURNITURE,
            center=(0.0, 0.0, 0.25),
            size=(1.1, 0.55, 0.5),
        )
    )
    scene.add_object(
        _box_object(
            "tv_stand_0",
            "tv stand media console",
            ObjectType.FURNITURE,
            center=(0.0, -1.7, 0.3),
            size=(1.4, 0.35, 0.6),
        )
    )

    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
        extra={"stable_orientation_contracts": True},
    )
    first = room_scene_to_case_pack(scene, stage="scene_after_furniture")
    stabilize_orientation_contracts(first, scene, config, stage="scene_after_furniture")
    second = room_scene_to_case_pack(scene, stage="final_scene")
    stabilize_orientation_contracts(second, scene, config, stage="final_scene")

    first_check = _contract_check_for(first, "armchair_0")
    second_check = _contract_check_for(second, "armchair_0")

    assert first_check["relation_type"] == "seating_to_media"
    assert first_check["target_ids"] == ["tv_stand_0"]
    assert second_check["relation_type"] == "seating_to_media"
    assert second_check["target_ids"] == ["tv_stand_0"]


def test_orientation_contract_ignores_noisy_sittable_affordance_on_non_seats(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.room_type = "living_room"
    scene.text_description = "A living room with an armchair, TV, and coffee table."
    scene.objects.clear()
    for obj in [
        _box_object(
            "armchair_0",
            "armchair",
            ObjectType.FURNITURE,
            center=(1.2, 0.4, 0.4),
            size=(0.8, 0.8, 0.8),
            yaw_deg=90.0,
        ),
        _box_object(
            "tv_stand_0",
            "tv stand",
            ObjectType.FURNITURE,
            center=(0.0, -1.7, 0.3),
            size=(1.4, 0.35, 0.6),
        ),
        _box_object(
            "coffee_table_0",
            "coffee table",
            ObjectType.FURNITURE,
            center=(0.0, 0.0, 0.25),
            size=(1.1, 0.55, 0.5),
        ),
    ]:
        scene.add_object(obj)

    config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
        extra={"stable_orientation_contracts": True},
    )
    case_pack = room_scene_to_case_pack(scene, stage="final_scene")
    for record in case_pack["scene_geometry"]["objects"]:
        if record["id"] in {"tv_stand_0", "coffee_table_0"}:
            # 2026-07-08 修改原因：复现资产标注把非座椅误带 sittable 的回归场景。
            record["functional_hints"]["affordances"] = ["sittable", "supportable"]
            record["functional_hints"]["functional_categories"] = [
                "sittable",
                "supportable",
            ]
            record["object_function_profile"]["is_seating"] = False

    stabilize_orientation_contracts(case_pack, scene, config, stage="final_scene")

    contract_subjects = [
        check["subject_id"]
        for check in case_pack["checks"]
        if check.get("check_source") == CONTRACT_CHECK_SOURCE
    ]
    assert contract_subjects == ["armchair_0"]


def _contract_check_for(case_pack: dict[str, Any], subject_id: str) -> dict[str, Any]:
    return next(
        check
        for check in case_pack["checks"]
        if check.get("check_source") == CONTRACT_CHECK_SOURCE
        and check.get("subject_id") == subject_id
    )


def test_direct_evaluate_room_scene_runs_with_default_config(tmp_path: Path) -> None:
    payload = evaluate_room_scene(_scene(tmp_path), stage="final_scene")

    assert payload["schema_version"] == "scenesmith.scenebenchmark_critic.report.v1"
    assert payload["scope"] == "room:main"
    assert payload["stage"] == "final_scene"
    assert (
        payload["case_pack"]["schema_version"] == "scenesmith.scenebenchmark_critic.v1"
    )
    assert payload["gate"]["enabled"] is False
    assert payload["gate"]["label"] == "report_only"
    assert {result["metric"] for result in payload["results"]} >= {
        "functional_dependency"
    }


def test_evaluate_room_scene_respects_metric_filter(tmp_path: Path) -> None:
    fd_payload = evaluate_room_scene(
        _scene(tmp_path / "fd_only"),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["functional_dependency"],
            }
        },
        stage="final_scene",
    )
    sa_payload = evaluate_room_scene(
        _scene(tmp_path / "sa_only"),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["spatial_accessibility"],
            }
        },
        stage="final_scene",
    )

    assert {result["metric"] for result in fd_payload["results"]} == {
        "functional_dependency"
    }
    assert {result["metric"] for result in sa_payload["results"]} == {
        "spatial_accessibility"
    }


def test_evaluate_room_scene_accepts_string_metric_filter(tmp_path: Path) -> None:
    payload = evaluate_room_scene(
        _scene(tmp_path),
        config={
            "scenebenchmark_critic": {
                "enabled": "true",
                "metrics": "functional_dependency",
            }
        },
        stage="final_scene",
    )

    assert {check["metric"] for check in payload["case_pack"]["checks"]} == {
        "functional_dependency"
    }
    assert {result["metric"] for result in payload["results"]} == {
        "functional_dependency"
    }


def test_hard_gate_records_blocked_metadata_without_rewriting_scene() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"]["functional_categories"] = ["sittable"]
    blockers = [
        _benchmark_obj("front_block", "cabinet", (2.0, 2.85, 0.5), (1.2, 0.8, 1.0)),
        _benchmark_obj("left_block", "cabinet", (1.15, 2.0, 0.5), (0.8, 1.2, 1.0)),
        _benchmark_obj("right_block", "cabinet", (2.85, 2.0, 0.5), (0.8, 1.2, 1.0)),
    ]
    case_pack = _benchmark_case_pack(
        [chair, *blockers],
        [
            {
                "check_id": "sa_blocked",
                "metric": "spatial_accessibility",
                "subject_id": "chair_1",
                "affordance": "sittable",
            }
        ],
    )
    results = _run_direct_case_pack(
        case_pack,
        metrics=["spatial_accessibility"],
    )

    from scenesmith.scenebenchmark_critic.reports import build_evaluation_payload

    payload = build_evaluation_payload(
        case_pack=case_pack,
        results=results,
        stage="scene_after_furniture",
        scope="room:test",
        config=CriticConfig(
            enabled=True,
            metrics=("spatial_accessibility",),
            hard_gate=True,
            fail_gate_threshold=1,
        ),
    )

    assert payload["gate"]["enabled"] is True
    assert payload["gate"]["blocked"] is True
    assert payload["gate"]["label"] == "fail"
    assert case_pack["scene_geometry"]["objects"][0]["id"] == "chair_1"


def test_base_experiment_config_defaults_are_report_only_template_mode() -> None:
    cfg = OmegaConf.load("configurations/experiment/base_experiment.yaml")
    critic_config = critic_config_from_any({"experiment": cfg})

    assert critic_config.enabled is False
    assert critic_config.metrics == (
        "spatial_accessibility",
        "functional_dependency",
    )
    assert critic_config.room_stage_hooks == ("scene_after_furniture", "final_scene")
    assert critic_config.house_stage_hooks == ()
    assert critic_config.inject_into_llm_critic is True
    assert critic_config.agent_prompt_context_filter_enabled is True
    assert critic_config.agent_prompt_context_debug_write is False
    assert critic_config.hard_gate is False
    assert critic_config.extra["fd_relation_proposer_mode"] == "template"
    assert critic_config.extra["max_fd_relation_proposals"] == 8
    assert critic_config.extra["accessibility_grid_resolution_m"] == 0.05


def test_config_helper_defaults_to_room_only_when_section_is_minimal() -> None:
    critic_config = critic_config_from_any(
        {"experiment": {"scenebenchmark_critic": {"enabled": True}}}
    )

    assert critic_config.enabled is True
    assert critic_config.room_stage_hooks == ("scene_after_furniture", "final_scene")
    assert critic_config.house_stage_hooks == ()


def test_config_helper_parses_string_booleans_for_direct_api_configs() -> None:
    critic_config = critic_config_from_any(
        {
            "scenebenchmark_critic": {
                "enabled": "false",
                "inject_into_llm_critic": "off",
                "hard_gate": "yes",
            }
        }
    )

    assert critic_config.enabled is False
    assert critic_config.inject_into_llm_critic is False
    assert critic_config.hard_gate is True


def test_config_helper_parses_string_sequences_for_direct_api_configs() -> None:
    critic_config = critic_config_from_any(
        {
            "scenebenchmark_critic": {
                "metrics": "functional_dependency",
                "room_stage_hooks": "scene_after_furniture, final_scene",
                "house_stage_hooks": "",
            }
        }
    )

    assert critic_config.metrics == ("functional_dependency",)
    assert critic_config.room_stage_hooks == ("scene_after_furniture", "final_scene")
    assert critic_config.house_stage_hooks == ()


def test_readme_documents_memsafe_config_wrapper_without_secrets() -> None:
    readme = Path("scenesmith/scenebenchmark_critic/README.md").read_text(
        encoding="utf-8"
    )
    vendor_readme = Path("scenesmith/scenebenchmark_critic/vendor/README.md").read_text(
        encoding="utf-8"
    )

    assert "scenebenchmark_critic_memsafe_smoke" in readme
    assert "critic_eval_config" in readme
    assert "hydra.run.dir=/path/to/existing/output_dir" in readme
    assert "fd_proposer_checks" in readme
    assert "scenebenchmark_critic.json" in readme
    assert "single-room" in readme
    assert "RoomScene" in readme
    assert "explicit opt-in" in readme
    assert "grouped functional-dependency" in readme
    assert "access_direction" in readme
    assert "interaction faces" in readme
    assert "benchmark_relevance" in readme
    assert "category_keywords" in readme
    assert "support_region_summary" in readme
    assert "front_hint" in readme
    assert "metric_relevance" in readme
    assert "bedroom_nightstand_1_f0_c" in readme
    assert 'Path("config.yml")' in readme
    assert 'os.environ["OPENAI_API_KEY"]' in readme
    assert "house_stage_hooks" in readme
    assert "vendor/README.md" in readme
    assert "room_scene_to_case_pack" in readme
    assert "metrics/functional_dependency/" in vendor_readme
    assert "metrics/spatial_accessibility/" in vendor_readme
    assert "AST parity" in vendor_readme
    assert "sk-" not in readme
    assert "sk-" not in vendor_readme


def test_public_package_exports_embedding_api() -> None:
    import scenesmith.scenebenchmark_critic as critic

    expected = {
        "CriticConfig",
        "annotate_room_scene",
        "evaluate_house_scene",
        "evaluate_room_scene",
        "format_prompt_context",
        "house_scene_to_case_pack",
        "room_scene_to_case_pack",
        "write_house_stage_report",
        "write_room_stage_report",
    }

    assert set(critic.__all__) == expected
    for name in expected:
        assert getattr(critic, name) is not None


def test_scenesmith_utils_is_regular_package_for_wheel_imports() -> None:
    import scenesmith.utils as utils

    assert Path("scenesmith/utils/__init__.py").exists()
    assert utils.__file__ is not None


def test_public_room_case_pack_export_builds_single_room_case_pack(
    tmp_path: Path,
) -> None:
    import scenesmith.scenebenchmark_critic as critic

    case_pack = critic.room_scene_to_case_pack(
        _scene(tmp_path),
        stage="scene_after_furniture",
    )

    assert case_pack["schema_version"] == "scenesmith.scenebenchmark_critic.v1"
    assert case_pack["scene_id"] == "main:scene_after_furniture"
    assert case_pack["room_type"] == "bedroom"
    assert len(case_pack["scene_geometry"]["rooms"]) == 1
    assert {check["metric"] for check in case_pack["checks"]} == {
        "spatial_accessibility",
        "functional_dependency",
    }


def test_vendored_metric_packages_preserve_scenebenchmark_lazy_exports() -> None:
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics import (
        functional_dependency,
        spatial_accessibility,
    )

    assert functional_dependency.evaluate_functional_dependency is not None
    assert functional_dependency.augment_functional_dependency_checks is not None
    assert functional_dependency.PLUGIN.name == "functional_dependency"
    assert spatial_accessibility.evaluate_spatial_accessibility is not None
    assert spatial_accessibility.PLUGIN.name == "spatial_accessibility"


def test_vendored_scenebenchmark_code_has_no_external_repo_imports() -> None:
    vendor_root = Path("scenesmith/scenebenchmark_critic/vendor/scenebenchmark")
    forbidden_prefixes = (
        "from critic ",
        "from critic.",
        "import critic",
        "from metrics ",
        "from metrics.",
        "import metrics",
        "import bootstrap",
        "from bootstrap ",
    )
    offenders: list[str] = []
    for path in sorted(vendor_root.rglob("*.py")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if stripped.startswith(forbidden_prefixes):
                offenders.append(f"{path}:{line_number}: {stripped}")

    assert offenders == []


def test_vendored_scenebenchmark_fd_sa_rule_source_manifest_is_complete() -> None:
    vendor_root = Path("scenesmith/scenebenchmark_critic/vendor/scenebenchmark")
    expected_files = {
        "__init__.py",
        "critic/__init__.py",
        "critic/accessibility.py",
        "critic/config.py",
        "critic/dependency.py",
        "critic/geometry.py",
        "critic/models.py",
        "metrics/__init__.py",
        "metrics/base.py",
        "metrics/functional_dependency/__init__.py",
        "metrics/functional_dependency/augmenter.py",
        "metrics/functional_dependency/constants.py",
        "metrics/functional_dependency/evaluator.py",
        "metrics/functional_dependency/legacy.py",
        "metrics/functional_dependency/plugin.py",
        "metrics/functional_dependency/profiles.py",
        "metrics/functional_dependency/proposer.py",
        "metrics/functional_dependency/relations.py",
        "metrics/functional_dependency/results.py",
        "metrics/functional_dependency/semantics.py",
        "metrics/functional_dependency/support.py",
        "metrics/functional_dependency/support_scoring.py",
        "metrics/spatial_accessibility/__init__.py",
        "metrics/spatial_accessibility/config.py",
        "metrics/spatial_accessibility/core.py",
        "metrics/spatial_accessibility/evaluator.py",
        "metrics/spatial_accessibility/grid.py",
        "metrics/spatial_accessibility/legacy.py",
        "metrics/spatial_accessibility/obstacles.py",
        "metrics/spatial_accessibility/plugin.py",
        "metrics/spatial_accessibility/reach.py",
        "metrics/spatial_accessibility/results.py",
        "metrics/spatial_accessibility/zones.py",
    }

    actual_files = {
        path.relative_to(vendor_root).as_posix()
        for path in vendor_root.rglob("*.py")
        if path.is_file()
    }

    assert actual_files == expected_files


def test_vendored_scenebenchmark_rule_bodies_match_upstream_when_available() -> None:
    upstream_root = Path.home() / "proj" / "SceneBenchmark" / "src"
    if not upstream_root.exists():
        pytest.skip("SceneBenchmark source checkout is not available")

    vendor_root = Path("scenesmith/scenebenchmark_critic/vendor/scenebenchmark")
    # relations.py carries SceneSmith-only asset dependency evaluators, so it is
    # covered by local behavioral tests instead of upstream AST parity.
    parity_files = {
        "critic/accessibility.py",
        "critic/config.py",
        "critic/dependency.py",
        "critic/geometry.py",
        "critic/models.py",
        "metrics/base.py",
        "metrics/functional_dependency/augmenter.py",
        "metrics/functional_dependency/constants.py",
        "metrics/functional_dependency/evaluator.py",
        "metrics/functional_dependency/legacy.py",
        "metrics/functional_dependency/plugin.py",
        "metrics/functional_dependency/profiles.py",
        "metrics/functional_dependency/proposer.py",
        "metrics/functional_dependency/results.py",
        "metrics/functional_dependency/semantics.py",
        "metrics/functional_dependency/support.py",
        "metrics/functional_dependency/support_scoring.py",
        "metrics/spatial_accessibility/config.py",
        "metrics/spatial_accessibility/core.py",
        "metrics/spatial_accessibility/evaluator.py",
        "metrics/spatial_accessibility/grid.py",
        "metrics/spatial_accessibility/legacy.py",
        "metrics/spatial_accessibility/obstacles.py",
        "metrics/spatial_accessibility/plugin.py",
        "metrics/spatial_accessibility/reach.py",
        "metrics/spatial_accessibility/results.py",
        "metrics/spatial_accessibility/zones.py",
    }
    diffs = [
        rel
        for rel in sorted(parity_files)
        if _normalized_rule_ast(upstream_root / rel)
        != _normalized_rule_ast(vendor_root / rel)
    ]

    assert diffs == []


class _NormalizeVendoredImports(ast.NodeTransformer):
    _PREFIX = "scenesmith.scenebenchmark_critic.vendor.scenebenchmark."

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module and node.module.startswith(self._PREFIX):
            node.module = node.module[len(self._PREFIX) :]
        return node


def _normalized_rule_ast(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    normalized = _NormalizeVendoredImports().visit(tree)
    assert isinstance(normalized, ast.Module)
    normalized.body = [
        node
        for node in normalized.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not _is_optional_agent_import_guard(path, node)
    ]
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _is_optional_agent_import_guard(path: Path, node: ast.AST) -> bool:
    if not isinstance(node, ast.Try):
        return False
    segment = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
    return "build_structured_agent" in segment


def test_vendored_scenebenchmark_modules_import_cleanly() -> None:
    import scenesmith.scenebenchmark_critic.vendor.scenebenchmark as vendored

    package_prefix = vendored.__name__ + "."
    modules = [
        module_info.name
        for module_info in pkgutil.walk_packages(
            vendored.__path__, prefix=package_prefix
        )
    ]

    assert modules
    for module_name in modules:
        importlib.import_module(module_name)


def test_vendored_structured_agent_uses_project_vlm_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic import (
        agent as agent_module,
    )
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.agent import (
        build_structured_agent,
    )
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.models import (
        FunctionalDependencyProposalSet,
    )

    calls: list[dict[str, Any]] = []

    class FakeVLMService:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

        def create_completion(self, **kwargs: Any) -> str:
            calls.append({"cfg": self.cfg, "kwargs": kwargs})
            return json.dumps(
                {
                    "proposals": [
                        {
                            "subject_id": "chair_1",
                            "target_ids": ["desk_1"],
                            "relation_type": "seating_to_work_surface",
                            "expected_use": "sit at and use the work surface",
                            "priority": 0.9,
                            "reason": "unit test",
                        }
                    ]
                }
            )

    monkeypatch.setattr(agent_module, "VLMService", FakeVLMService)

    cfg = OmegaConf.create({"provider": {"model": "local-model"}})
    agent = build_structured_agent(
        cfg,
        output_type=FunctionalDependencyProposalSet,
        system_prompt="system",
        name="unit_test_agent",
    )

    result = agent.run_sync('{"subjects": [], "targets": []}')

    assert result.output.proposals[0].subject_id == "chair_1"
    assert calls[0]["cfg"] is cfg
    assert calls[0]["kwargs"]["model"] == "local-model"
    assert calls[0]["kwargs"]["messages"][0]["role"] == "system"
    assert calls[0]["kwargs"]["response_format"] == {"type": "json_object"}


def test_rule_config_forwards_asset_annotation_model_to_vlm_proposer() -> None:
    from scenesmith.scenebenchmark_critic.vendor.rules import _to_rule_config

    rule_config = _to_rule_config(
        CriticConfig(
            enabled=True,
            asset_annotation={"enabled": True, "backend": "vlm", "model": "qwen-local"},
        )
    )

    assert rule_config.provider == {"model": "qwen-local"}


def test_vendored_critic_wrappers_evaluate_rules_without_external_repo() -> None:
    from scenesmith.scenebenchmark_critic.vendor.rules import _to_rule_config
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic import (
        accessibility,
        dependency,
    )
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.geometry import (
        load_geometry,
    )

    chair = _benchmark_obj("chair_1", "chair", (2.0, 1.2, 0.45), (0.5, 0.5, 0.9))
    desk = _benchmark_obj("desk_1", "desk", (2.0, 2.0, 0.4), (1.2, 0.7, 0.8))
    case_pack = _benchmark_case_pack([chair, desk])
    store = load_geometry(case_pack)

    assert store is not None

    accessibility_result = accessibility.evaluate_spatial_accessibility(
        store,
        {
            "check_id": "sa_chair_1",
            "metric": "spatial_accessibility",
            "subject_id": "chair_1",
            "affordance": "sittable",
        },
        _to_rule_config(CriticConfig(enabled=True)),
    )
    dependency_result = dependency.evaluate_functional_dependency(
        store,
        {
            "check_id": "fd_chair_desk",
            "metric": "functional_dependency",
            "subject_id": "chair_1",
            "target_ids": ["desk_1"],
            "relation_type": "seating_to_work_surface",
        },
    )

    assert accessibility_result["metric"] == "spatial_accessibility"
    assert accessibility_result["check_id"] == "sa_chair_1"
    assert accessibility_result["label"] in {"pass", "degraded", "fail", "unknown"}
    assert dependency_result["metric"] == "functional_dependency"
    assert dependency_result["primary_object"] == "chair_1"


def test_aggregate_results_excludes_ignored_scoring_tier_from_summary() -> None:
    summary = aggregate_results(
        [
            {
                "check_id": "core_pass",
                "metric": "functional_dependency",
                "label": "pass",
                "primary_object": "mug_0",
            },
            {
                "check_id": "ignored_fail",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "decor_0",
                "scoring_tier": "ignored",
            },
        ]
    )
    scene_summary = summary["scene_summary"]
    scene_diag_summary = summary["scene_diagnostic_summary"]
    fd_summary = summary["metric_summary"]["functional_dependency"]
    fd_diag_summary = summary["metric_diagnostic_summary"]["functional_dependency"]
    object_rows = {row["subject_id"]: row for row in summary["object_results"]}

    assert scene_summary["all_checks"] == 2
    assert scene_summary["total_checks"] == 1
    assert scene_summary["effective_checks"] == 1
    assert scene_summary["fail"] == 0
    assert scene_summary["excluded_auxiliary"] == 0
    assert scene_summary["excluded_ignored"] == 1
    assert scene_summary["score"] == 1.0
    assert scene_diag_summary["total_checks"] == 2
    assert scene_diag_summary["fail"] == 1
    assert fd_summary["excluded_checks"] == 1
    assert fd_diag_summary["fail"] == 1
    assert object_rows["mug_0"]["checks"][0]["counted_in_summary"] is True
    assert object_rows["decor_0"]["checks"][0]["counted_in_summary"] is False


def test_aggregate_results_uses_case_pack_check_metadata_fallback() -> None:
    case_pack = {
        "checks": [
            {
                "check_id": "ignored_fail",
                "metric": "functional_dependency",
                "subject_id": "decor_0",
                "target_ids": ["table_0"],
                "scoring_tier": "ignored",
            }
        ]
    }
    summary = aggregate_results(
        [
            {
                "check_id": "ignored_fail",
                "label": "fail",
            }
        ],
        case_pack=case_pack,
    )
    object_row = summary["object_results"][0]
    check_row = object_row["checks"][0]

    assert summary["scene_summary"]["total_checks"] == 0
    assert summary["scene_summary"]["excluded_ignored"] == 1
    assert summary["metric_summary"]["functional_dependency"]["excluded_ignored"] == 1
    assert object_row["subject_id"] == "decor_0"
    assert check_row["primary_object"] == "decor_0"
    assert check_row["related_objects"] == ["table_0"]
    assert check_row["scoring_tier"] == "ignored"
    assert check_row["counted_in_summary"] is False


def test_aggregate_results_accepts_upstream_argument_order() -> None:
    case_pack = {
        "checks": [
            {
                "check_id": "fd1",
                "metric": "functional_dependency",
                "subject_id": "book_0",
                "target_ids": ["shelf_0"],
                "scoring_tier": "auxiliary",
            }
        ]
    }

    summary = aggregate_results(
        case_pack,
        [
            {
                "check_id": "fd1",
                "label": "pass",
            }
        ],
    )

    assert summary["scene_summary"]["total_checks"] == 1
    assert summary["scene_summary"]["excluded_auxiliary"] == 0
    assert summary["object_results"][0]["subject_id"] == "book_0"
    assert summary["object_results"][0]["checks"][0]["related_objects"] == ["shelf_0"]


def test_template_fd_proposer_adds_seating_work_surface_relation(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    chair = _box_object(
        "chair_0",
        "dining chair",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.45),
        size=(0.5, 0.5, 0.9),
    )
    desk = _box_object(
        "desk_0",
        "writing desk",
        ObjectType.FURNITURE,
        center=(0.85, 0.0, 0.4),
        size=(1.2, 0.7, 0.8),
    )
    scene.add_object(chair)
    scene.add_object(desk)

    payload = evaluate_room_scene(
        scene,
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["functional_dependency"],
            }
        },
        stage="scene_after_furniture",
    )

    fd_results = [
        result
        for result in payload["results"]
        if result.get("relation_type") == "seating_to_work_surface"
    ]
    assert fd_results
    assert fd_results[0]["primary_object"] == "chair_0"
    assert fd_results[0]["related_objects"] == ["desk_0"]
    assert fd_results[0]["label"] in {"pass", "degraded"}


def test_report_case_pack_keeps_fd_proposer_augmented_checks(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    chair = _box_object(
        "chair_0",
        "dining chair",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.45),
        size=(0.5, 0.5, 0.9),
    )
    desk = _box_object(
        "desk_0",
        "writing desk",
        ObjectType.FURNITURE,
        center=(0.85, 0.0, 0.4),
        size=(1.2, 0.7, 0.8),
    )
    scene.add_object(chair)
    scene.add_object(desk)

    payload = evaluate_room_scene(
        scene,
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["functional_dependency"],
                "fd_relation_proposer_mode": "template",
            }
        },
        stage="scene_after_furniture",
    )

    proposer_checks = [
        check
        for check in payload["case_pack"]["checks"]
        if check.get("check_source") == "fd_relation_proposer"
    ]
    proposer_result_ids = {
        result["check_id"]
        for result in payload["results"]
        if result.get("metric") == "functional_dependency"
    }

    assert any(
        check["relation_type"] == "seating_to_work_surface"
        and check["subject_id"] == "chair_0"
        and check["target_ids"] == ["desk_0"]
        for check in proposer_checks
    )
    assert {check["check_id"] for check in proposer_checks} <= proposer_result_ids


def test_single_room_offline_smoke_runs_fd_sa_and_template_proposer(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    scene.objects.clear()
    scene.text_description = (
        "A compact bedroom with a bed, nightstand, mug, desk, and chair."
    )
    bed = _box_object(
        "bed_0",
        "bed",
        ObjectType.FURNITURE,
        center=(-1.7, 0.0, 0.3),
        size=(1.4, 2.0, 0.6),
    )
    nightstand = _box_object(
        "nightstand_0",
        "nightstand",
        ObjectType.FURNITURE,
        center=(-0.55, -0.75, 0.35),
        size=(0.5, 0.5, 0.7),
    )
    nightstand.support_surfaces = [
        SupportSurface(
            surface_id=UniqueID("nightstand_top"),
            bounding_box_min=np.array([-0.25, -0.25, 0.0]),
            bounding_box_max=np.array([0.25, 0.25, 0.0]),
            transform=RigidTransform(p=[-0.55, -0.75, 0.7]),
        )
    ]
    mug = _box_object(
        "mug_0",
        "coffee mug",
        ObjectType.MANIPULAND,
        center=(-0.55, -0.75, 0.82),
        size=(0.16, 0.16, 0.2),
    )
    mug.placement_info = PlacementInfo(
        parent_surface_id=UniqueID("nightstand_top"),
        position_2d=np.array([0.0, 0.0]),
        rotation_2d=0.0,
    )
    desk = _box_object(
        "desk_0",
        "writing desk",
        ObjectType.FURNITURE,
        center=(1.25, 0.35, 0.4),
        size=(1.1, 0.65, 0.8),
    )
    chair = _box_object(
        "chair_0",
        "office chair",
        ObjectType.FURNITURE,
        center=(1.25, -0.45, 0.45),
        size=(0.55, 0.55, 0.9),
    )
    for obj in (bed, nightstand, mug, desk, chair):
        scene.add_object(obj)

    payload = evaluate_room_scene(
        scene,
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["spatial_accessibility", "functional_dependency"],
                "fd_relation_proposer_mode": "template",
                "max_fd_relation_proposals": 16,
            }
        },
        stage="scene_after_furniture",
    )
    checks = payload["case_pack"]["checks"]
    results = payload["results"]

    assert payload["scope"] == "room:main"
    assert {"spatial_accessibility", "functional_dependency"} <= {
        result["metric"] for result in results
    }
    assert any(
        check["metric"] == "functional_dependency"
        and check["subject_id"] == "mug_0"
        and check["target_ids"] == ["nightstand_0"]
        for check in checks
    )
    assert any(
        check.get("check_source") == "fd_relation_proposer"
        and check.get("relation_type") == "seating_to_work_surface"
        and check.get("subject_id") == "chair_0"
        and check.get("target_ids") == ["desk_0"]
        for check in checks
    )
    assert (
        payload["summary"]["metric_summary"]["spatial_accessibility"]["total_checks"]
        >= 3
    )
    assert (
        payload["summary"]["metric_summary"]["functional_dependency"]["total_checks"]
        >= 2
    )


def test_spatial_accessibility_checks_manipuland_placeable_objects(
    tmp_path: Path,
) -> None:
    payload = evaluate_room_scene(
        _scene(tmp_path),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["spatial_accessibility"],
            }
        },
        stage="final_scene",
    )

    subjects = {
        result["primary_object"]
        for result in payload["results"]
        if result["metric"] == "spatial_accessibility"
    }
    assert "mug_0" in subjects


def test_spatial_accessibility_promotes_cached_ignored_manipuland_policy(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    mug = scene.objects[UniqueID("mug_0")]
    mug.metadata["functional_hints"] = {
        "functional_categories": ["graspable"],
        "scene_object_type": "manipuland",
        "accessibility_policy": "ignored",
    }

    case_pack = room_scene_to_case_pack(
        scene,
        stage="final_scene",
        metrics=["spatial_accessibility"],
    )
    mug_record = next(
        obj for obj in case_pack["scene_geometry"]["objects"] if obj["id"] == "mug_0"
    )
    checked_subjects = {
        check["subject_id"]
        for check in case_pack["checks"]
        if check["metric"] == "spatial_accessibility"
    }

    assert mug_record["functional_hints"]["accessibility_policy"] == "required"
    assert "mug_0" in checked_subjects


def test_spatial_accessibility_uses_grid_reach_diagnostics(tmp_path: Path) -> None:
    payload = evaluate_room_scene(
        _scene(tmp_path),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["spatial_accessibility"],
            }
        },
        stage="final_scene",
    )

    sa_result = next(
        result
        for result in payload["results"]
        if result["metric"] == "spatial_accessibility"
    )
    diagnostics = sa_result["diagnostics"]
    assert "access_ratio" in diagnostics
    assert "min_reach_distance_m" in diagnostics
    assert "zone_scores" in diagnostics


def test_rule_spatial_accessibility_open_room_is_accessible() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"]["functional_categories"] = ["sittable"]
    check = {
        "check_id": "sa_open_room",
        "metric": "spatial_accessibility",
        "subject_id": "chair_1",
        "affordance": "sittable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair], [check]),
        metrics=["spatial_accessibility"],
    )

    result = next(item for item in results if item["check_id"] == "sa_open_room")
    assert result["label"] == "pass"
    assert result["blocking_objects"] == []


def test_rule_spatial_accessibility_missing_subject_is_unknown() -> None:
    check = {
        "check_id": "sa_missing_subject",
        "metric": "spatial_accessibility",
        "subject_id": "missing_chair",
        "affordance": "sittable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([], [check]),
        metrics=["spatial_accessibility"],
    )

    result = next(item for item in results if item["check_id"] == "sa_missing_subject")
    assert result["label"] == "unknown"
    assert "could not find subject object" in result["reason"]


def test_rule_spatial_accessibility_blocked_access_zone_fails() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"]["functional_categories"] = ["sittable"]
    blockers = [
        _benchmark_obj("front_block", "cabinet", (2.0, 2.85, 0.5), (1.2, 0.8, 1.0)),
        _benchmark_obj("left_block", "cabinet", (1.15, 2.0, 0.5), (0.8, 1.2, 1.0)),
        _benchmark_obj("right_block", "cabinet", (2.85, 2.0, 0.5), (0.8, 1.2, 1.0)),
    ]
    check = {
        "check_id": "sa_blocked",
        "metric": "spatial_accessibility",
        "subject_id": "chair_1",
        "affordance": "sittable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, *blockers], [check]),
        metrics=["spatial_accessibility"],
    )

    result = next(item for item in results if item["check_id"] == "sa_blocked")
    assert result["label"] == "fail"
    assert result["blocking_objects"]


def test_rule_spatial_accessibility_partially_blocked_zone_is_degraded() -> None:
    cabinet = _benchmark_obj("cabinet_1", "cabinet", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    cabinet["functional_hints"]["functional_categories"] = ["openable"]
    blocker = _benchmark_obj(
        "partial_block", "cabinet", (1.45, 2.45, 0.5), (0.8, 0.2, 1.0)
    )
    check = {
        "check_id": "sa_partial",
        "metric": "spatial_accessibility",
        "subject_id": "cabinet_1",
        "affordance": "openable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([cabinet, blocker], [check]),
        metrics=["spatial_accessibility"],
    )

    result = next(item for item in results if item["check_id"] == "sa_partial")
    assert result["label"] == "degraded"


def test_rule_spatial_accessibility_threshold_overrides_affect_label() -> None:
    cabinet = _benchmark_obj("cabinet_1", "cabinet", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    cabinet["functional_hints"]["functional_categories"] = ["openable"]
    blocker = _benchmark_obj(
        "partial_block", "cabinet", (2.85, 2.35, 0.5), (0.8, 0.2, 1.0)
    )
    check = {
        "check_id": "sa_threshold_override",
        "metric": "spatial_accessibility",
        "subject_id": "cabinet_1",
        "affordance": "openable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([cabinet, blocker], [check]),
        metrics=["spatial_accessibility"],
        extra={
            "accessibility_pass_ratio": 0.01,
            "accessibility_degraded_ratio": 0.0,
        },
    )

    result = next(
        item for item in results if item["check_id"] == "sa_threshold_override"
    )
    assert result["label"] == "pass"


def test_rule_spatial_accessibility_ignores_small_and_high_blockers() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"]["functional_categories"] = ["sittable"]
    spoon = _benchmark_obj("tablespoon_1", "table", (2.9, 2.0, 0.25), (0.04, 0.04, 0.2))
    spoon["category"] = "room_tablespoon"
    pendant = _benchmark_obj(
        "pendant_light_1", "pendant_light", (2.0, 2.9, 2.1), (0.3, 0.3, 0.3)
    )
    sofa = _benchmark_obj("sofa_1", "sofa", (1.25, 2.0, 0.5), (1.0, 0.7, 1.0))
    check = {
        "check_id": "sa_small_high",
        "metric": "spatial_accessibility",
        "subject_id": "chair_1",
        "affordance": "sittable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, spoon, pendant, sofa], [check]),
        metrics=["spatial_accessibility"],
    )

    result = next(item for item in results if item["check_id"] == "sa_small_high")
    assert result["label"] == "pass"
    assert "tablespoon_1" not in result["blocking_objects"]
    assert "pendant_light_1" not in result["blocking_objects"]


def test_rule_spatial_accessibility_reports_limiting_agent_profile() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"]["functional_categories"] = ["sittable"]
    blocker = _benchmark_obj(
        "narrow_block", "cabinet", (2.9, 2.0, 0.5), (0.25, 1.2, 1.0)
    )
    check = {
        "check_id": "sa_profiles",
        "metric": "spatial_accessibility",
        "subject_id": "chair_1",
        "affordance": "sittable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, blocker], [check]),
        metrics=["spatial_accessibility"],
        extra={
            "accessibility_agent_profiles": [
                {
                    "id": "adult",
                    "clearance_width_m": 0.35,
                    "reach_radius_m": 0.9,
                    "arm_origin_height_m": 1.0,
                },
                {
                    "id": "wheelchair",
                    "clearance_width_m": 0.95,
                    "reach_radius_m": 0.7,
                    "arm_origin_height_m": 0.9,
                },
            ]
        },
    )

    result = next(item for item in results if item["check_id"] == "sa_profiles")
    per_profile = result["diagnostics"]["per_profile"]
    assert "adult" in per_profile
    assert "wheelchair" in per_profile
    assert result["label"] == per_profile["wheelchair"]["label"]


def test_rule_spatial_accessibility_fails_when_target_is_out_of_reach() -> None:
    shelf = _benchmark_obj("shelf_1", "shelf", (2.0, 2.0, 1.6), (0.6, 0.6, 3.2))
    shelf["functional_hints"]["functional_categories"] = ["supportable"]
    shelf["interaction_height_m"] = {"supportable": 2.8}
    check = {
        "check_id": "sa_reach",
        "metric": "spatial_accessibility",
        "subject_id": "shelf_1",
        "affordance": "supportable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([shelf], [check]),
        metrics=["spatial_accessibility"],
        extra={
            "accessibility_agent_profiles": [
                {
                    "id": "child",
                    "clearance_width_m": 0.4,
                    "reach_radius_m": 0.45,
                    "arm_origin_height_m": 0.9,
                }
            ]
        },
    )

    result = next(item for item in results if item["check_id"] == "sa_reach")
    assert result["label"] == "fail"
    assert result["diagnostics"]["min_reach_distance_m"] > 0.45


def test_rule_spatial_accessibility_crouch_posture_reaches_low_support() -> None:
    low_shelf = _benchmark_obj(
        "low_shelf_1", "shelf", (2.0, 2.0, 0.25), (0.6, 0.6, 0.5)
    )
    low_shelf["functional_hints"]["functional_categories"] = ["supportable"]
    low_shelf["interaction_height_m"] = {"supportable": 0.50}
    check = {
        "check_id": "sa_crouch",
        "metric": "spatial_accessibility",
        "subject_id": "low_shelf_1",
        "affordance": "supportable",
    }

    standing_result = _run_direct_case_pack(
        _benchmark_case_pack([low_shelf], [check]),
        metrics=["spatial_accessibility"],
        extra={
            "accessibility_agent_profiles": [
                {
                    "id": "standing_only",
                    "clearance_width_m": 0.4,
                    "reach_radius_m": 0.40,
                    "arm_origin_height_m": 1.10,
                    "crouch_factor": 0.0,
                }
            ]
        },
    )[0]
    crouch_result = _run_direct_case_pack(
        _benchmark_case_pack([low_shelf], [check]),
        metrics=["spatial_accessibility"],
        extra={
            "accessibility_agent_profiles": [
                {
                    "id": "crouch_enabled",
                    "clearance_width_m": 0.4,
                    "reach_radius_m": 0.40,
                    "arm_origin_height_m": 1.10,
                }
            ]
        },
    )[0]

    assert standing_result["label"] == "fail"
    assert standing_result["diagnostics"]["reach_posture"] == "standing"
    assert crouch_result["label"] == "pass"
    assert crouch_result["diagnostics"]["reach_posture"] == "crouch"
    assert "crouch/lean" in crouch_result["reason"]


def test_rule_spatial_accessibility_fails_unreachable_manipuland() -> None:
    island = _benchmark_obj("island_1", "table", (2.5, 2.5, 0.4), (3.0, 3.0, 0.8))
    island["functional_hints"]["functional_categories"] = ["supportable"]
    mug = _benchmark_obj("mug_1", "mug", (2.5, 2.5, 0.9), (0.12, 0.12, 0.18))
    mug["object_type"] = "manipuland"
    mug["functional_hints"] = {
        "functional_categories": ["graspable"],
        "scene_object_type": "manipuland",
        "accessibility_policy": "required",
    }
    check = {
        "check_id": "sa_manipuland_reach",
        "metric": "spatial_accessibility",
        "subject_id": "mug_1",
        "affordance": "graspable",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([island, mug], [check]),
        metrics=["spatial_accessibility"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "fail"
    assert result["diagnostics"]["access_side"] == "connected_floor"
    assert result["diagnostics"]["min_reach_distance_m"] > 0.75


def test_rule_functional_dependency_fails_when_chair_back_faces_desk() -> None:
    chair = _benchmark_obj(
        "chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0), yaw=180.0
    )
    desk = _benchmark_obj("desk_1", "desk", (2.9, 2.0, 0.4), (1.0, 0.8, 0.8))
    check = {
        "check_id": "fd_back_facing",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["desk_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, desk], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_back_facing")
    assert result["label"] == "fail"


def test_dependency_annotation_checks_generate_orientation_and_wall_relations() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"].update(
        {
            "functional_categories": ["sittable"],
            "orientation_dependencies": [
                {
                    "relation_type": "seat_faces_surface",
                    "target_kind": "object",
                    "target_category": ["desk", "table"],
                    "max_distance_m": 1.5,
                }
            ],
        }
    )
    desk = _benchmark_obj("desk_1", "desk", (2.9, 2.0, 0.4), (1.0, 0.8, 0.8))
    bed = _benchmark_obj("bed_1", "bed", (2.5, 4.35, 0.35), (1.2, 0.8, 0.7))
    bed["functional_hints"].update(
        {
            "functional_categories": ["sleepable"],
            "attachment_dependencies": [
                {
                    "relation_type": "back_against_wall",
                    "target_kind": "architecture",
                    "target_category": "wall",
                    "subject_face": "back",
                    "max_distance_m": 0.25,
                }
            ],
        }
    )
    wall = _benchmark_obj("wall_n", "wall", (2.5, 4.95, 1.5), (5.0, 0.1, 3.0))

    checks = build_checks(
        _benchmark_case_pack([chair, desk, bed, wall]),
        metrics=["functional_dependency"],
    )

    annotation_checks = {
        check["relation_type"]: check
        for check in checks
        if str(check.get("check_source", "")).startswith("asset_")
    }
    assert annotation_checks["seat_faces_surface"]["target_ids"] == ["desk_1"]
    assert annotation_checks["back_against_wall"]["target_ids"] == ["wall_n"]


def test_dependency_annotation_checks_keep_wall_relation_beyond_threshold() -> None:
    bed = _benchmark_obj("bed_1", "bed", (2.5, 4.35, 0.35), (1.2, 0.8, 0.7), yaw=180.0)
    bed["functional_hints"].update(
        {
            "functional_categories": ["sleepable"],
            "attachment_dependencies": [
                {
                    "relation_type": "back_against_wall",
                    "target_kind": "architecture",
                    "target_category": "wall",
                    "subject_face": "back",
                    "max_angle_deg": 45.0,
                    "max_distance_m": 0.25,
                }
            ],
        }
    )
    wall = _benchmark_obj("wall_n", "wall", (2.5, 5.055, 1.5), (5.0, 0.1, 3.0))

    case_pack = _benchmark_case_pack([bed, wall])
    checks = build_checks(case_pack, metrics=["functional_dependency"])

    wall_check = next(
        check
        for check in checks
        if check.get("relation_type") == "back_against_wall"
        and check.get("subject_id") == "bed_1"
    )
    assert wall_check["target_ids"] == ["wall_n"]

    result = _run_direct_case_pack(
        _benchmark_case_pack([bed, wall], [wall_check]),
        metrics=["functional_dependency"],
    )[0]
    assert result["label"] == "fail"


def test_dependency_annotation_checks_skip_floor_attachment_for_surface_object() -> (
    None
):
    table = _benchmark_obj("table_1", "dining_table", (2.0, 2.0, 0.35), (1.2, 0.8, 0.7))
    table["functional_hints"]["functional_categories"] = ["supportable"]
    table["support_regions"] = [
        {
            "region_id": "S_table",
            "support_kind": "top_surface",
            "height_world_z": 0.7,
            "polygon_world_xy": [
                [1.4, 1.6],
                [2.6, 1.6],
                [2.6, 2.4],
                [1.4, 2.4],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    vase = _benchmark_obj("vase_1", "vase", (2.0, 2.0, 0.8), (0.1, 0.1, 0.2))
    vase["placement_info"] = {
        "parent_surface_id": "S_table",
        "placement_method": "surface_placement",
    }
    vase["functional_hints"].update(
        {
            "scene_object_type": "manipuland",
            "placement_class": "surface_object",
            "attachment_dependencies": [
                {
                    "relation_type": "object_on_floor",
                    "target_kind": "architecture",
                    "target_category": "floor",
                }
            ],
        }
    )
    floor = _benchmark_obj("floor_1", "floor", (2.0, 2.0, -0.05), (5.0, 5.0, 0.1))

    checks = build_checks(
        _benchmark_case_pack([table, vase, floor]),
        metrics=["functional_dependency"],
    )

    vase_checks = [check for check in checks if check["subject_id"] == "vase_1"]
    assert {check["relation_type"] for check in vase_checks} == {"object_on_support"}
    assert vase_checks[0]["target_ids"] == ["table_1"]


def test_dependency_annotation_skips_nondirectional_front_faces_noise() -> None:
    coffee_table = _benchmark_obj(
        "coffee_table_1", "coffee_table", (2.0, 2.0, 0.25), (1.0, 0.55, 0.5)
    )
    coffee_table["functional_hints"]["orientation_dependencies"] = [
        {
            "relation_type": "front_faces",
            "target_kind": "object",
            "target_category": ["sofa"],
            "subject_face": "front",
            "max_distance_m": 2.0,
        }
    ]
    sofa = _benchmark_obj("sofa_1", "sofa", (2.0, 3.0, 0.4), (1.6, 0.8, 0.8))
    sofa["functional_hints"].update(
        {
            "category_group": "seating",
            "orientation_dependencies": [
                {
                    "relation_type": "front_faces",
                    "target_kind": "object",
                    "target_category": ["coffee_table"],
                    "subject_face": "front",
                    "max_distance_m": 2.0,
                }
            ],
        }
    )

    checks = build_checks(
        _benchmark_case_pack([coffee_table, sofa]),
        metrics=["functional_dependency"],
    )
    orientation_checks = [
        check
        for check in checks
        if check.get("check_source") == "asset_orientation_dependency"
    ]

    observed = [
        (check["subject_id"], check["relation_type"]) for check in orientation_checks
    ]
    assert observed == [("sofa_1", "furniture_faces_furniture")]


def test_rule_functional_dependency_seat_faces_surface_passes_and_fails() -> None:
    desk = _benchmark_obj("desk_1", "desk", (2.9, 2.0, 0.4), (1.0, 0.8, 0.8))
    check = {
        "check_id": "fd_seat_faces_surface",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["desk_1"],
        "relation_type": "seat_faces_surface",
    }

    pass_chair = _benchmark_obj(
        "chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0), yaw=-90.0
    )
    fail_chair = _benchmark_obj(
        "chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0), yaw=90.0
    )

    pass_result = _run_direct_case_pack(
        _benchmark_case_pack([pass_chair, desk], [check]),
        metrics=["functional_dependency"],
    )[0]
    fail_result = _run_direct_case_pack(
        _benchmark_case_pack([fail_chair, desk], [check]),
        metrics=["functional_dependency"],
    )[0]

    assert pass_result["relation_type"] == "seat_faces_surface"
    assert pass_result["label"] == "pass"
    assert fail_result["label"] == "fail"


def test_rule_functional_dependency_display_faces_user_uses_face_orientation() -> None:
    alarm_clock = _benchmark_obj(
        "alarm_clock_1", "alarm_clock", (2.0, 2.0, 0.6), (0.2, 0.1, 0.12), yaw=0.0
    )
    bed = _benchmark_obj("bed_1", "bed", (2.0, 2.8, 0.45), (1.2, 1.0, 0.9))
    check = {
        "check_id": "fd_alarm_display_faces_bed",
        "metric": "functional_dependency",
        "subject_id": "alarm_clock_1",
        "target_ids": ["bed_1"],
        "relation_type": "display_faces_user",
        "evidence": {
            "dependency": {
                "relation_type": "display_faces_user",
                "subject_face": "front",
                "target_face": "any",
                "max_angle_deg": 45,
                "max_distance_m": 1.0,
            }
        },
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([alarm_clock, bed], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "pass"
    assert "`display_faces_user` holds" in result["reason"]
    assert "support region" not in result["reason"]


def test_rule_functional_dependency_back_against_wall_passes_and_fails() -> None:
    wall = _benchmark_obj("wall_n", "wall", (2.5, 4.95, 1.5), (5.0, 0.1, 3.0))
    check = {
        "check_id": "fd_back_against_wall",
        "metric": "functional_dependency",
        "subject_id": "bed_1",
        "target_ids": ["wall_n"],
        "relation_type": "back_against_wall",
        "evidence": {
            "dependency": {
                "subject_face": "back",
                "max_angle_deg": 45.0,
                "max_distance_m": 0.25,
            }
        },
    }
    pass_bed = _benchmark_obj(
        "bed_1", "bed", (2.5, 4.35, 0.35), (1.2, 0.8, 0.7), yaw=180.0
    )
    fail_bed = _benchmark_obj(
        "bed_1", "bed", (2.5, 4.35, 0.35), (1.2, 0.8, 0.7), yaw=90.0
    )

    pass_result = _run_direct_case_pack(
        _benchmark_case_pack([pass_bed, wall], [check]),
        metrics=["functional_dependency"],
    )[0]
    fail_result = _run_direct_case_pack(
        _benchmark_case_pack([fail_bed, wall], [check]),
        metrics=["functional_dependency"],
    )[0]

    assert pass_result["label"] == "pass"
    assert fail_result["label"] == "fail"


def test_rule_functional_dependency_wall_mounted_thin_axis_fallback() -> None:
    wall = _benchmark_obj("north_wall", "wall", (0.0, 2.0, 1.5), (4.0, 0.1, 3.0))
    check = {
        "check_id": "fd_wall_art_mount",
        "metric": "functional_dependency",
        "subject_id": "wall_art_1",
        "target_ids": ["north_wall"],
        "relation_type": "mounted_to_wall",
        "evidence": {
            "dependency": {
                "subject_face": "back",
                "max_angle_deg": 10.0,
                "max_distance_m": 0.05,
            }
        },
    }
    wall_art = _benchmark_obj(
        "wall_art_1",
        "wall_art",
        (0.0, 1.93, 1.4),
        (0.8, 0.04, 0.5),
        yaw=0.0,
    )
    wall_art["functional_hints"]["scene_object_type"] = "wall_mounted"
    furniture_panel = _benchmark_obj(
        "wall_art_1",
        "wall_art",
        (0.0, 1.93, 1.4),
        (0.8, 0.04, 0.5),
        yaw=0.0,
    )
    furniture_panel["functional_hints"]["scene_object_type"] = "furniture"

    wall_mounted_result = _run_direct_case_pack(
        _benchmark_case_pack([wall_art, wall], [check]),
        metrics=["functional_dependency"],
    )[0]
    furniture_result = _run_direct_case_pack(
        _benchmark_case_pack([furniture_panel, wall], [check]),
        metrics=["functional_dependency"],
    )[0]

    assert wall_mounted_result["label"] == "pass"
    assert "thin wall-mounted footprint" in wall_mounted_result["reason"]
    assert furniture_result["label"] == "fail"


def test_rule_functional_dependency_wall_mounted_shelf_projection_fallback() -> None:
    wall = _benchmark_obj("west_wall", "wall", (0.0, 2.0, 1.5), (0.1, 4.0, 3.0))
    shelf = _benchmark_obj(
        "floating_shelf_1",
        "shelf",
        (0.14, 2.0, 1.2),
        (0.18, 0.8, 0.04),
        yaw=90.0,
    )
    shelf["functional_hints"].update(
        {
            "scene_object_type": "wall_mounted",
            "category_group": "storage_surface",
        }
    )
    check = {
        "check_id": "fd_floating_shelf_mount",
        "metric": "functional_dependency",
        "subject_id": "floating_shelf_1",
        "target_ids": ["west_wall"],
        "relation_type": "mounted_to_wall",
        "evidence": {
            "dependency": {
                "subject_face": "back",
                "max_angle_deg": 5.0,
                "max_distance_m": 0.05,
            }
        },
    }

    result = _run_direct_case_pack(
        _benchmark_case_pack([shelf, wall], [check]),
        metrics=["functional_dependency"],
    )[0]

    assert result["label"] == "pass"
    assert "wall-mounted footprint is flush" in result["reason"]


def test_rule_functional_dependency_storage_backed_by_wall_footprint_fallback() -> None:
    wall = _benchmark_obj("east_wall", "wall", (2.5, 0.0, 1.5), (0.1, 4.0, 3.0))
    bookshelf = _benchmark_obj(
        "bookshelf_1",
        "bookshelf",
        (2.30, 0.0, 0.8),
        (0.35, 1.0, 1.6),
        yaw=0.0,
    )
    bookshelf["functional_hints"].update(
        {
            "scene_object_type": "furniture",
            "category_group": "storage_surface",
        }
    )
    check = {
        "check_id": "fd_bookshelf_wall",
        "metric": "functional_dependency",
        "subject_id": "bookshelf_1",
        "target_ids": ["east_wall"],
        "relation_type": "back_against_wall",
        "evidence": {
            "dependency": {
                "subject_face": "back",
                "max_angle_deg": 5.0,
                "max_distance_m": 0.05,
            }
        },
    }

    result = _run_direct_case_pack(
        _benchmark_case_pack([bookshelf, wall], [check]),
        metrics=["functional_dependency"],
    )[0]

    assert result["label"] == "pass"
    assert "storage/work furniture footprint is flush" in result["reason"]


def test_rule_functional_dependency_sideboard_back_against_north_wall_passes() -> None:
    wall = _benchmark_obj("north_wall", "wall", (0.0, 2.0, 1.5), (4.0, 0.1, 3.0))
    check = {
        "check_id": "fd_sideboard_back_against_wall",
        "metric": "functional_dependency",
        "subject_id": "sideboard_1",
        "target_ids": ["north_wall"],
        "relation_type": "back_against_wall",
        "evidence": {
            "dependency": {
                "subject_face": "back",
                "max_angle_deg": 15.0,
                "max_distance_m": 0.05,
            }
        },
    }
    pass_sideboard = _benchmark_obj(
        "sideboard_1",
        "sideboard",
        (1.6, 1.65, 0.35),
        (1.2, 0.6, 0.7),
        yaw=-90.0,
    )
    fail_sideboard = _benchmark_obj(
        "sideboard_1",
        "sideboard",
        (1.6, 1.65, 0.35),
        (1.2, 0.6, 0.7),
        yaw=90.0,
    )

    pass_result = _run_direct_case_pack(
        _benchmark_case_pack([pass_sideboard, wall], [check]),
        metrics=["functional_dependency"],
    )[0]
    fail_result = _run_direct_case_pack(
        _benchmark_case_pack([fail_sideboard, wall], [check]),
        metrics=["functional_dependency"],
    )[0]

    assert pass_result["label"] == "pass"
    assert fail_result["label"] == "fail"


def test_adapter_front_vector_uses_y_axis_canonical_convention() -> None:
    scene = _scene(Path("/tmp/scenesmith_front_vector_test"))
    scene.objects.clear()
    sideboard = _box_object(
        "sideboard_0",
        "sideboard cabinet",
        ObjectType.FURNITURE,
        center=(0.0, 0.0, 0.35),
        size=(1.2, 0.6, 0.7),
        yaw_deg=180.0,
    )
    sideboard.metadata["functional_hints"] = {
        "functional_categories": ["openable", "supportable", "storage"],
        "candidate_affordances": ["openable", "supportable", "storage"],
        "front_hint": "front",
    }
    scene.add_object(sideboard)

    case_pack = room_scene_to_case_pack(
        scene,
        stage="debug",
        metrics=["functional_dependency"],
    )
    exported = next(
        obj
        for obj in case_pack["scene_geometry"]["objects"]
        if obj["id"] == "sideboard_0"
    )
    front_face = next(
        face for face in exported["interaction_faces"] if face["name"] == "front"
    )

    assert front_face["normal_xy"][0] == pytest.approx(0.0, abs=1e-6)
    assert front_face["normal_xy"][1] == pytest.approx(-1.0, abs=1e-6)


def test_rule_functional_dependency_front_hint_rotates_seating_orientation() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    chair["functional_hints"].update(
        {
            "functional_categories": ["sittable"],
            "classification_source": "asset_annotation",
            "front_hint": "left",
        }
    )
    table = _benchmark_obj("table_1", "dining_table", (2.0, 2.9, 0.4), (1.2, 0.8, 0.8))
    check = {
        "check_id": "fd_front_hint_left",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["table_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, table], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_front_hint_left")
    assert result["label"] == "pass"
    assert "depth-axis fallback" not in result["reason"]
    assert "facing angle 0deg" in result["reason"]


def test_rule_functional_dependency_dining_chair_depth_axis_fallback() -> None:
    chair = _benchmark_obj(
        "chair_1", "chair", (2.0, 2.0, 0.5), (0.5, 0.85, 1.0), yaw=180.0
    )
    chair["functional_hints"].update(
        {
            "functional_categories": ["sittable"],
            "classification_source": "heuristic",
        }
    )
    table = _benchmark_obj("table_1", "dining_table", (2.9, 2.0, 0.4), (1.2, 0.8, 0.8))
    check = {
        "check_id": "fd_dining_axis_fallback",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["table_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, table], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_dining_axis_fallback"
    )
    assert result["label"] == "pass"
    assert "depth-axis fallback" in result["reason"]


def test_rule_functional_dependency_long_table_uses_edge_fallback() -> None:
    chair = _benchmark_obj(
        "chair_1", "chair", (3.15, 3.31, 0.5), (0.5, 0.85, 1.0), yaw=180.0
    )
    chair["functional_hints"].update(
        {
            "functional_categories": ["sittable"],
            "classification_source": "asset_annotation",
            "front_hint": "back",
        }
    )
    table = _benchmark_obj(
        "table_1", "dining_table", (3.06, 3.29, 0.4), (2.61, 0.81, 0.8)
    )
    check = {
        "check_id": "fd_long_table_edge_fallback",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["table_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, table], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_long_table_edge_fallback"
    )
    assert result["label"] == "pass"
    assert "table-edge fallback" in result["reason"]


def test_rule_functional_dependency_long_table_uses_nearest_edge_per_chair() -> None:
    table = _benchmark_obj("table_1", "dining_table", (2.5, 2.5, 0.4), (2.4, 0.8, 0.8))
    table["footprint_world"] = [
        [1.3, 2.1],
        [3.7, 2.1],
        [3.7, 2.9],
        [1.3, 2.9],
    ]
    chair_south = _benchmark_obj(
        "chair_south", "chair", (2.0, 1.75, 0.5), (0.5, 0.5, 1.0), yaw=90.0
    )
    chair_north = _benchmark_obj(
        "chair_north", "chair", (3.0, 3.25, 0.5), (0.5, 0.5, 1.0), yaw=-90.0
    )
    for chair in (chair_south, chair_north):
        chair["functional_hints"].update(
            {
                "functional_categories": ["sittable"],
                "classification_source": "asset_annotation",
                "front_hint": "front",
            }
        )
    checks = [
        {
            "check_id": "fd_chair_south",
            "metric": "functional_dependency",
            "subject_id": "chair_south",
            "target_ids": ["table_1"],
            "relation_type": "seating_to_work_surface",
        },
        {
            "check_id": "fd_chair_north",
            "metric": "functional_dependency",
            "subject_id": "chair_north",
            "target_ids": ["table_1"],
            "relation_type": "seating_to_work_surface",
        },
    ]

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, chair_south, chair_north], checks),
        metrics=["functional_dependency"],
    )

    by_check = {result["check_id"]: result for result in results}
    assert by_check["fd_chair_south"]["label"] == "pass"
    assert by_check["fd_chair_north"]["label"] == "pass"
    assert "table-edge fallback" in by_check["fd_chair_south"]["reason"]
    assert "table-edge fallback" in by_check["fd_chair_north"]["reason"]


def test_rule_functional_dependency_allows_oblique_media_view() -> None:
    sofa = _benchmark_obj("sofa_1", "sofa", (2.5, 2.5, 0.5), (2.0, 0.9, 1.0), yaw=180.0)
    sofa["functional_hints"]["functional_categories"] = ["sittable"]
    television = _benchmark_obj("tv_1", "television", (2.5, 0.0, 1.2), (1.2, 0.1, 0.7))
    check = {
        "check_id": "fd_sofa_tv_oblique",
        "metric": "functional_dependency",
        "subject_id": "sofa_1",
        "target_ids": ["tv_1"],
        "relation_type": "seating_to_media",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([sofa, television], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_sofa_tv_oblique")
    assert result["label"] == "pass"
    assert "usable media view" in result["reason"]


def test_rule_functional_dependency_standalone_chair_ignores_far_counter() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    counter = _benchmark_obj(
        "counter_1", "counter_a_rectangular", (3.43, 2.0, 0.45), (1.0, 0.8, 0.9)
    )
    check = {
        "check_id": "fd_far_counter",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["counter_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, counter], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_far_counter")
    assert result["label"] == "unknown"
    assert "chair can stand alone" in result["reason"]


def test_rule_functional_dependency_selects_valid_multi_target() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    poster = _benchmark_obj("poster_1", "poster", (2.5, 2.0, 1.5), (0.2, 0.05, 0.5))
    desk = _benchmark_obj("desk_1", "desk", (2.9, 2.0, 0.4), (1.0, 0.8, 0.8))
    check = {
        "check_id": "fd_multi",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["poster_1", "desk_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, poster, desk], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_multi")
    assert result["label"] == "pass"
    assert result["diagnostics"]["selected_target_ids"] == ["desk_1"]
    assert result["selected_related_objects"] == ["desk_1"]


def test_rule_functional_dependency_scores_dining_set_multiple_targets() -> None:
    table = _benchmark_obj("table_1", "dining_table", (2.5, 2.5, 0.4), (1.2, 0.8, 0.8))
    chair_a = _benchmark_obj("chair_a", "chair", (1.6, 2.5, 0.5), (0.6, 0.6, 1.0))
    chair_b = _benchmark_obj(
        "chair_b", "chair", (3.4, 2.5, 0.5), (0.6, 0.6, 1.0), yaw=180.0
    )
    check = {
        "check_id": "fd_dining",
        "metric": "functional_dependency",
        "subject_id": "table_1",
        "target_ids": ["chair_a", "chair_b"],
        "relation_type": "dining_set",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, chair_a, chair_b], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_dining")
    assert result["label"] == "pass"
    assert result["diagnostics"]["cardinality_score"] >= 1.0


def test_rule_functional_dependency_support_regions_allow_internal_shelf() -> None:
    bookshelf = _benchmark_obj(
        "bookshelf_1", "bookshelf", (2.0, 2.0, 0.75), (0.4, 1.2, 1.5)
    )
    bookshelf["support_regions"] = [
        {
            "region_id": "layer_1",
            "support_kind": "internal_shelf",
            "height_world_z": 0.62,
            "polygon_world_xy": [
                [1.8, 1.4],
                [2.2, 1.4],
                [2.2, 2.6],
                [1.8, 2.6],
            ],
            "clearance_above_m": 0.45,
            "access_type": "front_open",
        }
    ]
    book = _benchmark_obj("book_1", "book", (2.0, 2.0, 0.77), (0.2, 0.18, 0.3))
    check = {
        "check_id": "fd_internal_region",
        "metric": "functional_dependency",
        "subject_id": "book_1",
        "target_ids": ["bookshelf_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([bookshelf, book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_internal_region")
    assert result["label"] == "pass"
    assert "matched internal_shelf" in result["reason"]
    assert "layer_1" in result["reason"]


def test_rule_functional_dependency_support_regions_allow_top_surface() -> None:
    bookshelf = _benchmark_obj(
        "bookshelf_1", "bookshelf", (2.0, 2.0, 0.75), (0.4, 1.2, 1.5)
    )
    bookshelf["support_regions"] = [
        {
            "region_id": "top",
            "support_kind": "top_surface",
            "height_world_z": 1.5,
            "polygon_world_xy": [
                [1.8, 1.4],
                [2.2, 1.4],
                [2.2, 2.6],
                [1.8, 2.6],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    book = _benchmark_obj("book_1", "book", (2.0, 2.0, 1.65), (0.2, 0.18, 0.3))
    check = {
        "check_id": "fd_top_region",
        "metric": "functional_dependency",
        "subject_id": "book_1",
        "target_ids": ["bookshelf_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([bookshelf, book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_top_region")
    assert result["label"] == "pass"
    assert "matched top_surface" in result["reason"]


def test_rule_functional_dependency_internal_shelf_prefers_unknown() -> None:
    table = _benchmark_obj("table_1", "table", (2.0, 2.0, 0.235), (0.68, 1.14, 0.47))
    table["functional_hints"].update(
        {
            "functional_categories": ["supportable"],
            "candidate_affordances": ["supportable"],
            "category_group": "work_surface",
            "category_keywords": ["table", "work table"],
            "access_type": {"primary": "top", "secondary": "perimeter"},
            "interaction_surface_map": {
                "front": ["lower shelf access", "edge contact"],
                "top": ["support", "place objects"],
            },
            "placement_class": "floor",
            "benchmark_relevance": "functional",
            "classification_source": "asset_annotation",
        }
    )
    book = _benchmark_obj("book_1", "book", (2.0, 2.0, 0.17), (0.26, 0.24, 0.08))
    check = {
        "check_id": "fd_support_unknown",
        "metric": "functional_dependency",
        "subject_id": "book_1",
        "target_ids": ["table_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_support_unknown")
    assert result["label"] == "unknown"
    assert "internal shelf height" in result["reason"]


def test_rule_functional_dependency_conservative_top_polygon_uses_bbox_fallback() -> (
    None
):
    cabinet = _benchmark_obj("cabinet_1", "cabinet", (2.0, 2.0, 0.5), (1.0, 0.6, 1.0))
    cabinet["support_regions"] = [
        {
            "region_id": "top",
            "support_kind": "top_surface",
            "height_world_z": 1.0,
            "polygon_world_xy": [
                [1.9, 1.75],
                [2.1, 1.75],
                [2.1, 2.05],
                [1.9, 2.05],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    book = _benchmark_obj("book_1", "book", (2.35, 2.2, 1.04), (0.2, 0.16, 0.08))
    check = {
        "check_id": "fd_conservative_top",
        "metric": "functional_dependency",
        "subject_id": "book_1",
        "target_ids": ["cabinet_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([cabinet, book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_conservative_top")
    assert result["label"] == "pass"
    assert "bbox top fallback" in result["reason"]


def test_rule_functional_dependency_unknown_profile_surface_supports_placeable() -> (
    None
):
    surface = _benchmark_obj(
        "surface",
        "dataset_specific_surface_931",
        (2.0, 2.0, 0.4),
        (1.0, 0.8, 0.8),
    )
    surface["object_function_profile"] = {"can_support_top": True}
    subject = _benchmark_obj(
        "object",
        "dataset_specific_placeable_274",
        (2.0, 2.0, 0.9),
        (0.16, 0.12, 0.2),
    )
    subject["object_function_profile"] = {"is_small_placeable": True}
    check = {
        "check_id": "fd_unknown_profile_support",
        "metric": "functional_dependency",
        "subject_id": "object",
        "target_ids": ["surface"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([surface, subject], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "pass"
    assert "unified support score" in result["reason"]


def test_rule_functional_dependency_weak_overlap_floating_object_not_blanket_passed() -> (
    None
):
    shelf = _benchmark_obj(
        "wall_shelf",
        "wall_shelf",
        (3.5, 0.0, 1.28),
        (0.90, 0.05, 0.235),
    )
    shelf["object_function_profile"] = {"can_support_top": True}
    shelf["support_regions"] = [
        {
            "region_id": "collapsed_top",
            "support_kind": "top_surface",
            "height_world_z": 1.28,
            "clearance_above_m": 1.0,
            "access_type": "top",
            "polygon_world_xy": [
                [3.05, 0.025],
                [3.95, 0.025],
                [3.95, 0.025],
                [3.05, 0.025],
            ],
        }
    ]
    plant = _benchmark_obj("plant", "plant", (3.39, 0.60, 1.43), (0.152, 0.203, 0.158))
    plant["object_function_profile"] = {"is_small_placeable": True}
    check = {
        "check_id": "fd_weak_overlap_floating",
        "metric": "functional_dependency",
        "subject_id": "plant",
        "target_ids": ["wall_shelf"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([shelf, plant], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] in {"fail", "degraded"}


def test_rule_functional_dependency_cabinet_base_region_passes_and_floating_fails() -> (
    None
):
    cabinet = _benchmark_obj("cabinet_1", "cabinet", (2.0, 2.0, 0.6), (0.8, 0.6, 1.2))
    cabinet["support_regions"] = [
        {
            "region_id": "cabinet_floor",
            "support_kind": "cabinet_base",
            "height_world_z": 0.18,
            "polygon_world_xy": [
                [1.6, 1.7],
                [2.4, 1.7],
                [2.4, 2.3],
                [1.6, 2.3],
            ],
            "clearance_above_m": 0.5,
            "access_type": "openable_storage",
        }
    ]
    cup_supported = _benchmark_obj(
        "cup_supported", "cup", (2.0, 2.0, 0.28), (0.14, 0.14, 0.2)
    )
    cup_floating = _benchmark_obj(
        "cup_floating", "cup", (2.0, 2.0, 0.8), (0.14, 0.14, 0.2)
    )
    checks = [
        {
            "check_id": "fd_cabinet_region",
            "metric": "functional_dependency",
            "subject_id": "cup_supported",
            "target_ids": ["cabinet_1"],
            "relation_type": "object_on_support",
        },
        {
            "check_id": "fd_floating_region",
            "metric": "functional_dependency",
            "subject_id": "cup_floating",
            "target_ids": ["cabinet_1"],
            "relation_type": "object_on_support",
        },
    ]

    results = _run_direct_case_pack(
        _benchmark_case_pack([cabinet, cup_supported, cup_floating], checks),
        metrics=["functional_dependency"],
    )

    supported = next(
        item for item in results if item["check_id"] == "fd_cabinet_region"
    )
    floating = next(
        item for item in results if item["check_id"] == "fd_floating_region"
    )
    assert supported["label"] == "pass"
    assert "matched cabinet_base" in supported["reason"]
    assert floating["label"] == "fail"
    assert "does not match any support region" in floating["reason"]


def test_rule_functional_dependency_template_support_skips_noise_targets() -> None:
    nightstand = _benchmark_obj(
        "nightstand_1", "nightstand", (2.0, 2.0, 0.4), (0.8, 0.6, 0.8)
    )
    fork = _benchmark_obj("fork_1", "fork", (2.0, 2.0, 0.85), (0.04, 0.02, 0.2))
    recessed = _benchmark_obj(
        "recessed_light_1", "light", (2.0, 2.0, 2.3), (0.2, 0.2, 0.2)
    )
    alarm_clock = _benchmark_obj(
        "alarm_clock_1", "alarm_clock", (2.0, 2.0, 0.85), (0.12, 0.08, 0.12)
    )

    results = _run_direct_case_pack(
        _benchmark_case_pack([nightstand, fork, recessed, alarm_clock]),
        metrics=["functional_dependency"],
        extra={"max_fd_relation_proposals": 16},
    )

    support_pairs = {
        (result["primary_object"], result["related_objects"][0])
        for result in results
        if result.get("relation_type") == "object_on_support"
        and result.get("related_objects")
    }
    assert ("alarm_clock_1", "nightstand_1") in support_pairs
    assert ("fork_1", "nightstand_1") not in support_pairs
    assert ("recessed_light_1", "nightstand_1") not in support_pairs


def test_rule_functional_dependency_templates_ignore_book_named_furniture() -> None:
    shelf = _benchmark_obj("shelf_1", "shelf", (4.8, 2.0, 1.2), (0.2, 1.2, 0.2))
    book = _benchmark_obj("book_1", "book", (4.72, 2.0, 1.35), (0.12, 0.05, 0.2))
    cabinet = _benchmark_obj(
        "bookcabinet_1", "cabinet", (4.7, 1.1, 1.6), (0.5, 0.5, 1.2)
    )
    nightstand = _benchmark_obj(
        "booknightstand_1", "nightstand", (4.7, 2.9, 0.4), (0.6, 0.5, 0.8)
    )

    results = _run_direct_case_pack(
        _benchmark_case_pack([shelf, book, cabinet, nightstand]),
        metrics=["functional_dependency"],
        extra={"max_fd_relation_proposals": 16},
    )

    pairs = {
        (result["primary_object"], result["related_objects"][0])
        for result in results
        if result.get("relation_type") == "object_on_support"
        and result.get("related_objects")
    }
    assert ("book_1", "shelf_1") in pairs
    assert ("bookcabinet_1", "shelf_1") not in pairs
    assert ("booknightstand_1", "shelf_1") not in pairs


def test_rule_functional_dependency_books_on_bookshelf_skip_nearby_desk() -> None:
    bookshelf = _benchmark_obj(
        "bookshelf_1", "bookshelf", (0.2, 0.7, 0.75), (0.3, 1.2, 1.5)
    )
    desk = _benchmark_obj("desk_1", "desk", (1.2, 3.4, 0.35), (0.8, 1.1, 0.7))
    top_book = _benchmark_obj("book_top", "book", (0.2, 0.7, 1.58), (0.2, 0.16, 0.16))
    floor_book = _benchmark_obj(
        "book_floor", "book", (1.1, 0.3, 0.05), (0.2, 0.16, 0.1)
    )

    results = _run_direct_case_pack(
        _benchmark_case_pack([bookshelf, desk, top_book, floor_book]),
        metrics=["functional_dependency"],
        extra={"max_fd_relation_proposals": 16},
    )

    pairs = {
        (result["primary_object"], result["related_objects"][0])
        for result in results
        if result.get("relation_type") == "object_on_support"
        and result.get("related_objects")
    }
    assert ("book_top", "bookshelf_1") in pairs
    assert ("book_top", "desk_1") not in pairs
    assert ("book_floor", "desk_1") not in pairs


def test_rule_functional_dependency_name_collision_skips_fake_work_surface() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    coffee_table = _benchmark_obj(
        "coffee_table_1", "coffee_table", (2.9, 2.0, 0.35), (1.0, 0.8, 0.7)
    )
    disguised_book = _benchmark_obj(
        "book_coffeetable_1", "book", (2.4, 2.0, 0.15), (0.2, 0.16, 0.1)
    )
    disguised_book["category"] = "book_coffeetable"
    check = {
        "check_id": "fd_name_collision",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["book_coffeetable_1", "coffee_table_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, coffee_table, disguised_book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_name_collision")
    assert result["label"] == "pass"
    assert result["selected_related_objects"] == ["coffee_table_1"]


def test_rule_functional_dependency_table_lamp_is_not_work_surface_target() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    lamp = _benchmark_obj("lamp_1", "tablelamp1", (2.45, 2.0, 0.55), (0.3, 0.3, 0.6))
    lamp["category_norm"] = "desk_lamp"
    lamp["functional_hints"].update(
        {
            "category_group": "lighting",
            "category_keywords": ["desk lamp", "table lamp", "reading lamp"],
            "classification_source": "asset_annotation",
            "benchmark_relevance": "functional",
        }
    )
    table = _benchmark_obj("table_1", "table", (2.95, 2.0, 0.4), (1.0, 0.8, 0.8))
    check = {
        "check_id": "fd_lamp_not_surface",
        "metric": "functional_dependency",
        "subject_id": "chair_1",
        "target_ids": ["lamp_1", "table_1"],
        "relation_type": "seating_to_work_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, lamp, table], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_lamp_not_surface")
    assert result["label"] == "pass"
    assert result["selected_related_objects"] == ["table_1"]


def test_rule_functional_dependency_lamp_to_surface_rejects_seating() -> None:
    lamp = _benchmark_obj("lamp_1", "tablelamp1", (2.0, 2.0, 0.7), (0.3, 0.3, 0.6))
    lamp["category_norm"] = "desk_lamp"
    lamp["functional_hints"].update(
        {
            "category_group": "lighting",
            "category_keywords": ["desk lamp", "table lamp"],
            "classification_source": "asset_annotation",
            "benchmark_relevance": "functional",
        }
    )
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.9, 0.9, 1.0))
    chair["functional_hints"].update(
        {
            "category_group": "seating",
            "interaction_surface_map": {"top": ["seat cushion top surface"]},
            "classification_source": "asset_annotation",
            "benchmark_relevance": "functional",
        }
    )
    nightstand = _benchmark_obj(
        "nightstand_1", "nightstand", (2.0, 2.0, 0.2), (0.8, 0.6, 0.4)
    )
    check = {
        "check_id": "fd_lamp_surface_rejects_seating",
        "metric": "functional_dependency",
        "subject_id": "lamp_1",
        "target_ids": ["chair_1", "nightstand_1"],
        "relation_type": "lamp_to_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([lamp, chair, nightstand], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item
        for item in results
        if item["check_id"] == "fd_lamp_surface_rejects_seating"
    )
    assert result["label"] == "pass"
    assert result["selected_related_objects"] == ["nightstand_1"]


def test_rule_functional_dependency_mounted_lamps_do_not_propose_surface_relation() -> (
    None
):
    table = _benchmark_obj("table_1", "table", (2.0, 2.0, 0.4), (1.0, 0.8, 0.8))
    ceiling_lamp = _benchmark_obj(
        "ceiling_lamp_1", "lamp", (2.0, 2.0, 2.4), (0.35, 0.35, 0.3)
    )
    wall_lamp = _benchmark_obj(
        "wall_lamp_1", "lamp", (2.0, 1.5, 1.4), (0.35, 0.35, 0.3)
    )
    ceiling_lamp["functional_hints"].update(
        {"category_group": "lighting", "scene_object_type": "ceiling_mounted"}
    )
    wall_lamp["functional_hints"].update(
        {"category_group": "lighting", "scene_object_type": "wall_mounted"}
    )

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, ceiling_lamp, wall_lamp]),
        metrics=["functional_dependency"],
        extra={"max_fd_relation_proposals": 16},
    )

    pairs = {
        (
            result["primary_object"],
            result.get("relation_type"),
            result["related_objects"][0],
        )
        for result in results
        if result.get("related_objects")
    }
    assert ("ceiling_lamp_1", "lamp_to_surface", "table_1") not in pairs
    assert ("wall_lamp_1", "lamp_to_surface", "table_1") not in pairs


def test_rule_functional_dependency_support_rejects_seating_and_cushions() -> None:
    cup = _benchmark_obj("cup_1", "cup", (2.0, 2.0, 0.48), (0.12, 0.12, 0.16))
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.9, 0.9, 1.0))
    chair["functional_hints"].update(
        {
            "category_group": "seating",
            "interaction_surface_map": {"top": ["seat cushion top surface"]},
            "classification_source": "asset_annotation",
            "benchmark_relevance": "functional",
        }
    )
    cushion = _benchmark_obj(
        "cushion_1", "cushion", (2.0, 2.0, 0.36), (0.45, 0.45, 0.12)
    )
    cushion["functional_hints"].update(
        {
            "functional_categories": ["supportable"],
            "classification_source": "asset_annotation",
            "benchmark_relevance": "functional",
        }
    )
    console = _benchmark_obj("console_1", "console", (2.0, 2.0, 0.2), (1.0, 0.4, 0.4))
    check = {
        "check_id": "fd_support_rejects_soft_targets",
        "metric": "functional_dependency",
        "subject_id": "cup_1",
        "target_ids": ["chair_1", "cushion_1", "console_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([cup, chair, cushion, console], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item
        for item in results
        if item["check_id"] == "fd_support_rejects_soft_targets"
    )
    assert result["label"] == "pass"
    assert result["selected_related_objects"] == ["console_1"]


def test_rule_functional_dependency_indirect_support_via_tray() -> None:
    table = _benchmark_obj("table_1", "coffee_table", (2.0, 2.0, 0.0), (1.2, 0.7, 0.6))
    tray = _benchmark_obj("tray_1", "tray", (2.0, 2.0, 0.52), (0.46, 0.18, 0.24))
    bowl = _benchmark_obj("bowl_1", "bowl", (1.92, 2.0, 0.64), (0.18, 0.14, 0.16))
    check = {
        "check_id": "fd_indirect_tray_support",
        "metric": "functional_dependency",
        "subject_id": "bowl_1",
        "target_ids": ["table_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, tray, bowl], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_indirect_tray_support"
    )
    assert result["label"] == "pass"
    assert "indirectly via `tray_1`" in result["reason"]


def test_rule_functional_dependency_indirect_support_rejects_unrelated_shelf_target() -> (
    None
):
    shelf = _benchmark_obj(
        "pantry_shelving_unit_1", "bookshelf", (3.30, 2.80, 0.0), (0.90, 1.90, 0.45)
    )
    shelf["functional_hints"].update(
        {
            "category_group": "storage",
            "functional_categories": ["supportable"],
            "category_keywords": ["pantry shelving unit", "shelf", "bookcase"],
        }
    )
    tray = _benchmark_obj(
        "serving_tray_1", "tray", (2.825, 1.905, 1.173), (0.46, 0.28, 0.08)
    )
    cup = _benchmark_obj(
        "cup_1", "water_tumbler_cup", (2.749, 1.944, 1.276), (0.10, 0.10, 0.20)
    )
    check = {
        "check_id": "fd_unrelated_shelf_tray",
        "metric": "functional_dependency",
        "subject_id": "cup_1",
        "target_ids": ["pantry_shelving_unit_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([shelf, tray, cup], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_unrelated_shelf_tray"
    )
    assert result["label"] == "fail", result["reason"]
    assert "indirectly via" not in result["reason"]


def test_rule_functional_dependency_floating_cup_in_shelf_footprint_skips_fallback() -> (
    None
):
    shelf = _benchmark_obj("shelf_1", "bookshelf", (2.0, 2.0, 0.0), (0.8, 1.4, 0.32))
    shelf["functional_hints"].update(
        {
            "category_group": "storage",
            "functional_categories": ["supportable"],
            "category_keywords": ["bookshelf", "bookcase", "shelving unit"],
        }
    )
    cup = _benchmark_obj(
        "cup_1", "water_tumbler_cup", (2.0, 2.0, 1.10), (0.10, 0.10, 0.20)
    )
    check = {
        "check_id": "fd_floating_cup_shelf",
        "metric": "functional_dependency",
        "subject_id": "cup_1",
        "target_ids": ["shelf_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([shelf, cup], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_floating_cup_shelf"
    )
    assert result["label"] == "fail", result["reason"]
    assert "multilevel shelf support" not in result["reason"]


def test_rule_functional_dependency_cup_on_coaster_uses_indirect_support() -> None:
    nightstand = _benchmark_obj(
        "nightstand_1", "nightstand", (2.0, 2.0, 0.0), (0.5, 0.5, 0.4)
    )
    nightstand["support_regions"] = [
        {
            "region_id": "small_top_patch",
            "support_kind": "top_surface",
            "height_world_z": 0.2,
            "polygon_world_xy": [
                [1.9, 1.9],
                [2.05, 1.9],
                [2.05, 2.05],
                [1.9, 2.05],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    coaster = _benchmark_obj(
        "coaster_1", "cork_coaster_0_s0", (2.0, 2.0, 0.50), (0.12, 0.12, 0.10)
    )
    cup = _benchmark_obj(
        "cup_1", "glass_tumbler_0_s0", (2.0, 2.0, 0.54), (0.10, 0.10, 0.08)
    )
    check = {
        "check_id": "fd_cup_on_coaster_nightstand",
        "metric": "functional_dependency",
        "subject_id": "cup_1",
        "target_ids": ["nightstand_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([nightstand, coaster, cup], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_cup_on_coaster_nightstand"
    )
    assert result["label"] == "pass"
    assert "indirectly via `coaster_1`" in result["reason"]


def test_rule_functional_dependency_small_upright_tumbler_on_table_top() -> None:
    table = _benchmark_obj(
        "table_1", "dining_table", (2.25, 1.185, 0.00716), (1.92, 0.63, 0.9)
    )
    table["support_regions"] = [
        {
            "region_id": "support_region_0",
            "support_kind": "top_surface",
            "height_world_z": 0.4524,
            "polygon_world_xy": [
                [3.1704, 1.4606],
                [1.3296, 1.4606],
                [1.3296, 0.8710],
                [3.1704, 0.8710],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    tumbler = _benchmark_obj(
        "tumbler_1",
        "tumbler",
        (2.6669, 1.3985, 0.6383),
        (0.1176, 0.0762, 0.0758),
        yaw=90.0,
    )
    tumbler["functional_hints"].update(
        {
            "scene_object_type": "manipuland",
            "category_keywords": ["tumbler chocolattice tooth"],
            "interaction_surface_map": {
                "top": ["open cavity for holding toothbrushes"]
            },
        }
    )
    check = {
        "check_id": "fd_tumbler_conservative_top_height",
        "metric": "functional_dependency",
        "subject_id": "tumbler_1",
        "target_ids": ["table_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, tumbler], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item
        for item in results
        if item["check_id"] == "fd_tumbler_conservative_top_height"
    )
    assert result["label"] == "pass"
    assert "small upright object" in result["reason"]


def test_rule_functional_dependency_stacked_books_use_lower_book() -> None:
    nightstand = _benchmark_obj(
        "nightstand_1", "nightstand", (2.0, 2.0, 0.0), (0.5, 0.5, 0.4)
    )
    nightstand["support_regions"] = [
        {
            "region_id": "small_top_patch",
            "support_kind": "top_surface",
            "height_world_z": 0.2,
            "polygon_world_xy": [
                [1.9, 1.9],
                [2.05, 1.9],
                [2.05, 2.05],
                [1.9, 2.05],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    lower_book = _benchmark_obj(
        "book_lower", "paperback_book", (2.0, 2.0, 0.45), (0.22, 0.16, 0.24)
    )
    upper_book = _benchmark_obj(
        "book_upper", "hardcover_book_1_s0", (2.0, 2.0, 0.57), (0.21, 0.15, 0.14)
    )
    check = {
        "check_id": "fd_stacked_book_nightstand",
        "metric": "functional_dependency",
        "subject_id": "book_upper",
        "target_ids": ["nightstand_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([nightstand, lower_book, upper_book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_stacked_book_nightstand"
    )
    assert result["label"] == "pass"
    assert "indirectly via `book_lower`" in result["reason"]


def test_rule_functional_dependency_manipuland_stack_chain_to_table() -> None:
    table = _benchmark_obj("table_1", "table", (2.0, 2.0, 0.35), (1.0, 1.0, 0.7))
    table["functional_hints"].update(
        {"functional_categories": ["supportable"], "scene_object_type": "furniture"}
    )
    lower_book = _benchmark_obj(
        "book_lower", "book", (2.0, 2.0, 0.85), (0.24, 0.18, 0.30)
    )
    lower_book["functional_hints"]["scene_object_type"] = "manipuland"
    upper_book = _benchmark_obj(
        "book_upper", "book", (2.0, 2.0, 1.05), (0.22, 0.16, 0.10)
    )
    upper_book["functional_hints"]["scene_object_type"] = "manipuland"
    check = {
        "check_id": "fd_manipuland_book_stack",
        "metric": "functional_dependency",
        "subject_id": "book_upper",
        "target_ids": ["table_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, lower_book, upper_book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_manipuland_book_stack"
    )
    assert result["label"] == "pass"
    assert "manipuland stack chain" in result["reason"]
    assert "book_upper" in result["reason"]
    assert "book_lower" in result["reason"]
    assert result["diagnostics"]["support_evaluation_path"] == "stack"
    assert result["diagnostics"]["support_bottom_object_id"] == "book_lower"
    assert result["diagnostics"]["support_chain_ids"] == [
        "book_upper",
        "book_lower",
        "table_1",
    ]


def test_rule_functional_dependency_cup_tray_stack_uses_bottom_table_support() -> None:
    table = _benchmark_obj(
        "table_1",
        "bar_height_table",
        (2.2609, 2.6746, -0.0003),
        (2.0232, 1.2181, 0.7039),
    )
    table["functional_hints"].update(
        {
            "functional_categories": ["supportable"],
            "category_group": "work_surface",
            "scene_object_type": "furniture",
            "interaction_height_m": {"min": 1.0, "max": 1.2},
        }
    )
    tray = _benchmark_obj(
        "tray_1", "tray", (2.8255, 1.9050, 1.1726), (0.3612, 0.0644, 0.1828)
    )
    tray["functional_hints"].update(
        {
            "scene_object_type": "manipuland",
            "category_keywords": ["tray"],
            "interaction_surface_map": {"top": ["supportable", "graspable"]},
        }
    )
    cup = _benchmark_obj(
        "cup_1", "water_tumbler_cup", (2.7494, 1.9441, 1.1761), (0.0809, 0.0897, 0.0855)
    )
    cup["functional_hints"].update(
        {"scene_object_type": "manipuland", "category_keywords": ["cup", "tumbler"]}
    )
    check = {
        "check_id": "fd_cup_tray_bar_table_stack",
        "metric": "functional_dependency",
        "subject_id": "cup_1",
        "target_ids": ["table_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, tray, cup], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_cup_tray_bar_table_stack"
    )
    assert result["label"] == "pass", result["reason"]
    assert "manipuland stack chain" in result["reason"]
    assert result["diagnostics"]["support_evaluation_path"] == "stack"
    assert result["diagnostics"]["support_bottom_evaluation_path"] == "stack_bottom"


def test_rule_functional_dependency_book_inside_low_coffee_table_lower_shelf() -> None:
    coffee_table = _benchmark_obj(
        "coffee_table_1", "coffee_table", (2.0, 2.0, 0.235), (0.67, 1.14, 0.47)
    )
    coffee_table["functional_hints"].update(
        {
            "functional_categories": ["supportable"],
            "category_group": "work_surface",
            "category_keywords": ["coffee table", "center table", "low table"],
            "interaction_surface_map": {"top": ["flat tabletop surface"]},
        }
    )
    book = _benchmark_obj("book_1", "book", (2.0, 2.0, 0.163), (0.36, 0.40, 0.056))
    check = {
        "check_id": "fd_book_low_coffee_table_shelf",
        "metric": "functional_dependency",
        "subject_id": "book_1",
        "target_ids": ["coffee_table_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([coffee_table, book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_book_low_coffee_table_shelf"
    )
    assert result["label"] == "pass"
    assert "lower-shelf support" in result["reason"]


def test_rule_functional_dependency_tray_on_bed_accepts_soft_surface() -> None:
    bed = _benchmark_obj("bed_1", "bed", (2.0, 2.0, 0.0), (1.0, 1.8, 1.8))
    bed["functional_hints"].update(
        {
            "functional_categories": ["sittable", "sleepable", "supportable"],
            "category_group": "sleeping",
            "interaction_surface_map": {"top": ["sleeping surface", "sitting surface"]},
        }
    )
    bed["support_regions"] = [
        {
            "region_id": "support_region_0",
            "support_kind": "top_surface",
            "height_world_z": 0.88,
            "polygon_world_xy": [
                [1.4, 1.3],
                [2.6, 1.3],
                [2.6, 2.7],
                [1.4, 2.7],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    tray = _benchmark_obj("tray_1", "tray", (2.0, 2.0, 0.53), (0.52, 0.28, 0.72))
    tray["functional_hints"]["functional_categories"] = [
        "graspable",
        "supportable",
    ]
    check = {
        "check_id": "fd_tray_on_bed_soft_surface",
        "metric": "functional_dependency",
        "subject_id": "tray_1",
        "target_ids": ["bed_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([bed, tray], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_tray_on_bed_soft_surface"
    )
    assert result["label"] == "pass"
    assert "soft-surface support" in result["reason"]


def test_rule_functional_dependency_lamp_declared_on_bed_rescues_to_stool() -> None:
    bed = _benchmark_obj("bed_1", "bed", (1.44, 3.0, 0.0), (0.67, 2.06, 2.14))
    bed["functional_hints"].update(
        {
            "functional_categories": ["sittable", "sleepable", "supportable"],
            "category_group": "sleeping",
            "interaction_surface_map": {"top": ["sleeping surface", "mattress"]},
        }
    )
    stool = _benchmark_obj("stool_1", "stool", (1.32, 4.575, 0.0), (0.44, 0.395, 0.395))
    stool["functional_hints"].update(
        {
            "functional_categories": ["sittable"],
            "category_keywords": ["stool"],
        }
    )
    lamp = _benchmark_obj(
        "lamp_1", "desk_lamp", (1.10, 4.405, 0.488), (0.136, 0.341, 0.233)
    )
    lamp["functional_hints"].update(
        {
            "functional_categories": ["toggleable"],
            "category_group": "lighting",
            "category_keywords": ["desk lamp", "table lamp"],
            "scene_object_type": "manipuland",
        }
    )
    check = {
        "check_id": "fd_lamp_bed_rescue",
        "metric": "functional_dependency",
        "subject_id": "lamp_1",
        "target_ids": ["bed_1"],
        "relation_type": "lamp_to_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([bed, stool, lamp], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "pass"
    assert result["related_objects"] == ["bed_1"]
    assert result["selected_related_objects"] == ["stool_1"]
    assert result["diagnostics"]["support_evaluation_path"] == "target_rescue"
    assert result["diagnostics"]["rescue_selected_target_id"] == "stool_1"


def test_rule_functional_dependency_manipuland_stack_to_wall_art_fails() -> None:
    wall_art = _benchmark_obj(
        "wall_art_1", "wall_art", (2.0, 2.0, 0.95), (0.9, 0.05, 0.4)
    )
    wall_art["functional_hints"]["scene_object_type"] = "wall_mounted"
    plate = _benchmark_obj("plate_1", "plate", (2.0, 2.0, 1.15), (0.32, 0.28, 0.30))
    plate["functional_hints"]["scene_object_type"] = "manipuland"
    bowl = _benchmark_obj("bowl_1", "bowl", (2.0, 2.0, 1.38), (0.18, 0.16, 0.16))
    bowl["functional_hints"]["scene_object_type"] = "manipuland"
    check = {
        "check_id": "fd_bowl_plate_wall_art",
        "metric": "functional_dependency",
        "subject_id": "bowl_1",
        "target_ids": ["wall_art_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([wall_art, plate, bowl], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "fail"
    assert "target category is not compatible" in result["reason"]


def test_rule_functional_dependency_thin_shelf_without_regions_allows_front_edge() -> (
    None
):
    shelf = _benchmark_obj("shelf_1", "shelf", (2.0, 2.0, 0.75), (1.2, 0.05, 0.30))
    shelf["functional_hints"]["category_group"] = "storage"
    shelf["functional_hints"]["functional_categories"] = ["supportable"]
    book = _benchmark_obj("book_1", "book", (2.0, 1.86, 0.96), (0.18, 0.14, 0.12))
    check = {
        "check_id": "fd_thin_shelf_front_edge",
        "metric": "functional_dependency",
        "subject_id": "book_1",
        "target_ids": ["shelf_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([shelf, book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "pass"
    assert "thin shelf support" in result["reason"]


def test_rule_functional_dependency_thin_wall_shelf_bbox_fallback() -> None:
    shelf = _benchmark_obj(
        "wall_shelf_1", "wall_shelf", (2.0, 2.0, 1.1), (0.03, 0.8, 0.18)
    )
    shelf["functional_hints"]["category_group"] = "storage"
    shelf["functional_hints"]["functional_categories"] = ["supportable"]
    shelf["support_regions"] = [
        {
            "region_id": "thin_top",
            "support_kind": "top_surface",
            "height_world_z": 1.1,
            "polygon_world_xy": [
                [1.985, 1.6],
                [1.985, 2.4],
                [1.985, 2.4],
                [1.985, 1.6],
            ],
            "clearance_above_m": 1.0,
            "access_type": "top",
        }
    ]
    book = _benchmark_obj("book_1", "book", (2.0, 2.0, 1.23), (0.3, 0.3, 0.08))
    check = {
        "check_id": "fd_thin_wall_shelf",
        "metric": "functional_dependency",
        "subject_id": "book_1",
        "target_ids": ["wall_shelf_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([shelf, book], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == "fd_thin_wall_shelf")
    assert result["label"] == "pass"
    assert "bbox top fallback" in result["reason"]


def test_rule_functional_dependency_desk_lamp_manipuland_matches_table_surface() -> (
    None
):
    table = _benchmark_obj("table_1", "table", (2.0, 2.0, 0.35), (1.0, 0.8, 0.7))
    table["functional_hints"].update(
        {
            "functional_categories": ["supportable"],
            "category_group": "work_surface",
            "scene_object_type": "furniture",
        }
    )
    lamp = _benchmark_obj("lamp_1", "desk_lamp", (2.0, 2.0, 0.9), (0.24, 0.24, 0.4))
    lamp["functional_hints"].update(
        {"category_group": "lighting", "scene_object_type": "manipuland"}
    )
    check = {
        "check_id": "fd_desk_lamp_manipuland",
        "metric": "functional_dependency",
        "subject_id": "lamp_1",
        "target_ids": ["table_1"],
        "relation_type": "lamp_to_surface",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([table, lamp], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "pass"


def test_rule_functional_dependency_reversed_support_check_evaluates_surface() -> None:
    cabinet = _benchmark_obj(
        "sink_cabinet",
        "cabinet",
        (2.0, 2.0, 0.4),
        (1.2, 0.8, 0.8),
    )
    cabinet["object_function_profile"] = {"can_support_top": True}
    dispenser = _benchmark_obj(
        "soap_dispenser",
        "soap_dispenser",
        (2.0, 2.0, 0.9),
        (0.14, 0.12, 0.2),
    )
    dispenser["object_function_profile"] = {"is_small_placeable": True}
    check = {
        "check_id": "fd_reversed_support",
        "metric": "functional_dependency",
        "subject_id": "sink_cabinet",
        "target_ids": ["soap_dispenser"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([cabinet, dispenser], [check]),
        metrics=["functional_dependency"],
    )

    result = next(item for item in results if item["check_id"] == check["check_id"])
    assert result["label"] == "pass"
    assert result["selected_related_objects"] == ["soap_dispenser"]
    assert "reversed support direction" in result["reason"]


def test_rule_functional_dependency_wall_mounted_small_object_rejects_table_support() -> (
    None
):
    clock = _benchmark_obj("clock_1", "clock", (2.0, 2.0, 1.8), (0.2, 0.08, 0.2))
    clock["functional_hints"]["scene_object_type"] = "wall_mounted"
    table = _benchmark_obj("table_1", "table", (2.0, 2.0, 0.4), (1.0, 0.8, 0.8))
    check = {
        "check_id": "fd_wall_clock_support",
        "metric": "functional_dependency",
        "subject_id": "clock_1",
        "target_ids": ["table_1"],
        "relation_type": "object_on_support",
    }

    results = _run_direct_case_pack(
        _benchmark_case_pack([clock, table], [check]),
        metrics=["functional_dependency"],
    )

    result = next(
        item for item in results if item["check_id"] == "fd_wall_clock_support"
    )
    assert result["label"] == "fail"
    assert "target category is not compatible" in result["reason"]


def test_rule_functional_dependency_default_template_mode_does_not_call_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    desk = _benchmark_obj("desk_1", "desk", (2.9, 2.0, 0.4), (1.0, 0.8, 0.8))

    def fail_if_vlm_called(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise AssertionError("default template mode should not call VLM proposer")

    monkeypatch.setattr(fd_proposer, "_propose_via_vlm", fail_if_vlm_called)

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, desk]),
        metrics=["functional_dependency"],
        extra={"max_fd_relation_proposals": 4},
    )

    assert any(
        result.get("relation_type") == "seating_to_work_surface"
        and result.get("primary_object") == "chair_1"
        and result.get("related_objects") == ["desk_1"]
        for result in results
    )


def test_rule_functional_dependency_max_relation_proposals_limits_results() -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    desk = _benchmark_obj("desk_1", "desk", (2.9, 2.0, 0.4), (1.0, 0.8, 0.8))
    sofa = _benchmark_obj("sofa_1", "sofa", (2.0, 3.5, 0.5), (1.4, 0.8, 1.0))
    television = _benchmark_obj("tv_1", "television", (2.0, 4.6, 1.2), (1.2, 0.1, 0.7))
    case_pack = _benchmark_case_pack([chair, desk, sofa, television])

    results = _run_direct_case_pack(
        case_pack,
        metrics=["functional_dependency"],
        extra={
            "fd_relation_proposer_mode": "template",
            "max_fd_relation_proposals": 1,
        },
    )

    assert len(results) == 1
    assert results[0]["metric"] == "functional_dependency"


def test_rule_functional_dependency_vlm_mode_uses_proposer_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    desk = _benchmark_obj("desk_1", "desk", (2.9, 2.0, 0.4), (1.0, 0.8, 0.8))
    calls: list[dict[str, Any]] = []

    def fake_vlm_proposer(
        *args: Any, **kwargs: Any
    ) -> list[FunctionalDependencyProposal]:
        calls.append({"args": args, "kwargs": kwargs})
        return [
            FunctionalDependencyProposal(
                subject_id="chair_1",
                target_ids=["desk_1"],
                relation_type="seating_to_work_surface",
                expected_use="sit at and use the work surface",
                priority=0.99,
                reason="fake VLM proposal",
            )
        ]

    monkeypatch.setattr(fd_proposer, "_propose_via_vlm", fake_vlm_proposer)

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, desk]),
        metrics=["functional_dependency"],
        extra={
            "fd_relation_proposer_mode": "vlm",
            "max_fd_relation_proposals": 1,
        },
    )

    assert calls
    assert calls[0]["kwargs"]["max_proposals"] == 1
    fd_result = next(
        result
        for result in results
        if result.get("relation_type") == "seating_to_work_surface"
    )
    assert fd_result["primary_object"] == "chair_1"
    assert fd_result["related_objects"] == ["desk_1"]
    assert fd_result["label"] == "pass"


def test_rule_functional_dependency_bad_vlm_surface_target_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chair = _benchmark_obj("chair_1", "chair", (2.0, 2.0, 0.5), (0.6, 0.6, 1.0))
    table = _benchmark_obj(
        "dining_table_1", "dining_table", (2.9, 2.0, 0.4), (1.2, 0.8, 0.8)
    )
    spoon = _benchmark_obj(
        "tablespoon_1", "table", (2.35, 2.0, 0.25), (0.04, 0.04, 0.2)
    )
    spoon["category"] = "room_tablespoon"

    def fake_vlm_proposer(
        *_args: Any, **_kwargs: Any
    ) -> list[FunctionalDependencyProposal]:
        return [
            FunctionalDependencyProposal(
                subject_id="chair_1",
                target_ids=["tablespoon_1"],
                relation_type="seating_to_work_surface",
                expected_use="sit at and use the work/table surface",
                priority=0.9,
                reason="bad VLM proposal",
            )
        ]

    monkeypatch.setattr(fd_proposer, "_propose_via_vlm", fake_vlm_proposer)

    results = _run_direct_case_pack(
        _benchmark_case_pack([chair, table, spoon]),
        metrics=["functional_dependency"],
        extra={"fd_relation_proposer_mode": "vlm", "max_fd_relation_proposals": 4},
    )

    pairs = {
        (result["primary_object"], result["related_objects"][0])
        for result in results
        if result.get("relation_type") == "seating_to_work_surface"
        and result.get("related_objects")
    }
    assert ("chair_1", "dining_table_1") in pairs
    assert ("chair_1", "tablespoon_1") not in pairs


def test_rule_functional_dependency_bad_vlm_media_target_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    armchair = _benchmark_obj(
        "armchair_1", "armchair", (2.0, 2.0, 0.5), (0.8, 0.8, 1.0)
    )
    tv_stand = _benchmark_obj(
        "tv_stand_1", "tv_stand", (4.0, 2.0, 0.5), (1.2, 0.5, 1.0), yaw=180.0
    )
    remote = _benchmark_obj(
        "tv_remote_control_1", "television", (2.6, 2.0, 0.25), (0.05, 0.08, 0.2)
    )
    remote["category"] = "room_tv_remote_control"

    def fake_vlm_proposer(
        *_args: Any, **_kwargs: Any
    ) -> list[FunctionalDependencyProposal]:
        return [
            FunctionalDependencyProposal(
                subject_id="armchair_1",
                target_ids=["tv_remote_control_1"],
                relation_type="seating_to_media",
                expected_use="sit and view the media object",
                priority=0.85,
                reason="bad VLM proposal",
            )
        ]

    monkeypatch.setattr(fd_proposer, "_propose_via_vlm", fake_vlm_proposer)

    results = _run_direct_case_pack(
        _benchmark_case_pack([armchair, tv_stand, remote]),
        metrics=["functional_dependency"],
        extra={"fd_relation_proposer_mode": "vlm", "max_fd_relation_proposals": 4},
    )

    pairs = [
        (result["primary_object"], result["related_objects"][0])
        for result in results
        if result.get("relation_type") == "seating_to_media"
        and result.get("related_objects")
    ]
    assert pairs == [("armchair_1", "tv_stand_1")]


def test_rule_functional_dependency_support_templates_prefer_credenza() -> None:
    dining_table = _benchmark_obj(
        "dining_table_1", "dining_table", (2.5, 2.1, 0.0), (2.0, 0.8, 1.0)
    )
    credenza = _benchmark_obj(
        "credenza_1", "credenza", (2.5, 0.66, 0.0), (1.8, 0.74, 0.45)
    )
    credenza["category_norm"] = "credenza_rowan_modular"
    bowl = _benchmark_obj("bowl_1", "bowl", (2.5, 0.42, 0.445), (0.23, 0.07, 0.23))
    book = _benchmark_obj("book_1", "book", (3.0, 0.42, 0.15), (0.20, 0.04, 0.21))

    results = _run_direct_case_pack(
        _benchmark_case_pack([dining_table, credenza, bowl, book]),
        metrics=["functional_dependency"],
        extra={"max_fd_relation_proposals": 16},
    )

    pairs = {
        (result["primary_object"], result["related_objects"][0])
        for result in results
        if result.get("relation_type") == "object_on_support"
        and result.get("related_objects")
    }
    assert ("bowl_1", "credenza_1") in pairs
    assert ("book_1", "credenza_1") in pairs
    assert ("bowl_1", "dining_table_1") not in pairs
    assert ("book_1", "dining_table_1") not in pairs


def test_rule_functional_dependency_fd_proposer_limits_support_subject_duplicates() -> (
    None
):
    table = _benchmark_obj("table_1", "dining_table", (2.5, 2.5, 0.4), (1.2, 0.8, 0.8))
    chair = _benchmark_obj("chair_1", "chair", (1.7, 2.5, 0.5), (0.6, 0.6, 1.0))
    candle = _benchmark_obj("candle_1", "candle", (2.1, 2.5, 0.82), (0.08, 0.08, 0.14))
    books = [
        _benchmark_obj(
            f"book_{idx}",
            "book",
            (2.2 + idx * 0.05, 2.5, 0.85),
            (0.12, 0.08, 0.12),
        )
        for idx in range(5)
    ]
    case_pack = _benchmark_case_pack([table, chair, candle, *books])
    store = fd_proposer.load_geometry(case_pack)
    assert store is not None

    payload = fd_proposer._build_fd_proposer_payload(case_pack, store, max_proposals=8)

    subject_ids = [subject["id"] for subject in payload["subjects"]]
    book_subjects = [
        subject_id for subject_id in subject_ids if subject_id.startswith("book_")
    ]
    assert len(book_subjects) == 2
    assert "candle_1" not in subject_ids


def test_rule_functional_dependency_template_proposer_uses_function_profiles() -> None:
    surface = _benchmark_obj(
        "surface",
        "dataset_specific_surface_931",
        (2.0, 2.0, 0.4),
        (1.0, 0.8, 0.8),
    )
    surface["object_function_profile"] = {"can_support_top": True}
    placeable = _benchmark_obj(
        "object",
        "dataset_specific_placeable_274",
        (2.0, 2.0, 0.9),
        (0.16, 0.12, 0.2),
    )
    placeable["object_function_profile"] = {"is_small_placeable": True}
    seat = _benchmark_obj(
        "seat",
        "dataset_specific_seat",
        (0.0, 0.0, 0.45),
        (0.6, 0.6, 0.9),
    )
    seat["object_function_profile"] = {"is_seating": True}
    desk = _benchmark_obj(
        "desk_like",
        "dataset_specific_surface",
        (0.8, 0.0, 0.4),
        (0.8, 0.6, 0.8),
    )
    desk["object_function_profile"] = {
        "is_work_surface": True,
        "can_support_top": True,
    }
    case_pack = _benchmark_case_pack([surface, placeable, seat, desk])
    store = fd_proposer.load_geometry(case_pack)
    assert store is not None
    config = type(
        "Config",
        (),
        {
            "run": type(
                "Run",
                (),
                {
                    "max_fd_relation_proposals": 8,
                    "fd_relation_proposer_mode": "template",
                },
            )()
        },
    )()

    proposals = fd_proposer.propose_dependency_relations(case_pack, store, config)
    proposal_pairs = {
        (proposal.subject_id, tuple(proposal.target_ids), proposal.relation_type)
        for proposal in proposals
    }

    assert ("object", ("surface",), "object_on_support") in proposal_pairs
    assert ("seat", ("desk_like",), "seating_to_work_surface") in proposal_pairs


def test_house_case_pack_filter_matches_combined_furniture_stage(
    tmp_path: Path,
) -> None:
    case_pack = house_scene_to_case_pack(
        _house(tmp_path),
        stage="combined_house_after_furniture",
        metrics=["spatial_accessibility", "functional_dependency"],
        include_object_types=[ObjectType.FURNITURE],
    )

    object_ids = {obj["id"] for obj in case_pack["scene_geometry"]["objects"]}
    assert case_pack["task_instruction"] == "A house with one bedroom."
    assert "table_0" in object_ids
    assert "mug_0" not in object_ids
    assert all(
        check["subject_id"] != "mug_0"
        for check in case_pack["checks"]
        if check["metric"] == "functional_dependency"
    )


def test_write_room_stage_report_uses_stage_directory(tmp_path: Path) -> None:
    stage_dir = tmp_path / "scene_states" / "final_scene"
    payload = write_room_stage_report(
        _scene(tmp_path),
        stage_dir,
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "room_stage_hooks": ["final_scene"],
            }
        },
        stage="final_scene",
    )

    assert payload is not None
    report_path = stage_dir / "scenebenchmark_critic.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["stage"] == "final_scene"
    markdown = (stage_dir / "scenebenchmark_critic.md").read_text(encoding="utf-8")
    assert "All checks / ignored" in markdown


def test_write_room_stage_report_skips_when_disabled_or_stage_not_hooked(
    tmp_path: Path,
) -> None:
    disabled_dir = tmp_path / "disabled" / "scene_states" / "final_scene"
    skipped_stage_dir = tmp_path / "skipped" / "scene_states" / "wall_objects"

    disabled_payload = write_room_stage_report(
        _scene(tmp_path / "disabled_scene"),
        disabled_dir,
        config={
            "scenebenchmark_critic": {
                "enabled": False,
                "room_stage_hooks": ["final_scene"],
            }
        },
        stage="final_scene",
    )
    skipped_stage_payload = write_room_stage_report(
        _scene(tmp_path / "skipped_scene"),
        skipped_stage_dir,
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "room_stage_hooks": ["final_scene"],
            }
        },
        stage="wall_objects",
    )

    assert disabled_payload is None
    assert skipped_stage_payload is None
    assert not (disabled_dir / "scenebenchmark_critic.json").exists()
    assert not (skipped_stage_dir / "scenebenchmark_critic.json").exists()


def test_write_house_stage_report_is_opt_in_by_default(tmp_path: Path) -> None:
    house_dir = tmp_path / "scene_000" / "combined_house_after_furniture"

    payload = write_house_stage_report(
        _house(tmp_path),
        house_dir,
        config={
            "scenebenchmark_critic": {
                "enabled": True,
            }
        },
        stage="combined_house_after_furniture",
    )

    assert payload is None
    assert not (house_dir / "scenebenchmark_critic.json").exists()


def test_room_stage_reports_do_not_collide_across_rooms(tmp_path: Path) -> None:
    room_a = _scene(tmp_path / "scene_000" / "room_a")
    room_b = _scene(tmp_path / "scene_000" / "room_b")
    room_a.room_id = "room_a"
    room_b.room_id = "room_b"
    stage_a = tmp_path / "scene_000" / "room_a" / "scene_states" / "final_scene"
    stage_b = tmp_path / "scene_000" / "room_b" / "scene_states" / "final_scene"
    config = {
        "scenebenchmark_critic": {
            "enabled": True,
            "room_stage_hooks": ["final_scene"],
        }
    }

    payload_a = write_room_stage_report(
        room_a, stage_a, config=config, stage="final_scene"
    )
    payload_b = write_room_stage_report(
        room_b, stage_b, config=config, stage="final_scene"
    )

    assert payload_a is not None
    assert payload_b is not None
    assert (stage_a / "scenebenchmark_critic.json").exists()
    assert (stage_b / "scenebenchmark_critic.json").exists()
    saved_a = json.loads(
        (stage_a / "scenebenchmark_critic.json").read_text(encoding="utf-8")
    )
    saved_b = json.loads(
        (stage_b / "scenebenchmark_critic.json").read_text(encoding="utf-8")
    )
    assert saved_a["scope"] == "room:room_a"
    assert saved_b["scope"] == "room:room_b"


def test_monitor_clearance_ignores_same_surface_desktop_peripherals() -> None:
    monitor = _benchmark_obj(
        "computer_monitor_1", "monitor", (0.0, 0.0, 0.9), (0.4, 0.08, 0.3)
    )
    monitor["metadata"] = {
        "clearance": {
            "clearance_type": "接近",
            "direction": "前",
            "depth_m": 0.45,
            "width_m": 0.52,
            "height_m": 1.8,
            "confidence": "high",
            "inherits_from_support": False,
        }
    }
    monitor["object_type"] = "manipuland"
    monitor["functional_hints"]["scene_object_type"] = "manipuland"
    monitor["placement_info"] = {"parent_surface_id": "desk_top"}
    keyboard = _benchmark_obj(
        "keyboard_1", "keyboard", (0.0, 0.18, 0.75), (0.32, 0.12, 0.04)
    )
    keyboard["object_type"] = "manipuland"
    keyboard["functional_hints"]["scene_object_type"] = "manipuland"
    keyboard["placement_info"] = {"parent_surface_id": "desk_top"}
    book = _benchmark_obj("book_1", "book", (0.18, 0.18, 0.75), (0.16, 0.12, 0.04))
    book["object_type"] = "manipuland"
    book["functional_hints"]["scene_object_type"] = "manipuland"
    book["placement_info"] = {"parent_surface_id": "desk_top"}

    checks = build_checks(
        _benchmark_case_pack([monitor, keyboard, book]),
        metrics=["interaction_clearance"],
    )
    result = _run_direct_case_pack(
        _benchmark_case_pack([monitor, keyboard, book], checks),
        metrics=["interaction_clearance"],
    )[0]

    assert checks[0]["target_ids"] == ["book_1"]
    assert result["label"] == "fail"
    assert result["blocking_objects"] == ["book_1"]
    assert "keyboard_1" not in result["reason"]


def test_prompt_context_is_concise(tmp_path: Path) -> None:
    payload = evaluate_room_scene(
        _scene(tmp_path),
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["functional_dependency"],
            }
        },
        stage="final_scene",
    )

    context = format_prompt_context(payload, max_issues=2)

    assert "SceneBenchmark geometry critic" in context


def test_prompt_context_respects_max_issues() -> None:
    payload = {
        "results": [
            {
                "label": "fail",
                "metric": "spatial_accessibility",
                "primary_object": f"chair_{idx}",
                "related_objects": [],
                "reason": f"issue {idx}",
            }
            for idx in range(3)
        ]
    }

    context = format_prompt_context(payload, max_issues=2)

    assert "chair_0" in context
    assert "chair_1" in context
    assert "chair_2" not in context


def test_prompt_context_excludes_ignored_issues() -> None:
    payload = {
        "results": [
            {
                "check_id": "ignored_fail",
                "label": "fail",
                "metric": "functional_dependency",
                "primary_object": "decor_0",
                "related_objects": [],
                "reason": "ignored decorative support issue",
                "scoring_tier": "ignored",
            },
            {
                "check_id": "core_degraded",
                "label": "degraded",
                "metric": "spatial_accessibility",
                "primary_object": "chair_0",
                "related_objects": [],
                "reason": "real access issue",
            },
        ]
    }

    context = format_prompt_context(payload)

    assert "chair_0" in context
    assert "decor_0" not in context
    assert "ignored decorative support issue" not in context


def _workstation_payload() -> dict[str, Any]:
    desk = _benchmark_obj("study_desk_0", "desk", (0.0, 0.0, 0.35), (1.4, 0.7, 0.7))
    desk["object_type"] = "furniture"
    desk["functional_hints"]["scene_object_type"] = "furniture"
    desk["functional_hints"]["category_group"] = "work_surface"
    desk["support_regions"] = [{"region_id": "desk_top", "support_kind": "top_surface"}]
    chair = _benchmark_obj(
        "office_chair_0", "office_chair", (0.0, -0.8, 0.45), (0.6, 0.6, 0.9)
    )
    chair["object_type"] = "furniture"
    chair["functional_hints"]["scene_object_type"] = "furniture"
    chair["functional_hints"]["category_group"] = "seating"
    chair["functional_hints"]["functional_categories"] = ["sittable"]
    monitor = _benchmark_obj(
        "computer_monitor_0", "monitor", (0.0, 0.12, 0.82), (0.45, 0.08, 0.32)
    )
    monitor["object_type"] = "manipuland"
    monitor["functional_hints"]["scene_object_type"] = "manipuland"
    monitor["functional_hints"]["category_group"] = "media"
    monitor["placement_info"] = {"parent_surface_id": "desk_top"}
    mouse = _benchmark_obj(
        "wireless_mouse_0", "mouse", (0.35, -0.08, 0.73), (0.08, 0.13, 0.04)
    )
    mouse["object_type"] = "manipuland"
    mouse["functional_hints"]["scene_object_type"] = "manipuland"
    mouse["placement_info"] = {"parent_surface_id": "desk_top"}
    book = _benchmark_obj(
        "hardcover_book_0", "book", (-0.3, -0.08, 0.73), (0.2, 0.3, 0.04)
    )
    book["object_type"] = "manipuland"
    book["functional_hints"]["scene_object_type"] = "manipuland"
    book["placement_info"] = {"parent_surface_id": "desk_top"}
    shelf = _benchmark_obj(
        "shelving_unit_0", "bookshelf", (2.0, 0.0, 0.9), (0.8, 0.35, 1.8)
    )
    shelf["object_type"] = "furniture"
    shelf["functional_hints"]["scene_object_type"] = "furniture"
    sideboard = _benchmark_obj(
        "sideboard_0", "sideboard", (2.0, 1.0, 0.45), (1.2, 0.45, 0.9)
    )
    sideboard["object_type"] = "furniture"
    sideboard["functional_hints"]["scene_object_type"] = "furniture"
    sideboard["functional_hints"]["category_group"] = "storage_surface"
    sideboard["functional_hints"]["functional_categories"] = [
        "openable",
        "sittable",
        "storage",
        "supportable",
    ]
    return {
        "case_pack": _benchmark_case_pack(
            [desk, chair, monitor, mouse, book, shelf, sideboard]
        ),
        "results": [
            {
                "check_id": "fd_monitor_on_desk",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "computer_monitor_0",
                "related_objects": ["study_desk_0"],
                "relation_type": "object_on_support",
                "reason": "monitor is not on desk",
            },
            {
                "check_id": "fd_mouse_faces_monitor",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "wireless_mouse_0",
                "related_objects": ["computer_monitor_0"],
                "relation_type": "computer_peripheral_faces_screen",
                "reason": "mouse does not face monitor",
            },
            {
                "check_id": "fd_chair_faces_monitor",
                "metric": "functional_dependency",
                "label": "degraded",
                "primary_object": "office_chair_0",
                "related_objects": ["computer_monitor_0"],
                "relation_type": "seating_to_media",
                "reason": "chair is weakly aligned to monitor",
            },
            {
                "check_id": "fd_book_monitor_noise",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "hardcover_book_0",
                "related_objects": ["computer_monitor_0"],
                "relation_type": "seating_to_media",
                "reason": "book is not a seating subject",
            },
            {
                "check_id": "fd_monitor_self_noise",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "computer_monitor_0",
                "related_objects": ["computer_monitor_0"],
                "relation_type": "seating_to_media",
                "reason": "self relation",
            },
            {
                "check_id": "fd_shelf_monitor_noise",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "shelving_unit_0",
                "related_objects": ["computer_monitor_0"],
                "relation_type": "seating_to_media",
                "reason": "remote furniture is not actionable now",
            },
        ],
    }


def test_agent_prompt_context_filters_manipuland_workstation_noise() -> None:
    payload = _workstation_payload()

    filtered = filter_prompt_results_for_agent(
        payload,
        agent_type=AgentType.MANIPULAND,
        current_furniture_id="study_desk_0",
    )
    context = format_agent_prompt_context(
        payload,
        agent_type=AgentType.MANIPULAND,
        current_furniture_id="study_desk_0",
    )
    check_ids = {result["check_id"] for result in filtered}

    assert {
        "fd_monitor_on_desk",
        "fd_mouse_faces_monitor",
        "fd_chair_faces_monitor",
    } <= check_ids
    assert "fd_book_monitor_noise" not in check_ids
    assert "fd_monitor_self_noise" not in check_ids
    assert "fd_shelf_monitor_noise" not in check_ids
    assert "wireless_mouse_0" in context
    assert "hardcover_book_0" not in context
    assert "shelving_unit_0" not in context


def test_agent_prompt_context_keeps_furniture_layout_issues() -> None:
    payload = _workstation_payload()
    payload["results"].extend(
        [
            {
                "check_id": "fd_desk_chair_faces",
                "metric": "functional_dependency",
                "label": "degraded",
                "primary_object": "study_desk_0",
                "related_objects": ["office_chair_0"],
                "relation_type": "furniture_faces_furniture",
                "reason": "desk and chair alignment is weak",
            },
            {
                "check_id": "spatial_desk",
                "metric": "spatial_accessibility",
                "label": "fail",
                "primary_object": "study_desk_0",
                "related_objects": [],
                "reason": "desk is blocked",
            },
        ]
    )

    filtered = filter_prompt_results_for_agent(
        payload,
        agent_type=AgentType.FURNITURE,
    )
    check_ids = {result["check_id"] for result in filtered}

    assert "fd_desk_chair_faces" in check_ids
    assert "spatial_desk" in check_ids
    assert "fd_monitor_on_desk" not in check_ids
    assert "fd_mouse_faces_monitor" not in check_ids


def test_agent_prompt_context_filters_invalid_seating_relation_targets() -> None:
    payload = _workstation_payload()
    payload["results"] = [
        {
            "check_id": "fd_chair_desk",
            "metric": "functional_dependency",
            "label": "fail",
            "primary_object": "office_chair_0",
            "related_objects": ["study_desk_0"],
            "relation_type": "seating_to_work_surface",
            "reason": "chair should face the desk",
        },
        {
            "check_id": "fd_shelf_desk_noise",
            "metric": "functional_dependency",
            "label": "fail",
            "primary_object": "shelving_unit_0",
            "related_objects": ["study_desk_0"],
            "relation_type": "seating_to_work_surface",
            "reason": "storage furniture is not a seating subject",
        },
        {
            "check_id": "fd_chair_shelf_noise",
            "metric": "functional_dependency",
            "label": "fail",
            "primary_object": "office_chair_0",
            "related_objects": ["sideboard_0"],
            "relation_type": "seating_to_work_surface",
            "reason": "storage furniture is not a work surface target",
        },
        {
            "check_id": "fd_chair_shelf_media_noise",
            "metric": "functional_dependency",
            "label": "fail",
            "primary_object": "office_chair_0",
            "related_objects": ["sideboard_0"],
            "relation_type": "seating_to_media",
            "reason": "sideboard is not media",
        },
    ]

    filtered = filter_prompt_results_for_agent(
        payload,
        agent_type=AgentType.FURNITURE,
    )
    check_ids = {result["check_id"] for result in filtered}

    assert "fd_chair_desk" in check_ids
    assert "fd_shelf_desk_noise" not in check_ids
    assert "fd_chair_shelf_noise" not in check_ids
    assert "fd_chair_shelf_media_noise" not in check_ids


def test_agent_prompt_context_keeps_furniture_media_targets() -> None:
    sofa = _benchmark_obj("sofa_0", "sofa", (0.0, 0.0, 0.45), (1.8, 0.8, 0.9))
    sofa["object_type"] = "furniture"
    sofa["functional_hints"]["scene_object_type"] = "furniture"
    sofa["functional_hints"]["category_group"] = "seating"
    media = _benchmark_obj(
        "entertainment_center_0",
        "entertainment_center_entertainment",
        (0.0, 2.0, 0.45),
        (1.8, 0.45, 0.9),
    )
    media["object_type"] = "furniture"
    media["functional_hints"]["scene_object_type"] = "furniture"
    media["functional_hints"]["category_group"] = "storage_surface"
    sideboard = _benchmark_obj(
        "sideboard_0", "sideboard", (2.0, 2.0, 0.45), (1.2, 0.45, 0.9)
    )
    sideboard["object_type"] = "furniture"
    sideboard["functional_hints"]["scene_object_type"] = "furniture"
    sideboard["functional_hints"]["category_group"] = "storage_surface"
    payload = {
        "case_pack": _benchmark_case_pack([sofa, media, sideboard]),
        "results": [
            {
                "check_id": "fd_sofa_media",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "sofa_0",
                "related_objects": ["entertainment_center_0"],
                "relation_type": "seating_to_media",
                "reason": "sofa should face the media furniture",
            },
            {
                "check_id": "fd_sofa_sideboard_noise",
                "metric": "functional_dependency",
                "label": "fail",
                "primary_object": "sofa_0",
                "related_objects": ["sideboard_0"],
                "relation_type": "seating_to_media",
                "reason": "sideboard is not media",
            },
        ],
    }

    filtered = filter_prompt_results_for_agent(
        payload,
        agent_type=AgentType.FURNITURE,
    )
    check_ids = {result["check_id"] for result in filtered}

    assert "fd_sofa_media" in check_ids
    assert "fd_sofa_sideboard_noise" not in check_ids


def test_markdown_report_excludes_ignored_issues() -> None:
    payload = {
        "scope": "room:main",
        "stage": "final_scene",
        "gate": {"label": "report_only"},
        "summary": aggregate_results(
            [
                {
                    "check_id": "ignored_fail",
                    "label": "fail",
                    "metric": "functional_dependency",
                    "scoring_tier": "ignored",
                }
            ]
        ),
        "results": [
            {
                "check_id": "ignored_fail",
                "label": "fail",
                "metric": "functional_dependency",
                "primary_object": "decor_0",
                "reason": "ignored decorative support issue",
                "scoring_tier": "ignored",
            }
        ],
    }

    markdown = format_markdown_report(payload)

    assert "All checks / ignored: 1/1" in markdown
    assert "No degraded or failed checks." in markdown
    assert "ignored decorative support issue" not in markdown


class _DummyAgent(BaseStatefulAgent):
    @property
    def agent_type(self) -> AgentType:
        return AgentType.FURNITURE

    def _get_design_change_prompt_enum(self) -> Any:
        raise NotImplementedError

    def _get_initial_design_prompt_enum(self) -> Any:
        raise NotImplementedError

    def _get_initial_design_prompt_kwargs(self) -> dict:
        raise NotImplementedError

    def _get_critique_prompt_enum(self) -> Any:
        return "critique_prompt"

    def _get_final_scores_directory(self) -> Path:
        raise NotImplementedError

    def _set_placement_noise_profile(self, mode: Any) -> None:
        raise NotImplementedError


class _DummyCeilingAgent(_DummyAgent):
    @property
    def agent_type(self) -> AgentType:
        return AgentType.CEILING_MOUNTED


def test_agent_context_helper_is_disabled_by_default(tmp_path: Path) -> None:
    agent = object.__new__(_DummyAgent)
    agent.cfg = {"scenebenchmark_critic": {"enabled": False}}
    agent.scene = _scene(tmp_path)

    assert agent._build_scenebenchmark_critic_context() is None


def test_agent_context_helper_returns_context_when_enabled(tmp_path: Path) -> None:
    agent = object.__new__(_DummyAgent)
    agent.cfg = {
        "scenebenchmark_critic": {
            "enabled": True,
            "inject_into_llm_critic": True,
            "metrics": ["functional_dependency"],
        }
    }
    agent.scene = _scene(tmp_path)

    context = agent._build_scenebenchmark_critic_context()

    assert context is not None
    assert "SceneBenchmark geometry critic" in context


def test_agent_context_helper_does_not_inject_for_ceiling_agent(
    tmp_path: Path,
) -> None:
    agent = object.__new__(_DummyCeilingAgent)
    agent.cfg = {
        "scenebenchmark_critic": {
            "enabled": True,
            "inject_into_llm_critic": True,
            "metrics": ["functional_dependency"],
        }
    }
    agent.scene = _scene(tmp_path)

    assert agent._build_scenebenchmark_critic_context() is None


def test_create_run_config_adds_session_input_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_openai_run_config(_cfg: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "run-config"

    monkeypatch.setattr(
        base_stateful_agent,
        "create_openai_run_config",
        fake_create_openai_run_config,
    )

    agent = object.__new__(_DummyAgent)
    agent.cfg = OmegaConf.create(
        {
            "session_memory": {
                "intra_turn_observation_stripping": {
                    "enabled": False,
                }
            }
        }
    )

    result = agent._create_run_config()

    assert result == "run-config"
    callback = captured["session_input_callback"]
    assert callable(callback)
    assert callback([{"role": "assistant"}], [{"role": "user"}]) == [
        {"role": "assistant"},
        {"role": "user"},
    ]


@pytest.mark.asyncio
async def test_request_critique_disabled_does_not_change_physics_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakePromptRegistry:
        def get_prompt(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "critic instruction"

    class FakeResult:
        def final_output_as(self, _type: Any) -> FurnitureCritiqueWithScores:
            score = CategoryScore(name="x", grade=8, comment="ok")
            return FurnitureCritiqueWithScores(
                critique="ok",
                realism=score,
                functionality=score,
                layout=score,
                holistic_completeness=score,
                prompt_following=score,
                reachability=score,
            )

    async def fake_run(**_kwargs: Any) -> FakeResult:
        return FakeResult()

    monkeypatch.setattr(
        base_stateful_agent,
        "check_physics_violations",
        lambda **_kwargs: "physics-only",
    )
    monkeypatch.setattr(base_stateful_agent.Runner, "run", fake_run)
    monkeypatch.setattr(base_stateful_agent, "log_agent_usage", lambda **_kwargs: None)
    monkeypatch.setattr(
        base_stateful_agent, "log_agent_response", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        base_stateful_agent, "log_critique_scores", lambda *_args, **_kwargs: None
    )

    agent = object.__new__(_DummyAgent)
    agent.cfg = OmegaConf.create(
        {
            "scenebenchmark_critic": {
                "enabled": False,
                "inject_into_llm_critic": True,
            },
            "agents": {"critic_agent": {"max_turns": 1}},
        }
    )
    agent.scene = _scene(tmp_path)
    agent.prompt_registry = FakePromptRegistry()
    agent.placement_style = "natural"
    agent.critic = object()
    agent.critic_session = object()
    agent.rendering_manager = type("RenderingManager", (), {"last_render_dir": None})()
    agent.previous_scores = None
    agent.final_render_dir = None
    agent._create_run_config = lambda: None

    result = await agent._request_critique_impl(update_checkpoint=False)

    assert result == "ok"
    assert captured["physics_context"] == "physics-only"


@pytest.mark.asyncio
async def test_request_critique_injects_scenebenchmark_context_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakePromptRegistry:
        def get_prompt(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "critic instruction"

    class FakeResult:
        def final_output_as(self, _type: Any) -> FurnitureCritiqueWithScores:
            score = CategoryScore(name="x", grade=8, comment="ok")
            return FurnitureCritiqueWithScores(
                critique="ok",
                realism=score,
                functionality=score,
                layout=score,
                holistic_completeness=score,
                prompt_following=score,
                reachability=score,
            )

    async def fake_run(**_kwargs: Any) -> FakeResult:
        return FakeResult()

    monkeypatch.setattr(
        base_stateful_agent,
        "check_physics_violations",
        lambda **_kwargs: "physics-only",
    )
    monkeypatch.setattr(base_stateful_agent.Runner, "run", fake_run)
    monkeypatch.setattr(base_stateful_agent, "log_agent_usage", lambda **_kwargs: None)
    monkeypatch.setattr(
        base_stateful_agent, "log_agent_response", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        base_stateful_agent, "log_critique_scores", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        _DummyAgent,
        "_build_scenebenchmark_critic_context",
        lambda _self: "benchmark-context",
    )

    agent = object.__new__(_DummyAgent)
    agent.cfg = OmegaConf.create(
        {
            "scenebenchmark_critic": {
                "enabled": True,
                "inject_into_llm_critic": True,
            },
            "agents": {"critic_agent": {"max_turns": 1}},
        }
    )
    agent.scene = _scene(tmp_path)
    agent.prompt_registry = FakePromptRegistry()
    agent.placement_style = "natural"
    agent.critic = object()
    agent.critic_session = object()
    agent.rendering_manager = type("RenderingManager", (), {"last_render_dir": None})()
    agent.previous_scores = None
    agent.final_render_dir = None
    agent._create_run_config = lambda: None

    result = await agent._request_critique_impl(update_checkpoint=False)

    assert result == "ok"
    assert "physics-only" in captured["physics_context"]
    assert (
        "Additional SceneBenchmark geometry critic context"
        in captured["physics_context"]
    )
    assert "benchmark-context" in captured["physics_context"]


@pytest.mark.asyncio
async def test_request_critique_retries_with_inline_context_when_model_skips_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    class FakePromptRegistry:
        def get_prompt(self, **_kwargs: Any) -> str:
            return "critic instruction"

    def _invalid_scores() -> FurnitureCritiqueWithScores:
        zero = CategoryScore(name="x", grade=0, comment="missing data")
        return FurnitureCritiqueWithScores(
            critique=(
                "I'm unable to provide a ceiling fixture evaluation because the "
                "required scene observation tools were not executed."
            ),
            realism=zero,
            functionality=zero,
            layout=zero,
            holistic_completeness=zero,
            prompt_following=zero,
            reachability=zero,
        )

    def _valid_scores() -> FurnitureCritiqueWithScores:
        good = CategoryScore(name="x", grade=8, comment="ok")
        return FurnitureCritiqueWithScores(
            critique="Recovered critique",
            realism=good,
            functionality=good,
            layout=good,
            holistic_completeness=good,
            prompt_following=good,
            reachability=good,
        )

    class FakeResult:
        def __init__(
            self,
            response: FurnitureCritiqueWithScores,
            new_items: list[Any] | None = None,
        ):
            self._response = response
            self.new_items = new_items or []

        def final_output_as(self, _type: Any) -> FurnitureCritiqueWithScores:
            return self._response

    async def fake_run(**kwargs: Any) -> FakeResult:
        calls.append(kwargs)
        if len(calls) == 1:
            return FakeResult(_invalid_scores())
        return FakeResult(_valid_scores())

    monkeypatch.setattr(
        base_stateful_agent,
        "check_physics_violations",
        lambda **_kwargs: "physics-only",
    )
    monkeypatch.setattr(base_stateful_agent.Runner, "run", fake_run)
    monkeypatch.setattr(base_stateful_agent, "log_agent_usage", lambda **_kwargs: None)
    monkeypatch.setattr(
        base_stateful_agent, "log_agent_response", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        base_stateful_agent, "log_critique_scores", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        base_stateful_agent, "encode_image_to_base64", lambda _path: "abc123"
    )

    render_dir = tmp_path / "renders_001"
    render_dir.mkdir()
    (render_dir / "view_0.png").write_bytes(b"not-a-real-png")

    agent = object.__new__(_DummyAgent)
    agent.cfg = OmegaConf.create(
        {
            "scenebenchmark_critic": {
                "enabled": False,
                "inject_into_llm_critic": True,
            },
            "agents": {"critic_agent": {"max_turns": 1}},
        }
    )
    agent.scene = _scene(tmp_path / "scene")
    agent.prompt_registry = FakePromptRegistry()
    agent.placement_style = "natural"
    agent.critic = object()
    agent.critic_session = object()
    agent.rendering_manager = type(
        "RenderingManager", (), {"last_render_dir": render_dir}
    )()
    agent.previous_scores = None
    agent.final_render_dir = None
    agent._create_run_config = lambda: None

    result = await agent._request_critique_impl(update_checkpoint=False)

    assert result == "Recovered critique"
    assert len(calls) == 2
    retry_input = calls[1]["input"]
    assert isinstance(retry_input, list)
    assert "IMPORTANT FALLBACK CONTEXT" in retry_input[0]["content"][0]["text"]
    assert retry_input[0]["content"][1]["type"] == "input_image"
    assert (render_dir / "scores.yaml").exists()


@pytest.mark.asyncio
async def test_request_critique_retries_when_critic_hallucinates_without_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    class FakePromptRegistry:
        def get_prompt(self, **_kwargs: Any) -> str:
            return "critic instruction"

    def _hallucinated_scores() -> FurnitureCritiqueWithScores:
        zero = CategoryScore(name="x", grade=0, comment="missing fixtures")
        return FurnitureCritiqueWithScores(
            critique=(
                "Scene observation shows Ceiling Objects Found: None. "
                "The ceiling is completely bare and the room has zero lighting."
            ),
            realism=zero,
            functionality=zero,
            layout=zero,
            holistic_completeness=zero,
            prompt_following=zero,
            reachability=zero,
        )

    def _valid_scores() -> FurnitureCritiqueWithScores:
        good = CategoryScore(name="x", grade=8, comment="ok")
        return FurnitureCritiqueWithScores(
            critique="Recovered critique",
            realism=good,
            functionality=good,
            layout=good,
            holistic_completeness=good,
            prompt_following=good,
            reachability=good,
        )

    class FakeResult:
        def __init__(
            self,
            response: FurnitureCritiqueWithScores,
            new_items: list[Any] | None = None,
        ):
            self._response = response
            self.new_items = new_items or []

        def final_output_as(self, _type: Any) -> FurnitureCritiqueWithScores:
            return self._response

    async def fake_run(**kwargs: Any) -> FakeResult:
        calls.append(kwargs)
        if len(calls) == 1:
            return FakeResult(_hallucinated_scores(), new_items=[])
        return FakeResult(_valid_scores(), new_items=[])

    monkeypatch.setattr(
        base_stateful_agent,
        "check_physics_violations",
        lambda **_kwargs: "physics-only",
    )
    monkeypatch.setattr(base_stateful_agent.Runner, "run", fake_run)
    monkeypatch.setattr(base_stateful_agent, "log_agent_usage", lambda **_kwargs: None)
    monkeypatch.setattr(
        base_stateful_agent, "log_agent_response", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        base_stateful_agent, "log_critique_scores", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        base_stateful_agent, "encode_image_to_base64", lambda _path: "abc123"
    )

    render_dir = tmp_path / "renders_001"
    render_dir.mkdir()
    (render_dir / "view_0.png").write_bytes(b"not-a-real-png")

    agent = object.__new__(_DummyAgent)
    agent.cfg = OmegaConf.create(
        {
            "scenebenchmark_critic": {
                "enabled": False,
                "inject_into_llm_critic": True,
            },
            "agents": {"critic_agent": {"max_turns": 1}},
        }
    )
    agent.scene = _scene(tmp_path / "scene")
    agent.prompt_registry = FakePromptRegistry()
    agent.placement_style = "natural"
    agent.critic = object()
    agent.critic_session = object()
    agent.rendering_manager = type(
        "RenderingManager", (), {"last_render_dir": render_dir}
    )()
    agent.previous_scores = None
    agent.final_render_dir = None
    agent._create_run_config = lambda: None

    result = await agent._request_critique_impl(update_checkpoint=False)

    assert result == "Recovered critique"
    assert len(calls) == 2
    retry_input = calls[1]["input"]
    assert isinstance(retry_input, list)
    assert "IMPORTANT FALLBACK CONTEXT" in retry_input[0]["content"][0]["text"]


def test_evaluate_scenes_disabled_does_not_overwrite_existing_reports(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    scene_dir = output_dir / "scene_000"
    room = _scene(scene_dir / "room_main")
    room_stage_dir = scene_dir / "room_main" / "scene_states" / "final_scene"
    room_stage_dir.mkdir(parents=True)
    (room_stage_dir / "scene_state.json").write_text(
        json.dumps(room.to_state_dict(), indent=2),
        encoding="utf-8",
    )
    report_path = room_stage_dir / "scenebenchmark_critic.json"
    report_path.write_text("stale", encoding="utf-8")

    experiment = object.__new__(IndoorSceneGenerationExperiment)
    experiment.output_dir = output_dir
    experiment.geometry_server = None
    experiment.hssd_server = None
    experiment.objaverse_server = None
    experiment.articulated_server = None
    experiment.materials_server = None
    experiment.cfg = {"experiment": {"scenebenchmark_critic": {"enabled": False}}}

    experiment.evaluate_scenes()

    assert report_path.read_text(encoding="utf-8") == "stale"


def test_evaluate_scenes_defaults_to_room_only_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    scene_dir = output_dir / "scene_000"
    room = _scene(scene_dir / "room_main")
    room_stage_dir = scene_dir / "room_main" / "scene_states" / "final_scene"
    room_stage_dir.mkdir(parents=True)
    (room_stage_dir / "scene_state.json").write_text(
        json.dumps(room.to_state_dict(), indent=2),
        encoding="utf-8",
    )

    house = _house(scene_dir)
    house_stage_dir = scene_dir / "combined_house_after_furniture"
    house_stage_dir.mkdir(parents=True)
    (house_stage_dir / "house_state.json").write_text(
        json.dumps(house.to_state_dict(), indent=2),
        encoding="utf-8",
    )
    house_report_path = house_stage_dir / "scenebenchmark_critic.json"
    house_report_path.write_text("stale", encoding="utf-8")

    experiment = object.__new__(IndoorSceneGenerationExperiment)
    experiment.output_dir = output_dir
    experiment.geometry_server = None
    experiment.hssd_server = None
    experiment.objaverse_server = None
    experiment.articulated_server = None
    experiment.materials_server = None
    experiment.cfg = {
        "experiment": {
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["spatial_accessibility", "functional_dependency"],
            }
        }
    }

    experiment.evaluate_scenes()

    room_report = json.loads(
        (room_stage_dir / "scenebenchmark_critic.json").read_text(encoding="utf-8")
    )

    assert room_report["stage"] == "final_scene"
    assert house_report_path.read_text(encoding="utf-8") == "stale"


def test_evaluate_scenes_refreshes_existing_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    scene_dir = output_dir / "scene_000"
    room = _scene(scene_dir / "room_main")
    room_stage_dir = scene_dir / "room_main" / "scene_states" / "final_scene"
    room_stage_dir.mkdir(parents=True)
    (room_stage_dir / "scene_state.json").write_text(
        json.dumps(room.to_state_dict(), indent=2),
        encoding="utf-8",
    )

    house = _house(scene_dir)
    house_stage_dir = scene_dir / "combined_house_after_furniture"
    house_stage_dir.mkdir(parents=True)
    (house_stage_dir / "house_state.json").write_text(
        json.dumps(house.to_state_dict(), indent=2),
        encoding="utf-8",
    )
    (house_stage_dir / "scenebenchmark_critic.json").write_text(
        "stale", encoding="utf-8"
    )

    experiment = object.__new__(IndoorSceneGenerationExperiment)
    experiment.output_dir = output_dir
    experiment.geometry_server = None
    experiment.hssd_server = None
    experiment.objaverse_server = None
    experiment.articulated_server = None
    experiment.materials_server = None
    experiment.cfg = {
        "experiment": {
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["spatial_accessibility", "functional_dependency"],
                "room_stage_hooks": ["final_scene"],
                "house_stage_hooks": ["combined_house_after_furniture"],
            }
        }
    }

    experiment.evaluate_scenes()

    room_report = json.loads(
        (room_stage_dir / "scenebenchmark_critic.json").read_text(encoding="utf-8")
    )
    house_report = json.loads(
        (house_stage_dir / "scenebenchmark_critic.json").read_text(encoding="utf-8")
    )
    house_object_ids = {
        obj["id"] for obj in house_report["case_pack"]["scene_geometry"]["objects"]
    }

    assert room_report["stage"] == "final_scene"
    assert house_report["stage"] == "combined_house_after_furniture"
    assert "table_0" in house_object_ids
    assert "mug_0" not in house_object_ids
