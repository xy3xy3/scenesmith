from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import lxml.etree as ET
import numpy as np
import pytest

from omegaconf import OmegaConf
from pydrake.math import RigidTransform

import scenesmith.agent_utils.base_stateful_agent as base_stateful_agent

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.house import (
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
    evaluate_room_scene,
    format_prompt_context,
    write_room_stage_report,
)
from scenesmith.scenebenchmark_critic.adapter import (
    house_scene_to_case_pack,
    room_scene_to_case_pack,
)


def _box_object(
    object_id: str,
    name: str,
    object_type: ObjectType,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> SceneObject:
    sx, sy, sz = size
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=name,
        description=name,
        transform=RigidTransform(p=list(center)),
        bbox_min=np.array([-sx / 2.0, -sy / 2.0, -sz / 2.0]),
        bbox_max=np.array([sx / 2.0, sy / 2.0, sz / 2.0]),
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
    assert room["floor_polygon"] == [[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]]
    table = next(
        obj for obj in case_pack["scene_geometry"]["objects"] if obj["id"] == "table_0"
    )
    assert table["support_regions"][0]["region_id"] == "S_0"
    assert any(
        check["metric"] == "functional_dependency"
        and check["subject_id"] == "mug_0"
        and check["target_ids"] == ["table_0"]
        for check in case_pack["checks"]
    )


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
    assert (stage_dir / "scenebenchmark_critic.md").exists()


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
