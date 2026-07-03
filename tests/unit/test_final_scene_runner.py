from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from scenesmith.utils.final_scene_runner import (
    apply_runtime_overrides,
    discover_final_scene_states,
    find_nearest_resolved_config,
    relative_output_subdir,
)


def test_discover_final_scene_states_recurses(tmp_path: Path) -> None:
    state_a = (
        tmp_path
        / "critic_off"
        / "batch_001"
        / "scene_001"
        / "room_living_room"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    state_b = (
        tmp_path
        / "critic_off"
        / "batch_002"
        / "scene_002"
        / "room_bedroom"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    state_a.parent.mkdir(parents=True)
    state_b.parent.mkdir(parents=True)
    state_a.write_text("{}", encoding="utf-8")
    state_b.write_text("{}", encoding="utf-8")

    matches = discover_final_scene_states(tmp_path / "critic_off")

    assert matches == [state_a.resolve(), state_b.resolve()]


def test_find_nearest_resolved_config_walks_upwards(tmp_path: Path) -> None:
    resolved_config = tmp_path / "batch_001" / "resolved_config.yaml"
    scene_state = (
        tmp_path
        / "batch_001"
        / "scene_001"
        / "room_living_room"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    resolved_config.parent.mkdir(parents=True)
    scene_state.parent.mkdir(parents=True)
    resolved_config.write_text("{}", encoding="utf-8")
    scene_state.write_text("{}", encoding="utf-8")

    assert find_nearest_resolved_config(scene_state) == resolved_config.resolve()


def test_relative_output_subdir_preserves_target_relative_layout(
    tmp_path: Path,
) -> None:
    target = tmp_path / "critic_off"
    scene_state = (
        target
        / "batch_001"
        / "scene_001"
        / "room_living_room"
        / "scene_states"
        / "final_scene"
        / "scene_state.json"
    )
    scene_state.parent.mkdir(parents=True)
    scene_state.write_text("{}", encoding="utf-8")

    rel = relative_output_subdir(scene_state, target)

    assert rel == Path(
        "batch_001/scene_001/room_living_room/scene_states/final_scene"
    )


def test_apply_runtime_overrides_enables_final_scene_only() -> None:
    cfg = OmegaConf.create(
        {
            "openai": {"base_url": None, "use_responses": True},
            "experiment": {
                "scenebenchmark_critic": {
                    "enabled": False,
                    "room_stage_hooks": ["scene_after_furniture"],
                    "house_stage_hooks": ["combined_house"],
                    "asset_annotation": {
                        "skip_existing": True,
                        "refresh": False,
                        "write_scene_state": True,
                    },
                }
            },
        }
    )

    updated = apply_runtime_overrides(
        cfg,
        base_url="http://127.0.0.1:8002/v1",
        model="demo-model",
        use_responses=False,
        write_scene_state=False,
    )

    assert updated.openai.base_url == "http://127.0.0.1:8002/v1"
    assert updated.openai.model == "demo-model"
    assert updated.openai.use_responses is False
    assert updated.experiment.scenebenchmark_critic.enabled is True
    assert list(updated.experiment.scenebenchmark_critic.room_stage_hooks) == [
        "final_scene"
    ]
    assert list(updated.experiment.scenebenchmark_critic.house_stage_hooks) == []
    assert updated.experiment.scenebenchmark_critic.asset_annotation.skip_existing is False
    assert updated.experiment.scenebenchmark_critic.asset_annotation.refresh is True
    assert (
        updated.experiment.scenebenchmark_critic.asset_annotation.write_scene_state
        is False
    )
