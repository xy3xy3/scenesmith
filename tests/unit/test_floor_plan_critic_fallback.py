"""Unit tests for floor-plan critic render fallbacks."""

import asyncio

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from scenesmith.agent_utils.scoring import CategoryScore, FloorPlanCritiqueWithScores
from scenesmith.floor_plan_agents import stateful_floor_plan_agent as floor_plan_module


def _scores() -> FloorPlanCritiqueWithScores:
    score = CategoryScore(name="ok", grade=8, comment="ok")
    return FloorPlanCritiqueWithScores(
        critique="ok",
        room_proportions=score,
        spatial_flow=score,
        natural_lighting=score,
        material_consistency=score,
        prompt_following=score,
    )


@pytest.mark.asyncio
async def test_floor_plan_critique_reuses_prior_render_when_model_skips_observe_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final critique should not crash when a local model skips observe_scene."""

    class FakeResult:
        def final_output_as(self, _type):
            return _scores()

    async def fake_run(**_kwargs):
        return FakeResult()

    monkeypatch.setattr(floor_plan_module.Runner, "run", fake_run)
    monkeypatch.setattr(floor_plan_module, "log_agent_usage", lambda **_kwargs: None)
    monkeypatch.setattr(
        floor_plan_module, "log_agent_response", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        floor_plan_module, "log_critique_scores", lambda *_args, **_kwargs: None
    )

    render_dir = tmp_path / "renders_001"
    render_dir.mkdir()

    agent = object.__new__(floor_plan_module.StatefulFloorPlanAgent)
    agent.cfg = OmegaConf.create({"agents": {"critic_agent": {"max_turns": 1}}})
    agent.prompt_registry = SimpleNamespace(get_prompt=lambda **_kwargs: "critic")
    agent.critic = object()
    agent.critic_session = object()
    agent.layout = SimpleNamespace(to_dict=lambda: {}, content_hash=lambda: "hash")
    agent.previous_scores = None
    agent.scene_checkpoint = None
    agent.checkpoint_scores = None
    agent.previous_scene_checkpoint = None
    agent.previous_checkpoint_scores = None
    agent.previous_checkpoint_render_dir = None
    agent.checkpoint_render_dir = render_dir
    agent.checkpoint_scene_hash = None
    agent.final_render_dir = None
    agent._create_run_config = lambda: None

    vision_tools = Mock()
    vision_tools.last_render_dir = None
    vision_tools._observe_scene_impl = Mock()
    agent._get_vision_tools = lambda: vision_tools

    result = await agent._request_critique_impl(update_checkpoint=False)

    assert result == "ok"
    assert agent.final_render_dir == render_dir
    assert (render_dir / "scores.yaml").exists()
    vision_tools._observe_scene_impl.assert_not_called()
