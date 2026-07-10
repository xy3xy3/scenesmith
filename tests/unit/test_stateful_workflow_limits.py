"""Tests for hard limits around planner/designer/critic workflows."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.room import AgentType


class _WorkflowAgent(BaseStatefulAgent):
    _is_placement_agent = False

    @property
    def agent_type(self) -> AgentType:
        return AgentType.FURNITURE

    def _get_design_change_prompt_enum(self):
        raise NotImplementedError

    def _get_initial_design_prompt_enum(self):
        raise NotImplementedError

    def _get_initial_design_prompt_kwargs(self):
        raise NotImplementedError

    def _get_critique_prompt_enum(self):
        raise NotImplementedError

    def _get_final_scores_directory(self) -> Path:
        raise NotImplementedError

    def _set_placement_noise_profile(self, mode):
        raise NotImplementedError


class _MutableScene:
    def __init__(self) -> None:
        self.version = 0

    def content_hash(self) -> str:
        return f"scene-{self.version}"


def _agent(max_rounds: int = 2, no_progress_rounds: int = 2) -> _WorkflowAgent:
    agent = object.__new__(_WorkflowAgent)
    agent.cfg = OmegaConf.create(
        {
            "max_critique_rounds": max_rounds,
            "max_no_progress_rounds": no_progress_rounds,
        }
    )
    agent.scene = _MutableScene()
    agent._request_initial_design_impl = AsyncMock(return_value="initial")
    agent._request_critique_impl = AsyncMock(return_value="critique")
    agent._request_design_change_impl = AsyncMock(return_value="changed")
    agent._reset_planner_workflow_tracking()
    return agent


def _tool(agent: _WorkflowAgent, name: str):
    return next(tool for tool in agent._create_planner_tools() if tool.name == name)


@pytest.mark.asyncio
async def test_critique_tool_enforces_configured_round_budget() -> None:
    agent = _agent(max_rounds=1)
    critique = _tool(agent, "request_critique")

    assert await critique.on_invoke_tool(Mock(), {}) == "critique"
    stopped = await critique.on_invoke_tool(Mock(), {})

    assert "HARD STOP" in stopped
    assert "max_critique_rounds=1" in stopped
    agent._request_critique_impl.assert_awaited_once()


@pytest.mark.asyncio
async def test_critique_tool_stops_when_scene_did_not_change() -> None:
    agent = _agent(max_rounds=3)
    critique = _tool(agent, "request_critique")

    assert await critique.on_invoke_tool(Mock(), {}) == "critique"
    stopped = await critique.on_invoke_tool(Mock(), {})

    assert "scene is unchanged" in stopped
    agent._request_critique_impl.assert_awaited_once()


@pytest.mark.asyncio
async def test_design_change_tool_stops_after_consecutive_no_progress() -> None:
    agent = _agent(max_rounds=3, no_progress_rounds=2)
    design_change = _tool(agent, "request_design_change")

    assert await design_change.on_invoke_tool(
        Mock(), '{"instruction": "first fix"}'
    ) == "changed"
    stopped = await design_change.on_invoke_tool(
        Mock(), '{"instruction": "second fix"}'
    )

    assert "HARD STOP" in stopped
    assert "no scene-state change" in stopped
    assert agent._request_design_change_impl.await_count == 2
