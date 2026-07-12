"""Tests for hard limits around planner/designer/critic workflows."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.room import AgentType
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)


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


def test_manipuland_rejects_destructive_change_that_contradicts_complete_inventory(
    monkeypatch,
) -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = "dining_table_0"
    agent.scene = object()
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
        lambda scene, stage: {"scene_geometry": {}},
    )
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.evaluate_manipuland_completeness",
        lambda case_pack: [
            {
                "label": "pass",
                "primary_object": "dining_table_0",
                "diagnostics": {"counts": {"plate": 4, "utensil": 4}},
            }
        ],
    )

    result = agent._validate_design_change_instruction(
        "All plates and cutlery are missing. Remove everything and regenerate the set."
    )

    assert result is not None
    assert "DESIGN CHANGE REJECTED" in result
    assert "plate=4" in result


def test_manipuland_allows_completion_fix_when_deterministic_check_fails(
    monkeypatch,
) -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = "dining_table_0"
    agent.scene = object()
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
        lambda scene, stage: {"scene_geometry": {}},
    )


def test_manipuland_rejects_addition_that_contradicts_complete_inventory(
    monkeypatch,
) -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = "dining_table_0"
    agent.scene = object()
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
        lambda scene, stage: {"scene_geometry": {}},
    )
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.evaluate_manipuland_completeness",
        lambda case_pack: [
            {
                "label": "pass",
                "primary_object": "dining_table_0",
                "diagnostics": {
                    "counts": {"plate": 4, "utensil": 4, "drinkware": 4}
                },
            }
        ],
    )

    result = agent._validate_design_change_instruction(
        "Only two place settings exist. Add two complete place settings with plates, "
        "cutlery, and glasses."
    )

    assert result is not None
    assert "DESIGN CHANGE REJECTED" in result
    assert "drinkware=4" in result


def test_manipuland_rejects_inventory_generation_without_missing_word(
    monkeypatch,
) -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = "dining_table_0"
    agent.scene = object()
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.room_scene_to_case_pack",
        lambda scene, stage: {"scene_geometry": {}},
    )
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.evaluate_manipuland_completeness",
        lambda case_pack: [
            {
                "label": "pass",
                "primary_object": "dining_table_0",
                "diagnostics": {"counts": {"plate": 4, "utensil": 4}},
            }
        ],
    )

    result = agent._validate_design_change_instruction(
        "Generate separate fork, knife, and spoon assets for formal dining."
    )

    assert result is not None
    assert "DESIGN CHANGE REJECTED" in result
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.evaluate_manipuland_completeness",
        lambda case_pack: [
            {"label": "fail", "primary_object": "dining_table_0"}
        ],
    )

    assert (
        agent._validate_design_change_instruction(
            "All plates are missing. Clear the incomplete placeholders and rebuild them."
        )
        is None
    )


def test_manipuland_allows_non_destructive_geometry_correction(monkeypatch) -> None:
    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = "dining_table_0"
    agent.scene = object()

    assert (
        agent._validate_design_change_instruction(
            "Rotate two existing glasses and increase spacing between the plates."
        )
        is None
    )
