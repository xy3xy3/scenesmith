"""Tests for 7.3 wall prompt requirement extraction and prompt wiring."""

import unittest

from scenesmith.prompts import prompt_manager, prompt_registry
from scenesmith.prompts.registry import WallAgentPrompts
from scenesmith.wall_agents.prompt_constraints import (
    build_required_wall_object_constraints,
)


class TestWallPromptRequirements(unittest.TestCase):
    """Validate deterministic wall constraint extraction for media displays."""

    def test_tv_on_opposite_wall_becomes_required_wall_object(self):
        """TV + TV stand prompts should produce a hard wall-stage requirement."""
        constraints = build_required_wall_object_constraints(
            "A living room with a sofa facing a TV stand and television on the opposite wall."
        )

        self.assertIn("REQUIRED media display", constraints)
        self.assertIn("opposite wall", constraints)
        self.assertIn("TV stand/media console", constraints)

    def test_desktop_monitor_without_wall_context_is_not_promoted(self):
        """Desk monitors should stay out of wall-stage obligations."""
        constraints = build_required_wall_object_constraints(
            "A home office with a desk, monitor, keyboard, and office chair."
        )

        self.assertIn("No explicit wall-object obligations", constraints)

    def test_wall_prompts_accept_required_wall_objects_template_var(self):
        """Prompt files should render cleanly with the new required-wall context."""
        required_wall_objects = "- REQUIRED media display: place a wall-mounted television on the opposite wall."

        planner_prompt = prompt_manager.get_prompt(
            prompt_name=WallAgentPrompts.STATEFUL_PLANNER_AGENT,
            room_description="Living room with a media wall.",
            wall_count=4,
            required_wall_objects=required_wall_objects,
            max_critique_rounds=2,
            reset_single_category_threshold=2,
            reset_total_sum_threshold=4,
            early_finish_min_score=8,
        )
        self.assertIn(required_wall_objects, planner_prompt)

        designer_prompt = prompt_manager.get_prompt(
            prompt_name=WallAgentPrompts.DESIGNER_AGENT,
            room_description="Living room with a media wall.",
            wall_count=4,
            required_wall_objects=required_wall_objects,
        )
        self.assertIn(required_wall_objects, designer_prompt)

        critic_prompt = prompt_manager.get_prompt(
            prompt_name=WallAgentPrompts.STATEFUL_CRITIC_AGENT,
            room_description="Living room with a media wall.",
            wall_count=4,
            required_wall_objects=required_wall_objects,
        )
        self.assertIn(required_wall_objects, critic_prompt)

        initial_instruction = prompt_registry.get_prompt(
            prompt_enum=WallAgentPrompts.DESIGNER_INITIAL_INSTRUCTION,
            wall_summary="- wall_0: 4.0m wide x 2.6m tall",
            required_wall_objects=required_wall_objects,
        )
        self.assertIn(required_wall_objects, initial_instruction)


if __name__ == "__main__":
    unittest.main()
