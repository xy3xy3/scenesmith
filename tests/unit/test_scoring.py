"""Unit tests for score normalization and comparison helpers."""

import unittest

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.room import AgentType
from scenesmith.agent_utils.scoring import (
    CategoryScore,
    CritiqueWithScores,
    FurnitureCritiqueWithScores,
    compute_score_deltas,
    format_score_deltas_for_planner,
)


def _make_furniture_scores(
    realism_name: str = "Realism",
    functionality_name: str = "Functionality",
    layout_name: str = "Layout",
    holistic_name: str = "Holistic Completeness",
    prompt_name: str = "Prompt Following",
    reachability_name: str = "Reachability",
    realism_grade: int = 7,
    functionality_grade: int = 7,
    layout_grade: int = 7,
    holistic_grade: int = 7,
    prompt_grade: int = 7,
    reachability_grade: int = 7,
) -> FurnitureCritiqueWithScores:
    """Build furniture critique scores with configurable nested score names."""
    return FurnitureCritiqueWithScores(
        critique="test critique",
        realism=CategoryScore(
            name=realism_name,
            grade=realism_grade,
            comment="test",
        ),
        functionality=CategoryScore(
            name=functionality_name,
            grade=functionality_grade,
            comment="test",
        ),
        layout=CategoryScore(name=layout_name, grade=layout_grade, comment="test"),
        holistic_completeness=CategoryScore(
            name=holistic_name,
            grade=holistic_grade,
            comment="test",
        ),
        prompt_following=CategoryScore(
            name=prompt_name,
            grade=prompt_grade,
            comment="test",
        ),
        reachability=CategoryScore(
            name=reachability_name,
            grade=reachability_grade,
            comment="test",
        ),
    )


@dataclass
class AdhocCritiqueWithScores(CritiqueWithScores):
    """Simple score container for comparison edge cases."""

    scores: list[CategoryScore]

    def get_scores(self) -> list[CategoryScore]:
        return self.scores


class TestScoreComparison(unittest.TestCase):
    """Tests for score normalization and delta formatting."""

    def test_compute_score_deltas_normalizes_model_authored_names(self):
        """Score deltas should not depend on critic-provided casing."""
        previous = _make_furniture_scores(
            realism_name="realism",
            functionality_name="functionality",
            layout_name="layout",
            holistic_name="holistic completeness",
            prompt_name="prompt_following",
            reachability_name="reachability",
            realism_grade=6,
        )
        current = _make_furniture_scores(realism_grade=8)

        deltas = compute_score_deltas(current=current, previous=previous)

        self.assertEqual(deltas["Realism"], 2)
        self.assertEqual(
            set(deltas),
            {
                "Realism",
                "Functionality",
                "Layout",
                "Holistic Completeness",
                "Prompt Following",
                "Reachability",
            },
        )

    def test_format_score_deltas_handles_missing_categories_gracefully(self):
        """Planner formatting should degrade gracefully when categories differ."""
        previous = AdhocCritiqueWithScores(
            critique="before",
            scores=[
                CategoryScore(name="Functionality", grade=6, comment="test"),
                CategoryScore(name="Layout", grade=7, comment="test"),
            ],
        )
        current = AdhocCritiqueWithScores(
            critique="after",
            scores=[
                CategoryScore(name="Realism", grade=8, comment="test"),
                CategoryScore(name="Layout", grade=5, comment="test"),
            ],
        )

        message = format_score_deltas_for_planner(
            current_scores=current,
            previous_scores=previous,
            format_style="detailed",
        )

        self.assertIn("Layout: 7", message)
        self.assertNotIn("Realism:", message)

    def test_should_reset_aligns_categories_by_name_not_list_position(self):
        """Checkpoint reset logic should compare matching categories only."""

        class TestableAgent(BaseStatefulAgent):
            def __init__(self):
                self.cfg = MagicMock()
                self.cfg.reset_single_category_threshold = 3
                self.cfg.reset_total_sum_threshold = 99

            @property
            def agent_type(self) -> AgentType:
                return AgentType.FURNITURE

            def _get_final_scores_directory(self) -> Path:
                return Path(".")

            def _get_critique_prompt_enum(self) -> Any:
                return None

            def _get_design_change_prompt_enum(self) -> Any:
                return None

            def _get_initial_design_prompt_enum(self) -> Any:
                return None

            def _get_initial_design_prompt_kwargs(self) -> dict:
                return {}

            def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
                pass

        previous = AdhocCritiqueWithScores(
            critique="before",
            scores=[
                CategoryScore(name="Realism", grade=9, comment="test"),
                CategoryScore(name="Layout", grade=8, comment="test"),
            ],
        )
        current = AdhocCritiqueWithScores(
            critique="after",
            scores=[
                CategoryScore(name="Layout", grade=8, comment="test"),
                CategoryScore(name="Realism", grade=5, comment="test"),
            ],
        )

        should_reset, reason = TestableAgent()._should_reset_to_checkpoint(
            current_scores=current,
            previous_scores=previous,
        )

        self.assertTrue(should_reset)
        self.assertEqual(reason, "Realism dropped 4 points")


if __name__ == "__main__":
    unittest.main()
