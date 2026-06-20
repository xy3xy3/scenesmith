"""Structured scoring and critique utilities for design evaluation.

This module provides data structures and utilities for critic-based evaluation
with categorical scoring. Includes score computation, delta tracking, and formatted
output for iterative design improvement.
"""

import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

console_logger = logging.getLogger(__name__)


@dataclass
class CategoryScore:
    """Score for a single evaluation category.

    Used by both furniture and manipuland agents for structured critique output.
    """

    name: str
    """Category name (e.g., 'realism', 'functionality') - for logging."""
    grade: int
    """Score from 0-10."""
    comment: str
    """Brief justification for the score."""


@dataclass
class CritiqueWithScores(ABC):
    """Base class for agent-specific critique scoring.

    Subclasses define agent-specific categories and implement get_scores()
    for generic utility function access.
    """

    critique: str
    """Natural language critique for the planner."""

    @abstractmethod
    def get_scores(self) -> list[CategoryScore]:
        """Return all category scores for generic processing.

        Returns:
            List of CategoryScore objects for all categories.
        """

    def _canonicalize_scores(
        self, named_scores: list[tuple[str, CategoryScore]]
    ) -> list[CategoryScore]:
        """Normalize model-authored score names to stable category labels."""
        canonicalized_scores = []

        for canonical_name, score in named_scores:
            if score.name != canonical_name:
                console_logger.warning(
                    "Normalizing %s score category name from %r to %r.",
                    type(self).__name__,
                    score.name,
                    canonical_name,
                )
                score.name = canonical_name
            canonicalized_scores.append(score)

        return canonicalized_scores


@dataclass
class FloorPlanCritiqueWithScores(CritiqueWithScores):
    """Floor plan agent critique with layout-specific categories.

    Categories evaluate floor plan design quality.
    """

    room_proportions: CategoryScore
    """How well room sizes match intended use (bedrooms vs living areas)."""
    spatial_flow: CategoryScore
    """How well rooms connect and traffic flows through the space."""
    natural_lighting: CategoryScore
    """Window placement for natural light distribution."""
    material_consistency: CategoryScore
    """How well materials match room purposes and coordinate visually."""
    prompt_following: CategoryScore
    """How well the design adheres to the original house description requirements."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all floor plan critique category scores."""
        return self._canonicalize_scores(
            [
                ("Room Proportions", self.room_proportions),
                ("Spatial Flow", self.spatial_flow),
                ("Natural Lighting", self.natural_lighting),
                ("Material Consistency", self.material_consistency),
                ("Prompt Following", self.prompt_following),
            ]
        )


@dataclass
class FurnitureCritiqueWithScores(CritiqueWithScores):
    """Furniture agent critique with standard categories.

    Categories evaluate furniture placement quality (6 categories, 0-60 total).
    """

    realism: CategoryScore
    """How realistic the arrangement appears."""
    functionality: CategoryScore
    """How well items support intended activities."""
    layout: CategoryScore
    """Whether items are arranged logically."""
    holistic_completeness: CategoryScore
    """How complete and appropriately populated the arrangement is."""
    prompt_following: CategoryScore
    """How well the scene adheres to the original prompt requirements."""
    reachability: CategoryScore
    """Whether all areas of the room are traversable."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all furniture critique category scores."""
        return self._canonicalize_scores(
            [
                ("Realism", self.realism),
                ("Functionality", self.functionality),
                ("Layout", self.layout),
                ("Holistic Completeness", self.holistic_completeness),
                ("Prompt Following", self.prompt_following),
                ("Reachability", self.reachability),
            ]
        )


@dataclass
class ManipulandCritiqueWithScores(CritiqueWithScores):
    """Manipuland agent critique with standard categories.

    Categories evaluate manipuland placement quality.
    """

    realism: CategoryScore
    """How realistic the arrangement appears."""
    functionality: CategoryScore
    """How well items support intended activities."""
    layout: CategoryScore
    """Whether items are arranged logically."""
    holistic_completeness: CategoryScore
    """How complete and appropriately populated the arrangement is."""
    prompt_following: CategoryScore
    """How well the scene adheres to the original prompt requirements."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all manipuland critique category scores."""
        return self._canonicalize_scores(
            [
                ("Realism", self.realism),
                ("Functionality", self.functionality),
                ("Layout", self.layout),
                ("Holistic Completeness", self.holistic_completeness),
                ("Prompt Following", self.prompt_following),
            ]
        )


@dataclass
class WallCritiqueWithScores(CritiqueWithScores):
    """Wall agent critique with standard categories.

    Categories evaluate wall-mounted object placement quality.
    """

    realism: CategoryScore
    """Natural placement patterns (art at eye level, appropriate heights)."""
    functionality: CategoryScore
    """Objects serve purpose, accessible, not blocked by furniture."""
    layout: CategoryScore
    """Distribution across walls, spacing from doors/windows, visual balance."""
    holistic_completeness: CategoryScore
    """Walls appropriately decorated for room type (not bare, not overcrowded)."""
    prompt_following: CategoryScore
    """Requested wall items present."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all wall critique category scores."""
        return self._canonicalize_scores(
            [
                ("Realism", self.realism),
                ("Functionality", self.functionality),
                ("Layout", self.layout),
                ("Holistic Completeness", self.holistic_completeness),
                ("Prompt Following", self.prompt_following),
            ]
        )


@dataclass
class CeilingCritiqueWithScores(CritiqueWithScores):
    """Ceiling agent critique with 4 standard categories.

    Categories evaluate ceiling-mounted object placement quality.
    Note: Has 4 categories (not 5) - skips Holistic Completeness since ceiling
    decoration is minimal (usually just lights) and completeness is implicitly
    covered by Functionality (adequate lighting).
    """

    realism: CategoryScore
    """Natural fixture placement (centered, over functional areas like tables)."""
    functionality: CategoryScore
    """Adequate lighting coverage, appropriate fixture types for room."""
    layout: CategoryScore
    """Symmetry, spacing, clearance from tall furniture below."""
    prompt_following: CategoryScore
    """Requested ceiling fixtures present."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all ceiling critique category scores."""
        return self._canonicalize_scores(
            [
                ("Realism", self.realism),
                ("Functionality", self.functionality),
                ("Layout", self.layout),
                ("Prompt Following", self.prompt_following),
            ]
        )


def _normalize_score_key(name: str) -> str:
    """Create a stable comparison key for score category names."""
    return " ".join(name.replace("_", " ").split()).casefold()


def align_scores_for_comparison(
    current: CritiqueWithScores, previous: CritiqueWithScores
) -> list[tuple[CategoryScore, CategoryScore]]:
    """Align score categories by normalized name for safe comparisons."""
    current_by_key = {
        _normalize_score_key(score.name): score for score in current.get_scores()
    }
    previous_by_key = {
        _normalize_score_key(score.name): score for score in previous.get_scores()
    }

    missing_from_previous = [
        score.name
        for key, score in current_by_key.items()
        if key not in previous_by_key
    ]
    missing_from_current = [
        score.name
        for key, score in previous_by_key.items()
        if key not in current_by_key
    ]

    if missing_from_previous or missing_from_current:
        console_logger.warning(
            "Score categories did not fully align for comparison. "
            "Missing from previous: %s. Missing from current: %s.",
            missing_from_previous or "none",
            missing_from_current or "none",
        )

    aligned_pairs: list[tuple[CategoryScore, CategoryScore]] = []
    for score in current.get_scores():
        key = _normalize_score_key(score.name)
        previous_score = previous_by_key.get(key)
        if previous_score is not None:
            aligned_pairs.append((score, previous_score))

    return aligned_pairs


def compute_total_score(scores: CritiqueWithScores) -> int:
    """Compute total score across all categories.

    Args:
        scores: Critique scores containing all categories.

    Returns:
        Sum of all category grades (range depends on number of categories).
    """
    return sum(score.grade for score in scores.get_scores())


def compute_score_deltas(
    current: CritiqueWithScores, previous: CritiqueWithScores
) -> dict[str, int]:
    """Compute per-category score changes.

    Args:
        current: Current critique scores.
        previous: Previous critique scores.

    Returns:
        Dictionary mapping category names to score changes (can be negative).
    """
    return {
        current_score.name: current_score.grade - previous_score.grade
        for current_score, previous_score in align_scores_for_comparison(
            current=current,
            previous=previous,
        )
    }


def scores_to_dict(scores: CritiqueWithScores) -> dict[str, dict[str, Any]]:
    """Convert scores to YAML-serializable dictionary.

    Args:
        scores: Critique scores to convert.

    Returns:
        Dictionary with category scores and summary, suitable for YAML output.
    """
    result = {
        score.name: {
            "grade": score.grade,
            "comment": score.comment,
        }
        for score in scores.get_scores()
    }
    result["summary"] = scores.critique
    return result


def log_critique_scores(
    scores: CritiqueWithScores, title: str = "CRITIQUE SCORES"
) -> None:
    """Log critique scores with consistent formatting.

    Args:
        scores: Critique scores to log.
        title: Title to display in log header.
    """
    console_logger = logging.getLogger(__name__)
    console_logger.info("=" * 60)
    console_logger.info(title)
    console_logger.info("=" * 60)
    for score in scores.get_scores():
        console_logger.info(
            f"{score.name.replace('_', ' ').title()}: {score.grade}/10"
        )
        console_logger.info(f"  {score.comment}")
    console_logger.info("=" * 60)


def log_score_deltas(deltas: dict[str, int]) -> None:
    """Log score changes with formatted output.

    Args:
        deltas: Dictionary mapping category names to score changes.
    """
    console_logger = logging.getLogger(__name__)
    if not deltas:
        console_logger.info("Score changes unavailable: no comparable categories.")
        return

    sum_delta = sum(deltas.values())
    max_drop = abs(min(min(deltas.values(), default=0), 0))

    parts = [
        f"{name.replace('_', ' ').title()} {delta:+d}"
        for name, delta in deltas.items()
    ]
    console_logger.info(
        f"Score changes: {', '.join(parts)}. "
        f"**Total: {sum_delta:+d}** (Max drop: {max_drop})"
    )


def format_score_deltas_for_planner(
    current_scores: CritiqueWithScores,
    previous_scores: CritiqueWithScores,
    format_style: str = "detailed",
) -> str:
    """Format score changes for planner agent feedback.

    Args:
        current_scores: Current critique scores.
        previous_scores: Previous critique scores.
        format_style: Formatting style - "detailed" (with transitions) or
            "compact" (deltas only).

    Returns:
        Formatted string showing score changes.
    """
    aligned_scores = align_scores_for_comparison(
        current=current_scores,
        previous=previous_scores,
    )
    deltas = {
        current_score.name: current_score.grade - previous_score.grade
        for current_score, previous_score in aligned_scores
    }

    if not aligned_scores:
        return "\n\n**Score Changes:** unavailable (no comparable score categories)."

    sum_delta = sum(deltas.values())

    if format_style == "detailed":
        max_drop = abs(min(min(deltas.values(), default=0), 0))
        lines = ["\n\n**Score Changes:**"]
        for current_score, previous_score in aligned_scores:
            display_name = current_score.name.replace("_", " ").title()
            prev_grade = previous_score.grade
            curr_grade = current_score.grade
            lines.append(
                f"{display_name}: {prev_grade}→{curr_grade} "
                f"({curr_grade - prev_grade:+d})"
            )
        lines.append(f"**Total: {sum_delta:+d}** (Max drop: {max_drop})")
        return "\n".join(lines)
    else:  # compact
        delta_lines = [
            f"- {name.replace('_', ' ').title()}: {delta:+d}"
            for name, delta in deltas.items()
        ]
        return (
            f"\n\n## Score Changes from Previous Iteration\n"
            f"{chr(10).join(delta_lines)}\n"
            f"- **Total: {sum_delta:+d}**"
        )


def log_agent_response(response: str, agent_name: str) -> None:
    """Log agent response with consistent formatting.

    Args:
        response: The agent's final response text.
        agent_name: Name of the agent (e.g., "PLANNER", "DESIGNER").
    """
    console_logger = logging.getLogger(__name__)
    console_logger.info("=" * 60)
    console_logger.info(f"{agent_name} RESPONSE")
    console_logger.info("=" * 60)
    console_logger.info(response)
    console_logger.info("=" * 60)
