"""Deterministic scale checks for common manipuland objects.

7.3 fix: common desktop manipulands were keeping bad retrieved-mesh aspect
ratios after uniform scaling, so this module provides hard size envelopes and
fit diagnostics before the LLM critic has to reason about scale from images.
"""

from __future__ import annotations

import re

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DimensionRange:
    """Inclusive physical size bounds for one axis."""

    min: float
    max: float

    def clamp(self, value: float) -> float:
        """Clamp a value into this range."""
        return min(max(value, self.min), self.max)

    def contains(self, value: float) -> bool:
        """Return True when value lies in this range."""
        return self.min <= value <= self.max


@dataclass(frozen=True)
class SizeProfile:
    """Real-world size envelope for a common manipuland category."""

    name: str
    patterns: tuple[str, ...]
    width: DimensionRange
    depth: DimensionRange
    height: DimensionRange
    footprint_swappable: bool = False

    @property
    def ranges(self) -> tuple[DimensionRange, DimensionRange, DimensionRange]:
        """Return width/depth/height ranges."""
        return self.width, self.depth, self.height


@dataclass(frozen=True)
class UniformScaleFit:
    """Result of fitting a mesh to target dimensions using uniform scale."""

    current_dimensions: tuple[float, float, float]
    desired_dimensions: tuple[float, float, float]
    uniform_scale: float
    actual_dimensions: tuple[float, float, float]
    axis_relative_errors: tuple[float, float, float]
    max_axis_relative_error: float
    axis_order: tuple[int, int, int]

    @property
    def passes_default_threshold(self) -> bool:
        """Whether the fit passes the default manipuland threshold."""
        return self.max_axis_relative_error <= DEFAULT_MAX_AXIS_RELATIVE_ERROR


DEFAULT_MAX_AXIS_RELATIVE_ERROR = 0.75

SIZE_PROFILES: tuple[SizeProfile, ...] = (
    SizeProfile(
        name="notebook_book",
        patterns=(
            r"\bnotebook\b",
            r"\bbook\b",
            r"\bjournal\b",
            r"\bsketchbook\b",
            r"\bmagazine\b",
        ),
        width=DimensionRange(0.09, 0.30),
        depth=DimensionRange(0.09, 0.30),
        height=DimensionRange(0.008, 0.06),
        footprint_swappable=True,
    ),
    SizeProfile(
        name="computer_monitor",
        patterns=(r"\bmonitor\b", r"\bscreen\b", r"\bdisplay\b"),
        width=DimensionRange(0.35, 0.75),
        depth=DimensionRange(0.06, 0.25),
        height=DimensionRange(0.22, 0.55),
    ),
    SizeProfile(
        name="desk_lamp",
        patterns=(r"\bdesk lamp\b", r"\blamp\b", r"\btable lamp\b"),
        width=DimensionRange(0.08, 0.25),
        depth=DimensionRange(0.08, 0.25),
        height=DimensionRange(0.25, 0.65),
        footprint_swappable=True,
    ),
    SizeProfile(
        name="pen_holder",
        patterns=(
            r"\bpen holder\b",
            r"\bpencil holder\b",
            r"\bpen cup\b",
            r"\bpencil cup\b",
        ),
        width=DimensionRange(0.06, 0.12),
        depth=DimensionRange(0.06, 0.12),
        height=DimensionRange(0.08, 0.18),
        footprint_swappable=True,
    ),
    SizeProfile(
        name="keyboard",
        patterns=(r"\bkeyboard\b",),
        width=DimensionRange(0.25, 0.50),
        depth=DimensionRange(0.08, 0.20),
        height=DimensionRange(0.01, 0.05),
    ),
    SizeProfile(
        name="mouse",
        patterns=(r"\bmouse\b", r"\bcomputer mouse\b"),
        width=DimensionRange(0.05, 0.09),
        depth=DimensionRange(0.08, 0.14),
        height=DimensionRange(0.02, 0.06),
    ),
    SizeProfile(
        name="mug_cup",
        patterns=(r"\bmug\b", r"\bcup\b", r"\bteacup\b", r"\bcoffee cup\b"),
        width=DimensionRange(0.06, 0.12),
        depth=DimensionRange(0.06, 0.12),
        height=DimensionRange(0.06, 0.16),
        footprint_swappable=True,
    ),
    SizeProfile(
        name="plate_bowl",
        patterns=(r"\bplate\b", r"\bbowl\b", r"\bdish\b"),
        width=DimensionRange(0.10, 0.32),
        depth=DimensionRange(0.10, 0.32),
        height=DimensionRange(0.015, 0.14),
        footprint_swappable=True,
    ),
)


def match_size_profile(
    description: str, short_name: str | None = None
) -> SizeProfile | None:
    """Match a manipuland description to a known deterministic size profile."""
    haystack = " ".join(part for part in [description, short_name] if part).lower()
    for profile in SIZE_PROFILES:
        if any(re.search(pattern, haystack) for pattern in profile.patterns):
            return profile
    return None


def _validate_dimensions(
    dimensions: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    dims = np.asarray(dimensions, dtype=float)
    if dims.shape != (3,):
        raise ValueError(f"dimensions must contain exactly 3 values, got {dimensions}")
    if np.any(dims <= 0):
        raise ValueError(f"dimensions must be positive, got {dimensions}")
    return dims


def normalize_manipuland_dimensions(
    description: str,
    dimensions: list[float],
    short_name: str | None = None,
) -> tuple[list[float], SizeProfile | None]:
    """Clamp known manipuland categories to realistic profile ranges.

    Unknown categories are returned unchanged. Known categories preserve valid LLM
    dimensions and only clamp axes that are outside the profile envelope.
    """
    profile = match_size_profile(description, short_name)
    if profile is None:
        return dimensions, None

    dims = _validate_dimensions(dimensions)
    normalized = [
        axis_range.clamp(float(dim)) for dim, axis_range in zip(dims, profile.ranges)
    ]
    return normalized, profile


def compute_uniform_scale_fit(
    current_dimensions: list[float] | tuple[float, ...] | np.ndarray,
    desired_dimensions: list[float] | tuple[float, ...] | np.ndarray,
    *,
    footprint_swappable: bool = False,
) -> UniformScaleFit:
    """Compute the best uniform-scale fit and resulting axis relative errors."""
    current = _validate_dimensions(current_dimensions)
    desired = _validate_dimensions(desired_dimensions)

    axis_orders = [(0, 1, 2)]
    if footprint_swappable:
        axis_orders.append((1, 0, 2))

    best_fit: UniformScaleFit | None = None
    for axis_order in axis_orders:
        ordered_current = current[list(axis_order)]
        scale_factors = desired / ordered_current
        uniform_scale = float(np.median(scale_factors))
        actual = ordered_current * uniform_scale
        errors = np.abs(actual - desired) / desired
        fit = UniformScaleFit(
            current_dimensions=tuple(float(v) for v in ordered_current),
            desired_dimensions=tuple(float(v) for v in desired),
            uniform_scale=uniform_scale,
            actual_dimensions=tuple(float(v) for v in actual),
            axis_relative_errors=tuple(float(v) for v in errors),
            max_axis_relative_error=float(np.max(errors)),
            axis_order=axis_order,
        )
        if (
            best_fit is None
            or fit.max_axis_relative_error < best_fit.max_axis_relative_error
        ):
            best_fit = fit

    if best_fit is None:
        raise ValueError("Unable to compute uniform scale fit")
    return best_fit


def diagnose_manipuland_scale(
    description: str,
    actual_dimensions: list[float] | tuple[float, ...] | np.ndarray,
    *,
    requested_dimensions: list[float] | tuple[float, ...] | np.ndarray | None = None,
    short_name: str | None = None,
    max_axis_relative_error: float = DEFAULT_MAX_AXIS_RELATIVE_ERROR,
) -> dict[str, Any] | None:
    """Return a JSON-friendly diagnostic for known manipuland scale profiles."""
    profile = match_size_profile(description, short_name)
    if profile is None:
        return None

    actual = _validate_dimensions(actual_dimensions)
    expected_range = {
        "width": asdict(profile.width),
        "depth": asdict(profile.depth),
        "height": asdict(profile.height),
    }
    out_of_range_axes = [
        axis
        for axis, value, axis_range in zip(
            ("width", "depth", "height"), actual, profile.ranges
        )
        if not axis_range.contains(float(value))
    ]

    diagnostic: dict[str, Any] = {
        "profile": profile.name,
        "status": "ok" if not out_of_range_axes else "out_of_profile_range",
        "expected_range_m": expected_range,
        "actual_dimensions_m": [float(v) for v in actual],
        "out_of_range_axes": out_of_range_axes,
    }

    if requested_dimensions is not None:
        requested = _validate_dimensions(requested_dimensions)
        fit = compute_uniform_scale_fit(
            actual,
            requested,
            footprint_swappable=profile.footprint_swappable,
        )
        diagnostic.update(
            {
                "requested_dimensions_m": [float(v) for v in requested],
                "recommended_uniform_scale": fit.uniform_scale,
                "post_scale_dimensions_m": list(fit.actual_dimensions),
                "axis_relative_errors": list(fit.axis_relative_errors),
                "max_axis_relative_error": fit.max_axis_relative_error,
            }
        )
        if fit.max_axis_relative_error > max_axis_relative_error:
            diagnostic["status"] = "bad_uniform_scale_fit"

    return diagnostic
