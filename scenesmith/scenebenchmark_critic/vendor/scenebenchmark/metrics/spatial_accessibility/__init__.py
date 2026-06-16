from __future__ import annotations

__all__ = ["PLUGIN", "evaluate_spatial_accessibility"]


def __getattr__(name: str):
    if name == "PLUGIN":
        from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.spatial_accessibility.plugin import (
            PLUGIN,
        )

        return PLUGIN
    if name == "evaluate_spatial_accessibility":
        from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.spatial_accessibility.evaluator import (
            evaluate_spatial_accessibility,
        )

        return evaluate_spatial_accessibility
    raise AttributeError(
        f"module 'metrics.spatial_accessibility' has no attribute {name!r}"
    )
