from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.base import (
    MetricPlugin,
)


def evaluate_spatial_accessibility(
    case_pack: dict[str, Any], check: dict[str, Any], config: Any
) -> dict[str, Any] | None:
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.spatial_accessibility.evaluator import (
        evaluate_spatial_accessibility as _evaluate_spatial_accessibility,
    )

    return _evaluate_spatial_accessibility(case_pack, check, config)


PLUGIN = MetricPlugin(
    name="spatial_accessibility",
    display_label_zh="空间可达",
    rule_evaluator=evaluate_spatial_accessibility,
)
