from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.base import (
    MetricPlugin,
)


def evaluate_functional_dependency(
    case_pack: dict[str, Any], check: dict[str, Any], config: Any
) -> dict[str, Any] | None:
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.evaluator import (
        evaluate_functional_dependency as _evaluate_functional_dependency,
    )

    return _evaluate_functional_dependency(case_pack, check, config)


def augment_functional_dependency_checks(
    case_pack, config, metric_filter, progress
) -> bool:
    from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.augmenter import (
        augment_functional_dependency_checks as _augment_functional_dependency_checks,
    )

    return _augment_functional_dependency_checks(
        case_pack, config, metric_filter, progress
    )


PLUGIN = MetricPlugin(
    name="functional_dependency",
    display_label_zh="功能依赖",
    rule_evaluator=evaluate_functional_dependency,
    check_augmenter=augment_functional_dependency_checks,
)
