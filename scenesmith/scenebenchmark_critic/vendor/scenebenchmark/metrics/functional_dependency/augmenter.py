from __future__ import annotations

from typing import Any, Callable

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.proposer import (
    augment_functional_dependency_checks as _augment_functional_dependency_checks,
)

ProgressFn = Callable[[str], None]


def augment_functional_dependency_checks(
    case_pack: dict[str, Any],
    config: Any,
    metric_filter: list[str] | None,
    progress: ProgressFn | None,
) -> bool:
    return _augment_functional_dependency_checks(
        case_pack,
        config,
        metric_filter=metric_filter,
        progress=progress,
    )
