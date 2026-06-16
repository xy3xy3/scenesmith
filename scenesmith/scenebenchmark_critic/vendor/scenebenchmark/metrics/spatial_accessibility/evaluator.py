from __future__ import annotations

from typing import Any

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.critic.geometry import (
    load_geometry,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.spatial_accessibility.core import (
    evaluate_spatial_accessibility as _evaluate_spatial_accessibility,
)


def evaluate_spatial_accessibility(
    case_pack: dict[str, Any], check: dict[str, Any], config: Any
) -> dict[str, Any] | None:
    store = load_geometry(case_pack)
    if store is None:
        return None
    return _evaluate_spatial_accessibility(store, check, config)
