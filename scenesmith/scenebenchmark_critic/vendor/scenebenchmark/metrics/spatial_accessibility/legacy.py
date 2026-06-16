from __future__ import annotations

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.spatial_accessibility import (
    config as _config,
    core as _core,
    grid as _grid,
    obstacles as _obstacles,
    reach as _reach,
    results as _results,
    zones as _zones,
)

for _module in (_config, _grid, _obstacles, _reach, _zones, _results, _core):
    globals().update(
        {
            _name: _value
            for _name, _value in vars(_module).items()
            if not _name.startswith("__") and _name not in {"annotations"}
        }
    )

evaluate_spatial_accessibility = _core.evaluate_spatial_accessibility
