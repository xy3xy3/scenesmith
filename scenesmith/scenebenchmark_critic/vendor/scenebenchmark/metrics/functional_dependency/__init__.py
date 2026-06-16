from __future__ import annotations

from importlib import import_module

_BASE = (
    "scenesmith.scenebenchmark_critic.vendor.scenebenchmark."
    "metrics.functional_dependency"
)

__all__ = [
    "PLUGIN",
    "augment_functional_dependency_checks",
    "evaluate_functional_dependency",
]


def __getattr__(name: str):
    if name == "PLUGIN":
        return import_module(f"{_BASE}.plugin").PLUGIN
    if name == "augment_functional_dependency_checks":
        return import_module(f"{_BASE}.augmenter").augment_functional_dependency_checks
    if name == "evaluate_functional_dependency":
        return import_module(f"{_BASE}.evaluator").evaluate_functional_dependency
    raise AttributeError(
        f"module 'metrics.functional_dependency' has no attribute {name!r}"
    )
