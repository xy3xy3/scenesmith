from __future__ import annotations

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.proposer import (
    _build_fd_proposer_payload,
    _compact_task_instruction,
    _propose_via_vlm as _default_propose_via_vlm,
    augment_functional_dependency_checks,
    propose_dependency_relations as _propose_dependency_relations,
)
from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency.relations import (
    evaluate_functional_dependency,
)

_propose_via_vlm = _default_propose_via_vlm


def propose_dependency_relations(*args, **kwargs):
    kwargs.setdefault("vlm_proposer", _propose_via_vlm)
    return _propose_dependency_relations(*args, **kwargs)


__all__ = [
    "_build_fd_proposer_payload",
    "_compact_task_instruction",
    "_propose_via_vlm",
    "augment_functional_dependency_checks",
    "evaluate_functional_dependency",
    "propose_dependency_relations",
]
