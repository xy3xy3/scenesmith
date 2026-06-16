from __future__ import annotations

from scenesmith.scenebenchmark_critic.vendor.scenebenchmark.metrics.functional_dependency import (
    constants as _constants,
    proposer as _proposer,
    relations as _relations,
    results as _results,
    semantics as _semantics,
    support as _support,
)

for _module in (_constants, _semantics, _support, _results, _relations, _proposer):
    globals().update(
        {
            _name: _value
            for _name, _value in vars(_module).items()
            if not _name.startswith("__") and _name not in {"annotations"}
        }
    )

_propose_via_vlm = _proposer._propose_via_vlm
_build_fd_proposer_payload = _proposer._build_fd_proposer_payload
_compact_task_instruction = _proposer._compact_task_instruction
evaluate_functional_dependency = _relations.evaluate_functional_dependency


def propose_dependency_relations(*args, **kwargs):
    kwargs.setdefault("vlm_proposer", _propose_via_vlm)
    return _proposer.propose_dependency_relations(*args, **kwargs)


def augment_functional_dependency_checks(*args, **kwargs):
    kwargs.setdefault("vlm_proposer", _propose_via_vlm)
    return _proposer.augment_functional_dependency_checks(*args, **kwargs)
