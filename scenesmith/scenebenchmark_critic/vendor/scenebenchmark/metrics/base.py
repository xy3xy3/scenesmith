from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ProgressFn = Callable[[str], None]
RuleEvaluator = Callable[[dict[str, Any], dict[str, Any], Any], dict[str, Any] | None]
CheckAugmenter = Callable[
    [dict[str, Any], Any, list[str] | None, ProgressFn | None], bool
]
SummaryPolicy = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class MetricPlugin:
    name: str
    display_label_zh: str
    rule_evaluator: RuleEvaluator | None = None
    check_augmenter: CheckAugmenter | None = None
    counts_toward_summary: SummaryPolicy | None = None
