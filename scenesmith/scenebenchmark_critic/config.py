"""Configuration helpers for the embedded SceneBenchmark critic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_METRICS = ("spatial_accessibility", "functional_dependency")


@dataclass(frozen=True)
class CriticConfig:
    enabled: bool = False
    metrics: tuple[str, ...] = DEFAULT_METRICS
    room_stage_hooks: tuple[str, ...] = ("scene_after_furniture", "final_scene")
    house_stage_hooks: tuple[str, ...] = ()
    inject_into_llm_critic: bool = True
    hard_gate: bool = False
    max_issues_for_prompt: int = 8
    fail_gate_threshold: int = 1
    degraded_gate_threshold: int = 999999
    extra: dict[str, Any] = field(default_factory=dict)

    def metric_enabled(self, metric: str) -> bool:
        return metric in set(self.metrics)

    def room_stage_enabled(self, stage: str) -> bool:
        return stage in set(self.room_stage_hooks)

    def house_stage_enabled(self, stage: str) -> bool:
        return stage in set(self.house_stage_hooks)


def critic_config_from_any(cfg: Any) -> CriticConfig:
    """Extract critic config from a full experiment config or an agent config."""
    raw = _get(cfg, "scenebenchmark_critic", None)
    if raw is None:
        experiment = _get(cfg, "experiment", None)
        raw = _get(experiment, "scenebenchmark_critic", None)
    if raw is None:
        return CriticConfig()

    data = _to_plain_dict(raw)
    known = {
        "enabled",
        "metrics",
        "room_stage_hooks",
        "house_stage_hooks",
        "inject_into_llm_critic",
        "hard_gate",
        "max_issues_for_prompt",
        "fail_gate_threshold",
        "degraded_gate_threshold",
    }
    extra = {key: value for key, value in data.items() if key not in known}
    return CriticConfig(
        enabled=_as_bool(data.get("enabled", False)),
        metrics=_as_tuple(data.get("metrics", DEFAULT_METRICS), DEFAULT_METRICS),
        room_stage_hooks=_as_tuple(
            data.get("room_stage_hooks", ("scene_after_furniture", "final_scene")),
            ("scene_after_furniture", "final_scene"),
        ),
        house_stage_hooks=_as_tuple(data.get("house_stage_hooks", ()), ()),
        inject_into_llm_critic=_as_bool(data.get("inject_into_llm_critic", True)),
        hard_gate=_as_bool(data.get("hard_gate", False)),
        max_issues_for_prompt=int(data.get("max_issues_for_prompt", 8)),
        fail_gate_threshold=int(data.get("fail_gate_threshold", 1)),
        degraded_gate_threshold=int(data.get("degraded_gate_threshold", 999999)),
        extra=extra,
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "items"):
        return {key: value for key, value in obj.items()}
    return {
        key: getattr(obj, key)
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value or ())
